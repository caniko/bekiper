impl AnnotationSession {
    pub fn load(annotation_dir: &Path, video_path: &Path) -> Result<Self> {
        Self::load_with_context(annotation_dir, video_path, None)
    }

    pub fn load_with_context(
        annotation_dir: &Path,
        video_path: &Path,
        context: Option<IngressSessionContext>,
    ) -> Result<Self> {
        let annotation_dir = annotation_dir.to_path_buf();
        let plugin_dir = annotation_dir.join("plugin_files");
        let video_context = extract_video_context(video_path);

        if let Some(store) = CanonicalAnnotationStore::open(&annotation_dir)? {
            let session = context
                .as_ref()
                .and_then(|ctx| {
                    Some(VideoContext {
                        animal: ctx.animal.as_deref()?.parse().ok()?,
                        stage: ctx.stage.as_deref()?.parse().ok()?,
                        phase_part: ctx.phase_part.as_deref()?.parse().ok()?,
                    })
                })
                .as_ref()
                .or(video_context.as_ref())
                .map(|ctx| store.resolve_session(ctx))
                .transpose()?
                .flatten()
                .or_else(|| {
                    store
                        .load_sessions()
                        .ok()
                        .and_then(|mut sessions| (sessions.len() == 1).then(|| sessions.remove(0)))
                })
                .ok_or_else(|| {
                    BekiperError::Data(format!(
                        "could not resolve canonical annotation mapping for {}",
                        video_path.display()
                    ))
                })?;
            let specs = store.resolve_session_perimeters(&session)?;
            let perimeter_id = session.perimeter_id.clone();
            let current_target_image = session.target_image.clone();
            let reference_id = session
                .source_perimeter_id
                .as_deref()
                .unwrap_or(&session.perimeter_id)
                .to_string();
            let canonical_reference = store.load_reference_document(&reference_id)?;
            let (source_reference, change_reference, current_target_reference_row) =
                canonical_reference_to_editable(
                    &store.reference_path(&reference_id),
                    canonical_reference,
                    current_target_image.as_deref(),
                );
            let asset = EditablePerimeterAsset::Canonical {
                perimeter_id: perimeter_id.clone(),
                store,
                session_key: session.key.clone(),
                session,
                specs,
            };
            let mut loaded = Self {
                annotation_dir,
                plugin_dir,
                video_path: video_path.to_path_buf(),
                context,
                video_context,
                mapping: SessionMapping::Fallback {
                    perimeter_id: perimeter_id.clone(),
                },
                perimeter_id,
                asset,
                resolved_specs: Vec::new(),
                current_target_image,
                metadata: None,
                video_map: None,
                source_reference,
                change_reference,
                current_target_reference_row,
                pending_geometry: None,
            };
            loaded.resolved_specs = loaded.asset.to_specs()?;
            return Ok(loaded);
        }

        let metadata_path = annotation_dir.join("metadata.csv");
        let video_map_path = annotation_dir.join("video_map.csv");

        let metadata = if metadata_path.exists() {
            Some(MetadataDocument::load(&metadata_path)?)
        } else {
            None
        };
        let video_map = if video_map_path.exists() {
            Some(VideoMapDocument::load(&video_map_path)?)
        } else {
            None
        };

        let mapping = metadata
            .as_ref()
            .and_then(|metadata| metadata.resolve_context_row(context.as_ref()))
            .or_else(|| {
                let (Some(metadata), Some(ctx)) = (&metadata, &video_context) else {
                    return None;
                };
                metadata.resolve_row(ctx)
            })
            .map(|(row_index, source)| SessionMapping::Metadata { row_index, source })
            .or_else(|| {
                let video_number = extract_video_number(video_path)?;
                video_map.as_ref().and_then(|doc| {
                    doc.resolve_row(video_number)
                        .map(|(row_index, perimeter_id)| SessionMapping::VideoMap {
                            row_index,
                            perimeter_id,
                        })
                })
            })
            .or_else(|| {
                sole_perimeter_id(&plugin_dir)
                    .ok()
                    .flatten()
                    .map(|perimeter_id| SessionMapping::Fallback { perimeter_id })
            })
            .ok_or_else(|| {
                BekiperError::Data(format!(
                    "could not resolve annotation mapping for {}",
                    video_path.display()
                ))
            })?;

        let (perimeter_id, current_target_image) = match &mapping {
            SessionMapping::Metadata { source, .. } => match source {
                PerimeterSource::Direct { perimeter_id } => (perimeter_id.clone(), None),
                PerimeterSource::ReReferenced {
                    perimeter_id,
                    target_image,
                } => (perimeter_id.clone(), Some(target_image.clone())),
            },
            SessionMapping::VideoMap { perimeter_id, .. } => (perimeter_id.clone(), None),
            SessionMapping::Fallback { perimeter_id } => (perimeter_id.clone(), None),
        };

        let asset = EditablePerimeterAsset::load(&plugin_dir, &perimeter_id)?;

        let mut session = Self {
            annotation_dir,
            plugin_dir,
            video_path: video_path.to_path_buf(),
            context,
            video_context,
            perimeter_id: perimeter_id.clone(),
            asset,
            resolved_specs: Vec::new(),
            current_target_image,
            metadata,
            video_map,
            mapping,
            source_reference: None,
            change_reference: None,
            current_target_reference_row: None,
            pending_geometry: None,
        };
        session.refresh()?;
        Ok(session)
    }

    pub fn is_rereferenced(&self) -> bool {
        if matches!(
            self.asset,
            EditablePerimeterAsset::Canonical {
                session: CanonicalSession {
                    target_image: Some(_),
                    ..
                },
                ..
            }
        ) {
            return true;
        }
        matches!(
            self.mapping,
            SessionMapping::Metadata {
                source: PerimeterSource::ReReferenced { .. },
                ..
            }
        )
    }

    pub fn family(&self) -> PerimeterAssetFamily {
        self.asset.family()
    }

    pub fn is_metadata_backed(&self) -> bool {
        matches!(self.mapping, SessionMapping::Metadata { .. })
    }

    pub fn can_edit_source_reference(&self) -> bool {
        self.source_reference.is_some()
    }

    pub fn can_edit_current_target_reference(&self) -> bool {
        self.current_target_image.is_some()
    }

    pub fn can_edit_current_target_geometry(&self) -> bool {
        self.current_target_image.is_some()
    }

    pub fn is_canonical_backed(&self) -> bool {
        matches!(self.asset, EditablePerimeterAsset::Canonical { .. })
    }

    pub fn source_image_stem(&self) -> Option<String> {
        if let Some(name) = self.asset.source_image_name() {
            return Some(strip_png_extension(name).to_string());
        }
        self.source_reference
            .as_ref()
            .and_then(ReferenceDocument::first_point)
            .map(|point| strip_png_extension(&point.image_name).to_string())
    }

    pub fn target_image_stem(&self) -> Option<&str> {
        self.current_target_image.as_deref()
    }

    pub fn has_source_reference_row(&self) -> bool {
        self.source_reference
            .as_ref()
            .is_some_and(|document| !document.rows.is_empty())
    }

    pub fn perimeter_selection_source(&self) -> PerimeterSelectionSource {
        match &self.mapping {
            SessionMapping::Metadata { .. } => {
                if self
                    .metadata_value("Perimeter")
                    .is_some_and(|value| !value.trim().is_empty())
                {
                    PerimeterSelectionSource::MetadataPerimeter
                } else {
                    PerimeterSelectionSource::MetadataChangeReference
                }
            }
            SessionMapping::VideoMap { .. } => PerimeterSelectionSource::VideoMap,
            SessionMapping::Fallback { .. } => PerimeterSelectionSource::Fallback,
        }
    }

    pub fn source_reference_point(&self) -> Option<ReferencePoint> {
        self.source_reference
            .as_ref()
            .and_then(ReferenceDocument::first_point)
    }

    pub fn current_target_reference_point(&self) -> Option<ReferencePoint> {
        let doc = self.change_reference.as_ref()?;
        let index = self.current_target_reference_row?;
        doc.as_reference_point(index)
    }

    pub fn target_reference_state(&self) -> TargetReferenceState {
        if self.current_target_image.is_none() {
            TargetReferenceState::Unavailable
        } else if self.current_target_reference_row.is_some() {
            TargetReferenceState::SavedRow
        } else {
            TargetReferenceState::FallbackPreview
        }
    }

    pub fn annotation_group_summary(&self) -> Option<AnnotationGroupSummary> {
        match &self.asset {
            EditablePerimeterAsset::Canonical { store, session, .. } => {
                let info = store.group_info_for_session(session).ok().flatten()?;
                Some(canonical_group_summary(&info, session))
            }
            _ => self.legacy_group_summary(),
        }
    }

    fn legacy_group_summary(&self) -> Option<AnnotationGroupSummary> {
        let (source_perimeter_id, target_image) = match &self.mapping {
            SessionMapping::Metadata {
                source:
                    PerimeterSource::ReReferenced {
                        perimeter_id,
                        target_image,
                    },
                ..
            } => (perimeter_id.clone(), target_image.clone()),
            _ => return None,
        };
        let metadata = self.metadata.as_ref()?;
        let mut member_labels = Vec::new();
        for row_index in 0..metadata.rows.len() {
            let Some(PerimeterSource::ReReferenced {
                perimeter_id,
                target_image: row_target,
            }) = metadata.source_for_row(row_index)
            else {
                continue;
            };
            if perimeter_id == source_perimeter_id && row_target == target_image {
                member_labels.push(format!("metadata.csv row {}", row_index + 2));
            }
        }
        Some(AnnotationGroupSummary {
            source_perimeter_id,
            target_image,
            member_count: member_labels.len().max(1),
            member_labels,
            correction_kind: AnnotationCorrectionKind::SourceShared,
        })
    }

    fn resolve_imported_canonical_session(
        &self,
        store: &CanonicalAnnotationStore,
    ) -> Result<CanonicalSession> {
        let ctx = self.video_context.as_ref().ok_or_else(|| {
            BekiperError::Data(
                "cannot resolve imported canonical session without video context".into(),
            )
        })?;
        store.resolve_session(ctx)?.ok_or_else(|| {
            BekiperError::Data("canonical import did not resolve current session".into())
        })
    }

    pub fn metadata_value(&self, field: &str) -> Option<&str> {
        let SessionMapping::Metadata { row_index, .. } = self.mapping else {
            return None;
        };
        self.metadata.as_ref()?.get(row_index, field)
    }

    pub fn metadata_has_column(&self, field: &str) -> bool {
        self.metadata
            .as_ref()
            .is_some_and(|doc| doc.has_column(field))
    }

    pub fn ymaze_arm_labels(&self) -> Vec<String> {
        match &self.asset {
            EditablePerimeterAsset::RadialMaze { asset, .. } => asset
                .arm_lines
                .iter()
                .map(|row| row.label.clone())
                .collect(),
            _ => Vec::new(),
        }
    }

    pub fn resolved_object_labels(&self) -> Vec<String> {
        self.resolved_specs
            .iter()
            .map(|spec| spec.label().clone())
            .collect()
    }

    pub fn set_metadata_field(&mut self, field: &str, value: String) -> Result<()> {
        let row_index = match self.mapping {
            SessionMapping::Metadata { row_index, .. } => row_index,
            _ => {
                return Err(BekiperError::Data(
                    "metadata fields are only available for metadata-backed sessions".into(),
                ));
            }
        };
        let metadata = self.metadata.as_mut().ok_or_else(|| {
            BekiperError::Data("session does not contain a metadata document".into())
        })?;
        metadata.set(row_index, field, value)?;

        let source = metadata.source_for_row(row_index).ok_or_else(|| {
            BekiperError::Data(format!(
                "metadata no longer resolves a mapping for {}",
                self.video_path.display()
            ))
        })?;

        let (perimeter_id, current_target_image) = match &source {
            PerimeterSource::Direct { perimeter_id } => (perimeter_id.clone(), None),
            PerimeterSource::ReReferenced {
                perimeter_id,
                target_image,
            } => (perimeter_id.clone(), Some(target_image.clone())),
        };

        self.mapping = SessionMapping::Metadata { row_index, source };

        if perimeter_id != self.perimeter_id {
            return self.refresh();
        }

        self.current_target_image = current_target_image.clone();
        self.current_target_reference_row = current_target_image.as_ref().and_then(|target| {
            self.change_reference
                .as_ref()
                .and_then(|doc| doc.find_row_by_stem(target))
        });
        self.recompute_resolved_specs()
    }

    pub fn set_shared_geometry_from_resolved(
        &mut self,
        resolved_specs: &[PerimeterSpec],
    ) -> Result<()> {
        let mut source_specs = resolved_specs.to_vec();
        if let Some((dx, dy)) = self.current_translation() {
            translate_perimeters(&mut source_specs, -dx, -dy);
        }
        let original = self.asset.to_specs()?;
        if matches!(self.asset, EditablePerimeterAsset::Canonical { .. })
            || (!matches!(self.asset, EditablePerimeterAsset::CocoJson { .. })
                && original.len() == source_specs.len()
                && original
                    .iter()
                    .zip(&source_specs)
                    .all(|(old, new)| std::mem::discriminant(old) == std::mem::discriminant(new)))
        {
            self.asset.replace_from_specs(&source_specs)?;
        }
        self.pending_geometry = Some((GeometryWriteScope::SharedSource, source_specs));
        self.recompute_resolved_specs()
    }
}
