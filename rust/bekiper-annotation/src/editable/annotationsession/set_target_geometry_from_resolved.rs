impl AnnotationSession {
    pub fn set_target_geometry_from_resolved(
        &mut self,
        resolved_specs: &[PerimeterSpec],
    ) -> Result<()> {
        if !self.can_edit_current_target_geometry() {
            return Err(BekiperError::Data(
                "session does not expose target geometry".into(),
            ));
        }
        self.pending_geometry = Some((GeometryWriteScope::CurrentVideo, resolved_specs.to_vec()));
        self.resolved_specs = resolved_specs.to_vec();
        Ok(())
    }

    pub fn set_group_target_geometry_from_resolved(
        &mut self,
        resolved_specs: &[PerimeterSpec],
    ) -> Result<()> {
        if !self.can_edit_current_target_geometry() {
            return Err(BekiperError::Data(
                "session does not expose target geometry".into(),
            ));
        }
        self.pending_geometry = Some((GeometryWriteScope::EntireGroup, resolved_specs.to_vec()));
        self.resolved_specs = resolved_specs.to_vec();
        Ok(())
    }

    pub fn split_current_session_from_group(&mut self) -> Result<()> {
        let specs = self.resolved_specs.clone();
        match &mut self.asset {
            EditablePerimeterAsset::Canonical {
                store,
                session_key,
                session,
                perimeter_id,
                specs: stored_specs,
            } => {
                let updated = store.split_session_from_group(session_key, &specs)?;
                *perimeter_id = updated.perimeter_id.clone();
                *session = updated;
                *stored_specs = specs.clone();
                self.perimeter_id = perimeter_id.clone();
                self.resolved_specs = specs;
                Ok(())
            }
            _ => self.set_target_geometry_from_resolved(&specs),
        }
    }

    pub fn set_source_reference_point(&mut self, x: f64, y: f64) -> Result<()> {
        let source_doc = self.source_reference.as_mut().ok_or_else(|| {
            BekiperError::Data("session does not expose a source reference".into())
        })?;
        let template = source_doc.rows.first().cloned();
        let image_stem = self
            .asset
            .source_image_name()
            .map(strip_png_extension)
            .map(ToOwned::to_owned)
            .or_else(|| {
                source_doc
                    .rows
                    .first()
                    .map(|row| strip_png_extension(&row.image_name).to_string())
            })
            .unwrap_or_else(|| "source".into());
        source_doc.upsert_point(&image_stem, x, y, template.as_ref());
        self.recompute_resolved_specs()
    }

    pub fn set_current_target_reference_point(&mut self, x: f64, y: f64) -> Result<()> {
        let target_image = self.current_target_image.clone().ok_or_else(|| {
            BekiperError::Data("session does not expose a target reference".into())
        })?;
        let doc = self.change_reference.as_mut().ok_or_else(|| {
            BekiperError::Data("session does not expose a change-reference document".into())
        })?;
        let template = doc.rows.first().cloned().or_else(|| {
            self.source_reference
                .as_ref()
                .and_then(|source| source.rows.first().cloned())
        });
        let index = doc.upsert_point(&target_image, x, y, template.as_ref());
        self.current_target_reference_row = Some(index);
        self.recompute_resolved_specs()
    }

    pub fn save(&self) -> Result<()> {
        if let Some((scope, specs)) = &self.pending_geometry {
            let (store, session) = match &self.asset {
                EditablePerimeterAsset::Canonical { store, session, .. } => {
                    (store.clone(), session.clone())
                }
                _ => {
                    let store = CanonicalAnnotationStore::import_legacy(&self.annotation_dir)?;
                    let session = self.resolve_imported_canonical_session(&store)?;
                    (store, session)
                }
            };
            let source_id = session
                .source_perimeter_id
                .clone()
                .unwrap_or_else(|| session.perimeter_id.clone());
            let mut members = Vec::new();
            let id = match scope {
                GeometryWriteScope::SharedSource => session.perimeter_id.clone(),
                GeometryWriteScope::CurrentVideo | GeometryWriteScope::EntireGroup => {
                    let key = session.group_key().ok_or_else(|| {
                        BekiperError::Data("Target edit requires a target image".into())
                    })?;
                    let id = if *scope == GeometryWriteScope::EntireGroup {
                        members = store.group_members(&key)?;
                        crate::canonical::group_target_perimeter_id(
                            &key.source_perimeter_id,
                            &key.target_image,
                        )
                    } else {
                        members.push(session.clone());
                        crate::canonical::individual_target_perimeter_id(
                            &source_id,
                            &key.target_image,
                            &session.key,
                        )
                    };
                    for member in &mut members {
                        member.perimeter_id = id.clone();
                        member.source_perimeter_id = Some(source_id.clone());
                        member.coordinate_space = CoordinateSpace::Target;
                    }
                    id
                }
            };
            let reference = editable_references_to_canonical(
                self.source_reference.as_ref(),
                self.change_reference.as_ref(),
            );
            let reference = (reference.source.is_some() || !reference.targets.is_empty())
                .then_some((source_id.as_str(), &reference));
            return store.write_geometry_edit(&id, specs, &members, reference);
        }
        self.asset.write()?;
        if let EditablePerimeterAsset::Canonical { store, session, .. } = &self.asset {
            let reference_id = session
                .source_perimeter_id
                .as_deref()
                .unwrap_or(&session.perimeter_id);
            let reference = editable_references_to_canonical(
                self.source_reference.as_ref(),
                self.change_reference.as_ref(),
            );
            if reference.source.is_some() || !reference.targets.is_empty() {
                store.write_reference_document(reference_id, &reference)?;
                store.write_sync_state()?;
            }
            return Ok(());
        }
        if let Some(metadata) = &self.metadata {
            metadata.write()?;
        }
        if let Some(reference) = &self.source_reference {
            reference.write()?;
        }
        if let Some(reference) = &self.change_reference {
            reference.write()?;
        }
        Ok(())
    }

    pub fn refresh(&mut self) -> Result<()> {
        self.pending_geometry = None;
        if let EditablePerimeterAsset::Canonical {
            perimeter_id,
            store,
            session_key,
            session,
            specs,
        } = &mut self.asset
        {
            let refreshed = store.load_session(session_key)?;
            *perimeter_id = refreshed.perimeter_id.clone();
            *session = refreshed;
            *specs = store.resolve_session_perimeters(session)?;
            self.perimeter_id = perimeter_id.clone();
            self.current_target_image = session.target_image.clone();
            self.resolved_specs = specs.clone();
            return Ok(());
        }

        let mut refreshed_mapping = None;
        let (perimeter_id, target_image) = match &self.mapping {
            SessionMapping::Metadata { row_index, .. } => {
                let metadata = self
                    .metadata
                    .as_ref()
                    .ok_or_else(|| BekiperError::Data("session is missing metadata rows".into()))?;
                let (resolved_row_index, source) = metadata
                    .resolve_context_row(self.context.as_ref())
                    .or_else(|| {
                        let ctx = self.video_context.as_ref()?;
                        metadata.resolve_row(ctx)
                    })
                    .ok_or_else(|| {
                        BekiperError::Data(format!(
                            "metadata no longer resolves a mapping for {}",
                            self.video_path.display()
                        ))
                    })?;
                if resolved_row_index != *row_index {
                    return Err(BekiperError::Data(format!(
                        "metadata mapping for {} moved from row {} to row {}",
                        self.video_path.display(),
                        row_index,
                        resolved_row_index
                    )));
                }
                refreshed_mapping = Some(SessionMapping::Metadata {
                    row_index: *row_index,
                    source: source.clone(),
                });
                match source {
                    PerimeterSource::Direct { perimeter_id } => (perimeter_id, None),
                    PerimeterSource::ReReferenced {
                        perimeter_id,
                        target_image,
                    } => (perimeter_id, Some(target_image)),
                }
            }
            SessionMapping::VideoMap {
                row_index,
                perimeter_id,
            } => {
                let doc = self.video_map.as_ref().ok_or_else(|| {
                    BekiperError::Data("session is missing video_map rows".into())
                })?;
                let video_number = extract_video_number(&self.video_path).ok_or_else(|| {
                    BekiperError::Data(format!(
                        "could not extract video number from {}",
                        self.video_path.display()
                    ))
                })?;
                let (resolved_row_index, resolved_id) =
                    doc.resolve_row(video_number).ok_or_else(|| {
                        BekiperError::Data(format!(
                            "video_map no longer resolves {}",
                            self.video_path.display()
                        ))
                    })?;
                if resolved_row_index != *row_index || resolved_id != *perimeter_id {
                    return Err(BekiperError::Data(format!(
                        "video_map mapping for {} changed since session load",
                        self.video_path.display()
                    )));
                }
                (perimeter_id.clone(), None)
            }
            SessionMapping::Fallback { perimeter_id } => (perimeter_id.clone(), None),
        };

        if let Some(mapping) = refreshed_mapping {
            self.mapping = mapping;
        }
        self.perimeter_id = perimeter_id.clone();
        self.current_target_image = target_image.clone();
        self.asset = EditablePerimeterAsset::load(&self.plugin_dir, &perimeter_id)?;

        let source_reference_path = self
            .plugin_dir
            .join(format!("reference-{perimeter_id}.csv"));
        self.source_reference = if source_reference_path.exists() {
            Some(ReferenceDocument::load(&source_reference_path)?)
        } else {
            None
        };

        let change_reference_path = self
            .plugin_dir
            .join(format!("change_reference-{perimeter_id}.csv"));
        self.change_reference = if change_reference_path.exists() || target_image.is_some() {
            Some(ReferenceDocument::load_or_empty(&change_reference_path)?)
        } else {
            None
        };
        self.current_target_reference_row = target_image.as_ref().and_then(|target| {
            self.change_reference
                .as_ref()
                .and_then(|doc| doc.find_row_by_stem(target))
        });

        self.recompute_resolved_specs()
    }

    fn current_translation(&self) -> Option<(f64, f64)> {
        if matches!(
            self.asset,
            EditablePerimeterAsset::Canonical {
                session: CanonicalSession {
                    coordinate_space: CoordinateSpace::Target,
                    ..
                },
                ..
            }
        ) {
            return None;
        }
        let target_image = self.current_target_image.as_ref()?;
        let source = self.source_reference_point()?;
        let target = self.current_target_reference_point().or_else(|| {
            self.change_reference.as_ref().and_then(|doc| {
                doc.find_row_by_stem(target_image)
                    .and_then(|index| doc.as_reference_point(index))
            })
        });
        let target = target?;
        if strip_png_extension(&source.image_name) == target_image {
            return None;
        }
        Some((target.x - source.x, target.y - source.y))
    }

    fn recompute_resolved_specs(&mut self) -> Result<()> {
        if let Some((scope, specs)) = &self.pending_geometry {
            let mut resolved = specs.clone();
            if *scope == GeometryWriteScope::SharedSource
                && let Some((dx, dy)) = self.current_translation()
            {
                translate_perimeters(&mut resolved, dx, dy);
            }
            self.resolved_specs = resolved;
            return Ok(());
        }
        let mut resolved = self.asset.to_specs()?;
        if let Some((dx, dy)) = self.current_translation() {
            translate_perimeters(&mut resolved, dx, dy);
        }
        self.resolved_specs = resolved;
        Ok(())
    }
}
