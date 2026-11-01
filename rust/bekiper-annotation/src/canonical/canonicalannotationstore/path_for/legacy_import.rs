impl CanonicalAnnotationStore {
    pub fn import_legacy(annotation_dir: &Path) -> Result<Self> {
        Self::import_legacy_with_review(annotation_dir, false)
    }

    pub fn import_legacy_with_review(
        annotation_dir: &Path,
        reviewed_externally: bool,
    ) -> Result<Self> {
        let store = Self::create_or_open(annotation_dir);
        store.import_with_review(reviewed_externally)?;
        Ok(store)
    }

    pub fn import_legacy_in_place(&self) -> Result<()> {
        self.import_with_review(false)
    }

    pub fn ingestion(&self) -> Result<BTreeMap<String, IngestionProvenance>> {
        let path = self.root.join("ingestion.toml");
        match fs::read_to_string(&path) {
            Ok(raw) => toml::from_str(&raw)
                .map_err(|e| BekiperError::Config(format!("invalid {}: {e}", path.display()))),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(BTreeMap::new()),
            Err(e) => Err(e.into()),
        }
    }

    fn import_with_review(&self, reviewed_externally: bool) -> Result<()> {
        // Hold a cross-process lock from the authoritative provenance read until
        // publication finishes. Closing the file releases it on every exit path.
        fs::create_dir_all(&self.root)?;
        let lock = fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(self.root.join(".ingestion.lock"))?;
        lock.lock()?;
        let mut ingestion = self.ingestion()?;
        let existing_keys: BTreeSet<_> = self.load_sessions()?.into_iter().map(|s| s.key).collect();

        let plugin_dir = self.annotation_dir.join("plugin_files");
        let mut perimeter_ids = BTreeSet::new();
        let mut sessions = Vec::new();

        let video_map_path = self.annotation_dir.join("video_map.csv");
        let metadata_path = self.annotation_dir.join("metadata.csv");

        if video_map_path.exists() {
            for (video_number, perimeter_id) in read_video_map(&video_map_path)? {
                perimeter_ids.insert(perimeter_id.clone());
                sessions.push(CanonicalSession {
                    key: format!("video_{video_number}"),
                    video_number: Some(video_number),
                    animal: None,
                    stage: None,
                    phase_part: None,
                    perimeter_id,
                    source_perimeter_id: None,
                    coordinate_space: CoordinateSpace::Source,
                    target_image: None,
                });
            }
        } else if metadata_path.exists() {
            for (row_index, row) in read_metadata_csv(&metadata_path)?.into_iter().enumerate() {
                let Some(source) = row.source else {
                    continue;
                };
                let (perimeter_id, target_image) = match source {
                    PerimeterSource::Direct { perimeter_id } => (perimeter_id, None),
                    PerimeterSource::ReReferenced {
                        perimeter_id,
                        target_image,
                    } => (perimeter_id, Some(target_image)),
                };
                perimeter_ids.insert(perimeter_id.clone());
                sessions.push(CanonicalSession {
                    key: format!(
                        "metadata_row_{}_animal_{}_stage_{}_part_{}",
                        row_index + 2,
                        row.animal,
                        row.stage,
                        row.phase_part
                    ),
                    video_number: Some(row.animal),
                    animal: Some(row.animal),
                    stage: Some(row.stage),
                    phase_part: Some(row.phase_part),
                    perimeter_id,
                    source_perimeter_id: None,
                    coordinate_space: CoordinateSpace::Source,
                    target_image,
                });
            }
        } else if let Some(perimeter_id) = sole_legacy_perimeter_id(&plugin_dir)? {
            perimeter_ids.insert(perimeter_id.clone());
            sessions.push(CanonicalSession {
                key: "fallback".into(),
                video_number: None,
                animal: None,
                stage: None,
                phase_part: None,
                perimeter_id,
                source_perimeter_id: None,
                coordinate_space: CoordinateSpace::Source,
                target_image: None,
            });
        }

        for session in &sessions {
            ingestion.entry(session.key.clone()).or_insert_with(|| {
                let now = now_unix_s();
                // Prior ingestion without provenance is not evidence of external
                // review. Only a newly ingested session may create a request.
                let reviewed_externally =
                    reviewed_externally && !existing_keys.contains(&session.key);
                IngestionProvenance {
                    imported: true,
                    reviewed_externally,
                    ingested_unix_s: now,
                    validation_request_id: reviewed_externally
                        .then(|| format!("import-{now}-{}", session.key)),
                }
            });
        }
        // Persist the ingestion identity before publishing geometry. A failed
        // import retries with this same identity and immutable eligibility.
        write_toml(&self.root.join("ingestion.toml"), &ingestion)?;
        // Validate all legacy assets before replacing usable canonical data.
        let prepared = perimeter_ids
            .iter()
            .map(|id| {
                Ok((
                    id,
                    load_perimeter_by_id(&plugin_dir, id)?,
                    import_reference_document(&plugin_dir, id)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut previous_paths = Vec::new();
        for name in ["sessions", "perimeters", "references"] {
            let path = self.root.join(name);
            fs::create_dir_all(&path)?;
            previous_paths.extend(sorted_files(&path)?);
        }
        fs::create_dir_all(self.root.join("conflicts"))?;
        let mut writes = Vec::new();
        for session in &sessions {
            writes.push((
                self.session_path(&session.key),
                toml::to_string_pretty(session),
            ));
        }

        for (perimeter_id, perimeters, reference) in prepared {
            writes.push((
                self.perimeter_path(perimeter_id),
                toml::to_string_pretty(&PerimetersFile { perimeters }),
            ));
            if reference.source.is_some() || !reference.targets.is_empty() {
                writes.push((
                    self.reference_path(perimeter_id),
                    toml::to_string_pretty(&reference),
                ));
            }
        }

        let manifest = StoreManifest {
            schema_version: SCHEMA_VERSION,
            imported_unix_s: now_unix_s(),
            source_layout: "legacy_plugin_files".into(),
            external_fingerprints: legacy_fingerprints(&self.annotation_dir)?,
        };
        writes.push((
            self.root.join("manifest.toml"),
            toml::to_string_pretty(&manifest),
        ));
        let published: BTreeSet<_> = writes.iter().map(|(path, _)| path.clone()).collect();
        let removals = previous_paths
            .into_iter()
            .filter(|path| !published.contains(path))
            .collect();
        Self::publish_toml_batch(writes, removals)?;
        self.write_sync_state()?;
        Ok(())
    }
}
