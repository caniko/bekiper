impl CanonicalAnnotationStore {
    pub fn sync(annotation_dir: &Path, options: SyncOptions) -> Result<SyncReport> {
        Self::sync_with_review(annotation_dir, options, false)
    }

    pub fn sync_with_review(
        annotation_dir: &Path,
        options: SyncOptions,
        reviewed_externally: bool,
    ) -> Result<SyncReport> {
        let canonical_exists = Self::exists(annotation_dir);
        let (legacy_changed, canonical_changed) = change_state(annotation_dir)?;
        let action = match options.mode {
            SyncMode::ImportOnly => {
                if canonical_changed && legacy_changed {
                    SyncAction::Conflict
                } else if legacy_changed || !canonical_exists {
                    SyncAction::Import
                } else {
                    SyncAction::Noop
                }
            }
            SyncMode::ExportOnly => {
                if canonical_changed || canonical_exists {
                    SyncAction::Export
                } else {
                    SyncAction::Noop
                }
            }
            SyncMode::TwoWay => {
                if !canonical_exists || (legacy_changed && !canonical_changed) {
                    SyncAction::Import
                } else if canonical_changed && !legacy_changed {
                    SyncAction::Export
                } else if canonical_changed && legacy_changed {
                    SyncAction::Conflict
                } else {
                    SyncAction::Noop
                }
            }
        };

        let mut conflict_path = None;
        if !options.dry_run {
            match action {
                SyncAction::Import => {
                    Self::import_legacy_with_review(annotation_dir, reviewed_externally)?;
                }
                SyncAction::Export => {
                    let store = Self::open(annotation_dir)?.ok_or_else(|| {
                        BekiperError::Data(format!(
                            "canonical annotation store does not exist at {}",
                            Self::path_for(annotation_dir).display()
                        ))
                    })?;
                    store.export_legacy()?;
                }
                SyncAction::Conflict => {
                    let path =
                        write_conflict_report(annotation_dir, legacy_changed, canonical_changed)?;
                    conflict_path = Some(path);
                    return Err(BekiperError::Data(format!(
                        "annotation sync conflict; report written to {}",
                        conflict_path.as_ref().unwrap().display()
                    )));
                }
                SyncAction::Noop => {}
            }
        } else if action == SyncAction::Conflict {
            conflict_path = Some(
                Self::path_for(annotation_dir)
                    .join("conflicts")
                    .join("dry-run.toml"),
            );
        }

        Ok(SyncReport {
            action,
            canonical_exists,
            legacy_changed,
            canonical_changed,
            conflict_path,
            dry_run: options.dry_run,
        })
    }

    pub fn write_sync_state(&self) -> Result<()> {
        let state = SyncState {
            legacy_fingerprints: legacy_fingerprints(&self.annotation_dir)?,
            canonical_fingerprints: canonical_fingerprints(&self.root)?,
            synced_unix_s: now_unix_s(),
        };
        write_toml(&self.root.join("sync_state.toml"), &state)
    }

    pub fn perimeter_path(&self, perimeter_id: &str) -> PathBuf {
        self.root
            .join("perimeters")
            .join(format!("{perimeter_id}.toml"))
    }

    pub fn reference_path(&self, perimeter_id: &str) -> PathBuf {
        self.root
            .join("references")
            .join(format!("{perimeter_id}.toml"))
    }

    pub fn session_path(&self, session_key: &str) -> PathBuf {
        self.root
            .join("sessions")
            .join(format!("{session_key}.toml"))
    }
}
