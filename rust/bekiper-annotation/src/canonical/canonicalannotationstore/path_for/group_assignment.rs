impl CanonicalAnnotationStore {
    pub fn assign_sessions_to_source_group(
        &self,
        session_keys: &[String],
        source_perimeter_id: &str,
        target_image: &str,
    ) -> Result<Vec<CanonicalSession>> {
        let source_perimeter_id = source_perimeter_id.trim();
        let target_image = target_image.trim();
        if source_perimeter_id.is_empty() {
            return Err(BekiperError::Data(
                "source perimeter id must not be empty".into(),
            ));
        }
        if target_image.is_empty() {
            return Err(BekiperError::Data("target image must not be empty".into()));
        }
        if session_keys.is_empty() {
            return Err(BekiperError::Data(
                "at least one session is required to assign an annotation group".into(),
            ));
        }

        self.load_perimeter(source_perimeter_id)?;

        let mut updated = Vec::new();
        for session_key in session_keys {
            let mut session = self.load_session(session_key)?;
            session.perimeter_id = source_perimeter_id.to_string();
            session.source_perimeter_id = None;
            session.coordinate_space = CoordinateSpace::Source;
            session.target_image = Some(target_image.to_string());
            updated.push(session);
        }

        let session_backups = updated
            .iter()
            .map(|session| {
                let path = self.session_path(&session.key);
                let content = fs::read_to_string(&path)?;
                Ok((path, content))
            })
            .collect::<Result<Vec<_>>>()?;
        let sync_state_path = self.root.join("sync_state.toml");
        let sync_state_backup = fs::read_to_string(&sync_state_path).ok();

        let write_result = (|| {
            for session in &updated {
                self.write_session(session)?;
            }
            self.write_sync_state()?;
            Ok(())
        })();

        if let Err(error) = write_result {
            for (path, content) in session_backups {
                fs::write(path, content)?;
            }
            match sync_state_backup {
                Some(content) => {
                    fs::write(sync_state_path, content)?;
                }
                None => {
                    if sync_state_path.exists() {
                        fs::remove_file(sync_state_path)?;
                    }
                }
            }
            return Err(error);
        }

        Ok(updated)
    }
}
