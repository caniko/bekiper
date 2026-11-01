impl CanonicalAnnotationStore {
    pub fn write_perimeter(&self, perimeter_id: &str, perimeters: &[PerimeterSpec]) -> Result<()> {
        fs::create_dir_all(self.root.join("perimeters"))?;
        let file = PerimetersFile {
            perimeters: perimeters.to_vec(),
        };
        let content = toml::to_string_pretty(&file)
            .map_err(|e| BekiperError::Other(format!("failed to serialize perimeter: {e}")))?;
        fs::write(self.perimeter_path(perimeter_id), content)?;
        Ok(())
    }

    /// Publish one GUI geometry edit and its mappings together, rolling back
    /// completed writes if a later file cannot be published.
    // ponytail: rollback handles I/O errors, not process death between files;
    // add a durable journal if crash-atomic multi-file commits are required.
    pub fn write_geometry_edit(
        &self,
        perimeter_id: &str,
        perimeters: &[PerimeterSpec],
        sessions: &[CanonicalSession],
        reference: Option<(&str, &ReferenceDocument)>,
    ) -> Result<()> {
        let mut writes = vec![(
            self.perimeter_path(perimeter_id),
            toml::to_string_pretty(&PerimetersFile {
                perimeters: perimeters.to_vec(),
            }),
        )];
        writes.extend(sessions.iter().map(|session| {
            (
                self.session_path(&session.key),
                toml::to_string_pretty(session),
            )
        }));
        if let Some((id, reference)) = reference {
            writes.push((self.reference_path(id), toml::to_string_pretty(reference)));
        }
        // Do not advance sync fingerprints: these edits still need exporting.
        Self::publish_toml_batch(writes, Vec::new())
    }

    /// Preflight all destinations, then publish atomically per file with rollback
    /// on I/O errors. The caller serializes mutations for the store.
    fn publish_toml_batch(
        writes: Vec<(PathBuf, std::result::Result<String, toml::ser::Error>)>,
        removals: Vec<PathBuf>,
    ) -> Result<()> {
        let writes = writes
            .into_iter()
            .map(|(path, content)| {
                content
                    .map(|content| (path, Some(content)))
                    .map_err(|error| BekiperError::Data(error.to_string()))
            })
            .chain(removals.into_iter().map(|path| Ok((path, None))))
            .map(|entry| {
                let (path, content) = entry?;
                let previous = match fs::read(&path) {
                    Ok(bytes) => Some(bytes),
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
                    Err(error) => return Err(error.into()),
                };
                Ok((path, content, previous))
            })
            .collect::<Result<Vec<_>>>()?;
        for (index, (path, content, _)) in writes.iter().enumerate() {
            let result = match content {
                Some(content) => atomic_write(path, content.as_bytes()),
                None => fs::remove_file(path).map_err(BekiperError::from),
            };
            if let Err(error) = result {
                for (path, _, previous) in writes[..=index].iter().rev() {
                    let restored = match previous {
                        Some(bytes) => atomic_write(path, bytes),
                        None if path.exists() => fs::remove_file(path).map_err(BekiperError::from),
                        None => Ok(()),
                    };
                    if let Err(rollback) = restored {
                        return Err(BekiperError::Data(format!(
                            "Save failed: {error}; rollback failed for {}: {rollback}",
                            path.display()
                        )));
                    }
                }
                return Err(error);
            }
        }
        Ok(())
    }
}
