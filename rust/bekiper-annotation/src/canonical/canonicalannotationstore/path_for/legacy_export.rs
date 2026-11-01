impl CanonicalAnnotationStore {
    pub fn export_legacy(&self) -> Result<()> {
        let plugin_dir = self.annotation_dir.join("plugin_files");
        fs::create_dir_all(&plugin_dir)?;
        for path in sorted_files(&self.root.join("perimeters"))? {
            if path.extension().and_then(|ext| ext.to_str()) != Some("toml") {
                continue;
            }
            let perimeter_id =
                path.file_stem()
                    .and_then(|stem| stem.to_str())
                    .ok_or_else(|| {
                        BekiperError::Data(format!("invalid perimeter path {}", path.display()))
                    })?;
            let perimeters = self.load_perimeter(perimeter_id)?;
            write_legacy_perimeter(&plugin_dir, perimeter_id, &perimeters)?;
        }
        for path in sorted_files(&self.root.join("references"))? {
            if path.extension().and_then(|ext| ext.to_str()) != Some("toml") {
                continue;
            }
            let perimeter_id =
                path.file_stem()
                    .and_then(|stem| stem.to_str())
                    .ok_or_else(|| {
                        BekiperError::Data(format!("invalid reference path {}", path.display()))
                    })?;
            let content = fs::read_to_string(&path)?;
            let reference: ReferenceDocument = toml::from_str(&content)
                .map_err(|e| BekiperError::Config(format!("invalid {}: {e}", path.display())))?;
            if let Some(source) = reference.source {
                write_reference_csv(
                    &plugin_dir.join(format!("reference-{perimeter_id}.csv")),
                    &[source],
                )?;
            }
            if !reference.targets.is_empty() {
                write_reference_csv(
                    &plugin_dir.join(format!("change_reference-{perimeter_id}.csv")),
                    &reference.targets.into_values().collect::<Vec<_>>(),
                )?;
            }
        }
        self.export_session_mappings()?;
        self.write_sync_state()?;
        Ok(())
    }
}
