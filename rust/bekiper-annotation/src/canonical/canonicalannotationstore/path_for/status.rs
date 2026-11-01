impl CanonicalAnnotationStore {
    pub fn status(annotation_dir: &Path) -> Result<StoreStatus> {
        let canonical_exists = Self::exists(annotation_dir);
        let (legacy_changed, canonical_changed) = change_state(annotation_dir)?;
        let conflicts_dir = Self::path_for(annotation_dir).join("conflicts");
        let conflicts = if conflicts_dir.exists() {
            sorted_files(&conflicts_dir)?
        } else {
            Vec::new()
        };
        Ok(StoreStatus {
            canonical_exists,
            legacy_changed,
            canonical_changed,
            preferred_source: if canonical_exists {
                "canonical"
            } else {
                "legacy"
            },
            conflicts,
        })
    }
}
