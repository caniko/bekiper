impl CanonicalAnnotationStore {
    pub fn path_for(annotation_dir: &Path) -> PathBuf {
        annotation_dir.join(CANONICAL_DIR)
    }

    pub fn exists(annotation_dir: &Path) -> bool {
        Self::path_for(annotation_dir)
            .join("manifest.toml")
            .exists()
    }

    pub fn open(annotation_dir: &Path) -> Result<Option<Self>> {
        if Self::exists(annotation_dir) {
            Ok(Some(Self {
                annotation_dir: annotation_dir.to_path_buf(),
                root: Self::path_for(annotation_dir),
            }))
        } else {
            Ok(None)
        }
    }

    pub fn create_or_open(annotation_dir: &Path) -> Self {
        Self {
            annotation_dir: annotation_dir.to_path_buf(),
            root: Self::path_for(annotation_dir),
        }
    }
}
