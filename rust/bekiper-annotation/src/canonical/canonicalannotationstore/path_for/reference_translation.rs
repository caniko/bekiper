impl CanonicalAnnotationStore {
    fn apply_reference_translation(
        &self,
        perimeter_id: &str,
        target_image: &str,
        specs: &mut [PerimeterSpec],
    ) -> Result<()> {
        let path = self.reference_path(perimeter_id);
        if !path.exists() {
            return Ok(());
        }
        let content = fs::read_to_string(&path)?;
        let refs: ReferenceDocument = toml::from_str(&content)
            .map_err(|e| BekiperError::Config(format!("invalid {}: {e}", path.display())))?;
        let Some(source) = refs.source else {
            return Ok(());
        };
        let source_stem = strip_png_extension(&source.image_name);
        if source_stem == target_image {
            return Ok(());
        }
        let Some(target) = refs.targets.get(target_image) else {
            return Ok(());
        };
        translate_perimeters(specs, target.x - source.x, target.y - source.y);
        Ok(())
    }
}
