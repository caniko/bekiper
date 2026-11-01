impl CanonicalAnnotationStore {
    pub fn resolve_session_perimeters(
        &self,
        session: &CanonicalSession,
    ) -> Result<Vec<PerimeterSpec>> {
        let path = self.perimeter_path(&session.perimeter_id);
        let mut specs = if path.exists() {
            self.load_perimeter(&session.perimeter_id)?
        } else {
            Vec::new()
        };
        if !specs.is_empty()
            && session.coordinate_space == CoordinateSpace::Source
            && let Some(target_image) = &session.target_image
        {
            self.apply_reference_translation(&session.perimeter_id, target_image, &mut specs)?;
        }
        Ok(specs)
    }

    pub fn load_reference_document(&self, perimeter_id: &str) -> Result<Option<ReferenceDocument>> {
        let path = self.reference_path(perimeter_id);
        if !path.exists() {
            return Ok(None);
        }
        let content = fs::read_to_string(&path)?;
        let reference: ReferenceDocument = toml::from_str(&content)
            .map_err(|e| BekiperError::Config(format!("invalid {}: {e}", path.display())))?;
        Ok(Some(reference))
    }

    pub fn write_reference_document(
        &self,
        perimeter_id: &str,
        reference: &ReferenceDocument,
    ) -> Result<()> {
        write_toml(&self.reference_path(perimeter_id), reference)
    }

    pub fn materialize_target_perimeter(
        &self,
        session_key: &str,
        resolved_specs: &[PerimeterSpec],
    ) -> Result<CanonicalSession> {
        let mut session = self.load_session(session_key)?;
        let source_id = session
            .source_perimeter_id
            .clone()
            .unwrap_or_else(|| session.perimeter_id.clone());
        let target_image = session.target_image.clone().ok_or_else(|| {
            BekiperError::Data(format!(
                "canonical session '{session_key}' has no target image to materialize"
            ))
        })?;
        let target_id = individual_target_perimeter_id(&source_id, &target_image, session_key);
        self.write_perimeter(&target_id, resolved_specs)?;
        session.perimeter_id = target_id;
        session.source_perimeter_id = Some(source_id);
        session.coordinate_space = CoordinateSpace::Target;
        self.write_session(&session)?;
        self.write_sync_state()?;
        Ok(session)
    }

    pub fn materialize_group_target_perimeter(
        &self,
        session_key: &str,
        resolved_specs: &[PerimeterSpec],
    ) -> Result<Vec<CanonicalSession>> {
        let session = self.load_session(session_key)?;
        let key = session.group_key().ok_or_else(|| {
            BekiperError::Data(format!(
                "canonical session '{session_key}' does not belong to a re-referenced group"
            ))
        })?;
        let target_id = group_target_perimeter_id(&key.source_perimeter_id, &key.target_image);
        self.write_perimeter(&target_id, resolved_specs)?;

        let mut updated = Vec::new();
        for mut member in self.group_members(&key)? {
            member.perimeter_id = target_id.clone();
            member.source_perimeter_id = Some(key.source_perimeter_id.clone());
            member.coordinate_space = CoordinateSpace::Target;
            self.write_session(&member)?;
            updated.push(member);
        }
        self.write_sync_state()?;
        Ok(updated)
    }
}
