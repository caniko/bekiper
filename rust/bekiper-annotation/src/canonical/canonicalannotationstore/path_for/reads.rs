impl CanonicalAnnotationStore {
    pub fn resolve(&self, ctx: &VideoContext) -> Result<Option<Vec<PerimeterSpec>>> {
        let Some(session) = self.resolve_session(ctx)? else {
            return Ok(None);
        };
        let mut specs = self.load_perimeter(&session.perimeter_id)?;
        if session.coordinate_space == CoordinateSpace::Source
            && let Some(target_image) = &session.target_image
        {
            self.apply_reference_translation(&session.perimeter_id, target_image, &mut specs)?;
        }
        Ok(Some(specs))
    }

    pub fn resolve_session(&self, ctx: &VideoContext) -> Result<Option<CanonicalSession>> {
        for session in self.load_sessions()? {
            if session.matches_context(ctx) {
                return Ok(Some(session));
            }
        }
        Ok(None)
    }

    pub fn load_sessions(&self) -> Result<Vec<CanonicalSession>> {
        let sessions_dir = self.root.join("sessions");
        if !sessions_dir.exists() {
            return Ok(Vec::new());
        }
        let mut sessions = Vec::new();
        for path in sorted_files(&sessions_dir)? {
            if path.extension().and_then(|ext| ext.to_str()) != Some("toml") {
                continue;
            }
            let content = fs::read_to_string(&path)?;
            let session: CanonicalSession = toml::from_str(&content)
                .map_err(|e| BekiperError::Config(format!("invalid {}: {e}", path.display())))?;
            sessions.push(session);
        }
        Ok(sessions)
    }

    pub fn load_perimeter(&self, perimeter_id: &str) -> Result<Vec<PerimeterSpec>> {
        let path = self.perimeter_path(perimeter_id);
        let content = fs::read_to_string(&path)?;
        let file: PerimetersFile = toml::from_str(&content)
            .map_err(|e| BekiperError::Config(format!("invalid {}: {e}", path.display())))?;
        Ok(file.perimeters)
    }
}
