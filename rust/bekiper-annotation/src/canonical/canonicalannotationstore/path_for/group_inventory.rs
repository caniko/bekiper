impl CanonicalAnnotationStore {
    pub fn split_session_from_group(
        &self,
        session_key: &str,
        resolved_specs: &[PerimeterSpec],
    ) -> Result<CanonicalSession> {
        self.materialize_target_perimeter(session_key, resolved_specs)
    }

    pub fn group_info_for_session(
        &self,
        session: &CanonicalSession,
    ) -> Result<Option<AnnotationGroupInfo>> {
        let Some(key) = session.group_key() else {
            return Ok(None);
        };
        Ok(Some(AnnotationGroupInfo {
            members: self.group_members(&key)?,
            key,
        }))
    }

    pub fn group_members(&self, key: &AnnotationGroupKey) -> Result<Vec<CanonicalSession>> {
        Ok(self
            .load_sessions()?
            .into_iter()
            .filter(|session| session.group_key().as_ref() == Some(key))
            .collect())
    }

    pub fn load_session(&self, session_key: &str) -> Result<CanonicalSession> {
        let path = self.session_path(session_key);
        let content = fs::read_to_string(&path)?;
        toml::from_str(&content)
            .map_err(|e| BekiperError::Config(format!("invalid {}: {e}", path.display())))
    }

    pub fn write_session(&self, session: &CanonicalSession) -> Result<()> {
        write_toml(&self.session_path(&session.key), session)
    }
}
