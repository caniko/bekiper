impl TaskParamsOverrides {
    pub fn with_pose_label_defaults(model: PoseLabelModel) -> Self {
        let mut overrides = Self::default();
        overrides.fill_pose_label_defaults(model);
        overrides
    }

    pub fn fill_pose_label_defaults(&mut self, model: PoseLabelModel) {
        if matches!(model, PoseLabelModel::ImenDlc) {
            // Default body centroid is the nose↔tail midpoint, matching the
            // behavioural-spec sense of "body centroid" used for OFT/NORT
            // zone-entry tracking. Requires a `[[pipeline.midpoint_groups]]`
            // entry that produces `body_center_x/y`.
            self.centroid_coords
                .get_or_insert_with(|| bekiper_core::types::CoordinateColumns {
                    x: "body_center_x".into(),
                    y: "body_center_y".into(),
                });
            self.head_coords
                .get_or_insert_with(|| bekiper_core::types::CoordinateColumns {
                    x: "center_ear_x".into(),
                    y: "center_ear_y".into(),
                });
            self.tail_coords
                .get_or_insert_with(|| bekiper_core::types::CoordinateColumns {
                    x: "tail_x".into(),
                    y: "tail_y".into(),
                });
        }
    }

    /// Merge two override layers, preferring `other` where it sets a value.
    ///
    /// Nested task-specific sections merge field-by-field, so a variant can
    /// override one Y-maze threshold without clearing another inherited setting.
    /// Map fields, such as `exploration.merge_rules`, are replaced as whole
    /// values when `other` provides them.
    pub fn merged(&self, other: &Self) -> Self {
        Self {
            freezing_window_seconds: other
                .freezing_window_seconds
                .or(self.freezing_window_seconds),
            freezing_displacement_threshold: other
                .freezing_displacement_threshold
                .or(self.freezing_displacement_threshold),
            min_dwell_seconds: other.min_dwell_seconds.or(self.min_dwell_seconds),
            oft_window_seconds: other.oft_window_seconds.or(self.oft_window_seconds),
            peripheral_band_fraction: other
                .peripheral_band_fraction
                .or(self.peripheral_band_fraction),
            centroid_coords: other
                .centroid_coords
                .clone()
                .or_else(|| self.centroid_coords.clone()),
            nose_coords: other
                .nose_coords
                .clone()
                .or_else(|| self.nose_coords.clone()),
            head_coords: other
                .head_coords
                .clone()
                .or_else(|| self.head_coords.clone()),
            tail_coords: other
                .tail_coords
                .clone()
                .or_else(|| self.tail_coords.clone()),
            exploration: merge_optional_nested(&self.exploration, &other.exploration),
            ymaze: merge_optional_nested(&self.ymaze, &other.ymaze),
            rearing: merge_optional_nested(&self.rearing, &other.rearing),
            grooming: merge_optional_nested(&self.grooming, &other.grooming),
        }
    }

    /// Apply overrides to a `TaskParams`, returning the modified params.
    pub fn apply(
        &self,
        mut params: bekiper_behaviour::TaskParams,
    ) -> bekiper_behaviour::TaskParams {
        if let Some(v) = self.freezing_window_seconds {
            params.freezing_window_seconds = v;
        }
        if let Some(v) = self.freezing_displacement_threshold {
            params.freezing_displacement_threshold = v;
        }
        if let Some(v) = self.min_dwell_seconds {
            params.min_dwell_seconds = v;
        }
        if let Some(v) = self.oft_window_seconds {
            params.oft_window_seconds = v;
        }
        if let Some(v) = self.peripheral_band_fraction {
            params.peripheral_band_fraction = v;
        }
        if let Some(ref v) = self.centroid_coords {
            params.centroid_coords = v.clone();
        }
        if let Some(ref v) = self.nose_coords {
            params.nose_coords = v.clone();
        }
        if let Some(ref v) = self.head_coords {
            params.head_coords = v.clone();
        }
        if let Some(ref v) = self.tail_coords {
            params.tail_coords = v.clone();
        }
        params
    }
}
