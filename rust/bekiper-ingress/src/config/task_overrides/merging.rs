trait MergeOverrides {
    fn merged(&self, other: &Self) -> Self;
}

fn merge_optional_nested<T>(base: &Option<T>, other: &Option<T>) -> Option<T>
where
    T: MergeOverrides + Clone,
{
    match (base, other) {
        (Some(base), Some(other)) => Some(base.merged(other)),
        (None, Some(other)) => Some(other.clone()),
        (Some(base), None) => Some(base.clone()),
        (None, None) => None,
    }
}

impl MergeOverrides for ExplorationOverrides {
    fn merged(&self, other: &Self) -> Self {
        Self {
            nose_max_distance: other.nose_max_distance.or(self.nose_max_distance),
            contact_distance: other.contact_distance.or(self.contact_distance),
            max_angle_deg: other.max_angle_deg.or(self.max_angle_deg),
            max_nose_speed: other.max_nose_speed.or(self.max_nose_speed),
            speed_filter_mode: other.speed_filter_mode.or(self.speed_filter_mode),
            angle_filter_mode: other.angle_filter_mode.or(self.angle_filter_mode),
            min_bout_seconds: other.min_bout_seconds.or(self.min_bout_seconds),
            max_gap_seconds: other.max_gap_seconds.or(self.max_gap_seconds),
            strict_boundaries: other.strict_boundaries.or(self.strict_boundaries),
            min_total_exploration: other.min_total_exploration.or(self.min_total_exploration),
            analysis_window_seconds: other
                .analysis_window_seconds
                .or(self.analysis_window_seconds),
            window_mode: other.window_mode.or(self.window_mode),
            gap_density_threshold: other.gap_density_threshold.or(self.gap_density_threshold),
            perimeter_erosion_meters: other
                .perimeter_erosion_meters
                .or(self.perimeter_erosion_meters),
            exclude_posture: other.exclude_posture.or(self.exclude_posture),
            enable_qualia: other.enable_qualia.or(self.enable_qualia),
            qualia_body_proximity_distance: other
                .qualia_body_proximity_distance
                .or(self.qualia_body_proximity_distance),
            qualia_whisker_distance: other
                .qualia_whisker_distance
                .or(self.qualia_whisker_distance),
            qualia_whisker_angle: other.qualia_whisker_angle.or(self.qualia_whisker_angle),
            qualia_olfaction_distance: other
                .qualia_olfaction_distance
                .or(self.qualia_olfaction_distance),
            qualia_olfaction_angle: other.qualia_olfaction_angle.or(self.qualia_olfaction_angle),
            merge_rules: other
                .merge_rules
                .clone()
                .or_else(|| self.merge_rules.clone()),
            whisker: match (&self.whisker, &other.whisker) {
                (Some(base), Some(over)) => Some(base.merged(over)),
                (base, over) => over.clone().or_else(|| base.clone()),
            },
        }
    }
}

impl MergeOverrides for YMazeOverrides {
    fn merged(&self, other: &Self) -> Self {
        Self {
            temporal_policy: other.temporal_policy.or(self.temporal_policy),
            threshold_geometry: other.threshold_geometry.or(self.threshold_geometry),
            proximal_threshold: other.proximal_threshold.or(self.proximal_threshold),
            principal_thresholds: other
                .principal_thresholds
                .clone()
                .or_else(|| self.principal_thresholds.clone()),
            start_mode: other.start_mode.or(self.start_mode),
            apply_dwell_filter: other.apply_dwell_filter.or(self.apply_dwell_filter),
            arm_membership_hysteresis: other
                .arm_membership_hysteresis
                .or(self.arm_membership_hysteresis),
            zone_max_gap_seconds: other.zone_max_gap_seconds.or(self.zone_max_gap_seconds),
            pose_gap_fill_seconds: other.pose_gap_fill_seconds.or(self.pose_gap_fill_seconds),
            zone_occupancy: other.zone_occupancy.or(self.zone_occupancy),
        }
    }
}

impl MergeOverrides for RearingOverrides {
    fn merged(&self, other: &Self) -> Self {
        Self {
            max_speed: other.max_speed.or(self.max_speed),
            max_body_length_ratio: other.max_body_length_ratio.or(self.max_body_length_ratio),
            max_body_length: other.max_body_length.or(self.max_body_length),
            forepaw_proximity: other.forepaw_proximity.or(self.forepaw_proximity),
        }
    }
}

impl MergeOverrides for GroomingOverrides {
    fn merged(&self, other: &Self) -> Self {
        Self {
            max_speed: other.max_speed.or(self.max_speed),
            max_body_length: other.max_body_length.or(self.max_body_length),
            min_body_length: other.min_body_length.or(self.min_body_length),
            forepaw_proximity: other.forepaw_proximity.or(self.forepaw_proximity),
        }
    }
}
