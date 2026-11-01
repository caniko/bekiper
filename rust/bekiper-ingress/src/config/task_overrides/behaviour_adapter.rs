impl bekiper_behaviour::TaskOverrides for TaskParamsOverrides {
    fn exploration_nose_max_distance(&self) -> Option<f64> {
        self.exploration.as_ref()?.nose_max_distance
    }
    fn exploration_contact_distance(&self) -> Option<f64> {
        self.exploration.as_ref()?.contact_distance
    }
    fn exploration_max_angle_deg(&self) -> Option<f64> {
        self.exploration.as_ref()?.max_angle_deg
    }
    fn exploration_max_nose_speed(&self) -> Option<f64> {
        self.exploration.as_ref()?.max_nose_speed
    }
    fn exploration_speed_filter_mode(&self) -> Option<BoutFilterMode> {
        self.exploration.as_ref()?.speed_filter_mode
    }
    fn exploration_angle_filter_mode(&self) -> Option<BoutFilterMode> {
        self.exploration.as_ref()?.angle_filter_mode
    }
    fn exploration_min_bout_seconds(&self) -> Option<f64> {
        self.exploration.as_ref()?.min_bout_seconds
    }
    fn exploration_max_gap_seconds(&self) -> Option<f64> {
        self.exploration.as_ref()?.max_gap_seconds
    }
    fn exploration_strict_boundaries(&self) -> Option<bool> {
        self.exploration.as_ref()?.strict_boundaries
    }
    fn exploration_min_total_exploration(&self) -> Option<f64> {
        self.exploration.as_ref()?.min_total_exploration
    }
    fn exploration_analysis_window_seconds(&self) -> Option<f64> {
        self.exploration.as_ref()?.analysis_window_seconds
    }
    fn exploration_window_mode(&self) -> Option<NortWindowMode> {
        self.exploration.as_ref()?.window_mode
    }
    fn exploration_perimeter_erosion_meters(&self) -> Option<f64> {
        self.exploration.as_ref()?.perimeter_erosion_meters
    }
    fn exploration_exclude_posture(&self) -> Option<bool> {
        self.exploration.as_ref()?.exclude_posture
    }
    fn exploration_gap_density_threshold(&self) -> Option<f64> {
        self.exploration.as_ref()?.gap_density_threshold
    }

    fn exploration_whisker(
        &self,
    ) -> Option<bekiper_feature::heuristic::solo::whisker_config::WhiskerOverrides> {
        self.exploration.as_ref()?.whisker.clone()
    }

    fn qualia_enable(&self) -> Option<bool> {
        self.exploration.as_ref()?.enable_qualia
    }
    fn qualia_body_proximity_distance(&self) -> Option<f64> {
        self.exploration.as_ref()?.qualia_body_proximity_distance
    }
    fn qualia_whisker_distance(&self) -> Option<f64> {
        self.exploration.as_ref()?.qualia_whisker_distance
    }
    fn qualia_whisker_angle(&self) -> Option<f64> {
        self.exploration.as_ref()?.qualia_whisker_angle
    }
    fn qualia_olfaction_distance(&self) -> Option<f64> {
        self.exploration.as_ref()?.qualia_olfaction_distance
    }
    fn qualia_olfaction_angle(&self) -> Option<f64> {
        self.exploration.as_ref()?.qualia_olfaction_angle
    }
    fn qualia_merge_rules(
        &self,
    ) -> Option<
        std::collections::HashMap<String, bekiper_feature::heuristic::combined::merge::MergeRule>,
    > {
        self.exploration.as_ref()?.merge_rules.clone()
    }

    fn ymaze_threshold_geometry(
        &self,
    ) -> Option<bekiper_behaviour::radial_arm::YMazeThresholdGeometry> {
        self.ymaze.as_ref()?.threshold_geometry
    }
    fn ymaze_proximal_threshold(&self) -> Option<f64> {
        self.ymaze.as_ref()?.proximal_threshold
    }
    fn ymaze_principal_thresholds(&self) -> Option<Vec<f64>> {
        self.ymaze.as_ref()?.principal_thresholds.clone()
    }
    fn ymaze_temporal_policy(&self) -> Option<bekiper_feature::tolerance::TemporalPolicy> {
        self.ymaze.as_ref()?.temporal_policy
    }
    fn ymaze_start_mode(&self) -> Option<bekiper_behaviour::radial_arm::YMazeStartMode> {
        self.ymaze.as_ref()?.start_mode
    }
    fn ymaze_apply_dwell_filter(&self) -> Option<bool> {
        self.ymaze.as_ref()?.apply_dwell_filter
    }
    fn ymaze_arm_membership_hysteresis(&self) -> Option<bool> {
        self.ymaze.as_ref()?.arm_membership_hysteresis
    }
    fn ymaze_zone_max_gap_seconds(&self) -> Option<f64> {
        self.ymaze.as_ref()?.zone_max_gap_seconds
    }
    fn ymaze_pose_gap_fill_seconds(&self) -> Option<f64> {
        self.ymaze.as_ref()?.pose_gap_fill_seconds
    }
    fn ymaze_zone_occupancy(&self) -> Option<bekiper_feature::zone::ZoneOccupancy> {
        self.ymaze.as_ref()?.zone_occupancy
    }

    fn rearing_max_speed(&self) -> Option<f64> {
        self.rearing.as_ref()?.max_speed
    }
    fn rearing_max_body_length_ratio(&self) -> Option<f64> {
        self.rearing.as_ref()?.max_body_length_ratio
    }
    fn rearing_max_body_length(&self) -> Option<f64> {
        self.rearing.as_ref()?.max_body_length
    }
    fn rearing_forepaw_proximity(&self) -> Option<f64> {
        self.rearing.as_ref()?.forepaw_proximity
    }

    fn grooming_max_speed(&self) -> Option<f64> {
        self.grooming.as_ref()?.max_speed
    }
    fn grooming_max_body_length(&self) -> Option<f64> {
        self.grooming.as_ref()?.max_body_length
    }
    fn grooming_min_body_length(&self) -> Option<f64> {
        self.grooming.as_ref()?.min_body_length
    }
    fn grooming_forepaw_proximity(&self) -> Option<f64> {
        self.grooming.as_ref()?.forepaw_proximity
    }
}
