impl BehaviouralTask for YMaze {
    fn task_type(&self) -> TaskType {
        TaskType::YMaze
    }

    fn effective_parameters(&self) -> Option<serde_json::Value> {
        let mut value = self.params.effective_parameters();
        value["ymaze"] = serde_json::json!({
            "temporal_policy": self.effective_temporal_policy(),
            "threshold_geometry": self.threshold_geometry,
            "proximal_threshold": self.proximal_threshold,
            "principal_thresholds": self.principal_thresholds,
            "start_mode": self.start_mode,
            "apply_dwell_filter": self.apply_dwell_filter,
            "arm_membership_hysteresis": self.arm_membership_hysteresis,
            "zone_max_gap_seconds": self.zone_max_gap_seconds,
            "pose_gap_fill_seconds": self.pose_gap_fill_seconds,
            "zone_occupancy": self.zone_occupancy,
        });
        Some(value)
    }

    fn heuristics(&self) -> Vec<Box<dyn Heuristic>> {
        self.heuristics_with_hysteresis(self.arm_membership_hysteresis)
    }

    fn analyze_detailed(
        &self,
        lf: LazyFrame,
        timing: &bekiper_core::video::FrameTiming,
    ) -> (crate::AnalysisResult, Option<TaskDetail>) {
        let heuristics = self.heuristics();
        let decompositions = crate::decompositions_from_heuristics(&heuristics);
        let result = self.full_analyze_timed(lf, timing);
        let mut col_names: Vec<String> = self.arm_col_names();
        col_names.push("in_center".to_string());
        col_names.push("freezing".to_string());
        let col_refs: Vec<&str> = col_names.iter().map(|s| s.as_str()).collect();
        let summaries = crate::summaries_from_df(
            result
                .evaluated_df
                .as_ref()
                .expect("evaluated_df is always Some after full_analyze"),
            &col_refs,
            timing,
        );
        let analysis = crate::AnalysisResult {
            task_type: self.task_type(),
            summaries,
            decompositions,
            evaluated_df: result.evaluated_df.clone(),
        };
        (analysis, Some(TaskDetail::YMaze(result)))
    }
}
