impl<S> ExplorationHeuristic<S>
where
    S: Expandable + ClosestPoint + Clone + Send + Sync + 'static,
    S::Output: Clone + 'static,
{
    pub fn new(
        object_perimeter: Perimeter<S>,
        object_label: impl Into<Label>,
        object_centroid: (f64, f64),
        fps: f64,
    ) -> Self {
        Self {
            object_perimeter,
            object_label: object_label.into(),
            object_centroid,
            centroid_coords: CoordinateColumns::new("mouse_center_x", "mouse_center_y"),
            nose_coords: CoordinateColumns::new("nose_x", "nose_y"),
            head_base_coords: CoordinateColumns::new("center_ear_x", "center_ear_y"),
            footprint_erosion_meters: 0.0,
            nose_max_distance: DEFAULT_NOSE_MAX_DISTANCE,
            contact_distance: DEFAULT_CONTACT_DISTANCE,
            max_angle_deg: DEFAULT_MAX_ANGLE_DEG,
            max_nose_speed: DEFAULT_MAX_NOSE_SPEED,
            speed_filter_mode: BoutFilterMode::PerFrame,
            angle_filter_mode: BoutFilterMode::PerFrame,
            min_bout_seconds: DEFAULT_MIN_BOUT_SECONDS,
            max_gap_seconds: DEFAULT_MAX_GAP_SECONDS,
            strict_boundaries: false,
            gap_density_threshold: DEFAULT_GAP_DENSITY_THRESHOLD,
            posture_col: None,
            whisker: None,
            fps,
            shared: None,
        }
    }

    /// Column name for the raw (pre-tolerance) exploration boolean.
    fn raw_col(&self) -> String {
        format!("exploring_{}_raw", self.object_label)
    }

    fn heuristic_name(&self) -> String {
        format!("exploring_{}", self.object_label)
    }

    fn outside_col(&self) -> String {
        diag_column_name(&self.heuristic_name(), "outside_object")
    }

    fn nose_near_col(&self) -> String {
        diag_column_name(&self.heuristic_name(), "nose_near")
    }

    fn nose_contact_col(&self) -> String {
        diag_column_name(&self.heuristic_name(), "nose_contact")
    }

    fn within_angle_col(&self) -> String {
        diag_column_name(&self.heuristic_name(), "within_angle")
    }

    fn contact_or_angle_col(&self) -> String {
        diag_column_name(&self.heuristic_name(), "contact_or_angle")
    }

    fn within_angle_min_deg_col(&self) -> String {
        diag_column_name(&self.heuristic_name(), "within_angle_deg")
    }

    fn within_angle_margin_deg_col(&self) -> String {
        diag_column_name(&self.heuristic_name(), "within_angle_margin_deg")
    }

    fn within_angle_target_x_col(&self) -> String {
        diag_column_name(&self.heuristic_name(), "within_angle_target_x")
    }

    fn within_angle_target_y_col(&self) -> String {
        diag_column_name(&self.heuristic_name(), "within_angle_target_y")
    }

    fn whisker_hit_col(&self) -> String {
        diag_column_name(&self.heuristic_name(), "whisker_hit")
    }

    fn whisker_side_col(&self) -> String {
        diag_column_name(&self.heuristic_name(), "whisker_side")
    }

    fn whisker_nearest_distance_col(&self) -> String {
        diag_column_name(&self.heuristic_name(), "whisker_nearest_hit_distance_m")
    }

    fn whisker_candidate_col(&self) -> String {
        diag_column_name(&self.heuristic_name(), "whisker_candidate")
    }

    /// Per-frame column whose true frames rescue a bout from the median-angle
    /// filter: nose contact, plus whisker candidates when the fan is enabled.
    pub fn bout_contact_col(&self) -> String {
        if self.whisker.is_some() {
            diag_column_name(&self.heuristic_name(), "bout_contact")
        } else {
            self.nose_contact_col()
        }
    }

    fn slow_nose_col(&self) -> String {
        diag_column_name(&self.heuristic_name(), "slow_nose")
    }

    fn nose_speed_mps_col(&self) -> String {
        diag_column_name(&self.heuristic_name(), "nose_speed_mps")
    }

    pub fn raw_diag_col(&self) -> String {
        diag_column_name(&self.heuristic_name(), "raw_combined")
    }

    pub fn final_toleranced_col(&self) -> String {
        diag_column_name(&self.heuristic_name(), "final_toleranced")
    }

    pub fn attribution_summary(
        &self,
        df: &DataFrame,
        timing: &bekiper_core::video::FrameTiming,
    ) -> ExplorationAttribution {
        let clock = BoutTiming::frames(timing, df.height()).expect("valid exploration timing");
        self.attribution_summary_with_timing(df, &clock)
    }

    pub fn attribution_summary_with_timing(
        &self,
        df: &DataFrame,
        timing: &BoutTiming,
    ) -> ExplorationAttribution {
        let raw =
            bool_column(df, &self.raw_col()).or_else(|| bool_column(df, &self.raw_diag_col()));
        let final_toleranced = bool_column(df, &self.final_toleranced_col());
        let nose_speed = df
            .column(&self.nose_speed_mps_col())
            .ok()
            .and_then(|series| series.f64().ok());
        let angle = df
            .column(&self.within_angle_min_deg_col())
            .ok()
            .and_then(|series| series.f64().ok());
        let diagnostics = raw
            .as_ref()
            .map(|mask| {
                bout_filter_diagnostics(
                    mask,
                    final_toleranced.as_ref(),
                    select_exploration_bouts_for_column(
                        df,
                        &self.raw_col(),
                        mask,
                        timing,
                        self.min_bout_seconds,
                        (self.max_gap_seconds, self.strict_boundaries),
                        self.gap_density_threshold,
                    ),
                    BoutFilterSpec {
                        strict_boundaries: self.strict_boundaries,
                        speed_mode: self.speed_filter_mode,
                        angle_mode: self.angle_filter_mode,
                        max_nose_speed: self.max_nose_speed,
                        max_angle_deg: self.max_angle_deg,
                        nose_speed_mps: nose_speed,
                        angle_deg: angle,
                        nose_contact: bool_column(df, &self.bout_contact_col()).as_ref(),
                    },
                )
            })
            .unwrap_or_default();
        // Frames outside the object with the nose near it, split by which
        // criterion rejected them.
        let near = [self.outside_col(), self.nose_near_col()];
        let count = |required: &[String], rejected: &[&str]| -> u32 {
            let columns: Option<Vec<_>> =
                required.iter().map(|name| bool_column(df, name)).collect();
            let Some(columns) = columns else { return 0 };
            let flags: Vec<_> = rejected.iter().map(|name| bool_column(df, name)).collect();
            (0..df.height())
                .filter(|&i| {
                    columns.iter().all(|c| c.get(i).unwrap_or(false))
                        && flags
                            .iter()
                            .all(|c| !c.as_ref().is_some_and(|c| c.get(i).unwrap_or(false)))
                })
                .count() as u32
        };
        let contact = self.nose_contact_col();
        let angle = self.within_angle_col();
        let removed_by_angle = count(&near, &[&contact, &angle]);
        let removed_by_posture = self.posture_col.as_ref().map_or(0, |posture| {
            let required = [
                near[0].clone(),
                near[1].clone(),
                contact.clone(),
                posture.clone(),
            ];
            count(&required, &[&angle])
        });
        ExplorationAttribution {
            frames_in_window: df.height() as u64,
            frames_pose_ok: count_pose_ok(df, &self.nose_coords),
            frames_outside_object: count_true_column(df, &self.outside_col()),
            frames_nose_near: count_true_column(df, &self.nose_near_col()),
            frames_nose_contact: count_true_column(df, &self.nose_contact_col()),
            frames_within_angle: count_true_column(df, &self.within_angle_col()),
            frames_contact_or_angle: count_true_column(df, &self.contact_or_angle_col()),
            frames_slow_nose: count_true_column(df, &self.slow_nose_col()),
            frames_all_gates_raw: raw
                .as_ref()
                .map(|mask| mask.sum().unwrap_or(0) as u64)
                .unwrap_or(0),
            frames_after_tolerance: final_toleranced
                .as_ref()
                .map(|mask| mask.sum().unwrap_or(0) as u64)
                .unwrap_or(0),
            bouts_pre_filter: raw.as_ref().map(count_bouts).unwrap_or(0),
            bouts_post_filter: final_toleranced.as_ref().map(count_bouts).unwrap_or(0),
            bout_lengths_ms_pre: raw
                .as_ref()
                .map(|mask| bout_lengths_ms(mask, timing))
                .unwrap_or_default(),
            bout_lengths_ms_post: final_toleranced
                .as_ref()
                .map(|mask| bout_lengths_ms(mask, timing))
                .unwrap_or_default(),
            candidate_bouts_pre_merge: diagnostics.candidate_bouts_pre_merge,
            bouts_post_merge: diagnostics.bouts_post_merge,
            bouts_removed_by_duration_filter: diagnostics.removed_by_duration,
            bouts_removed_by_speed_filter: diagnostics.removed_by_speed,
            final_valid_bouts: diagnostics.final_valid_bouts,
            bouts_removed_by_angle_filter: diagnostics.removed_by_angle,
            candidate_frames_removed_by_angle_criterion: removed_by_angle,
            candidate_frames_removed_by_posture: removed_by_posture,
            median_nose_speed_per_bout: diagnostics.per_bout_median_nose_speed,
            median_head_angle_per_bout: diagnostics.per_bout_median_head_angle_deg,
            bouts_rescued_by_contact_override: diagnostics.bouts_rescued_by_contact_override,
        }
    }
}
