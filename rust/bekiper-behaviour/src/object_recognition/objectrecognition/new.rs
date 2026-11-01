impl ObjectRecognition {
    /// Create an NORT task with the given object perimeters and video FPS,
    /// using default exploration thresholds.
    pub fn new(object_perimeters: Vec<Perimeter<ObjectShape>>, fps: f64) -> Self {
        let object_count = object_perimeters
            .iter()
            .filter(|perimeter| !is_context_label(&perimeter.label))
            .count();
        Self {
            params: TaskParams::new(fps),
            protocol: if object_count <= 2 {
                ObjectRecognitionProtocol::NortNovelty
            } else {
                ObjectRecognitionProtocol::OulTest
            },
            object_perimeters,
            center_perimeter: None,
            nose_max_distance: DEFAULT_NOSE_MAX_DISTANCE,
            contact_distance: DEFAULT_CONTACT_DISTANCE,
            max_angle_deg: DEFAULT_MAX_ANGLE_DEG,
            max_nose_speed: DEFAULT_MAX_NOSE_SPEED,
            speed_filter_mode: BoutFilterMode::PerFrame,
            strict_boundaries: false,
            angle_filter_mode: BoutFilterMode::PerFrame,
            min_bout_seconds: DEFAULT_MIN_BOUT_SECONDS,
            max_gap_seconds: DEFAULT_MAX_GAP_SECONDS,
            min_total_exploration: DEFAULT_MIN_TOTAL_EXPLORATION,
            analysis_window_seconds: DEFAULT_ANALYSIS_WINDOW_SECONDS,
            window_mode: NortWindowMode::Unknown,
            gap_density_threshold: DEFAULT_GAP_DENSITY_THRESHOLD,
            perimeter_erosion_meters: DEFAULT_PERIMETER_EROSION_METERS,
            exclude_posture: false,
            enable_qualia: true,
            qualia_body_proximity_distance: 0.05,
            qualia_whisker_distance: 0.035,
            qualia_whisker_angle: 45.0,
            qualia_olfaction_distance: 0.05,
            qualia_olfaction_angle: 60.0,
            merge_rules: HashMap::new(),
            whisker_fan: None,
            whisker_age_days: None,
            whisker_length_source: None,
            shared_geometry: HashMap::new(),
        }
    }

    /// Objects in output order. For NORT, A is the left object and B the
    /// right one (smaller outline centroid x first), whatever the annotation order.
    fn actual_object_perimeters(&self) -> Vec<&Perimeter<ObjectShape>> {
        let mut objects: Vec<_> = self
            .object_perimeters
            .iter()
            .filter(|perimeter| !is_context_label(&perimeter.label))
            .collect();
        if matches!(
            self.protocol,
            ObjectRecognitionProtocol::NortTraining | ObjectRecognitionProtocol::NortNovelty
        ) {
            objects.sort_by(|a, b| a.shape.centroid().0.total_cmp(&b.shape.centroid().0));
        }
        objects
    }

    /// Grooming (≥1 s) and rearing (≥0.5 s) proxies with the OFT defaults,
    /// combined into [`POSTURE_COLUMN`].
    fn posture_heuristics(&self) -> Vec<Box<dyn Heuristic>> {
        let p = &self.params;
        let mut rearing = RearingHeuristic::new(p.fps);
        let mut grooming = GroomingHeuristic::new(p.fps);
        (rearing.nose_x, rearing.nose_y) = (p.nose_coords.x.clone(), p.nose_coords.y.clone());
        (rearing.tail_base_x, rearing.tail_base_y) =
            (p.tail_coords.x.clone(), p.tail_coords.y.clone());
        (rearing.centroid_x, rearing.centroid_y) =
            (p.centroid_coords.x.clone(), p.centroid_coords.y.clone());
        (grooming.nose_x, grooming.nose_y) = (rearing.nose_x.clone(), rearing.nose_y.clone());
        (grooming.tail_base_x, grooming.tail_base_y) =
            (rearing.tail_base_x.clone(), rearing.tail_base_y.clone());
        (grooming.centroid_x, grooming.centroid_y) =
            (rearing.centroid_x.clone(), rearing.centroid_y.clone());
        let sources = vec![
            (grooming.name().to_string(), DEFAULT_OFT_GROOMING_MIN_BOUT_S),
            (rearing.name().to_string(), DEFAULT_OFT_REARING_MIN_BOUT_S),
        ];
        vec![
            Box::new(rearing),
            Box::new(grooming),
            Box::new(PostureBoutHeuristic {
                name: POSTURE_COLUMN.to_string(),
                sources,
                fps: p.fps,
            }),
        ]
    }

    /// The combined exploration heuristic for one object, with this task's
    /// thresholds and landmarks.
    fn exploration_heuristic(
        &self,
        perimeter: &Perimeter<ObjectShape>,
    ) -> ExplorationHeuristic<ObjectShape> {
        let mut heuristic = ExplorationHeuristic::new(
            perimeter.clone(),
            perimeter.label.as_str(),
            perimeter.shape.centroid(),
            self.params.fps,
        );
        heuristic.footprint_erosion_meters = self.perimeter_erosion_meters;
        heuristic.nose_max_distance = self.nose_max_distance;
        heuristic.contact_distance = self.contact_distance;
        heuristic.max_angle_deg = self.max_angle_deg;
        heuristic.max_nose_speed = self.max_nose_speed;
        heuristic.speed_filter_mode = self.speed_filter_mode;
        heuristic.strict_boundaries = self.strict_boundaries;
        heuristic.angle_filter_mode = self.angle_filter_mode;
        heuristic.min_bout_seconds = self.min_bout_seconds;
        heuristic.max_gap_seconds = self.max_gap_seconds;
        heuristic.gap_density_threshold = self.gap_density_threshold;
        heuristic.centroid_coords = self.params.centroid_coords.clone();
        heuristic.nose_coords = self.params.nose_coords.clone();
        heuristic.head_base_coords = self.params.head_coords.clone();
        if self.exclude_posture {
            heuristic.posture_col = Some(POSTURE_COLUMN.to_string());
        }
        heuristic.whisker = self
            .whisker_fan
            .map(|params| WhiskerFanSpec::new(params, &perimeter.shape));
        heuristic.shared = self.shared_geometry.get(perimeter.label.as_str()).cloned();
        heuristic
    }

    fn effective_perimeter(&self, perimeter: &Perimeter<ObjectShape>) -> Perimeter<ObjectShape> {
        if self.perimeter_erosion_meters > 0.0 {
            perimeter.expanded(-self.perimeter_erosion_meters)
        } else {
            perimeter.clone()
        }
    }

    fn sorted_merge_rule_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.merge_rules.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
    }

    fn arena_perimeter(&self) -> Option<Perimeter<ObjectShape>> {
        self.object_perimeters
            .iter()
            .find(|perimeter| perimeter.label == "arena")
            .cloned()
    }

    /// Run full NORT analysis.
    ///
    /// For the test phase: object_perimeters[0] = familiar, [1] = novel.
    /// For familiarization: both are identical (use side_bias/object_bias).
    /// Run full NORT analysis, returning the result and generic heuristic summaries.
    ///
    /// The summaries are computed on the windowed DataFrame (matching the detailed
    /// metrics), while `evaluated_df` contains the full trial for visualization.
    fn full_analyze_with_summaries(
        &self,
        lf: LazyFrame,
        timing: &FrameTiming,
    ) -> (NortResult, Vec<HeuristicSummary>) {
        let inner = self.full_analyze_inner(lf, timing);
        self.with_summaries(inner, timing)
    }

    fn with_summaries(
        &self,
        (result, _heuristics, windowed_df): (NortResult, Vec<Box<dyn Heuristic>>, DataFrame),
        timing: &FrameTiming,
    ) -> (NortResult, Vec<HeuristicSummary>) {
        let full_clock = BoutTiming::frames(
            timing,
            result.evaluated_df.as_ref().map_or(0, DataFrame::height),
        )
        .expect("valid NORT timing");
        let clock = full_clock.sliced(
            result.window_start_frame.unwrap_or(0),
            result
                .window_end_frame_exclusive
                .unwrap_or(full_clock.len()),
        );
        let summaries = self.summaries_from_windowed_df(&windowed_df, timing, &clock);
        (result, summaries)
    }

    fn summaries_from_windowed_df(
        &self,
        df: &DataFrame,
        timing: &FrameTiming,
        clock: &BoutTiming,
    ) -> Vec<HeuristicSummary> {
        let actual_perimeters = self.actual_object_perimeters();
        let mut summaries = Vec::new();

        for perimeter in &actual_perimeters {
            summaries.push(self.object_exploration_summary(
                df,
                clock,
                format!("exploring_{}", perimeter.label),
                format!("exploring_{}_raw", perimeter.label),
            ));
        }

        if self.enable_qualia {
            for perimeter in &actual_perimeters {
                let label = perimeter.label.as_str();
                summaries.push(self.object_exploration_summary(
                    df,
                    clock,
                    format!("body_proximity_{label}"),
                    format!("body_proximity_{label}"),
                ));
                summaries.push(self.object_exploration_summary(
                    df,
                    clock,
                    format!("olfaction_{label}"),
                    format!("olfaction_{label}"),
                ));
                summaries.push(self.object_exploration_summary(
                    df,
                    clock,
                    format!("whisker_interaction_{label}"),
                    format!("whisker_interaction_{label}"),
                ));
                for merge_name in self.sorted_merge_rule_names() {
                    summaries.push(self.object_exploration_summary(
                        df,
                        clock,
                        format!("{merge_name}_{label}"),
                        format!("{merge_name}_{label}"),
                    ));
                }
            }
        }

        if self.center_perimeter.is_some() && self.arena_perimeter().is_some() {
            summaries.push(self.zone_summary_entry(df, timing, "in_periphery"));
        }
        if self.center_perimeter.is_some() {
            summaries.push(self.zone_summary_entry(df, timing, "in_center"));
        }
        summaries.push(self.freezing_summary_entry(df, timing));

        summaries
    }

    fn object_exploration_summary(
        &self,
        df: &DataFrame,
        clock: &BoutTiming,
        name: String,
        column: String,
    ) -> HeuristicSummary {
        let filters = if is_combined_exploration_raw_column(&column) {
            self.bout_filter_spec(df, &column)
        } else {
            BoutFilterSpec::unfiltered()
        };
        let (seconds, _, mask) = object_exploration_with_timing(
            df,
            &column,
            clock,
            self.min_bout_seconds,
            self.max_gap_seconds,
            self.gap_density_threshold,
            filters,
        );
        HeuristicSummary {
            name,
            true_frames: mask.sum().unwrap_or(0) as u64,
            total_frames: mask.len() as u64,
            seconds,
        }
    }

    fn bout_filter_spec<'a>(&self, df: &'a DataFrame, raw_col: &str) -> BoutFilterSpec<'a> {
        let heuristic_name = raw_col.strip_suffix("_raw").unwrap_or(raw_col);
        let nose_speed_col = diag_column_name(heuristic_name, "nose_speed_mps");
        let angle_col = diag_column_name(heuristic_name, "within_angle_deg");
        // Whisker-enabled runs rescue bouts on nose contact OR a whisker hit.
        let nose_contact_col = diag_column_name(
            heuristic_name,
            if self.whisker_fan.is_some() {
                "bout_contact"
            } else {
                "nose_contact"
            },
        );
        BoutFilterSpec {
            strict_boundaries: self.strict_boundaries,
            speed_mode: self.speed_filter_mode,
            angle_mode: self.angle_filter_mode,
            max_nose_speed: self.max_nose_speed,
            max_angle_deg: self.max_angle_deg,
            nose_speed_mps: df
                .column(&nose_speed_col)
                .ok()
                .and_then(|series| series.f64().ok()),
            angle_deg: df
                .column(&angle_col)
                .ok()
                .and_then(|series| series.f64().ok()),
            nose_contact: df
                .column(&nose_contact_col)
                .ok()
                .and_then(|series| series.bool().ok()),
        }
    }

    fn zone_summary_entry(
        &self,
        df: &DataFrame,
        timing: &FrameTiming,
        zone_col: &str,
    ) -> HeuristicSummary {
        let mask = zone_toleranced_mask(df, zone_col, self.params.min_dwell_frames())
            .unwrap_or_else(|| {
                BooleanChunked::from_slice(zone_col.into(), &vec![false; df.height()])
            });
        summary_from_mask(zone_col.to_string(), &mask, timing)
    }

    fn freezing_summary_entry(&self, df: &DataFrame, timing: &FrameTiming) -> HeuristicSummary {
        let mask = toleranced_freezing_mask(df, self.params.fps).unwrap_or_else(|| {
            BooleanChunked::from_slice("freezing".into(), &vec![false; df.height()])
        });
        summary_from_mask("freezing".into(), &mask, timing)
    }

    pub fn full_analyze(&self, lf: LazyFrame) -> NortResult {
        self.full_analyze_inner(lf, &FrameTiming::cfr(self.params.fps))
            .0
    }
}

fn is_combined_exploration_raw_column(column: &str) -> bool {
    column.starts_with("exploring_") && column.ends_with("_raw")
}
