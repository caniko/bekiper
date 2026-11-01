impl ObjectRecognition {
    fn full_analyze_inner(
        &self,
        lf: LazyFrame,
        timing: &FrameTiming,
    ) -> (NortResult, Vec<Box<dyn Heuristic>>, DataFrame) {
        let heuristics = self.heuristics();
        let evaluated = heuristics.iter().fold(lf, |lf, h| h.evaluate(lf));
        let full_df = evaluated.collect().expect("failed to collect NORT results");
        self.finish_analysis(heuristics, full_df, timing)
    }

    /// Everything after heuristic evaluation (overlap resolution, windowing,
    /// bout selection, metrics). `full_df` holds this variant's evaluated columns.
    fn finish_analysis(
        &self,
        heuristics: Vec<Box<dyn Heuristic>>,
        mut full_df: DataFrame,
        timing: &FrameTiming,
    ) -> (NortResult, Vec<Box<dyn Heuristic>>, DataFrame) {
        let full_clock = BoutTiming::frames(timing, full_df.height()).expect("valid NORT timing");
        let actual_perimeters = self.actual_object_perimeters();
        // Nose-to-object distance is measured to the real outline; the
        // erosion applies only to the centroid-outside footprint check.
        let actual_outline_perimeters: Vec<Perimeter<ObjectShape>> = actual_perimeters
            .iter()
            .map(|&perimeter| perimeter.clone())
            .collect();

        let raw_cols: Vec<String> = actual_perimeters
            .iter()
            .map(|p| format!("exploring_{}_raw", p.label))
            .collect();

        // Resolve overlapping explorations: when nose is near both objects
        // on the same frame, assign to the nearest object only.
        if actual_outline_perimeters.len() >= 2 {
            resolve_overlapping_explorations(&mut full_df, &actual_outline_perimeters, &raw_cols);
        }

        full_df = add_object_recognition_zone_columns(
            full_df,
            self.arena_perimeter().as_ref(),
            &self.params.centroid_coords,
        )
        .expect("failed to add object-recognition zone columns");

        for perimeter in &actual_perimeters {
            let raw_col = format!("exploring_{}_raw", perimeter.label);
            let (_, _, toleranced) = object_exploration_with_timing(
                &full_df,
                &raw_col,
                &full_clock,
                self.min_bout_seconds,
                self.max_gap_seconds,
                self.gap_density_threshold,
                self.bout_filter_spec(&full_df, &raw_col),
            );
            let final_col = diag_column_name(
                &format!("exploring_{}", perimeter.label),
                "final_toleranced",
            );
            persist_boolean_mask(&mut full_df, &final_col, &toleranced);
        }

        self.persist_inspection_comparisons(&mut full_df, &full_clock);
        let first_approach = find_first_exploration_frame(&full_df, &raw_cols);
        let selection = select_analysis_window(
            self.window_mode,
            full_df.height(),
            self.analysis_window_seconds,
            self.params.fps,
            first_approach,
        );
        let df = full_df.slice(
            selection.start_frame as i64,
            selection
                .end_frame_exclusive
                .saturating_sub(selection.start_frame),
        );

        let mut exploration_times = Vec::new();
        let clock = full_clock.sliced(selection.start_frame, selection.end_frame_exclusive);
        let mut bout_counts = Vec::new();
        let mut toleranced_masks: Vec<BooleanChunked> = Vec::new();

        for perimeter in &actual_perimeters {
            let raw_col = format!("exploring_{}_raw", perimeter.label);
            let (seconds, bouts, toleranced) = object_exploration_with_timing(
                &df,
                &raw_col,
                &clock,
                self.min_bout_seconds,
                self.max_gap_seconds,
                self.gap_density_threshold,
                self.bout_filter_spec(&df, &raw_col),
            );
            exploration_times.push(seconds);
            bout_counts.push(bouts);
            toleranced_masks.push(toleranced);
        }
        let mut attribution_df = df.clone();
        for (perimeter, toleranced) in actual_perimeters.iter().zip(toleranced_masks.iter()) {
            let final_col = diag_column_name(
                &format!("exploring_{}", perimeter.label),
                "final_toleranced",
            );
            persist_boolean_mask(&mut attribution_df, &final_col, toleranced);
        }

        let actual_object_labels: Vec<String> = actual_perimeters
            .iter()
            .map(|perimeter| perimeter.label.clone())
            .collect();

        let exploration_attributions: Vec<ExplorationAttribution> = actual_perimeters
            .iter()
            .map(|perimeter| {
                self.exploration_heuristic(perimeter)
                    .attribution_summary_with_timing(&attribution_df, &clock)
            })
            .collect();

        // Compute median speed during observation vs non-observation intervals.
        let (median_speed_observing, median_speed_not_observing) = if !toleranced_masks.is_empty() {
            // Combined mask: observing ANY object
            let observing = toleranced_masks
                .iter()
                .skip(1)
                .fold(toleranced_masks[0].clone(), |acc, m| &acc | m);

            // Frame-to-frame speed (m/s) from head coordinates
            let head_x = &self.params.head_coords.x;
            let head_y = &self.params.head_coords.y;
            let xs = df.column(head_x).unwrap().f64().unwrap();
            let ys = df.column(head_y).unwrap().f64().unwrap();

            let mut obs_speeds = Vec::new();
            let mut not_obs_speeds = Vec::new();
            for i in 1..df.height() {
                if let (Some(x), Some(y), Some(px), Some(py)) =
                    (xs.get(i), ys.get(i), xs.get(i - 1), ys.get(i - 1))
                {
                    let dx = x - px;
                    let dy = y - py;
                    let speed = (dx * dx + dy * dy).sqrt() * self.params.fps;
                    if observing.get(i).unwrap_or(false) {
                        obs_speeds.push(speed);
                    } else {
                        not_obs_speeds.push(speed);
                    }
                }
            }

            let median_obs = if obs_speeds.is_empty() {
                None
            } else {
                Float64Chunked::from_slice("obs".into(), &obs_speeds).median()
            };
            let median_not = if not_obs_speeds.is_empty() {
                None
            } else {
                Float64Chunked::from_slice("not_obs".into(), &not_obs_speeds).median()
            };
            (median_obs, median_not)
        } else {
            (None, None)
        };

        let freezing_seconds = toleranced_freezing_mask(&df, self.params.fps)
            .map(|mask| {
                clock.mask_seconds(
                    &mask
                        .into_iter()
                        .map(|v| v.unwrap_or(false))
                        .collect::<Vec<_>>(),
                )
            })
            .unwrap_or(0.0);

        let total_duration = clock.duration(0, clock.len());

        let total_distance = path_distance(
            &df,
            &self.params.head_coords.x,
            &self.params.head_coords.y,
            None,
        );
        let total_mean_speed = if total_duration > 0.0 {
            total_distance / total_duration
        } else {
            0.0
        };

        // Compute latency to center zone (if arena info is available).
        let latency_to_center = if self.center_perimeter.is_some() {
            let center_summary = zone_summary(
                &df,
                "in_center",
                &self.params.centroid_coords.x,
                &self.params.centroid_coords.y,
                self.params.fps,
                self.params.min_dwell_frames(),
            );
            center_summary.latency_seconds
        } else {
            None
        };

        // Compute side bias and object bias for 2-object paradigms.
        // Side bias: T_B - T_A / (T_B + T_A) — object A = left, B = right.
        // Object bias: T_A - T_B / (T_A + T_B) — for identical objects in T1.
        let (sb, ob) =
            if exploration_times.len() >= 2 && exploration_times[0] + exploration_times[1] > 0.0 {
                (
                    // NORT objects are ordered left to right (outline centroid
                    // x); post-analysis still requires its registered side mapping.
                    (!self.strict_boundaries)
                        .then(|| side_bias(exploration_times[0], exploration_times[1])),
                    Some(object_bias(exploration_times[0], exploration_times[1])),
                )
            } else {
                (None, None)
            };

        // Qualia decomposition: per-heuristic per-object statistics.
        let qualia = if self.enable_qualia && !actual_perimeters.is_empty() {
            let mut heuristic_types = vec!["body_proximity", "olfaction", "whisker_interaction"];
            if self.whisker_fan.is_some() {
                heuristic_types.push("whisker_fan");
            }
            let mut channels = Vec::new();
            let mut relative_bias = HashMap::new();
            let mut absolute_bias = HashMap::new();

            for ht in &heuristic_types {
                let mut per_object_seconds = Vec::new();
                let mut per_object_bouts = Vec::new();

                for perimeter in &actual_perimeters {
                    let col_name = format!("{ht}_{}", perimeter.label);
                    let (seconds, bouts, _) = object_exploration_with_timing(
                        &df,
                        &col_name,
                        &clock,
                        self.min_bout_seconds,
                        self.max_gap_seconds,
                        self.gap_density_threshold,
                        BoutFilterSpec::unfiltered(),
                    );
                    per_object_seconds.push(seconds);
                    per_object_bouts.push(bouts);
                }

                let channel_total: f64 = per_object_seconds.iter().sum();
                let rel: Vec<f64> = per_object_seconds
                    .iter()
                    .map(|s| {
                        if channel_total > 0.0 {
                            100.0 * s / channel_total
                        } else {
                            0.0
                        }
                    })
                    .collect();
                let abs: Vec<f64> = per_object_seconds
                    .iter()
                    .map(|s| {
                        if total_duration > 0.0 {
                            100.0 * s / total_duration
                        } else {
                            0.0
                        }
                    })
                    .collect();

                relative_bias.insert(ht.to_string(), rel);
                absolute_bias.insert(ht.to_string(), abs);

                channels.push(QualiaChannel {
                    heuristic_type: ht.to_string(),
                    per_object_seconds,
                    per_object_bouts,
                });
            }

            // Build observation sequence from exploration columns (already overlap-resolved).
            let observation_sequence = build_observation_sequence(&df, &raw_cols);
            let min_tolerance_frames = (self.min_bout_seconds * self.params.fps).round() as usize;
            let alternation_count = count_alternations(&observation_sequence, min_tolerance_frames);

            // Compute merged heuristic seconds per object.
            let mut merged = HashMap::new();
            for merge_name in self.sorted_merge_rule_names() {
                let mut per_object_secs = Vec::new();
                for perimeter in &actual_perimeters {
                    let col_name = format!("{merge_name}_{}", perimeter.label);
                    let (seconds, _, _) = object_exploration_with_timing(
                        &df,
                        &col_name,
                        &clock,
                        self.min_bout_seconds,
                        self.max_gap_seconds,
                        self.gap_density_threshold,
                        BoutFilterSpec::unfiltered(),
                    );
                    per_object_secs.push(seconds);
                }
                merged.insert(merge_name.to_string(), per_object_secs);
            }

            Some(QualiaDecomposition {
                channels,
                relative_bias,
                absolute_bias,
                observation_sequence,
                alternation_count,
                merged,
            })
        } else {
            None
        };

        let metric_ctx = ObjectRecognitionMetricContext {
            task: self,
            df: &df,
            clock: &clock,
            object_labels: actual_object_labels.clone(),
            observation_seconds: exploration_times.clone(),
            observation_bouts: bout_counts.clone(),
            arena: self.arena_perimeter(),
        };
        let mut metrics = ObjectRecognitionMetrics::default();
        let metric_units: [&dyn ObjectRecognitionMetricUnit; 5] = [
            &ObservationTotalsUnit,
            &PairwiseObservationUnit,
            &NoveltyIndexUnit,
            &QualiaBreakdownUnit,
            &ZoneMotionUnit,
        ];
        for unit in metric_units {
            unit.compute(&metric_ctx, &mut metrics);
        }

        let total_observation_seconds = metrics
            .observation_totals
            .as_ref()
            .map(|totals| totals.total_seconds)
            .unwrap_or(0.0);

        let result = NortResult {
            protocol: self.protocol,
            exploration_times,
            bout_counts,
            object_labels: actual_object_labels,
            indices: metrics.novelty_indices,
            observation_totals: metrics.observation_totals,
            pairwise_observations: metrics.pairwise_observations,
            qualia_rollups: metrics.qualia_rollups,
            zone_metrics: metrics.zone_metrics,
            exploration_attributions,
            scoring_provenance: Some(serde_json::json!({
                "zone_extension_m": self.nose_max_distance,
                "contact_distance_m": self.contact_distance,
                "body_centroid_outside_object_mask": true,
                "candidate_orientation_or_contact": true,
                "angle_filter_mode": self.angle_filter_mode,
                "angle_max_degrees": self.max_angle_deg,
                "speed_filter_mode": self.speed_filter_mode,
                "diagnostic_nose_speed_reference_m_per_s": self.max_nose_speed,
                "merge_before_duration": true,
                "merge_gap_seconds": self.max_gap_seconds,
                "merge_gap_comparison": if self.strict_boundaries { "<" } else { "<=" },
                "speed_comparison": if self.strict_boundaries { "<" } else { "<=" },
                "physical_side_mapping_required": self.strict_boundaries,
                "minimum_dwell_seconds": self.min_bout_seconds,
                "duration_comparison": ">=",
                "gap_density_threshold": self.gap_density_threshold,
                "gap_duration_policy": "include_gaps",
                "gap_median_policy": "include_gaps",
                "bout_contact_policy": if self.whisker_fan.is_some() {
                    bekiper_feature::heuristic::combined::exploration::BOUT_CONTACT_POLICY_WHISKER
                } else {
                    bekiper_feature::heuristic::combined::exploration::BOUT_CONTACT_POLICY
                },
                "timing_basis": if timing.timestamps.is_empty() { "frame_count_over_fps" } else { "frame_timestamps_nominal_last_frame" },
            }))
            .map(|mut provenance| {
                // Absent rather than null: TOML exports drop nulls, and archived
                // and whisker-free results must round-trip unchanged.
                if let Some(p) = self.whisker_fan {
                    let mut fan = serde_json::json!({
                        "enabled": true,
                        "apex_fraction": p.apex_fraction,
                        "half_angle_deg": p.half_angle_deg,
                        "length_m": p.length_m,
                        "requires_outside_object_centroid": true,
                        "posture_frames_need_head_angle": self.exclude_posture,
                    });
                    if let Some(age) = self.whisker_age_days {
                        fan["age_days"] = age.into();
                    }
                    if let Some(source) = &self.whisker_length_source {
                        fan["length_source"] = source.clone().into();
                    }
                    provenance["whisker_fan"] = fan;
                }
                provenance
            }),
            side_bias: sb,
            object_bias: ob,
            latency_to_center,
            freezing_seconds,
            total_distance_meters: total_distance,
            total_mean_speed,
            median_speed_observing,
            median_speed_not_observing,
            total_duration_seconds: total_duration,
            first_approach_frame: first_approach,
            first_object_approach_time_s: first_approach
                .map(|frame| full_clock.boundary(frame) - full_clock.boundary(0)),
            analysis_window_seconds: self.analysis_window_seconds,
            window_mode: selection.mode,
            window_start_frame: Some(selection.start_frame),
            window_end_frame_exclusive: Some(selection.end_frame_exclusive),
            excluded: false,
            passes_15s_exploration_threshold: total_observation_seconds
                >= self.min_total_exploration,
            qualia,
            evaluated_df: Some(full_df),
        };
        (result, heuristics, df)
    }
}
