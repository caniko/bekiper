/// `json!` turns non-finite floats into `null`, which the flat export drops;
/// keep them as "inf" / "-inf" / "nan" (e.g. `analysis_window_seconds` = full trial).
fn finite_or_text(v: f64) -> serde_json::Value {
    if v.is_finite() {
        serde_json::json!(v)
    } else if v.is_nan() {
        "nan".into()
    } else {
        (if v > 0.0 { "inf" } else { "-inf" }).into()
    }
}

impl BehaviouralTask for ObjectRecognition {
    fn task_type(&self) -> TaskType {
        TaskType::ObjectRecognition
    }

    fn analyze(&self, lf: LazyFrame, timing: &FrameTiming) -> crate::AnalysisResult {
        let heuristics = self.heuristics();
        let decompositions = self.inspection_decompositions(&heuristics);
        let (result, summaries) = self.full_analyze_with_summaries(lf, timing);
        crate::AnalysisResult {
            task_type: self.task_type(),
            summaries,
            decompositions,
            evaluated_df: result.evaluated_df.clone(),
        }
    }

    fn effective_parameters(&self) -> Option<serde_json::Value> {
        let mut value = self.params.effective_parameters();
        let whisker = match &self.whisker_fan {
            Some(fan) => serde_json::json!({
                "enabled": true,
                "apex_fraction": fan.apex_fraction,
                "half_angle_deg": fan.half_angle_deg,
                "length_m": fan.length_m,
                "age_days": self.whisker_age_days,
                "length_source": self.whisker_length_source,
            }),
            None => serde_json::json!({"enabled": false}),
        };
        value["exploration"] = serde_json::json!({
            "nose_max_distance": self.nose_max_distance,
            "contact_distance": self.contact_distance,
            "max_angle_deg": self.max_angle_deg,
            "max_nose_speed": finite_or_text(self.max_nose_speed),
            "speed_filter_mode": self.speed_filter_mode,
            "angle_filter_mode": self.angle_filter_mode,
            "strict_boundaries": self.strict_boundaries,
            "min_bout_seconds": self.min_bout_seconds,
            "max_gap_seconds": self.max_gap_seconds,
            "min_total_exploration": self.min_total_exploration,
            "analysis_window_seconds": finite_or_text(self.analysis_window_seconds),
            "window_mode": self.window_mode,
            "gap_density_threshold": self.gap_density_threshold,
            "perimeter_erosion_meters": self.perimeter_erosion_meters,
            "exclude_posture": self.exclude_posture,
            "enable_qualia": self.enable_qualia,
            "qualia_body_proximity_distance": self.qualia_body_proximity_distance,
            "qualia_whisker_distance": self.qualia_whisker_distance,
            "qualia_whisker_angle": self.qualia_whisker_angle,
            "qualia_olfaction_distance": self.qualia_olfaction_distance,
            "qualia_olfaction_angle": self.qualia_olfaction_angle,
            "merge_rules": self.merge_rules,
            "whisker": whisker,
        });
        Some(value)
    }

    fn heuristics(&self) -> Vec<Box<dyn Heuristic>> {
        let mut heuristics: Vec<Box<dyn Heuristic>> = Vec::new();
        let actual_perimeters = self.actual_object_perimeters();

        if self.exclude_posture {
            heuristics.extend(self.posture_heuristics());
        }
        for perimeter in &actual_perimeters {
            // The 1 cm erosion applies to the centroid footprint only; the
            // 3 cm / 0.5 cm nose zones are offsets of the real outline.
            heuristics.push(Box::new(self.exploration_heuristic(perimeter)));
        }

        // Qualia decomposition: per-object solo heuristics wrapped with
        // an outside-perimeter filter so the animal must be investigating
        // from outside the object, not from within.
        if self.enable_qualia {
            for perimeter in &actual_perimeters {
                let label = &perimeter.label;
                let effective = self.effective_perimeter(perimeter);

                let perimeter_label = Some(effective.label.clone());
                let head_label = coordinate_label_from_xy(&self.params.head_coords.x);
                let nose_label = coordinate_label_from_xy(&self.params.nose_coords.x);

                let body_name = format!("body_proximity_{label}");
                let body_raw_col = diag_column_name(&body_name, "raw_combined");
                let body_outside_col = diag_column_name(&body_name, "outside_object");
                let body_prox = BodyProximityHeuristic {
                    torso_coords: Some(self.params.centroid_coords.clone()),
                    perimeter: effective.clone(),
                    max_distance: self.qualia_body_proximity_distance,
                    head_coords: self.params.head_coords.clone(),
                    tail_coords: self.params.tail_coords.clone(),
                    name: body_name.clone(),
                    result_column: Some(body_raw_col.clone()),
                };
                let body_decomposition = HeuristicDecompositionSpec {
                    heuristic_name: body_name.clone(),
                    result_column: body_name.clone(),
                    coordinate_labels: vec![head_label.clone()],
                    perimeter_label: perimeter_label.clone(),
                    components: vec![
                        decomposition_component(
                            "center_ear_near",
                            diag_column_name(&body_name, "center_ear_near"),
                            vec![head_label.clone()],
                            perimeter_label.clone(),
                        ),
                        decomposition_component(
                            "tail_base_near",
                            diag_column_name(&body_name, "tail_base_near"),
                            vec![coordinate_label_from_xy(&self.params.tail_coords.x)],
                            perimeter_label.clone(),
                        ),
                        decomposition_component(
                            "torso_near",
                            diag_column_name(&body_name, "torso_near"),
                            vec![coordinate_label_from_xy(&self.params.centroid_coords.x)],
                            perimeter_label.clone(),
                        ),
                        decomposition_component(
                            "raw_combined",
                            body_raw_col.clone(),
                            vec![head_label.clone()],
                            perimeter_label.clone(),
                        ),
                        decomposition_component(
                            "outside_object",
                            body_outside_col.clone(),
                            vec![head_label.clone()],
                            perimeter_label.clone(),
                        ),
                    ],
                    show_final_result: true,
                };
                heuristics.push(Box::new(
                    CombinedHeuristic::new(
                        body_name.clone(),
                        body_prox,
                        vec![Box::new(OutsidePerimeterHeuristic {
                            perimeter: effective.clone(),
                            coords: self.params.centroid_coords.clone(),
                            name: body_outside_col,
                        })],
                    )
                    .with_decomposition(body_decomposition),
                ));

                let olfaction_name = format!("olfaction_{label}");
                let olfaction_raw_col = diag_column_name(&olfaction_name, "raw_combined");
                let olfaction_outside_col = diag_column_name(&olfaction_name, "outside_object");
                let olfaction = OlfactionHeuristic {
                    perimeter: effective.clone(),
                    max_distance: self.qualia_olfaction_distance,
                    max_angle: self.qualia_olfaction_angle,
                    contact_distance: 0.01,
                    name: olfaction_name.clone(),
                    result_column: Some(olfaction_raw_col.clone()),
                };
                let olfaction_decomposition = HeuristicDecompositionSpec {
                    heuristic_name: olfaction_name.clone(),
                    result_column: olfaction_name.clone(),
                    coordinate_labels: vec![nose_label.clone()],
                    perimeter_label: perimeter_label.clone(),
                    components: vec![
                        decomposition_component(
                            "nose_near",
                            diag_column_name(&olfaction_name, "nose_near"),
                            vec![nose_label.clone()],
                            perimeter_label.clone(),
                        ),
                        decomposition_component(
                            "ray_hits_object",
                            diag_column_name(&olfaction_name, "ray_hits_object"),
                            vec![nose_label.clone()],
                            perimeter_label.clone(),
                        ),
                        decomposition_component(
                            "nose_contact",
                            diag_column_name(&olfaction_name, "nose_contact"),
                            vec![nose_label.clone()],
                            perimeter_label.clone(),
                        ),
                        decomposition_component(
                            "raw_combined",
                            olfaction_raw_col.clone(),
                            vec![nose_label.clone()],
                            perimeter_label.clone(),
                        ),
                        decomposition_component(
                            "outside_object",
                            olfaction_outside_col.clone(),
                            vec![head_label.clone()],
                            perimeter_label.clone(),
                        ),
                    ],
                    show_final_result: true,
                };
                heuristics.push(Box::new(
                    CombinedHeuristic::new(
                        olfaction_name.clone(),
                        olfaction,
                        vec![Box::new(OutsidePerimeterHeuristic {
                            perimeter: effective.clone(),
                            coords: self.params.centroid_coords.clone(),
                            name: olfaction_outside_col,
                        })],
                    )
                    .with_decomposition(olfaction_decomposition),
                ));

                let whisker_name = format!("whisker_interaction_{label}");
                let whisker_raw_col = diag_column_name(&whisker_name, "raw_combined");
                let whisker_outside_col = diag_column_name(&whisker_name, "outside_object");
                let whiskers = WhiskerHeuristic {
                    perimeter: effective.clone(),
                    max_distance: self.qualia_whisker_distance,
                    max_angle: self.qualia_whisker_angle,
                    name: whisker_name.clone(),
                    result_column: Some(whisker_raw_col.clone()),
                };
                let whisker_decomposition = HeuristicDecompositionSpec {
                    heuristic_name: whisker_name.clone(),
                    result_column: whisker_name.clone(),
                    coordinate_labels: vec![head_label.clone()],
                    perimeter_label: perimeter_label.clone(),
                    components: vec![
                        decomposition_component(
                            "left_ear_near",
                            diag_column_name(&whisker_name, "left_ear_near"),
                            vec![coordinate_label_from_xy("left_ear_x")],
                            perimeter_label.clone(),
                        ),
                        decomposition_component(
                            "right_ear_near",
                            diag_column_name(&whisker_name, "right_ear_near"),
                            vec![coordinate_label_from_xy("right_ear_x")],
                            perimeter_label.clone(),
                        ),
                        decomposition_component(
                            "left_ray_hits_object",
                            diag_column_name(&whisker_name, "left_ray_hits_object"),
                            vec![head_label.clone()],
                            perimeter_label.clone(),
                        ),
                        decomposition_component(
                            "right_ray_hits_object",
                            diag_column_name(&whisker_name, "right_ray_hits_object"),
                            vec![head_label.clone()],
                            perimeter_label.clone(),
                        ),
                        decomposition_component(
                            "raw_combined",
                            whisker_raw_col.clone(),
                            vec![head_label.clone()],
                            perimeter_label.clone(),
                        ),
                        decomposition_component(
                            "outside_object",
                            whisker_outside_col.clone(),
                            vec![head_label.clone()],
                            perimeter_label.clone(),
                        ),
                        decomposition_component(
                            "left_combined",
                            diag_column_name(&whisker_name, "left_combined"),
                            vec![coordinate_label_from_xy("left_ear_x")],
                            perimeter_label.clone(),
                        ),
                        decomposition_component(
                            "right_combined",
                            diag_column_name(&whisker_name, "right_combined"),
                            vec![coordinate_label_from_xy("right_ear_x")],
                            perimeter_label.clone(),
                        ),
                    ],
                    show_final_result: true,
                };
                heuristics.push(Box::new(
                    CombinedHeuristic::new(
                        whisker_name.clone(),
                        whiskers,
                        vec![Box::new(OutsidePerimeterHeuristic {
                            perimeter: effective.clone(),
                            coords: self.params.centroid_coords.clone(),
                            name: whisker_outside_col,
                        })],
                    )
                    .with_decomposition(whisker_decomposition),
                ));

                if let Some(params) = self.whisker_fan {
                    let fan_name = format!("whisker_fan_{label}");
                    let fan_outside_col = diag_column_name(&fan_name, "outside_object");
                    heuristics.push(Box::new(
                        CombinedHeuristic::new(
                            fan_name.clone(),
                            WhiskerFanHeuristic {
                                spec: WhiskerFanSpec::new(params, &perimeter.shape),
                                ear_coords: self.params.head_coords.clone(),
                                nose_coords: self.params.nose_coords.clone(),
                                name: fan_name.clone(),
                                result_column: Some(diag_column_name(&fan_name, "hit")),
                                shared: self.shared_geometry.get(label.as_str()).cloned(),
                            },
                            vec![Box::new(OutsidePerimeterHeuristic {
                                perimeter: effective.clone(),
                                coords: self.params.centroid_coords.clone(),
                                name: fan_outside_col,
                            })],
                        )
                        .with_decomposition(HeuristicDecompositionSpec {
                            heuristic_name: fan_name.clone(),
                            result_column: fan_name.clone(),
                            coordinate_labels: vec![head_label.clone()],
                            perimeter_label: perimeter_label.clone(),
                            components: vec![decomposition_component(
                                "fan_hit",
                                diag_column_name(&fan_name, "hit"),
                                vec![head_label.clone()],
                                perimeter_label.clone(),
                            )],
                            show_final_result: true,
                        }),
                    ));
                }

                // Merge rules: expand per-object column names and emit MergedHeuristic.
                for merge_name in self.sorted_merge_rule_names() {
                    let rule = &self.merge_rules[merge_name];
                    let expanded = MergeRule {
                        require: rule
                            .require
                            .iter()
                            .map(|c| format!("{c}_{label}"))
                            .collect(),
                        any_of: rule.any_of.iter().map(|c| format!("{c}_{label}")).collect(),
                        exclude: rule
                            .exclude
                            .iter()
                            .map(|c| format!("{c}_{label}"))
                            .collect(),
                    };
                    let merged_name = format!("{merge_name}_{label}");
                    let merged_components =
                        merge_rule_components(&expanded, &head_label, perimeter_label.clone());
                    heuristics.push(Box::new(MergedHeuristic {
                        name: merged_name.clone(),
                        rule: expanded,
                        decomposition: Some(HeuristicDecompositionSpec {
                            heuristic_name: merged_name.clone(),
                            result_column: merged_name,
                            coordinate_labels: vec![head_label.clone()],
                            perimeter_label: perimeter_label.clone(),
                            components: merged_components,
                            show_final_result: true,
                        }),
                    }));
                }
            }
        }

        if let (Some(arena), Some(center)) = (self.arena_perimeter(), self.center_perimeter.clone())
        {
            heuristics.push(Box::new(PeripheryHeuristic {
                arena,
                center,
                coords: self.params.centroid_coords.clone(),
            }));
        }

        // Center zone confinement (for latency computation) if arena is known
        if let Some(ref center) = self.center_perimeter {
            heuristics.push(Box::new(ZoneConfinementHeuristic {
                zone_name: "in_center".into(),
                perimeter: center.clone(),
                coords: self.params.centroid_coords.clone(),
            }));
        }

        heuristics.push(Box::new(FreezingHeuristic::new(
            &self.params.head_coords.x,
            &self.params.head_coords.y,
            self.params.freezing_window_seconds,
            self.params.freezing_displacement_threshold,
            self.params.fps,
        )));

        heuristics
    }

    fn quadrants(&self) -> Vec<QuadrantSpec> {
        let Some(arena) = self.arena_perimeter() else {
            return Vec::new();
        };
        let bb = arena.shape.bounding_box();
        vec![QuadrantSpec {
            label: "object_recognition_grid".to_string(),
            x_min: bb.min_x,
            y_min: bb.min_y,
            x_max: bb.max_x,
            y_max: bb.max_y,
            rows: OBJECT_RECOGNITION_QUADRANT_ROWS as u32,
            cols: OBJECT_RECOGNITION_QUADRANT_COLS as u32,
        }]
    }

    fn analyze_detailed(
        &self,
        lf: LazyFrame,
        timing: &bekiper_core::video::FrameTiming,
    ) -> (crate::AnalysisResult, Option<TaskDetail>) {
        let heuristics = self.heuristics();
        let decompositions = self.inspection_decompositions(&heuristics);
        let (result, summaries) = self.full_analyze_with_summaries(lf, timing);
        self.detailed_from(decompositions, result, summaries)
    }
}

impl ObjectRecognition {
    fn detailed_from(
        &self,
        decompositions: Vec<HeuristicDecompositionSpec>,
        result: NortResult,
        summaries: Vec<HeuristicSummary>,
    ) -> (crate::AnalysisResult, Option<TaskDetail>) {
        let analysis = crate::AnalysisResult {
            task_type: self.task_type(),
            summaries,
            decompositions,
            evaluated_df: result.evaluated_df.clone(),
        };
        (analysis, Some(TaskDetail::ObjectRecognition(result)))
    }
}

/// Full NORT analysis result.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NortResult {
    /// Stage-aware protocol semantics for this trial.
    pub protocol: ObjectRecognitionProtocol,
    /// Per-object exploration time in seconds (ordered by `object_labels`).
    pub exploration_times: Vec<f64>,
    /// Per-object bout count (ordered by `object_labels`).
    pub bout_counts: Vec<u32>,
    /// Labels of the object perimeters, in order.
    pub object_labels: Vec<String>,
    /// Discrimination/recognition indices (present when >= 2 objects).
    pub indices: Option<NortIndices>,
    /// Per-object observing totals for actual objects (excludes arena / center zone).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observation_totals: Option<ObservationTotals>,
    /// Pairwise absolute observation deltas for actual objects.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pairwise_observations: Vec<PairwiseObservationMetric>,
    /// Diagnostic qualia rollups derived from behaviour parquet component columns.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub qualia_rollups: Vec<QualiaDiagnosticRollup>,
    /// Center/periphery/quadrant dwell and motion summaries.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub zone_metrics: Option<ObjectRecognitionZoneMetrics>,
    /// Per-object exploration filter-attribution diagnostics.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub exploration_attributions: Vec<ExplorationAttribution>,
    /// Effective scoring settings; absent on historical results.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scoring_provenance: Option<serde_json::Value>,
    /// Side bias: (T_B − T_A) / (T_B + T_A). Range [-1, 1].
    pub side_bias: Option<f64>,
    /// Object bias: (T_A − T_B) / (T_A + T_B). Range [-1, 1].
    pub object_bias: Option<f64>,
    /// Latency to center zone: time from trial start to first center entry (seconds).
    pub latency_to_center: Option<f64>,
    /// Total freezing duration in the analysis window (seconds).
    pub freezing_seconds: f64,
    /// Total path distance in the analysis window (meters).
    pub total_distance_meters: f64,
    /// Mean speed in the analysis window (m/s).
    pub total_mean_speed: f64,
    /// Median speed during object observation intervals (m/s).
    pub median_speed_observing: Option<f64>,
    /// Median speed during non-observation intervals (m/s).
    pub median_speed_not_observing: Option<f64>,
    /// Duration of the analysis window (seconds).
    pub total_duration_seconds: f64,
    /// Frame index of first object approach (None if never approached).
    pub first_approach_frame: Option<usize>,
    /// Time from trial start to first object approach (seconds).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_object_approach_time_s: Option<f64>,
    /// Analysis window duration used (seconds from first approach).
    pub analysis_window_seconds: f64,
    /// How the analysis window was selected for this trial.
    #[serde(default)]
    pub window_mode: NortWindowMode,
    /// Inclusive start frame of the selected analysis window.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window_start_frame: Option<usize>,
    /// Exclusive end frame of the selected analysis window.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window_end_frame_exclusive: Option<usize>,
    /// Whether the animal was excluded for insufficient total exploration.
    pub excluded: bool,
    /// Whether total exploration meets the configured reporting threshold.
    #[serde(default)]
    pub passes_15s_exploration_threshold: bool,
    /// Qualia decomposition: per-sensory-channel per-object statistics.
    /// Present when `enable_qualia` is true and at least one object exists.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub qualia: Option<QualiaDecomposition>,
    #[serde(skip)]
    pub evaluated_df: Option<DataFrame>,
}

#[cfg(test)]
mod finite_or_text_tests {
    use super::finite_or_text;
    use serde_json::json;

    #[test]
    fn non_finite_values_become_text() {
        assert_eq!(finite_or_text(1.5), json!(1.5));
        assert_eq!(finite_or_text(f64::NAN), json!("nan"));
        assert_eq!(finite_or_text(f64::INFINITY), json!("inf"));
        assert_eq!(finite_or_text(f64::NEG_INFINITY), json!("-inf"));
    }
}
