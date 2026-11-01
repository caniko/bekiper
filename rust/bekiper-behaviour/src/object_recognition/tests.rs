#[cfg(test)]
mod tests {
    use super::*;

    use bekiper_perimeter::circle::Circle;
    use bekiper_perimeter::polygon::Polygon;
    use bekiper_perimeter::rectangle::Rectangle;

    #[test]
    fn infinite_analysis_window_is_exported_as_text() {
        let mut task = ObjectRecognition::new(Vec::new(), 10.0);
        task.analysis_window_seconds = f64::INFINITY;
        let value = BehaviouralTask::effective_parameters(&task).unwrap();
        assert_eq!(value["exploration"]["analysis_window_seconds"], "inf");
    }

    #[test]
    fn default_protocol_counts_objects_without_arena_or_center_context() {
        for (objects, expected) in [
            (2, ObjectRecognitionProtocol::NortNovelty),
            (3, ObjectRecognitionProtocol::OulTest),
        ] {
            let mut perimeters: Vec<_> = (0..objects)
                .map(|index| {
                    Perimeter::new(
                        format!("object_{index}"),
                        Circle::new(index as f64, 0.0, 0.03).into(),
                    )
                })
                .collect();
            let before = ObjectRecognition::new(perimeters.clone(), 10.0);
            assert_eq!(before.protocol, expected);
            for label in ["arena", "center_zone"] {
                perimeters.push(Perimeter::new(
                    label,
                    Rectangle::new(0.0, 0.0, 0.4, 0.4).into(),
                ));
            }
            let with_context = ObjectRecognition::new(perimeters, 10.0);
            assert_eq!(with_context.protocol, expected);
            assert_eq!(with_context.actual_object_perimeters().len(), objects);
        }
    }

    #[test]
    fn unavailable_bout_medians_round_trip_native_json_and_toml_checkpoints() {
        use bekiper_core::inspection::{json_value_to_toml_string, toml_str_to_json_value};
        let attribution = serde_json::json!({
            "frames_in_window":1000, "frames_pose_ok":1000, "frames_outside_object":1000,
            "frames_nose_near":30, "frames_nose_contact":10, "frames_within_angle":20,
            "frames_contact_or_angle":30, "frames_slow_nose":1000, "frames_all_gates_raw":30,
            "frames_after_tolerance":30, "bouts_pre_filter":3, "bouts_post_filter":3,
            "bout_lengths_ms_pre":[500,1000,500], "bout_lengths_ms_post":[500,1000,500],
            "final_valid_bouts":3, "median_nose_speed_per_bout":[0.02, f64::NAN, 0.04],
            "median_head_angle_per_bout":[f64::NAN,20.0,f64::NAN]
        });
        let value = serde_json::json!({
            "protocol":"nort_novelty", "exploration_times":[2.0], "bout_counts":[3], "object_labels":["novel"],
            "exploration_attributions":[attribution], "freezing_seconds":0.0, "total_distance_meters":1.0,
            "total_mean_speed":0.01, "total_duration_seconds":100.0, "analysis_window_seconds":120.0,
            "excluded":false
        });
        // JSON's numeric missing slots and the retained TOML checkpoint's
        // empty-string slots must both preserve bout position and finite data.
        let checkpoint =
            toml_str_to_json_value(&json_value_to_toml_string(&value).unwrap()).unwrap();
        assert_eq!(
            checkpoint["exploration_attributions"][0]["median_nose_speed_per_bout"][1],
            ""
        );
        for input in [value.clone(), checkpoint] {
            let detail = crate::TaskDetail::from_json_value(
                bekiper_core::types::TaskType::ObjectRecognition,
                input,
            )
            .unwrap();
            let restored = detail.as_json_value();
            for field in [
                "median_nose_speed_per_bout",
                "median_head_angle_per_bout",
                "final_valid_bouts",
            ] {
                assert_eq!(
                    restored["exploration_attributions"][0][field],
                    value["exploration_attributions"][0][field]
                );
            }
            assert_eq!(restored["bout_counts"], value["bout_counts"]);
            assert_eq!(restored["exploration_times"], value["exploration_times"]);
        }
        for invalid in [
            serde_json::json!("unavailable"),
            serde_json::json!(true),
            serde_json::json!({}),
        ] {
            let mut malformed = value.clone();
            malformed["exploration_attributions"][0]["median_nose_speed_per_bout"][1] = invalid;
            assert!(
                crate::TaskDetail::from_json_value(
                    bekiper_core::types::TaskType::ObjectRecognition,
                    malformed,
                )
                .is_err()
            );
        }
    }

    #[test]
    fn tilted_nort_arena_clips_quadrant_dwell_distance_and_motion_in_both_phases() {
        for protocol in [
            ObjectRecognitionProtocol::NortTraining,
            ObjectRecognitionProtocol::NortNovelty,
        ] {
            let mut task = ObjectRecognition::new(
                vec![Perimeter::new(
                    "arena",
                    ObjectShape::Polygon(Polygon::new(vec![
                        (0.0, -0.2),
                        (0.2, 0.0),
                        (0.0, 0.2),
                        (-0.2, 0.0),
                    ])),
                )],
                10.0,
            );
            task.protocol = protocol;
            task.center_perimeter = Some(Perimeter::new(
                "center_zone",
                Rectangle::new(0.0, 0.0, 0.1, 0.1).into(),
            ));
            task.params.min_dwell_seconds = 0.0;
            task.enable_qualia = false;
            task.window_mode = NortWindowMode::FullAnalysis;
            // The last two samples are inside the bounding box but outside the diamond.
            let df = df! {
                "mouse_center_x" => &[0.15, 0.04, 0.18, 0.16],
                "mouse_center_y" => &[0.02, 0.02, 0.18, 0.18],
                "nose_x" => &[0.15, 0.04, 0.18, 0.16], "nose_y" => &[0.02, 0.02, 0.18, 0.18],
                "center_ear_x" => &[0.15, 0.04, 0.18, 0.16], "center_ear_y" => &[0.02, 0.02, 0.18, 0.18],
                "snout_dx" => &[1.0; 4], "snout_dy" => &[0.0; 4],
            }.unwrap();
            let result = task.full_analyze(df.lazy());
            let zones = result.zone_metrics.as_ref().unwrap();
            assert_eq!(zones.center_summary.as_ref().unwrap().duration_seconds, 0.1);
            assert_eq!(
                zones.periphery_summary.as_ref().unwrap().duration_seconds,
                0.1
            );
            let quadrant_duration: f64 = zones
                .quadrant_summaries
                .iter()
                .map(|zone| zone.duration_seconds)
                .sum();
            assert!((quadrant_duration - 0.2).abs() < 1e-12);
            let quadrant_distance: f64 = zones
                .quadrant_summaries
                .iter()
                .map(|zone| zone.distance_meters)
                .sum();
            assert!((quadrant_distance - 0.11).abs() < 1e-12);
            let motion_distance: f64 = zones
                .quadrant_motion_summaries
                .iter()
                .map(|zone| zone.total_displacement_meters)
                .sum();
            assert!((motion_distance - 0.11).abs() < 1e-12);
            let df = result.evaluated_df.as_ref().unwrap();
            let mask = df.column("q_1_1").unwrap().bool().unwrap();
            assert_eq!(
                mask.into_iter().collect::<Vec<_>>(),
                [Some(true), Some(true), Some(false), Some(false)]
            );
        }
    }

    #[test]
    fn imen_training_and_novelty_retain_arena_metrics_and_disabled_speed_policy() {
        for protocol in [
            ObjectRecognitionProtocol::NortTraining,
            ObjectRecognitionProtocol::NortNovelty,
        ] {
            let label = if protocol == ObjectRecognitionProtocol::NortTraining {
                "variable"
            } else {
                "novel"
            };
            let mut task = ObjectRecognition::new(
                vec![
                    Perimeter::new(
                        "arena",
                        ObjectShape::Rectangle(Rectangle::new(0.0, 0.0, 0.4, 0.4)),
                    ),
                    Perimeter::new(
                        "familiar",
                        ObjectShape::Rectangle(Rectangle::new(-0.15, 0.0, 0.04, 0.04)),
                    ),
                    Perimeter::new(
                        label,
                        ObjectShape::Rectangle(Rectangle::new(0.0, 0.0, 0.04, 0.04)),
                    ),
                ],
                10.0,
            );
            task.protocol = protocol;
            task.center_perimeter = Some(Perimeter::new(
                "center_zone",
                Rectangle::new(0.0, 0.0, 0.24, 0.24).into(),
            ));
            task.params.min_dwell_seconds = 0.0;
            task.enable_qualia = false;
            task.perimeter_erosion_meters = 0.0;
            task.speed_filter_mode = BoutFilterMode::Disabled;
            task.angle_filter_mode = BoutFilterMode::PerBoutMedian;
            task.min_bout_seconds =
                bekiper_feature::heuristic::combined::exploration::IMEN_MIN_BOUT_SECONDS;
            task.max_gap_seconds =
                bekiper_feature::heuristic::combined::exploration::IMEN_MAX_GAP_SECONDS;
            task.window_mode = NortWindowMode::FullAnalysis;
            let nose: Vec<_> = (0..10)
                .map(|i| if i % 2 == 0 { 0.022 } else { -0.022 })
                .collect();
            let dx: Vec<_> = nose
                .iter()
                .map(|x| if *x > 0.0 { -1.0 } else { 1.0 })
                .collect();
            let body: Vec<_> = (0..10).map(|i| if i < 5 { 0.15 } else { 0.1 }).collect();
            let df = df! {
                "mouse_center_x" => &body, "mouse_center_y" => &[0.0; 10],
                "nose_x" => &nose, "nose_y" => &[0.0; 10],
                "center_ear_x" => &[0.1; 10], "center_ear_y" => &[0.0; 10],
                "snout_dx" => &dx, "snout_dy" => &[0.0; 10],
            }
            .unwrap();
            let (result, summaries) =
                task.full_analyze_with_summaries(df.lazy(), &FrameTiming::cfr(10.0));
            assert_eq!(result.object_labels, ["familiar", label]);
            assert_eq!(result.exploration_times, [0.0, 1.0]);
            assert_eq!(result.bout_counts, [0, 1]);
            let zones = result.zone_metrics.as_ref().unwrap();
            assert_eq!(zones.center_summary.as_ref().unwrap().duration_seconds, 0.5);
            assert_eq!(
                zones.periphery_summary.as_ref().unwrap().duration_seconds,
                0.5
            );
            assert_eq!(zones.quadrant_summaries.len(), 4);
            let diagnostics = &result.exploration_attributions[1];
            assert_eq!(diagnostics.bouts_removed_by_speed_filter, 0);
            assert!(diagnostics.median_nose_speed_per_bout[0] > 0.05);
            assert_eq!(
                summaries
                    .iter()
                    .find(|s| s.name == format!("exploring_{label}"))
                    .unwrap()
                    .seconds,
                1.0
            );
            let serialized = serde_json::to_value(&result).unwrap();
            assert_eq!(
                serialized["scoring_provenance"]["speed_filter_mode"],
                "disabled"
            );
            assert_eq!(
                serialized["scoring_provenance"]["minimum_dwell_seconds"],
                0.5
            );
            assert_eq!(serialized["scoring_provenance"]["merge_gap_seconds"], 1.0);
            assert_eq!(
                serialized["scoring_provenance"]["gap_median_policy"],
                "include_gaps"
            );
            assert_eq!(
                serialized["scoring_provenance"]["bout_contact_policy"],
                bekiper_feature::heuristic::combined::exploration::BOUT_CONTACT_POLICY
            );
        }
    }

    #[test]
    fn overlap_resolution_prefers_nearest_boundary() {
        let mut df = df! {
            "nose_x" => &[1.0],
            "nose_y" => &[0.0],
            "exploring_large_raw" => &[true],
            "exploring_small_raw" => &[true],
        }
        .unwrap();
        let perimeters = vec![
            Perimeter::new("large", ObjectShape::Circle(Circle::new(0.0, 0.0, 5.0))),
            Perimeter::new("small", ObjectShape::Circle(Circle::new(3.0, 0.0, 2.1))),
        ];
        let raw_cols = vec![
            "exploring_large_raw".to_string(),
            "exploring_small_raw".to_string(),
        ];

        resolve_overlapping_explorations(&mut df, &perimeters, &raw_cols);

        assert!(
            !df.column("exploring_large_raw")
                .unwrap()
                .bool()
                .unwrap()
                .get(0)
                .unwrap()
        );
        assert!(
            df.column("exploring_small_raw")
                .unwrap()
                .bool()
                .unwrap()
                .get(0)
                .unwrap()
        );
    }

    #[test]
    fn first_exploration_frame_is_the_earliest_across_objects() {
        let df = df! {
            "late" => &[false, false, true, true],
            "early" => &[false, true, false, true],
            "speed" => &[1.0, 2.0, 3.0, 4.0],
        }
        .unwrap();
        let cols = ["late", "missing", "speed", "early"].map(String::from);
        assert_eq!(find_first_exploration_frame(&df, &cols), Some(1));
        assert_eq!(find_first_exploration_frame(&df, &cols[1..3]), None);
    }

    #[test]
    fn non_finite_or_non_positive_windows_use_the_full_trial() {
        for seconds in [f64::INFINITY, f64::NAN, 0.0, -5.0] {
            let window = select_analysis_window(
                NortWindowMode::AnchoredFirstApproach,
                100,
                seconds,
                10.0,
                Some(3),
            );
            assert_eq!(window.mode, NortWindowMode::ConfiguredFullTrial);
            assert_eq!((window.start_frame, window.end_frame_exclusive), (0, 100));
        }
    }

    #[test]
    fn overlap_resolution_needs_nose_coordinates() {
        let perimeters = vec![
            Perimeter::new("a", ObjectShape::Circle(Circle::new(0.0, 0.0, 1.0))),
            Perimeter::new("b", ObjectShape::Circle(Circle::new(0.5, 0.0, 1.0))),
        ];
        let raw_cols = vec!["a_raw".to_string(), "b_raw".to_string()];
        let both = |df: &DataFrame| -> Vec<bool> {
            raw_cols
                .iter()
                .map(|c| df.column(c).unwrap().bool().unwrap().get(0).unwrap())
                .collect()
        };
        // Without nose_x or nose_y the overlap is left unresolved.
        for frame in [
            df! { "a_raw" => &[true], "b_raw" => &[true] }.unwrap(),
            df! { "nose_x" => &[0.0], "a_raw" => &[true], "b_raw" => &[true] }.unwrap(),
        ] {
            let mut frame = frame;
            resolve_overlapping_explorations(&mut frame, &perimeters, &raw_cols);
            assert_eq!(both(&frame), [true, true]);
        }
        // A null nose on the overlapping frame also keeps both.
        let mut null_nose = df! {
            "nose_x" => &[None::<f64>], "nose_y" => &[Some(0.0)],
            "a_raw" => &[true], "b_raw" => &[true],
        }
        .unwrap();
        resolve_overlapping_explorations(&mut null_nose, &perimeters, &raw_cols);
        assert_eq!(both(&null_nose), [true, true]);
    }

    #[test]
    fn zone_and_freezing_summaries_default_to_zero_without_their_columns() {
        let task = ObjectRecognition::new(vec![], 10.0);
        let df = df! { "nose_x" => &[0.0, 1.0, 2.0] }.unwrap();
        let timing = FrameTiming::cfr(10.0);
        let zone = task.zone_summary_entry(&df, &timing, "in_center");
        assert_eq!(
            (zone.name.as_str(), zone.true_frames, zone.total_frames),
            ("in_center", 0, 3)
        );
        let freezing = task.freezing_summary_entry(&df, &timing);
        assert_eq!((freezing.true_frames, freezing.total_frames), (0, 3));
    }

    #[test]
    fn speeds_skip_frame_pairs_with_missing_head_pose() {
        let mut task = ObjectRecognition::new(
            vec![Perimeter::new(
                "object",
                ObjectShape::Circle(Circle::new(1.0, 0.0, 0.05)),
            )],
            10.0,
        );
        task.enable_qualia = false;
        task.window_mode = NortWindowMode::FullAnalysis;
        let head = [Some(0.0), Some(0.01), None, Some(0.03), Some(0.04)];
        let df = df! {
            "mouse_center_x" => &head, "mouse_center_y" => &[0.0; 5],
            "nose_x" => &head, "nose_y" => &[0.0; 5],
            "center_ear_x" => &head, "center_ear_y" => &[0.0; 5],
            "snout_dx" => &[1.0; 5], "snout_dy" => &[0.0; 5],
        }
        .unwrap();
        let result = task.full_analyze(df.lazy());
        // Only the two complete pairs (0→1, 3→4) contribute 0.1 m/s each.
        let median = result.median_speed_not_observing.unwrap();
        assert!((median - 0.1).abs() < 1e-9, "{median}");
        assert_eq!(result.median_speed_observing, None);
    }
}
