fn coord_cols(labels: &[&str]) -> HashMap<Label, CoordinateColumnSpec> {
    labels
        .iter()
        .map(|l| {
            (
                (*l).to_string(),
                CoordinateColumnSpec {
                    x: format!("{l}_x"),
                    y: format!("{l}_y"),
                },
            )
        })
        .collect()
}

fn h(name: &str) -> HeuristicMeta {
    HeuristicMeta {
        name: name.into(),
        result_column: name.into(),
        true_frames: 10,
        total_frames: 100,
        seconds: 1.0,
        truth_spans: vec![],
    }
}

fn ymaze_perimeter() -> Vec<PerimeterSpec> {
    vec![PerimeterSpec::RadialMaze {
        label: "ymaze".into(),
        center_vertices: vec![(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)],
        arms: vec![
            vec![(1.0, 0.0), (2.0, 0.0), (2.0, 0.4), (1.0, 0.4)],
            vec![(0.0, 1.0), (-1.0, 2.0), (-0.7, 2.3), (0.3, 1.3)],
            vec![(1.0, 1.0), (2.0, 2.0), (1.7, 2.3), (0.7, 1.3)],
        ],
    }]
}

fn nort_detail() -> TaskDetail {
    TaskDetail::ObjectRecognition(NortResult {
        protocol: bekiper_core::types::ObjectRecognitionProtocol::NortNovelty,
        exploration_times: vec![12.5, 7.5],
        bout_counts: vec![3, 2],
        object_labels: vec!["novel".into(), "familiar".into()],
        indices: Some(bekiper_behaviour::nort::NortIndices::compute(
            12.5, 7.5, 3, 2,
        )),
        observation_totals: None,
        pairwise_observations: vec![],
        qualia_rollups: vec![],
        zone_metrics: None,
        exploration_attributions: vec![],
        scoring_provenance: None,
        side_bias: None,
        object_bias: None,
        latency_to_center: None,
        freezing_seconds: 0.0,
        total_distance_meters: 0.0,
        total_mean_speed: 0.0,
        median_speed_observing: None,
        median_speed_not_observing: None,
        total_duration_seconds: 30.0,
        first_approach_frame: Some(10),
        first_object_approach_time_s: Some(10.0 / 30.0),
        analysis_window_seconds: 120.0,
        window_mode: bekiper_behaviour::object_recognition::NortWindowMode::FullTrialShortTrial,
        window_start_frame: Some(0),
        window_end_frame_exclusive: Some(900),
        excluded: false,
        passes_15s_exploration_threshold: true,
        qualia: None,
        evaluated_df: None,
    })
}

fn ymaze_detail() -> TaskDetail {
    TaskDetail::YMaze(YMazeResult {
        scoring_provenance: None,
        analysis: bekiper_behaviour::ymaze::YMazeAnalysis {
            zone_sequence: vec![
                "A".into(),
                "Center".into(),
                "B".into(),
                "Center".into(),
                "C".into(),
            ],
            arm_sequence: vec!["A".into(), "B".into(), "C".into(), "A".into()],
            raw_arm_sequence: vec!["A".into(), "B".into(), "C".into(), "A".into()],
            first_arm_entered: Some("A".into()),
            last_arm_entered: Some("A".into()),
            arm_entries: [2, 1, 1],
            center_entries: 3,
            correct_alternations: 2,
            incorrect_alternations: 0,
            neutral_triads: 0,
            spontaneous_alternation_pct: Some(66.7),
            arm_preference: 0.1,
            permutation_distribution: bekiper_behaviour::ymaze::TriadPermutationDistribution {
                abc: 2,
                ..Default::default()
            },
            sum_of_permutation_distribution: 2,
            correct_triad_distribution: Default::default(),
            incorrect_triad_distribution: Default::default(),
            neutral_triad_distribution: Default::default(),
            raw_correct_alternations: 2,
            raw_incorrect_alternations: 0,
            raw_neutral_triads: 0,
            raw_spontaneous_alternation_pct: Some(66.7),
        },
        zone_summaries: vec![],
        area_motion_summaries: vec![],
        seconds_in_arms: 20.0,
        seconds_in_arms_and_center: 30.0,
        freezing_seconds: 0.0,
        total_distance_meters: 0.0,
        total_mean_speed: 0.0,
        total_duration_seconds: 30.0,
        analysis_duration_seconds: 30.0,
        proximal_threshold: 0.25,
        start_mode: bekiper_behaviour::radial_arm::YMazeStartMode::FirstDetection,
        zone_max_gap_seconds: 0.1,
        pose_gap_fill_seconds: 0.2,
        arm_membership_hysteresis: true,
        evaluated_df: None,
    })
}

fn oft_detail(center_time_pct: f64, periphery_time_pct: f64) -> TaskDetail {
    let zone = |name: &str, seconds: f64| bekiper_feature::zone::ZoneSummary {
        zone_name: name.into(),
        entries: 1,
        duration_seconds: seconds,
        distance_meters: 0.0,
        mean_speed: 0.0,
        latency_seconds: Some(0.0),
    };

    TaskDetail::OpenField(OftResult {
        window: Default::default(),
        center_summary: zone("center", center_time_pct),
        periphery_summary: zone("periphery", periphery_time_pct),
        quadrant_summaries: vec![],
        quadrant_motion_summaries: vec![],
        center_motion_summary: bekiper_behaviour::motion_detail::ZoneMotionSummary::default(),
        periphery_motion_summary: bekiper_behaviour::motion_detail::ZoneMotionSummary::default(),
        center_motion_islands: vec![],
        periphery_motion_islands: vec![],
        center_time_pct,
        periphery_time_pct,
        freezing_seconds: 0.0,
        immobility_seconds: 0.0,
        immobility_pct: 0.0,
        rearing_seconds: 0.0,
        grooming_seconds: 0.0,
        total_distance_meters: 0.0,
        total_mean_speed: 0.0,
        total_duration_seconds: center_time_pct + periphery_time_pct,
        evaluated_df: None,
    })
}

fn cheeseboard_detail() -> TaskDetail {
    TaskDetail::Cheeseboard(bekiper_behaviour::reward_tracing::CheeseboardResult {
        enclosure_label: "board".into(),
        reward_labels: vec![],
        enclosure_summary: bekiper_feature::zone::ZoneSummary {
            zone_name: "board".into(),
            entries: 0,
            duration_seconds: 0.0,
            distance_meters: 0.0,
            mean_speed: 0.0,
            latency_seconds: None,
        },
        reward_summaries: vec![],
        start_area_exit_frame: None,
        start_area_exit_seconds: None,
        reward_arrival_frame: None,
        reward_arrival_seconds: None,
        seconds_to_find_reward: None,
        seconds_in_reward_area: 0.0,
        reward_entries: 0,
        search_path_distance_meters: None,
        start_to_reward_displacement_meters: None,
        search_direct_distance_meters: None,
        search_mean_speed_mps: None,
        search_tortuosity: None,
        total_distance_meters: 0.0,
        total_mean_speed_mps: 0.0,
        total_duration_seconds: 0.0,
        evaluated_df: None,
    })
}

// ── Task-detailed emission tests ─────────────────────────────────

#[test]
fn nort_discrimination_emitted_from_detailed() {
    let detail = nort_detail();
    let primary = "snout".to_string();
    let features = collect_task_detailed_inspections(
        TaskType::ObjectRecognition,
        Some(&detail),
        Some(&primary),
    );
    assert_eq!(features.len(), 1);
    match &features[0] {
        FeatureInspection::NortDiscrimination {
            protocol,
            object_labels,
            exploration_times,
            discrimination_index,
            recognition_index,
            coordinate_labels,
            title_override,
            ..
        } => {
            assert_eq!(protocol.as_deref(), Some("nort_novelty"));
            assert_eq!(object_labels, &["novel", "familiar"]);
            assert_eq!(exploration_times, &[12.5, 7.5]);
            assert!((discrimination_index.unwrap() - 0.25).abs() < f64::EPSILON);
            assert!((recognition_index.unwrap() - 0.625).abs() < f64::EPSILON);
            assert_eq!(coordinate_labels, &["snout"]);
            assert_eq!(
                title_override.as_deref(),
                Some("Object Discrimination: NORT Novelty")
            );
        }
        other => panic!("expected NortDiscrimination, got {other:?}"),
    }
}

#[test]
fn nort_discrimination_skipped_without_object_labels() {
    let mut detail = nort_detail();
    if let TaskDetail::ObjectRecognition(ref mut detail) = detail {
        detail.object_labels.clear();
    }
    let features =
        collect_task_detailed_inspections(TaskType::ObjectRecognition, Some(&detail), None);
    assert!(features.is_empty());
}

#[test]
fn arm_sequence_emitted_from_detailed() {
    let detail = ymaze_detail();
    let primary = "nose".to_string();
    let features =
        collect_task_detailed_inspections(TaskType::YMaze, Some(&detail), Some(&primary));
    assert_eq!(features.len(), 1);
    match &features[0] {
        FeatureInspection::ArmSequence {
            arm_sequence,
            alternation_pct,
            coordinate_labels,
            ..
        } => {
            assert_eq!(arm_sequence, &["A", "B", "C", "A"]);
            assert!((alternation_pct.unwrap() - 66.7).abs() < 0.01);
            assert_eq!(coordinate_labels, &["nose"]);
        }
        other => panic!("expected ArmSequence, got {other:?}"),
    }
}

#[test]
fn arm_sequence_skipped_with_empty_sequence() {
    let mut detail = ymaze_detail();
    if let TaskDetail::YMaze(ref mut detail) = detail {
        detail.analysis.arm_sequence.clear();
    }
    let features = collect_task_detailed_inspections(TaskType::YMaze, Some(&detail), None);
    assert!(features.is_empty());
}

#[test]
fn ymaze_arm_sections_emitted_with_defaults() {
    let perimeters = ymaze_perimeter();
    let features = collect_ymaze_arm_section_inspections(
        TaskType::YMaze,
        &perimeters,
        Some(&"snout".to_string()),
        None,
        None,
    );

    assert_eq!(features.len(), 1);
    match &features[0] {
        FeatureInspection::YMazeArmSections {
            perimeter_label,
            proximal_threshold,
            coordinate_labels,
            ..
        } => {
            assert_eq!(perimeter_label, "ymaze");
            assert!((*proximal_threshold - DEFAULT_PROXIMAL_THRESHOLD).abs() < f64::EPSILON);
            assert_eq!(coordinate_labels, &["snout"]);
        }
        other => panic!("expected YMazeArmSections, got {other:?}"),
    }
}

#[test]
fn ymaze_arm_sections_emitted_with_overrides() {
    let perimeters = ymaze_perimeter();
    let features =
        collect_ymaze_arm_section_inspections(TaskType::YMaze, &perimeters, None, Some(0.30), None);

    match &features[0] {
        FeatureInspection::YMazeArmSections {
            proximal_threshold,
            coordinate_labels,
            ..
        } => {
            assert!((*proximal_threshold - 0.30).abs() < f64::EPSILON);
            assert_eq!(coordinate_labels, &["nose"]);
        }
        other => panic!("expected YMazeArmSections, got {other:?}"),
    }
}

#[test]
fn center_periphery_emitted_from_detailed() {
    let detail = oft_detail(35.2, 64.8);
    let primary = "snout".to_string();
    let features =
        collect_task_detailed_inspections(TaskType::OpenField, Some(&detail), Some(&primary));
    assert_eq!(features.len(), 1);
    match &features[0] {
        FeatureInspection::CenterPeriphery {
            center_perimeter_label,
            arena_perimeter_label,
            center_time_pct,
            periphery_time_pct,
            coordinate_labels,
            ..
        } => {
            assert_eq!(center_perimeter_label, "center");
            assert_eq!(arena_perimeter_label, "arena");
            assert!((center_time_pct - 35.2).abs() < 0.01);
            assert!((periphery_time_pct - 64.8).abs() < 0.01);
            assert_eq!(coordinate_labels, &["snout"]);
        }
        other => panic!("expected CenterPeriphery, got {other:?}"),
    }
}

#[test]
fn oft_center_mask_name_binds_to_the_recorded_center_geometry() {
    let TaskDetail::OpenField(mut detail) = oft_detail(17.2, 78.9) else {
        unreachable!()
    };
    detail.center_summary.zone_name = "in_center".into();
    let features = collect_task_detailed_inspections(
        TaskType::OpenField,
        Some(&TaskDetail::OpenField(detail)),
        None,
    );
    let FeatureInspection::CenterPeriphery {
        center_perimeter_label,
        ..
    } = &features[0]
    else {
        panic!("expected center/periphery feature")
    };
    assert_eq!(center_perimeter_label, "center_zone");
}

#[test]
fn center_periphery_emits_zero_center_time() {
    let detail = oft_detail(0.0, 64.8);
    let features = collect_task_detailed_inspections(TaskType::OpenField, Some(&detail), None);
    assert_eq!(features.len(), 1);
}

#[test]
fn nort_arena_occupancy_is_emitted_for_training_and_novelty() {
    for protocol in [
        bekiper_core::types::ObjectRecognitionProtocol::NortTraining,
        bekiper_core::types::ObjectRecognitionProtocol::NortNovelty,
    ] {
        let TaskDetail::ObjectRecognition(mut detail) = nort_detail() else {
            unreachable!()
        };
        detail.protocol = protocol;
        detail.total_duration_seconds = 20.0;
        detail.zone_metrics = Some(
            bekiper_behaviour::object_recognition::ObjectRecognitionZoneMetrics {
                center_summary: Some(bekiper_feature::zone::ZoneSummary {
                    zone_name: "in_center".into(),
                    duration_seconds: 5.0,
                    ..Default::default()
                }),
                periphery_summary: Some(bekiper_feature::zone::ZoneSummary {
                    zone_name: "in_periphery".into(),
                    duration_seconds: 10.0,
                    ..Default::default()
                }),
                ..Default::default()
            },
        );
        let features = collect_task_detailed_inspections(
            TaskType::ObjectRecognition,
            Some(&TaskDetail::ObjectRecognition(detail)),
            Some(&"body_center".into()),
        );
        let feature = features
            .iter()
            .find(|feature| feature.kind() == "center_periphery")
            .unwrap();
        match feature {
            FeatureInspection::CenterPeriphery {
                center_time_pct,
                periphery_time_pct,
                center_perimeter_label,
                coordinate_labels,
                ..
            } => {
                assert_eq!((*center_time_pct, *periphery_time_pct), (25.0, 50.0));
                assert_eq!(center_perimeter_label, "center_zone");
                assert_eq!(coordinate_labels, &["body_center"]);
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn task_detailed_returns_empty_for_none_detail() {
    let features = collect_task_detailed_inspections(TaskType::ObjectRecognition, None, None);
    assert!(features.is_empty());
}

#[test]
fn task_detailed_returns_empty_for_cheeseboard() {
    let detail = cheeseboard_detail();
    let features = collect_task_detailed_inspections(TaskType::Cheeseboard, Some(&detail), None);
    assert!(features.is_empty());
}

// ── Generic feature emission tests ────────────────────────────────

#[test]
fn empty_inputs_yield_no_features() {
    let coords = coord_cols(&[]);
    let features =
        collect_feature_inspections(&[], &[], &[], &[], &coords, None, None, &[], &[], &[]);
    assert!(features.is_empty());
}

#[test]
fn perimeter_emits_perimeter_feature() {
    let perimeters = vec![PerimeterSpec::Circle {
        label: "arena".into(),
        center_x: 0.0,
        center_y: 0.0,
        radius: 1.0,
    }];
    let coords = coord_cols(&["snout"]);
    let features = collect_feature_inspections(
        &perimeters,
        &[],
        &[],
        &[],
        &coords,
        None,
        None,
        &[],
        &[],
        &[],
    );
    // Perimeter + trajectory (snout primary), no heuristics, no angle pair
    assert_eq!(features.len(), 2);
    assert!(matches!(features[0], FeatureInspection::Perimeter { .. }));
    assert!(matches!(features[1], FeatureInspection::Trajectory { .. }));
}

#[test]
fn ymaze_arm_sections_copy_crossline_provenance_only_for_crossline_geometry() {
    let perimeters = ymaze_perimeter();
    let with = |geometry: &str| {
        let TaskDetail::YMaze(mut result) = ymaze_detail() else {
            unreachable!()
        };
        result.scoring_provenance = Some(serde_json::json!({
            "geometry_version": geometry,
            "crossline_endpoints": [
                {"p1": [0.35, -0.15], "p2": [0.325, 0.15]},
                {"p1": [1.0, 2.0], "p2": [3.0, 4.0]},
            ],
        }));
        let detail = TaskDetail::YMaze(result);
        match collect_ymaze_arm_section_inspections(
            TaskType::YMaze,
            &perimeters,
            None,
            Some(0.25),
            Some(&detail),
        )
        .remove(0)
        {
            FeatureInspection::YMazeArmSections {
                threshold_lines, ..
            } => threshold_lines,
            other => panic!("expected YMazeArmSections, got {other:?}"),
        }
    };
    let lines = with("imen_arm_start_crossline_v1");
    assert_eq!(lines.len(), 2);
    assert_eq!(
        (lines[0].arm.as_str(), lines[0].p1, lines[0].p2),
        ("A", (0.35, -0.15), (0.325, 0.15))
    );
    assert_eq!((lines[1].arm.as_str(), lines[1].p2), ("B", (3.0, 4.0)));
    assert!(with("imen_radial_vertex_span_v1").is_empty());
}
