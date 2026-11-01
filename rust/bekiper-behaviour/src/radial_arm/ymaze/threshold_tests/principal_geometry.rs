#[test]
fn threshold_distances_use_arm_span_from_nearest_vertex() {
    let ymaze = synthetic_ymaze();

    let proximal = ymaze.arm_threshold_distances_sq(DEFAULT_PROXIMAL_THRESHOLD);
    let forty_five = ymaze.arm_threshold_distances_sq(0.45);

    assert_eq!(proximal.len(), 1);
    assert_eq!(forty_five.len(), 1);

    assert!((proximal[0] - 0.1225).abs() < f64::EPSILON);
    assert!((forty_five[0] - 0.3025).abs() < f64::EPSILON);
}

fn principal_input(head: &[f64], tail: &[f64]) -> LazyFrame {
    assert_eq!(head.len(), tail.len());
    df! {
        "center_ear_x" => head.to_vec(), "center_ear_y" => vec![0.0; head.len()],
        "tail_base_x" => tail.to_vec(), "tail_base_y" => vec![0.0; head.len()],
        // Deliberately irrelevant centroid/nose proxies.
        "mouse_center_x" => vec![0.9; head.len()], "mouse_center_y" => vec![0.0; head.len()],
        "nose_x" => vec![0.9; head.len()], "nose_y" => vec![0.0; head.len()],
    }
    .unwrap()
    .lazy()
}

#[test]
fn explicit_zero_dwell_retains_second_landmark_crossing_and_subsecond_entries() {
    for threshold in [0.25, 0.45, 0.65] {
        let mut task = synthetic_ymaze();
        task.proximal_threshold = threshold;
        task.temporal_policy = Some(super::TemporalPolicy::default());
        task.params.fps = 30.0;
        task.params.min_dwell_seconds = 100.0; // must be irrelevant
        task.apply_dwell_filter = true; // principal rule overrides legacy request
        task.zone_max_gap_seconds = 10.0;
        task.pose_gap_fill_seconds = 10.0;
        task.arm_membership_hysteresis = true;
        let low = 0.2;
        let high = 0.9;
        for (head, tail, expected_frames) in [
            (vec![low, high, low], vec![low, low, low], vec![]),
            (vec![low, low, low], vec![low, high, low], vec![]),
            (
                vec![low, high, high, low],
                vec![low, low, high, low],
                vec![2],
            ),
            (
                vec![low, low, high, low],
                vec![low, high, high, low],
                vec![2],
            ),
            (vec![low, high, low], vec![low, high, low], vec![1]),
            // Single-landmark retreat releases; next dual crossing is a new entry.
            (
                vec![low, high, high, high, low],
                vec![low, high, low, high, low],
                vec![1, 3],
            ),
        ] {
            let result = task.full_analyze(principal_input(&head, &tail));
            assert_eq!(result.analysis.arm_sequence.len(), expected_frames.len());
            assert_eq!(
                result.analysis.arm_sequence,
                result.analysis.raw_arm_sequence
            );
            let p = result.scoring_provenance.unwrap();
            assert_eq!(p["entry_frames"]["in_arm_A"], json!(expected_frames));
            assert_eq!(p["apply_dwell_filter"], false);
            assert_eq!(p["minimum_dwell_seconds"], 0.0);
            assert!(!result.arm_membership_hysteresis);
            assert_eq!(result.zone_max_gap_seconds, 0.0);
            assert_eq!(result.pose_gap_fill_seconds, 0.0);
        }
    }
}

#[test]
fn principal_geometry_is_radial_span_not_center_to_tip_or_axis_projection() {
    for threshold in [0.25, 0.45, 0.65] {
        let mut task = synthetic_ymaze();
        task.proximal_threshold = threshold;
        let radius = 0.1 + threshold * 1.0;
        let alternate = threshold * 1.1;
        let between = (radius + alternate) / 2.0;
        let values = [between, radius - 1e-9, radius, radius + 1e-9];
        let result = task.full_analyze(principal_input(&values, &values));
        assert_eq!(in_arm_a_values(&result), [false, false, true, true]);
        // Off-axis radial crossing lies before a hypothetical axial line.
        let off_axis = df! {
            "center_ear_x" => [radius - 0.00005], "center_ear_y" => [0.01],
            "tail_base_x" => [radius - 0.00005], "tail_base_y" => [0.01],
            "mouse_center_x" => [0.9], "mouse_center_y" => [0.0],
        }
        .unwrap()
        .lazy();
        assert_eq!(in_arm_a_values(&task.full_analyze(off_axis)), [true]);
    }
    let mut shallow = synthetic_ymaze();
    shallow.proximal_threshold = 0.25;
    let mut deep = synthetic_ymaze();
    deep.proximal_threshold = 0.65;
    assert_eq!(
        shallow
            .full_analyze(principal_input(&[0.5], &[0.5]))
            .analysis
            .arm_sequence,
        ["A"]
    );
    assert!(
        deep.full_analyze(principal_input(&[0.5], &[0.5]))
            .analysis
            .arm_sequence
            .is_empty()
    );
}

#[test]
fn principal_landmarks_must_cross_into_the_same_arm() {
    for threshold in [0.25, 0.45, 0.65] {
        let mut task = synthetic_three_arm_ymaze();
        task.proximal_threshold = threshold;
        let different_arms = df! {
            "center_ear_x" => [0.9], "center_ear_y" => [0.0],
            "tail_base_x" => [0.0], "tail_base_y" => [0.9],
            "mouse_center_x" => [0.9], "mouse_center_y" => [0.0],
        }
        .unwrap()
        .lazy();
        assert!(
            task.full_analyze(different_arms)
                .analysis
                .arm_sequence
                .is_empty()
        );
    }
}

#[test]
fn shared_python_rust_principal_entry_fixture() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../tests/golden/principal_entry_parity.json"
    ))
    .expect("parse shared principal-entry fixture");
    let center: Vec<(f64, f64)> =
        serde_json::from_value(fixture["center"].clone()).expect("center vertices");
    let arms = ["A", "B", "C"]
        .into_iter()
        .map(|arm| {
            let vertices: Vec<(f64, f64)> =
                serde_json::from_value(fixture["arms"][arm].clone()).expect("arm vertices");
            Polygon::new(vertices)
        })
        .collect();
    let maze = Perimeter::new("maze", RadialMaze::new(Polygon::new(center), arms));
    for case in fixture["cases"].as_array().expect("fixture cases") {
        let samples = case["samples"].as_array().expect("sample rows");
        let values = |column: usize| -> Vec<Option<f64>> {
            samples.iter().map(|row| row[column].as_f64()).collect()
        };
        let input = df! {
            "center_ear_x" => values(1), "center_ear_y" => values(2),
            "tail_base_x" => values(3), "tail_base_y" => values(4),
            "mouse_center_x" => vec![0.9; samples.len()],
            "mouse_center_y" => vec![0.0; samples.len()],
        }
        .expect("parity input");
        for threshold in case["thresholds"].as_array().expect("thresholds") {
            let pct = threshold.as_u64().expect("threshold percentage");
            let mut task = YMaze::new(maze.clone(), 30.0);
            // Frozen October 4 fixture records the explicit zero-dwell policy.
            task.temporal_policy = Some(super::TemporalPolicy::default());
            task.start_mode = super::YMazeStartMode::VideoStart;
            task.proximal_threshold = pct as f64 / 100.0;
            task.params.min_dwell_seconds = 100.0;
            task.zone_max_gap_seconds = 10.0;
            task.pose_gap_fill_seconds = 10.0;
            let result = task.full_analyze(input.clone().lazy());
            let expected = &case["expected"][pct.to_string()];
            assert_eq!(json!(result.analysis.arm_sequence), expected["sequence"]);
            assert_eq!(
                result.analysis.arm_sequence,
                result.analysis.raw_arm_sequence
            );
            let provenance = result.scoring_provenance.as_ref().expect("provenance");
            for arm in ["A", "B", "C"] {
                assert_eq!(
                    provenance["entry_frames"][format!("in_arm_{arm}")],
                    expected["entry_frames"][arm]
                );
            }
            if let Some(predicate) = expected.get("arm_A_predicate") {
                assert_eq!(json!(in_arm_a_values(&result)), *predicate);
            }
        }
    }
}

#[test]
fn total_distance_includes_movement_outside_every_zone_predicate() {
    // Head crosses the threshold while the tail stays behind: those frames
    // satisfy no arm predicate, but the path still counts (criteria #29).
    let mut task = synthetic_ymaze();
    task.params.fps = 30.0;
    let result = task.full_analyze(principal_input(&[0.2, 0.9, 0.2], &[0.2, 0.2, 0.2]));
    assert!((result.total_distance_meters - 1.4).abs() < 1e-12);
    let in_zones: f64 = result
        .zone_summaries
        .iter()
        .map(|zone| zone.distance_meters)
        .sum();
    assert!(result.total_distance_meters > in_zones);
}
