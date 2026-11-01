#[test]
fn recipe_strict_gap_and_speed_boundaries_preserve_historical_mode() {
    for strict in [false, true] {
        for (gap_frames, expected) in [(4, 1), (5, if strict { 0 } else { 1 })] {
            let mut raw = vec![true; 5];
            raw.extend(vec![false; gap_frames]);
            raw.extend(vec![true; 5]);
            let df = df! { "raw" => raw }.unwrap();
            let (_, count, _) = object_exploration_with_bout_filters(
                &df,
                "raw",
                10.0,
                1.0,
                0.5,
                0.0,
                BoutFilterSpec {
                    strict_boundaries: strict,
                    ..BoutFilterSpec::unfiltered()
                },
            );
            assert_eq!(count, expected);
        }
        for (speed, expected) in [(0.049, 1), (0.05, if strict { 0 } else { 1 }), (0.051, 0)] {
            let df = df! { "raw" => &[true; 10] }.unwrap();
            let speeds = floats(&[speed; 10]);
            let (_, count, _) = object_exploration_with_bout_filters(
                &df,
                "raw",
                10.0,
                1.0,
                0.5,
                0.0,
                BoutFilterSpec {
                    strict_boundaries: strict,
                    speed_mode: BoutFilterMode::PerBoutMedian,
                    nose_speed_mps: Some(&speeds),
                    ..BoutFilterSpec::unfiltered()
                },
            );
            assert_eq!(count, expected);
        }
    }
}

#[test]
fn imen_bouts_include_equal_duration_and_gap_without_speed_rejection() {
    for (gap_frames, expected_bouts) in [(10, 1), (11, 2)] {
        let mut raw = vec![true; 5];
        raw.extend(vec![false; gap_frames]);
        raw.extend(vec![true; 5]);
        let speeds = floats(&vec![0.5; raw.len()]);
        let angles = floats(&vec![10.0; raw.len()]);
        let raw = bools(&raw);
        let (mask, diagnostics) = filter_selected_bouts(
            &raw,
            select_exploration_bouts(
                &raw,
                &BoutTiming::cfr(10.0, raw.len()).unwrap(),
                IMEN_MIN_BOUT_SECONDS,
                IMEN_MAX_GAP_SECONDS,
                0.0,
            ),
            BoutFilterSpec {
                strict_boundaries: false,
                speed_mode: BoutFilterMode::Disabled,
                angle_mode: BoutFilterMode::PerBoutMedian,
                max_nose_speed: 0.05,
                max_angle_deg: 45.0,
                nose_speed_mps: Some(&speeds),
                angle_deg: Some(&angles),
                nose_contact: None,
            },
        );
        assert_eq!(count_bouts(&mask), expected_bouts);
        assert_eq!(diagnostics.removed_by_speed, 0);
        assert_eq!(
            diagnostics.per_bout_median_nose_speed,
            vec![0.5; expected_bouts as usize]
        );
    }
}

#[test]
fn disabled_speed_does_not_gate_candidate_frames() {
    use bekiper_perimeter::rectangle::Rectangle;
    let mut heuristic = ExplorationHeuristic::new(
        Perimeter::new("object", Rectangle::new(0.0, 0.0, 0.04, 0.04)),
        "object",
        (0.0, 0.0),
        10.0,
    );
    heuristic.speed_filter_mode = BoutFilterMode::Disabled;
    assert!(
        heuristic
            .decomposition()
            .unwrap()
            .components
            .iter()
            .all(|component| component.label != "slow_nose")
    );
    let df = df! {
        "mouse_center_x" => &[0.1, 0.1], "mouse_center_y" => &[0.0, 0.0],
        "nose_x" => &[0.022, -0.022], "nose_y" => &[0.0, 0.0],
        "snout_dx" => &[-1.0, 1.0], "snout_dy" => &[0.0, 0.0],
    }
    .unwrap();
    let evaluated = heuristic.evaluate(df.lazy()).collect().unwrap();
    assert_eq!(
        evaluated
            .column("exploring_object_raw")
            .unwrap()
            .bool()
            .unwrap()
            .sum(),
        Some(2)
    );
    assert_eq!(
        evaluated
            .column(&heuristic.slow_nose_col())
            .unwrap()
            .bool()
            .unwrap()
            .get(1),
        Some(false)
    );
}

#[test]
fn per_frame_speed_mode_identical_to_baseline() {
    let raw = bools(&[true, true, false, true, true, true, false, true]);
    let filtered = apply_bout_level_filters(
        &raw,
        2,
        1,
        0.0,
        test_filter_spec(
            BoutFilterMode::PerFrame,
            BoutFilterMode::PerFrame,
            None,
            None,
            None,
        ),
    );
    let baseline = apply_tolerance_series_with_density(&raw, 2, 1, 0.0);
    assert_eq!(
        filtered
            .into_iter()
            .map(|value| value.unwrap_or(false))
            .collect::<Vec<_>>(),
        baseline
            .into_iter()
            .map(|value| value.unwrap_or(false))
            .collect::<Vec<_>>()
    );
}

#[test]
fn per_bout_median_speed_passes_slow_bout_with_fast_spikes() {
    let mut raw_per_frame = vec![true; 30];
    raw_per_frame[10] = false;
    raw_per_frame[20] = false;
    let raw_per_bout = vec![true; 30];
    let mut speeds = vec![0.02; 30];
    speeds[10] = 0.15;
    speeds[20] = 0.15;
    let angles = vec![20.0; 30];

    assert_eq!(
        filtered_count(
            &raw_per_frame,
            &speeds,
            &angles,
            BoutFilterMode::PerFrame,
            BoutFilterMode::PerFrame,
            30,
            0,
        ),
        0
    );
    assert_eq!(
        filtered_count(
            &raw_per_bout,
            &speeds,
            &angles,
            BoutFilterMode::PerBoutMedian,
            BoutFilterMode::PerFrame,
            30,
            0,
        ),
        30
    );
}

#[test]
fn per_bout_median_speed_rejects_fast_overall_bout() {
    let raw = vec![true; 30];
    let speeds = vec![0.10; 30];
    let angles = vec![20.0; 30];

    assert_eq!(
        filtered_count(
            &[false; 30],
            &speeds,
            &angles,
            BoutFilterMode::PerFrame,
            BoutFilterMode::PerFrame,
            30,
            0,
        ),
        0
    );
    assert_eq!(
        filtered_count(
            &raw,
            &speeds,
            &angles,
            BoutFilterMode::PerBoutMedian,
            BoutFilterMode::PerFrame,
            30,
            0,
        ),
        0
    );
}

#[test]
fn per_bout_median_speed_even_length_bout_tie_breaker() {
    let raw = vec![true; 4];
    let speeds = vec![0.02, 0.03, 0.08, 0.12];
    let angles = vec![20.0; 4];
    assert_eq!(
        filtered_count(
            &raw,
            &speeds,
            &angles,
            BoutFilterMode::PerBoutMedian,
            BoutFilterMode::PerFrame,
            4,
            0,
        ),
        0
    );
}
