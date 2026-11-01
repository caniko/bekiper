#[test]
fn half_second_bout_does_not_round_down_at_fifteen_fps() {
    let df = df! { "raw" => &[true; 7] }.unwrap();
    let (seconds, bouts, _) = object_exploration_with_bout_filters(
        &df,
        "raw",
        15.0,
        0.5,
        1.0,
        0.0,
        test_filter_spec(
            BoutFilterMode::PerFrame,
            BoutFilterMode::PerFrame,
            None,
            None,
            None,
        ),
    );
    assert_eq!((seconds, bouts), (0.0, 0));
}

#[test]
fn exploration_uses_timestamped_duration_and_window_end_boundary() {
    let full = BoutTiming::timestamps(&[0.0, 0.2, 0.5, 1.5, 1.7], 2.0).unwrap();
    let df = df! { "raw" => &[true, true, false, true, true] }.unwrap();
    let (seconds, bouts, _) = object_exploration_with_timing(
        &df,
        "raw",
        &full,
        0.5,
        1.0,
        0.0,
        BoutFilterSpec::unfiltered(),
    );
    assert_eq!((seconds, bouts), (2.0, 1));
    let window = df.slice(0, 2);
    let (seconds, bouts, _) = object_exploration_with_timing(
        &window,
        "raw",
        &full.sliced(0, 2),
        0.5,
        1.0,
        0.0,
        BoutFilterSpec::unfiltered(),
    );
    assert_eq!((seconds, bouts), (0.5, 1));
}

#[test]
fn native_bouts_cannot_merge_through_another_object_assignment() {
    let raw = [
        true, true, true, true, false, false, false, false, true, true, true, true,
    ];
    for other_object in [false, true] {
        let other = [
            false,
            false,
            false,
            false,
            other_object,
            other_object,
            other_object,
            other_object,
            false,
            false,
            false,
            false,
        ];
        let df = df! { "exploring_A_raw" => &raw, "exploring_B_raw" => &other }.unwrap();
        let (seconds, bouts, _) = object_exploration_with_bout_filters(
            &df,
            "exploring_A_raw",
            10.0,
            0.5,
            1.0,
            0.0,
            BoutFilterSpec::unfiltered(),
        );
        assert_eq!(
            (seconds, bouts),
            if other_object { (0.0, 0) } else { (1.2, 1) }
        );
    }
}

#[test]
fn merge_before_min_duration() {
    let mut raw = Vec::new();
    raw.extend(std::iter::repeat_n(true, 20));
    raw.extend(std::iter::repeat_n(false, 10));
    raw.extend(std::iter::repeat_n(true, 20));
    let speeds = vec![0.02; 50];
    let angles = vec![20.0; 50];
    assert_eq!(
        filtered_count(
            &raw,
            &speeds,
            &angles,
            BoutFilterMode::PerBoutMedian,
            BoutFilterMode::PerBoutMedian,
            30,
            15,
        ),
        50
    );
}

#[test]
fn bout_filter_mode_default_is_per_frame() {
    assert_eq!(BoutFilterMode::default(), BoutFilterMode::PerFrame);
}
