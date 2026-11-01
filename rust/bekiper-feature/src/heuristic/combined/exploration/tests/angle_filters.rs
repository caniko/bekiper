#[test]
fn per_frame_angle_mode_identical_to_baseline() {
    let raw = bools(&[true, false, true, true, false, true]);
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
fn per_bout_median_angle_passes_with_brief_angle_failures() {
    let mut raw_per_frame = vec![true; 30];
    raw_per_frame[8] = false;
    raw_per_frame[16] = false;
    raw_per_frame[24] = false;
    let raw_per_bout = vec![true; 30];
    let speeds = vec![0.02; 30];
    let mut angles = vec![20.0; 30];
    angles[8] = 60.0;
    angles[16] = 60.0;
    angles[24] = 60.0;

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
            BoutFilterMode::PerFrame,
            BoutFilterMode::PerBoutMedian,
            30,
            0,
        ),
        30
    );
}

#[test]
fn per_bout_median_angle_drops_bout_with_sustained_bad_angle() {
    let raw = vec![true; 30];
    let speeds = vec![0.02; 30];
    let mut angles = vec![60.0; 30];
    for angle in angles.iter_mut().take(10) {
        *angle = 20.0;
    }
    assert_eq!(
        filtered_count(
            &raw,
            &speeds,
            &angles,
            BoutFilterMode::PerFrame,
            BoutFilterMode::PerBoutMedian,
            30,
            0,
        ),
        0
    );
}

#[test]
fn per_bout_median_angle_keeps_bout_with_brief_angle_failures() {
    let raw = vec![true; 30];
    let speeds = vec![0.02; 30];
    let mut angles = vec![20.0; 30];
    angles[0] = 60.0;
    angles[10] = 60.0;
    angles[20] = 60.0;
    assert_eq!(
        filtered_count(
            &raw,
            &speeds,
            &angles,
            BoutFilterMode::PerFrame,
            BoutFilterMode::PerBoutMedian,
            30,
            0,
        ),
        30
    );
}
