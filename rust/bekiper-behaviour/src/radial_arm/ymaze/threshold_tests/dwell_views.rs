fn brief_arm_bout_input() -> LazyFrame {
    // 6 frames @ 1 fps. The mouse enters the single arm for a single
    // frame, leaves to the center for three frames, then dwells in the
    // arm for two frames. With min_dwell_seconds=2 the leading single
    // frame is below the dwell threshold; with the filter off it
    // survives as a distinct arm entry.
    df! {
        "center_ear_x" => [0.9, 0.0, 0.0, 0.0, 0.9, 0.9],
        "center_ear_y" => [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "tail_base_x" => [0.9, 0.0, 0.0, 0.0, 0.9, 0.9],
        "tail_base_y" => [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "mouse_center_x" => [0.9, 0.0, 0.0, 0.0, 0.9, 0.9],
        "mouse_center_y" => [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }
    .unwrap()
    .lazy()
}

#[test]
fn apply_dwell_filter_true_drops_brief_arm_bout_from_headline_view() {
    let mut ymaze = synthetic_ymaze();
    ymaze.params.min_dwell_seconds = 2.0;
    ymaze.apply_dwell_filter = true;
    ymaze.start_mode = super::YMazeStartMode::VideoStart;

    let result = ymaze.full_analyze(brief_arm_bout_input());

    // Headline arm_sequence is dwell-filtered → only the 2-frame bout
    // survives the ≥2 s gate. raw_arm_sequence keeps both arm visits.
    assert_eq!(result.analysis.arm_sequence, ["A"]);
    assert_eq!(result.analysis.raw_arm_sequence, ["A", "A"]);
}

#[test]
fn apply_dwell_filter_false_keeps_brief_arm_bout_in_headline_view() {
    let mut ymaze = synthetic_ymaze();
    ymaze.params.min_dwell_seconds = 2.0;
    ymaze.apply_dwell_filter = false;
    ymaze.arm_membership_hysteresis = false;
    ymaze.start_mode = super::YMazeStartMode::VideoStart;

    let result = ymaze.full_analyze(brief_arm_bout_input());

    // With the dwell filter off the headline arm_sequence equals the raw
    // arm-threshold crossings: both arm visits are counted.
    assert_eq!(result.analysis.arm_sequence, ["A", "A"]);
    assert_eq!(
        result.analysis.arm_sequence, result.analysis.raw_arm_sequence,
        "raw and headline views must match when the dwell filter is off"
    );
}

#[test]
fn raw_sa_fields_are_always_populated_regardless_of_toggle() {
    // The raw SA view is computed from raw_arm_sequence and is therefore
    // independent of apply_dwell_filter. Both toggle values must produce
    // identical raw_* counts for the same trial.
    let baseline = {
        let mut ymaze = synthetic_ymaze();
        ymaze.params.min_dwell_seconds = 2.0;
        ymaze.apply_dwell_filter = true;
        ymaze.start_mode = super::YMazeStartMode::VideoStart;
        ymaze.full_analyze(brief_arm_bout_input())
    };
    let toggled = {
        let mut ymaze = synthetic_ymaze();
        ymaze.params.min_dwell_seconds = 2.0;
        ymaze.apply_dwell_filter = false;
        ymaze.start_mode = super::YMazeStartMode::VideoStart;
        ymaze.full_analyze(brief_arm_bout_input())
    };
    assert_eq!(
        baseline.analysis.raw_correct_alternations,
        toggled.analysis.raw_correct_alternations,
    );
    assert_eq!(
        baseline.analysis.raw_incorrect_alternations,
        toggled.analysis.raw_incorrect_alternations,
    );
    assert_eq!(
        baseline.analysis.raw_neutral_triads,
        toggled.analysis.raw_neutral_triads,
    );
    assert_eq!(
        baseline.analysis.raw_spontaneous_alternation_pct,
        toggled.analysis.raw_spontaneous_alternation_pct,
    );
}
