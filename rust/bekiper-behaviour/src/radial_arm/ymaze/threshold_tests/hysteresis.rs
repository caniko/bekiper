#[test]
fn brief_mid_bout_dropout_is_latched_when_hysteresis_is_on() {
    let mut ymaze = synthetic_ymaze();
    ymaze.apply_dwell_filter = false;
    ymaze.zone_max_gap_seconds = 0.0;
    ymaze.start_mode = super::YMazeStartMode::VideoStart;

    let result = ymaze.full_analyze(mid_bout_tail_drop_input());

    assert_eq!(in_arm_a_values(&result), vec![true; 11]);
    assert_eq!(result.analysis.arm_sequence, ["A"]);
}

#[test]
fn brief_mid_bout_dropout_releases_without_hysteresis() {
    let mut ymaze = synthetic_ymaze();
    ymaze.apply_dwell_filter = false;
    ymaze.arm_membership_hysteresis = false;
    ymaze.zone_max_gap_seconds = 0.0;
    ymaze.start_mode = super::YMazeStartMode::VideoStart;

    let result = ymaze.full_analyze(mid_bout_tail_drop_input());

    assert_eq!(
        in_arm_a_values(&result),
        vec![
            true, true, true, true, true, false, true, true, true, true, true
        ]
    );
    assert_eq!(result.analysis.arm_sequence, ["A", "A"]);
}

#[test]
fn genuine_exit_still_releases_with_hysteresis() {
    let mut ymaze = synthetic_ymaze();
    ymaze.apply_dwell_filter = false;
    ymaze.zone_max_gap_seconds = 0.0;
    ymaze.start_mode = super::YMazeStartMode::VideoStart;

    let result = ymaze.full_analyze(genuine_exit_input());

    assert_eq!(
        in_arm_a_values(&result),
        vec![
            true, true, true, true, true, false, false, false, false, false, false
        ]
    );
    assert_eq!(result.analysis.arm_sequence, ["A"]);
}

#[test]
fn polygon_exit_releases_even_when_threshold_is_still_satisfied() {
    let mut ymaze = synthetic_ymaze();
    ymaze.apply_dwell_filter = false;
    ymaze.zone_max_gap_seconds = 0.0;
    ymaze.start_mode = super::YMazeStartMode::VideoStart;

    let result = ymaze.full_analyze(polygon_exit_input());

    assert_eq!(
        in_arm_a_values(&result),
        vec![true, true, true, true, true, false, true, true]
    );
    assert_eq!(result.analysis.arm_sequence, ["A", "A"]);
}
