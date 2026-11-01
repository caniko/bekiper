#[test]
fn zone_gap_bridge_snapshot_matches() {
    let legacy = {
        let mut ymaze = synthetic_ymaze();
        ymaze.params.fps = 30.0;
        ymaze.start_mode = super::YMazeStartMode::VideoStart;
        ymaze.apply_dwell_filter = false;
        ymaze.zone_max_gap_seconds = 0.0;
        ymaze.full_analyze(zone_gap_bridge_input())
    };
    let bridged = {
        let mut ymaze = synthetic_ymaze();
        ymaze.params.fps = 30.0;
        ymaze.start_mode = super::YMazeStartMode::VideoStart;
        ymaze.apply_dwell_filter = false;
        ymaze.zone_max_gap_seconds = DEFAULT_ZONE_MAX_GAP_SECONDS;
        ymaze.full_analyze(zone_gap_bridge_input())
    };

    assert_eq!(legacy.analysis.arm_sequence, ["A", "A"]);
    assert_eq!(bridged.analysis.arm_sequence, ["A"]);
    assert_eq!(legacy.analysis.raw_arm_sequence, ["A", "A"]);
    assert_eq!(bridged.analysis.raw_arm_sequence, ["A", "A"]);

    let actual = serde_json::to_string_pretty(&json!({
        "legacy": {
            "zone_max_gap_seconds": legacy.zone_max_gap_seconds,
            "arm_sequence": legacy.analysis.arm_sequence,
            "raw_arm_sequence": legacy.analysis.raw_arm_sequence,
        },
        "bridged": {
            "zone_max_gap_seconds": bridged.zone_max_gap_seconds,
            "arm_sequence": bridged.analysis.arm_sequence,
            "raw_arm_sequence": bridged.analysis.raw_arm_sequence,
        }
    }))
    .expect("serialize zone gap bridge snapshot");
    let golden = zone_gap_bridge_golden_path();

    if std::env::var_os("UPDATE_GOLDEN").is_some() {
        std::fs::create_dir_all(
            golden
                .parent()
                .expect("zone gap bridge golden should have a parent directory"),
        )
        .expect("create zone gap bridge golden directory");
        std::fs::write(&golden, &actual).expect("write zone gap bridge golden");
        eprintln!("wrote golden fixture to {}", golden.display());
        return;
    }

    let expected = std::fs::read_to_string(&golden).unwrap_or_else(|e| {
        panic!(
            "golden fixture missing at {} ({e}); run with UPDATE_GOLDEN=1 to create it",
            golden.display()
        )
    });
    assert_eq!(
        actual, expected,
        "zone gap bridge golden drifted; rerun with UPDATE_GOLDEN=1 if intentional"
    );
}

#[test]
fn arm_membership_hysteresis_snapshot_matches() {
    let legacy = {
        let mut ymaze = synthetic_three_arm_ymaze();
        ymaze.apply_dwell_filter = false;
        ymaze.arm_membership_hysteresis = false;
        ymaze.zone_max_gap_seconds = 0.0;
        ymaze.start_mode = super::YMazeStartMode::VideoStart;
        ymaze.full_analyze(hysteresis_golden_input())
    };
    let latched = {
        let mut ymaze = synthetic_three_arm_ymaze();
        ymaze.apply_dwell_filter = false;
        ymaze.arm_membership_hysteresis = true;
        ymaze.zone_max_gap_seconds = 0.0;
        ymaze.start_mode = super::YMazeStartMode::VideoStart;
        ymaze.full_analyze(hysteresis_golden_input())
    };

    assert_eq!(legacy.analysis.arm_sequence, ["A", "B", "B", "C"]);
    assert_eq!(legacy.analysis.correct_alternations, 0);
    assert_eq!(legacy.analysis.incorrect_alternations, 1);
    assert_eq!(legacy.analysis.neutral_triads, 1);
    assert_eq!(latched.analysis.arm_sequence, ["A", "B", "C"]);
    assert_eq!(latched.analysis.correct_alternations, 1);
    assert_eq!(latched.analysis.incorrect_alternations, 0);
    assert_eq!(latched.analysis.neutral_triads, 0);

    let actual = serde_json::to_string_pretty(&json!({
        "legacy": {
            "arm_membership_hysteresis": legacy.arm_membership_hysteresis,
            "arm_sequence": legacy.analysis.arm_sequence,
            "correct_alternations": legacy.analysis.correct_alternations,
            "incorrect_alternations": legacy.analysis.incorrect_alternations,
            "neutral_triads": legacy.analysis.neutral_triads,
        },
        "latched": {
            "arm_membership_hysteresis": latched.arm_membership_hysteresis,
            "arm_sequence": latched.analysis.arm_sequence,
            "correct_alternations": latched.analysis.correct_alternations,
            "incorrect_alternations": latched.analysis.incorrect_alternations,
            "neutral_triads": latched.analysis.neutral_triads,
        }
    }))
    .expect("serialize arm membership hysteresis snapshot");
    let golden = arm_membership_hysteresis_golden_path();

    if std::env::var_os("UPDATE_GOLDEN").is_some() {
        std::fs::create_dir_all(
            golden
                .parent()
                .expect("arm membership hysteresis golden should have a parent directory"),
        )
        .expect("create arm membership hysteresis golden directory");
        std::fs::write(&golden, &actual).expect("write arm membership hysteresis golden");
        eprintln!("wrote golden fixture to {}", golden.display());
        return;
    }

    let expected = std::fs::read_to_string(&golden).unwrap_or_else(|e| {
        panic!(
            "golden fixture missing at {} ({e}); run with UPDATE_GOLDEN=1 to create it",
            golden.display()
        )
    });
    assert_eq!(
        actual, expected,
        "arm membership hysteresis golden drifted; rerun with UPDATE_GOLDEN=1 if intentional"
    );
}

#[test]
fn pose_gap_fill_snapshot_matches() {
    let disabled = {
        let mut ymaze = synthetic_ymaze();
        ymaze.params.fps = 30.0;
        ymaze.apply_dwell_filter = false;
        ymaze.arm_membership_hysteresis = false;
        ymaze.zone_max_gap_seconds = 0.0;
        ymaze.pose_gap_fill_seconds = 0.0;
        ymaze.start_mode = super::YMazeStartMode::VideoStart;
        ymaze.full_analyze(pose_gap_fill_input())
    };
    let filled = {
        let mut ymaze = synthetic_ymaze();
        ymaze.params.fps = 30.0;
        ymaze.apply_dwell_filter = false;
        ymaze.arm_membership_hysteresis = false;
        ymaze.zone_max_gap_seconds = 0.0;
        ymaze.pose_gap_fill_seconds = DEFAULT_POSE_GAP_FILL_SECONDS;
        ymaze.start_mode = super::YMazeStartMode::VideoStart;
        ymaze.full_analyze(pose_gap_fill_input())
    };

    assert_eq!(disabled.analysis.arm_sequence, ["A", "A"]);
    assert_eq!(filled.analysis.arm_sequence, ["A"]);
    assert_eq!(disabled.analysis.raw_arm_sequence, ["A", "A"]);
    assert_eq!(filled.analysis.raw_arm_sequence, ["A", "A"]);

    let actual = serde_json::to_string_pretty(&json!({
        "disabled": {
            "pose_gap_fill_seconds": disabled.pose_gap_fill_seconds,
            "arm_sequence": disabled.analysis.arm_sequence,
            "raw_arm_sequence": disabled.analysis.raw_arm_sequence,
        },
        "filled": {
            "pose_gap_fill_seconds": filled.pose_gap_fill_seconds,
            "arm_sequence": filled.analysis.arm_sequence,
            "raw_arm_sequence": filled.analysis.raw_arm_sequence,
        }
    }))
    .expect("serialize pose gap fill snapshot");
    let golden = pose_gap_fill_golden_path();

    if std::env::var_os("UPDATE_GOLDEN").is_some() {
        std::fs::create_dir_all(
            golden
                .parent()
                .expect("pose gap fill golden should have a parent directory"),
        )
        .expect("create pose gap fill golden directory");
        std::fs::write(&golden, &actual).expect("write pose gap fill golden");
        eprintln!("wrote golden fixture to {}", golden.display());
        return;
    }

    let expected = std::fs::read_to_string(&golden).unwrap_or_else(|e| {
        panic!(
            "golden fixture missing at {} ({e}); run with UPDATE_GOLDEN=1 to create it",
            golden.display()
        )
    });
    assert_eq!(
        actual, expected,
        "pose gap fill golden drifted; rerun with UPDATE_GOLDEN=1 if intentional"
    );
}

mod truly_raw_arm_sequence_tests {
    use super::{
        DEFAULT_POSE_GAP_FILL_SECONDS, all_mitigations_input, mid_bout_tail_drop_input,
        pose_gap_fill_input, synthetic_ymaze, truly_raw_arm_sequence_golden_path,
        zone_gap_bridge_input,
    };
    use serde_json::json;

    #[test]
    fn raw_sequence_bypasses_zone_gap_bridge() {
        let mut ymaze = synthetic_ymaze();
        ymaze.params.fps = 30.0;
        ymaze.apply_dwell_filter = false;
        ymaze.zone_max_gap_seconds = 0.5;
        ymaze.start_mode = super::super::YMazeStartMode::VideoStart;

        let result = ymaze.full_analyze(zone_gap_bridge_input());

        assert_eq!(result.analysis.raw_arm_sequence, ["A", "A"]);
        assert_eq!(result.analysis.arm_sequence, ["A"]);
    }

    #[test]
    fn raw_sequence_bypasses_hysteresis() {
        let mut ymaze = synthetic_ymaze();
        ymaze.apply_dwell_filter = false;
        ymaze.arm_membership_hysteresis = true;
        ymaze.zone_max_gap_seconds = 0.0;
        ymaze.start_mode = super::super::YMazeStartMode::VideoStart;

        let result = ymaze.full_analyze(mid_bout_tail_drop_input());

        assert_eq!(result.analysis.raw_arm_sequence, ["A", "A"]);
        assert_eq!(result.analysis.arm_sequence, ["A"]);
    }

    #[test]
    fn raw_sequence_bypasses_pose_gap_fill() {
        let mut ymaze = synthetic_ymaze();
        ymaze.params.fps = 30.0;
        ymaze.apply_dwell_filter = false;
        ymaze.arm_membership_hysteresis = false;
        ymaze.zone_max_gap_seconds = 0.0;
        ymaze.pose_gap_fill_seconds = DEFAULT_POSE_GAP_FILL_SECONDS;
        ymaze.start_mode = super::super::YMazeStartMode::VideoStart;

        let result = ymaze.full_analyze(pose_gap_fill_input());

        assert_eq!(result.analysis.raw_arm_sequence, ["A", "A"]);
        assert_eq!(result.analysis.arm_sequence, ["A"]);
    }

    #[test]
    fn truly_raw_arm_sequence_snapshot_matches() {
        let mut ymaze = synthetic_ymaze();
        ymaze.params.fps = 30.0;
        ymaze.apply_dwell_filter = false;
        ymaze.arm_membership_hysteresis = true;
        ymaze.zone_max_gap_seconds = 0.1;
        ymaze.pose_gap_fill_seconds = DEFAULT_POSE_GAP_FILL_SECONDS;
        ymaze.start_mode = super::super::YMazeStartMode::VideoStart;

        let result = ymaze.full_analyze(all_mitigations_input());

        assert_eq!(result.analysis.arm_sequence, ["A"]);
        assert_eq!(result.analysis.raw_arm_sequence, ["A", "A", "A", "A"]);

        let actual = serde_json::to_string_pretty(&json!({
            "arm_membership_hysteresis": result.arm_membership_hysteresis,
            "pose_gap_fill_seconds": result.pose_gap_fill_seconds,
            "zone_max_gap_seconds": result.zone_max_gap_seconds,
            "arm_sequence": result.analysis.arm_sequence,
            "raw_arm_sequence": result.analysis.raw_arm_sequence,
        }))
        .expect("serialize truly raw arm-sequence snapshot");
        let golden = truly_raw_arm_sequence_golden_path();

        if std::env::var_os("UPDATE_GOLDEN").is_some() {
            std::fs::create_dir_all(
                golden
                    .parent()
                    .expect("truly raw arm-sequence golden should have a parent directory"),
            )
            .expect("create truly raw arm-sequence golden directory");
            std::fs::write(&golden, &actual).expect("write truly raw arm-sequence golden");
            eprintln!("wrote golden fixture to {}", golden.display());
            return;
        }

        let expected = std::fs::read_to_string(&golden).unwrap_or_else(|e| {
            panic!(
                "golden fixture missing at {} ({e}); run with UPDATE_GOLDEN=1 to create it",
                golden.display()
            )
        });
        assert_eq!(
            actual, expected,
            "truly raw arm-sequence golden drifted; rerun with UPDATE_GOLDEN=1 if intentional"
        );
    }
}
