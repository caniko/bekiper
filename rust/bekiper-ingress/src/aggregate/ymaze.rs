use indexmap::IndexMap;
use serde_json::Value;

use super::cell::{CellValue, pluck, pluck_trial_metadata};
use super::experiment::{
    Experiment, ExperimentId, SheetKey, legacy_routing_key_for_trial, sheet_key_from_routing_key,
};
use super::trial::TrialResult;

/// Column schema for [`bekiper_core::types::TaskType::YMaze`].
pub struct YMazeExperiment;

const YMAZE_STATIC_COLUMNS: &[&str] = &[
    "animal",
    "timepoint",
    "cohort",
    "gene",
    "sex",
    "age",
    "fps",
    "proximal_threshold",
    "start_mode",
    "zone_max_gap_seconds",
    "arm_membership_hysteresis",
    "pose_gap_fill_seconds",
    "scoring_provenance",
    "total_duration_s",
    "analysis_duration_s",
    "duration_within_expected",
    "total_distance_m",
    "total_mean_speed",
    "raw_arm_A_entries",
    "arm_A_entries",
    "arm_A_duration_s",
    "arm_A_distance_m",
    "arm_A_mean_speed",
    "arm_A_latency_s",
    "arm_A_motion_distance_m",
    "arm_A_motion_median_speed",
    "arm_A_motion_median_acceleration",
    "arm_A_motion_freezing_s",
    "raw_arm_B_entries",
    "arm_B_entries",
    "arm_B_duration_s",
    "arm_B_distance_m",
    "arm_B_mean_speed",
    "arm_B_latency_s",
    "arm_B_motion_distance_m",
    "arm_B_motion_median_speed",
    "arm_B_motion_median_acceleration",
    "arm_B_motion_freezing_s",
    "raw_arm_C_entries",
    "arm_C_entries",
    "arm_C_duration_s",
    "arm_C_distance_m",
    "arm_C_mean_speed",
    "arm_C_latency_s",
    "arm_C_motion_distance_m",
    "arm_C_motion_median_speed",
    "arm_C_motion_median_acceleration",
    "arm_C_motion_freezing_s",
    "center_entries",
    "center_duration_s",
    "center_distance_m",
    "center_mean_speed",
    "center_latency_s",
    "center_motion_distance_m",
    "center_motion_median_speed",
    "center_motion_median_acceleration",
    "center_motion_freezing_s",
    "zone_entry_sequence",
    "raw_arm_entry_sequence",
    "valid_arm_entry_sequence",
    "first_arm_entered",
    "last_arm_entered",
    // Valid arm-entry counts: derived from `valid_arm_entry_sequence`, the cleaned
    // alternation sequence used for SA scoring. By construction
    // `valid_arm_A + valid_arm_B + valid_arm_C == valid_total_arm_entries
    // == len(valid_arm_entry_sequence)`.
    "valid_arm_A_entries",
    "valid_arm_B_entries",
    "valid_arm_C_entries",
    "raw_total_arm_entries",
    "raw_to_scored_entry_difference",
    "dwell_rejected_entries",
    // Historical alias: raw sequence count minus scored sequence count.
    "removed_short_entries",
    "valid_total_arm_entries",
    "total_triads",
    "correct_alternations",
    "incorrect_alternations",
    "usable_triads",
    "neutral_triads",
    "neutral_triads_pct",
    // `spontaneous_alternation_pct`           : legacy criteria denominator
    //                                           (correct / (correct + incorrect)
    //                                           x 100, neutral triads excluded).
    //                                           Stays exposed for backwards
    //                                           compatibility with prior
    //                                           reports. See
    //                                           `docs/src/pipeline/ymaze-thresholds.md`.
    // `spontaneous_alternation_pct_standard`  : the experimentalist-requested
    //                                           SA% = correct / total_triads
    //                                           x 100, where
    //                                           total_triads = max(0,
    //                                           valid_total_arm_entries - 2).
    "spontaneous_alternation_pct",
    "spontaneous_alternation_pct_standard",
    // Parallel SA view scored from `raw_arm_entry_sequence`, which bypasses
    // pose gap-fill, arm-membership hysteresis, and zone-gap bridging.
    // Lets analysts compare the mitigated headline against the original
    // detected arm-threshold crossings.
    "raw_correct_alternations",
    "raw_incorrect_alternations",
    "raw_neutral_triads",
    "raw_spontaneous_alternation_pct",
    "raw_spontaneous_alternation_pct_standard",
    "arm_preference",
    "perm_ABC",
    "perm_ACB",
    "perm_BAC",
    "perm_BCA",
    "perm_CAB",
    "perm_CBA",
    "permutation_sum",
    "incorrect_AAB",
    "incorrect_AAC",
    "incorrect_ABA",
    "incorrect_ACA",
    "incorrect_BAB",
    "incorrect_BBA",
    "incorrect_BBC",
    "incorrect_BCB",
    "incorrect_CAC",
    "incorrect_CBC",
    "incorrect_CCA",
    "incorrect_CCB",
    "incorrect_sum",
    "neutral_AAA",
    "neutral_ABB",
    "neutral_ACC",
    "neutral_BAA",
    "neutral_BBB",
    "neutral_BCC",
    "neutral_CAA",
    "neutral_CBB",
    "neutral_CCC",
    "neutral_sum",
    "seconds_in_arms",
    "seconds_in_arms_and_center",
    "freezing_s",
    "low_valid_entries_flag",
    "low_usable_triads_flag",
    "experiment",
    "excluded",
];

const YMAZE_ARM_LABELS: [&str; 3] = ["A", "B", "C"];

/// A complete arm-bout ledger can establish zero rejections; absent or
/// unrecognized evidence cannot. Center-zone bouts do not count as arm entries.
fn dwell_rejected_entries(detail: &Value) -> CellValue {
    let count = YMAZE_ARM_LABELS.iter().try_fold(0_i64, |total, arm| {
        let bouts =
            detail["scoring_provenance"]["dwell_bouts"][format!("in_arm_{arm}")].as_array()?;
        bouts
            .iter()
            .try_fold(total, |count, bout| match bout["accepted"].as_bool()? {
                true => Some(count),
                false if bout["rejection_reason"] == "minimum_dwell" => count.checked_add(1),
                false => None,
            })
    });
    count.map(CellValue::Int).unwrap_or(CellValue::Null)
}

const YMAZE_ANIMAL_OVERVIEW_COLUMNS: &[&str] = &[
    "spontaneous_alternation_pct_standard",
    "valid_total_arm_entries",
    "total_triads",
    "arm_preference",
    "total_distance_m",
    "total_duration_s",
    "proximal_threshold",
];

include!("ymaze/experiment.rs");

include!("ymaze/routing.rs");

include!("ymaze/zone_columns.rs");

include!("ymaze/arm_sequences.rs");

#[cfg(test)]
mod ymaze_timepoint_tests {
    use super::{CellValue, ymaze_timepoint};

    #[test]
    fn ymaze_timepoint_detects_pre() {
        assert_eq!(
            ymaze_timepoint("y_maze_videos_y_maze_pre_pre_video_3__proximal25"),
            CellValue::String("pre".into())
        );
    }

    #[test]
    fn ymaze_timepoint_detects_post() {
        assert_eq!(
            ymaze_timepoint("y_maze_videos_y_maze_post_post_video_5__proximal65"),
            CellValue::String("post".into())
        );
    }

    #[test]
    fn ymaze_timepoint_defaults_to_unknown_for_unlabeled() {
        assert_eq!(
            ymaze_timepoint("y_maze_videos_y_maze_unlabeled_video_7__proximal45"),
            CellValue::String("unknown".into())
        );
    }

    #[test]
    fn ymaze_timepoint_prefers_pre_when_both_markers_are_present() {
        assert_eq!(
            ymaze_timepoint("y_maze_videos_y_maze_pre_post_combo_video"),
            CellValue::String("pre".into())
        );
    }

    #[test]
    fn ymaze_timepoint_defaults_to_unknown_for_empty_strings() {
        assert_eq!(ymaze_timepoint(""), CellValue::String("unknown".into()));
    }
}
