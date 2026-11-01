#[cfg(test)]
mod ymaze_dwell_filtered_sequence_tests {
    use super::ARM_LABELS;
    use bekiper_feature::zone::{dwell_filtered_mask, zone_summary};
    use polars::prelude::*;

    use crate::ymaze::{extract_zone_sequence, reduce_to_arm_sequence};

    fn dwell_filtered_sequence_from_masks(
        arm_a: &[bool],
        arm_b: &[bool],
        arm_c: &[bool],
        center: &[bool],
        min_dwell_frames: u32,
    ) -> Vec<String> {
        let zone_values_owned = [
            (
                "A".to_string(),
                dwell_filtered_mask(arm_a, min_dwell_frames),
            ),
            (
                "B".to_string(),
                dwell_filtered_mask(arm_b, min_dwell_frames),
            ),
            (
                "C".to_string(),
                dwell_filtered_mask(arm_c, min_dwell_frames),
            ),
            (
                "Center".to_string(),
                dwell_filtered_mask(center, min_dwell_frames),
            ),
        ];
        let zone_values: Vec<(&str, Vec<bool>)> = zone_values_owned
            .iter()
            .map(|(label, values)| (label.as_str(), values.clone()))
            .collect();

        extract_zone_sequence(&zone_values, arm_a.len(), 0)
    }

    fn raw_sequence_from_masks(
        arm_a: &[bool],
        arm_b: &[bool],
        arm_c: &[bool],
        center: &[bool],
    ) -> Vec<String> {
        let zone_values_owned = [
            ("A".to_string(), arm_a.to_vec()),
            ("B".to_string(), arm_b.to_vec()),
            ("C".to_string(), arm_c.to_vec()),
            ("Center".to_string(), center.to_vec()),
        ];
        let zone_values: Vec<(&str, Vec<bool>)> = zone_values_owned
            .iter()
            .map(|(label, values)| (label.as_str(), values.clone()))
            .collect();

        extract_zone_sequence(&zone_values, arm_a.len(), 0)
    }

    fn all_false(n: usize) -> Vec<bool> {
        vec![false; n]
    }

    #[test]
    fn single_frame_zone_transit_produces_zero_sequence_entries() {
        let zone_sequence = dwell_filtered_sequence_from_masks(
            &[false, true, false],
            &all_false(3),
            &all_false(3),
            &all_false(3),
            2,
        );

        assert!(zone_sequence.is_empty());
    }

    #[test]
    fn same_arm_reentry_separated_by_dwell_center_crossing_is_preserved() {
        let arm_a = vec![true, true, true, false, false, false, true, true, true];
        let center = vec![false, false, false, true, true, true, false, false, false];
        let zone_sequence =
            dwell_filtered_sequence_from_masks(&arm_a, &all_false(9), &all_false(9), &center, 3);
        let arm_sequence = reduce_to_arm_sequence(&zone_sequence, &ARM_LABELS);

        assert_eq!(zone_sequence, ["A", "Center", "A"]);
        assert_eq!(arm_sequence, ["A", "A"]);
    }

    #[test]
    fn short_arm_blip_between_dwell_arm_bouts_is_filtered_out() {
        let arm_a = vec![false, false, false, true, true, false, false, false];
        let arm_b = vec![true, true, true, false, false, true, true, true];
        let zone_sequence =
            dwell_filtered_sequence_from_masks(&arm_a, &arm_b, &all_false(8), &all_false(8), 3);
        let arm_sequence = reduce_to_arm_sequence(&zone_sequence, &ARM_LABELS);

        assert_eq!(zone_sequence, ["B", "B"]);
        assert_eq!(arm_sequence, ["B", "B"]);
    }

    #[test]
    fn valid_sequence_len_matches_dwell_filtered_zone_summary() -> PolarsResult<()> {
        let arm_a = vec![true, true, true, false, false, true, true, true];
        let arm_b = vec![false, false, false, true, true, false, false, false];
        let arm_c = all_false(8);
        let center = all_false(8);
        let zone_sequence = dwell_filtered_sequence_from_masks(&arm_a, &arm_b, &arm_c, &center, 3);
        let arm_sequence = reduce_to_arm_sequence(&zone_sequence, &ARM_LABELS);

        let df = df![
            "in_arm_A" => arm_a,
            "in_arm_B" => arm_b,
            "in_arm_C" => arm_c,
            "x" => vec![0.0; 8],
            "y" => vec![0.0; 8],
        ]?;
        let summary_entries: u32 = ["in_arm_A", "in_arm_B", "in_arm_C"]
            .iter()
            .map(|zone| zone_summary(&df, zone, "x", "y", 30.0, 3).entries)
            .sum();

        assert_eq!(arm_sequence, ["A", "A"]);
        assert_eq!(arm_sequence.len() as u32, summary_entries);
        Ok(())
    }

    #[test]
    fn raw_sequence_len_can_exceed_valid() {
        let arm_a = vec![true, true, true, false, true, false, false, false];
        let arm_b = all_false(8);
        let arm_c = all_false(8);
        let center = vec![false, false, false, true, false, true, true, true];

        let valid_zone_sequence =
            dwell_filtered_sequence_from_masks(&arm_a, &arm_b, &arm_c, &center, 3);
        let raw_zone_sequence = raw_sequence_from_masks(&arm_a, &arm_b, &arm_c, &center);
        let valid_arm_sequence = reduce_to_arm_sequence(&valid_zone_sequence, &ARM_LABELS);
        let raw_arm_sequence = reduce_to_arm_sequence(&raw_zone_sequence, &ARM_LABELS);

        assert_eq!(valid_arm_sequence, ["A"]);
        assert_eq!(raw_arm_sequence, ["A", "A"]);
        assert!(raw_arm_sequence.len() > valid_arm_sequence.len());
    }

    #[test]
    fn raw_sequence_len_matches_valid_without_short_crossings() {
        let arm_a = vec![true, true, true, false, false, false, true, true, true];
        let arm_b = all_false(9);
        let arm_c = all_false(9);
        let center = vec![false, false, false, true, true, true, false, false, false];

        let valid_zone_sequence =
            dwell_filtered_sequence_from_masks(&arm_a, &arm_b, &arm_c, &center, 3);
        let raw_zone_sequence = raw_sequence_from_masks(&arm_a, &arm_b, &arm_c, &center);
        let valid_arm_sequence = reduce_to_arm_sequence(&valid_zone_sequence, &ARM_LABELS);
        let raw_arm_sequence = reduce_to_arm_sequence(&raw_zone_sequence, &ARM_LABELS);

        assert_eq!(valid_arm_sequence, ["A", "A"]);
        assert_eq!(raw_arm_sequence, valid_arm_sequence);
    }

    #[test]
    fn max_gap_zero_reproduces_legacy_none_reset() {
        let arm_a = vec![true, false, true];
        let zone_sequence =
            raw_sequence_from_masks(&arm_a, &all_false(3), &all_false(3), &all_false(3));
        let arm_sequence = reduce_to_arm_sequence(&zone_sequence, &ARM_LABELS);

        assert_eq!(zone_sequence, ["A", "A"]);
        assert_eq!(arm_sequence, ["A", "A"]);
    }

    #[test]
    fn short_none_gap_is_bridged_within_budget() {
        let zone_values = vec![
            ("A", vec![true, false, true]),
            ("Center", vec![false, false, false]),
        ];

        let sequence = extract_zone_sequence(&zone_values, 3, 3);

        assert_eq!(sequence, ["A"]);
    }

    #[test]
    fn none_gap_longer_than_budget_resets_prev_zone() {
        let zone_values = vec![
            ("A", vec![true, false, false, false, false, true]),
            ("Center", vec![false, false, false, false, false, false]),
        ];

        let sequence = extract_zone_sequence(&zone_values, 6, 3);

        assert_eq!(sequence, ["A", "A"]);
    }

    #[test]
    fn bridging_does_not_merge_across_real_intervening_zone() {
        let zone_values = vec![
            ("A", vec![true, false, false, true]),
            ("Center", vec![false, false, true, false]),
        ];

        let sequence = extract_zone_sequence(&zone_values, 4, 3);
        let arm_sequence = reduce_to_arm_sequence(&sequence, &ARM_LABELS);

        assert_eq!(sequence, ["A", "Center", "A"]);
        assert_eq!(arm_sequence, ["A", "A"]);
    }
}
