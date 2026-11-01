#[test]
fn per_bout_medians_emit_nan_when_diagnostic_columns_missing() {
    // Regression guard: a bout-level pipeline running without
    // diagnostic columns (PerFrame mode, no speed/angle series) must
    // not panic and must align the per-bout vectors with the bout
    // count by emitting NaN sentinels.
    let raw = bools(&[true, true, true, false, true, true, true]);

    let (_, diagnostics) = apply_bout_level_filters_with_diagnostics(
        &raw,
        2,
        0,
        0.0,
        test_filter_spec(
            BoutFilterMode::PerFrame,
            BoutFilterMode::PerFrame,
            None,
            None,
            None,
        ),
    );

    assert_eq!(diagnostics.final_valid_bouts, 2);
    assert_eq!(diagnostics.per_bout_median_nose_speed.len(), 2);
    assert_eq!(diagnostics.per_bout_median_head_angle_deg.len(), 2);
    for value in &diagnostics.per_bout_median_nose_speed {
        assert!(value.is_nan(), "expected NaN, got {value}");
    }
    for value in &diagnostics.per_bout_median_head_angle_deg {
        assert!(value.is_nan(), "expected NaN, got {value}");
    }
}

#[test]
fn diagnostics_count_each_bout_filter_stage() {
    let raw = bools(&[
        true, true, false, true, true, false, false, true, true, false, false, true, true, true,
        false, false, true, true, true,
    ]);
    let speeds = floats(&[
        0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.20, 0.20, 0.20, 0.02,
        0.02, 0.02, 0.02, 0.02,
    ]);
    let angles = floats(&[
        20.0, 20.0, 20.0, 60.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0,
        20.0, 60.0, 60.0, 60.0,
    ]);
    let (_, diagnostics) = apply_bout_level_filters_with_diagnostics(
        &raw,
        3,
        1,
        0.0,
        test_filter_spec(
            BoutFilterMode::PerBoutMedian,
            BoutFilterMode::PerBoutMedian,
            Some(&speeds),
            Some(&angles),
            None,
        ),
    );

    assert_eq!(diagnostics.candidate_bouts_pre_merge, 5);
    assert_eq!(diagnostics.bouts_post_merge, 4);
    assert_eq!(diagnostics.removed_by_duration, 1);
    assert_eq!(diagnostics.removed_by_speed, 1);
    assert_eq!(diagnostics.removed_by_angle, 1);
    assert_eq!(diagnostics.final_valid_bouts, 1);
    assert_eq!(diagnostics.per_bout_median_nose_speed, vec![0.02]);
    assert_eq!(diagnostics.per_bout_median_head_angle_deg, vec![20.0]);
}

#[test]
fn bout_medians_accept_only_finite_nan_and_missing_forms() {
    use serde::de::value::{Error, SeqDeserializer};
    let numbers = deserialize_bout_medians(SeqDeserializer::<_, Error>::new(
        [1.5, f64::NAN].into_iter(),
    ))
    .unwrap();
    assert_eq!(numbers[0], 1.5);
    assert!(numbers[1].is_nan());
    // JSON null and the TOML adapter's empty string both mean "unavailable".
    let missing =
        deserialize_bout_medians(SeqDeserializer::<_, Error>::new([(), ()].into_iter())).unwrap();
    assert!(missing.len() == 2 && missing.iter().all(|v| v.is_nan()));
    let empty =
        deserialize_bout_medians(SeqDeserializer::<_, Error>::new([""].into_iter())).unwrap();
    assert!(empty[0].is_nan());
    assert!(
        deserialize_bout_medians(SeqDeserializer::<_, Error>::new(["fast"].into_iter())).is_err()
    );
    assert!(
        deserialize_bout_medians(SeqDeserializer::<_, Error>::new(
            [f64::INFINITY].into_iter()
        ))
        .is_err()
    );
}

#[test]
fn bout_medians_ignore_non_finite_samples() {
    let values = floats(&[f64::NAN, 2.0, f64::INFINITY, 4.0]);
    assert_eq!(median_for_range(Some(&values), 0, 4), Some(3.0));
    assert_eq!(median_for_range(Some(&values), 0, 1), None);
    assert_eq!(median_for_range(Some(&values), 1, 2), Some(2.0));
    assert_eq!(median_for_range(None, 0, 4), None);
}
