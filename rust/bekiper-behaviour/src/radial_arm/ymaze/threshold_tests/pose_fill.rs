#[test]
fn zone_max_gap_seconds_defaults_to_point_one() {
    let ymaze = synthetic_ymaze();
    assert!((ymaze.zone_max_gap_seconds - DEFAULT_ZONE_MAX_GAP_SECONDS).abs() < f64::EPSILON);
}

#[test]
fn arm_membership_hysteresis_defaults_to_true() {
    let ymaze = synthetic_ymaze();
    assert!(ymaze.arm_membership_hysteresis);
}

#[test]
fn pose_gap_fill_seconds_defaults_to_point_two() {
    let ymaze = synthetic_ymaze();
    assert!((ymaze.pose_gap_fill_seconds - DEFAULT_POSE_GAP_FILL_SECONDS).abs() < f64::EPSILON);
}

#[test]
fn ttl_zero_is_a_noop() {
    assert_eq!(
        filled_values(&[Some(1.0), None, Some(2.0)], 0),
        vec![Some(1.0), None, Some(2.0)]
    );
}

#[test]
fn ttl_forward_fill_within_ttl_fills() {
    assert_eq!(
        filled_values(&[Some(1.0), None, None, Some(2.0)], 2),
        vec![Some(1.0), Some(1.0), Some(1.0), Some(2.0)]
    );
}

#[test]
fn ttl_forward_fill_beyond_ttl_stops() {
    assert_eq!(
        filled_values(&[Some(1.0), None, None, None, Some(2.0)], 2),
        vec![Some(1.0), Some(1.0), Some(1.0), None, Some(2.0)]
    );
}

#[test]
fn ttl_forward_fill_leading_nulls_stay_null() {
    assert_eq!(
        filled_values(&[None, None, Some(2.0)], 2),
        vec![None, None, Some(2.0)]
    );
}

#[test]
fn ttl_forward_fill_resets_after_each_real_value() {
    assert_eq!(
        filled_values(
            &[Some(1.0), None, Some(2.0), None, None, None, Some(3.0)],
            2,
        ),
        vec![
            Some(1.0),
            Some(1.0),
            Some(2.0),
            Some(2.0),
            Some(2.0),
            None,
            Some(3.0),
        ]
    );
}

#[test]
fn pose_gap_fill_ttl_negative_seconds_coerce_to_zero_frames() {
    assert_eq!(ttl_frames_from_seconds(-1.0, 30.0), 0);
}

#[test]
fn pose_gap_fill_ttl_zero_leaves_lazyframe_bit_identical() {
    let original = df! {
        "center_ear_x" => [Some(0.9), None, Some(0.9)],
        "center_ear_y" => [Some(0.0), Some(0.0), Some(0.0)],
        "tail_base_x" => [Some(0.9), Some(0.9), Some(0.9)],
        "tail_base_y" => [Some(0.0), Some(0.0), Some(0.0)],
        "mouse_center_x" => [Some(0.5), Some(0.6), Some(0.7)],
        "mouse_center_y" => [Some(0.0), Some(0.0), Some(0.0)],
    }
    .unwrap();
    let transformed = apply_ttl_forward_fill(
        original.clone().lazy(),
        &["center_ear_x", "center_ear_y", "tail_base_x", "tail_base_y"],
        0,
    )
    .collect()
    .expect("collect no-op gap fill");

    assert_eq!(transformed.get_column_names(), original.get_column_names());
    for column_name in original.get_column_names() {
        assert_eq!(
            transformed.column(column_name.as_str()).unwrap(),
            original.column(column_name.as_str()).unwrap(),
            "column {column_name} changed under ttl=0"
        );
    }
}

#[test]
fn pose_gap_fill_targets_only_head_and_tail_axes() {
    let ymaze = synthetic_ymaze();
    assert_eq!(
        ymaze.pose_gap_fill_columns(),
        [
            "center_ear_x".to_string(),
            "center_ear_y".to_string(),
            "tail_base_x".to_string(),
            "tail_base_y".to_string(),
        ]
    );
}
