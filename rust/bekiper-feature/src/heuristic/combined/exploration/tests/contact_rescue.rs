#[test]
fn per_frame_contact_override_admits_high_angle_frame_to_bout() {
    let raw_contact_or_angle = vec![true, true, true];
    let speeds = vec![0.02; 3];
    let angles = vec![20.0, 60.0, 20.0];
    assert_eq!(
        filtered_count(
            &raw_contact_or_angle,
            &speeds,
            &angles,
            BoutFilterMode::PerFrame,
            BoutFilterMode::PerFrame,
            1,
            0,
        ),
        3
    );
}

#[test]
fn per_bout_median_angle_rescued_by_nose_contact_frame() {
    // Bout-level median angle filter would drop this 30-frame bout
    // (median = 60°). The spec's nose-contact override rescues the
    // bout because one frame had nose_contact == true.
    let raw = bools(&[true; 30]);
    let speeds = floats(&[0.02; 30]);
    let angles = floats(&[60.0; 30]);
    let mut contact_vec = vec![false; 30];
    contact_vec[5] = true;
    let nose_contact = bools(&contact_vec);

    let (filtered, diagnostics) = apply_bout_level_filters_with_diagnostics(
        &raw,
        10,
        0,
        0.0,
        test_filter_spec(
            BoutFilterMode::PerFrame,
            BoutFilterMode::PerBoutMedian,
            Some(&speeds),
            Some(&angles),
            Some(&nose_contact),
        ),
    );

    assert_eq!(filtered.sum().unwrap_or(0), 30);
    assert_eq!(diagnostics.final_valid_bouts, 1);
    assert_eq!(diagnostics.removed_by_angle, 0);
    assert_eq!(diagnostics.bouts_rescued_by_contact_override, 1);
}

#[test]
fn per_bout_median_angle_no_rescue_when_no_contact_frame() {
    let raw = bools(&[true; 30]);
    let speeds = floats(&[0.02; 30]);
    let angles = floats(&[60.0; 30]);
    let nose_contact = bools(&[false; 30]);

    let (filtered, diagnostics) = apply_bout_level_filters_with_diagnostics(
        &raw,
        10,
        0,
        0.0,
        test_filter_spec(
            BoutFilterMode::PerFrame,
            BoutFilterMode::PerBoutMedian,
            Some(&speeds),
            Some(&angles),
            Some(&nose_contact),
        ),
    );

    assert_eq!(filtered.sum().unwrap_or(0), 0);
    assert_eq!(diagnostics.final_valid_bouts, 0);
    assert_eq!(diagnostics.removed_by_angle, 1);
    assert_eq!(diagnostics.bouts_rescued_by_contact_override, 0);
}

#[test]
fn contact_override_inactive_when_nose_contact_arg_is_none() {
    // High-angle bout, contact arg = None → no rescue path available.
    let raw = bools(&[true; 30]);
    let speeds = floats(&[0.02; 30]);
    let angles = floats(&[60.0; 30]);

    let (_, diagnostics) = apply_bout_level_filters_with_diagnostics(
        &raw,
        10,
        0,
        0.0,
        test_filter_spec(
            BoutFilterMode::PerFrame,
            BoutFilterMode::PerBoutMedian,
            Some(&speeds),
            Some(&angles),
            None,
        ),
    );

    assert_eq!(diagnostics.removed_by_angle, 1);
    assert_eq!(diagnostics.bouts_rescued_by_contact_override, 0);
}

#[test]
fn contact_override_does_not_fire_in_per_frame_angle_mode() {
    // Spec attaches the override to the per-bout-median path. In
    // PerFrame angle mode the candidate-frame mask already gates on
    // contact_or_angle, so we must not double-count rescues here.
    let raw = bools(&[true; 30]);
    let speeds = floats(&[0.02; 30]);
    let angles = floats(&[60.0; 30]);
    let mut contact_vec = vec![false; 30];
    contact_vec[5] = true;
    let nose_contact = bools(&contact_vec);

    let (_, diagnostics) = apply_bout_level_filters_with_diagnostics(
        &raw,
        10,
        0,
        0.0,
        test_filter_spec(
            BoutFilterMode::PerFrame,
            BoutFilterMode::PerFrame,
            Some(&speeds),
            Some(&angles),
            Some(&nose_contact),
        ),
    );

    assert_eq!(diagnostics.bouts_rescued_by_contact_override, 0);
}

#[test]
fn contact_override_does_not_rescue_speed_failures() {
    // Bout fails BOTH the per-bout median speed and angle checks. A
    // contact frame present must NOT rescue speed-fail bouts — the
    // spec scopes the override to the angle filter.
    let raw = bools(&[true; 30]);
    let speeds = floats(&[0.10; 30]); // median > 0.05 → fails speed
    let angles = floats(&[60.0; 30]); // median > 45° → would fail angle
    let mut contact_vec = vec![false; 30];
    contact_vec[5] = true;
    let nose_contact = bools(&contact_vec);

    let (filtered, diagnostics) = apply_bout_level_filters_with_diagnostics(
        &raw,
        10,
        0,
        0.0,
        test_filter_spec(
            BoutFilterMode::PerBoutMedian,
            BoutFilterMode::PerBoutMedian,
            Some(&speeds),
            Some(&angles),
            Some(&nose_contact),
        ),
    );

    assert_eq!(filtered.sum().unwrap_or(0), 0);
    assert_eq!(diagnostics.final_valid_bouts, 0);
    assert_eq!(diagnostics.removed_by_speed, 1);
    assert_eq!(diagnostics.removed_by_angle, 0);
    assert_eq!(diagnostics.bouts_rescued_by_contact_override, 0);
}

#[test]
fn contact_override_handles_multiple_bouts_independently() {
    // Two bouts both fail per-bout median angle. The first has a
    // contact frame and should be rescued; the second does not and
    // should be dropped.
    let mut raw_vec = Vec::new();
    raw_vec.extend(vec![true; 30]);
    raw_vec.extend(vec![false; 10]);
    raw_vec.extend(vec![true; 30]);
    let raw = bools(&raw_vec);
    let speeds = floats(&[0.02; 70]);
    let angles = floats(&[60.0; 70]);
    let mut contact_vec = vec![false; 70];
    contact_vec[10] = true; // contact frame inside the first bout
    let nose_contact = bools(&contact_vec);

    let (filtered, diagnostics) = apply_bout_level_filters_with_diagnostics(
        &raw,
        10,
        0,
        0.0,
        test_filter_spec(
            BoutFilterMode::PerFrame,
            BoutFilterMode::PerBoutMedian,
            Some(&speeds),
            Some(&angles),
            Some(&nose_contact),
        ),
    );

    assert_eq!(
        filtered.sum().unwrap_or(0),
        30,
        "only the first bout survives"
    );
    assert_eq!(diagnostics.final_valid_bouts, 1);
    assert_eq!(diagnostics.removed_by_angle, 1);
    assert_eq!(diagnostics.bouts_rescued_by_contact_override, 1);
}

#[test]
fn rescued_bout_appears_in_per_bout_medians() {
    // A rescued bout is a final valid bout — its per-bout medians
    // must be appended so the vectors stay aligned with
    // `final_valid_bouts`.
    let raw = bools(&[true; 30]);
    let speeds = floats(&[0.02; 30]);
    let mut angles_vec = vec![60.0; 30];
    angles_vec[14] = 60.0;
    let angles = floats(&angles_vec);
    let mut contact_vec = vec![false; 30];
    contact_vec[2] = true;
    let nose_contact = bools(&contact_vec);

    let (_, diagnostics) = apply_bout_level_filters_with_diagnostics(
        &raw,
        10,
        0,
        0.0,
        test_filter_spec(
            BoutFilterMode::PerFrame,
            BoutFilterMode::PerBoutMedian,
            Some(&speeds),
            Some(&angles),
            Some(&nose_contact),
        ),
    );

    assert_eq!(diagnostics.bouts_rescued_by_contact_override, 1);
    assert_eq!(
        diagnostics.per_bout_median_nose_speed.len(),
        diagnostics.final_valid_bouts as usize
    );
    assert_eq!(
        diagnostics.per_bout_median_head_angle_deg.len(),
        diagnostics.final_valid_bouts as usize
    );
    assert_eq!(diagnostics.per_bout_median_nose_speed, vec![0.02]);
    assert_eq!(diagnostics.per_bout_median_head_angle_deg, vec![60.0]);
}

#[test]
fn conservation_identity_holds_with_rescue() {
    // post_merge − removed_by_duration − removed_by_speed − removed_by_angle
    //     == final_valid_bouts
    // Verified across a mixed scenario with one rescued bout.
    let mut raw_vec = Vec::new();
    raw_vec.extend(vec![true; 30]); // bout A: passes outright
    raw_vec.extend(vec![false; 5]);
    raw_vec.extend(vec![true; 30]); // bout B: high angle, rescued
    raw_vec.extend(vec![false; 5]);
    raw_vec.extend(vec![true; 30]); // bout C: high angle, dropped
    let raw = bools(&raw_vec);
    let mut speeds_vec = Vec::new();
    speeds_vec.extend(vec![0.02; 30]);
    speeds_vec.extend(vec![0.02; 5]);
    speeds_vec.extend(vec![0.02; 30]);
    speeds_vec.extend(vec![0.02; 5]);
    speeds_vec.extend(vec![0.02; 30]);
    let speeds = floats(&speeds_vec);
    let mut angles_vec = Vec::new();
    angles_vec.extend(vec![20.0; 30]);
    angles_vec.extend(vec![20.0; 5]);
    angles_vec.extend(vec![60.0; 30]);
    angles_vec.extend(vec![20.0; 5]);
    angles_vec.extend(vec![60.0; 30]);
    let angles = floats(&angles_vec);
    let mut contact_vec = vec![false; 100];
    contact_vec[40] = true; // contact in bout B → rescues
    let nose_contact = bools(&contact_vec);

    let (_, diagnostics) = apply_bout_level_filters_with_diagnostics(
        &raw,
        10,
        0,
        0.0,
        test_filter_spec(
            BoutFilterMode::PerFrame,
            BoutFilterMode::PerBoutMedian,
            Some(&speeds),
            Some(&angles),
            Some(&nose_contact),
        ),
    );

    // Bouts A + B (rescued) survive; bout C is removed by angle.
    assert_eq!(diagnostics.final_valid_bouts, 2);
    assert_eq!(diagnostics.removed_by_angle, 1);
    assert_eq!(diagnostics.bouts_rescued_by_contact_override, 1);

    let conservation = diagnostics.bouts_post_merge
        - diagnostics.removed_by_duration
        - diagnostics.removed_by_speed
        - diagnostics.removed_by_angle;
    assert_eq!(
        conservation, diagnostics.final_valid_bouts,
        "post_merge − dur − speed − angle should equal final_valid_bouts",
    );
}

#[test]
fn rescue_at_first_frame_of_bout_counts() {
    // Edge: contact frame sits at the very first frame of the bout.
    let raw = bools(&[true; 20]);
    let speeds = floats(&[0.02; 20]);
    let angles = floats(&[60.0; 20]);
    let mut contact_vec = vec![false; 20];
    contact_vec[0] = true;
    let nose_contact = bools(&contact_vec);

    let (_, diagnostics) = apply_bout_level_filters_with_diagnostics(
        &raw,
        10,
        0,
        0.0,
        test_filter_spec(
            BoutFilterMode::PerFrame,
            BoutFilterMode::PerBoutMedian,
            Some(&speeds),
            Some(&angles),
            Some(&nose_contact),
        ),
    );

    assert_eq!(diagnostics.bouts_rescued_by_contact_override, 1);
    assert_eq!(diagnostics.final_valid_bouts, 1);
}

#[test]
fn rescue_at_last_frame_of_bout_counts() {
    // Edge: contact frame sits at the very last frame of the bout.
    let raw = bools(&[true; 20]);
    let speeds = floats(&[0.02; 20]);
    let angles = floats(&[60.0; 20]);
    let mut contact_vec = vec![false; 20];
    contact_vec[19] = true;
    let nose_contact = bools(&contact_vec);

    let (_, diagnostics) = apply_bout_level_filters_with_diagnostics(
        &raw,
        10,
        0,
        0.0,
        test_filter_spec(
            BoutFilterMode::PerFrame,
            BoutFilterMode::PerBoutMedian,
            Some(&speeds),
            Some(&angles),
            Some(&nose_contact),
        ),
    );

    assert_eq!(diagnostics.bouts_rescued_by_contact_override, 1);
    assert_eq!(diagnostics.final_valid_bouts, 1);
}

#[test]
fn rescued_bout_already_passing_does_not_increment_counter() {
    // A bout whose median angle is already ≤ 45° passes outright; the
    // rescue counter must not double-count it even if a contact frame
    // happens to be present.
    let raw = bools(&[true; 30]);
    let speeds = floats(&[0.02; 30]);
    let angles = floats(&[20.0; 30]);
    let mut contact_vec = vec![false; 30];
    contact_vec[15] = true;
    let nose_contact = bools(&contact_vec);

    let (_, diagnostics) = apply_bout_level_filters_with_diagnostics(
        &raw,
        10,
        0,
        0.0,
        test_filter_spec(
            BoutFilterMode::PerFrame,
            BoutFilterMode::PerBoutMedian,
            Some(&speeds),
            Some(&angles),
            Some(&nose_contact),
        ),
    );

    assert_eq!(diagnostics.final_valid_bouts, 1);
    assert_eq!(diagnostics.removed_by_angle, 0);
    assert_eq!(
        diagnostics.bouts_rescued_by_contact_override, 0,
        "rescue counter must not fire for bouts that pass on their own merits",
    );
}
