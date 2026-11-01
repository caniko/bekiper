#[test]
fn redetect_replaces_only_flagged_labels() {
    let df = polars::df! {
        "frame" => &[0i64, 1, 2],
        "a_x" => &[0.0, 0.0, 0.0],
        "a_y" => &[0.0, 0.0, 0.0],
        "b_x" => &[1.0, 1.0, 1.0],
        "b_y" => &[0.0, 0.0, 0.0],
        "c_x" => &[2.0, 2.0, 10.0],
        "c_y" => &[0.0, 0.0, 0.0],
        "c_likelihood" => &[0.9, 0.9, 0.1],
    }
    .unwrap();

    let step = SkeletonRepairStep {
        skeleton: line_skeleton(),
        relative_tolerance: 0.35,
        mad_multiplier: 4.0,
        min_baseline_frames: 2,
    };

    let fitted = step.fit(&df).unwrap();
    let initial_eval = fitted.evaluate(&df).unwrap();
    let requests = initial_eval.flagged_rows(&df).unwrap();
    let mut final_eval = initial_eval.clone();
    let mut redetected_rows = BTreeSet::new();
    let mut repaired_df = fitted
        .apply(df.clone(), &initial_eval, &BTreeSet::new())
        .unwrap();

    apply_redetected_candidates_with(
        &fitted,
        &requests,
        0.5,
        &mut repaired_df,
        &mut final_eval,
        &mut redetected_rows,
        |_| {
            Ok(Some(HashMap::from([(
                "c".to_string(),
                RedetectedKeypoint {
                    x: 2.0,
                    y: 0.0,
                    likelihood: 0.95,
                },
            )])))
        },
    )
    .unwrap();

    let final_df = fitted
        .apply(repaired_df, &final_eval, &redetected_rows)
        .unwrap();
    assert_eq!(
        final_df.column("a_x").unwrap().get(2).unwrap(),
        AnyValue::Float64(0.0)
    );
    assert_eq!(
        final_df.column("b_x").unwrap().get(2).unwrap(),
        AnyValue::Float64(1.0)
    );
    assert_eq!(
        final_df.column("c_x").unwrap().get(2).unwrap(),
        AnyValue::Float64(2.0)
    );
    assert_eq!(
        final_df
            .column("skeleton_redetected")
            .unwrap()
            .bool()
            .unwrap()
            .get(2),
        Some(true)
    );
    assert_eq!(
        final_df
            .column("skeleton_violation_count")
            .unwrap()
            .u32()
            .unwrap()
            .get(2),
        Some(0)
    );
}
