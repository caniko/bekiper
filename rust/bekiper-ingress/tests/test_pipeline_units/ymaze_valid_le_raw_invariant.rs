#[test]
fn ymaze_valid_le_raw_invariant() {
    let root = unique_pipeline_dir("ymaze_valid_le_raw_invariant");
    let result_path = root.join("data").join("trial").join("result.toml");
    write_ymaze_invariant_result(
        &result_path,
        "ymaze_trial_valid",
        "ymaze__proximal25",
        &["A", "B", "C"],
        &[("A", 1), ("B", 1), ("C", 1)],
        1,
        0,
        0,
        100.0,
    );

    let trial = bekiper_ingress::aggregate::TrialResult::load(&result_path).unwrap();
    let sheets =
        bekiper_ingress::aggregate::build_sheet_dataframes(vec![trial], &Default::default())
            .unwrap();
    let (_, df, _) = sheets
        .iter()
        .find(|(name, _, _)| name == "ymaze__proximal25")
        .expect("ymaze proximal25 sheet");

    assert_eq!(df.height(), 1);
    assert_eq!(
        df.column("valid_total_arm_entries")
            .unwrap()
            .i64()
            .unwrap()
            .get(0),
        Some(3)
    );
    assert_eq!(
        df.column("raw_total_arm_entries")
            .unwrap()
            .i64()
            .unwrap()
            .get(0),
        Some(3)
    );
    assert_eq!(
        df.column("total_triads").unwrap().i64().unwrap().get(0),
        Some(1)
    );
    assert_eq!(
        df.column("spontaneous_alternation_pct_standard")
            .unwrap()
            .f64()
            .unwrap()
            .get(0),
        Some(100.0)
    );

    std::fs::remove_dir_all(&root).ok();
}

#[allow(clippy::too_many_arguments)]
fn write_ymaze_invariant_result(
    path: &std::path::Path,
    experiment: &str,
    analysis_role: &str,
    arm_sequence: &[&str],
    arm_entries: &[(&str, i64)],
    correct_alternations: i64,
    incorrect_alternations: i64,
    neutral_triads: i64,
    spontaneous_alternation_pct: f64,
) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();

    let arm_sequence_toml = arm_sequence
        .iter()
        .map(|label| format!("\"{label}\""))
        .collect::<Vec<_>>()
        .join(", ");
    let raw_arm_sequence_toml = arm_entries
        .iter()
        .flat_map(|(label, entries)| std::iter::repeat_n(*label, *entries as usize))
        .map(|label| format!("\"{label}\""))
        .collect::<Vec<_>>()
        .join(", ");
    let zone_summaries_toml = arm_entries
        .iter()
        .map(|(label, entries)| {
            format!(
                r#"
                [[detailed.zone_summaries]]
                zone_name = "in_arm_{label}"
                entries = {entries}
                "#
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    std::fs::write(
        path,
        format!(
            r#"
            experiment = "{experiment}"
            analysis_role = "{analysis_role}"
            fps = 30.0
            meters_per_pixel = 0.001
            task_type = "y_maze"

            [animal_metadata]
            animal = 42

            [detailed]
            proximal_threshold = 0.25
            start_mode = "video_start"
            arm_membership_hysteresis = true
            total_duration_seconds = 120.0
            analysis_duration_seconds = 120.0
            total_distance_meters = 2.0
            total_mean_speed = 0.02
            seconds_in_arms = 90.0
            seconds_in_arms_and_center = 100.0
            freezing_seconds = 1.0
            area_motion_summaries = []
            excluded = false

            [detailed.analysis]
            arm_sequence = [{arm_sequence_toml}]
            raw_arm_sequence = [{raw_arm_sequence_toml}]
            zone_sequence = [{arm_sequence_toml}]
            first_arm_entered = "A"
            last_arm_entered = "C"
            correct_alternations = {correct_alternations}
            incorrect_alternations = {incorrect_alternations}
            neutral_triads = {neutral_triads}
            spontaneous_alternation_pct = {spontaneous_alternation_pct}
            arm_preference = 0.25

            {zone_summaries_toml}
            "#,
        ),
    )
    .unwrap();
}
