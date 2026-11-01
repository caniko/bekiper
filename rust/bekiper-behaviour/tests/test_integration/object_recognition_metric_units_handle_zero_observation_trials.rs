#[test]
fn object_recognition_metric_units_handle_zero_observation_trials() {
    let mut task = ObjectRecognition::new(
        vec![
            circle_perimeter("object_1", 100.0, 100.0, 0.1),
            circle_perimeter("object_2", 110.0, 110.0, 0.1),
        ],
        30.0,
    );
    task.protocol = ObjectRecognitionProtocol::OulTraining;

    let nort = task.full_analyze(synthetic_tracking_df(90));
    let totals = nort
        .observation_totals
        .as_ref()
        .expect("zero-observation trials should still serialize totals");

    assert_eq!(totals.object_labels, vec!["object_1", "object_2"]);
    assert_eq!(totals.observation_seconds, vec![0.0, 0.0]);
    assert_eq!(totals.observation_bouts, vec![0, 0]);
    assert_eq!(totals.total_seconds, 0.0);
    assert_eq!(totals.total_bouts, 0);
    assert!(!nort.excluded);
    assert!(!nort.passes_15s_exploration_threshold);
    assert_eq!(nort.pairwise_observations.len(), 1);
    assert_eq!(nort.pairwise_observations[0].absolute_delta_seconds, 0.0);
}

#[test]
fn object_recognition_metric_units_cover_four_object_trials() {
    let mut task = ObjectRecognition::new(
        vec![
            circle_perimeter("object_1", 0.8, 0.8, 0.2),
            circle_perimeter("object_2", 1.8, 1.8, 0.2),
            circle_perimeter("object_3", 3.2, 3.2, 0.2),
            circle_perimeter("object_4", 4.4, 4.4, 0.2),
        ],
        30.0,
    );
    task.protocol = ObjectRecognitionProtocol::OulTest;

    let nort = task.full_analyze(synthetic_tracking_df(180));
    let totals = nort
        .observation_totals
        .as_ref()
        .expect("four-object trials should populate observation totals");

    assert_eq!(totals.object_labels.len(), 4);
    assert_eq!(totals.observation_seconds.len(), 4);
    assert_eq!(totals.observation_bouts.len(), 4);
    assert_eq!(nort.pairwise_observations.len(), 6);
    assert_eq!(nort.qualia_rollups.len(), 4);
    assert!(
        nort.indices.is_none(),
        "OUL test should not emit NORT indices"
    );
}

#[test]
fn object_recognition_zone_summaries_apply_dwell_threshold() {
    let mut task = ObjectRecognition::new(vec![circle_perimeter("arena", 0.0, 0.0, 5.0)], 10.0);
    task.center_perimeter = Some(rectangle_perimeter("center_zone", 0.0, 0.0, 1.0, 1.0));

    let result = task.analyze(
        center_periphery_df(10),
        &bekiper_core::video::FrameTiming::cfr(10.0),
    );
    let in_center = result
        .summaries
        .iter()
        .find(|summary| summary.name == "in_center")
        .expect("in_center summary should exist");
    let in_periphery = result
        .summaries
        .iter()
        .find(|summary| summary.name == "in_periphery")
        .expect("in_periphery summary should exist");

    assert_eq!(
        in_center.true_frames, 0,
        "0.5s center dwell should be filtered out"
    );
    assert_eq!(in_periphery.true_frames, 15);
}

#[test]
fn object_recognition_freezing_summary_uses_minimum_bout_length() {
    let task = ObjectRecognition::new(vec![], 10.0);

    let result = task.analyze(
        short_freezing_df(),
        &bekiper_core::video::FrameTiming::cfr(10.0),
    );
    let freezing = result
        .summaries
        .iter()
        .find(|summary| summary.name == "freezing")
        .expect("freezing summary should exist");
    let raw_freezing_frames = result
        .evaluated_df
        .as_ref()
        .unwrap()
        .column("freezing")
        .unwrap()
        .bool()
        .unwrap()
        .sum()
        .unwrap_or(0);

    assert!(
        raw_freezing_frames > 0,
        "raw freezing diagnostic should remain present"
    );
    assert_eq!(
        freezing.true_frames, 0,
        "sub-second freezing bouts should be filtered out"
    );

    let nort = task.full_analyze(short_freezing_df());
    assert_eq!(nort.freezing_seconds, 0.0);
}

#[test]
fn analyze_seconds_consistent_with_fps() {
    let task = ObjectRecognition::new(vec![circle_perimeter("obj", 5.0, 5.0, 0.5)], 60.0);

    let lf = synthetic_tracking_df(60);
    let result = task.analyze(lf, &bekiper_core::video::FrameTiming::cfr(60.0));

    for s in &result.summaries {
        let expected_seconds = s.true_frames as f64 / 60.0;
        assert!(
            (s.seconds - expected_seconds).abs() < 1e-10,
            "seconds should be true_frames / fps"
        );
    }
}

#[test]
fn analysis_result_serde_roundtrip() {
    let task = ObjectRecognition::new(vec![circle_perimeter("obj", 5.0, 5.0, 0.5)], 30.0);

    let lf = synthetic_tracking_df(20);
    let result = task.analyze(lf, &bekiper_core::video::FrameTiming::cfr(30.0));

    let toml_str = toml::to_string(&result).unwrap();
    let deser: bekiper_behaviour::AnalysisResult = toml::from_str(&toml_str).unwrap();

    assert_eq!(deser.task_type, result.task_type);
    assert_eq!(deser.summaries.len(), result.summaries.len());
    assert!(deser.evaluated_df.is_none());
}

#[test]
fn qualia_decomposition_populated_for_two_objects() {
    let task = ObjectRecognition::new(
        vec![
            circle_perimeter("obj1", 5.0, 5.0, 0.5),
            circle_perimeter("obj2", 1.0, 1.0, 0.3),
        ],
        30.0,
    );

    let lf = synthetic_tracking_df(100);
    let nort = task.full_analyze(lf);

    assert_eq!(nort.protocol, ObjectRecognitionProtocol::NortNovelty);
    let qualia = nort.qualia.as_ref().expect("qualia should be present");
    assert_eq!(qualia.channels.len(), 3);

    let types: Vec<&str> = qualia
        .channels
        .iter()
        .map(|c| c.heuristic_type.as_str())
        .collect();
    assert!(types.contains(&"body_proximity"));
    assert!(types.contains(&"olfaction"));
    assert!(types.contains(&"whisker_interaction"));

    // Each channel should have per-object data matching the 2 objects.
    for ch in &qualia.channels {
        assert_eq!(ch.per_object_seconds.len(), 2);
        assert_eq!(ch.per_object_bouts.len(), 2);
        assert!(ch.per_object_seconds.iter().all(|&s| s >= 0.0));
    }

    // Relative bias should sum to ~100% per channel (or all zeros).
    for bias in qualia.relative_bias.values() {
        let total: f64 = bias.iter().sum();
        assert!(total < 1e-10 || (total - 100.0).abs() < 1e-10);
    }

    // Observation sequence length matches windowed DataFrame.
    assert!(!qualia.observation_sequence.is_empty());
    assert!(qualia.observation_sequence.iter().all(|&v| v <= 2));
}

#[test]
fn qualia_disabled_produces_none() {
    let mut task = ObjectRecognition::new(vec![circle_perimeter("obj1", 5.0, 5.0, 0.5)], 30.0);
    task.enable_qualia = false;

    let lf = synthetic_tracking_df(50);
    let nort = task.full_analyze(lf);

    assert!(nort.qualia.is_none());
    // Only exploration + freezing heuristics, no qualia.
}

#[test]
fn qualia_with_merge_rule() {
    let mut task = ObjectRecognition::new(
        vec![
            circle_perimeter("obj1", 5.0, 5.0, 0.5),
            circle_perimeter("obj2", 1.0, 1.0, 0.3),
        ],
        30.0,
    );

    // Merge rule: require both olfaction AND body_proximity.
    let mut rules = HashMap::new();
    rules.insert(
        "close_investigation".into(),
        MergeRule {
            require: vec!["olfaction".into(), "body_proximity".into()],
            any_of: vec![],
            exclude: vec![],
        },
    );
    task.merge_rules = rules;

    let lf = synthetic_tracking_df(100);
    let nort = task.full_analyze(lf);

    let qualia = nort.qualia.as_ref().expect("qualia should be present");
    assert!(qualia.merged.contains_key("close_investigation"));
    let merged_secs = &qualia.merged["close_investigation"];
    assert_eq!(merged_secs.len(), 2); // 2 objects

    // Merged result should be <= min(olfaction, body_proximity) for each object
    // since require = AND.
    let olf = qualia
        .channels
        .iter()
        .find(|c| c.heuristic_type == "olfaction")
        .unwrap();
    let bp = qualia
        .channels
        .iter()
        .find(|c| c.heuristic_type == "body_proximity")
        .unwrap();
    for (i, merged) in merged_secs.iter().enumerate().take(2) {
        assert!(*merged <= olf.per_object_seconds[i] + 1e-6);
        assert!(*merged <= bp.per_object_seconds[i] + 1e-6);
    }
}

#[test]
fn qualia_merge_rule_adds_heuristics() {
    let mut task = ObjectRecognition::new(vec![circle_perimeter("obj1", 5.0, 5.0, 0.5)], 30.0);
    let mut rules = HashMap::new();
    rules.insert(
        "combined".into(),
        MergeRule {
            require: vec!["olfaction".into()],
            any_of: vec!["whisker_interaction".into()],
            exclude: vec![],
        },
    );
    task.merge_rules = rules;

    let heuristics = task.heuristics();
    let names: Vec<&str> = heuristics.iter().map(|h| h.name()).collect();

    // Should include the merged heuristic per object.
    assert!(names.contains(&"combined_obj1"));
}

#[test]
fn posture_exclusion_runs_end_to_end_and_reports_its_removals() {
    let perimeters = vec![
        circle_perimeter("obj1", 5.0, 5.0, 0.5),
        circle_perimeter("obj2", 1.0, 1.0, 0.3),
    ];
    let plain =
        ObjectRecognition::new(perimeters.clone(), 30.0).full_analyze(synthetic_tracking_df(100));
    let mut task = ObjectRecognition::new(perimeters, 30.0);
    task.exclude_posture = true;
    let nort = task.full_analyze(synthetic_tracking_df(100));
    assert_eq!(nort.exploration_attributions.len(), 2);
    // Posture removes frames only from the contact rescue, never adds any.
    for (with, without) in nort
        .exploration_attributions
        .iter()
        .zip(&plain.exploration_attributions)
    {
        assert!(with.frames_all_gates_raw <= without.frames_all_gates_raw);
        assert!(
            with.frames_all_gates_raw + with.candidate_frames_removed_by_posture as u64
                >= without.frames_all_gates_raw
        );
    }
}

#[test]
fn qualia_nort_result_serde_roundtrip() {
    let task = ObjectRecognition::new(
        vec![
            circle_perimeter("obj1", 5.0, 5.0, 0.5),
            circle_perimeter("obj2", 1.0, 1.0, 0.3),
        ],
        30.0,
    );

    let lf = synthetic_tracking_df(100);
    let nort = task.full_analyze(lf);

    // Serialize NortResult to JSON and back (mirrors the detailed result path).
    let json = serde_json::to_value(&nort).unwrap();
    let restored: bekiper_behaviour::object_recognition::NortResult =
        serde_json::from_value(json).unwrap();

    let totals = restored
        .observation_totals
        .as_ref()
        .expect("observation totals should roundtrip");
    // Object A is the left object (smaller centroid x), not the first annotated.
    assert_eq!(totals.object_labels, vec!["obj2", "obj1"]);
    assert_eq!(restored.pairwise_observations.len(), 1);
    assert_eq!(restored.qualia_rollups.len(), 2);
    assert!(restored.zone_metrics.is_none());

    let q = restored.qualia.as_ref().unwrap();
    assert_eq!(q.channels.len(), 3);
    assert_eq!(
        q.observation_sequence.len(),
        nort.qualia.as_ref().unwrap().observation_sequence.len()
    );
    assert_eq!(
        q.alternation_count,
        nort.qualia.as_ref().unwrap().alternation_count
    );
}

#[test]
fn nort_center_latency_is_the_first_dwell_qualified_entry_from_trial_start() {
    // 10 fps: outside 2 s, a 0.5 s centre visit (too short for the 1 s dwell),
    // outside 1 s, then 2.5 s in the centre: latency 3.5 s from frame 0.
    let inside = |i: usize| (20..25).contains(&i) || (35..60).contains(&i);
    let n = 60;
    let pos: Vec<f64> = (0..n).map(|i| if inside(i) { 0.0 } else { 3.0 }).collect();
    let shifted = |d: f64| pos.iter().map(|p| p + d).collect::<Vec<_>>();
    let lf = df! {
        "nose_x" => shifted(0.03), "nose_y" => pos.clone(),
        "center_ear_x" => shifted(0.01), "center_ear_y" => pos.clone(),
        "tail_base_x" => shifted(-0.05), "tail_base_y" => pos.clone(),
        "mouse_center_x" => pos.clone(), "mouse_center_y" => pos.clone(),
        "snout_dx" => vec![1.0; n], "snout_dy" => vec![0.0; n],
    }
    .unwrap()
    .lazy();
    let mut task = ObjectRecognition::new(vec![circle_perimeter("obj", 9.0, 9.0, 0.1)], 10.0);
    task.center_perimeter = Some(rectangle_perimeter("center_zone", 0.0, 0.0, 1.0, 1.0));
    task.enable_qualia = false;
    let nort = task.full_analyze(lf);
    assert_eq!(nort.latency_to_center, Some(3.5));
}
