use super::*;
use crate::config::{AggregateConfig, AggregationRule};
use indexmap::IndexMap;

fn write_result_toml(path: &Path) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(
        path,
        r#"
            experiment = "721_seq0_oultraining"
            fps = 30.0
            meters_per_pixel = 0.001
            task_type = "object_recognition"

            [detailed]
            protocol = "oul_training"
            excluded = false
            total_duration_seconds = 120.0
            total_distance_meters = 5.0
            total_mean_speed = 0.04
            freezing_seconds = 1.2
            analysis_window_seconds = 120.0
            object_labels = []
            exploration_times = []
            bout_counts = []
            "#,
    )
    .unwrap();
}

#[test]
fn result_walk_keeps_depth_bounds_and_stops_below_result_directories() {
    let root = std::env::temp_dir().join(format!("bekiper_result_walk_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    for relative in [
        "direct/result.toml",
        "direct/hidden/result.toml",
        "variant/trial/result.toml",
        "too/deep/trial/result.toml",
    ] {
        write_result_toml(&root.join(relative));
    }
    let corrupt = root.join("corrupt/result.toml");
    std::fs::create_dir_all(corrupt.parent().unwrap()).unwrap();
    std::fs::write(&corrupt, "not a result").unwrap();
    write_result_toml(&root.join("result.toml"));
    let mut paths = collect_result_paths(&root, 0, 1);
    paths.sort();
    let mut expected = [
        "corrupt/result.toml",
        "direct/result.toml",
        "variant/trial/result.toml",
    ]
    .map(|path| root.join(path))
    .to_vec();
    expected.sort();
    assert_eq!(paths, expected);
    let mut trials = Vec::new();
    collect_trial_results(&root, &mut trials, 0, 1);
    assert_eq!(
        trials.len(),
        2,
        "a corrupt discovered path is retained by inventory but skipped by decoding"
    );
    assert_eq!(collect_result_paths(&root, 0, 0).len(), 2);
    std::fs::remove_dir_all(root).unwrap();
}

fn write_unkeyed_result_toml(path: &Path) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(
        path,
        r#"
            experiment = "unkeyed_trial"
            fps = 30.0
            meters_per_pixel = 0.001
            task_type = "object_recognition"

            [detailed]
            excluded = false
            total_duration_seconds = 120.0
            total_distance_meters = 5.0
            total_mean_speed = 0.04
            freezing_seconds = 1.2
            analysis_window_seconds = 120.0
            object_labels = []
            exploration_times = []
            bout_counts = []
            "#,
    )
    .unwrap();
}

fn write_open_field_result_toml(path: &Path, experiment: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(
        path,
        format!(
            r#"
                experiment = "{experiment}"
                fps = 30.0
                meters_per_pixel = 0.001
                task_type = "open_field"

                [session_context]
                source_file = "examples/neo/imen/annotations/nort/metadata.csv"
                source_row = 12
                fecal_boli_count = 3

                [detailed]
                center_time_pct = 10.0
                periphery_time_pct = 90.0
                total_distance_meters = 4.0
                total_duration_seconds = 120.0
                total_mean_speed = 0.03
                freezing_seconds = 0.0
                grooming_seconds = 0.0
                rearing_seconds = 0.0
                immobility_pct = 0.0
                "#
        ),
    )
    .unwrap();
}

fn write_nort_result_toml(path: &Path, experiment: &str, protocol: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(
        path,
        format!(
            r#"
                experiment = "{experiment}"
                fps = 30.0
                meters_per_pixel = 0.001
                task_type = "object_recognition"

                [detailed]
                protocol = "{protocol}"
                excluded = false
                total_duration_seconds = 120.0
                total_distance_meters = 5.0
                total_mean_speed = 0.04
                freezing_seconds = 1.2
                analysis_window_seconds = 120.0
                object_labels = []
                exploration_times = []
                bout_counts = []
                "#
        ),
    )
    .unwrap();
}

fn write_ymaze_result_toml(path: &Path, experiment: &str, entries: [u32; 3], durations: [f64; 3]) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(
        path,
        format!(
            r#"
                experiment = "{experiment}"
                fps = 30.0
                meters_per_pixel = 0.001
                task_type = "y_maze"

                [detailed]
                seconds_in_arms = 1.0
                seconds_in_arms_and_center = 1.0
                freezing_seconds = 0.0
                total_distance_meters = 1.0
                total_duration_seconds = 120.0
                total_mean_speed = 0.01

                [detailed.analysis]
                arm_entries = [{}, {}, {}]
                arm_preference = 0.0
                arm_sequence = ["A"]
                center_entries = 0
                correct_alternations = 0
                incorrect_alternations = 0
                neutral_triads = 0
                spontaneous_alternation_pct = 0.0
                sum_of_permutation_distribution = 0
                zone_sequence = ["A"]

                [detailed.analysis.permutation_distribution]
                abc = 0
                acb = 0
                bac = 0
                bca = 0
                cab = 0
                cba = 0

                [[detailed.zone_summaries]]
                zone_name = "in_arm_A"
                entries = {}
                duration_seconds = {}
                distance_meters = 1.0
                mean_speed = 0.1

                [[detailed.zone_summaries]]
                zone_name = "in_arm_B"
                entries = {}
                duration_seconds = {}
                distance_meters = 0.0
                mean_speed = 0.0

                [[detailed.zone_summaries]]
                zone_name = "in_arm_C"
                entries = {}
                duration_seconds = {}
                distance_meters = 0.0
                mean_speed = 0.0

                [[detailed.zone_summaries]]
                zone_name = "in_center"
                entries = 0
                duration_seconds = 0.0
                distance_meters = 0.0
                mean_speed = 0.0
                "#,
            entries[0],
            entries[1],
            entries[2],
            entries[0],
            durations[0],
            entries[1],
            durations[1],
            entries[2],
            durations[2],
        ),
    )
    .unwrap();
}

fn write_pixel_perimeters(path: &Path, labels: &[&str]) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let mut content = String::new();
    for label in labels {
        content.push_str(&format!(
            r#"
                [[perimeters]]
                label = "{label}"
                shape = "polygon"
                vertices = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
                "#
        ));
    }
    std::fs::write(path, content).unwrap();
}

fn config_with_aggregate(aggregate: AggregateConfig) -> IngressConfig {
    IngressConfig {
        inspect: Default::default(),
        review: Default::default(),
        project: bekiper_core::config::ProjectConfig {
            project_name: "aggregate".into(),
            data_directory: ".".into(),
            output_dir: Some("output".into()),
            runtime: Default::default(),
        },
        pipeline: crate::config::PipelineConfig::default(),
        gpu: crate::config::GpuConfig::default(),
        aggregate,
        post_analysis: crate::config::PostAnalysisConfig::default(),
        pose_model: None,
        scan: Vec::new(),
        animal_scan: Vec::new(),
        ingress_profile: Vec::new(),
        experiments: Vec::new(),
        animal_metadata: Default::default(),
        override_record: Vec::new(),
        toml_paths: Vec::new(),
    }
}

fn assert_aggregate_layout(name: &str, result_path: &str) {
    let dir = std::env::temp_dir().join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    write_result_toml(&dir.join(result_path));

    let config = config_with_aggregate(AggregateConfig::default());
    run_aggregate(&config, &dir).unwrap();
    assert!(dir.join("results.ods").exists());
    assert!(is_aggregate_fresh(&config, &dir));

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn aggregate_freshness_invalidates_when_recipe_mapping_changes() {
    let dir = std::env::temp_dir().join("bekiper_aggregate_freshness_recipe_map");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    write_result_toml(&dir.join("trial/result.toml"));

    let config = config_with_aggregate(AggregateConfig::default());
    run_aggregate(&config, &dir).unwrap();
    assert!(is_aggregate_fresh(&config, &dir));

    let mut recipe_to_sheet = IndexMap::new();
    recipe_to_sheet.insert("oultraining".to_string(), "training".to_string());
    let changed = config_with_aggregate(AggregateConfig {
        recipe_to_sheet,
        ..AggregateConfig::default()
    });
    assert!(!is_aggregate_fresh(&changed, &dir));

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn aggregate_prefers_data_dir_for_new_layout() {
    assert_aggregate_layout("bekiper_aggregate_data_layout", "data/trial/result.toml");
}

#[test]
fn aggregate_uses_legacy_layout_when_data_dir_is_absent() {
    assert_aggregate_layout("bekiper_aggregate_legacy_layout", "trial/result.toml");
}

#[test]
fn aggregate_freshness_invalidates_when_passthrough_declarations_change() {
    let dir = std::env::temp_dir().join("bekiper_aggregate_freshness_passthrough");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    write_result_toml(&dir.join("trial/result.toml"));

    let mut recipe_to_sheet = IndexMap::new();
    recipe_to_sheet.insert("oultraining".to_string(), "training".to_string());
    let config = config_with_aggregate(AggregateConfig {
        recipe_to_sheet: recipe_to_sheet.clone(),
        ..AggregateConfig::default()
    });
    run_aggregate(&config, &dir).unwrap();
    assert!(is_aggregate_fresh(&config, &dir));

    let mut aggregation = IndexMap::new();
    aggregation.insert("training".to_string(), AggregationRule::Passthrough);
    let changed = config_with_aggregate(AggregateConfig {
        recipe_to_sheet,
        aggregation,
        ..AggregateConfig::default()
    });
    assert!(!is_aggregate_fresh(&changed, &dir));

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn aggregate_fails_when_trials_produce_no_sheets() {
    let dir = std::env::temp_dir().join("bekiper_aggregate_no_sheets");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    write_unkeyed_result_toml(&dir.join("trial/result.toml"));

    let config = config_with_aggregate(AggregateConfig::default());
    let err = run_aggregate(&config, &dir).unwrap_err();
    let message = err.to_string();
    assert!(
        message.contains("aggregate produced no sheets"),
        "{message}"
    );
    assert!(
        !aggregate_output_path_for_config(&config, &dir).exists(),
        "empty aggregate workbook should not be written"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn aggregate_allows_missing_oft_fecal_boli_source() {
    let dir = std::env::temp_dir().join("bekiper_aggregate_missing_fecal_boli");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("oft_trial")).unwrap();
    std::fs::write(
        dir.join("oft_trial/result.toml"),
        r#"
            experiment = "real_open_field_trial"
            fps = 30.0
            meters_per_pixel = 0.001
            task_type = "open_field"

            [session_context]
            source_file = "examples/neo/imen/annotations/nort/metadata.csv"
            source_row = 42

            [detailed]
            center_time_pct = 10.0
            periphery_time_pct = 90.0
            total_distance_meters = 4.0
            total_duration_seconds = 120.0
            total_mean_speed = 0.03
            freezing_seconds = 0.0
            grooming_seconds = 0.0
            rearing_seconds = 0.0
            immobility_pct = 0.0
            "#,
    )
    .unwrap();

    let config = config_with_aggregate(AggregateConfig::default());
    crate::traced(|| run_aggregate(&config, &dir))
        .expect("missing optional fecal boli should not block aggregate");
    assert!(aggregate_output_path_for_config(&config, &dir).exists());

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn aggregate_keeps_stale_imen_oft_habituation_results() {
    let dir = std::env::temp_dir().join("bekiper_aggregate_keeps_imen_oft");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    write_open_field_result_toml(
        &dir.join("oft_videos_oft_pre_pre_video_1/result.toml"),
        "oft_videos_oft_pre_pre_video_1",
    );
    write_open_field_result_toml(
        &dir.join("real_open_field_trial/result.toml"),
        "real_open_field_trial",
    );

    let config = config_with_aggregate(AggregateConfig::default());
    run_aggregate(&config, &dir).unwrap();

    assert!(dir.join("results.ods").exists());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn aggregate_fails_when_all_discovered_trials_are_analysis_excluded() {
    let dir = std::env::temp_dir().join("bekiper_aggregate_all_skipped");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::create_dir_all(dir.join("stage0")).unwrap();
    std::fs::write(
        dir.join("stage0/result.toml"),
        r#"
            experiment = "nort_videos_nort_pre_t1_config_a_pre_t1_configa_video_22"
            fps = 30.0
            meters_per_pixel = 0.001
            task_type = "object_recognition"

            [session_context]
            profile = "videomap"
            stage = "0"
            phase = "A"
            "#,
    )
    .unwrap();

    let config = config_with_aggregate(AggregateConfig::default());
    let err = run_aggregate(&config, &dir).unwrap_err();
    let message = err.to_string();
    assert!(message.contains("skipped all 1 result.toml"), "{message}");
    assert!(!dir.join("results.ods").exists());

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn aggregate_rejects_stale_imen_nort_protocol_mismatch() {
    let dir = std::env::temp_dir().join("bekiper_aggregate_nort_protocol_mismatch");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    write_nort_result_toml(
        &dir.join("nort_videos_nort_pre_t2_config_b_pre_t2_configb_video_59/result.toml"),
        "nort_videos_nort_pre_t2_config_b_pre_t2_configb_video_59",
        "nort_training",
    );

    let config = config_with_aggregate(AggregateConfig::default());
    let err = run_aggregate(&config, &dir).unwrap_err();
    let message = err.to_string();
    assert!(
        message.contains("aggregate NORT consistency check failed"),
        "{message}"
    );
    assert!(message.contains("expected nort_novelty"), "{message}");
    assert!(!dir.join("results.ods").exists());

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn aggregate_rejects_stale_imen_nort_perimeter_label_mismatch() {
    let dir = std::env::temp_dir().join("bekiper_aggregate_nort_perimeter_mismatch");
    let _ = std::fs::remove_dir_all(&dir);
    let experiment = "nort_videos_nort_pre_t2_config_b_pre_t2_configb_video_59";
    write_nort_result_toml(
        &dir.join("data").join(experiment).join("result.toml"),
        experiment,
        "nort_novelty",
    );
    write_pixel_perimeters(
        &checkpoint_root(&dir)
            .join(experiment)
            .join("pixel_perimeters.toml"),
        &["familiar", "variable"],
    );

    let config = config_with_aggregate(AggregateConfig::default());
    let err = run_aggregate(&config, &dir).unwrap_err();
    let message = err.to_string();
    assert!(message.contains("perimeter labels"), "{message}");
    assert!(message.contains("familiar + novel"), "{message}");
    assert!(!dir.join("results.ods").exists());

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn aggregate_rejects_degenerate_ymaze_source_summaries() {
    let dir = std::env::temp_dir().join("bekiper_aggregate_ymaze_degenerate");
    let _ = std::fs::remove_dir_all(&dir);
    write_ymaze_result_toml(
        &dir.join("y_maze_videos_y_maze_pre_pre_video_14/result.toml"),
        "y_maze_videos_y_maze_pre_pre_video_14",
        [13, 0, 0],
        [20.0, 0.0, 0.0],
    );
    write_ymaze_result_toml(
        &dir.join("y_maze_videos_y_maze_pre_pre_video_15/result.toml"),
        "y_maze_videos_y_maze_pre_pre_video_15",
        [9, 0, 0],
        [15.0, 0.0, 0.0],
    );

    let config = config_with_aggregate(AggregateConfig::default());
    let err = run_aggregate(&config, &dir).unwrap_err();
    let message = err.to_string();
    assert!(
        message.contains("aggregate Y-maze source validation failed"),
        "{message}"
    );
    assert!(message.contains("zero arm(s): B, C"), "{message}");
    assert!(
        message.contains("y_maze_videos_y_maze_pre_pre_video_14"),
        "{message}"
    );
    assert!(!dir.join("results.ods").exists());

    std::fs::remove_dir_all(&dir).ok();
}

fn scratch_root(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("bekiper_aggregate_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn metadata_config(path: &str, policy: crate::config::MissingMetadataPolicy) -> IngressConfig {
    config_with_aggregate(AggregateConfig {
        metadata: Some(crate::config::AggregateMetadataConfig {
            path: path.into(),
            sheet: "Updated_Mice-overview".into(),
            missing_policy: policy,
        }),
        ..AggregateConfig::default()
    })
}

#[test]
fn aggregate_freshness_needs_a_readable_state_and_result_files() {
    let dir = scratch_root("freshness_state");
    write_result_toml(&dir.join("trial/result.toml"));
    let config = config_with_aggregate(AggregateConfig::default());
    run_aggregate(&config, &dir).unwrap();
    assert!(is_aggregate_fresh(&config, &dir));
    let state = aggregate_state_path_for_config(&config, &dir);
    std::fs::write(&state, "format_version = 'corrupt'").unwrap();
    assert!(!is_aggregate_fresh(&config, &dir), "corrupt state");
    std::fs::remove_file(&state).unwrap();
    assert!(!is_aggregate_fresh(&config, &dir), "missing state");
    run_aggregate(&config, &dir).unwrap();
    std::fs::remove_dir_all(dir.join("trial")).unwrap();
    assert!(!is_aggregate_fresh(&config, &dir), "no results to compare");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn aggregate_metadata_enrichment_follows_its_missing_policy() {
    use crate::config::MissingMetadataPolicy::{AllowBlanks, ExcludeMissing, Fail};
    let root = scratch_root("metadata_policy");
    let run_root = root.join("run");
    let trials = || {
        let mut trials = Vec::new();
        for experiment in [
            "oft_videos_oft_pre_pre_video_1",
            "oft_videos_oft_pre_pre_video_2",
        ] {
            write_open_field_result_toml(
                &run_root.join(experiment).join("result.toml"),
                experiment,
            );
        }
        collect_trial_results(&run_root, &mut trials, 0, 1);
        trials
    };
    // Relative metadata resolves beside the run root's data directory.
    crate::write_test_xlsx(
        &root.join("animals.xlsx"),
        &[&["Animal ID", "Sex"], &["1", "f"]],
    );
    let config = metadata_config("animals.xlsx", ExcludeMissing);
    assert_eq!(
        aggregate_metadata_base_dir(&config, &run_root),
        root.join(".")
    );
    assert!(aggregate_metadata_modified_unix_nanos(&config, &run_root).is_some());
    let absolute = metadata_config(root.join("animals.xlsx").to_str().unwrap(), ExcludeMissing);
    assert_eq!(
        aggregate_metadata_modified_unix_nanos(&absolute, &run_root),
        aggregate_metadata_modified_unix_nanos(&config, &run_root)
    );
    crate::traced(|| export_aggregate_trials(&config, &run_root, trials())).unwrap();
    assert!(is_aggregate_fresh(&config, &run_root));

    let mut kept = trials();
    let exclusions = apply_aggregate_metadata(&config, &run_root, &mut kept).unwrap();
    assert_eq!(kept.len(), 1);
    let sheet = format!("{:?}", metadata_exclusions_sheet(&exclusions));
    assert!(sheet.contains("metadata_exclusions") && sheet.contains("missing_animal_metadata"));

    let only_two = metadata_config("animals.xlsx", ExcludeMissing);
    let mut second = trials();
    second.remove(0);
    let error = export_aggregate_trials(&only_two, &run_root, second)
        .unwrap_err()
        .to_string();
    assert!(error.contains("excluded every result.toml row"), "{error}");

    let missing = metadata_config("absent.xlsx", AllowBlanks);
    let mut blank = trials();
    assert!(
        crate::traced(|| apply_aggregate_metadata(&missing, &run_root, &mut blank))
            .unwrap()
            .is_empty()
    );
    assert_eq!(blank.len(), 2);
    let required = metadata_config("absent.xlsx", Fail);
    assert!(
        apply_aggregate_metadata(&required, &run_root, &mut trials())
            .unwrap_err()
            .to_string()
            .contains("workbook is missing")
    );
    assert_eq!(
        aggregate_metadata_modified_unix_nanos(&required, &run_root),
        None
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn aggregate_export_honours_cancellation_disabled_output_and_empty_input() {
    let dir = scratch_root("export_flags");
    write_result_toml(&dir.join("trial/result.toml"));
    let mut trials = Vec::new();
    collect_trial_results(&dir, &mut trials, 0, 1);
    let config = config_with_aggregate(AggregateConfig::default());
    let cancelled = |result: Result<()>| result.unwrap_err().to_string().contains("cancelled");
    assert!(cancelled(export_aggregate_trials_with_cancel(
        &config,
        &dir,
        trials.clone(),
        &|| true
    )));
    let calls = std::sync::atomic::AtomicUsize::new(0);
    assert!(cancelled(export_aggregate_trials_with_cancel(
        &config,
        &dir,
        trials.clone(),
        &|| calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) > 0
    )));
    crate::traced(|| export_aggregate_trials(&config, &dir, Vec::new())).unwrap();
    let disabled = config_with_aggregate(AggregateConfig {
        enabled: false,
        ..AggregateConfig::default()
    });
    export_aggregate_trials(&disabled, &dir, trials.clone()).unwrap();
    assert!(!aggregate_output_path_for_config(&config, &dir).exists());

    // A post-analysis failure stops before the workbook is written.
    let mut post = config_with_aggregate(AggregateConfig::default());
    post.post_analysis.enabled = true;
    let file_root = dir.join("not-a-directory");
    std::fs::write(&file_root, "").unwrap();
    assert!(export_aggregate_trials(&post, &file_root, trials).is_err());
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn aggregate_skips_excluded_trials_and_adds_nort_diagnostics() {
    let dir = scratch_root("diagnostics");
    let attribution = r#"
        [[detailed.exploration_attributions]]
        frames_in_window = 3600
        frames_pose_ok = 3500
        frames_outside_object = 3000
        frames_nose_near = 200
        frames_nose_contact = 50
        frames_within_angle = 100
        frames_contact_or_angle = 120
        frames_slow_nose = 80
        frames_all_gates_raw = 20
        frames_after_tolerance = 0
        bouts_pre_filter = 4
        bouts_post_filter = 0
        bout_lengths_ms_pre = [100, 200]
        bout_lengths_ms_post = []
        candidate_bouts_pre_merge = 4
        bouts_post_merge = 2
        bouts_removed_by_duration_filter = 1
        bouts_removed_by_speed_filter = 1
        final_valid_bouts = 0
        bouts_removed_by_angle_filter = 0
        candidate_frames_removed_by_angle_criterion = 3
        median_nose_speed_per_bout = []
        median_head_angle_per_bout = []
    "#;
    let experiment = "nort_videos_nort_pre_t1_config_a_pre_t1_configa_video_7";
    let path = dir.join(experiment).join("result.toml");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(
        &path,
        format!(
            r#"
            experiment = "{experiment}"
            fps = 30.0
            meters_per_pixel = 0.001
            task_type = "object_recognition"

            [detailed]
            protocol = "nort_training"
            excluded = false
            total_duration_seconds = 120.0
            total_distance_meters = 5.0
            total_mean_speed = 0.04
            freezing_seconds = 1.2
            analysis_window_seconds = 120.0
            object_labels = ["familiar", "variable", "spare"]
            exploration_times = [1.0, 2.0, 3.0]
            bout_counts = [1, 1, 1]
            {}
            "#,
            attribution.repeat(3)
        ),
    )
    .unwrap();
    write_pixel_perimeters(
        &checkpoint_root(&dir)
            .join(experiment)
            .join("pixel_perimeters.toml"),
        &["familiar", "variable"],
    );
    std::fs::create_dir_all(dir.join("stage0")).unwrap();
    std::fs::write(
        dir.join("stage0/result.toml"),
        r#"
            experiment = "nort_videos_nort_pre_t1_config_a_pre_t1_configa_video_22"
            fps = 30.0
            meters_per_pixel = 0.001
            task_type = "object_recognition"

            [session_context]
            profile = "videomap"
            stage = "0"
            phase = "A"
            "#,
    )
    .unwrap();
    let config = config_with_aggregate(AggregateConfig::default());
    crate::traced(|| run_aggregate(&config, &dir)).unwrap();
    assert!(dir.join("results.ods").exists());
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn nort_consistency_reports_unreadable_or_mismatched_perimeters() {
    let dir = scratch_root("nort_perimeters");
    let training = "nort_videos_nort_pre_t1_config_a_pre_t1_configa_video_3";
    write_nort_result_toml(
        &dir.join(training).join("result.toml"),
        training,
        "nort_training",
    );
    let checkpoint = checkpoint_root(&dir)
        .join(training)
        .join("pixel_perimeters.toml");
    let mut trials = Vec::new();
    collect_trial_results(&dir, &mut trials, 0, 1);
    let error = |trials: &[TrialResult]| {
        validate_nort_aggregate_consistency(&dir, trials)
            .unwrap_err()
            .to_string()
    };
    write_pixel_perimeters(&checkpoint, &["familiar", "novel"]);
    assert!(error(&trials).contains("expected familiar + variable for nort_training"));
    std::fs::write(&checkpoint, "perimeters = 5").unwrap();
    assert!(error(&trials).contains("invalid"));
    std::fs::remove_file(&checkpoint).unwrap();
    std::fs::create_dir_all(&checkpoint).unwrap();
    assert!(error(&trials).contains("failed to read"));
    // Only object-recognition results are checked against NORT names.
    let mut renamed = trials[0].clone();
    renamed.task_type = TaskType::OpenField;
    validate_nort_aggregate_consistency(&dir, &[renamed]).unwrap();
    assert_eq!(
        expected_imen_nort_sheet_from_name("nort_videos_habituation"),
        None
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn ymaze_source_validation_reports_missing_zone_summaries() {
    let dir = scratch_root("ymaze_missing_zones");
    write_ymaze_result_toml(
        &dir.join("ymaze_a/result.toml"),
        "ymaze_a",
        [13, 0, 0],
        [20.0, 0.0, 0.0],
    );
    let mut trials = Vec::new();
    collect_trial_results(&dir, &mut trials, 0, 1);
    let mut bare = trials[0].clone();
    bare.experiment = "ymaze_bare".into();
    bare.detailed = serde_json::json!({});
    trials.push(bare);
    let error = validate_ymaze_aggregate_source_summaries(&dir, &trials)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("missing Y-maze zone_summaries in 1 result(s): ymaze_bare"),
        "{error}"
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn result_discovery_logs_unreadable_directories_and_results() {
    let dir = scratch_root("discovery_logs");
    std::fs::create_dir_all(dir.join("corrupt")).unwrap();
    std::fs::write(dir.join("corrupt/result.toml"), "not a result").unwrap();
    let mut trials = Vec::new();
    crate::traced(|| {
        collect_trial_results(&dir, &mut trials, 0, 1);
        collect_trial_results(&dir.join("missing"), &mut trials, 0, 1);
    });
    assert!(trials.is_empty());
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn nort_variant_trials_validate_against_their_recording_protocol_and_get_their_own_sheet() {
    use super::super::experiment::Experiment as _;
    let dir = scratch_root("nort_variant");
    let name = "nort_videos_nort_pre_t1_config_a_pre_t1_configa_video_3__whisker20mm";
    write_nort_result_toml(&dir.join(name).join("result.toml"), name, "nort_training");
    let mut trials = Vec::new();
    collect_trial_results(&dir, &mut trials, 0, 1);
    trials[0].analysis_role = Some("nort_training__whisker20mm".into());
    assert_eq!(
        expected_imen_nort_sheet_from_name(name),
        Some("nort_training")
    );
    // The suffix trips neither the protocol nor the sheet-key comparison...
    validate_nort_aggregate_consistency(&dir, &trials).unwrap();
    // ...and, like Y-maze variants, the variant is its own aggregate sheet.
    let key = experiment_for_trial(&trials[0])
        .sheet_key(&trials[0])
        .unwrap()
        .routing_key();
    assert_eq!(key, "nort_training__whisker20mm");
    std::fs::remove_dir_all(dir).unwrap();
}
