#[test]
fn run_augment_user_midpoints_can_depend_on_model_midpoints() {
    let exp_dir = prepare_output("user_midpoint_depends_on_model_midpoint");
    write_pixel_perimeters(&exp_dir, vec![]);

    let frames: Vec<i64> = (0..3).collect();
    let mut df = imen_pose_data(&frames, vec![10.0; frames.len()]);
    write_parquet(&mut df, &exp_dir.poses_parquet()).unwrap();

    let pipeline = PipelineConfig {
        pose_label_model: PoseLabelModel::ImenDlc,
        midpoint_groups: vec![
            bekiper_core::types::MidpointGroup {
                output_label: "center_ear".into(),
                source_labels: vec!["left_ear".into(), "right_ear".into()],
            },
            bekiper_core::types::MidpointGroup {
                output_label: "head_tail_mid".into(),
                source_labels: vec!["center_ear".into(), "tail".into()],
            },
        ],
        ..PipelineConfig::default()
    };

    run_augment(&experiment("exp"), &test_config(pipeline), &exp_dir).unwrap();

    let augmented = read_augmented(&exp_dir);

    assert!(augmented.column("center_ear_x").is_ok());
    assert!(augmented.column("head_tail_mid_x").is_ok());
}

#[test]
fn run_augment_rejects_conflicting_user_midpoint_for_model_required_output() {
    let exp_dir = prepare_output("conflicting_model_midpoint");
    write_pixel_perimeters(&exp_dir, vec![]);

    let frames: Vec<i64> = (0..3).collect();
    let mut df = imen_pose_data(&frames, vec![10.0; frames.len()]);
    write_parquet(&mut df, &exp_dir.poses_parquet()).unwrap();

    let pipeline = PipelineConfig {
        pose_label_model: PoseLabelModel::ImenDlc,
        midpoint_groups: vec![bekiper_core::types::MidpointGroup {
            output_label: "center_ear".into(),
            source_labels: vec!["nose".into(), "tail".into()],
        }],
        ..PipelineConfig::default()
    };

    let err = run_augment(&experiment("exp"), &test_config(pipeline), &exp_dir)
        .unwrap_err()
        .to_string();
    assert!(err.contains("conflicts with model-required midpoint group"));
}

#[test]
fn run_augment_rejects_missing_source_for_model_required_midpoint() {
    let exp_dir = prepare_output("missing_model_midpoint_source");
    write_pixel_perimeters(&exp_dir, vec![]);

    let frames: Vec<i64> = (0..3).collect();
    let mut df = imen_pose_data(&frames, vec![10.0; frames.len()]);
    for column in ["right_ear_x", "right_ear_y", "right_ear_likelihood"] {
        df.drop_in_place(column).unwrap();
    }
    write_parquet(&mut df, &exp_dir.poses_parquet()).unwrap();

    let pipeline = PipelineConfig {
        pose_label_model: PoseLabelModel::ImenDlc,
        ..PipelineConfig::default()
    };

    let err = run_augment(&experiment("exp"), &test_config(pipeline), &exp_dir)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains(
            "model-required midpoint group 'center_ear' missing source label(s): right_ear"
        )
    );
}

#[test]
fn resolve_midpoint_groups_validates_shapes_and_deduplicates() {
    let group = |output: &str, sources: &[&str]| bekiper_core::types::MidpointGroup {
        output_label: output.into(),
        source_labels: sources.iter().map(|s| (*s).into()).collect(),
    };
    let labels: Vec<String> = ["left_ear", "right_ear", "tail"].map(String::from).to_vec();
    let ears = group("center_ear", &["left_ear", "right_ear"]);
    let error = |model: &[bekiper_core::types::MidpointGroup],
                 user: &[bekiper_core::types::MidpointGroup]| {
        resolve_midpoint_groups(&labels, model, user)
            .unwrap_err()
            .to_string()
    };
    for (bad, message) in [
        (group("", &["tail"]), "output_label cannot be empty"),
        (group("mid", &[]), "at least one source label"),
        (group("mid", &["tail", ""]), "empty source label"),
        (
            group("mid", &["tail", "tail"]),
            "duplicate source label 'tail'",
        ),
    ] {
        assert!(
            error(std::slice::from_ref(&bad), &[]).contains(message),
            "model {message}"
        );
        assert!(error(&[], &[bad]).contains(message), "user {message}");
    }
    assert!(error(&[group("tail", &["left_ear"])], &[]).contains("duplicates a raw pose label"));
    assert!(error(&[ears.clone(), ears.clone()], &[]).contains("defined more than once"));
    assert!(error(&[], &[group("tail", &["left_ear"])]).contains("duplicates a raw pose label"));
    assert!(
        error(
            &[],
            &[ears.clone(), group("center_ear", &["tail", "left_ear"])]
        )
        .contains("defined more than once with different sources")
    );
    // Identical user duplicates collapse; groups with unavailable sources are skipped.
    let merged = resolve_midpoint_groups(
        &labels,
        &[],
        &[
            ears.clone(),
            ears.clone(),
            group("nose_tail", &["nose", "tail"]),
        ],
    )
    .unwrap();
    assert_eq!(merged.len(), 1);
    assert!(midpoint_groups_equal(&merged[0], &ears));
}
