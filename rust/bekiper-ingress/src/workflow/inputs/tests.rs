use super::*;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("bekiper_inputs_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn experiment(body: &str) -> ExperimentConfig {
    toml::from_str(&format!(
        "name = 'trial'\ntask_type = 'open_field'\nmeters_per_pixel = 0.001\n{body}"
    ))
    .unwrap()
}

#[test]
fn pose_sources_report_precomputed_files_or_inference_inputs() {
    let dir = scratch("pose_sources");
    let present = dir.join("pose.parquet");
    std::fs::write(&present, "").unwrap();
    let resolver = PipelineInputResolver::new(&dir);
    let mut trial = experiment("");
    trial.data_files = vec![present.clone(), dir.join("absent.parquet")];
    let status = resolver.pose_source(&trial);
    assert_eq!(status.sources.len(), 1);
    assert!(status.missing[0].contains("missing pose data file"));
    assert!(!status.is_available());
    trial.data_files = vec![present.clone()];
    assert!(resolver.pose_source(&trial).is_available());

    let none = resolver.pose_source(&experiment(""));
    assert!(none.missing[0].contains("no precomputed pose data"));
    let model = |path: &Path| {
        format!(
            "[pose_model]\npath = '{}'\nspec = 'spec.json'\n",
            path.display()
        )
    };
    for (video, model_path, expected) in [
        (present.clone(), present.clone(), None),
        (
            dir.join("absent.mp4"),
            dir.join("absent.onnx"),
            Some("missing pose inference video"),
        ),
    ] {
        let trial = experiment(&format!(
            "video = '{}'\n{}",
            video.display(),
            model(&model_path)
        ));
        let status = resolver.pose_source(&trial);
        match expected {
            None => assert!(status.is_available(), "{status:?}"),
            Some(message) => {
                assert!(status.missing[0].contains(message));
                assert!(status.missing[1].contains("missing pose model"));
            }
        }
    }
    // Inference needs both a video and a model; each is reported when absent.
    let mut no_video = experiment(&model(&present));
    no_video.video = None;
    assert!(!no_video.needs_pose_inference());
    let mut no_model = experiment(&format!("video = '{}'", present.display()));
    no_model.pose_model = None;
    assert!(resolver.pose_source(&no_model).missing[0].contains("no precomputed pose data"));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn perimeter_sources_summarize_every_backend() {
    let dir = scratch("perimeter_sources");
    let resolver = PipelineInputResolver::new(&dir);
    let precomputed = dir.join("perimeters.toml");
    std::fs::write(
        &precomputed,
        "[[perimeters]]\nlabel = 'arena'\nshape = 'rectangle'\ncenter_x = 1.0\ncenter_y = 1.0\nwidth = 1.0\nheight = 1.0\n",
    )
    .unwrap();
    let video = dir.join("video.mp4");
    std::fs::write(&video, "").unwrap();
    let trial: ExperimentConfig = toml::from_str(&format!(
        "name = 'trial'\ntask_type = 'open_field'\nvideo = '{}'\nmeters_per_pixel = {{ method = 'arena_side', meters = 0.4 }}\n[detection]\nperimeters = '{}'\nbackend = ['classical_cv', 'arena_calibration_cv', 'manual_annotation']",
        video.display(),
        precomputed.display()
    ))
    .unwrap();
    let status = resolver.perimeter_source(&trial, &resolver.exp_dir(&trial));
    assert_eq!(status.artifact_state, InputArtifactState::Missing);
    let summary = status.source_summary();
    for expected in [
        "precomputed",
        "(1 perimeter(s))",
        "automatic backend classical_cv",
        "arena calibration cv source available",
        "manual_annotation backend requires annotations.dir",
    ] {
        assert!(summary.contains(expected), "{expected}: {summary}");
    }
    assert!(status.has_non_auto_source());
    assert!(status.first_manual_annotation().is_none());

    let empty = PerimeterInputStatus {
        artifact_state: InputArtifactState::Missing,
        artifact_path: dir.clone(),
        sources: Vec::new(),
    };
    assert_eq!(empty.source_summary(), "no perimeter source configured");
    let checkpoint = PerimeterInputStatus {
        sources: vec![PerimeterInputSource::Checkpoint { path: dir.clone() }],
        ..empty.clone()
    };
    assert!(checkpoint.source_summary().starts_with("checkpoint"));
    assert!(checkpoint.first_manual_annotation().is_none());
    let manual = PerimeterInputStatus {
        sources: vec![PerimeterInputSource::ManualAnnotation(AnnotationInput {
            specs: Vec::new(),
            labels: Vec::new(),
            detail: "row 2".into(),
            scale_plugin_dir: dir.clone(),
        })],
        ..empty
    };
    assert_eq!(manual.first_manual_annotation().unwrap().detail, "row 2");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn unusable_perimeter_sources_explain_why() {
    let dir = scratch("unusable_sources");
    let detection = |body: &str| -> DetectionConfig { toml::from_str(body).unwrap() };
    let reason = |source: PerimeterInputSource| match source {
        PerimeterInputSource::Missing { reason } => reason,
        other => panic!("expected a missing source, got {other:?}"),
    };
    assert!(
        reason(resolve_precomputed_source(&detection(""))).contains("requires a perimeters path")
    );
    assert!(
        reason(resolve_precomputed_source(&detection(
            "perimeters = '/absent.toml'"
        )))
        .contains("missing precomputed perimeters")
    );
    let corrupt = dir.join("corrupt.toml");
    std::fs::write(&corrupt, "perimeters = 5").unwrap();
    assert!(
        reason(resolve_precomputed_source(&detection(&format!(
            "perimeters = '{}'",
            corrupt.display()
        ))))
        .contains("invalid TOML")
    );

    let mut ymaze = experiment("");
    ymaze.task_type = bekiper_core::types::TaskType::YMaze;
    assert!(reason(resolve_arena_calibration_source(&ymaze)).contains("only valid for open_field"));
    assert!(
        reason(resolve_arena_calibration_source(&experiment(""))).contains("requires a video path")
    );
    let mut oft = experiment("video = '/absent.mp4'");
    assert!(
        reason(resolve_arena_calibration_source(&oft)).contains("missing arena calibration video")
    );
    let video = dir.join("video.mp4");
    std::fs::write(&video, "").unwrap();
    oft.video = Some(video.clone());
    assert!(reason(resolve_arena_calibration_source(&oft)).contains("requires meters_per_pixel"));

    let mut manual = experiment("");
    manual.annotation_dir = Some(dir.join("absent"));
    assert!(reason(resolve_manual_annotation_source(&manual)).contains("missing annotation dir"));
    manual.annotation_dir = Some(dir.clone());
    assert!(reason(resolve_manual_annotation_source(&manual)).contains("requires a video path"));
    manual.video = Some(video);
    assert!(
        reason(resolve_manual_annotation_source(&manual))
            .contains("manual annotation resolution failed")
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn manual_annotations_describe_metadata_rows_and_canonical_sessions() {
    let dir = scratch("annotation_detail");
    let legacy = dir.join("legacy");
    std::fs::create_dir_all(legacy.join("plugin_files")).unwrap();
    std::fs::write(
        legacy.join("plugin_files/perimeter-rectangle-training_1.csv"),
        "familiar,48,49,25,38,img.png,224,224\nvariable,152,150,28,27,img.png,224,224\n",
    )
    .unwrap();
    std::fs::write(
        legacy.join("metadata.csv"),
        "Phase,PhasePart,TrialNumber,Stage,Animal,Apparatus,Perimeter,ChangeReference,ChangeReferenceImageName\nA,1,47,1,1,1,training_1,,\n",
    )
    .unwrap();
    let video = dir.join("video.mp4");
    std::fs::write(&video, "").unwrap();
    let mut trial = experiment("");
    trial.video = Some(video.clone());
    trial.annotation_dir = Some(legacy);
    trial.session_context = Some(bekiper_core::inspection::IngressSessionContext {
        profile: "videomap".into(),
        animal: Some("1".into()),
        stage: Some("1".into()),
        phase_part: Some("1".into()),
        source_row: Some(2),
        ..Default::default()
    });
    let PerimeterInputSource::ManualAnnotation(annotation) =
        resolve_manual_annotation_source(&trial)
    else {
        panic!("legacy metadata resolves");
    };
    assert!(
        annotation.detail.starts_with("metadata.csv row 2"),
        "{}",
        annotation.detail
    );
    assert_eq!(annotation.specs.len(), 2);

    // A canonical store with a single session resolves as a fallback asset.
    let canonical = dir.join("canonical");
    let root = canonical.join("bekiper");
    std::fs::create_dir_all(root.join("sessions")).unwrap();
    std::fs::create_dir_all(root.join("perimeters")).unwrap();
    std::fs::write(root.join("manifest.toml"), "").unwrap();
    std::fs::write(
        root.join("sessions/animal_1_stage_1_part_1.toml"),
        "key = \"animal_1_stage_1_part_1\"\nvideo_number = 1\nanimal = 1\nstage = 1\nphase_part = 1\nperimeter_id = \"training_1\"\ncoordinate_space = \"target\"\n",
    )
    .unwrap();
    std::fs::write(
        root.join("perimeters/training_1.toml"),
        "[[perimeters]]\nlabel = \"familiar\"\nshape = \"polygon\"\nvertices = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]\n",
    )
    .unwrap();
    trial.annotation_dir = Some(canonical);
    let PerimeterInputSource::ManualAnnotation(annotation) =
        resolve_manual_annotation_source(&trial)
    else {
        panic!("canonical session resolves");
    };
    assert_eq!(annotation.detail, "fallback perimeter training_1");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn materialized_annotations_fall_back_to_the_configured_scale() {
    let dir = scratch("materialize_scale");
    let output = ExperimentOutputDir::new(&dir.join("run"), "trial");
    let annotation = AnnotationInput {
        specs: vec![PerimeterSpec::Rectangle {
            label: "arena".into(),
            center_x: 10.0,
            center_y: 10.0,
            width: 10.0,
            height: 10.0,
        }],
        labels: vec!["arena".into()],
        detail: "recorded".into(),
        scale_plugin_dir: dir.join("no_plugins"),
    };
    materialize_manual_annotation(&experiment(""), &annotation, &output).unwrap();
    assert!(output.pixel_perimeters_toml().is_file());
    assert!(
        !output.annotation_scale_toml().exists(),
        "no annotation calibration was found"
    );
    // A partial reviewed outline pair fails before anything is written.
    let mut nort = experiment(
        "[detection.arena_calibration]\ncenter_zone_rule = { method = 'frame_centered', arena_width_meters = 0.4, arena_height_meters = 0.4, center_width_meters = 0.2, center_height_meters = 0.2 }",
    );
    nort.task_type = bekiper_core::types::TaskType::ObjectRecognition;
    nort.video_resolution = Some((100, 100));
    let failed = ExperimentOutputDir::new(&dir.join("failed"), "trial");
    assert!(materialize_manual_annotation(&nort, &annotation, &failed).is_err());
    assert!(!failed.pixel_perimeters_toml().exists());
    std::fs::remove_dir_all(dir).unwrap();
}
