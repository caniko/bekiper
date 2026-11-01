use super::*;

#[cfg(unix)]
#[test]
fn ready_rgb_pipes_transfer_full_frames_without_chunk_sleep_penalties() {
    let frames = 64;
    let frame = RgbImage::from_pixel(256, 256, image::Rgb([7, 8, 9]));
    let mut child = Command::new("cat")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .spawn()
        .unwrap();
    let stdin = child.stdin.take().unwrap();
    #[cfg(target_os = "linux")]
    rustix::pipe::fcntl_setpipe_size(&stdin, 4096).unwrap();
    let flags = rustix::fs::fcntl_getfl(&stdin).unwrap();
    rustix::fs::fcntl_setfl(&stdin, flags | rustix::fs::OFlags::NONBLOCK).unwrap();
    let started = std::time::Instant::now();
    let mut encoder = FfmpegEncoder {
        child,
        stdin: Some(stdin),
        output_path: PathBuf::new(),
        stderr_path: PathBuf::new(),
    };
    for _ in 0..frames {
        encoder
            .write_frame(&frame, &|| {
                started.elapsed() > std::time::Duration::from_secs(5)
            })
            .unwrap();
    }
    encoder.stdin.take();
    assert!(encoder.child.wait().unwrap().success());

    let mut child = Command::new("head")
        .args([
            "-c",
            &(frames * frame.as_raw().len()).to_string(),
            "/dev/zero",
        ])
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let stdout = child.stdout.take().unwrap();
    #[cfg(target_os = "linux")]
    rustix::pipe::fcntl_setpipe_size(&stdout, 4096).unwrap();
    let flags = rustix::fs::fcntl_getfl(&stdout).unwrap();
    rustix::fs::fcntl_setfl(&stdout, flags | rustix::fs::OFlags::NONBLOCK).unwrap();
    let mut decoder = FfmpegDecoder {
        child,
        stdout,
        width: 256,
        height: 256,
        fps: 30.,
        frame_count: None,
        stderr_path: PathBuf::new(),
    };
    let started = std::time::Instant::now();
    for _ in 0..frames {
        let image = decoder
            .next_frame(&|| started.elapsed() > std::time::Duration::from_secs(5))
            .unwrap()
            .unwrap();
        assert_eq!(image.dimensions(), (256, 256));
        assert!(image.as_raw().iter().all(|byte| *byte == 0));
    }
    assert!(decoder.next_frame(&|| false).unwrap().is_none());
}

#[cfg(unix)]
#[test]
fn stalled_encoder_write_is_cancellable_and_reaped() {
    let mut child = Command::new("sleep")
        .arg("60")
        .stdin(Stdio::piped())
        .spawn()
        .unwrap();
    let pid = child.id();
    let stdin = child.stdin.take().unwrap();
    let flags = rustix::fs::fcntl_getfl(&stdin).unwrap();
    rustix::fs::fcntl_setfl(&stdin, flags | rustix::fs::OFlags::NONBLOCK).unwrap();
    let started = std::time::Instant::now();
    let mut encoder = FfmpegEncoder {
        child,
        stdin: Some(stdin),
        output_path: PathBuf::new(),
        stderr_path: PathBuf::new(),
    };
    let result = encoder.write_frame(&RgbImage::new(1024, 1024), &|| {
        started.elapsed() >= std::time::Duration::from_millis(100)
    });
    assert!(result.unwrap_err().to_string().contains("cancelled"));
    drop(encoder);
    assert!(started.elapsed() < std::time::Duration::from_secs(2));
    #[cfg(target_os = "linux")]
    assert!(!Path::new(&format!("/proc/{pid}")).exists());
    #[cfg(not(target_os = "linux"))]
    let _ = pid;
}

use bekiper_core::inspection::{HeuristicMeta, InspectionSettings};
use bekiper_core::types::MetersPerPixel;
use bekiper_core::video::VideoMetadata;

fn sample_manifest() -> InspectionManifest {
    let mut coordinate_columns = HashMap::new();
    coordinate_columns.insert(
        "nose".into(),
        CoordinateColumnSpec {
            x: "nose_x".into(),
            y: "nose_y".into(),
        },
    );
    coordinate_columns.insert(
        "tail_base".into(),
        CoordinateColumnSpec {
            x: "tail_base_x".into(),
            y: "tail_base_y".into(),
        },
    );

    InspectionManifest {
        video: VideoMetadata {
            fps: 30.0,
            total_frames: 10,
            resolution: (320, 240),
            meters_per_pixel: MetersPerPixel(0.001),
        },
        video_path: Some(PathBuf::from("video.mp4")),
        session_context: None,
        frame_image: None,
        labels: vec!["nose".into(), "tail_base".into()],
        label_colors: HashMap::new(),
        coordinate_columns,
        perimeters: vec![PerimeterSpec::Circle {
            label: "arena".into(),
            center_x: 0.16,
            center_y: 0.12,
            radius: 0.08,
        }],
        heuristics: vec![
            HeuristicMeta {
                name: "in_arena".into(),
                result_column: "in_arena".into(),
                true_frames: 5,
                total_frames: 10,
                seconds: 0.2,
                truth_spans: vec![],
            },
            HeuristicMeta {
                name: "ambiguous".into(),
                result_column: "shared".into(),
                true_frames: 2,
                total_frames: 10,
                seconds: 0.1,
                truth_spans: vec![],
            },
        ],
        settings: InspectionSettings {
            invert_y: false,
            minimum_seconds_tolerance: 0.5,
            maximum_seconds_distraction: 0.333,
            meters_per_pixel: 0.001,
        },
        scale_derivation: None,
        quadrants: vec![],
        feature_inspections: vec![
            FeatureInspection::Heuristic {
                label: "heuristic_in_arena".into(),
                result_column: "in_arena".into(),
                coordinate_labels: vec!["nose".into()],
                perimeter_label: Some("arena".into()),
                description: "arena".into(),
                representative_frame_index: None,
                color: None,
                linewidth: None,
                title_override: None,
                story_key: None,
            },
            FeatureInspection::Zone {
                label: "zone_shared".into(),
                perimeter_label: "arena".into(),
                result_column: "shared".into(),
                coordinate_labels: vec!["nose".into()],
                description: "shared".into(),
                representative_frame_index: None,
                color: None,
                linewidth: None,
                title_override: None,
                story_key: None,
            },
            FeatureInspection::Proximity {
                label: "proximity_shared".into(),
                perimeter_label: "arena".into(),
                buffer_meters: 0.01,
                result_column: "shared".into(),
                coordinate_labels: vec!["nose".into()],
                description: "shared".into(),
                representative_frame_index: None,
                color: None,
                linewidth: None,
                title_override: None,
                story_key: None,
            },
            FeatureInspection::ScaleDerivation {
                label: "scale_derivation".into(),
                description: "skip".into(),
                representative_frame_index: None,
                color: None,
                linewidth: None,
                title_override: None,
                story_key: None,
            },
        ],
        pixel_perimeters: vec![PerimeterSpec::Circle {
            label: "arena".into(),
            center_x: 160.0,
            center_y: 120.0,
            radius: 80.0,
        }],
        pixel_quadrants: vec![],
    }
}

fn sample_bundle(dir: &Path) -> LoadedInspection {
    let df = df! {
        "nose_x" => &[0.10, 0.11],
        "nose_y" => &[0.12, 0.13],
        "tail_base_x" => &[0.09, 0.10],
        "tail_base_y" => &[0.11, 0.12],
        "in_arena" => &[true, false],
        "shared" => &[true, true],
    }
    .unwrap();

    LoadedInspection {
        dir: dir.to_path_buf(),
        manifest: sample_manifest(),
        df,
        video_path: dir.join("video.mp4"),
        recorded_video_path: None,
    }
}

#[test]
fn cropped_video_uses_source_frame_keys_and_does_not_draw_future_or_missing_points() {
    let mut bundle = sample_bundle(Path::new("."));
    bundle
        .df
        .with_column(Series::new("frame".into(), [76u64, 78]).into_column())
        .unwrap();
    let job = RenderJob {
        target: JobTarget::Feature(
            bundle
                .manifest
                .feature_inspections
                .iter()
                .find(|feature| feature.result_column() == Some("in_arena"))
                .unwrap()
                .clone(),
        ),
        output_path: "unused.mp4".into(),
    };
    let state = JobState::build(&bundle, &job, &FrameScale { sx: 1.0, sy: 1.0 }).unwrap();
    assert_eq!(state.coord_at("nose", 0), None);
    assert_eq!(state.bool_at(0), None);
    assert_eq!(state.coord_at("nose", 76), Some((100.0, 120.0)));
    assert_eq!(state.bool_at(76), Some(true));
    assert_eq!(state.coord_at("nose", 77), None);
    assert_eq!(state.bool_at(77), None);
    assert_eq!(state.coord_at("nose", 78), Some((110.0, 130.0)));
    assert_eq!(state.bool_at(78), Some(false));
}

#[test]
fn video_trajectory_preserves_frame_gaps_and_isolated_observations() {
    let mut bundle = sample_bundle(Path::new("."));
    bundle
        .df
        .with_column(Series::new("frame".into(), [76u64, 78]).into_column())
        .unwrap();
    let job = RenderJob {
        target: JobTarget::Feature(FeatureInspection::Trajectory {
            label: "trajectory_nose".into(),
            coordinate_labels: vec!["nose".into()],
            description: "Recorded trajectory".into(),
            representative_frame_index: None,
            color: None,
            linewidth: None,
            title_override: None,
            story_key: None,
        }),
        output_path: "unused.mp4".into(),
    };
    let state = JobState::build(&bundle, &job, &FrameScale { sx: 1.0, sy: 1.0 }).unwrap();
    assert_eq!(
        state.segments_until("nose", 0).unwrap(),
        Vec::<Vec<(f64, f64)>>::new()
    );
    assert_eq!(
        state.segments_until("nose", 77).unwrap(),
        vec![vec![(100.0, 120.0)]]
    );
    assert_eq!(
        state.segments_until("nose", 78).unwrap(),
        vec![vec![(100.0, 120.0)], vec![(110.0, 130.0)]]
    );
    let mut image = RgbImage::new(320, 240);
    render_frame(
        &mut image,
        78,
        &bundle,
        &VideoTheme::default(),
        &job,
        &state,
    );
    assert_ne!(*image.get_pixel(100, 120), image::Rgb([0, 0, 0]));
    assert_ne!(*image.get_pixel(110, 130), image::Rgb([0, 0, 0]));
    assert_eq!(*image.get_pixel(105, 125), image::Rgb([0, 0, 0]));
}

#[test]
fn per_object_heuristic_video_uses_parent_decision_and_preserves_native_bindings() {
    use bekiper_core::inspection::HeuristicDecompositionComponent;
    let mut bundle = sample_bundle(Path::new("."));
    bundle
        .df
        .with_column(Series::new("dx".into(), [1.0, 1.0]).into_column())
        .unwrap();
    bundle
        .df
        .with_column(Series::new("dy".into(), [0.0, 0.0]).into_column())
        .unwrap();
    bundle
        .df
        .with_column(Series::new("tx".into(), [0.15, 0.15]).into_column())
        .unwrap();
    bundle
        .df
        .with_column(Series::new("ty".into(), [0.12, 0.13]).into_column())
        .unwrap();
    bundle
        .manifest
        .heuristics
        .push(bekiper_core::inspection::HeuristicMeta {
            name: "olfaction_A".into(),
            result_column: "in_arena".into(),
            true_frames: 1,
            total_frames: 2,
            seconds: 0.1,
            truth_spans: vec![],
        });
    bundle
        .manifest
        .feature_inspections
        .push(FeatureInspection::HeuristicDecomposition {
            label: "heuristic_decomposition_olfaction_A".into(),
            heuristic_name: "olfaction_A".into(),
            result_column: "in_arena".into(),
            coordinate_labels: vec!["nose".into()],
            perimeter_label: None,
            components: vec![
                HeuristicDecompositionComponent {
                    label: "heading".into(),
                    result_column: "shared".into(),
                    coordinate_labels: vec!["nose".into()],
                    ray_origin_label: Some("nose".into()),
                    ray_direction_x_column: Some("dx".into()),
                    ray_direction_y_column: Some("dy".into()),
                    ray_target_x_column: Some("tx".into()),
                    ray_target_y_column: Some("ty".into()),
                    evaluated_perimeter: Some(PerimeterSpec::Circle {
                        label: "native_A".into(),
                        center_x: 0.15,
                        center_y: 0.12,
                        radius: 0.01,
                    }),
                    ..Default::default()
                },
                HeuristicDecompositionComponent {
                    label: "proximity".into(),
                    result_column: "shared".into(),
                    ..Default::default()
                },
            ],
            show_final_result: true,
            description: "native object A".into(),
            representative_frame_index: None,
            color: None,
            linewidth: None,
            title_override: None,
            story_key: None,
        });
    let jobs = build_jobs(
        &bundle,
        &RenderOptions {
            heuristic: Some("olfaction_A".into()),
            ..Default::default()
        },
    )
    .unwrap();
    let JobTarget::Feature(FeatureInspection::HeuristicDecomposition {
        components,
        show_final_result,
        ..
    }) = &jobs[0].target
    else {
        panic!("expected native ray feature");
    };
    assert_eq!(components.len(), 1);
    assert!(!show_final_result);
    assert_eq!(components[0].result_column, "in_arena");
    assert!(components[0].evaluated_perimeter.is_some());
    let state = JobState::build(&bundle, &jobs[0], &FrameScale { sx: 1.0, sy: 1.0 }).unwrap();
    assert_eq!(state.angles["tx"], vec![Some(0.15); 2]);
    assert_eq!(
        state.component_bools["in_arena"],
        vec![Some(true), Some(false)]
    );
    let mut second = RgbImage::new(320, 240);
    let theme = VideoTheme {
        video_true_color: "#00ff00".into(),
        video_false_color: "#ff0000".into(),
        ..Default::default()
    };
    render_frame(&mut second, 1, &bundle, &theme, &jobs[0], &state);
    assert!(second.get_pixel(114, 130)[0] > 0);
    assert_eq!(second.get_pixel(114, 130)[1], 0);
    assert!(second.get_pixel(150, 130)[1] > 0 && second.get_pixel(150, 130)[2] > 0);
}

#[test]
fn nonuniform_proximity_buffer_is_expanded_before_display_scaling() {
    let mut bundle = sample_bundle(Path::new("."));
    bundle.manifest.video.resolution = (400, 200);
    bundle.manifest.perimeters = vec![PerimeterSpec::Circle {
        label: "arena".into(),
        center_x: 0.1,
        center_y: 0.1,
        radius: 0.02,
    }];
    bundle.manifest.pixel_perimeters.clear();
    let job = RenderJob {
        target: JobTarget::Feature(FeatureInspection::Proximity {
            label: "buffer".into(),
            perimeter_label: "arena".into(),
            buffer_meters: 0.02,
            result_column: "in_arena".into(),
            coordinate_labels: vec!["nose".into()],
            description: "buffer".into(),
            representative_frame_index: None,
            color: None,
            linewidth: None,
            title_override: None,
            story_key: None,
        }),
        output_path: "unused.mp4".into(),
    };
    let state = JobState::build(&bundle, &job, &FrameScale { sx: 0.25, sy: 0.5 }).unwrap();
    let mut image = RgbImage::new(100, 100);
    let theme = VideoTheme {
        video_true_color: "#00ff00".into(),
        ..Default::default()
    };
    render_frame(&mut image, 0, &bundle, &theme, &job, &state);
    for (x, y) in [(15, 50), (35, 50), (25, 30), (25, 70)] {
        assert!(
            image.get_pixel(x, y)[1] > 0,
            "buffer boundary at ({x}, {y})"
        );
    }
}

#[test]
fn multiobject_family_video_uses_component_masks_and_direction_bindings() {
    let mut bundle = sample_bundle(Path::new("."));
    bundle
        .df
        .with_column(Series::new("dx".into(), [1.0, 1.0]).into_column())
        .unwrap();
    bundle
        .df
        .with_column(Series::new("dy".into(), [0.0, 0.0]).into_column())
        .unwrap();
    let feature = FeatureInspection::HeuristicDecomposition {
        label: "multiobject_olfaction".into(),
        heuristic_name: "multiobject_olfaction".into(),
        result_column: "in_arena".into(),
        coordinate_labels: vec!["nose".into()],
        perimeter_label: None,
        components: vec![bekiper_core::inspection::HeuristicDecompositionComponent {
            label: "object direction".into(),
            result_column: "in_arena".into(),
            coordinate_labels: vec!["nose".into()],
            ray_origin_label: Some("nose".into()),
            ray_direction_x_column: Some("dx".into()),
            ray_direction_y_column: Some("dy".into()),
            ..Default::default()
        }],
        show_final_result: false,
        description: "all objects".into(),
        representative_frame_index: None,
        color: None,
        linewidth: None,
        title_override: None,
        story_key: None,
    };
    bundle.manifest.feature_inspections.push(feature);
    let jobs = build_jobs(
        &bundle,
        &RenderOptions {
            heuristic: Some("olfaction".into()),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(jobs.len(), 1);
    let state = JobState::build(&bundle, &jobs[0], &FrameScale { sx: 1.0, sy: 1.0 }).unwrap();
    let mut first = RgbImage::new(320, 240);
    let mut second = RgbImage::new(320, 240);
    let theme = VideoTheme {
        video_true_color: "#00ff00".into(),
        video_false_color: "#ff0000".into(),
        ..VideoTheme::default()
    };
    render_frame(&mut first, 0, &bundle, &theme, &jobs[0], &state);
    render_frame(&mut second, 1, &bundle, &theme, &jobs[0], &state);
    assert!(first.get_pixel(104, 120)[1] > 0);
    assert_eq!(first.get_pixel(104, 120)[0], 0);
    assert!(second.get_pixel(114, 130)[0] > 0);
    assert_eq!(second.get_pixel(114, 130)[1], 0);

    // A missing decision must remain visibly unknown, distinct from red
    // rejection. Zero direction vectors must not invent a heading.
    bundle
        .df
        .with_column(Series::new("in_arena".into(), [None, Some(false)]).into_column())
        .unwrap();
    let state = JobState::build(&bundle, &jobs[0], &FrameScale { sx: 1.0, sy: 1.0 }).unwrap();
    let mut unknown = RgbImage::new(320, 240);
    render_frame(&mut unknown, 0, &bundle, &theme, &jobs[0], &state);
    let pixel = unknown.get_pixel(104, 120);
    assert!(pixel[0] > 0);
    assert_eq!(pixel[0], pixel[1]);
    assert_eq!(pixel[1], pixel[2]);
    bundle
        .df
        .with_column(Series::new("dx".into(), [0.0, 0.0]).into_column())
        .unwrap();
    let state = JobState::build(&bundle, &jobs[0], &FrameScale { sx: 1.0, sy: 1.0 }).unwrap();
    let mut missing_direction = RgbImage::new(320, 240);
    render_frame(&mut missing_direction, 0, &bundle, &theme, &jobs[0], &state);
    assert_eq!(missing_direction.get_pixel(104, 120).0, [0, 0, 0]);
}

#[test]
fn inverse_projection_draws_source_geometry_headings_and_bound_rays_at_source_keys() {
    let mut bundle = sample_bundle(Path::new("."));
    bundle.manifest.settings.invert_y = true;
    bundle.manifest.perimeters = vec![PerimeterSpec::Circle {
        label: "arena".into(),
        center_x: 0.16,
        center_y: 0.18,
        radius: 0.01,
    }];
    bundle.manifest.compute_pixel_geometry();
    let ray: FeatureInspection = serde_json::from_value(serde_json::json!({
        "kind":"ray_cast", "label":"heading", "origin_label":"nose", "angle_column":"heading",
        "length_meters":0.03,"cone_width_deg":20.,"description":"Native heading"
    }))
    .unwrap();
    let bound: FeatureInspection = serde_json::from_value(serde_json::json!({
        "kind":"heuristic_decomposition", "label":"bound", "heuristic_name":"olfaction",
        "result_column":"in_arena", "coordinate_labels":["nose"],"description":"Native ray",
        "show_final_result":false,"components":[{"label":"ray","result_column":"in_arena",
            "coordinate_labels":["nose"],"ray_origin_label":"nose","ray_direction_x_column":"dx",
            "ray_direction_y_column":"dy","ray_target_x_column":"tx","ray_target_y_column":"ty"}]
    }))
    .unwrap();
    bundle.manifest.feature_inspections = vec![ray.clone(), bound.clone()];
    bundle.df = df! {"frame"=>[76_u64],"nose_x"=>[0.10],"nose_y"=>[0.20],
    "tail_base_x"=>[0.09],"tail_base_y"=>[0.21],"heading"=>[-std::f64::consts::FRAC_PI_2],
    "dx"=>[0.0],"dy"=>[-0.01],"tx"=>[0.10],"ty"=>[0.17],"in_arena"=>[true]}
    .unwrap();
    let native = bundle.df.clone();
    bundle.df = crate::inspection::source_overlay_dataframe(&bundle.manifest, bundle.df).unwrap();
    bundle.manifest = bundle.manifest.source_overlay_geometry();
    let theme = VideoTheme {
        video_true_color: "#00ff00".into(),
        video_point_color: "#00ff00".into(),
        video_perimeter_color: "#00ff00".into(),
        ..Default::default()
    };
    for target in [
        JobTarget::Overview,
        JobTarget::Feature(ray),
        JobTarget::Feature(bound),
    ] {
        let job = RenderJob {
            target,
            output_path: "unused.mp4".into(),
        };
        let state = JobState::build(&bundle, &job, &FrameScale { sx: 1., sy: 1. }).unwrap();
        let mut image = RgbImage::new(320, 240);
        render_frame(&mut image, 76, &bundle, &theme, &job, &state);
        assert!(image.get_pixel(100, 40)[1] > 0, "source pose marker");
        assert_eq!(
            image.get_pixel(100, 200).0,
            [0, 0, 0],
            "analysis pose must not be drawn"
        );
        if matches!(job.target, JobTarget::Overview) {
            assert!(image.get_pixel(170, 60)[1] > 0, "source perimeter");
            assert_eq!(image.get_pixel(170, 180).0, [0, 0, 0]);
        } else {
            assert!(image.get_pixel(100, 47)[1] > 0, "source ray points down");
            if matches!(
                job.target,
                JobTarget::Feature(FeatureInspection::HeuristicDecomposition { .. })
            ) {
                assert!(
                    image.get_pixel(100, 70)[1] > 0,
                    "recorded source ray target"
                );
            }
            assert_eq!(
                image.get_pixel(100, 20).0,
                [0, 0, 0],
                "heading must be reflected"
            );
            let mut missing = RgbImage::new(320, 240);
            render_frame(&mut missing, 0, &bundle, &theme, &job, &state);
            assert!(
                missing.pixels().all(|p| p.0 == [0, 0, 0]),
                "no evaluated observation on source frame zero"
            );
        }
    }
    assert_eq!(
        native.column("nose_y").unwrap().f64().unwrap().get(0),
        Some(0.20)
    );
}

#[test]
fn theme_precedence_cli_over_env_over_file() {
    let dir = std::env::temp_dir().join("bekiper_theme_precedence");
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("bekiper_config.toml");
    fs::write(
        &path,
        r#"
[inspect.plot]
ffmpeg_preset = "file_preset"
video_true_color = "orange"
linewidth = 3.0
quiver_alpha = 0.2
"#,
    )
    .unwrap();

    let envs = HashMap::from([
        ("BEKIPER_PLOT_FFMPEG_PRESET", "env_preset".to_string()),
        ("BEKIPER_PLOT_VIDEO_TRUE_COLOR", "lime".to_string()),
        ("BEKIPER_PLOT_LINEWIDTH", "4.5".to_string()),
    ]);
    let theme = VideoTheme::load_with_env(Some(&path), Some("cli_preset"), |key| {
        envs.get(key).cloned()
    });
    assert_eq!(theme.ffmpeg_preset, "cli_preset");
    assert_eq!(theme.video_true_color, "lime");
    assert!((theme.linewidth - 4.5).abs() < f64::EPSILON);
    assert!((theme.quiver_alpha - 0.2).abs() < 1e-6);
}

#[test]
fn loader_finds_pair() {
    let dir = std::env::temp_dir().join("bekiper_find_inspection_pair");
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("a_inspection.toml"), "").unwrap();
    fs::write(dir.join("a_evaluated.parquet"), "").unwrap();
    let (toml, parquet) = find_inspection_pair(&dir).unwrap();
    assert!(toml.ends_with("a_inspection.toml"));
    assert!(parquet.ends_with("a_evaluated.parquet"));
}

#[test]
fn selector_defaults_to_overview() {
    let dir = std::env::temp_dir().join("bekiper_job_overview");
    let bundle = sample_bundle(&dir);
    let jobs = build_jobs(
        &bundle,
        &RenderOptions {
            codec: "h264".into(),
            ..RenderOptions::default()
        },
    )
    .unwrap();
    assert_eq!(jobs.len(), 1);
    assert!(matches!(jobs[0].target, JobTarget::Overview));
    assert!(jobs[0].output_path.ends_with("overview.inspection.mp4"));
}

#[test]
fn selector_feature_uses_feature_filename() {
    let dir = std::env::temp_dir().join("bekiper_job_feature");
    let bundle = sample_bundle(&dir);
    let jobs = build_jobs(
        &bundle,
        &RenderOptions {
            codec: "h264".into(),
            feature: Some("heuristic_in_arena".into()),
            ..RenderOptions::default()
        },
    )
    .unwrap();
    assert_eq!(jobs.len(), 1);
    assert!(
        jobs[0]
            .output_path
            .ends_with("feature_heuristic_in_arena.inspection.mp4")
    );
}

#[test]
fn selector_heuristic_ambiguity_lists_candidates() {
    let dir = std::env::temp_dir().join("bekiper_job_heuristic");
    let bundle = sample_bundle(&dir);
    let err = build_jobs(
        &bundle,
        &RenderOptions {
            codec: "h264".into(),
            heuristic: Some("ambiguous".into()),
            ..RenderOptions::default()
        },
    )
    .unwrap_err();
    let text = err.to_string();
    assert!(text.contains("zone_shared"));
    assert!(text.contains("proximity_shared"));
}

#[test]
fn batch_mode_writes_overview_and_renderable_features() {
    let dir = std::env::temp_dir().join("bekiper_job_batch");
    let bundle = sample_bundle(&dir);
    let jobs = build_jobs(
        &bundle,
        &RenderOptions {
            codec: "h264".into(),
            all_features: true,
            ..RenderOptions::default()
        },
    )
    .unwrap();
    assert_eq!(jobs.len(), 4);
    assert!(
        jobs.iter()
            .any(|job| matches!(job.target, JobTarget::Overview))
    );
    assert!(jobs.iter().any(|job| {
        matches!(
            &job.target,
            JobTarget::Feature(FeatureInspection::Heuristic { label, .. })
            if label == "heuristic_in_arena"
        )
    }));
    assert!(!jobs.iter().any(|job| {
        matches!(
            &job.target,
            JobTarget::Feature(FeatureInspection::ScaleDerivation { .. })
        )
    }));
}

#[test]
fn strip_x_lands_in_the_bar_that_holds_the_frame() {
    for (frames, bars) in [(10u64, 3u32), (10, 4), (4, 8), (7, 7), (1, 5)] {
        for frame in 0..frames {
            let x = u64::from(strip_x(frame, bars, frames));
            let start = x * frames / u64::from(bars);
            let end = ((x + 1) * frames / u64::from(bars)).max(start + 1);
            assert!(
                (start..end).contains(&frame),
                "F={frames} B={bars} f={frame} -> bar {x} spans {start}..{end}"
            );
            // With F <= B the playhead opens the run of bars showing the frame.
            if frames <= u64::from(bars) && x > 0 {
                assert!(((x - 1) * frames / u64::from(bars)) < frame);
            }
        }
    }
}

#[test]
fn validate_bounds_upscale_and_crf_and_env_crf_skips_non_crf_codecs() {
    let qualia = RenderOptions {
        qualia: true,
        codec: "h264".into(),
        ..RenderOptions::default()
    };
    for upscale in [0, 9] {
        let opts = RenderOptions {
            upscale: Some(upscale),
            ..qualia.clone()
        };
        assert!(
            validate_options(&opts)
                .unwrap_err()
                .to_string()
                .contains("--upscale")
        );
    }
    validate_options(&RenderOptions {
        upscale: Some(8),
        ..qualia
    })
    .unwrap();
    let crf = |codec: &str, crf| RenderOptions {
        codec: codec.into(),
        crf: Some(crf),
        ..RenderOptions::default()
    };
    assert!(validate_options(&crf("h264", 52)).is_err());
    validate_options(&crf("h264", 51)).unwrap();
    validate_options(&crf("libvpx-vp9", 63)).unwrap();
    assert!(validate_options(&crf("libvpx-vp9", 64)).is_err());

    let env = |key: &str| (key == "BEKIPER_VIDEO_CRF").then(|| "30".to_string());
    assert_eq!(crf_from_env("h264", env).unwrap(), Some(30));
    assert_eq!(crf_from_env("h264_nvenc", env).unwrap(), None);
    assert!(crf_from_env("h264", |_| Some("x".into())).is_err());
}

#[test]
fn validate_rejects_invalid_flag_combo() {
    let err = validate_options(&RenderOptions {
        output: Some(PathBuf::from("a.mp4")),
        output_dir: Some(PathBuf::from("videos")),
        codec: "h264".into(),
        ..RenderOptions::default()
    })
    .unwrap_err();
    assert!(err.to_string().contains("mutually exclusive"));
}

#[test]
fn scored_heuristics_mirror_the_composite_rule_and_drawn_perimeters() {
    let mut bundle = sample_bundle(Path::new("."));
    bundle.manifest.heuristics.push(HeuristicMeta {
        name: "olfaction_arena".into(),
        result_column: "shared".into(),
        true_frames: 0,
        total_frames: 2,
        seconds: 0.0,
        truth_spans: vec![],
    });
    bundle
        .manifest
        .quadrants
        .push(bekiper_core::inspection::QuadrantSpec {
            label: "grid".into(),
            x_min: 0.08,
            y_min: 0.06,
            x_max: 0.24,
            y_max: 0.18,
            rows: 2,
            cols: 2,
        });
    // Posture proxies count when only their primary decomposition is emitted;
    // a perimeter the overview does not draw stays unbound.
    bundle
        .manifest
        .feature_inspections
        .push(FeatureInspection::HeuristicDecomposition {
            label: "heuristic_decomposition_grooming".into(),
            heuristic_name: "grooming".into(),
            result_column: "shared".into(),
            coordinate_labels: vec![],
            perimeter_label: Some("undrawn".into()),
            components: vec![],
            show_final_result: true,
            description: "grooming".into(),
            representative_frame_index: None,
            color: None,
            linewidth: None,
            title_override: None,
            story_key: None,
        });
    assert_eq!(
        scored_heuristics(&bundle.manifest),
        vec![
            ("in_arena".into(), "in_arena".into(), Some("arena".into())),
            ("ambiguous".into(), "shared".into(), None),
            ("grooming".into(), "shared".into(), Some("undrawn".into())),
        ]
    );
    let job = RenderJob {
        target: JobTarget::Scored,
        output_path: "unused.mp4".into(),
    };
    let state = JobState::build(&bundle, &job, &FrameScale { sx: 1.0, sy: 1.0 }).unwrap();
    assert!(state.quadrants.is_empty() && state.perimeters.contains_key("arena"));
    let overview = RenderJob {
        target: JobTarget::Overview,
        output_path: "unused.mp4".into(),
    };
    let overview = JobState::build(&bundle, &overview, &FrameScale { sx: 1.0, sy: 1.0 }).unwrap();
    assert!(overview.quadrants.contains_key("grid") && overview.scored.is_empty());
    let rows: Vec<_> = state
        .scored
        .iter()
        .map(|row| (row.heuristic.as_str(), row.perimeter.as_deref()))
        .collect();
    assert_eq!(
        rows,
        [
            ("in_arena", Some("arena")),
            ("ambiguous", None),
            ("grooming", None)
        ]
    );
}

#[test]
fn scored_outline_follows_the_final_mask_and_the_strip_rasterises_it() {
    let bundle = sample_bundle(Path::new("."));
    let job = RenderJob {
        target: JobTarget::Scored,
        output_path: "unused.mp4".into(),
    };
    let state = JobState::build(&bundle, &job, &FrameScale { sx: 1.0, sy: 1.0 }).unwrap();
    let outline = |frame| {
        let mut image = RgbImage::new(320, 240);
        render_frame(
            &mut image,
            frame,
            &bundle,
            &VideoTheme::default(),
            &job,
            &state,
        );
        // Right-most point of the arena circle (centre 160,120, radius 80).
        *image.get_pixel(240, 120)
    };
    let close = |pixel: image::Rgb<u8>, token: &str| {
        let expected = parse_color(token);
        pixel
            .0
            .iter()
            .zip(expected.0)
            .all(|(a, b)| a.abs_diff(b) <= 16)
    };
    assert!(close(outline(0), themekit::DARK.true_mask));
    assert!(close(outline(1), themekit::DARK.false_mask));
    assert!(close(outline(5), themekit::DARK.unavailable));

    // Two rows over a four-frame recording, eight pixels wide.
    let strip = state.scored_strip(8, 18, 4);
    let row_y = |row: u32| SCORED_STRIP_TOP + row * SCORED_ROW_PITCH + 1;
    assert_eq!(*strip.get_pixel(0, row_y(0)), mask_color(Some(true)));
    assert_eq!(*strip.get_pixel(2, row_y(0)), mask_color(Some(false)));
    assert_eq!(*strip.get_pixel(5, row_y(0)), mask_color(None));
    assert_eq!(*strip.get_pixel(2, row_y(1)), mask_color(Some(true)));
    assert_eq!(
        *strip.get_pixel(0, 0),
        parse_color(themekit::DARK.background)
    );

    let frame = RgbImage::new(7, 4);
    let ink = parse_color(themekit::DARK.ink);
    let first = compose_scored(&frame, &strip, 0, 4, &state);
    let last = compose_scored(&frame, &strip, 3, 4, &state);
    assert_eq!(first.dimensions(), (8, 22));
    assert_eq!(*first.get_pixel(6, 0), image::Rgb([0, 0, 0]));
    assert_eq!(
        *first.get_pixel(7, 0),
        parse_color(themekit::DARK.background),
        "odd widths gain a background column"
    );
    assert_eq!(*first.get_pixel(1, 4 + row_y(0)), ink);
    assert_eq!(*first.get_pixel(2, 4 + row_y(0)), mask_color(Some(false)));
    assert_eq!(*last.get_pixel(7, 4 + row_y(0)), ink);
    assert_eq!(*last.get_pixel(0, 4 + row_y(0)), mask_color(Some(true)));
    assert_eq!(strip_x(99, 8, 4), 7, "late frames clamp to the last column");

    // An unscored perimeter is faint reference grey, never a mask colour.
    let mut bundle = bundle;
    bundle.manifest.heuristics.clear();
    let state = JobState::build(&bundle, &job, &FrameScale { sx: 1.0, sy: 1.0 }).unwrap();
    let mut image = RgbImage::new(320, 240);
    render_frame(&mut image, 0, &bundle, &VideoTheme::default(), &job, &state);
    let pixel = *image.get_pixel(240, 120);
    assert!(pixel.0.iter().all(|c| *c > 60) && pixel.0[2].abs_diff(pixel.0[0]) < 32);
}

#[test]
fn scored_selector_is_exclusive_and_uses_its_own_filename() {
    let bundle = sample_bundle(&std::env::temp_dir().join("bekiper_job_scored"));
    let jobs = build_jobs(
        &bundle,
        &RenderOptions {
            scored: true,
            ..RenderOptions::default()
        },
    )
    .unwrap();
    assert!(matches!(jobs[0].target, JobTarget::Scored));
    assert!(jobs[0].output_path.ends_with("scored.inspection.mp4"));
    assert_eq!(jobs[0].display_name(), "scored");
    let error = validate_options(&RenderOptions {
        scored: true,
        feature: Some("heuristic_in_arena".into()),
        ..RenderOptions::default()
    })
    .unwrap_err();
    assert!(error.to_string().contains("mutually exclusive"));
    let error = validate_options(&RenderOptions {
        codec: "mpeg4".into(),
        crf: Some(20),
        ..RenderOptions::default()
    })
    .unwrap_err();
    assert!(error.to_string().contains("--crf"));
    assert!(faststart_container(Path::new("a.MP4")));
    assert!(!faststart_container(Path::new("a.mkv")));
}

/// Two objects over four frames; `novel` lacks every channel but the nose rule
/// and the whisker fan. The animal sits at (100, 120) px facing down-right.
fn qualia_bundle(dir: &Path) -> LoadedInspection {
    let (near, gate, hit, side) = (
        |o: &str| format!("__bk_diag__exploring_{o}__nose_near"),
        |o: &str| format!("__bk_diag__exploring_{o}__contact_or_angle"),
        |o: &str| format!("__bk_diag__exploring_{o}__whisker_hit"),
        |o: &str| format!("__bk_diag__exploring_{o}__whisker_side"),
    );
    let df = df! {
        "nose_x" => &[0.10; 4], "nose_y" => &[0.12; 4],
        "tail_base_x" => &[0.09; 4], "tail_base_y" => &[0.11; 4],
        near("familiar") => &[true, false, false, true],
        gate("familiar") => &[true, false, false, false],
        hit("familiar") => &[false, true, false, false],
        side("familiar") => &[None, Some("left"), None, None],
        near("novel") => &[false; 4],
        gate("novel") => &[false; 4],
        hit("novel") => &[false; 4],
        "olfaction_familiar" => &[false, false, true, false],
        "body_proximity_familiar" => &[false, false, false, true],
        "__bk_diag__exploring_familiar__raw_combined" => &[true, true, false, false],
        "__bk_diag__exploring_familiar__final_toleranced" => &[true, true, true, false],
    }
    .unwrap();
    let mut bundle = LoadedInspection {
        dir: dir.to_path_buf(),
        manifest: sample_manifest(),
        df,
        video_path: dir.join("video.mp4"),
        recorded_video_path: None,
    };
    for (label, x) in [("familiar", 0.25), ("novel", 0.05)] {
        bundle.manifest.perimeters.push(PerimeterSpec::Rectangle {
            label: label.into(),
            center_x: x,
            center_y: 0.2,
            width: 0.04,
            height: 0.04,
        });
    }
    bundle
        .manifest
        .feature_inspections
        .push(FeatureInspection::DetectionFields {
            label: "detection_fields".into(),
            nose_label: "nose".into(),
            head_label: "tail_base".into(),
            cone_half_angle_deg: 45.0,
            nose_reach_m: 0.03,
            contact_distance_m: 0.005,
            whisker_apex_fraction: Some(0.9),
            whisker_half_angle_deg: Some(30.0),
            whisker_length_m: Some(0.02),
            object_labels: vec!["familiar".into(), "novel".into()],
            nose_near_columns: vec![near("familiar"), near("novel")],
            nose_gate_columns: vec![gate("familiar"), gate("novel")],
            whisker_hit_columns: vec![hit("familiar"), hit("novel")],
            description: "fields".into(),
            representative_frame_index: None,
            color: None,
            linewidth: None,
            title_override: None,
            story_key: None,
        });
    bundle
}

fn qualia_job(bundle: &LoadedInspection) -> RenderJob {
    build_jobs(
        bundle,
        &RenderOptions {
            qualia: true,
            ..RenderOptions::default()
        },
    )
    .unwrap()
    .remove(0)
}

#[test]
fn qualia_job_is_named_exclusive_and_needs_detection_fields() {
    let bundle = qualia_bundle(&std::env::temp_dir().join("bekiper_job_qualia"));
    let job = qualia_job(&bundle);
    assert!(matches!(job.target, JobTarget::Qualia(_)));
    assert!(job.output_path.ends_with("qualia.inspection.mp4"));
    assert_eq!(job.display_name(), "qualia");
    // Receipt and poster derive from the video name.
    assert!(
        job.output_path
            .with_extension("video.json")
            .ends_with("qualia.inspection.video.json")
    );
    assert!(
        job.output_path
            .with_extension("poster.png")
            .ends_with("qualia.inspection.poster.png")
    );
    let both = RenderOptions {
        qualia: true,
        scored: true,
        ..RenderOptions::default()
    };
    assert!(
        validate_options(&both)
            .unwrap_err()
            .to_string()
            .contains("mutually exclusive")
    );
    let plain = sample_bundle(Path::new("."));
    let error = build_jobs(
        &plain,
        &RenderOptions {
            qualia: true,
            ..RenderOptions::default()
        },
    )
    .unwrap_err();
    assert!(error.to_string().contains("detection_fields"));
}

#[test]
fn qualia_rows_group_by_object_and_colour_the_true_cell() {
    let bundle = qualia_bundle(Path::new("."));
    let job = qualia_job(&bundle);
    let state = JobState::build(&bundle, &job, &FrameScale { sx: 1.0, sy: 1.0 }).unwrap();
    let labels: Vec<_> = state.scored.iter().map(|row| row.label.as_str()).collect();
    assert_eq!(
        labels,
        [
            "A NOSE", "A WHISK", "A OLF", "A BODY", "A EXPL", "A BOUT", "B NOSE", "B WHISK"
        ],
        "absent channels have no row"
    );
    assert_eq!(state.scored[0].heuristic, "familiar:nose_rule");
    // The nose rule needs near AND gate: only frame 0 passes for object A.
    assert_eq!(
        state.scored[0].mask,
        [Some(true), Some(false), Some(false), Some(false)]
    );

    // One pixel column per frame right of the label gutter.
    let layout = qualia_layout(1);
    let height = layout.top + 8 * layout.pitch + layout.footer;
    let strip = state.scored_strip(layout.gutter + 4, height, 4);
    let cell = |row: u32, frame: u32| {
        *strip.get_pixel(
            layout.gutter + frame,
            layout.top + row * layout.pitch + layout.row_height / 2,
        )
    };
    // Familiar is blue, novel red; false cells are the dim palette border.
    let (blue, red, dim) = (
        parse_color(themekit::DARK.true_mask),
        parse_color(themekit::DARK.false_mask),
        parse_color(themekit::DARK.border),
    );
    assert_eq!(cell(0, 0), blue);
    assert_eq!(cell(0, 1), dim);
    assert_eq!(cell(1, 1), blue, "A whisker fan, frame 1");
    assert_eq!(cell(2, 2), blue, "A olfaction, frame 2");
    assert_eq!(cell(5, 2), blue, "A accepted bout, frame 2");
    assert_eq!(cell(7, 1), dim, "B rows follow A rows");
    assert_eq!(object_color("novel", 1), red);
    let ink = parse_color(themekit::DARK.ink);
    assert!(
        (0..layout.gutter)
            .any(|x| (0..layout.row_height).any(|y| *strip.get_pixel(x, layout.top + y) == ink)),
        "row label drawn in the gutter"
    );
}

#[test]
fn detection_fields_overlay_draws_cone_whisker_fans_and_hit_objects() {
    let bundle = qualia_bundle(Path::new("."));
    let job = qualia_job(&bundle);
    let state = JobState::build(&bundle, &job, &FrameScale { sx: 1.0, sy: 1.0 }).unwrap();
    let draw = |frame| {
        let mut image = RgbImage::new(320, 240);
        render_frame(
            &mut image,
            frame,
            &bundle,
            &VideoTheme::default(),
            &job,
            &state,
        );
        image
    };
    let black = image::Rgb([0, 0, 0]);
    let (nose_hit, whisker_hit) = (draw(0), draw(1));
    // Cone: nose (100,120) -> axis (1,1); 10 px along it is inside, 10 px behind the head is not.
    assert_ne!(*nose_hit.get_pixel(110, 130), black);
    assert_eq!(*nose_hit.get_pixel(80, 100), black);
    // Lateral fans at apex (99,119): left fan towards (-1,1), right towards (1,-1).
    assert_ne!(*nose_hit.get_pixel(92, 126), black, "idle left fan");
    assert_ne!(*nose_hit.get_pixel(106, 112), black, "idle right fan");
    let brightness = |pixel: &image::Rgb<u8>| pixel.0.iter().map(|c| u32::from(*c)).sum::<u32>();
    assert!(
        brightness(whisker_hit.get_pixel(92, 126)) > brightness(nose_hit.get_pixel(92, 126)),
        "the hit side fills strongly"
    );
    assert_eq!(
        whisker_hit.get_pixel(106, 112),
        nose_hit.get_pixel(106, 112),
        "the other side stays idle"
    );
    // Object A (250,200): filled on nose-rule and whisker-hit frames, outline only otherwise.
    assert_ne!(*whisker_hit.get_pixel(250, 200), black);
    assert_eq!(*draw(2).get_pixel(250, 200), black);
    assert_ne!(
        *nose_hit.get_pixel(250, 200),
        black,
        "and on its nose-rule frame"
    );
}

#[test]
fn qualia_axis_ticks_every_minute_and_layout_scales_with_the_upscale() {
    let bundle = qualia_bundle(Path::new("."));
    let job = qualia_job(&bundle);
    let mut state = JobState::build(&bundle, &job, &FrameScale { sx: 1.0, sy: 1.0 }).unwrap();
    // Two frames per minute: ticks at frames 2 (60 s) of a four-frame recording.
    state.fps = 2.0 / 60.0;
    let layout = state.layout;
    let strip = state.scored_strip(layout.gutter + 4, 400, 4);
    let tick_y = layout.top + 8 * layout.pitch + layout.text;
    let muted = parse_color(themekit::DARK.muted);
    assert_eq!(*strip.get_pixel(layout.gutter + 2, tick_y), muted);
    assert_ne!(
        *strip.get_pixel(layout.gutter + 1, tick_y),
        muted,
        "one tick only"
    );
    let big = qualia_layout(3);
    assert_eq!((big.text, big.row_height, big.pitch), (3, 15, 18));
    assert!(
        qualia_layout(1).row_height >= 10,
        "rows stay readable at unit scale"
    );
    assert_eq!(QUALIA_MIN_WIDTH.div_ceil(224), 3);
}
