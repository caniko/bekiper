use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use polars::prelude::*;

use bekiper_core::inspection::{
    CoordinateColumnSpec, FeatureInspection, HeuristicMeta, InspectionManifest, InspectionSettings,
    PerimeterSpec, QuadrantSpec,
};
use bekiper_core::types::MetersPerPixel;
use bekiper_core::video::VideoMetadata;
use bekiper_ingress::inspection_video::{RenderOptions, render_from_directory};
use bekiper_pose::io::write_parquet;
use bekiper_pose::video::{VideoDecoder, probe_video_info};

fn unique_temp_dir(name: &str) -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "bekiper_inspection_video_{name}_{}_{}",
        std::process::id(),
        stamp
    ))
}

fn generate_test_video(path: &Path) {
    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=10:duration=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "10",
            path.to_str().unwrap(),
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .expect("ffmpeg must be on PATH to run inspection-video tests");

    assert!(status.success(), "ffmpeg failed to generate test video");
}

fn sample_manifest(video_path: &Path) -> InspectionManifest {
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
            fps: 10.0,
            total_frames: 10,
            resolution: (320, 240),
            meters_per_pixel: MetersPerPixel(0.001),
        },
        video_path: Some(video_path.to_path_buf()),
        session_context: None,
        frame_image: None,
        labels: vec!["nose".into(), "tail_base".into()],
        label_colors: HashMap::from([
            ("nose".into(), "cyan".into()),
            ("tail_base".into(), "yellow".into()),
        ]),
        coordinate_columns,
        perimeters: vec![PerimeterSpec::Circle {
            label: "arena".into(),
            center_x: 0.160,
            center_y: 0.120,
            radius: 0.080,
        }],
        heuristics: vec![HeuristicMeta {
            name: "zone_hit".into(),
            result_column: "zone_hit".into(),
            true_frames: 6,
            total_frames: 10,
            seconds: 0.6,
            truth_spans: vec![(0, 6)],
        }],
        settings: InspectionSettings {
            invert_y: false,
            minimum_seconds_tolerance: 0.5,
            maximum_seconds_distraction: 0.333,
            meters_per_pixel: 0.001,
        },
        scale_derivation: None,
        quadrants: vec![QuadrantSpec {
            label: "arena_grid".into(),
            x_min: 0.080,
            y_min: 0.060,
            x_max: 0.240,
            y_max: 0.180,
            rows: 2,
            cols: 2,
        }],
        feature_inspections: vec![
            FeatureInspection::Heuristic {
                label: "heuristic_zone_hit".into(),
                result_column: "zone_hit".into(),
                coordinate_labels: vec!["nose".into()],
                perimeter_label: Some("arena".into()),
                description: "Nose inside arena zone".into(),
                representative_frame_index: None,
                color: None,
                linewidth: None,
                title_override: None,
                story_key: None,
            },
            FeatureInspection::Trajectory {
                label: "trajectory_nose".into(),
                coordinate_labels: vec!["nose".into()],
                description: "Nose trajectory".into(),
                representative_frame_index: None,
                color: None,
                linewidth: None,
                title_override: None,
                story_key: None,
            },
            FeatureInspection::Angle {
                label: "angle_body_axis".into(),
                from_coordinate_label: "tail_base".into(),
                to_coordinate_label: "nose".into(),
                description: "Body axis".into(),
                representative_frame_index: None,
                color: None,
                linewidth: None,
                title_override: None,
                story_key: None,
            },
            FeatureInspection::RayCast {
                label: "raycast_nose".into(),
                origin_label: "nose".into(),
                angle_column: "heading_rad".into(),
                length_meters: 0.050,
                cone_width_deg: 40.0,
                description: "Nose sensing cone".into(),
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
        pixel_quadrants: vec![QuadrantSpec {
            label: "arena_grid".into(),
            x_min: 80.0,
            y_min: 60.0,
            x_max: 240.0,
            y_max: 180.0,
            rows: 2,
            cols: 2,
        }],
    }
}

fn sample_dataframe() -> DataFrame {
    df! {
        "nose_x" => &[0.120, 0.125, 0.130, 0.135, 0.140, 0.145, 0.150, 0.155, 0.160, 0.165],
        "nose_y" => &[0.100, 0.102, 0.104, 0.106, 0.108, 0.110, 0.112, 0.114, 0.116, 0.118],
        "tail_base_x" => &[0.100, 0.105, 0.110, 0.115, 0.120, 0.125, 0.130, 0.135, 0.140, 0.145],
        "tail_base_y" => &[0.090, 0.092, 0.094, 0.096, 0.098, 0.100, 0.102, 0.104, 0.106, 0.108],
        "zone_hit" => &[true, true, true, true, true, true, false, false, false, false],
        "heading_rad" => &[0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58],
    }
    .unwrap()
}

fn build_inspection_dir(name: &str) -> PathBuf {
    let dir = unique_temp_dir(name);
    std::fs::create_dir_all(&dir).unwrap();

    let video_path = dir.join("fixture.mp4");
    generate_test_video(&video_path);

    let manifest = sample_manifest(&video_path);
    manifest
        .write_toml(&dir.join("fixture_inspection.toml"))
        .unwrap();

    let mut df = sample_dataframe();
    write_parquet(&mut df, &dir.join("fixture_evaluated.parquet")).unwrap();

    dir
}

fn first_frame_rgb(path: &Path) -> Vec<u8> {
    let mut decoder = VideoDecoder::open(path).unwrap();
    let (_, image) = decoder.next_frame().unwrap().unwrap();
    image.to_rgb8().into_raw()
}

#[test]
fn overview_render_produces_playable_output() {
    let dir = build_inspection_dir("overview");
    let outputs = render_from_directory(&dir, &RenderOptions::default()).unwrap();

    assert_eq!(outputs.len(), 1);
    let output = &outputs[0];
    assert!(output.ends_with("overview.inspection.mp4"));
    assert!(output.exists());

    let info = probe_video_info(output).unwrap();
    assert_eq!(info.output_width, 320);
    assert_eq!(info.output_height, 240);
    assert!(
        info.total_frames >= 8,
        "expected ~10 frames, got {}",
        info.total_frames
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn feature_renders_differ_from_overview() {
    let dir = build_inspection_dir("feature_compare");

    let overview = render_from_directory(
        &dir,
        &RenderOptions {
            output: Some(dir.join("overview_custom.mp4")),
            ..RenderOptions::default()
        },
    )
    .unwrap()
    .pop()
    .unwrap();
    let heuristic = render_from_directory(
        &dir,
        &RenderOptions {
            heuristic: Some("zone_hit".into()),
            output: Some(dir.join("heuristic_custom.mp4")),
            ..RenderOptions::default()
        },
    )
    .unwrap()
    .pop()
    .unwrap();
    let trajectory = render_from_directory(
        &dir,
        &RenderOptions {
            feature: Some("trajectory_nose".into()),
            output: Some(dir.join("trajectory_custom.mp4")),
            ..RenderOptions::default()
        },
    )
    .unwrap()
    .pop()
    .unwrap();
    let angle = render_from_directory(
        &dir,
        &RenderOptions {
            feature: Some("angle_body_axis".into()),
            output: Some(dir.join("angle_custom.mp4")),
            ..RenderOptions::default()
        },
    )
    .unwrap()
    .pop()
    .unwrap();
    let raycast = render_from_directory(
        &dir,
        &RenderOptions {
            feature: Some("raycast_nose".into()),
            output: Some(dir.join("raycast_custom.mp4")),
            ..RenderOptions::default()
        },
    )
    .unwrap()
    .pop()
    .unwrap();

    let overview_rgb = first_frame_rgb(&overview);
    assert_ne!(overview_rgb, first_frame_rgb(&heuristic));
    assert_ne!(overview_rgb, first_frame_rgb(&trajectory));
    assert_ne!(overview_rgb, first_frame_rgb(&angle));
    assert_ne!(overview_rgb, first_frame_rgb(&raycast));

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn batch_mode_writes_expected_default_outputs() {
    let dir = build_inspection_dir("batch");
    let outputs = render_from_directory(
        &dir,
        &RenderOptions {
            all_features: true,
            ..RenderOptions::default()
        },
    )
    .unwrap();

    let expected = vec![
        dir.join("videos/overview.inspection.mp4"),
        dir.join("videos/feature_heuristic_zone_hit.inspection.mp4"),
        dir.join("videos/feature_trajectory_nose.inspection.mp4"),
        dir.join("videos/feature_angle_body_axis.inspection.mp4"),
        dir.join("videos/feature_raycast_nose.inspection.mp4"),
    ];

    assert_eq!(outputs.len(), expected.len());
    for path in expected {
        assert!(outputs.contains(&path), "missing {:?}", path);
        assert!(path.exists(), "render did not write {:?}", path);
    }

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn captured_render_receipts_detect_changes_and_retry_without_reanalysis() {
    use bekiper_ingress::{
        config::{ExperimentConfig, InspectConfig},
        inspection,
        stage::ExperimentOutputDir,
    };
    let root = unique_temp_dir("receipt");
    let output = ExperimentOutputDir::new(&root, "video_sample");
    output.ensure_dir().unwrap();
    let fixture = build_inspection_dir("receipt_inputs");
    for entry in std::fs::read_dir(&fixture).unwrap() {
        let path = entry.unwrap().path();
        let target = match path.file_name().unwrap().to_str().unwrap() {
            "fixture_inspection.toml" => output.inspection_toml(),
            "fixture_evaluated.parquet" => output.evaluated_parquet(),
            name => output.root().join(name),
        };
        std::fs::copy(&path, target).unwrap();
    }
    let experiment: ExperimentConfig = toml::from_str(
        "name='video_sample'\ntask_type='open_field'\nfps=10.0\nmeters_per_pixel=0.001",
    )
    .unwrap();
    let mut settings = InspectConfig {
        plot: toml::from_str("fig_format='.png'\nmin_dpi=48").unwrap(),
        ..Default::default()
    };
    settings
        .plot
        .insert("ffmpeg_preset".into(), "ultrafast".into());
    settings.video_trials = vec![experiment.name.clone()];
    inspection::capture_context(&output, &experiment, &settings).unwrap();
    assert!(!inspection::is_fresh(&output, &settings));
    inspection::render(&output, &experiment, &settings, &|| false).unwrap();
    assert!(inspection::is_fresh(&output, &settings));
    let native_before = std::fs::read(output.evaluated_parquet()).unwrap();
    let receipt: serde_json::Value = serde_json::from_slice(
        &std::fs::read(output.root().join("inspection_receipt.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(receipt["effective_theme"]["min_dpi"], 48);
    assert_eq!(
        receipt["effective_video_settings"]["ffmpeg_preset"],
        "ultrafast"
    );
    assert!(output.videos_dir().join("overview.mp4").is_file());
    assert_eq!(receipt["video_outputs"][0]["source_fps"], 10.0);
    assert_eq!(receipt["video_outputs"][0]["frames_written"], 10);
    let artifact = receipt["artifacts"][0]["path"].as_str().unwrap();
    std::fs::write(artifact, b"damaged").unwrap();
    assert!(!inspection::is_fresh(&output, &settings));
    inspection::render(&output, &experiment, &settings, &|| false).unwrap();
    assert!(inspection::is_fresh(&output, &settings));
    assert_eq!(
        native_before,
        std::fs::read(output.evaluated_parquet()).unwrap()
    );
    // A captured theme wins over the renderer subprocess's later environment.
    let status = Command::new("python")
        .args(["-m", "bekiper_cli", "inspect", "plot"])
        .arg(output.root())
        .arg("--only")
        .arg("summary.png")
        .arg("--output")
        .arg(output.root().join("replay"))
        .env("BEKIPER_PLOT_DPI", "72")
        .env("BEKIPER_PLOT_FORMAT", ".svgz")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .unwrap();
    assert!(status.success());
    let replay: serde_json::Value = serde_json::from_slice(
        &std::fs::read(output.root().join("replay/render_receipt.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(replay["effective_theme"]["min_dpi"], 48);
    assert_eq!(replay["effective_theme"]["fig_format"], ".png");
    assert_eq!(replay["artifacts"].as_array().unwrap().len(), 1);
    // Source-media bytes are rendering inputs too, even with fresh analysis.
    let video = fixture.join("fixture.mp4");
    let original = std::fs::read(&video).unwrap();
    std::fs::write(&video, b"changed source").unwrap();
    assert!(!inspection::is_fresh(&output, &settings));
    std::fs::write(&video, original).unwrap();
    assert!(inspection::is_fresh(&output, &settings));
    // A declared renderer error leaves a failed receipt rather than freshness.
    settings
        .plot
        .insert("fig_format".into(), ".unsupported".into());
    inspection::capture_context(&output, &experiment, &settings).unwrap();
    assert!(inspection::render(&output, &experiment, &settings, &|| false).is_err());
    assert!(!inspection::is_fresh(&output, &settings));
    settings.plot.insert("fig_format".into(), ".png".into());
    inspection::capture_context(&output, &experiment, &settings).unwrap();
    inspection::render(&output, &experiment, &settings, &|| false).unwrap();
    assert!(inspection::is_fresh(&output, &settings));
    assert_eq!(
        native_before,
        std::fs::read(output.evaluated_parquet()).unwrap()
    );
    std::fs::remove_dir_all(root).unwrap();
    std::fs::remove_dir_all(fixture).unwrap();
}

#[test]
fn cropped_inverted_video_keeps_the_full_source_span() {
    let dir = build_inspection_dir("cropped_inverted");
    let source = dir.join("fixture.mp4");
    let mut manifest = sample_manifest(&source);
    manifest.settings.invert_y = true;
    manifest.video.fps = 2.0; // Evaluated timing must not slow the source video.
    manifest
        .feature_inspections
        .retain(|f| matches!(f, FeatureInspection::Trajectory { .. }));
    // Analysis-space geometry and poses are reflected; source overlays invert
    // only renderer copies. No evaluated pose is available on source frame 0.
    manifest.perimeters = manifest
        .perimeters
        .iter()
        .map(|p| p.reflect_y(0.24))
        .collect();
    manifest.compute_pixel_geometry();
    manifest
        .write_toml(&dir.join("fixture_inspection.toml"))
        .unwrap();
    let mut cropped = df! {"frame" => [5_u64,6,8], "nose_x" => [0.16,0.17,0.18],
    "nose_y" => [Some(0.12),None,Some(0.13)], "tail_base_x" => [0.12,0.13,0.14],
    "tail_base_y" => [0.12,0.13,0.14], "heading_rad" => [0.,0.,0.], "zone_hit" => [true,false,true]}
    .unwrap();
    write_parquet(&mut cropped, &dir.join("fixture_evaluated.parquet")).unwrap();
    let output = render_from_directory(&dir, &RenderOptions::default())
        .unwrap()
        .pop()
        .unwrap();
    assert_eq!(
        probe_video_info(&output).unwrap().total_frames,
        probe_video_info(&source).unwrap().total_frames
    );
    assert!(
        (probe_video_info(&output).unwrap().fps - probe_video_info(&source).unwrap().fps).abs()
            < 1e-6
    );
    assert_eq!(cropped.height(), 3);
    let metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(output.with_extension("video.json")).unwrap())
            .unwrap();
    assert_eq!(metadata["source_fps"], 10.0);
    assert_eq!(metadata["encoded_fps"], 10.0);
    assert_eq!(metadata["frames_written"], 10);
    std::fs::remove_dir_all(dir).unwrap();
}

/// Top-level MP4 box types in file order.
fn mp4_boxes(path: &Path) -> Vec<String> {
    let bytes = std::fs::read(path).unwrap();
    let mut boxes = Vec::new();
    let mut offset = 0usize;
    while offset + 8 <= bytes.len() {
        let size = u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap()) as u64;
        boxes.push(String::from_utf8_lossy(&bytes[offset + 4..offset + 8]).into_owned());
        let size = match size {
            0 => break,
            1 => u64::from_be_bytes(bytes[offset + 8..offset + 16].try_into().unwrap()),
            size => size,
        };
        offset += size as usize;
    }
    boxes
}

fn nth_frame_rgb(path: &Path, n: usize) -> image::RgbImage {
    let mut decoder = VideoDecoder::open(path).unwrap();
    for _ in 0..n {
        decoder.next_frame().unwrap().unwrap();
    }
    decoder.next_frame().unwrap().unwrap().1.to_rgb8()
}

#[test]
fn scored_render_uses_source_override_strip_playhead_and_faststart() {
    let dir = build_inspection_dir("scored");
    let recorded = dir.join("fixture.mp4");
    let moved = dir.join("rebased.mp4");
    std::fs::rename(&recorded, &moved).unwrap();
    assert!(render_from_directory(&dir, &RenderOptions::default()).is_err());

    let output = render_from_directory(
        &dir,
        &RenderOptions {
            scored: true,
            source_video: Some(moved.clone()),
            ffmpeg_preset: Some("ultrafast".into()),
            ..RenderOptions::default()
        },
    )
    .unwrap()
    .pop()
    .unwrap();
    assert!(output.ends_with("videos/scored.inspection.mp4"));

    let boxes = mp4_boxes(&output);
    let position = |name: &str| boxes.iter().position(|b| b == name).unwrap();
    assert!(position("moov") < position("mdat"), "{boxes:?}");

    let receipt: serde_json::Value =
        serde_json::from_slice(&std::fs::read(output.with_extension("video.json")).unwrap())
            .unwrap();
    assert_eq!(receipt["source_video"], moved.to_str().unwrap());
    assert_eq!(receipt["recorded_source_video"], recorded.to_str().unwrap());
    assert_eq!(receipt["crf"], 23);
    assert_eq!(receipt["faststart"], true);
    assert_eq!(receipt["frame_size"], serde_json::json!([320, 240]));
    // One scored row: 2 px top + 8 px pitch.
    assert_eq!(receipt["strip_height_px"], 10);
    assert_eq!(receipt["resolution"], serde_json::json!([320, 250]));
    assert_eq!(
        receipt["scored_rows"],
        serde_json::json!([{"heuristic": "zone_hit", "result_column": "zone_hit", "perimeter": "arena"}])
    );
    let poster = image::open(output.with_extension("poster.png"))
        .unwrap()
        .to_rgb8();
    assert_eq!(poster.dimensions(), (320, 250));

    let info = probe_video_info(&output).unwrap();
    assert_eq!((info.output_width, info.output_height), (320, 250));
    // zone_hit is true on frames 0-5 and false on 6-9: 32 px per frame.
    let row_y = 240 + 2 + 3;
    let first = nth_frame_rgb(&output, 0);
    let last = nth_frame_rgb(&output, 9);
    let blue = |p: &image::Rgb<u8>| p[2] > p[0] + 60;
    let red = |p: &image::Rgb<u8>| p[0] > p[2] + 60;
    // Luma: the ink playhead is far brighter than either mask colour.
    let bright = |p: &image::Rgb<u8>| {
        0.299 * p[0] as f64 + 0.587 * p[1] as f64 + 0.114 * p[2] as f64 > 200.0
    };
    assert!(blue(first.get_pixel(100, row_y)));
    assert!(red(first.get_pixel(260, row_y)));
    assert!(bright(first.get_pixel(0, row_y)), "playhead at frame 0");
    assert!(!bright(first.get_pixel(288, row_y)));
    assert!(bright(last.get_pixel(288, row_y)), "playhead at frame 9");
    assert!(!bright(last.get_pixel(0, row_y)));

    std::fs::remove_dir_all(&dir).ok();
}
