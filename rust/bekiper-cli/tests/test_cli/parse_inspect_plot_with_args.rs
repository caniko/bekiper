use super::*;
use clap::Parser;

#[test]
fn parse_inspect_plot_with_args() {
    let cli = TestCli::try_parse_from([
        "bekiper", "inspect", "plot", "/my/dir", "-o", "/out", "-f", ".png",
    ])
    .unwrap();
    let TestCli::Inspect(args) = cli else {
        panic!("expected Inspect");
    };
    let TestInspectCommand::Plot {
        dir,
        output,
        format,
    } = args.command
    else {
        panic!("expected Plot");
    };
    assert_eq!(dir.to_str().unwrap(), "/my/dir");
    assert_eq!(output.unwrap().to_str().unwrap(), "/out");
    assert_eq!(format, ".png");
}

#[test]
fn parse_inspect_video_defaults() {
    let cli = TestCli::try_parse_from(["bekiper", "inspect", "video"]).unwrap();
    let TestCli::Inspect(args) = cli else {
        panic!("expected Inspect");
    };
    let TestInspectCommand::Video {
        dir,
        output,
        output_dir,
        codec,
        heuristic,
        feature,
        all_features,
        config,
        ffmpeg_preset,
    } = args.command
    else {
        panic!("expected Video");
    };
    assert_eq!(dir.to_str().unwrap(), "inspection");
    assert!(output.is_none());
    assert!(output_dir.is_none());
    assert_eq!(codec, "h264");
    assert!(heuristic.is_none());
    assert!(feature.is_none());
    assert!(!all_features);
    assert!(config.is_none());
    assert!(ffmpeg_preset.is_none());
}

#[test]
fn parse_inspect_video_with_heuristic() {
    let cli = TestCli::try_parse_from([
        "bekiper",
        "inspect",
        "video",
        "/my/dir",
        "--codec",
        "mpeg4",
        "--heuristic",
        "body_proximity",
    ])
    .unwrap();
    let TestCli::Inspect(args) = cli else {
        panic!("expected Inspect");
    };
    let TestInspectCommand::Video {
        codec,
        heuristic,
        feature,
        all_features,
        ..
    } = args.command
    else {
        panic!("expected Video");
    };
    assert_eq!(codec, "mpeg4");
    assert_eq!(heuristic.unwrap(), "body_proximity");
    assert!(feature.is_none());
    assert!(!all_features);
}

#[test]
fn parse_inspect_video_with_feature() {
    let cli = TestCli::try_parse_from([
        "bekiper",
        "inspect",
        "video",
        "/my/dir",
        "--feature",
        "trajectory_nose",
        "--output",
        "/tmp/trajectory.mp4",
        "--ffmpeg-preset",
        "fast",
    ])
    .unwrap();
    let TestCli::Inspect(args) = cli else {
        panic!("expected Inspect");
    };
    let TestInspectCommand::Video {
        dir,
        output,
        feature,
        ffmpeg_preset,
        heuristic,
        ..
    } = args.command
    else {
        panic!("expected Video");
    };
    assert_eq!(dir.to_str().unwrap(), "/my/dir");
    assert_eq!(output.unwrap().to_str().unwrap(), "/tmp/trajectory.mp4");
    assert_eq!(feature.as_deref(), Some("trajectory_nose"));
    assert_eq!(ffmpeg_preset.as_deref(), Some("fast"));
    assert!(heuristic.is_none());
}

#[test]
fn parse_inspect_video_batch_mode() {
    let cli = TestCli::try_parse_from([
        "bekiper",
        "inspect",
        "video",
        "/my/dir",
        "--all-features",
        "--output-dir",
        "/videos",
        "--config",
        "/tmp/bekiper_config.toml",
    ])
    .unwrap();
    let TestCli::Inspect(args) = cli else {
        panic!("expected Inspect");
    };
    let TestInspectCommand::Video {
        dir,
        output,
        output_dir,
        all_features,
        config,
        feature,
        heuristic,
        ..
    } = args.command
    else {
        panic!("expected Video");
    };
    assert_eq!(dir.to_str().unwrap(), "/my/dir");
    assert!(output.is_none());
    assert_eq!(output_dir.unwrap().to_str().unwrap(), "/videos");
    assert!(all_features);
    assert_eq!(
        config.unwrap().to_str().unwrap(),
        "/tmp/bekiper_config.toml"
    );
    assert!(feature.is_none());
    assert!(heuristic.is_none());
}

#[test]
fn parse_inspect_info_defaults() {
    let cli = TestCli::try_parse_from(["bekiper", "inspect", "info"]).unwrap();
    let TestCli::Inspect(args) = cli else {
        panic!("expected Inspect");
    };
    let TestInspectCommand::Info { dir } = args.command else {
        panic!("expected Info");
    };
    assert_eq!(dir.to_str().unwrap(), "inspection");
}

#[test]
fn parse_inspect_info_with_dir() {
    let cli = TestCli::try_parse_from(["bekiper", "inspect", "info", "/my/output"]).unwrap();
    let TestCli::Inspect(args) = cli else {
        panic!("expected Inspect");
    };
    let TestInspectCommand::Info { dir } = args.command else {
        panic!("expected Info");
    };
    assert_eq!(dir.to_str().unwrap(), "/my/output");
}

#[test]
fn parse_inspect_check_annotation() {
    let cli = TestCli::try_parse_from([
        "bekiper",
        "inspect",
        "check-annotation",
        "-p",
        "/data/annotations.json",
    ])
    .unwrap();
    assert!(matches!(
        cli,
        TestCli::Inspect(TestInspectArgs {
            command: TestInspectCommand::CheckAnnotation { .. }
        })
    ));
}
