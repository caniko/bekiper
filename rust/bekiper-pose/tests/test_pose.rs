use std::io::Write;
use std::path::Path;

use bekiper_pose::deeplabcut::DeepLabCutReader;
use bekiper_pose::detect::{detect_backend, open};
use bekiper_pose::lightning::LightningPoseReader;
use bekiper_pose::skeleton::Skeleton;
use bekiper_pose::sleap::SleapCsvReader;
use bekiper_pose::{PoseBackend, PoseReader};

fn write_csv(dir: &Path, name: &str, content: &str) -> std::path::PathBuf {
    let path = dir.join(name);
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(content.as_bytes()).unwrap();
    path
}

fn test_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("bekiper_pose_test_{name}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

// ── DeepLabCut ──────────────────────────────────────────────────────────────

const DLC_CSV: &str = "\
scorer,scorer_name,scorer_name,scorer_name,scorer_name,scorer_name,scorer_name
bodyparts,nose,nose,nose,tail_base,tail_base,tail_base
coords,x,y,likelihood,x,y,likelihood
0,100.0,200.0,0.99,150.0,250.0,0.95
1,101.0,201.0,0.98,151.0,251.0,0.94
2,102.0,202.0,0.97,152.0,252.0,0.93
";

#[test]
fn dlc_auto_detects_labels() {
    let dir = test_dir("dlc_auto");
    let path = write_csv(&dir, "dlc.csv", DLC_CSV);
    let reader = DeepLabCutReader::from_file(&path).unwrap();
    assert_eq!(reader.labels(), &["nose", "tail_base"]);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn dlc_reads_to_flat_columns() {
    let dir = test_dir("dlc_flat");
    let path = write_csv(&dir, "dlc.csv", DLC_CSV);
    let reader = DeepLabCutReader::from_file(&path).unwrap();
    let df = reader.read(&path).unwrap().collect().unwrap();

    assert!(df.column("frame").is_ok());
    assert!(df.column("nose_x").is_ok());
    assert!(df.column("nose_y").is_ok());
    assert!(df.column("nose_likelihood").is_ok());
    assert!(df.column("tail_base_x").is_ok());
    assert!(df.column("tail_base_likelihood").is_ok());

    let nose_x: Vec<f64> = df
        .column("nose_x")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    assert_eq!(nose_x, vec![100.0, 101.0, 102.0]);

    assert_eq!(df.height(), 3);

    std::fs::remove_dir_all(&dir).ok();
}

// ── SLEAP ───────────────────────────────────────────────────────────────────

const SLEAP_CSV: &str = "\
track,frame_idx,nose.x,nose.y,nose.score,tail_base.x,tail_base.y,tail_base.score
track_0,0,100.0,200.0,0.99,150.0,250.0,0.95
track_0,1,101.0,201.0,0.98,151.0,251.0,0.94
track_0,2,102.0,202.0,0.97,152.0,252.0,0.93
";

#[test]
fn sleap_auto_detects_labels() {
    let dir = test_dir("sleap_auto");
    let path = write_csv(&dir, "sleap.csv", SLEAP_CSV);
    let reader = SleapCsvReader::from_file(&path).unwrap();
    assert_eq!(reader.labels(), &["nose", "tail_base"]);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn sleap_reads_to_flat_columns() {
    let dir = test_dir("sleap_flat");
    let path = write_csv(&dir, "sleap.csv", SLEAP_CSV);
    let reader = SleapCsvReader::from_file(&path).unwrap();
    let df = reader.read(&path).unwrap().collect().unwrap();

    assert!(df.column("frame").is_ok(), "expected 'frame' column");
    assert!(df.column("nose_x").is_ok());
    assert!(df.column("nose_y").is_ok());
    assert!(df.column("nose_likelihood").is_ok());
    assert!(df.column("tail_base_x").is_ok());

    // track column should be dropped
    assert!(
        df.column("track").is_err(),
        "track column should be dropped"
    );

    let nose_x: Vec<f64> = df
        .column("nose_x")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    assert_eq!(nose_x, vec![100.0, 101.0, 102.0]);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn sleap_without_track_column() {
    let csv = "\
frame_idx,nose.x,nose.y,nose.score,tail_base.x,tail_base.y,tail_base.score
0,100.0,200.0,0.99,150.0,250.0,0.95
1,101.0,201.0,0.98,151.0,251.0,0.94
";
    let dir = test_dir("sleap_notrack");
    let path = write_csv(&dir, "sleap.csv", csv);
    let reader = SleapCsvReader::from_file(&path).unwrap();
    let df = reader.read(&path).unwrap().collect().unwrap();

    assert!(df.column("frame").is_ok());
    assert_eq!(df.height(), 2);
    std::fs::remove_dir_all(&dir).ok();
}

// ── Lightning Pose ──────────────────────────────────────────────────────────

const LP_CSV: &str = "\
nose_x,nose_y,nose_likelihood,tail_base_x,tail_base_y,tail_base_likelihood
100.0,200.0,0.99,150.0,250.0,0.95
101.0,201.0,0.98,151.0,251.0,0.94
102.0,202.0,0.97,152.0,252.0,0.93
";

#[test]
fn lp_auto_detects_labels() {
    let dir = test_dir("lp_auto");
    let path = write_csv(&dir, "lp.csv", LP_CSV);
    let reader = LightningPoseReader::from_file(&path).unwrap();
    assert_eq!(reader.labels(), &["nose", "tail_base"]);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn lp_reads_to_flat_columns() {
    let dir = test_dir("lp_flat");
    let path = write_csv(&dir, "lp.csv", LP_CSV);
    let reader = LightningPoseReader::from_file(&path).unwrap();
    let df = reader.read(&path).unwrap().collect().unwrap();

    assert!(df.column("frame").is_ok());
    assert!(df.column("nose_x").is_ok());
    assert!(df.column("tail_base_likelihood").is_ok());

    let nose_x: Vec<f64> = df
        .column("nose_x")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    assert_eq!(nose_x, vec![100.0, 101.0, 102.0]);

    assert_eq!(df.height(), 3);
    std::fs::remove_dir_all(&dir).ok();
}

// ── Auto-detection ──────────────────────────────────────────────────────────

#[test]
fn detect_dlc_backend() {
    let dir = test_dir("detect_dlc");
    let path = write_csv(&dir, "dlc.csv", DLC_CSV);
    assert_eq!(detect_backend(&path).unwrap(), PoseBackend::DeepLabCut);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn detect_sleap_backend() {
    let dir = test_dir("detect_sleap");
    let path = write_csv(&dir, "sleap.csv", SLEAP_CSV);
    assert_eq!(detect_backend(&path).unwrap(), PoseBackend::Sleap);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn detect_lp_backend() {
    let dir = test_dir("detect_lp");
    let path = write_csv(&dir, "lp.csv", LP_CSV);
    assert_eq!(detect_backend(&path).unwrap(), PoseBackend::LightningPose);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn open_auto_detects_and_reads() {
    let dir = test_dir("open_auto");

    // Write all three formats
    let dlc_path = write_csv(&dir, "dlc.csv", DLC_CSV);
    let sleap_path = write_csv(&dir, "sleap.csv", SLEAP_CSV);
    let lp_path = write_csv(&dir, "lp.csv", LP_CSV);

    // Each should auto-detect and read successfully
    for (path, expected_backend) in [
        (&dlc_path, PoseBackend::DeepLabCut),
        (&sleap_path, PoseBackend::Sleap),
        (&lp_path, PoseBackend::LightningPose),
    ] {
        let reader = open(path).unwrap();
        assert_eq!(reader.backend(), expected_backend);

        let df = reader.read(path).unwrap().collect().unwrap();
        assert!(
            df.column("nose_x").is_ok(),
            "missing nose_x for {expected_backend:?}"
        );
        assert!(
            df.column("nose_y").is_ok(),
            "missing nose_y for {expected_backend:?}"
        );
        assert_eq!(df.height(), 3, "wrong row count for {expected_backend:?}");
    }

    std::fs::remove_dir_all(&dir).ok();
}

// ── All backends produce identical output ───────────────────────────────────

#[test]
fn all_backends_produce_same_schema() {
    let dir = test_dir("schema_compat");

    let dlc_path = write_csv(&dir, "dlc.csv", DLC_CSV);
    let sleap_path = write_csv(&dir, "sleap.csv", SLEAP_CSV);
    let lp_path = write_csv(&dir, "lp.csv", LP_CSV);

    let dlc_df = open(&dlc_path)
        .unwrap()
        .read(&dlc_path)
        .unwrap()
        .collect()
        .unwrap();
    let sleap_df = open(&sleap_path)
        .unwrap()
        .read(&sleap_path)
        .unwrap()
        .collect()
        .unwrap();
    let lp_df = open(&lp_path)
        .unwrap()
        .read(&lp_path)
        .unwrap()
        .collect()
        .unwrap();

    // All must have these columns
    for col_name in [
        "frame",
        "nose_x",
        "nose_y",
        "nose_likelihood",
        "tail_base_x",
        "tail_base_y",
        "tail_base_likelihood",
    ] {
        assert!(dlc_df.column(col_name).is_ok(), "DLC missing {col_name}");
        assert!(
            sleap_df.column(col_name).is_ok(),
            "SLEAP missing {col_name}"
        );
        assert!(lp_df.column(col_name).is_ok(), "LP missing {col_name}");
    }

    // Values should be identical across backends
    let dlc_nose: Vec<f64> = dlc_df
        .column("nose_x")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let sleap_nose: Vec<f64> = sleap_df
        .column("nose_x")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let lp_nose: Vec<f64> = lp_df
        .column("nose_x")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    assert_eq!(dlc_nose, sleap_nose);
    assert_eq!(dlc_nose, lp_nose);

    std::fs::remove_dir_all(&dir).ok();
}

// ── Skeleton ────────────────────────────────────────────────────────────────

#[test]
fn mouse_top_view_skeleton() {
    let skel = Skeleton::mouse_top_view();
    assert_eq!(skel.name, "mouse_top_view");
    assert_eq!(skel.nodes.len(), 27);
    assert_eq!(skel.edges.len(), 26);

    let labels = skel.labels();
    assert!(labels.contains(&"nose".to_string()));
    assert!(labels.contains(&"tail_base".to_string()));
    assert!(labels.contains(&"tail_end".to_string()));
}

#[test]
fn mouse_full_skeleton() {
    let skel = Skeleton::mouse_full();
    assert_eq!(skel.name, "mouse_full");
    assert_eq!(skel.nodes.len(), 13);
    assert_eq!(skel.edges.len(), 12);

    let labels = skel.labels();
    assert!(labels.contains(&"left_forepaw".to_string()));
    assert!(labels.contains(&"right_hindpaw".to_string()));
    assert!(labels.contains(&"belly".to_string()));
}

#[test]
fn skeleton_serializes_to_toml() {
    let skel = Skeleton::mouse_top_view();
    let toml_str = toml::to_string(&skel).unwrap();
    let deserialized: Skeleton = toml::from_str(&toml_str).unwrap();
    assert_eq!(deserialized.name, skel.name);
    assert_eq!(deserialized.nodes.len(), skel.nodes.len());
    assert_eq!(deserialized.edges.len(), skel.edges.len());
}

#[test]
fn skeleton_presets_resolve_to_their_topology() {
    use bekiper_pose::skeleton::SkeletonPreset;
    assert_eq!(
        SkeletonPreset::MouseTopView.resolve().name,
        "mouse_top_view"
    );
    assert_eq!(SkeletonPreset::MouseFull.resolve().name, "mouse_full");
    let tail = SkeletonPreset::ImenDlcTail.resolve();
    assert_eq!(tail.name, "imen_dlc_tail");
    assert_eq!(
        tail.labels(),
        ["nose", "left_ear", "right_ear", "center_ear", "tail"]
    );
    let base_tail = SkeletonPreset::ImenDlcBaseTail.resolve();
    assert_eq!(base_tail.name, "imen_dlc_base_tail");
    assert!(base_tail.edges.iter().any(|e| e.target == "base_tail"));
    base_tail.validate().unwrap();
}

#[test]
fn skeleton_validate_rejects_each_malformed_topology() {
    use bekiper_pose::skeleton::{Edge, Node};
    let nodes = |names: &[&str]| names.iter().map(|n| Node::new(*n)).collect::<Vec<_>>();
    let edge = |s: &str, t: &str| Edge::new(s, t);
    let cases = [
        (Skeleton::new("s", vec![], vec![]), "at least one node"),
        (Skeleton::new("s", nodes(&["a", ""]), vec![]), "empty node"),
        (
            Skeleton::new("s", nodes(&["a", "a"]), vec![]),
            "duplicate node",
        ),
        (
            Skeleton::new("s", nodes(&["a"]), vec![edge("a", "a")]),
            "self-edge",
        ),
        (
            Skeleton::new("s", nodes(&["a"]), vec![edge("x", "a")]),
            "source 'x'",
        ),
        (
            Skeleton::new("s", nodes(&["a"]), vec![edge("a", "y")]),
            "target 'y'",
        ),
        (
            Skeleton::new(
                "s",
                nodes(&["a", "b"]),
                vec![edge("a", "b"), edge("a", "b")],
            ),
            "duplicate edge",
        ),
    ];
    for (skeleton, expected) in cases {
        let error = skeleton.validate().unwrap_err().to_string();
        assert!(error.contains(expected), "{error} lacks {expected}");
    }
}

#[test]
fn skeleton_validate_against_labels_checks_label_list() {
    let skeleton = Skeleton::imen_dlc("tail");
    let labels = |names: &[&str]| names.iter().map(|n| n.to_string()).collect::<Vec<_>>();
    let all = labels(&[
        "nose",
        "left_ear",
        "right_ear",
        "center_ear",
        "tail",
        "extra",
    ]);
    skeleton.validate_against_labels(&all).unwrap();
    for (bad, expected) in [
        (labels(&["nose", ""]), "empty strings"),
        (labels(&["nose", "nose"]), "duplicate label 'nose'"),
        (
            labels(&["nose", "left_ear"]),
            "missing right_ear, center_ear, tail",
        ),
    ] {
        let error = skeleton
            .validate_against_labels(&bad)
            .unwrap_err()
            .to_string();
        assert!(error.contains(expected), "{error} lacks {expected}");
    }
    // An invalid skeleton fails before labels are inspected.
    let empty = Skeleton::new("empty", vec![], vec![]);
    assert!(empty.validate_against_labels(&all).is_err());
}

#[test]
fn dlc_header_errors_and_h5_paths_are_reported() {
    let dir = test_dir("dlc_header_errors");
    let only_scorer = write_csv(&dir, "scorer_only.csv", "scorer,a\n");
    assert!(
        DeepLabCutReader::from_file(&only_scorer)
            .err()
            .unwrap()
            .to_string()
            .contains("missing bodyparts")
    );
    let no_labels = write_csv(&dir, "no_labels.csv", "scorer,a,b\nbodyparts, ,\n");
    assert!(
        DeepLabCutReader::from_file(&no_labels)
            .err()
            .unwrap()
            .to_string()
            .contains("No body part labels")
    );
    assert!(DeepLabCutReader::from_file(&dir.join("absent.csv")).is_err());
    // Missing HDF5 files are reported both for header parsing and reading.
    let absent_h5 = dir.join("absent.h5");
    assert!(DeepLabCutReader::from_file(&absent_h5).is_err());
    let reader = DeepLabCutReader::with_labels(vec!["nose".into()]);
    assert!(reader.read(&absent_h5).is_err());
    assert_eq!(reader.backend(), PoseBackend::DeepLabCut);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn dlc_csv_with_fewer_columns_than_labels_keeps_complete_triples() {
    let dir = test_dir("dlc_short_columns");
    let path = write_csv(&dir, "short.csv", DLC_CSV);
    // A third label has no x/y/likelihood triple and is left unmapped.
    let reader =
        DeepLabCutReader::with_labels(vec!["nose".into(), "tail_base".into(), "ghost".into()]);
    let df = reader.read(&path).unwrap().collect().unwrap();
    assert!(df.column("tail_base_likelihood").is_ok());
    assert!(df.column("ghost_x").is_err());
    std::fs::remove_dir_all(&dir).ok();
}
