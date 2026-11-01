use polars::df;

use super::*;

#[test]
fn direct_pose_loading_matches_checkpoint_and_preserves_source() {
    let dir = std::env::temp_dir().join(format!("bekiper_direct_pose_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("source.parquet");
    let checkpoint = dir.join("checkpoint.parquet");
    let mut df = df! {
        "frame" => &[0i64, 1],
        "nose_x" => &[1.0f64, 2.0],
        "nose_y" => &[10.0f64, 12.0],
        "nose_likelihood" => &[0.9f64, 0.95],
    }
    .unwrap();
    write_parquet(&mut df, &src).unwrap();
    std::fs::write(
        src.with_extension("dlc_crop.toml"),
        "x_offset = 3.0\ny_offset = 4.0\n",
    )
    .unwrap();
    let before = std::fs::read(&src).unwrap();
    let options = PoseImportOptions {
        invert_y: true,
        frame_height_px: Some(30),
    };
    let (direct, report) = load_normalized_pose(&src, options).unwrap();
    let direct = direct.collect().unwrap();
    assert_eq!(report.labels, vec!["nose"]);
    assert_eq!(
        direct.column("nose_x").unwrap().f64().unwrap().get(0),
        Some(4.0)
    );
    assert_eq!(
        direct.column("nose_y").unwrap().f64().unwrap().get(0),
        Some(16.0)
    );
    assert!(!checkpoint.exists());
    import_pose_file_to_parquet(&src, &checkpoint, options).unwrap();
    assert!(direct.equals_missing(&scan_file(&checkpoint).unwrap().collect().unwrap()));
    assert_eq!(before, std::fs::read(&src).unwrap());
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn collect_convert_inputs_supports_file_and_dir() {
    let dir = std::env::temp_dir().join("bekiper_pose_import_collect");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("nested")).unwrap();

    let csv = dir.join("a.csv");
    let h5 = dir.join("nested/b.h5");
    let parquet = dir.join("nested/c.parquet");
    std::fs::write(&csv, b"scorer,a\n").unwrap();
    std::fs::write(&h5, b"h5").unwrap();
    std::fs::write(&parquet, b"pq").unwrap();

    let files = collect_convert_inputs(&dir).unwrap();
    assert_eq!(files, vec![csv.clone(), h5.clone()]);

    let one = collect_convert_inputs(&csv).unwrap();
    assert_eq!(one, vec![csv]);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn import_pose_file_to_parquet_inverts_y_when_requested() {
    let dir = std::env::temp_dir().join("bekiper_pose_import_invert_y");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let src = dir.join("poses.parquet");
    let out = dir.join("out.parquet");
    let mut df = df! {
        "frame" => &[0i64, 1],
        "nose_x" => &[1.0f64, 2.0],
        "nose_y" => &[10.0f64, 12.0],
        "nose_likelihood" => &[0.9f64, 0.95],
    }
    .unwrap();
    write_parquet(&mut df, &src).unwrap();

    let report = import_pose_file_to_parquet(
        &src,
        &out,
        PoseImportOptions {
            invert_y: true,
            frame_height_px: Some(20),
        },
    )
    .unwrap();
    assert!(matches!(report.format, InputFormat::Parquet));

    let df = scan_file(&out).unwrap().collect().unwrap();
    let y = df.column("nose_y").unwrap().f64().unwrap();
    assert_eq!(y.get(0), Some(10.0));
    assert_eq!(y.get(1), Some(8.0));

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn import_pose_file_to_parquet_applies_dlc_crop_sidecar() {
    let dir = std::env::temp_dir().join("bekiper_pose_import_dlc_crop");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let src = dir.join("poses.parquet");
    let out = dir.join("out.parquet");
    std::fs::write(
        src.with_extension("dlc_crop.toml"),
        "[dlc_crop]\nx_offset = 95.0\ny_offset = 75.0\n",
    )
    .unwrap();
    let mut df = df! {
        "frame" => &[0i64, 1],
        "nose_x" => &[213.0f64, 214.0],
        "nose_y" => &[252.0f64, 253.0],
        "nose_likelihood" => &[0.9f64, 0.95],
        "left_ear_x" => &[207.0f64, 208.0],
        "left_ear_y" => &[264.0f64, 265.0],
        "left_ear_likelihood" => &[0.8f64, 0.85],
    }
    .unwrap();
    write_parquet(&mut df, &src).unwrap();

    let report = import_pose_file_to_parquet(&src, &out, PoseImportOptions::default()).unwrap();
    assert!(matches!(report.format, InputFormat::Parquet));

    let df = scan_file(&out).unwrap().collect().unwrap();
    assert_eq!(
        df.column("nose_x").unwrap().f64().unwrap().get(0),
        Some(308.0)
    );
    assert_eq!(
        df.column("nose_y").unwrap().f64().unwrap().get(0),
        Some(327.0)
    );
    assert_eq!(
        df.column("left_ear_x").unwrap().f64().unwrap().get(1),
        Some(303.0)
    );
    assert_eq!(
        df.column("left_ear_y").unwrap().f64().unwrap().get(1),
        Some(340.0)
    );

    std::fs::remove_dir_all(&dir).ok();
}

fn scratch(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("bekiper_pose_import_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

const DLC_CSV: &str = "scorer,dlc,dlc,dlc\nbodyparts,nose,nose,nose\ncoords,x,y,likelihood\n0,1.0,2.0,0.9\n1,3.0,4.0,0.8\n";

#[test]
fn collect_convert_inputs_rejects_unsupported_and_missing_paths() {
    let dir = scratch("collect_errors");
    let parquet = dir.join("c.parquet");
    std::fs::write(&parquet, b"pq").unwrap();
    let error = collect_convert_inputs(&parquet).unwrap_err().to_string();
    assert!(error.contains("unsupported convert input"), "{error}");
    let error = collect_convert_inputs(&dir.join("missing"))
        .unwrap_err()
        .to_string();
    assert!(error.contains("does not exist"), "{error}");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[cfg(feature = "pose-hdf5")]
#[test]
fn dlc_csv_import_reports_backend_and_shape() {
    let dir = scratch("dlc_csv");
    let csv = dir.join("poses.csv");
    std::fs::write(&csv, DLC_CSV).unwrap();
    let (lf, report) = load_normalized_pose(&csv, PoseImportOptions::default()).unwrap();
    assert!(matches!(report.format, InputFormat::Csv));
    assert_eq!(report.backend, Some(PoseBackend::DeepLabCut));
    assert_eq!(report.labels, ["nose"]);
    let df = lf.collect().unwrap();
    assert_eq!(
        df.column("nose_y").unwrap().f64().unwrap().get(1),
        Some(4.0)
    );
    assert_eq!(pose_source_shape(&csv).unwrap(), (2, 4));
    let mut checkpoints = 0;
    let (batched, labels) = load_normalized_pose_batched(&csv, 1, || {
        checkpoints += 1;
        Ok(())
    })
    .unwrap();
    assert_eq!(checkpoints, 2, "before decoding and after collecting");
    assert_eq!(labels, ["nose"]);
    assert!(batched.equals_missing(&df));
    // A cancelled checkpoint stops before decoding.
    assert!(
        load_normalized_pose_batched(&csv, 1, || Err(BekiperError::Other("cancel".into())))
            .is_err()
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[cfg(feature = "pose-hdf5")]
#[test]
fn parquet_shape_counts_rows_and_columns_and_needs_labels() {
    let dir = scratch("parquet_shape");
    let src = dir.join("poses.parquet");
    let mut df = df! {
        "frame" => &[0i64, 1, 2],
        "nose_x" => &[1.0f64, 2.0, 3.0],
        "nose_y" => &[1.0f64, 2.0, 3.0],
        "nose_likelihood" => &[1.0f64, 1.0, 1.0],
    }
    .unwrap();
    write_parquet(&mut df, &src).unwrap();
    assert_eq!(pose_source_shape(&src).unwrap(), (3, 4));
    let unlabeled = dir.join("unlabeled.parquet");
    let mut df = df! { "frame_x" => &[0i64], "value" => &[1.0f64] }.unwrap();
    write_parquet(&mut df, &unlabeled).unwrap();
    let error = load_normalized_pose(&unlabeled, PoseImportOptions::default())
        .err()
        .unwrap()
        .to_string();
    assert!(
        error.contains("could not discover body-part labels"),
        "{error}"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn dlc_crop_sidecar_edge_cases() {
    let dir = scratch("sidecar");
    let src = dir.join("poses.csv");
    std::fs::write(&src, DLC_CSV).unwrap();
    let sidecar = src.with_extension("dlc_crop.toml");
    let x = |options| {
        let (lf, _) = load_normalized_pose(&src, options)?;
        Ok::<_, BekiperError>(lf.collect()?.column("nose_x")?.f64()?.get(0))
    };
    // A zero offset and a sidecar without offsets both leave coordinates alone.
    for contents in ["x_offset = 0.0\ny_offset = 0.0\n", "note = 'uncropped'\n"] {
        std::fs::write(&sidecar, contents).unwrap();
        assert_eq!(x(PoseImportOptions::default()).unwrap(), Some(1.0));
    }
    for (contents, message) in [
        ("x_offset = 5.0\n", "must both be present"),
        ("y_offset = 5.0\n", "must both be present"),
        ("x_offset = 'five'\n", "invalid DLC crop sidecar"),
    ] {
        std::fs::write(&sidecar, contents).unwrap();
        let error = x(PoseImportOptions::default()).unwrap_err().to_string();
        assert!(error.contains(message), "{contents}: {error}");
    }
    std::fs::remove_file(&sidecar).unwrap();
    std::fs::create_dir(&sidecar).unwrap();
    let error = x(PoseImportOptions::default()).unwrap_err().to_string();
    assert!(error.contains("failed to read DLC crop sidecar"), "{error}");
    std::fs::remove_dir(&sidecar).unwrap();
    let error = x(PoseImportOptions {
        invert_y: true,
        frame_height_px: None,
    })
    .unwrap_err()
    .to_string();
    assert!(error.contains("requires a source frame height"), "{error}");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[cfg(feature = "pose-hdf5")]
#[derive(hdf5_metno::H5Type, Clone)]
#[repr(C)]
struct DlcRow {
    index: i64,
    values_block_0: [f64; 12],
}

/// pandas/pytables DLC table with four labels, as written by DeepLabCut.
#[cfg(feature = "pose-hdf5")]
fn write_dlc_h5(path: &Path, rows: usize) {
    let file = hdf5_metno::File::create(path).unwrap();
    let group = file.create_group("df_with_missing").unwrap();
    let labels =
        hdf5_metno::types::FixedAscii::<512>::from_ascii(b"Vnose\nVleft_ear\nVright_ear\nVtail\n")
            .unwrap();
    group
        .new_attr::<hdf5_metno::types::FixedAscii<512>>()
        .shape(())
        .create("non_index_axes")
        .unwrap()
        .write_scalar(&labels)
        .unwrap();
    let rows: Vec<_> = (0..rows)
        .map(|index| DlcRow {
            index: index as i64,
            values_block_0: [index as f64, 1.0, 0.9].repeat(4).try_into().unwrap(),
        })
        .collect();
    group
        .new_dataset_builder()
        .with_data(&rows)
        .create("table")
        .unwrap();
}

#[cfg(feature = "pose-hdf5")]
#[test]
fn dlc_h5_batches_stack_offset_and_report_their_shape() {
    let dir = scratch("dlc_h5");
    let path = dir.join("poses.h5");
    write_dlc_h5(&path, 5);
    std::fs::write(
        path.with_extension("dlc_crop.toml"),
        "x_offset = 10.0\ny_offset = 5.0\n",
    )
    .unwrap();
    let mut checkpoints = 0;
    let (frame, labels) = load_normalized_pose_batched(&path, 2, || {
        checkpoints += 1;
        Ok(())
    })
    .unwrap();
    assert_eq!(labels, ["nose", "left_ear", "right_ear", "tail"]);
    assert_eq!(checkpoints, 4, "once before decoding and once per batch");
    assert_eq!(frame.height(), 5);
    let x = frame.column("tail_x").unwrap().f64().unwrap();
    assert_eq!(
        x.get(4),
        Some(14.0),
        "the crop offset is applied after stacking"
    );
    assert_eq!(pose_source_shape(&path).unwrap(), (5, 13));

    let empty = dir.join("empty.h5");
    write_dlc_h5(&empty, 0);
    let error = load_normalized_pose_batched(&empty, 2, || Ok(()))
        .unwrap_err()
        .to_string();
    assert!(error.contains("pose input is empty"), "{error}");
    std::fs::remove_dir_all(&dir).unwrap();
}
