use super::*;

#[test]
fn extract_dlc_labels_dejana_4_keypoint() {
    // Verbatim non_index_axes pickle from one of the dejana DLC files
    // (animal 1035, sequence 0). 4-keypoint mouse model with bodyparts
    // nose / left_ear / right_ear / tail_base.
    let pickle = "(lp0\n(I1\n(lp1\n(VDLC_resnet50_OULFeb6shuffle1_500000\n\
            p2\nVnose\np3\nVx\np4\ntp5\na(g2\ng3\nVy\np6\ntp7\na(g2\ng3\n\
            Vlikelihood\np8\ntp9\na(g2\nVleft_ear\np10\ng4\ntp11\na(g2\ng10\n\
            g6\ntp12\na(g2\ng10\ng8\ntp13\na(g2\nVright_ear\np14\ng4\ntp15\n\
            a(g2\ng14\ng6\ntp16\na(g2\ng14\ng8\ntp17\na(g2\nVtail_base\np18\n\
            g4\ntp19\na(g2\ng18\ng6\ntp20\na(g2\ng18\ng8\ntp21\natp22\na.";
    let labels = extract_dlc_labels_from_pickle(pickle);
    assert_eq!(labels, vec!["nose", "left_ear", "right_ear", "tail_base"]);
}

#[test]
fn extract_dlc_labels_skips_scorer_line() {
    // The scorer is on a `(V…` line and should be skipped (the leading
    // `(` makes strip_prefix('V') miss it). The body parts on plain `V…`
    // lines remain.
    let pickle = "(VScorer_Name_2024\np1\nVbp1\np2\nVx\np3";
    assert_eq!(extract_dlc_labels_from_pickle(pickle), vec!["bp1"]);
}

#[test]
fn extract_dlc_labels_filters_coord_names() {
    // Coord names x/y/likelihood must be removed even if they appear
    // multiple times. (In real pickles they're memo-referenced, but the
    // filter must still drop literal occurrences.)
    let pickle = "Vsnout\nVx\nVy\nVlikelihood\nVtail";
    assert_eq!(
        extract_dlc_labels_from_pickle(pickle),
        vec!["snout", "tail"]
    );
}

#[test]
fn is_hdf5_path_handles_extensions() {
    use std::path::PathBuf;
    assert!(is_hdf5_path(&PathBuf::from("foo.h5")));
    assert!(is_hdf5_path(&PathBuf::from("foo.H5")));
    assert!(is_hdf5_path(&PathBuf::from("foo.hdf5")));
    assert!(is_hdf5_path(&PathBuf::from("a/b/c.HDF5")));
    assert!(!is_hdf5_path(&PathBuf::from("foo.csv")));
    assert!(!is_hdf5_path(&PathBuf::from("foo")));
}

#[cfg(feature = "hdf5")]
#[test]
fn hdf5_batches_preserve_values_order_and_bound_decoding() {
    let path = std::env::temp_dir().join(format!("bekiper_dlc_batches_{}.h5", std::process::id()));
    let file = hdf5_metno::File::create(&path).unwrap();
    let group = file.create_group("df_with_missing").unwrap();
    let rows: Vec<_> = (0..23)
        .map(|i| DlcRow::<3> {
            index: i * 2,
            values_block_0: [i as f64, -(i as f64), 0.95],
        })
        .collect();
    group
        .new_dataset_builder()
        .with_data(&rows)
        .create("table")
        .unwrap();
    drop(group);
    drop(file);
    let reader = DeepLabCutReader::with_labels(vec!["nose".into()]);
    let mut sizes = Vec::new();
    let mut frames = Vec::new();
    let mut xs = Vec::new();
    reader
        .read_h5_batches(&path, 5, |batch| {
            sizes.push(batch.height());
            frames.extend(batch.column("frame")?.i64()?.into_no_null_iter());
            xs.extend(batch.column("nose_x")?.f64()?.into_no_null_iter());
            Ok(())
        })
        .unwrap();
    assert_eq!(sizes, [5, 5, 5, 5, 3]);
    assert_eq!(frames, (0..23).map(|i| i * 2).collect::<Vec<_>>());
    assert_eq!(xs, (0..23).map(|i| i as f64).collect::<Vec<_>>());
    assert!(reader.read_h5_batches(&path, 0, |_| Ok(())).is_err());
    let mut emitted = 0;
    assert!(
        reader
            .read_h5_batches(&path, 5, |_| {
                emitted += 1;
                Err(BekiperError::Other("cancelled".into()))
            })
            .is_err()
    );
    assert_eq!(emitted, 1);
    std::fs::remove_file(path).unwrap();
}

/// Pytables-style pickle naming `labels` as DLC body parts.
#[cfg(feature = "hdf5")]
fn pickle_for(labels: &[&str]) -> String {
    let mut pickle = String::from("(lp0\n(I1\n(lp1\n(VDLC_scorer\np2\n");
    for label in labels {
        pickle.push_str(&format!("V{label}\nVx\nVy\nVlikelihood\n"));
    }
    pickle + "a."
}

#[cfg(feature = "hdf5")]
fn h5_path(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("bekiper_dlc_{name}_{}.h5", std::process::id()))
}

/// Write a DLC-shaped file: the `df_with_missing` group, optionally its
/// `non_index_axes` pickle attribute, and a table produced by `table`.
#[cfg(feature = "hdf5")]
fn write_dlc_h5(
    name: &str,
    pickle: Option<&str>,
    table: impl FnOnce(&hdf5_metno::Group),
) -> std::path::PathBuf {
    use hdf5_metno::types::FixedAscii;
    let path = h5_path(name);
    let file = hdf5_metno::File::create(&path).unwrap();
    let group = file.create_group("df_with_missing").unwrap();
    if let Some(pickle) = pickle {
        group
            .new_attr::<FixedAscii<8192>>()
            .create("non_index_axes")
            .unwrap()
            .write_scalar(&FixedAscii::<8192>::from_ascii(pickle).unwrap())
            .unwrap();
    }
    table(&group);
    path
}

#[cfg(feature = "hdf5")]
fn two_label_rows(n: i64) -> Vec<DlcRow<6>> {
    (0..n)
        .map(|i| DlcRow::<6> {
            index: i,
            values_block_0: [i as f64, 2.0 * i as f64, 0.9, -(i as f64), 1.0, 0.5],
        })
        .collect()
}

#[cfg(feature = "hdf5")]
#[test]
fn hdf5_file_reads_labels_shape_and_rows_across_batches() {
    // 8193 rows force read_h5 to stack a second 8192-row slab.
    let rows = two_label_rows(8193);
    let path = write_dlc_h5("valid", Some(&pickle_for(&["nose", "tail"])), |g| {
        g.new_dataset_builder()
            .with_data(&rows)
            .create("table")
            .unwrap();
    });
    let reader = DeepLabCutReader::from_file(&path).unwrap();
    assert_eq!(reader.labels(), ["nose", "tail"]);
    assert_eq!(reader.h5_shape(&path).unwrap(), (8193, 7));
    let df = reader.read(&path).unwrap().collect().unwrap();
    assert_eq!(df.height(), 8193);
    let tail_y = df.column("tail_y").unwrap().f64().unwrap();
    assert_eq!(tail_y.get(8192), Some(1.0));
    let nose_y = df.column("nose_y").unwrap().f64().unwrap();
    assert_eq!(nose_y.get(8192), Some(16384.0));
    assert_eq!(
        df.column("frame").unwrap().i64().unwrap().get(8192),
        Some(8192)
    );

    // A reader configured for the wrong width cannot decode the compound.
    let wrong_width = DeepLabCutReader::with_labels(vec!["nose".into()]);
    assert!(wrong_width.read(&path).is_err());
    // Widths beyond the monomorphised set are rejected before reading.
    let too_wide = DeepLabCutReader::with_labels((0..33).map(|i| format!("p{i}")).collect());
    let error = too_wide.read(&path).err().unwrap().to_string();
    assert!(
        error.contains("unsupported DLC compound width 99"),
        "{error}"
    );
    std::fs::remove_file(path).unwrap();
}

#[cfg(feature = "hdf5")]
#[test]
fn hdf5_empty_table_is_an_error() {
    let rows = two_label_rows(0);
    let path = write_dlc_h5("empty", Some(&pickle_for(&["nose", "tail"])), |g| {
        g.new_dataset_builder()
            .with_data(&rows)
            .create("table")
            .unwrap();
    });
    let reader = DeepLabCutReader::from_file(&path).unwrap();
    assert_eq!(reader.h5_shape(&path).unwrap(), (0, 7));
    let error = reader.read(&path).err().unwrap().to_string();
    assert!(error.contains("table is empty"), "{error}");
    std::fs::remove_file(path).unwrap();
}

#[cfg(feature = "hdf5")]
#[test]
fn hdf5_multidimensional_or_missing_tables_are_rejected() {
    let path = write_dlc_h5("two_d", None, |g| {
        g.new_dataset::<f64>()
            .shape((2, 3))
            .create("table")
            .unwrap();
    });
    let reader = DeepLabCutReader::with_labels(vec!["nose".into()]);
    for error in [
        reader.h5_shape(&path).err().unwrap(),
        reader.read(&path).err().unwrap(),
    ] {
        assert!(error.to_string().contains("one dimension"), "{error}");
    }
    std::fs::remove_file(path).unwrap();

    let no_table = write_dlc_h5("no_table", None, |_| {});
    assert!(reader.h5_shape(&no_table).is_err());
    assert!(reader.read(&no_table).is_err());
    std::fs::remove_file(no_table).unwrap();

    let absent = h5_path("absent");
    assert!(reader.h5_shape(&absent).is_err());
    assert!(reader.read_h5_batches(&absent, 1, |_| Ok(())).is_err());
}

#[cfg(feature = "hdf5")]
#[test]
fn hdf5_header_rejects_each_malformed_layout() {
    let check = |name: &str, path: std::path::PathBuf, expected: &str| {
        let error = DeepLabCutReader::from_file(&path)
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains(expected), "{name}: {error} lacks {expected}");
        std::fs::remove_file(path).unwrap();
    };
    let two_labels = pickle_for(&["nose", "tail"]);

    let path = h5_path("no_group");
    drop(hdf5_metno::File::create(&path).unwrap());
    check("no group", path, "group df_with_missing");

    check(
        "no attribute",
        write_dlc_h5("no_attr", None, |_| {}),
        "attr non_index_axes",
    );

    let numeric_attr = write_dlc_h5("numeric_attr", None, |g| {
        g.new_attr::<i32>()
            .create("non_index_axes")
            .unwrap()
            .write_scalar(&7)
            .unwrap();
    });
    check("numeric attribute", numeric_attr, "read non_index_axes");

    check(
        "no table",
        write_dlc_h5("hdr_no_table", Some(&two_labels), |_| {}),
        "dataset table",
    );

    let narrow = write_dlc_h5("narrow", Some(&two_labels), |g| {
        g.new_dataset_builder()
            .with_data(&[1i32, 2])
            .create("table")
            .unwrap();
    });
    check("4-byte rows", narrow, "unexpected compound size 4");

    let two_floats: Vec<_> = (0..2)
        .map(|i| DlcRow::<2> {
            index: i,
            values_block_0: [0.0, 1.0],
        })
        .collect();
    let not_triples = write_dlc_h5("not_triples", Some(&two_labels), |g| {
        g.new_dataset_builder()
            .with_data(&two_floats)
            .create("table")
            .unwrap();
    });
    check("width 2", not_triples, "not a multiple of 3");

    let rows = two_label_rows(2);
    let mismatch = write_dlc_h5("mismatch", Some(&pickle_for(&["nose"])), |g| {
        g.new_dataset_builder()
            .with_data(&rows)
            .create("table")
            .unwrap();
    });
    check("label mismatch", mismatch, "extracted 1 body-part labels");

    // An index-only table matches an empty label list but is still unusable.
    let index_only = write_dlc_h5("index_only", Some(&pickle_for(&[])), |g| {
        g.new_dataset_builder()
            .with_data(&[0i64, 1])
            .create("table")
            .unwrap();
    });
    check("no labels", index_only, "no body part labels");
}
