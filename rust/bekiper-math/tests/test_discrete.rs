use polars::prelude::*;

use bekiper_math::discrete::*;

#[test]
fn count_true_sums_booleans() {
    let df = df! {
        "flag" => &[true, false, true, true, false],
    }
    .unwrap();

    let result = df
        .lazy()
        .select([count_true_expr("flag").alias("total")])
        .collect()
        .unwrap();

    let total = result.column("total").unwrap().u32().unwrap();
    assert_eq!(total.get(0).unwrap(), 3);
}

#[test]
fn count_true_all_false() {
    let df = df! {
        "flag" => &[false, false, false],
    }
    .unwrap();

    let result = df
        .lazy()
        .select([count_true_expr("flag").alias("total")])
        .collect()
        .unwrap();

    let total = result.column("total").unwrap().u32().unwrap();
    assert_eq!(total.get(0).unwrap(), 0);
}

#[test]
fn any_of_logical_or() {
    let df = df! {
        "a" => &[true, false, false],
        "b" => &[false, true, false],
    }
    .unwrap();

    let result = df
        .lazy()
        .with_column(any_of(vec![col("a"), col("b")]).alias("any"))
        .collect()
        .unwrap();

    let any_col = result.column("any").unwrap().bool().unwrap();
    assert!(any_col.get(0).unwrap());
    assert!(any_col.get(1).unwrap());
    assert!(!any_col.get(2).unwrap());
}

#[test]
fn all_of_logical_and() {
    let df = df! {
        "a" => &[true, true, false],
        "b" => &[true, false, true],
    }
    .unwrap();

    let result = df
        .lazy()
        .with_column(all_of(vec![col("a"), col("b")]).alias("all"))
        .collect()
        .unwrap();

    let all_col = result.column("all").unwrap().bool().unwrap();
    assert!(all_col.get(0).unwrap());
    assert!(!all_col.get(1).unwrap());
    assert!(!all_col.get(2).unwrap());
}

#[test]
fn bout_spans_lengths_and_first_frame() {
    let flags = BooleanChunked::from_iter_options(
        "f".into(),
        [
            Some(true),
            Some(true),
            None,
            Some(false),
            Some(true),
            None,
            Some(true),
        ]
        .into_iter(),
    );
    // Null breaks a bout like false; bouts touching both edges count.
    assert_eq!(true_series_spans(&flags), vec![0..2, 4..5, 6..7]);
    assert_eq!(count_bouts(&flags), 3);
    assert_eq!(bout_lengths_frames(&flags), vec![2, 1, 1]);
    assert_eq!(first_true_frame(&flags), Some(0));
    let empty = BooleanChunked::from_iter_options("e".into(), std::iter::empty());
    assert_eq!(count_bouts(&empty), 0);
    assert!(bout_lengths_frames(&empty).is_empty());
    assert_eq!(first_true_frame(&empty), None);
    let late =
        BooleanChunked::from_iter_options("l".into(), [None, Some(false), Some(true)].into_iter());
    assert_eq!(first_true_frame(&late), Some(2));
}

#[test]
fn zone_labels_take_first_matching_zone() {
    let df = df! {
        "a" => [Some(true), Some(false), Some(true), None],
        "b" => [Some(true), Some(true), Some(false), Some(false)],
        "x" => [1.0, 2.0, 3.0, 4.0],
    }
    .unwrap();
    // Missing and non-boolean columns are ignored.
    let labels = zone_label_series(&df, &["missing", "x", "a", "b"]);
    assert_eq!(labels.name().as_str(), "zone");
    assert_eq!(
        labels.into_iter().collect::<Vec<_>>(),
        vec![Some("a"), Some("b"), Some("a"), Some("none")]
    );
    assert_eq!(zone_label_series(&df, &[]).get(0), Some("none"));
}

#[test]
fn bool_column_seconds_counts_true_frames() {
    let df = df! {
        "f" => [Some(true), None, Some(true), Some(false)],
        "x" => [1.0, 2.0, 3.0, 4.0],
    }
    .unwrap();
    assert_eq!(bool_column_seconds(&df, "f", 4.0), 0.5);
    assert_eq!(bool_column_seconds(&df, "x", 4.0), 0.0);
    assert_eq!(bool_column_seconds(&df, "missing", 4.0), 0.0);
}
