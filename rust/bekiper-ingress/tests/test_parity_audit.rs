use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use bekiper_ingress::aggregate::{TrialResult, build_sheet_dataframes};
use bekiper_ingress::config::AggregateConfig;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct ParityFixture {
    source_workbook: String,
    source_sheet: String,
    source_row: String,
    expected_sheet: String,
    #[serde(default)]
    aggregate: AggregateConfig,
    #[serde(flatten)]
    trial: TrialResult,
    expected_cells: BTreeMap<String, toml::Value>,
}

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/parity")
}

fn load_fixture(name: &str) -> ParityFixture {
    let path = fixture_dir().join(name);
    let raw = fs::read_to_string(&path).unwrap_or_else(|err| {
        panic!("failed to read parity fixture {}: {err}", path.display());
    });
    toml::from_str(&raw).unwrap_or_else(|err| {
        panic!("failed to parse parity fixture {}: {err}", path.display());
    })
}

fn assert_expected_cell(df: &polars::prelude::DataFrame, column: &str, expected: &toml::Value) {
    let series = df.column(column).unwrap_or_else(|err| {
        panic!("missing expected column '{column}': {err}");
    });
    match expected {
        toml::Value::String(value) => {
            assert_eq!(
                series.str().unwrap().get(0),
                Some(value.as_str()),
                "string mismatch in column '{column}'"
            );
        }
        toml::Value::Integer(value) => {
            assert_eq!(
                series.i64().unwrap().get(0),
                Some(*value),
                "integer mismatch in column '{column}'"
            );
        }
        toml::Value::Float(value) => {
            let actual = series
                .f64()
                .unwrap()
                .get(0)
                .unwrap_or_else(|| panic!("expected a float in column '{column}'"));
            assert!(
                (actual - value).abs() < 1e-9,
                "float mismatch in column '{column}': expected {value}, got {actual}"
            );
        }
        toml::Value::Boolean(value) => {
            assert_eq!(
                series.bool().unwrap().get(0),
                Some(*value),
                "boolean mismatch in column '{column}'"
            );
        }
        other => panic!("unsupported expected value in column '{column}': {other:?}"),
    }
}

fn run_fixture(name: &str) {
    let fixture = load_fixture(name);
    let sheets = build_sheet_dataframes(vec![fixture.trial], &fixture.aggregate)
        .unwrap_or_else(|err| panic!("failed to build aggregate dataframe for {name}: {err}"));

    assert_eq!(
        sheets.len(),
        1,
        "fixture {name} ({}/{}) should produce exactly one sheet",
        fixture.source_workbook,
        fixture.source_sheet
    );
    let (sheet_name, df, _) = &sheets[0];
    assert_eq!(
        sheet_name, &fixture.expected_sheet,
        "sheet-name mismatch for legacy source {} [{} row {}]",
        fixture.source_workbook, fixture.source_sheet, fixture.source_row
    );
    assert_eq!(df.height(), 1, "fixture {name} should produce a single row");

    for (column, expected) in &fixture.expected_cells {
        assert_expected_cell(df, column, expected);
    }
}

#[test]
fn parity_audit_fixture_nort_novelty() {
    run_fixture("nort_novelty.toml");
}

#[test]
fn parity_audit_fixture_oul_training() {
    run_fixture("oul_training.toml");
}

#[test]
fn parity_audit_fixture_oul_update() {
    run_fixture("oul_update.toml");
}

#[test]
fn parity_audit_fixture_oul_test() {
    run_fixture("oul_test.toml");
}

#[test]
fn parity_audit_fixture_open_field() {
    run_fixture("open_field.toml");
}

#[test]
fn parity_audit_fixture_ymaze() {
    run_fixture("ymaze.toml");
}

#[test]
fn parity_audit_fixture_cheeseboard() {
    run_fixture("cheeseboard.toml");
}

#[test]
fn dejana_run_root_regression_populates_object_recognition_cells() {
    let rows = vec![
        load_fixture("oul_training.toml").trial,
        load_fixture("oul_update.toml").trial,
        load_fixture("oul_test.toml").trial,
    ];

    let mut cfg = AggregateConfig::default();
    cfg.recipe_to_sheet
        .insert("oultraining".into(), "Training".into());
    cfg.recipe_to_sheet
        .insert("oulupdate".into(), "Update".into());
    cfg.recipe_to_sheet.insert("oultest".into(), "Test".into());

    let sheets = build_sheet_dataframes(rows, &cfg).expect("dejana aggregate should build");
    assert_eq!(sheets.len(), 3);

    let training = sheets
        .iter()
        .find(|(name, _, _)| name == "Training")
        .expect("training sheet missing");
    let update = sheets
        .iter()
        .find(|(name, _, _)| name == "Update")
        .expect("update sheet missing");
    let test = sheets
        .iter()
        .find(|(name, _, _)| name == "Test")
        .expect("test sheet missing");

    let training_df = &training.1;
    assert!(
        training_df
            .column("total_observation_s")
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .unwrap()
            > 0.0
    );
    assert!(
        training_df
            .column("object_1_exploration_s")
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .unwrap()
            > 0.0
    );
    assert!(
        training_df
            .column("pair_object_1_object_2_delta_s")
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .unwrap()
            > 0.0
    );

    let update_df = &update.1;
    assert_eq!(
        update_df
            .column("object_3_exploration_s")
            .unwrap()
            .f64()
            .unwrap()
            .get(0),
        Some(0.0)
    );
    assert!(
        update_df
            .column("object_3_body_total_s")
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .unwrap()
            > 0.0
    );

    let test_df = &test.1;
    assert_eq!(
        test_df
            .column("object_4_exploration_s")
            .unwrap()
            .f64()
            .unwrap()
            .get(0),
        Some(0.0)
    );
    assert!(
        test_df
            .column("object_4_body_total_s")
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .unwrap()
            > 0.0
    );
}
