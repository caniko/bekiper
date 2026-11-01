use bekiper_post_analysis::statistics;
use serde_json::{Value, json};
use statistics::*;

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/statistics_oracle.json")).unwrap()
}

fn close(actual: &Value, expected: &Value, tolerance: f64) {
    let actual = actual.as_f64().expect("finite numeric output");
    let expected = expected.as_f64().unwrap();
    assert!(
        (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
        "{actual} != {expected}"
    );
}

fn probability_close(actual: &Value, expected: &Value) {
    let actual = actual.as_f64().expect("finite positive P value");
    let expected = expected.as_f64().unwrap();
    assert!(expected > 0.0 && actual > 0.0);
    assert!(
        (actual / expected - 1.0).abs() < 1e-9,
        "{actual} != {expected}"
    );
}

include!("statistics/models.rs");
include!("statistics/inclusion.rs");
include!("statistics/adjustments.rs");
include!("statistics/paired_checks.rs");
include!("statistics/probabilities.rs");
