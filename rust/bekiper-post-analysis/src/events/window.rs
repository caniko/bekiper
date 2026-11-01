//! Registered half-open time windows and finite event values.
use super::*;

pub(super) fn finite(value: &Value, name: &str) -> Result<f64> {
    let value = value
        .as_f64()
        .with_context(|| format!("{name}: requires a finite number"))?;
    ensure!(value.is_finite(), "{name}: requires a finite number");
    Ok(value)
}

pub(super) fn text<'a>(row: &'a Value, name: &str) -> Result<&'a str> {
    let value = row[name]
        .as_str()
        .with_context(|| format!("{name}: requires a nonempty string"))?;
    ensure!(
        !value.trim().is_empty(),
        "{name}: requires a nonempty string"
    );
    Ok(value)
}

/// Only explicitly pre-windowed input is supported. No origin inference,
/// interpolation, clipping or undocumented interval tolerance is introduced.
pub fn reconstruction_window(row: &Record, policy: Option<&Value>) -> Result<Option<Record>> {
    let assay = text(row, "assay")?;
    let origin = if assay == "ymaze" {
        "first_detection_s"
    } else {
        "analysis_start_s"
    };
    let mut expected = json!({"interval":"half_open","origin":origin,
        "duration":"analysis_duration_s","out_of_window":"reject"});
    if assay == "ymaze" {
        expected["missing_pose"] = "observed_samples_break_on_missing_no_interpolation".into();
    } else {
        expected["protocol_departures"] = "flag_only".into();
    }
    if policy != Some(&expected) || row[origin].is_null() || row["analysis_duration_s"].is_null() {
        return Ok(None);
    }
    let start = nonnegative(&row[origin], origin, None)?;
    let duration = nonnegative(&row["analysis_duration_s"], "analysis_duration_s", None)?;
    ensure!(
        duration > 0.0 && (start + duration).is_finite(),
        "Reconstruction requires a finite positive session window"
    );
    Ok(Some(
        json!({"start_s":start,"end_s":start+duration,"policy":expected}),
    ))
}

pub fn validate_window(items: &[Record], window: &Record, intervals: bool) -> Result<()> {
    let window_start = nonnegative(&window["start_s"], "window start", None)?;
    let window_end = nonnegative(&window["end_s"], "window end", None)?;
    ensure!(window_end > window_start, "Invalid session window");
    for item in items {
        let start = nonnegative(
            &item[if intervals { "start_s" } else { "timestamp_s" }],
            "window timestamp",
            None,
        )?;
        let end = if intervals {
            nonnegative(&item["end_s"], "window end", None)?
        } else {
            start
        };
        ensure!(
            start >= window_start && start < window_end && end <= window_end,
            "Timestamp/interval outside declared half-open session window"
        );
        ensure!(!intervals || end > start, "Invalid candidate interval");
    }
    Ok(())
}
