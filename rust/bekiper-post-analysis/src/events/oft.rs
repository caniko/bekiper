//! Center dwell, tracking gaps and censoring.
use super::*;

pub fn oft_zone_intervals(
    intervals: &[Record],
    first_detection_s: f64,
    occupancy: &str,
    no_entry: &str,
) -> Result<Record> {
    ensure!(
        matches!(occupancy, "all_observed" | "qualified_only")
            && matches!(no_entry, "missing" | "right_censored_300"),
        "D05 requires explicit occupancy and no-entry policy"
    );
    ensure!(
        first_detection_s.is_finite()
            && first_detection_s >= 0.0
            && (first_detection_s + 300.0).is_finite(),
        "Invalid first_detection_s"
    );
    let mut prior_end = first_detection_s;
    let mut runs: Vec<Value> = Vec::new();
    for (index, interval) in intervals.iter().enumerate() {
        ensure!(interval.is_object(), "OFT interval must be an object");
        let start = nonnegative(&interval["start_s"], "start_s", None)?;
        let end = nonnegative(&interval["end_s"], "end_s", None)?;
        ensure!(
            end > start
                && start >= prior_end
                && start >= first_detection_s
                && end <= first_detection_s + 300.0,
            "Invalid/overlapping OFT interval or interval outside first-detection 300s window"
        );
        ensure!(
            matches!(interval["zone"].as_str(), Some("center" | "periphery")),
            "Unknown OFT zone"
        );
        prior_end = end;
        if let Some(run) = runs
            .last_mut()
            .filter(|run| run["zone"] == interval["zone"] && run["end_s"].as_f64() == Some(start))
        {
            run["end_s"] = json!(end);
            run["component_indices"]
                .as_array_mut()
                .context("interval indices")?
                .push(json!(index));
            run["components"]
                .as_array_mut()
                .context("interval components")?
                .push(interval.clone());
        } else {
            let mut run = interval.clone();
            run["component_indices"] = json!([index]);
            run["components"] = json!([interval]);
            runs.push(run);
        }
    }
    let mut center_time = 0.0;
    let mut observed = 0.0;
    let mut entries = 0;
    let mut latency = None;
    for run in &mut runs {
        let duration = finite(&run["end_s"], "end_s")? - finite(&run["start_s"], "start_s")?;
        let qualified = duration >= 1.0;
        run["duration_s"] = json!(duration);
        run["valid_entry"] = json!(qualified);
        run["rejection_reason"] = if qualified {
            Value::Null
        } else {
            "short_dwell".into()
        };
        observed += duration;
        if run["zone"] == "center" {
            if occupancy == "all_observed" || qualified {
                center_time += duration;
            }
            if qualified {
                entries += 1;
                if latency.is_none() {
                    latency = Some(finite(&run["start_s"], "start_s")? - first_detection_s);
                }
            }
        }
    }
    let mut result = json!({"center_time_s":center_time,"center_entries":entries,"latency_to_first_center_entry_s":latency,
        "event_debug":runs,"observed_zone_time_s":observed,"latency_right_censored":false});
    if latency.is_none() {
        if no_entry == "right_censored_300" {
            result["latency_to_first_center_entry_s"] = json!(300.0);
            result["latency_right_censored"] = json!(true);
        } else {
            result["missing_reasons"] =
                json!({"latency_to_first_center_entry_s":"not_observed:no_center_entry"});
        }
    }
    Ok(result)
}
