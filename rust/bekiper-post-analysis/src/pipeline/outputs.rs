//! Rendering requests and dependency-scoped output accounting.
use super::*;

/// The pending bundle borrows the context's bundle: a re-read or clone of a
/// multi-gigabyte bundle per context is what bounds concurrent contexts.
#[derive(serde::Serialize)]
struct PendingBundle<'a> {
    #[serde(flatten)]
    rest: BTreeMap<&'a str, &'a Value>,
    figures: Vec<&'a Record>,
}

/// `bundle` is the written `plotting_bundle.json` (its `figures` may have been
/// taken out into `figures`).
pub(super) fn render_figures(
    directory: &Path,
    bundle: &Record,
    figures: &mut [Record],
    store: Option<&Path>,
    workers: Option<u64>,
    cancelled: &dyn Fn() -> bool,
) -> Result<Record> {
    let mut renderer = if figures.is_empty() {
        Value::Null
    } else if let Some(store) = store {
        // Pages another context already rendered are linked, not redrawn.
        let lookup = families::restore(store, directory, figures)?;
        let renderer = if lookup.misses.is_empty() {
            families::renderer_record(store)
        } else {
            let pending = PendingBundle {
                rest: bundle
                    .as_object()
                    .context("plotting bundle must be an object")?
                    .iter()
                    .filter(|(key, _)| *key != "figures")
                    .map(|(key, value)| (key.as_str(), value))
                    .collect(),
                figures: lookup.misses.iter().map(|&i| &figures[i]).collect(),
            };
            let name = "plotting_bundle.pending.json";
            write_json(&directory.join(name), &pending)?;
            let renderer =
                provenance::render_bundle_with_cancel(directory, name, workers, cancelled)?;
            std::fs::remove_file(directory.join(name))?;
            families::publish(&lookup, directory, figures, &renderer, store)?;
            renderer
        };
        renderer
    } else {
        provenance::render_bundle_with_cancel(
            directory,
            "plotting_bundle.json",
            workers,
            cancelled,
        )?
    };
    for figure in &mut *figures {
        if figure["status"] == "complete" {
            figure["status"] = "rendered".into(); // coverage: unreachable — plotting::figure labels pages rendered or partial, never complete
        }
        let name = figure["name"].as_str().context("figure needs name")?;
        ensure!(
            safe_relative_path(name) && !name.contains('/'),
            "unsafe figure name"
        );
        for extension in figure["formats"]
            .as_array()
            .context("figure needs formats")?
        {
            let extension = extension.as_str().context("figure format must be string")?;
            ensure!(
                ["svg", "png", "pdf"].contains(&extension),
                "unsupported figure format"
            );
            ensure!(
                directory
                    .join(format!("{name}.{extension}"))
                    .metadata()
                    .is_ok_and(|metadata| metadata.is_file() && metadata.len() > 0),
                "renderer did not produce selected figure {name}.{extension}"
            );
        }
    }
    crate::export::organize_figures(directory, figures)?;
    if renderer.is_object() {
        renderer["files"] = json!(
            figures
                .iter()
                .flat_map(|figure| figure["files"].as_array().into_iter().flatten().cloned())
                .collect::<Vec<_>>()
        );
    }
    Ok(renderer)
}

pub(super) fn report_request(config: &mut Record, requested: &[String]) -> (bool, bool) {
    let report = requested
        .iter()
        .any(|output| matches!(output.as_str(), "report" | "report_pdf"));
    let explicit_pdf = requested.iter().any(|output| output == "report_pdf");
    let pdf = explicit_pdf
        || (report
            && config["outputs"]["report"]["enabled"] == true
            && config["outputs"]["report"]["compiler"] != "none");
    if explicit_pdf {
        config["outputs"]["report"]["enabled"] = true.into();
    }
    (report, pdf)
}

pub(super) fn prepare_plotting_bundle(
    scored: &[Record],
    config: &mut Record,
    requested: &[String],
    pdf_requested: bool,
) -> Result<Record> {
    if !requested.iter().any(|output| output == "figures") {
        return crate::plotting::disabled_bundle(config);
    }
    if pdf_requested {
        let mut formats = config["outputs"]["figure_formats"]
            .as_array()
            .cloned()
            .unwrap_or_else(|| vec![json!("svg")]);
        if !formats.iter().any(|format| format == "pdf") {
            formats.push(json!("pdf"));
        }
        config["outputs"]["figure_formats"] = formats.into();
    }
    crate::plotting::build_bundle(scored, config)
}

pub(super) fn account_outputs(
    requested: &[String],
    scored: &[Record],
    inference: &InferenceResults,
    ledger: &[Record],
    figures: &[Record],
    subgroup_status: Record,
) -> OutputAccounting {
    let has = |name: &str| requested.iter().any(|output| output == name);
    let missing_count = ledger
        .iter()
        .filter(|row| {
            !matches!(
                row["metric_status"].as_str(),
                Some("available" | "not_applicable")
            )
        })
        .count();
    let mut statuses = json!({});
    if has("metrics") {
        let blocked = scored.iter().any(|row| {
            row["blocked_outputs"]
                .as_array()
                .is_some_and(|items| !items.is_empty())
        });
        let status = if scored.is_empty() {
            "unavailable"
        } else if missing_count > 0 || blocked {
            "partial"
        } else {
            "complete"
        };
        statuses["metrics"] = json!({"status":status,"missing_metric_count":missing_count});
    }
    for (name, attempts, success) in [
        ("models", inference.models.as_slice(), "fitted"),
        (
            "correlations",
            inference.correlations.as_slice(),
            "computed",
        ),
        (
            "within_group_tests",
            inference.within_group_tests.as_slice(),
            "computed",
        ),
        ("figures", figures, "rendered"),
    ] {
        if has(name) {
            statuses[name] = result_status(attempts, success);
        }
    }
    let mut supplementary = Vec::new();
    if has("subgroup_inference") {
        statuses["subgroup_inference"] = subgroup_status.clone();
        supplementary.push(subgroup_status);
    }
    if has("spatial_location_bias") {
        let blocked = json!({"output":"spatial_location_bias","status":"blocked","decision_ids":["D08"],"reason":"source side_bias is algebraic; spatial extension requires a registered object-location definition"});
        statuses["spatial_location_bias"] = blocked.clone();
        supplementary.push(blocked);
    }
    // Reports are published only if compilation succeeds. Failure retains evidence.
    for output in requested {
        if matches!(output.as_str(), "report" | "report_pdf") {
            statuses[output] = json!({"status":"complete"});
        }
    }
    OutputAccounting {
        statuses,
        supplementary,
        missing_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fully_stored_family_is_restored_without_running_the_renderer() {
        let root = tempfile::tempdir().unwrap();
        let (store, first, second) = (
            root.path().join("families"),
            root.path().join("a"),
            root.path().join("b"),
        );
        std::fs::create_dir_all(&first).unwrap();
        std::fs::create_dir_all(&second).unwrap();
        let figures = [json!({"name":"figure_oft_page","formats":["png"],"panels":[{}]})];
        std::fs::write(first.join("figure_oft_page.png"), "page").unwrap();
        let lookup = families::restore(&store, &first, &figures).unwrap();
        families::publish(&lookup, &first, &figures, &json!({"dpi": 150}), &store).unwrap();

        let mut again = figures.to_vec();
        let renderer = render_figures(&second, &json!({}), &mut again, Some(&store), None, &|| {
            false
        })
        .unwrap();
        assert_eq!(renderer["dpi"], 150);
        assert!(renderer["files"].is_array());
    }
}
