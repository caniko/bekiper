//! Atomic publication of recorded descriptive cohorts; no scoring or models.
use super::*;

pub fn publish_descriptive_with_cancel(
    rows: Vec<Record>,
    omissions: Vec<Record>,
    fingerprints: Vec<Record>,
    mut config: Record,
    output: &Path,
    cancelled: &(dyn Fn() -> bool + Sync),
) -> Result<Record> {
    let mut publication = publication::Publication::claim(output)?;
    let mut stage = "descriptive_populations";
    let result = (|| -> Result<Record> {
        check_cancel(cancelled)?;
        let directory = publication.directory();
        // No plan states the geometry: the trials' recorded provenance does.
        crate::inputs::inherit_threshold_geometry(&mut config, &rows)?;
        crate::inputs::resolve_primary_runs(&mut config, &RunOptions::default())?;
        if primary_runs::requested(&config) {
            stage = "primary_contexts";
            return primary_runs::run(
                directory,
                &config,
                &rows,
                fingerprints,
                Path::new("recorded-config.toml"),
                Some(omissions),
                cancelled,
            );
        }
        let store = families::take(&mut config);
        let workers = config["execution"]["render_workers"].as_u64();
        let report_requested = config["outputs"]["report"]["enabled"] == true;
        if report_requested {
            let mut formats = config["outputs"]["figure_formats"]
                .as_array()
                .cloned()
                .unwrap_or_else(|| vec![json!("png")]);
            if !formats.contains(&json!("pdf")) {
                formats.push(json!("pdf"));
            }
            config["outputs"]["figure_formats"] = formats.into();
        }
        let mut bundle = crate::plotting::build_descriptive_bundle(&rows, &config)?;
        write_json(&directory.join("plotting_bundle.json"), &bundle)?;
        stage = "descriptive_rendering";
        let Value::Array(mut figures) = bundle["figures"].take() else {
            anyhow::bail!("descriptive figures");
        };
        let renderer = outputs::render_figures(
            directory,
            &bundle,
            &mut figures,
            store.as_deref(),
            workers,
            cancelled,
        )?;
        check_cancel(cancelled)?;
        let mut ledger = omissions;
        ledger.extend(
            bundle["eligibility_ledger"]
                .as_array()
                .context("descriptive ledger")?
                .iter()
                .cloned(),
        );
        for (name, value) in [
            ("metrics", json!(rows)),
            ("inclusion_ledger", json!(ledger)),
            ("plotting_data", bundle["plotting_data"].clone()),
            ("summaries", bundle["summaries"].clone()),
        ] {
            write_json(&directory.join(format!("{name}.json")), &value)?;
        }
        let missing = ledger
            .iter()
            .filter(|entry| {
                entry["reason"]
                    .as_str()
                    .is_some_and(|reason| !reason.starts_with("excluded:"))
            })
            .count();
        let unavailable = bundle["unrendered_figures"]
            .as_array()
            .context("descriptive figure inventory")?
            .iter()
            .filter(|entry| entry["status"] == "unavailable")
            .count();
        let figure_missing = figures
            .iter()
            .flat_map(|f| f["panels"].as_array().into_iter().flatten())
            .any(|p| p["status"] != "rendered");
        let figures_requested = config["outputs"]["figures_enabled"] != false
            && config["outputs"]["figure_selection"]
                .as_array()
                .is_none_or(|s| !s.is_empty());
        let partial = missing > 0 || (figures_requested && (unavailable > 0 || figure_missing));
        let mut requested = vec!["metrics"];
        if figures_requested {
            requested.push("figures");
        }
        if report_requested {
            requested.extend(["report", "report_pdf"]);
        }
        let mut manifest = json!({"schema_version":SCHEMA_VERSION,"status":if partial {"partial"} else {"complete"},
            "run_mode":"descriptive","analysis_mode":"recorded_descriptive","scientific_results_recomputed":false,
            "scoring_provenance_by_analysis":bundle["scoring_provenance_by_analysis"],
            "inputs":fingerprints,"resolved_config":config,"code":crate::code_identity(),
            "requested_outputs":requested,"output_statuses":{"metrics":{"status":if rows.is_empty() {"unavailable"} else if missing > 0 {"partial"} else {"complete"},"omission_count":missing},
                "figures":{"status":if !figures_requested {"not_requested"} else if figures.is_empty() {"unavailable"} else if unavailable > 0 || figure_missing {"partial"} else {"complete"},"pages":figures.len(),"unavailable_families":unavailable}},
            "dependencies":{"renderer":renderer},"figures":figures,"unrendered_figures":bundle["unrendered_figures"],
            "limitations":["Recorded metadata populations; no registered inference, primary threshold, subgroup, uncertainty or new QC policy selected."],
            "report":{"status":"not_requested","pdf_status":"not_requested"}});
        if report_requested {
            stage = "descriptive_report";
            manifest["report"] = crate::report::write_report_with_cancel(
                directory,
                &manifest,
                &rows,
                &[],
                &ledger,
                &figures,
                &config,
                cancelled,
            )?;
            manifest["output_statuses"]["report"] = json!({"status":"complete"});
            manifest["output_statuses"]["report_pdf"] = json!({"status":"complete"});
        }
        stage = "descriptive_exports";
        crate::export::finalize(directory, &mut manifest, &rows)?;
        check_cancel(cancelled)?;
        write_json(&directory.join("run_manifest.json"), &manifest)?;
        write_artifact_manifest(directory)?;
        verify_artifacts(directory)?;
        Ok(manifest)
    })();
    match result {
        Ok(manifest) => {
            if let Err(error) = check_cancel(cancelled).and_then(|()| publication.publish()) {
                let path = publication.fail(
                    if cancelled() {
                        "cancelled"
                    } else {
                        "publication"
                    },
                    &error,
                )?;
                return Err(
                    error.context(format!("failure evidence retained in {}", path.display()))
                );
            }
            Ok(manifest)
        }
        Err(error) => {
            let path = publication.fail(if cancelled() { "cancelled" } else { stage }, &error)?;
            Err(error.context(format!("failure evidence retained in {}", path.display())))
        }
    }
}
