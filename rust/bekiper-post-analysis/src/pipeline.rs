//! Native orchestration. Scientific computations and registered populations are
//! shared by standalone analysis and project-level cohort exports.
use crate::{
    Record, RunOptions,
    contracts::{KEY, OFT_PRIMARY, OFT_SECONDARY, SCHEMA_VERSION, YM_SCORES},
    inputs::{choice, fingerprint},
    statistics,
};
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
    process::Command,
};

#[path = "pipeline/archive.rs"]
mod archive;
#[path = "pipeline/child.rs"]
mod child;
pub use child::run_child;
#[path = "pipeline/publication.rs"]
pub(crate) mod publication;
pub use archive::{ProjectArtifactSnapshot, verify_data_archive, verify_project_exports};
pub use publication::file_fingerprint as publication_fingerprint;
pub use publication::{
    ProjectDataLease, Publication, artifact_inventory, safe_relative_path, verify_artifacts,
    write_artifact_manifest, write_json,
};
#[path = "pipeline/descriptive.rs"]
mod descriptive;
pub use descriptive::publish_descriptive_with_cancel;

#[derive(Default, Clone)]
pub struct InferenceResults {
    pub models: Vec<Record>,
    pub pairing_ledger: Vec<Record>,
    pub correlations: Vec<Record>,
    pub within_group_tests: Vec<Record>,
}

#[path = "pipeline/genotypes.rs"]
pub mod genotypes;

pub fn run(config_path: &Path, options: &RunOptions) -> Result<Record> {
    let config =
        crate::inputs::resolved_config(config_path, options, &|key| std::env::var(key).ok())?;
    let (rows, fingerprints) =
        crate::inputs::load_inputs(&config, config_path.parent().unwrap_or(Path::new(".")))?;
    // The file adapter has already applied CLI/environment precedence. Do not
    // apply the same override twice or replace its configured-policy evidence.
    let options = RunOptions {
        ymaze_dwell: Default::default(),
        ..options.clone()
    };
    run_records(config, rows, fingerprints, config_path, &options)
}

pub fn run_records(
    config: Record,
    rows: Vec<Record>,
    fingerprints: Vec<Record>,
    config_path: &Path,
    options: &RunOptions,
) -> Result<Record> {
    run_records_with_cancel(config, rows, fingerprints, config_path, options, &|| false)
}

/// Cancellation is checked between every stage and after non-abortable child
/// work. The caller retains its memory lease until this function returns.
pub fn run_records_with_cancel(
    mut config: Record,
    rows: Vec<Record>,
    fingerprints: Vec<Record>,
    config_path: &Path,
    options: &RunOptions,
    cancelled: &(dyn Fn() -> bool + Sync),
) -> Result<Record> {
    ensure!(
        config.is_object(),
        "analysis configuration must be an object"
    );
    crate::inputs::resolve_ymaze_dwell(&mut config, options.ymaze_dwell)?;
    crate::inputs::resolve_primary_runs(&mut config, options)?;
    if !config["outputs"].is_object() {
        config["outputs"] = json!({});
    }
    if let Some(output) = &options.output {
        config["outputs"]["directory"] = output.to_string_lossy().into_owned().into();
    }
    if let Some(requests) = &options.requested_outputs {
        config["outputs"]["requested_outputs"] = json!(requests);
    }
    if let Some(thresholds) = &options.sensitivity_thresholds {
        config["assays"]["ymaze"]["requested_sensitivity_thresholds_arm_length_pct"] =
            json!(thresholds);
    }
    if !config["assays"]["ymaze"]["requested_sensitivity_thresholds_arm_length_pct"].is_null() {
        crate::inputs::validate_thresholds_in(
            &config["assays"]["ymaze"]["requested_sensitivity_thresholds_arm_length_pct"],
            crate::inputs::allowed_thresholds(&config),
        )?;
    }
    let configured = config["outputs"]["directory"]
        .as_str()
        .unwrap_or("post_analysis_output");
    let configured = Path::new(configured);
    let output = if configured.is_absolute() {
        configured.to_path_buf()
    } else {
        config_path
            .parent()
            .unwrap_or(Path::new("."))
            .join(configured)
    };
    let mut publication = publication::Publication::claim(&output)?;
    let mut stage = "validation";
    let result = execute(
        &mut config,
        &rows,
        fingerprints,
        config_path,
        publication.directory(),
        &mut stage,
        cancelled,
    );
    match result {
        Ok(manifest) => {
            if cancelled() {
                let error = anyhow::anyhow!("post-analysis cancelled before publication");
                let evidence = publication.fail("cancelled", &error)?;
                return Err(error.context(format!(
                    "failure evidence retained in {}",
                    evidence.display()
                )));
            }
            if let Err(error) = publication.publish() {
                let evidence = publication.fail("publication", &error)?;
                return Err(error.context(format!(
                    "failure evidence retained in {}",
                    evidence.display()
                )));
            }
            Ok(manifest)
        }
        Err(error) => {
            let failed_stage = if cancelled() { "cancelled" } else { stage };
            match publication.fail(failed_stage, &error) {
                Ok(evidence) => Err(error.context(format!(
                    "failure evidence retained in {}",
                    evidence.display()
                ))),
                Err(retention) => {
                    Err(error.context(format!("failed to retain failure manifest: {retention:#}")))
                }
            }
        }
    }
}

fn check_cancel(cancelled: &dyn Fn() -> bool) -> Result<()> {
    ensure!(!cancelled(), "post-analysis cancelled");
    Ok(())
}

fn execute(
    config: &mut Record,
    rows: &[Record],
    mut fingerprints: Vec<Record>,
    config_path: &Path,
    directory: &Path,
    stage: &mut &'static str,
    cancelled: &(dyn Fn() -> bool + Sync),
) -> Result<Record> {
    check_cancel(cancelled)?;
    if primary_runs::requested(config) {
        *stage = "primary_contexts";
        return primary_runs::run(
            directory,
            config,
            rows,
            fingerprints,
            config_path,
            None,
            cancelled,
        );
    }
    let store = families::take(config);
    let workers = config["execution"]["render_workers"].as_u64();
    ensure!(
        ["synthetic", "development", "production"]
            .contains(&config["run_mode"].as_str().unwrap_or("")),
        "invalid run_mode"
    );
    ensure!(
        config["run_mode"] != "production"
            || !rows.iter().any(|row| row["synthetic_observation"] == true),
        "synthetic observations cannot be a production execution"
    );
    for row in rows {
        crate::inputs::session_key(row)?;
    }
    let native_comparisons: Vec<Record> = rows
        .iter()
        .flat_map(|row| row["native_comparisons"].as_array().into_iter().flatten())
        .cloned()
        .collect();
    let requested = crate::inputs::requested_output_ids(config, rows)?;
    config["outputs"]["requested_outputs"] = json!(requested);
    let (recipe, amendment) = reference_fingerprints(config, config_path, directory)?;
    if config_path.is_file()
        && !fingerprints
            .iter()
            .any(|item| item["role"] == "post_analysis_plan")
    {
        let mut config_fingerprint = fingerprint(config_path)?;
        config_fingerprint["role"] = "post_analysis_plan".into();
        fingerprints.push(config_fingerprint);
    }
    *stage = "scoring";
    let (scored, debug) = crate::metrics::score_records(rows, config)?;
    check_cancel(cancelled)?;
    *stage = "inference";
    let policies = if crate::genotype::policy_plan(config)?.is_some() {
        genotypes::run_policies_with_cancel(&scored, config, cancelled)?
    } else {
        Vec::new()
    };
    check_cancel(cancelled)?;
    let inference = if let Some(primary) = policies.first() {
        primary.inference.clone()
    } else {
        run_models(&scored, config)?
    };
    let (subgroups, subgroup_status) = if let Some(primary) = policies.first() {
        (primary.subgroups.clone(), primary.subgroup_status.clone())
    } else if requested
        .iter()
        .any(|output| output == "subgroup_inference")
    {
        run_subgroups(&scored, config)?
    } else {
        (Vec::new(), Value::Null)
    };
    let ledger = inclusion_ledger(&scored);
    check_cancel(cancelled)?;
    let (report_requested, pdf_requested) = report_request(config, &requested);
    *stage = "plotting_bundle";
    let mut bundle = if policies.is_empty() {
        prepare_plotting_bundle(&scored, config, &requested, pdf_requested)?
    } else {
        genotypes::plotting_bundle(&policies, &requested, pdf_requested)?
    };
    write_json(&directory.join("plotting_bundle.json"), &bundle)?;
    let Value::Array(mut figures) = bundle["figures"].take() else {
        anyhow::bail!("plotting bundle needs figures");
    };
    *stage = "renderer";
    let renderer = render_figures(
        directory,
        &bundle,
        &mut figures,
        store.as_deref(),
        workers,
        cancelled,
    )?;
    check_cancel(cancelled)?;
    let OutputAccounting {
        statuses,
        mut supplementary,
        missing_count,
    } = account_outputs(
        &requested,
        &scored,
        &inference,
        &ledger,
        &figures,
        subgroup_status,
    );
    if !policies.is_empty() {
        supplementary.push(json!({"output":"hcar1_policy_sensitivity","primary":"A","policies":policies.iter().map(|p|json!({
            "policy":p.policy,"models":result_status(&p.inference.models,"fitted"),"subgroups":p.subgroup_status,
            "rank_deficient_contrasts_retained":true})).collect::<Vec<_>>() }));
    }
    let decisions: Vec<_> = (1..=14).map(|number| {
        let id = format!("D{number:02}");
        json!({"decision_id":id,"resolution":config["decision_resolutions"][&id],"status":if config["decision_resolutions"][&id].is_null() { "unresolved" } else { "recorded_dependency_scoped" }})
    }).collect();
    let limitations: BTreeSet<String> = inference
        .models
        .iter()
        .chain(&inference.within_group_tests)
        .chain(&inference.correlations)
        .chain(&supplementary)
        .filter_map(|item| item.get("reason"))
        .filter(|reason| !reason.is_null())
        .map(Value::to_string)
        .collect();
    let mut public_config = config.clone();
    sanitize_config(&mut public_config, &fingerprints);
    let mut manifest = json!({"schema_version":SCHEMA_VERSION,"status":overall_status(&statuses),"run_mode":config["run_mode"],
        "synthetic_fixture":config["synthetic_fixture"],"requested_outputs":requested,"output_statuses":statuses,
        "source_recipe":recipe,"ymaze_amendment":amendment,"inputs":fingerprints,"resolved_config":public_config,
        "decision_ids":config["decision_resolutions"].as_object().map(|items| items.keys().collect::<Vec<_>>()).unwrap_or_default(),
        "code":crate::code_identity(),"dependencies":{"engine":env!("CARGO_PKG_VERSION"),"statistics":"statrs 0.18.0","linear_algebra":"nalgebra 0.32.6","renderer":renderer},
        "random_seed":config["outputs"]["random_seed"],"missing_metric_count":missing_count,"limitations":limitations,
        "input_verification":scored.iter().map(|row| { let mut item=key_fields(row); item["threshold_pct"]=row["threshold_pct"].clone(); item.as_object_mut().unwrap().extend(row["input_verification"].as_object().cloned().unwrap_or_default()); item }).collect::<Vec<_>>(),
        "figures":figures,"unrendered_figures":bundle["unrendered_figures"],"supplementary_output_statuses":supplementary,
        "real_data_execution":config["run_mode"] == "production" && scored.iter().any(|row| row["measurement_available"] == true) && !scored.iter().any(|row| row["synthetic_observation"] == true),
        "scientific_readiness":if config["run_mode"] == "synthetic" { "fixture_only" } else { "see_dependency_scoped_decision_report" },
        "engineering_extensions":["D02 nonnegative triad-window safeguard","D13 undefined zero training biases"]});
    if !native_comparisons.is_empty() {
        manifest["native_comparisons"] = json!({"path":crate::export::relative_path("native_comparisons.json"),
            "count":native_comparisons.len(),"routing":"native_comparison_only",
            "principal_inference":false});
        write_json(
            &directory.join("native_comparisons.json"),
            &json!(native_comparisons),
        )?;
    }
    if !policies.is_empty() {
        manifest["hcar1_policies"] = json!({"primary":"A","path":crate::export::relative_path("genotype_policies.json"),"comparison":genotypes::comparison(&policies)});
        let mut record = genotypes::export_record(&policies);
        for policy in record["policies"]
            .as_array_mut()
            .context("Policy exports")?
        {
            sanitize_config(&mut policy["effective_config"], &fingerprints);
        }
        write_json(&directory.join("genotype_policies.json"), &record)?;
    }
    for (name, value) in [
        ("metrics", json!(scored)),
        ("event_bout_debug", json!(debug)),
        ("inclusion_ledger", json!(ledger)),
        ("models", json!(inference.models)),
        ("pairing_ledger", json!(inference.pairing_ledger)),
        ("correlations", json!(inference.correlations)),
        ("within_group_tests", json!(inference.within_group_tests)),
        ("subgroup_analyses", json!(subgroups)),
        ("plotting_data", bundle["plotting_data"].clone()),
        ("summaries", bundle["summaries"].clone()),
        ("decision_report", json!(decisions)),
        (
            "supplementary_statuses",
            manifest["supplementary_output_statuses"].clone(),
        ),
    ] {
        write_json(&directory.join(format!("{name}.json")), &value)?;
    }
    *stage = "report";
    if report_requested {
        let report = crate::report::write_report(
            directory,
            &manifest,
            &scored,
            &inference.models,
            &ledger,
            &figures,
            config,
        )?;
        if pdf_requested {
            ensure!(
                directory
                    .join("report.pdf")
                    .metadata()
                    .is_ok_and(|metadata| metadata.is_file() && metadata.len() > 0),
                "requested report PDF was not compiled"
            );
        }
        manifest["report"] = report;
    } else {
        manifest["report"] = json!({"status":"not_requested","pdf_status":"not_requested"});
    }
    check_cancel(cancelled)?;
    *stage = "human_exports";
    crate::export::finalize(directory, &mut manifest, &scored)?;
    check_cancel(cancelled)?;
    *stage = "manifest";
    write_json(&directory.join("run_manifest.json"), &manifest)?;
    write_artifact_manifest(directory)?;
    verify_artifacts(directory)?;
    Ok(manifest)
}

#[derive(Default)]
struct Scope {
    assays: BTreeSet<String>,
    baseline: BTreeSet<String>,
}

fn scope_for(rows: &[Record], config: &Record) -> Scope {
    let mut scope = Scope::default();
    for row in rows {
        if let Some(assay) = row["assay"].as_str() {
            scope.assays.insert(assay.into());
            if row["timepoint"] == "pre" && assay != "nort" {
                scope.baseline.insert(assay.into());
            }
        }
    }
    // Registered model families declare hypotheses even when their assay has
    // no observed member. This is essential for stable correction families.
    if let Some(families) =
        choice(config, "statistics.model_family_by_endpoint", "D09").and_then(Value::as_object)
    {
        for key in families.keys() {
            if let Some((assay, _)) = key.split_once('.')
                && ["oft", "ymaze", "nort"].contains(&assay)
            {
                scope.assays.insert(assay.into());
            }
        }
    }
    if choice(config, "statistics.oft_baseline_strategy", "D09")
        .is_some_and(|value| value == "baseline_adjusted")
    {
        scope.baseline.insert("oft".into());
    }
    if choice(config, "statistics.nort_within_group_checks", "D09").is_some() {
        scope.assays.insert("nort".into());
    }
    if config["outputs"]["requested_outputs"]
        .as_array()
        .is_some_and(|requests| requests.iter().any(|request| request == "correlations"))
    {
        scope.assays.insert("ymaze".into());
    }
    scope
}

pub fn run_models(rows: &[Record], config: &Record) -> Result<InferenceResults> {
    run_models_scoped(rows, config, &scope_for(rows, config))
}

fn failed_choices(config: &Record, requirements: &[(&str, &str, Option<&str>)]) -> Vec<String> {
    requirements
        .iter()
        .filter_map(|(path, decision, supported)| {
            let value = choice(config, path, decision);
            if value.is_none()
                || supported
                    .is_some_and(|supported| value.and_then(Value::as_str) != Some(supported))
            {
                Some(format!("{decision}:{path}"))
            } else {
                None
            }
        })
        .collect()
}

const INCLUSION_REQUIREMENTS: [(&str, &str, Option<&str>); 4] = [
    (
        "inclusion.existing_metadata_exclusions_policy",
        "D12",
        Some("honor_metadata"),
    ),
    (
        "inclusion.qc_flag_exclusion_policy",
        "D12",
        Some("flags_only"),
    ),
    (
        "inclusion.missing_metric_policy",
        "D12",
        Some("endpoint_complete_cases"),
    ),
    ("inclusion.upstream_verification_requirement", "D12", None),
];

fn run_models_scoped(rows: &[Record], config: &Record, scope: &Scope) -> Result<InferenceResults> {
    let requested = crate::inputs::requested_output_ids(config, rows)?;
    let has = |name: &str| requested.iter().any(|output| output == name);
    let mut result = InferenceResults::default();
    if !has("models") && !has("correlations") && !has("within_group_tests") {
        return Ok(result);
    }
    let requirement = choice(config, "inclusion.upstream_verification_requirement", "D12")
        .and_then(Value::as_str);
    ensure!(
        matches!(
            requirement,
            None | Some("allow_unverified" | "require_verified")
        ),
        "unknown upstream verification requirement"
    );
    for assay in &scope.assays {
        let mut subset: Vec<_> = rows
            .iter()
            .filter(|row| {
                row["assay"] == assay.as_str() && (assay != "nort" || row["phase"] == "test")
            })
            .cloned()
            .collect();
        if assay == "ymaze" {
            subset.retain(|row| row["primary_threshold"] == true);
        }
        if has("models") {
            fit_assay_models(
                &subset,
                assay,
                scope.baseline.contains(assay),
                config,
                requirement,
                &mut result,
            )?;
        }
        if assay == "ymaze" && has("correlations") {
            result
                .correlations
                .extend(ymaze_correlations(&subset, config, requirement)?);
        }
        if assay == "nort" && has("within_group_tests") {
            result.within_group_tests.extend(nort_within_group_tests(
                &subset,
                config,
                requirement,
            )?);
        }
    }
    correct_model_family(&mut result.models, config)?;
    Ok(result)
}

struct OutputAccounting {
    statuses: Record,
    supplementary: Vec<Record>,
    missing_count: usize,
}

mod outputs;
use outputs::{account_outputs, prepare_plotting_bundle, render_figures, report_request};

mod provenance;
use provenance::{reference_fingerprints, sanitize_config};

mod ledger;
pub use ledger::{inclusion_ledger, result_status};
use ledger::{key_fields, overall_status};

mod families;
mod models;
mod primary_runs;
use models::fit_assay_models;

mod correlations;
use correlations::ymaze_correlations;

mod checks;
use checks::nort_within_group_tests;

mod subgroups;
use subgroups::correct_model_family;
pub use subgroups::run_subgroups;

#[cfg(test)]
#[path = "pipeline/tests.rs"]
mod tests;
