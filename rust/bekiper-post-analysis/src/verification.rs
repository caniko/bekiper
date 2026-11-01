//! Read-only reconciliation of published primary contexts, policy payloads and render artifacts.
use crate::{Record, inputs, pipeline};
use anyhow::{Context, Result, ensure};
use serde_json::json;
use std::{
    collections::BTreeSet,
    fs::File,
    io::{BufReader, Read, Seek, SeekFrom},
    path::Path,
};
mod pdf;
#[cfg(test)]
mod tests;
pub use pdf::review_pdfs;

/// Check every declared registration against the effective plan, without inference or admission.
pub fn verify_plan(path: &Path) -> Result<Record> {
    let plan = inputs::resolved_config(path, &Default::default(), &|_| None)?;
    let mut checked = 0;
    for (decision, record) in plan["decision_resolutions"]
        .as_object()
        .context("decision registrations")?
    {
        for (field, expected) in record["value"].as_object().context("registered values")? {
            ensure!(
                inputs::choice(&plan, field, decision) == Some(expected),
                "unmatched or invalid registration: {decision}:{field}"
            );
            checked += 1;
        }
    }
    if plan["statistics"].get("hcar1_policy_plan").is_some() {
        crate::genotype::policy_plan(&plan)?;
    }
    Ok(
        json!({"schema_version":1,"status":"registered_choices_verified","plan":plan["_adapter_config_fingerprint"],
        "checked_registrations":checked,"analysis_admitted":false,"scientific_readiness":"requires_qualified_inputs_and_execution_seal"}),
    )
}

fn payload(root: &Path, manifest: &Record, name: &str) -> Result<Record> {
    let path = manifest["package_layout"]["paths"][name]
        .as_str()
        .with_context(|| format!("package path: {name}"))?;
    ensure!(
        pipeline::safe_relative_path(path),
        "unsafe payload path: {path}"
    );
    inputs::load_json(&root.join(path))
}

/// Artifact validity and scientific readiness remain separate in the receipt.
pub fn verify_run(root: &Path) -> Result<Record> {
    verify_run_at_source(root, None)
}

pub fn verify_run_at_source(root: &Path, source: Option<&Path>) -> Result<Record> {
    let inventory = pipeline::verify_artifacts(root)?;
    let manifest = inputs::load_json(&root.join("run_manifest.json"))?;
    ensure!(
        manifest["code"].is_object(),
        "run manifest requires code identity"
    );
    ensure!(
        manifest["schema_version"] == crate::contracts::SCHEMA_VERSION
            && matches!(
                manifest["status"].as_str(),
                Some("complete" | "partial" | "blocked")
            ),
        "run manifest has no supported terminal publication"
    );
    if let Some(source) = source {
        let lock = std::fs::read(source.join("Cargo.lock"))?;
        ensure!(
            crate::source_tree::digest(&source.join("rust/bekiper-post-analysis"), &lock)?
                == manifest["code"]["source_sha256"],
            "exported engine source identity differs from supplied checkout"
        );
        ensure!(
            inputs::fingerprint(&source.join("Cargo.lock"))?["sha256"]
                == manifest["code"]["source_lock_sha256"],
            "exported source lock differs from supplied checkout"
        );
    }
    let mut contexts = Vec::new();
    if matches!(
        manifest["publication_kind"].as_str(),
        Some("independent_ymaze_primaries" | "independent_primary_contexts")
    ) {
        // Configured contexts: `execution.contexts`, else the legacy Y-maze
        // thresholds (published as `primary<pct>`, 0% under the cross-line
        // geometry included).
        let execution = &manifest["effective_config"]["execution"];
        let listed = execution["contexts"].as_array().filter(|v| !v.is_empty());
        let configured = listed
            .or_else(|| execution["ymaze_primary_thresholds"].as_array())
            .context("registered primary context list")?;
        let children = manifest["primary_contexts"]
            .as_array()
            .context("primary context inventory")?;
        ensure!(
            children.len() == configured.len() && !children.is_empty(),
            "primary context count differs from effective configuration"
        );
        let mut seen = BTreeSet::new();
        for (child, expected) in children.iter().zip(configured) {
            let path = child["path"].as_str().context("primary context path")?;
            let bound = if listed.is_some() {
                expected["assay"] == child["assay"] && expected["variant"] == child["variant"]
            } else {
                // Legacy: the configured threshold names the context.
                child["threshold_pct"] == *expected
            };
            ensure!(
                pipeline::safe_relative_path(path)
                    && !path.contains('/')
                    && seen.insert(path)
                    && bound
                    && (listed.is_some() || path == format!("primary{}", child["threshold_pct"])),
                "invalid, repeated or misbound primary context"
            );
            let report = verify_context(&root.join(path), &manifest["code"])?;
            ensure!(
                report["status"] == child["status"]
                    && report["run_mode"] == child["run_mode"]
                    && (child["threshold_pct"].is_null()
                        || report["threshold_pct"] == child["threshold_pct"]),
                "primary context differs from parent inventory"
            );
            contexts.push(report);
        }
    } else {
        contexts.push(verify_context(root, &manifest["code"])?);
    }
    ensure!(
        pipeline::verify_artifacts(root)? == inventory,
        "published artifacts changed during verification"
    );
    Ok(
        json!({"schema_version":1,"artifact_status":"verified","analysis_admitted":false,"code":manifest["code"],"verifier_code":crate::code_identity(),
        "scientific_readiness":manifest["scientific_readiness"],"published_status":manifest["status"],
        "artifact_count":inventory["files"].as_array().context("artifact inventory")?.len(),"contexts":contexts}),
    )
}

fn verify_context(root: &Path, code: &Record) -> Result<Record> {
    let inventory = pipeline::verify_artifacts(root)?;
    let manifest = inputs::load_json(&root.join("run_manifest.json"))?;
    ensure!(
        &manifest["code"] == code,
        "primary context code identity differs"
    );
    let mut policies = Vec::new();
    let config = &manifest["resolved_config"];
    let metrics = payload(root, &manifest, "metrics.json")?;
    let metrics = metrics
        .as_array()
        .context("shared scored endpoint records")?;
    crate::export::verify_metrics_export(root, &manifest, metrics)?;
    ensure!(
        crate::genotype::policy_plan(config)?.is_some() == manifest["hcar1_policies"].is_object(),
        "declared policy plan and exported policy inventory disagree"
    );
    if manifest["hcar1_policies"].is_object() {
        ensure!(
            manifest["hcar1_policies"]["primary"] == "A",
            "primary HCAR1 policy must remain A"
        );
        let result = payload(root, &manifest, "genotype_policies.json")?;
        ensure!(
            result["primary_policy"] == "A"
                && result["comparison"]["endpoint_recalculation"] == false,
            "policy export changed the primary reference or endpoint-reuse contract"
        );
        let analyses = result["policies"]
            .as_array()
            .context("HCAR1 policy inventory")?;
        ensure!(
            analyses.len() == 3
                && analyses
                    .iter()
                    .map(|p| p["policy"].as_str())
                    .collect::<Vec<_>>()
                    == [Some("A"), Some("B"), Some("C")],
            "policy export must retain A/B/C in that order"
        );
        ensure!(
            payload(root, &manifest, "models.json")? == analyses[0]["models"],
            "main model payload differs from primary policy A"
        );
        for (name, field) in [
            ("within_group_tests.json", "within_group_tests"),
            ("subgroup_analyses.json", "subgroup_analyses"),
        ] {
            ensure!(
                payload(root, &manifest, name)? == analyses[0][field],
                "main {field} payload differs from primary policy A"
            );
        }
        ensure!(
            metrics
                .iter()
                .all(|row| row.get("analysis_factor_coding").is_none()),
            "shared endpoints contain private policy coding"
        );
        let mut families = BTreeSet::new();
        let comparison_rows = result["comparison"]["rows"]
            .as_array()
            .context("policy comparisons")?;
        ensure!(
            comparison_rows.len()
                == analyses
                    .iter()
                    .map(|a| a["models"].as_array().map_or(0, Vec::len))
                    .sum::<usize>(),
            "policy comparison count differs from retained model attempts"
        );
        let mut comparison_keys = BTreeSet::new();
        for row in comparison_rows {
            ensure!(
                comparison_keys.insert((
                    row["policy"].as_str().context("comparison policy")?,
                    row["model_id"].as_str().context("comparison model")?
                )),
                "repeated policy model comparison"
            );
        }
        for analysis in analyses {
            let policy = analysis["policy"].as_str().context("HCAR1 policy")?;
            let config = &analysis["effective_config"];
            ensure!(
                config["hcar1_policy_context"]["policy"] == policy
                    && config["hcar1_policy_context"]["endpoint_recalculation"] == false,
                "misbound policy effective configuration"
            );
            for section in ["assays", "inclusion"] {
                ensure!(
                    config[section] == manifest["resolved_config"][section],
                    "policy altered shared {section} settings"
                );
            }
            for (field, value) in manifest["resolved_config"]["statistics"]
                .as_object()
                .context("registered statistics")?
            {
                if !["multiplicity", "subgroup_analysis_plan"].contains(&field.as_str()) {
                    ensure!(
                        config["statistics"][field] == *value,
                        "policy altered registered statistical setting {field}"
                    );
                }
            }
            let (coded, _) = crate::genotype::project(metrics, config, policy)?;
            let eligible: Vec<_> = coded
                .iter()
                .filter(|r| r["excluded"] != true && r["genotype_eligibility"]["eligible"] == true)
                .collect();
            let animals: BTreeSet<_> = eligible
                .iter()
                .filter_map(|r| r["animal_id"].as_str())
                .collect();
            let models = analysis["models"]
                .as_array()
                .context("policy model attempts")?;
            verify_attempts(models, metrics, config)?;
            verify_test_families(analysis, config)?;
            for model in models {
                if let Some(family) = model["contrast"]["correction_family"].as_str() {
                    ensure!(
                        family.ends_with(&format!("HCAR1_policy_{policy}")),
                        "contrast family escapes its policy context"
                    );
                    ensure!(
                        model["contrast"]["family_size"].as_u64() == Some(models.len() as u64),
                        "Holm family omits model attempts"
                    );
                }
                let comparison = result["comparison"]["rows"]
                    .as_array()
                    .context("policy comparisons")?
                    .iter()
                    .find(|r| r["policy"] == policy && r["model_id"] == model["model_id"])
                    .context("missing policy model comparison")?;
                for field in [
                    "estimate",
                    "ci_lower",
                    "ci_upper",
                    "p_raw",
                    "log_p_raw",
                    "p_adjusted",
                    "log_p_adjusted",
                ] {
                    ensure!(
                        comparison[field] == model["contrast"][field],
                        "policy comparison differs from fitted {field}"
                    );
                }
                ensure!(
                    comparison["confidence_level"] == model["settings"]["confidence_level"],
                    "comparison confidence differs from registered fit"
                );
            }
            if let Some(family) = inputs::choice(config, "statistics.multiplicity", "D09")
                .and_then(|m| m["families"]["planned_contrasts"].as_str())
            {
                ensure!(
                    families.insert(family.to_string()),
                    "policy multiplicity families are shared"
                );
            }
            policies.push(json!({"policy":policy,"candidate_animals":animals.len(),"model_attempts":models.len(),
                "scope":"source_genotype_candidates_not_endpoint_or_model_eligibility"}));
        }
    }
    let mut png_count = 0;
    let figures = manifest["figures"].as_array().context("figure inventory")?;
    let mut names = BTreeSet::new();
    for figure in figures {
        let name = figure["name"].as_str().context("figure name")?;
        ensure!(
            !name.is_empty()
                && name.bytes().all(|b| b.is_ascii_lowercase()
                    || b.is_ascii_digit()
                    || b == b'_'
                    || b == b'-')
                && names.insert(name),
            "invalid or repeated figure slug"
        );
        if !policies.is_empty() {
            let policy = figure["hcar1_policy"]
                .as_str()
                .context("figure policy label")?;
            ensure!(["A", "B", "C"].contains(&policy), "unknown figure policy");
            if policy != "A" {
                ensure!(
                    name.starts_with(&format!(
                        "supplement_hcar1_{}_",
                        policy.to_ascii_lowercase()
                    )),
                    "unlabelled sensitivity figure"
                );
            }
        }
        for file in figure["files"].as_array().context("figure files")? {
            let path = file.as_str().context("figure artifact path")?;
            ensure!(
                pipeline::safe_relative_path(path)
                    && inventory["files"]
                        .as_array()
                        .context("artifact inventory")?
                        .iter()
                        .any(|r| r["path"] == path),
                "figure file is outside artifact inventory"
            );
            if path.ends_with(".png") {
                verify_png(
                    &root.join(path),
                    figure["dpi"]
                        .as_u64()
                        .or(manifest["resolved_config"]["outputs"]["figure_dpi"].as_u64())
                        .context("declared figure DPI")?,
                )?;
                png_count += 1;
            }
            if path.ends_with(".pdf") {
                verify_pdf(&root.join(path))?;
            }
        }
        let formats = figure["formats"].as_array().context("figure formats")?;
        let files = figure["files"].as_array().context("figure files")?;
        ensure!(
            formats.len() == files.len() && !formats.is_empty(),
            "figure format inventory is incomplete"
        );
        for format in formats {
            let format = format.as_str().context("figure format")?;
            ensure!(
                ["svg", "png", "pdf"].contains(&format)
                    && files
                        .iter()
                        .filter_map(serde_json::Value::as_str)
                        .filter(|p| p.ends_with(&format!("/{name}.{format}")))
                        .count()
                        == 1,
                "figure format is missing, repeated or misbound"
            );
        }
    }
    let report_pdf = inventory["files"]
        .as_array()
        .context("artifact inventory")?
        .iter()
        .any(|r| r["path"] == "report.pdf");
    ensure!(
        manifest["report"]["pdf_status"] != "compiled" || report_pdf,
        "compiled PDF missing from artifact inventory"
    );
    if report_pdf {
        verify_pdf(&root.join("report.pdf"))?;
    }
    Ok(
        json!({"status":manifest["status"],"run_mode":manifest["run_mode"],
        "threshold_pct":manifest["resolved_config"]["assays"]["ymaze"]["primary_threshold_arm_length_pct"],
        "artifact_count":inventory["files"].as_array().context("artifact inventory")?.len(),"png_count":png_count,"report_pdf":report_pdf,"policies":policies}),
    )
}

/// Decode every PNG row and check both physical-resolution axes against the captured DPI.
pub fn verify_png(path: &Path, dpi: u64) -> Result<()> {
    let mut reader = png::Decoder::new(BufReader::new(File::open(path)?)).read_info()?;
    let info = reader.info();
    ensure!(info.width > 0 && info.height > 0, "empty PNG dimensions");
    let physical = info
        .pixel_dims
        .context("PNG has no physical-resolution metadata")?;
    ensure!(
        physical.unit == png::Unit::Meter
            && (f64::from(physical.xppu) * 0.0254 - dpi as f64).abs() < 1.0
            && (f64::from(physical.yppu) * 0.0254 - dpi as f64).abs() < 1.0,
        "PNG DPI differs from declared render configuration: {}",
        path.display()
    );
    while reader.next_row()?.is_some() {}
    reader.finish()?;
    Ok(())
}

fn verify_attempts(models: &[Record], metrics: &[Record], config: &Record) -> Result<()> {
    if !crate::inputs::requested_output_ids(config, metrics)?
        .iter()
        .any(|r| r == "models")
    {
        ensure!(models.is_empty(), "unexpected model attempts");
        return Ok(());
    }
    let mut expected = BTreeSet::new();
    for descriptor in bekiper_core::experiment::post_analysis_descriptors() {
        let assay = descriptor
            .post_analysis
            .as_ref()
            .map_or("", |p| p.assay.as_str());
        if metrics.iter().any(|r| r["assay"] == assay)
            || inputs::choice(config, "statistics.model_family_by_endpoint", "D09")
                .and_then(serde_json::Value::as_object)
                .is_some_and(|families| {
                    families.keys().any(|k| k.starts_with(&format!("{assay}.")))
                })
        {
            expected.extend(
                descriptor
                    .modelled()
                    .into_iter()
                    .map(|e| format!("{assay}.{e}")),
            );
        }
    }
    let actual: Vec<_> = models
        .iter()
        .map(|m| m["model_id"].as_str().context("model identity"))
        .collect::<Result<_>>()?;
    ensure!(
        actual.len() == actual.iter().collect::<BTreeSet<_>>().len()
            && actual
                .into_iter()
                .map(str::to_owned)
                .collect::<BTreeSet<_>>()
                == expected,
        "registered model attempts are missing, repeated or unexpected"
    );
    for model in models {
        let levels = config["inputs"]["factor_levels"]["hcar1"]
            .as_array()
            .context("HCAR1 factor levels")?;
        let effects = model["genotype_lactate_effects"]
            .as_array()
            .context("registered genotype simple-effect attempts")?;
        ensure!(
            effects.len() == levels.len()
                && levels.iter().all(|level| effects
                    .iter()
                    .filter(|e| e["hcar1"] == *level)
                    .count()
                    == 1),
            "genotype simple-effect attempts are missing, repeated or unexpected"
        );
    }
    if let Some(settings) = inputs::choice(config, "statistics.multiplicity", "D09") {
        for (key, attempts) in [
            (
                "planned_contrasts",
                models
                    .iter()
                    .map(|m| m["contrast"].clone())
                    .collect::<Vec<_>>(),
            ),
            (
                "genotype_simple_effects",
                models
                    .iter()
                    .flat_map(|m| {
                        m["genotype_lactate_effects"]
                            .as_array()
                            .into_iter()
                            .flatten()
                    })
                    .cloned()
                    .collect(),
            ),
        ] {
            let Some(family) = settings["families"][key].as_str().filter(|f| !f.is_empty()) else {
                continue;
            };
            verify_family(
                &attempts,
                settings["method"]
                    .as_str()
                    .context("registered multiplicity method")?,
                family,
                key,
            )?;
        }
    }
    Ok(())
}

/// Recompute one published correction family from its retained raw attempts.
fn verify_family(attempts: &[Record], method: &str, family: &str, key: &str) -> Result<()> {
    let mut adjusted = attempts.to_vec();
    crate::statistics::adjust_family(&mut adjusted, method, family)?;
    for (actual, expected) in attempts.iter().zip(&adjusted) {
        for field in [
            "p_adjusted",
            "log_p_adjusted",
            "family_size",
            "correction_family",
        ] {
            ensure!(
                actual[field] == expected[field],
                "{key} correction differs from complete registered family: {field}"
            );
        }
    }
    Ok(())
}

/// The NORT preference and exploratory subgroup families, flattened in the
/// order the pipeline corrected them.
fn verify_test_families(analysis: &Record, config: &Record) -> Result<()> {
    let tests = analysis["within_group_tests"]
        .as_array()
        .context("NORT within-group attempts")?;
    if let Some(settings) = inputs::choice(config, "statistics.multiplicity", "D09")
        && settings["method"] == "holm"
        && let Some(family) = settings["families"]["nort_within_group"]
            .as_str()
            .filter(|f| !f.is_empty())
    {
        verify_family(tests, "holm", family, "nort_within_group")?;
    }
    let subgroups = analysis["subgroup_analyses"]
        .as_array()
        .context("subgroup attempts")?;
    if let Some(plan) = inputs::validated_subgroup_plan(config)?
        && !subgroups.is_empty()
    {
        let mut attempts = Vec::new();
        for item in subgroups {
            for kind in ["models", "correlations", "within_group_tests"] {
                for record in item[kind].as_array().context("subgroup attempts")? {
                    attempts.push(if kind == "models" {
                        record["contrast"].clone()
                    } else {
                        record.clone()
                    });
                }
            }
        }
        verify_family(&attempts, "holm", plan.family, "subgroup")?;
    }
    Ok(())
}

fn verify_pdf(path: &Path) -> Result<()> {
    let mut file = File::open(path)?;
    let mut header = [0; 5];
    file.read_exact(&mut header)?;
    ensure!(&header == b"%PDF-", "invalid report PDF header");
    let bytes = file.metadata()?.len();
    file.seek(SeekFrom::Start(bytes.saturating_sub(1024)))?;
    let mut tail = Vec::new();
    file.read_to_end(&mut tail)?;
    ensure!(
        tail.windows(5).any(|w| w == b"%%EOF"),
        "incomplete report PDF trailer"
    );
    Ok(())
}
