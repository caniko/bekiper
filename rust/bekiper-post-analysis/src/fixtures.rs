//! Explicitly synthetic fixtures for installed engine/policy/render qualification.
use crate::{Record, inputs, pipeline};
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::Path};

/// Reuse the registered plan and independent numerical oracle, with fixture-only registrations.
pub fn prepare_hcar1(plan_path: &Path, oracle_path: &Path, output: &Path) -> Result<Record> {
    let plan_identity = inputs::fingerprint(plan_path)?;
    let oracle_identity = inputs::fingerprint(oracle_path)?;
    let mut plan: Record = serde_json::to_value(toml::from_str::<toml::Value>(
        &std::fs::read_to_string(plan_path)?,
    )?)?;
    crate::genotype::policy_plan(&plan)?.context("fixture requires registered A/B/C plan")?;
    let oracle = inputs::load_json(oracle_path)?;
    ensure!(
        oracle["notice"]
            .as_str()
            .is_some_and(|s| s.starts_with("Synthetic")),
        "only a labelled synthetic oracle may seed fixtures"
    );
    let mut animals = oracle["rows"]
        .as_array()
        .context("synthetic oracle rows")?
        .clone();
    for animal in &mut animals {
        let group = plan["inputs"]["group_mapping"]
            .as_object()
            .context("assigned group mapping")?
            .iter()
            .find(|(_, factors)| {
                crate::contracts::FACTORS
                    .iter()
                    .all(|f| animal[*f] == factors[*f])
            })
            .context("synthetic factors need an assigned group")?
            .0;
        animal["group"] = group.clone().into();
    }
    let wt = animals
        .iter()
        .find(|a| a["group"] == "G5")
        .context("synthetic G5 seed")?
        .clone();
    for index in 0..2 {
        let mut row = wt.clone();
        row["animal_id"] = format!("SYNTHETIC_Het_{index}").into();
        row["hcar1"] = "Het".into();
        animals.push(row);
    }
    let mut ambiguous = animals
        .iter()
        .find(|a| a["group"] == "G4")
        .context("synthetic G4 seed")?
        .clone();
    ambiguous["animal_id"] = "37".into();
    ambiguous["hcar1"] = "Het/KO".into();
    animals.push(ambiguous);
    let SyntheticSessions {
        metadata,
        measurements,
    } = measurements(&animals)?;
    plan["run_mode"] = "synthetic".into();
    plan["synthetic_fixture"] = true.into();
    plan["execution"]["primary_workers"] = 2.into();
    plan["execution"]["ymaze_primary_thresholds"] = json!([25, 45, 65]);
    plan["inclusion"]["upstream_verification_requirement"] = "allow_unverified".into();
    plan["decision_resolutions"]["D12"]["value"]["inclusion.upstream_verification_requirement"] =
        "allow_unverified".into();
    plan["outputs"]["requested_outputs"] = json!([
        "metrics",
        "models",
        "correlations",
        "within_group_tests",
        "subgroup_inference",
        "figures",
        "report_pdf"
    ]);
    plan["outputs"]["figure_selection"] = json!(["figure_oft", "figure_ymaze", "figure_nort"]);
    for registration in plan["decision_resolutions"]
        .as_object_mut()
        .context("plan registrations")?
        .values_mut()
    {
        registration["origin"] = "development_fixture".into();
        registration["rationale"] =
            "Synthetic nine-analysis qualification only; no biological or production evidence"
                .into();
        registration["approval_or_plan_reference"] =
            "bekiper-post-analysis prepare-policy-fixture".into();
    }
    let mut publication = pipeline::Publication::claim(output)?;
    pipeline::write_json(
        &publication.directory().join("metadata.json"),
        &json!(metadata),
    )?; // coverage: unreachable — write into the freshly claimed private staging directory
    plan["inputs"]["metadata_path"] = "metadata.json".into();
    for (assay, rows) in &measurements {
        pipeline::write_json(
            &publication.directory().join(format!("{assay}.json")),
            &json!(rows),
        )?; // coverage: unreachable — write into the freshly claimed private staging directory
        plan["inputs"][format!("{assay}_path")] = format!("{assay}.json").into();
    }
    plan["outputs"]["directory"] = "synthetic-results".into();
    pipeline::write_json(&publication.directory().join("config.json"), &plan)?;
    ensure!(
        inputs::fingerprint(plan_path)? == plan_identity
            && inputs::fingerprint(oracle_path)? == oracle_identity,
        "fixture sources changed during preparation"
    );
    let receipt = json!({"schema_version":1,"scope":"synthetic_only","code":crate::code_identity(),
        "production_plan":plan_identity,"numerical_oracle":oracle_identity,"animals":animals.len(),
        "metadata_sessions":metadata.len(),"shared_measurements":measurements.values().map(Vec::len).sum::<usize>(),
        "analysis_admitted":false,"production_inputs_modified":false});
    pipeline::write_json(
        &publication.directory().join("source_receipt.json"),
        &receipt,
    )?; // coverage: unreachable — write into the freshly claimed private staging directory
    pipeline::write_artifact_manifest(publication.directory())?;
    publication.publish()?;
    Ok(receipt)
}

struct SyntheticSessions {
    metadata: Vec<Record>,
    measurements: BTreeMap<String, Vec<Record>>,
}

fn measurements(animals: &[Record]) -> Result<SyntheticSessions> {
    let mut metadata = Vec::new();
    let mut measurements: BTreeMap<String, Vec<Record>> = ["oft", "nort", "ymaze"]
        .into_iter()
        .map(|a| (a.into(), Vec::new()))
        .collect();
    for (index, animal) in animals.iter().enumerate() {
        for (assay, rows) in &mut measurements {
            for timepoint in ["pre", "post"] {
                let phases: &[&str] = if assay == "nort" {
                    &["training", "test"]
                } else {
                    &["session"]
                };
                for &phase in phases {
                    let key = json!({"animal_id":animal["animal_id"],"assay":assay,"session_id":format!("{assay}-{timepoint}-{phase}-{index}"),"timepoint":timepoint,"phase":phase});
                    let mut meta = key.clone();
                    for field in ["group", "five_xfad", "hcar1", "treatment"] {
                        meta[field] = animal[field].clone();
                    }
                    meta["sex"] = if index % 2 == 0 { "female" } else { "male" }.into();
                    meta["age_treatment_window"] =
                        ["old/late", "mid/late", "mid/early", "young/early"][index % 4].into();
                    meta["excluded"] = false.into();
                    meta["exclusion_reason"] = Value::Null;
                    metadata.push(meta);
                    let value = animal[if timepoint == "pre" {
                        "pre_outcome"
                    } else {
                        "DI"
                    }]
                    .as_f64()
                    .context("finite oracle endpoint")?;
                    let mut measured = key;
                    measured["analysis_duration_s"] = 300.into();
                    measured["synthetic_observation"] = true.into();
                    measured["qc_flags"] = json!({"fixture_qc_flag":true});
                    match assay.as_str() {
                        "oft" => {
                            measured["total_distance_m"] = json!(value * 10.0 + 1.0);
                            measured["mean_speed_m_per_s"] = json!((value * 10.0 + 1.0) / 300.0);
                            measured["center_time_s"] = json!(value * 100.0 + 20.0);
                            measured["center_entries"] = json!(if timepoint == "pre" {
                                index % 9 + 1
                            } else {
                                (index * 7 + 3) % 11 + 2
                            });
                            measured["latency_to_first_center_entry_s"] = json!(value * 20.0 + 3.0);
                            measured["periphery_time_pct"] =
                                json!(100.0 - (value * 100.0 + 20.0) / 3.0);
                            rows.push(measured);
                        }
                        "nort" => {
                            if phase == "training" {
                                measured["TA"] = json!(10 + index % 5);
                                measured["TB"] = json!(8 + index % 4);
                            } else {
                                measured["TN"] = json!((1.0 + value) * 20.0);
                                measured["TF"] = json!((1.0 - value) * 20.0);
                            }
                            rows.push(measured);
                        }
                        _ => {
                            for threshold in [25u64, 45, 65] {
                                let mut state = (index as u64 + 1) * 1009
                                    + threshold
                                    + if timepoint == "post" { 997 } else { 0 };
                                let sequence: Vec<_> = (0..12 + index % 9)
                                    .map(|_| {
                                        state =
                                            state.wrapping_mul(6364136223846793005).wrapping_add(1);
                                        ["A", "B", "C"][((state >> 32) % 3) as usize]
                                    })
                                    .collect();
                                let mut row = measured.clone();
                                row["threshold_pct"] = threshold.into();
                                row["valid_arm_entry_sequence"] = json!(sequence);
                                row["raw_arm_entry_sequence"] = json!(sequence);
                                row["total_distance_m"] = json!(value * 6.0 + 2.0);
                                row["mean_speed_m_per_s"] = json!((value * 6.0 + 2.0) / 300.0);
                                row["freezing_s"] = json!(index % 5);
                                rows.push(row);
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(SyntheticSessions {
        metadata,
        measurements,
    })
}
