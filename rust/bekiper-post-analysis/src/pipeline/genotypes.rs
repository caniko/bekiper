//! Three HCAR1 analyses reuse the scored endpoints inside each threshold context.
use super::*;

pub struct PolicyAnalysis {
    pub policy: String,
    pub rows: Vec<Record>,
    pub config: Record,
    pub inference: InferenceResults,
    pub subgroups: Vec<Record>,
    pub subgroup_status: Record,
    pub audit: Vec<Record>,
}

pub fn run_policies(rows: &[Record], config: &Record) -> Result<Vec<PolicyAnalysis>> {
    run_policies_with_cancel(rows, config, &|| false)
}

pub fn run_policies_with_cancel(
    rows: &[Record],
    config: &Record,
    cancelled: &dyn Fn() -> bool,
) -> Result<Vec<PolicyAnalysis>> {
    ensure!(
        crate::genotype::policy_plan(config)?.is_some(),
        "No HCAR1 policy plan"
    );
    ensure!(
        choice(config, "inputs.factor_levels", "D10").is_some(),
        "HCAR1 policies require registered factor levels"
    );
    let mut analyses = Vec::new();
    for policy in ["A", "B", "C"] {
        check_cancel(cancelled)?;
        let (coded, mut groups) = crate::genotype::project(rows, config, policy)?;
        let mut effective = config.clone();
        effective["hcar1_policy_context"] = json!({"policy":policy,"label":crate::genotype::label(policy),
            "primary":policy=="A","source_D10":config["decision_resolutions"]["D10"],"endpoint_recalculation":false});
        if policy == "B" {
            for (name, group) in config["inputs"]["group_mapping"]
                .as_object()
                .context("Group mapping")?
            {
                if group["hcar1"] == "WT" {
                    let mut group = group.clone();
                    group["hcar1"] = "Het".into();
                    groups[format!("{name} [Het]")] = group;
                }
            }
        }
        effective["inputs"]["group_mapping"] = groups.clone();
        effective["inputs"]["factor_levels"]["hcar1"] = if policy == "B" {
            json!(["KO", "WT", "Het"])
        } else {
            json!(["KO", "WT"])
        };
        for path in ["inputs.factor_levels", "inputs.group_mapping"] {
            effective["decision_resolutions"]["D10"]["value"][path] =
                if path.ends_with("factor_levels") {
                    effective["inputs"]["factor_levels"].clone()
                } else {
                    groups.clone()
                };
        }
        if choice(config, "statistics.multiplicity", "D09").is_some()
            && let Some(families) =
                effective["statistics"]["multiplicity"]["families"].as_object_mut()
        {
            for family in families.values_mut() {
                let name = family.as_str().context("Multiplicity family name")?;
                *family = format!("{name}:HCAR1_policy_{policy}").into();
            }
            effective["decision_resolutions"]["D09"]["value"]["statistics.multiplicity"] =
                effective["statistics"]["multiplicity"].clone();
        }
        if crate::inputs::validated_subgroup_plan(config)?.is_some() {
            let family =
                effective["statistics"]["subgroup_analysis_plan"]["multiplicity"]["family"]
                    .as_str()
                    .context("Subgroup family")?;
            effective["statistics"]["subgroup_analysis_plan"]["multiplicity"]["family"] =
                format!("{family}:HCAR1_policy_{policy}").into();
            for decision in ["D09", "D12"] {
                effective["decision_resolutions"][decision]["value"]["statistics.subgroup_analysis_plan"] =
                    effective["statistics"]["subgroup_analysis_plan"].clone();
            }
        }
        let inference = run_models(&coded, &effective)?;
        check_cancel(cancelled)?;
        let (subgroups, subgroup_status) = if effective["outputs"]["requested_outputs"]
            .as_array()
            .is_some_and(|r| r.iter().any(|r| r == "subgroup_inference"))
        {
            run_subgroups(&coded, &effective)?
        } else {
            (Vec::new(), Value::Null)
        };
        let mut audit = Vec::new();
        for row in &coded {
            for endpoint in ledger::endpoint_fields(row) {
                let model = inference
                    .models
                    .iter()
                    .find(|m| {
                        m["endpoint"] == endpoint
                            && m["model_id"]
                                == format!("{}.{endpoint}", row["assay"].as_str().unwrap_or(""))
                    })
                    .filter(|_| {
                        (row["assay"] != "nort" || row["phase"] == "test")
                            && (row["assay"] != "ymaze" || row["primary_threshold"] == true)
                    });
                let subset = model
                    .and_then(|m| m["subset"].as_array())
                    .and_then(|entries| {
                        entries.iter().find(|e| {
                            e["animal_id"] == row["animal_id"]
                                && e["session_id"] == row["session_id"]
                        })
                    });
                let pair_id = row["pair_id"].as_str().unwrap_or("primary");
                let pair = inference.pairing_ledger.iter().find(|p| {
                    p["animal_id"] == row["animal_id"]
                        && p["assay"] == row["assay"]
                        && p["phase"] == row["phase"]
                        && p["endpoint"] == endpoint
                        && p["threshold_pct"] == row["threshold_pct"]
                        && p["pair_id"] == pair_id
                });
                let mut entry = key_fields(row);
                entry["threshold_pct"] = row["threshold_pct"].clone();
                for field in ["recording_id", "analysis_id", "variant", "pair_id"] {
                    entry[field] = row[field].clone();
                }
                entry["endpoint"] = endpoint.into();
                entry["policy"] = policy.into();
                entry["literal_hcar1"] = row["literal_hcar1"].clone();
                entry["assigned_group"] = row["assigned_group"].clone();
                entry["analysis_factor_coding"] = row["analysis_factor_coding"].clone();
                entry["genotype_eligibility"] = row["genotype_eligibility"].clone();
                let value = row["qc_flags"].get(endpoint).unwrap_or(&row[endpoint]);
                entry["endpoint_available"] = (!value.is_null()).into();
                entry["value"] = value.clone();
                entry["input_verification"] = row["input_verification"].clone();
                entry["metric_source"] = row["metric_sources"][endpoint].clone();
                entry["factor_metadata_missing"] = json!(
                    crate::contracts::FACTORS
                        .into_iter()
                        .filter(|f| row[*f].is_null())
                        .collect::<Vec<_>>()
                );
                entry["missing_reason"] = row["missing_reasons"][endpoint].clone();
                entry["qc_flags"] = row["qc_flags"].clone();
                entry["qc_excluded"] = false.into();
                entry["metadata_excluded"] = row["excluded"].clone();
                entry["model_eligibility"] = subset.cloned().unwrap_or(Value::Null);
                entry["model_attempted_for_endpoint"] = model.is_some().into();
                entry["pair_eligibility"] = pair.cloned().unwrap_or(Value::Null);
                entry["model_status"] = model
                    .map(|m| m["status"].clone())
                    .unwrap_or(json!("not_applicable_or_not_requested"));
                entry["model_reason"] = model.map(|m| m["reason"].clone()).unwrap_or(Value::Null);
                audit.push(entry);
            }
        }
        analyses.push(PolicyAnalysis {
            policy: policy.into(),
            rows: coded,
            config: effective,
            inference,
            subgroups,
            subgroup_status,
            audit,
        });
    }
    Ok(analyses)
}

pub fn comparison(analyses: &[PolicyAnalysis]) -> Record {
    let mut rows = Vec::new();
    for analysis in analyses {
        for model in &analysis.inference.models {
            let contrast = &model["contrast"];
            let mut target_counts = json!({});
            let mut target_n = 0_u64;
            for (group, _) in statistics::WEIGHTS {
                let target = &analysis.config["inputs"]["group_mapping"][group];
                let count = model["factor_cell_counts"]
                    .as_array()
                    .and_then(|cells| {
                        cells.iter().find(|cell| {
                            crate::contracts::FACTORS
                                .iter()
                                .all(|f| cell[*f] == target[*f])
                        })
                    })
                    .and_then(|cell| cell["n_animals"].as_u64());
                target_counts[group] = json!(count);
                target_n += count.unwrap_or(0);
            }
            rows.push(json!({"policy":analysis.policy,"policy_label":crate::genotype::label(&analysis.policy),"primary":analysis.policy=="A",
                "model_id":model["model_id"],"endpoint":model["endpoint"],"n_model_animals":model["n_animals"],
                "n_contrast_target_animals":if model["factor_cell_counts"].is_array() {json!(target_n)}else{Value::Null},"target_counts":target_counts,
                "factor_cell_counts":model["factor_cell_counts"],"rank":model["rank"],"columns":model["columns"],
                "model_status":model["status"],"model_reason":model["reason"],
                "estimability_status":contrast.get("status").cloned().unwrap_or(json!("not_evaluated")),"estimability_reason":if contrast.is_object() {contrast["reason"].clone()}else{model["reason"].clone()},
                "estimate":contrast["estimate"],"ci_lower":contrast["ci_lower"],"ci_upper":contrast["ci_upper"],
                "confidence_level":model["settings"]["confidence_level"],
                "p_raw":contrast["p_raw"],"log_p_raw":contrast["log_p_raw"],
                "p_adjusted":contrast["p_adjusted"],"log_p_adjusted":contrast["log_p_adjusted"],
                "correction_family":contrast["correction_family"]}));
        }
    }
    let mut interpretation = Vec::new();
    for primary in rows.iter().filter(|r| r["policy"] == "A") {
        for sensitivity in rows
            .iter()
            .filter(|r| r["policy"] != "A" && r["model_id"] == primary["model_id"])
        {
            let (Some(a), Some(b)) = (
                primary["estimate"].as_f64(),
                sensitivity["estimate"].as_f64(),
            ) else {
                interpretation.push(json!({"model_id":primary["model_id"],"policy":sensitivity["policy"],"status":"comparison_unavailable","primary_status":primary["estimability_status"],"sensitivity_status":sensitivity["estimability_status"]}));
                continue;
            };
            let excludes_zero = |r: &Record| {
                r["ci_lower"]
                    .as_f64()
                    .zip(r["ci_upper"].as_f64())
                    .map(|(l, u)| l > 0.0 || u < 0.0)
            };
            interpretation.push(json!({"model_id":primary["model_id"],"policy":sensitivity["policy"],"status":"compared",
                "direction_changed":(a>0.0,a<0.0)!=(b>0.0,b<0.0),"estimate_difference_from_A":b-a,
                "confidence_level":primary["confidence_level"],
                "primary_interval_excludes_zero":excludes_zero(primary),"sensitivity_interval_excludes_zero":excludes_zero(sensitivity),
                "primary_adjusted_p":primary["p_adjusted"],"primary_log_adjusted_p":primary["log_p_adjusted"],
                "sensitivity_adjusted_p":sensitivity["p_adjusted"],"sensitivity_log_adjusted_p":sensitivity["log_p_adjusted"]}));
        }
    }
    json!({"primary_policy":"A","endpoint_recalculation":false,"contrast_direction":"(WT lactate - WT vehicle) - (KO lactate - KO vehicle)",
        "assigned_group_contrast":"(G5-G6)-(G1-G2)","rows":rows,
        "endpoint_interpretation":interpretation,
        "interpretation":"A remains the primary reference regardless of P values. B uses recorded Het as a separate level; C assumes assigned-group WT+Het pooling and animal 37's assigned KO, without establishing equivalence or genotype verification."})
}

pub(super) fn export_record(analyses: &[PolicyAnalysis]) -> Record {
    json!({"primary_policy":"A","comparison":comparison(analyses),"policies":analyses.iter().map(|a|json!({
        "policy":a.policy,"label":crate::genotype::label(&a.policy),"primary":a.policy=="A",
        "effective_config":a.config,"models":a.inference.models,"pairing_ledger":a.inference.pairing_ledger,
        "correlations":a.inference.correlations,"within_group_tests":a.inference.within_group_tests,
        "subgroup_analyses":a.subgroups,"subgroup_status":a.subgroup_status,"eligibility_and_coding_audit":a.audit})).collect::<Vec<_>>()})
}

pub(super) fn plotting_bundle(
    analyses: &[PolicyAnalysis],
    requested: &[String],
    pdf: bool,
) -> Result<Record> {
    let mut combined = Value::Null;
    for analysis in analyses {
        let mut config = analysis.config.clone();
        let mut bundle = prepare_plotting_bundle(&analysis.rows, &mut config, requested, pdf)?;
        let mut names = BTreeMap::new();
        for (index, figure) in bundle["figures"]
            .as_array_mut()
            .context("Policy figures")?
            .iter_mut()
            .enumerate()
        {
            let old = figure["name"].as_str().context("Figure name")?.to_owned();
            let assay = crate::contracts::assays_in_report_order()
                .iter().copied()
                .find(|assay| old.split('_').any(|p| p == *assay))
                .unwrap_or("all");
            let name = if analysis.policy == "A" {
                old.clone()
            } else {
                format!(
                    "supplement_hcar1_{}_{assay}_{index}",
                    analysis.policy.to_ascii_lowercase()
                )
            };
            figure["source_figure_name"] = old.clone().into();
            names.insert(old, name.clone());
            figure["name"] = name.into();
            figure["hcar1_policy"] = analysis.policy.clone().into();
            figure["summary_label"] = format!(
                "{}; {}",
                crate::genotype::label(&analysis.policy),
                figure["summary_label"].as_str().unwrap_or("")
            )
            .into();
            for panel in figure["panels"].as_array_mut().context("Policy panels")? {
                let label = match analysis.policy.as_str() {
                    "A" => "A recorded WT/KO primary",
                    "B" => "B recorded WT/Het/KO sensitivity",
                    _ => "C assigned-group sensitivity",
                };
                panel["label"] =
                    format!("{label} — {}", panel["label"].as_str().unwrap_or("")).into();
            }
        }
        for figure in bundle["figures"].as_array_mut().context("Policy figures")? {
            if let Some(name) = figure["parent_figure"]
                .as_str()
                .and_then(|name| names.get(name))
            {
                figure["parent_figure"] = name.clone().into();
            }
        }
        for figure in bundle["unrendered_figures"]
            .as_array_mut()
            .context("Unrendered policy figures")?
        {
            let old = figure["name"]
                .as_str()
                .context("Unrendered figure name")?
                .to_owned();
            figure["source_figure_name"] = old.clone().into();
            figure["hcar1_policy"] = analysis.policy.clone().into();
            if analysis.policy != "A" {
                figure["name"] =
                    format!("{}_hcar1_{}", old, analysis.policy.to_ascii_lowercase()).into();
            }
        }
        for field in ["plotting_data", "summaries"] {
            for row in bundle[field]
                .as_array_mut()
                .context("Policy plotting records")?
            {
                if let Some(name) = row["figure"].as_str().and_then(|name| names.get(name)) {
                    row["figure"] = name.clone().into();
                }
                row["hcar1_policy"] = analysis.policy.clone().into();
            }
        }
        if combined.is_null() {
            combined = bundle;
        } else {
            for field in [
                "figures",
                "plotting_data",
                "summaries",
                "unrendered_figures",
            ] {
                combined[field]
                    .as_array_mut()
                    .context("Combined policy array")?
                    .extend(
                        bundle[field]
                            .as_array()
                            .context("Policy array")?
                            .iter()
                            .cloned(),
                    );
            }
        }
    }
    Ok(combined)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_figures_have_unique_portable_names_and_preserve_distribution_page_links() {
        let config = json!({"outputs":{"figure_selection":["figure_nort_distributions"]}});
        let row = json!({"animal_id":"fixture","assay":"nort","session_id":"post-test",
            "timepoint":"post","phase":"test","group":"G1","excluded":false,
            "TN":12.0,"TF":8.0,"DI":0.2,"percent_novel":60.0});
        let analyses: Vec<_> = ["A", "B", "C"]
            .into_iter()
            .map(|policy| PolicyAnalysis {
                policy: policy.into(),
                rows: vec![row.clone()],
                config: config.clone(),
                inference: InferenceResults::default(),
                subgroups: Vec::new(),
                subgroup_status: Value::Null,
                audit: Vec::new(),
            })
            .collect();
        let bundle = plotting_bundle(&analyses, &["figures".into()], false).unwrap();
        let figures = bundle["figures"].as_array().unwrap();
        let names: BTreeSet<_> = figures
            .iter()
            .map(|f| f["name"].as_str().unwrap())
            .collect();
        assert_eq!(names.len(), figures.len());
        assert!(
            figures
                .iter()
                .any(|figure| !figure["parent_figure"].is_null())
        );
        for figure in figures
            .iter()
            .chain(bundle["unrendered_figures"].as_array().unwrap())
        {
            let name = figure["name"].as_str().unwrap();
            assert!(
                name.bytes()
                    .all(|c| matches!(c, b'a'..=b'z' | b'0'..=b'9' | b'_' | b'-'))
            );
            if let Some(parent) = figure["parent_figure"].as_str() {
                assert!(names.contains(parent));
                let parent = figures.iter().find(|f| f["name"] == parent).unwrap();
                assert_eq!(parent["hcar1_policy"], figure["hcar1_policy"]);
            }
        }
    }
}
