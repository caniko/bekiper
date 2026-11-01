//! Model-family corrections and registered subgroup orchestration.
use super::*;

pub(super) fn correct_model_family(models: &mut [Record], config: &Record) -> Result<()> {
    let mut contrasts:Vec<_>=models.iter().map(|model| if model["contrast"].is_object() { model["contrast"].clone() } else { json!({"p_raw":null,"model_id":model["model_id"],"reason":model["reason"],"status":model["status"]}) }).collect();
    let correction = choice(config, "statistics.multiplicity", "D09");
    let family = correction
        .and_then(|value| value["families"]["planned_contrasts"].as_str())
        .filter(|family| !family.is_empty());
    if let Some(family) = family {
        let correction = correction.context("multiplicity settings")?;
        statistics::adjust_family(
            &mut contrasts,
            correction["method"]
                .as_str()
                .context("multiplicity method")?,
            family,
        )?;
    } else {
        for contrast in &mut contrasts {
            contrast["p_adjusted"] = Value::Null;
            contrast["adjustment_reason"] = "blocked:D09:multiplicity_family".into();
        }
    }
    for (model, contrast) in models.iter_mut().zip(contrasts) {
        model["contrast"] = contrast;
    }
    let family = correction.and_then(|value| value["families"]["genotype_simple_effects"].as_str());
    let mut effects = Vec::new();
    let mut locations = Vec::new();
    for (model_index, model) in models.iter().enumerate() {
        for (effect_index, effect) in model["genotype_lactate_effects"]
            .as_array()
            .into_iter()
            .flatten()
            .enumerate()
        {
            effects.push(effect.clone());
            locations.push((model_index, effect_index));
        }
    }
    if let Some(family) = family.filter(|f| !f.is_empty()) {
        statistics::adjust_family(
            &mut effects,
            correction.unwrap()["method"]
                .as_str()
                .context("Correction method")?,
            family,
        )?;
    } else {
        for effect in &mut effects {
            effect["p_adjusted"] = Value::Null;
            effect["adjustment_reason"] = "blocked:D09:genotype_simple_effects_family".into();
        }
    }
    for ((model_index, effect_index), effect) in locations.into_iter().zip(effects) {
        models[model_index]["genotype_lactate_effects"][effect_index] = effect;
    }
    Ok(())
}

pub fn run_subgroups(rows: &[Record], config: &Record) -> Result<(Vec<Record>, Record)> {
    let Some(plan) = crate::inputs::validated_subgroup_plan(config)? else {
        return Ok((
            Vec::new(),
            json!({"output":"subgroup_inference","status":"blocked","decision_ids":["D09","D12"],"reason":"register the same subgroup plan under D09 and D12"}),
        ));
    };
    let parent_scope = scope_for(rows, config);
    let mut scoped_config = config.clone();
    scoped_config["outputs"]["requested_outputs"] = json!(plan.outputs);
    let mut results = Vec::new();
    for stratum in &plan.strata {
        let field = stratum.field;
        let assays: BTreeSet<String> = stratum.assays.iter().cloned().collect();
        let scope = Scope {
            baseline: parent_scope
                .baseline
                .intersection(&assays)
                .cloned()
                .collect(),
            assays: assays.clone(),
        };
        for level in &stratum.levels {
            let candidates: Vec<_> = rows
                .iter()
                .filter(|row| {
                    row["assay"]
                        .as_str()
                        .is_some_and(|assay| assays.contains(assay))
                })
                .collect();
            let members: Vec<_> = candidates
                .iter()
                .filter(|row| row[field] == *level)
                .map(|row| (*row).clone())
                .collect();
            let attempted = run_models_scoped(&members, &scoped_config, &scope)?;
            let membership: Vec<_> = candidates
                .iter()
                .map(|row| {
                    let mut item = key_fields(row);
                    item["threshold_pct"] = row["threshold_pct"].clone();
                    item["reason"] = if row[field] == *level {
                        Value::Null
                    } else if row[field].is_null() {
                        "missing_stratum_metadata".into()
                    } else {
                        "outside_registered_stratum".into()
                    };
                    item
                })
                .collect();
            results.push(json!({"field":field,"level":level,"assays":stratum.assays,"models":attempted.models,"pairing_ledger":attempted.pairing_ledger,
                "correlations":attempted.correlations,"within_group_tests":attempted.within_group_tests,"membership_ledger":membership}));
        }
    }
    let mut tests = Vec::new();
    let mut locations = Vec::new();
    for (index, item) in results.iter().enumerate() {
        for kind in ["models", "correlations", "within_group_tests"] {
            for (attempt, record) in item[kind]
                .as_array()
                .context("subgroup attempts")?
                .iter()
                .enumerate()
            {
                tests.push(if kind == "models" {
                    record["contrast"].clone()
                } else {
                    record.clone()
                });
                locations.push((index, kind, attempt));
            }
        }
    }
    statistics::adjust_family(&mut tests, "holm", plan.family)?;
    for ((index, kind, attempt), test) in locations.into_iter().zip(tests) {
        if kind == "models" {
            results[index][kind][attempt]["contrast"] = test;
        } else {
            results[index][kind][attempt] = test;
        }
    }
    for item in &mut results {
        let mut statuses = json!({});
        for name in &plan.outputs {
            let name = name.as_str();
            statuses[name] = result_status(
                item[name].as_array().context("subgroup attempts")?,
                if name == "models" {
                    "fitted"
                } else {
                    "computed"
                },
            );
        }
        let status = overall_status(&statuses);
        item["status"] = status.into();
        item["output_statuses"] = statuses;
        item["reason"] = if status == "complete" {
            Value::Null
        } else {
            item["output_statuses"].clone()
        };
    }
    let mut status = result_status(&results, "complete");
    status["output"] = "subgroup_inference".into();
    status["plan"] = plan.original.clone();
    Ok((results, status))
}
