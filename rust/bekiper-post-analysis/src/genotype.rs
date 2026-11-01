//! Registered HCAR1 sensitivity coding over shared, immutable scored endpoints.
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};

use crate::{Record, inputs::choice};

pub const PLAN_PATH: &str = "statistics.hcar1_policy_plan";

pub fn policy_plan(config: &Record) -> Result<Option<&Record>> {
    if config["statistics"].get("hcar1_policy_plan").is_none() {
        return Ok(None);
    }
    let plan = choice(config, PLAN_PATH, "D10")
        .context("HCAR1 policies require the matching registered D10 decision")?;
    ensure!(
        plan["primary"] == "A" && plan["policies"] == json!(["A", "B", "C"]),
        "HCAR1 policy plan must retain A as primary and A/B/C in that order"
    );
    Ok(Some(plan))
}

pub fn label(policy: &str) -> &'static str {
    match policy {
        "A" => "A: recorded literal WT/KO (primary)",
        "B" => "B: recorded WT/Het/KO (sensitivity)",
        "C" => "C: assigned groups; WT+Het pooling / assigned KO (sensitivity)",
        _ => "Unknown HCAR1 policy",
    }
}

/// Private statistical/plotting views retain original values in separate fields.
/// No endpoint, source record, metadata exclusion or QC flag is changed.
pub fn project(rows: &[Record], config: &Record, policy: &str) -> Result<(Vec<Record>, Record)> {
    ensure!(
        policy_plan(config)?.is_some(),
        "No registered HCAR1 policy plan"
    );
    ensure!(["A", "B", "C"].contains(&policy), "Unknown HCAR1 policy");
    let assigned = choice(config, "inputs.group_mapping", "D10")
        .context("Assigned-group coding requires registered group mappings")?;
    let mut projected = Vec::with_capacity(rows.len());
    let mut groups = assigned.clone();
    for source in rows {
        let mut row = source.clone();
        let literal = source.get("literal_hcar1").unwrap_or(&source["hcar1"]);
        let recorded = literal.as_str().map(str::trim);
        let group = source["group"].as_str();
        let eligible = match policy {
            "A" => matches!(recorded, Some("WT" | "KO")),
            "B" => matches!(recorded, Some("WT" | "Het" | "KO")),
            "C" => group.is_some_and(|g| assigned[g].is_object()),
            _ => unreachable!(), // coverage: unreachable — policy checked against A/B/C above
        };
        let reason = if eligible {
            None
        } else if policy == "C" {
            Some("missing_assigned_group")
        } else if recorded == Some("Het/KO") {
            Some("unresolved_recorded_Het/KO")
        } else if policy == "A" && recorded == Some("Het") {
            Some("recorded_Het_outside_WT_KO_primary")
        } else {
            Some("missing_or_unsupported_recorded_genotype")
        };
        row["literal_hcar1"] = literal.clone();
        row["assigned_group"] = source["group"].clone();
        row["source_factors"] = json!({"five_xfad":source["five_xfad"],"hcar1":literal,"treatment":source["treatment"],"group":source["group"]});
        if eligible {
            if policy == "C" {
                for factor in crate::contracts::FACTORS {
                    row[factor] = assigned[group.unwrap()][factor].clone();
                }
            } else {
                row["hcar1"] = recorded.unwrap().into();
            }
            if policy == "B" && recorded == Some("Het") {
                let name = format!(
                    "{} [Het]",
                    group.context("Recorded Het requires assigned group")?
                );
                row["group"] = name.clone().into();
                groups[&name] = json!({"five_xfad":row["five_xfad"],"hcar1":"Het","treatment":row["treatment"]});
            }
        }
        row["genotype_eligibility"] = json!({"policy":policy,"eligible":eligible,"reason":reason});
        row["analysis_factor_coding"] = json!({"policy":policy,"label":label(policy),"primary":policy=="A",
            "five_xfad":if eligible {row["five_xfad"].clone()}else{Value::Null},
            "hcar1":if eligible {row["hcar1"].clone()}else{Value::Null},
            "treatment":if eligible {row["treatment"].clone()}else{Value::Null},
            "group":if eligible {row["group"].clone()}else{Value::Null},
            "assumption":if policy=="C" {"assigned_group_not_genotype_verification"}else{"recorded_genotype_not_independently_verified"}});
        projected.push(row);
    }
    Ok((projected, groups))
}

pub fn eligibility_reason(row: &Record) -> Option<String> {
    (row["genotype_eligibility"]["eligible"] == false).then(|| {
        format!(
            "genotype_policy:{}",
            row["genotype_eligibility"]["reason"]
                .as_str()
                .unwrap_or("ineligible")
        )
    })
}
