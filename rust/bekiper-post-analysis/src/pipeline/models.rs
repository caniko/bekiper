//! Full registered assay model attempts.
use super::*;

pub(super) fn fit_assay_models(
    subset: &[Record],
    assay: &str,
    baseline: bool,
    config: &Record,
    requirement: Option<&str>,
    result: &mut InferenceResults,
) -> Result<()> {
    let Some(descriptor) = bekiper_core::experiment::post_analysis_descriptors()
        .find(|d| d.post_analysis.as_ref().is_some_and(|p| p.assay == assay))
    else {
        return Ok(());
    };
    let endpoints = descriptor.modelled();
    for endpoint in endpoints {
        let model_id = format!("{assay}.{endpoint}");
        let (post, pairs) = if baseline {
            statistics::pairing(subset, endpoint, requirement)?
        } else {
            (
                subset
                    .iter()
                    .filter(|row| row["timepoint"] == "post")
                    .cloned()
                    .collect(),
                Vec::new(),
            )
        };
        result.pairing_ledger.extend(pairs);
        let mut prerequisites = INCLUSION_REQUIREMENTS.to_vec();
        prerequisites.extend([
            ("inputs.group_mapping", "D10", None),
            ("inputs.factor_levels", "D10", None),
            (
                "statistics.model_failure_policy",
                "D09",
                Some("report_without_simplification"),
            ),
        ]);
        if baseline {
            prerequisites.push((
                "inclusion.unmatched_prepost_policy",
                "D12",
                Some("report_and_withhold_pair"),
            ));
        }
        if baseline && assay == "oft" {
            prerequisites.push((
                "statistics.oft_baseline_strategy",
                "D09",
                Some("baseline_adjusted"),
            ));
        }
        if !baseline && assay == "ymaze" {
            prerequisites.push((
                "statistics.ymaze_post_only_formula",
                "D09",
                Some("post_SA ~ 5XFAD_status * HCAR1_status * treatment"),
            ));
        }
        let mut failed = failed_choices(config, &prerequisites);
        if assay == "ymaze"
            && choice(
                config,
                "assays.ymaze.primary_threshold_arm_length_pct",
                "D03",
            )
            .is_none()
        {
            failed.push("D03:primary_threshold".into());
        }
        let mut settings = json!({});
        for field in [
            "factor_coding",
            "covariance_estimator",
            "contrast_estimand_and_weighting",
            "test_tails",
            "confidence_interval_method",
            "confidence_level",
            "contrast_roles",
        ] {
            settings[field] = choice(config, &format!("statistics.{field}"), "D09")
                .cloned()
                .unwrap_or(Value::Null);
        }
        settings["model_family"] = choice(config, "statistics.model_family_by_endpoint", "D09")
            .map(|families| families[&model_id].clone())
            .unwrap_or(Value::Null);
        settings["additional_covariates"] = if assay == "nort" {
            choice(config, "statistics.nort_additional_covariates", "D09")
                .cloned()
                .unwrap_or(Value::Null)
        } else {
            json!([])
        };
        if settings.as_object().unwrap().values().any(Value::is_null) {
            failed.push("D09:inference_settings".into());
        }
        let mut model = if failed.is_empty() {
            statistics::fit_factorial(
                &post,
                endpoint,
                &config["inputs"]["factor_levels"],
                &config["inputs"]["group_mapping"],
                &settings,
                baseline,
                requirement,
            )?
        } else {
            json!({"endpoint":endpoint,"status":"blocked","reason":failed,"n_animals":0,"contrast":null,"subset":[],"settings":settings})
        };
        model["model_id"] = model_id.into();
        if model.get("genotype_lactate_effects").is_none()
            && let Some(levels) = config["inputs"]["factor_levels"]["hcar1"].as_array()
        {
            // Unavailable models remain declared attempts in the supplementary
            // correction family, rather than silently reducing its size.
            model["genotype_lactate_effects"]=json!(levels.iter().map(|level|json!({"hcar1":level,
                "name":"AD background: lactate - vehicle","inference_role":"supplementary_genotype_simple_effect",
                "status":"not_evaluated","reason":model["reason"],"estimate":null,"p_raw":null})).collect::<Vec<_>>());
        }
        result.models.push(model);
    }
    Ok(())
}
