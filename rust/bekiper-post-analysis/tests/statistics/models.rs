#[test]
fn unbalanced_models_match_independent_cell_means_qr_oracle() {
    let fixture = fixture();
    for baseline in [false, true] {
        let mut settings = fixture["settings"].clone();
        if baseline {
            settings["additional_covariates"] = json!(["age", "sex"]);
            settings["contrast_estimand_and_weighting"]["reference_values"] =
                json!({"pre_outcome":"mean", "age":"mean", "sex":"female"});
        }
        let model = fit_factorial(
            fixture["rows"].as_array().unwrap(),
            "DI",
            &fixture["levels"],
            &fixture["groups"],
            &settings,
            baseline,
            Some("allow_unverified"),
        )
        .unwrap();
        let oracle = &fixture[if baseline {
            "baseline_adjusted"
        } else {
            "post_only"
        }];
        assert_eq!(model["status"], "fitted", "{model}");
        assert_eq!(model["n_animals"], oracle["n_animals"]);
        assert_eq!(model["residual_df"], oracle["df"]);
        close(
            &model["residual_variance"],
            &oracle["residual_variance"],
            1e-10,
        );
        for (index, coefficient) in model["coefficients"].as_array().unwrap().iter().enumerate() {
            close(
                &coefficient["estimate"],
                &oracle["coefficients"][index],
                1e-9,
            );
            for field in ["standard_error", "statistic", "ci_lower", "ci_upper"] {
                close(
                    &coefficient[field],
                    &oracle["coefficient_summaries"][index][field],
                    1e-9,
                );
            }
            probability_close(
                &coefficient["p_raw"],
                &oracle["coefficient_summaries"][index]["p_raw"],
            );
        }
        for (i, row) in model["covariance_matrix"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
        {
            for (j, value) in row.as_array().unwrap().iter().enumerate() {
                close(value, &oracle["covariance_matrix"][i][j], 1e-10);
            }
        }
        for key in [
            "estimate",
            "standard_error",
            "statistic",
            "p_raw",
            "log_p_raw",
            "ci_lower",
            "ci_upper",
        ] {
            close(&model["contrast"][key], &oracle["contrast"][key], 1e-9);
        }
        probability_close(&model["contrast"]["p_raw"], &oracle["contrast"]["p_raw"]);
        assert_eq!(
            model["contrast"]["weights"],
            json!({"G5":1,"G6":-1,"G1":-1,"G2":1})
        );
        assert!(model["contrast"]["estimate"].as_f64().unwrap() > 0.0);
        for group in ["G5", "G6", "G1", "G2"] {
            close(
                &model["contrast"]["group_estimates"][group],
                &oracle["contrast"]["group_estimates"][group],
                1e-9,
            );
        }
        for (index, diagnostic) in model["diagnostics"].as_array().unwrap().iter().enumerate() {
            close(&diagnostic["fitted"], &oracle["fitted"][index], 1e-9);
            close(&diagnostic["residual"], &oracle["residuals"][index], 1e-9);
            close(&diagnostic["leverage"], &oracle["leverage"][index], 1e-9);
        }
        assert_eq!(model["normality_diagnostic"]["status"], "unavailable");
    }
}

#[test]
fn missing_cell_and_exact_fit_are_errors_with_interactions_retained() {
    let fixture = fixture();
    let rows = fixture["rows"].as_array().unwrap();
    let model = fit_factorial(
        &rows[3..],
        "DI",
        &fixture["levels"],
        &fixture["groups"],
        &fixture["settings"],
        false,
        Some("allow_unverified"),
    )
    .unwrap();
    assert_eq!(model["reason"], "rank_deficient:interactions_retained");
    assert_eq!(model["columns"], 8);
    assert_eq!(model["contrast"]["status"], "computed");
    close(
        &model["contrast"]["estimate"],
        &fixture["post_only"]["contrast"]["estimate"],
        1e-9,
    );
    let exact: Vec<_> = rows
        .iter()
        .cloned()
        .map(|mut row| {
            row["DI"] = json!(if row["five_xfad"] == "AD" { 1.0 } else { 0.0 });
            row
        })
        .collect();
    let model = fit_factorial(
        &exact,
        "DI",
        &fixture["levels"],
        &fixture["groups"],
        &fixture["settings"],
        false,
        Some("allow_unverified"),
    )
    .unwrap();
    assert_eq!(model["reason"], "numerically_zero_residual_variation");
}

#[test]
fn missing_het_cells_retain_estimable_wt_ko_contrast_and_missing_target_is_not_estimable() {
    let fixture = fixture();
    let mut levels = fixture["levels"].clone();
    levels["hcar1"] = json!(["KO", "WT", "Het"]);
    let rows = fixture["rows"].as_array().unwrap();
    let model = fit_factorial(
        rows,
        "DI",
        &levels,
        &fixture["groups"],
        &fixture["settings"],
        false,
        Some("allow_unverified"),
    )
    .unwrap();
    assert_eq!(model["rank"], 8);
    assert_eq!(model["columns"], 12);
    assert_eq!(model["reason"], "rank_deficient:interactions_retained");
    assert_eq!(model["contrast"]["status"], "computed");
    for field in [
        "estimate",
        "standard_error",
        "ci_lower",
        "ci_upper",
        "p_raw",
    ] {
        close(
            &model["contrast"][field],
            &fixture["post_only"]["contrast"][field],
            1e-9,
        );
    }
    assert!(
        model["coefficients"]
            .as_array()
            .unwrap()
            .iter()
            .any(|coefficient| coefficient["status"] == "not_estimable"
                && coefficient["estimate"].is_null())
    );
    let missing_target: Vec<_> = rows
        .iter()
        .filter(|r| !(r["five_xfad"] == "AD" && r["hcar1"] == "WT" && r["treatment"] == "lactate"))
        .cloned()
        .collect();
    let model = fit_factorial(
        &missing_target,
        "DI",
        &levels,
        &fixture["groups"],
        &fixture["settings"],
        false,
        Some("allow_unverified"),
    )
    .unwrap();
    assert_eq!(model["contrast"]["status"], "not_estimable");
    assert!(model["contrast"]["estimate"].is_null());
    assert!(model["contrast"]["p_raw"].is_null());
    assert_eq!(
        model["contrast"]["group_estimates"]["G5"],
        serde_json::Value::Null
    );
}

#[test]
fn rank_deficient_baseline_design_retains_reference_adjusted_contrast() {
    let fixture = fixture();
    let mut levels = fixture["levels"].clone();
    levels["hcar1"] = json!(["KO", "WT", "Het"]);
    let mut settings = fixture["settings"].clone();
    settings["additional_covariates"] = json!(["age", "sex"]);
    settings["contrast_estimand_and_weighting"]["reference_values"] =
        json!({"pre_outcome":"mean","age":"mean","sex":"female"});
    let model = fit_factorial(
        fixture["rows"].as_array().unwrap(),
        "DI",
        &levels,
        &fixture["groups"],
        &settings,
        true,
        Some("allow_unverified"),
    )
    .unwrap();
    assert_eq!(model["contrast"]["status"], "computed");
    for field in [
        "estimate",
        "standard_error",
        "ci_lower",
        "ci_upper",
        "p_raw",
    ] {
        close(
            &model["contrast"][field],
            &fixture["baseline_adjusted"]["contrast"][field],
            1e-9,
        );
    }
}

#[test]
fn contrast_semantics_and_covariate_types_are_validated() {
    let fixture = fixture();
    let mut groups = fixture["groups"].clone();
    groups["G5"] = groups["G1"].clone();
    assert!(validate_contrast_mapping(&groups, &fixture["settings"]["contrast_roles"]).is_err());
    let mut rows = fixture["rows"].as_array().unwrap().clone();
    rows[0]["DI"] = "not a number".into();
    assert!(
        fit_factorial(
            &rows,
            "DI",
            &fixture["levels"],
            &fixture["groups"],
            &fixture["settings"],
            false,
            Some("allow_unverified")
        )
        .is_err()
    );
    let mut levels = fixture["levels"].clone();
    levels["hcar1"] = json!(["KO", "KO"]);
    assert!(
        fit_factorial(
            fixture["rows"].as_array().unwrap(),
            "DI",
            &levels,
            &fixture["groups"],
            &fixture["settings"],
            false,
            Some("allow_unverified")
        )
        .is_err()
    );
}

#[test]
fn balanced_planned_contrast_matches_hand_calculated_pooled_cell_variance() {
    let fixture = fixture();
    let mut rows = Vec::new();
    for (cell, (ad, hcar, treatment)) in ["WT", "AD"]
        .into_iter()
        .flat_map(|ad| {
            ["KO", "WT"].into_iter().flat_map(move |hcar| {
                ["vehicle", "lactate"]
                    .into_iter()
                    .map(move |treatment| (ad, hcar, treatment))
            })
        })
        .enumerate()
    {
        let mean = if treatment == "lactate" { 0.2 } else { 0.0 }
            + if (ad, hcar, treatment) == ("AD", "WT", "lactate") {
                0.4
            } else {
                0.0
            };
        for (rep, noise) in [-0.03, -0.01, 0.01, 0.03].into_iter().enumerate() {
            rows.push(
                json!({"animal_id":format!("a{cell}_{rep}"),"session_id":format!("s{cell}_{rep}"),
                "five_xfad":ad,"hcar1":hcar,"treatment":treatment,"DI":mean+noise}),
            );
        }
    }
    let model = fit_factorial(
        &rows,
        "DI",
        &fixture["levels"],
        &fixture["groups"],
        &fixture["settings"],
        false,
        Some("allow_unverified"),
    )
    .unwrap();
    let variance = 8.0
        * [-0.03_f64, -0.01, 0.01, 0.03]
            .iter()
            .map(|noise| noise.powi(2))
            .sum::<f64>()
        / 24.0;
    close(&model["contrast"]["estimate"], &json!(0.4), 1e-12);
    close(
        &model["contrast"]["standard_error"],
        &json!(variance.sqrt()),
        1e-12,
    );
}

#[test]
fn saturated_design_requires_residual_degrees_of_freedom() {
    let fixture = fixture();
    let mut seen = std::collections::BTreeSet::new();
    let rows: Vec<_> = fixture["rows"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|row| {
            seen.insert(
                serde_json::to_string(&json!([row["five_xfad"], row["hcar1"], row["treatment"]]))
                    .unwrap(),
            )
        })
        .cloned()
        .collect();
    let model = fit_factorial(
        &rows,
        "DI",
        &fixture["levels"],
        &fixture["groups"],
        &fixture["settings"],
        false,
        Some("allow_unverified"),
    )
    .unwrap();
    assert_eq!(model["rank"], 8);
    assert_eq!(model["reason"], "insufficient_residual_degrees_of_freedom");
}

#[test]
fn response_unit_changes_preserve_inference_without_an_absolute_variance_gate() {
    let fixture = fixture();
    let rows = fixture["rows"].as_array().unwrap();
    let original = fit_factorial(
        rows,
        "DI",
        &fixture["levels"],
        &fixture["groups"],
        &fixture["settings"],
        false,
        Some("allow_unverified"),
    )
    .unwrap();
    let multiplier = 1e-18;
    let scaled: Vec<_> = rows
        .iter()
        .cloned()
        .map(|mut row| {
            row["DI"] = json!(row["DI"].as_f64().unwrap() * multiplier);
            row
        })
        .collect();
    let model = fit_factorial(
        &scaled,
        "DI",
        &fixture["levels"],
        &fixture["groups"],
        &fixture["settings"],
        false,
        Some("allow_unverified"),
    )
    .unwrap();
    assert_eq!(model["status"], "fitted", "{model}");
    for field in ["estimate", "standard_error", "ci_lower", "ci_upper"] {
        close(
            &json!(model["contrast"][field].as_f64().unwrap() / multiplier),
            &original["contrast"][field],
            1e-10,
        );
    }
    for field in ["statistic", "p_raw", "log_p_raw"] {
        close(
            &model["contrast"][field],
            &original["contrast"][field],
            1e-10,
        );
    }
}

#[test]
fn multilevel_factorial_matches_independent_cell_means_variance() {
    let fixture = fixture();
    let levels = json!({"five_xfad":["WT", "AD", "other"], "hcar1":["KO", "WT"], "treatment":["vehicle", "lactate", "other"]});
    let mut rows = Vec::new();
    let mut means = std::collections::BTreeMap::new();
    for ad in ["WT", "AD", "other"] {
        for hcar in ["KO", "WT"] {
            for treatment in ["vehicle", "lactate", "other"] {
                let cell = means.len();
                let mean = cell as f64 * 0.01
                    + if (ad, hcar, treatment) == ("AD", "WT", "lactate") {
                        0.4
                    } else {
                        0.0
                    };
                means.insert((ad, hcar, treatment), mean);
                for (rep, noise) in [-0.03, -0.01, 0.04].into_iter().enumerate() {
                    rows.push(json!({"animal_id":format!("a{cell}_{rep}"), "session_id":format!("s{cell}_{rep}"),
                        "five_xfad":ad, "hcar1":hcar, "treatment":treatment, "DI":mean+noise}));
                }
            }
        }
    }
    let model = fit_factorial(
        &rows,
        "DI",
        &levels,
        &fixture["groups"],
        &fixture["settings"],
        false,
        Some("allow_unverified"),
    )
    .unwrap();
    assert_eq!(model["status"], "fitted", "{model}");
    assert_eq!(model["rank"], 18);
    assert_eq!(model["columns"], 18);
    assert_eq!(model["residual_df"], 36);
    let variance = 18.0
        * [-0.03_f64, -0.01, 0.04]
            .iter()
            .map(|value| value.powi(2))
            .sum::<f64>()
        / 36.0;
    close(&model["residual_variance"], &json!(variance), 1e-12);
    let estimate = means[&("AD", "WT", "lactate")]
        - means[&("AD", "WT", "vehicle")]
        - means[&("AD", "KO", "lactate")]
        + means[&("AD", "KO", "vehicle")];
    close(&model["contrast"]["estimate"], &json!(estimate), 1e-12);
    close(
        &model["contrast"]["standard_error"],
        &json!((variance * 4.0 / 3.0).sqrt()),
        1e-12,
    );
    let triples = model["coefficient_names"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|name| name.as_str().unwrap().matches(':').count() == 2)
        .count();
    assert_eq!(triples, 4);
    for diagnostic in model["diagnostics"].as_array().unwrap() {
        close(&diagnostic["leverage"], &json!(1.0 / 3.0), 1e-12);
    }
}

#[test]
fn categorical_reference_and_numeric_covariate_units_do_not_change_the_contrast() {
    let fixture = fixture();
    let rows = fixture["rows"].as_array().unwrap();
    let mut settings = fixture["settings"].clone();
    settings["additional_covariates"] = json!(["age", "sex"]);
    settings["contrast_estimand_and_weighting"]["reference_values"] =
        json!({"pre_outcome":"mean", "age":"mean", "sex":"female"});
    let original = fit_factorial(
        rows,
        "DI",
        &fixture["levels"],
        &fixture["groups"],
        &settings,
        true,
        Some("allow_unverified"),
    )
    .unwrap();
    let scaled: Vec<_> = rows
        .iter()
        .cloned()
        .map(|mut row| {
            row["pre_outcome"] = json!(row["pre_outcome"].as_f64().unwrap() * 1e6);
            row["age"] = json!(row["age"].as_f64().unwrap() * 1e-4);
            row
        })
        .collect();
    let mut levels = fixture["levels"].clone();
    levels["five_xfad"] = json!(["AD", "WT"]);
    levels["hcar1"] = json!(["WT", "KO"]);
    levels["treatment"] = json!(["lactate", "vehicle"]);
    levels["sex"] = json!(["male", "female"]);
    let model = fit_factorial(
        &scaled,
        "DI",
        &levels,
        &fixture["groups"],
        &settings,
        true,
        Some("allow_unverified"),
    )
    .unwrap();
    assert_eq!(model["status"], "fitted", "{model}");
    for field in [
        "estimate",
        "standard_error",
        "statistic",
        "p_raw",
        "log_p_raw",
        "ci_lower",
        "ci_upper",
    ] {
        close(
            &model["contrast"][field],
            &original["contrast"][field],
            1e-10,
        );
    }
}
