//! Registered decisions and effective execution configuration.
use super::*;

pub(super) fn at<'a>(value: &'a Value, path: &str) -> Option<&'a Value> {
    path.split('.')
        .try_fold(value, |v, key| v.as_object()?.get(key))
}

/// Invalid or unapproved registrations are unavailable, never implicit defaults.
pub fn choice<'a>(config: &'a Record, path: &str, decision: &str) -> Option<&'a Value> {
    let value = at(config, path).filter(|v| !v.is_null())?;
    let record = config
        .get("decision_resolutions")?
        .get(decision)?
        .as_object()?;
    for field in ["origin", "rationale", "approval_or_plan_reference"] {
        if record.get(field)?.as_str()?.trim().is_empty() {
            return None;
        }
    }
    match record.get("origin")?.as_str()? {
        "development_fixture"
            if config.get("run_mode")?.as_str()? == "synthetic"
                && config.get("synthetic_fixture")?.as_bool()? => {}
        "existing_analysis_plan" | "researcher_decision" => {}
        _ => return None,
    }
    if record.get("value")?.as_object()?.get(path)? != value {
        return None;
    }
    let scopes = record.get("scope")?.as_array()?;
    if scopes.iter().any(|s| s.as_str().is_none()) {
        return None;
    }
    scopes
        .iter()
        .any(|s| {
            let scope = s.as_str().unwrap_or_default();
            !scope.is_empty() && (scope == path || path.starts_with(&format!("{scope}.")))
        })
        .then_some(value)
}

pub(super) fn object_field<'a>(value: &'a mut Value, field: &str) -> Result<&'a mut Value> {
    let object = value
        .as_object_mut()
        .ok_or_else(|| anyhow!("Configuration must be an object"))?;
    let entry = object.entry(field).or_insert_with(|| json!({}));
    ensure!(entry.is_object(), "{field} must be an object");
    Ok(entry)
}

pub(super) fn output_list(value: &Value) -> Result<Vec<String>> {
    let values = value
        .as_array()
        .ok_or_else(|| anyhow!("requested_outputs must be a nonempty unique list"))?;
    ensure!(!values.is_empty(), "requested_outputs must be nonempty");
    let mut seen = BTreeSet::new();
    values
        .iter()
        .map(|v| {
            let s = v
                .as_str()
                .ok_or_else(|| anyhow!("requested_outputs must contain strings"))?;
            ensure!(
                OUTPUT_IDS.contains(&s) && seen.insert(s),
                "Invalid or duplicate requested_outputs: {s}"
            );
            Ok(s.to_owned())
        })
        .collect()
}

/// Primary/sensitivity contexts a plan may request: the 2026-10-28 cross-line
/// geometry adds 0% as a fourth equal context; radial keeps exactly 25/45/65.
pub fn allowed_thresholds(config: &Record) -> &'static [u64] {
    if config["assays"]["ymaze"]["threshold_geometry"] == crate::events::GEOMETRY_ARM_START {
        &[0, 25, 45, 65]
    } else {
        &[25, 45, 65]
    }
}

/// Without a plan, `assays.ymaze.threshold_geometry` is unset, which made the
/// cross-line 0% context unrepresentable. Inherit the geometry the trials
/// recorded in their scoring provenance; conflicting recordings are an error.
/// Archived `compare_*` variants keep their own geometry and are ignored here.
/// A geometry the plan already states is never replaced.
pub fn inherit_threshold_geometry(config: &mut Record, rows: &[Record]) -> Result<()> {
    if !config["assays"]["ymaze"]["threshold_geometry"].is_null() {
        return Ok(());
    }
    let recorded: BTreeSet<&str> = rows
        .iter()
        .filter(|row| {
            row["assay"] == "ymaze"
                && !row["variant"]
                    .as_str()
                    .is_some_and(|v| v.starts_with("compare_"))
        })
        .filter_map(|row| row["scoring_provenance"]["geometry_version"].as_str())
        .collect();
    ensure!(
        recorded.len() <= 1,
        "trials record conflicting Y-maze threshold geometries: {recorded:?}"
    );
    if let Some(geometry) = recorded.into_iter().next() {
        let geometry = geometry.to_owned();
        let assays = object_field(config, "assays")?;
        object_field(assays, "ymaze")?["threshold_geometry"] = geometry.into();
    }
    Ok(())
}

/// Threshold of a Y-maze variant name (`arm45`, `arm0_dwell1s`, `proximal25`).
pub fn context_threshold_pct(variant: &str) -> Option<u8> {
    let rest = variant
        .strip_prefix("arm")
        .or_else(|| variant.strip_prefix("proximal"))?;
    let digits = rest.len() - rest.trim_start_matches(|c: char| c.is_ascii_digit()).len();
    rest[..digits].parse().ok()
}

pub fn validate_sensitivity_thresholds(value: &Value) -> Result<()> {
    validate_thresholds_in(value, &[25, 45, 65])
}

pub fn validate_thresholds_in(value: &Value, allowed: &[u64]) -> Result<()> {
    let values = value.as_array().ok_or_else(|| {
        anyhow!("Requested sensitivity thresholds must be a unique list of 25/45/65")
    })?;
    let mut seen = BTreeSet::new();
    for value in values {
        let n = value
            .as_u64()
            .ok_or_else(|| anyhow!("Requested sensitivity thresholds must be integers"))?;
        ensure!(
            allowed.contains(&n) && seen.insert(n),
            "Invalid or duplicate sensitivity threshold"
        );
    }
    Ok(())
}

pub fn resolved_config(
    path: &Path,
    options: &RunOptions,
    lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<Record> {
    let (mut config, mut source) = snapshot::read_pinned(path, |file| {
        if path
            .extension()
            .is_some_and(|extension| extension == "toml")
        {
            let mut text = String::new();
            file.read_to_string(&mut text)?;
            Ok(serde_json::to_value(toml::from_str::<toml::Value>(&text)?)?)
        } else {
            Ok(json::from_reader(file)?)
        }
    })?;
    ensure!(
        config.is_object()
            && matches!(
                config["run_mode"].as_str(),
                Some("development" | "production" | "synthetic")
            ),
        "Config run_mode must be development, production or synthetic"
    );
    ensure!(
        config.get(CONFIG_FINGERPRINT).is_none(),
        "Configuration fingerprint is adapter-owned"
    );
    source["role"] = json!("post_analysis_plan");
    config[CONFIG_FINGERPRINT] = source;
    let outputs = object_field(&mut config, "outputs")?;
    let directory = options
        .output
        .as_ref()
        .map(|p| p.to_string_lossy().into_owned())
        .or_else(|| lookup("BEKIPER_POST_ANALYSIS_OUTPUT"));
    let directory = match directory {
        Some(value) => value,
        None => match outputs.get("directory") {
            None | Some(Value::Null) => "post_analysis_output".into(),
            Some(value) => value
                .as_str()
                .context("outputs.directory must be a string")?
                .to_owned(),
        },
    };
    ensure!(
        !directory.trim().is_empty(),
        "outputs.directory must be nonempty"
    );
    outputs["directory"] = directory.into();
    let requested = options
        .requested_outputs
        .as_ref()
        .map(|v| json!(v))
        .or_else(|| {
            lookup("BEKIPER_POST_ANALYSIS_REQUESTED_OUTPUTS")
                .map(|s| json!(s.split(',').map(str::trim).collect::<Vec<_>>()))
        })
        .unwrap_or_else(|| outputs["requested_outputs"].clone());
    if !requested.is_null() {
        output_list(&requested)?;
    }
    outputs["requested_outputs"] = requested;
    let report = object_field(outputs, "report")?;
    report
        .as_object_mut()
        .unwrap()
        .entry("enabled")
        .or_insert(json!(false));
    ensure!(
        report["enabled"].is_boolean(),
        "outputs.report.enabled must be boolean"
    );
    let overrides = options
        .ymaze_dwell
        .with_env(lookup)
        .map_err(anyhow::Error::msg)?;
    resolve_ymaze_dwell(&mut config, overrides)?;
    let primary = match &options.primary_thresholds {
        Some(values) => Some(values.clone()),
        None => lookup("BEKIPER_YMAZE_PRIMARY_THRESHOLDS")
            .map(|raw| {
                if raw.trim().is_empty() {
                    Ok(vec![])
                } else {
                    raw.split(',')
                        .map(|s| s.trim().parse::<u8>())
                        .collect::<std::result::Result<Vec<_>, _>>()
                }
            })
            .transpose()?,
    };
    let workers = match options.primary_workers {
        Some(value) => Some(value),
        None => lookup("BEKIPER_PRIMARY_ANALYSIS_WORKERS")
            .map(|v| v.parse::<usize>())
            .transpose()?,
    };
    resolve_primary_runs(
        &mut config,
        &RunOptions {
            primary_thresholds: primary,
            primary_workers: workers,
            ..Default::default()
        },
    )?;
    let allowed = allowed_thresholds(&config);
    let ymaze = object_field(object_field(&mut config, "assays")?, "ymaze")?;
    let thresholds = if let Some(values) = &options.sensitivity_thresholds {
        json!(values)
    } else if let Some(raw) = lookup("BEKIPER_POST_ANALYSIS_SENSITIVITY_THRESHOLDS") {
        if raw.is_empty() {
            json!([])
        } else {
            json!(
                raw.split(',')
                    .map(|s| s.trim().parse::<u8>())
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .context("Invalid sensitivity thresholds")?
            )
        }
    } else {
        ymaze
            .get("requested_sensitivity_thresholds_arm_length_pct")
            .cloned()
            .unwrap_or(json!(allowed))
    };
    validate_thresholds_in(&thresholds, allowed)?;
    ymaze["requested_sensitivity_thresholds_arm_length_pct"] = thresholds;
    Ok(config)
}

/// Native-record callers supply already-resolved values; no ambient env reread.
pub fn resolve_primary_runs(config: &mut Record, options: &RunOptions) -> Result<()> {
    let allowed = allowed_thresholds(config);
    let execution = object_field(config, "execution")?;
    if let Some(values) = &options.primary_thresholds {
        execution["ymaze_primary_thresholds"] = json!(values);
    }
    if let Some(values) = &options.contexts {
        execution["contexts"] = json!(values);
    }
    if let Some(workers) = options.render_workers {
        execution["render_workers"] = json!(workers);
    }
    if let Some(value) = execution.get("render_workers").filter(|v| !v.is_null()) {
        ensure!(
            value.as_u64().is_some_and(|v| v > 0),
            "render workers must be positive"
        );
    }
    if let Some(workers) = options.primary_workers {
        execution["primary_workers"] = json!(workers);
    }
    if let Some(values) = execution.get("ymaze_primary_thresholds") {
        validate_thresholds_in(values, allowed)?;
    }
    if let Some(values) = execution.get("contexts").filter(|v| !v.is_null()) {
        let specs: Vec<crate::contracts::ContextSpec> = serde_json::from_value(values.clone())
            .context("execution.contexts must be a list of {assay, variant}")?;
        ensure!(
            specs.is_empty()
                || execution["ymaze_primary_thresholds"]
                    .as_array()
                    .is_none_or(Vec::is_empty),
            "execution.contexts replaces ymaze_primary_thresholds; set only one"
        );
        let mut seen = BTreeSet::new();
        for spec in &specs {
            ensure!(
                ["oft", "ymaze", "nort"].contains(&spec.assay.as_str())
                    && !spec.variant.is_empty()
                    && spec
                        .variant
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
                    && seen.insert(spec.variant.clone()),
                "invalid or duplicate context {}/{}",
                spec.assay,
                spec.variant
            );
            if spec.assay == "ymaze" {
                let pct = context_threshold_pct(&spec.variant).with_context(|| {
                    format!(
                        "Y-maze context {} must name its threshold (arm45)",
                        spec.variant
                    )
                })?;
                ensure!(
                    allowed.contains(&u64::from(pct)),
                    "Invalid Y-maze context threshold {pct}"
                );
            }
        }
    }
    if let Some(value) = execution.get("primary_workers") {
        ensure!(
            value.as_u64().is_some_and(|v| v > 0),
            "primary workers must be positive"
        );
    }
    Ok(())
}

/// Apply already-resolved constructor/environment values to a captured or
/// file-backed plan. Native-record execution never rereads process environment.
/// The original file fingerprint remains unchanged; execution evidence records
/// the configured and effective policies separately from scientific approvals.
pub fn resolve_ymaze_dwell(
    config: &mut Record,
    overrides: bekiper_feature::tolerance::DwellOverrides,
) -> Result<()> {
    let ymaze = object_field(object_field(config, "assays")?, "ymaze")?;
    let base: bekiper_feature::tolerance::TemporalPolicy = ymaze
        .get("temporal_policy")
        .map(|v| serde_json::from_value(v.clone()))
        .transpose()?
        .unwrap_or_else(bekiper_feature::tolerance::TemporalPolicy::ymaze_default);
    let policy = overrides.apply(base);
    if ymaze["threshold_geometry"] == crate::events::GEOMETRY_ARM_START {
        policy.validate()
    } else {
        policy.validate_uninterrupted()
    }
    .map_err(anyhow::Error::msg)?;
    ymaze["temporal_policy"] = serde_json::to_value(policy)?;
    // Cross-line contexts prefer their own registered policy: override those too.
    if let Some(contexts) = ymaze
        .get_mut("threshold_policies")
        .and_then(Value::as_object_mut)
    {
        for value in contexts.values_mut() {
            let own: bekiper_feature::tolerance::TemporalPolicy =
                serde_json::from_value(value.clone())?;
            let own = overrides.apply(own);
            own.validate().map_err(anyhow::Error::msg)?;
            *value = serde_json::to_value(own)?;
        }
    }
    if overrides.seconds.is_some() && ymaze.get("principal_minimum_dwell_s").is_some() {
        ymaze["principal_minimum_dwell_s"] = json!(policy.min_duration_seconds);
    }
    if !overrides.is_empty() {
        let evidence = object_field(config, "execution_overrides")?;
        let history = evidence
            .as_object_mut()
            .unwrap()
            .entry("ymaze_dwell")
            .or_insert_with(|| json!([]));
        history
            .as_array_mut()
            .context("execution_overrides.ymaze_dwell must be an array")?
            .push(
                json!({"configured_policy":base,"overrides":overrides,"effective_policy":policy}),
            );
    }
    Ok(())
}

pub fn subgroup_plan(config: &Record) -> Result<Option<Record>> {
    Ok(validated_subgroup_plan(config)?.map(|plan| plan.original.clone()))
}

/// Internal consumers share the types established by the registration checks.
/// The original record is retained verbatim for public APIs and run evidence.
pub(crate) struct SubgroupPlan<'a> {
    pub original: &'a Record,
    pub strata: Vec<SubgroupStratum<'a>>,
    pub outputs: Vec<String>,
    pub family: &'a str,
}

pub(crate) struct SubgroupStratum<'a> {
    pub field: &'a str,
    pub levels: Vec<String>,
    pub assays: Vec<String>,
}

pub(crate) fn validated_subgroup_plan(config: &Record) -> Result<Option<SubgroupPlan<'_>>> {
    let path = "statistics.subgroup_analysis_plan";
    let Some(plan) = choice(config, path, "D09") else {
        return Ok(None);
    };
    if choice(config, path, "D12").is_none() {
        return Ok(None);
    }
    ensure!(plan.is_object(), "Subgroup plan must be an object");
    let strata = plan["strata"]
        .as_array()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| anyhow!("Subgroup plan requires explicit strata"))?;
    let mut seen = BTreeSet::new();
    let mut validated = Vec::new();
    for stratum in strata {
        let field = stratum["field"]
            .as_str()
            .ok_or_else(|| anyhow!("Missing subgroup field"))?;
        ensure!(
            ["sex", "age_treatment_window", "treatment"].contains(&field),
            "Invalid subgroup field"
        );
        let levels = unique_strings(&stratum["levels"], "subgroup levels")?;
        let assays = unique_strings(&stratum["assays"], "subgroup assays")?;
        ensure!(
            assays
                .iter()
                .all(|a| crate::contracts::assays_in_report_order().contains(&a.as_str())),
            "Invalid subgroup assay"
        );
        for level in &levels {
            ensure!(
                seen.insert((field.to_owned(), level.clone())),
                "Duplicate subgroup stratum"
            );
        }
        validated.push(SubgroupStratum {
            field,
            levels,
            assays,
        });
    }
    let outputs = unique_strings(&plan["outputs"], "subgroup outputs")?;
    ensure!(
        outputs
            .iter()
            .all(|o| ["models", "correlations", "within_group_tests"].contains(&o.as_str())),
        "Invalid subgroup outputs"
    );
    let family = plan["multiplicity"]["family"]
        .as_str()
        .filter(|s| !s.is_empty() && plan["multiplicity"]["method"] == "holm")
        .context("Subgroup inference requires a Holm family")?;
    Ok(Some(SubgroupPlan {
        original: plan,
        strata: validated,
        outputs,
        family,
    }))
}

pub(super) fn unique_strings(value: &Value, name: &str) -> Result<Vec<String>> {
    let array = value
        .as_array()
        .filter(|a| !a.is_empty())
        .ok_or_else(|| anyhow!("{name} requires explicit values"))?;
    let mut seen = BTreeSet::new();
    array
        .iter()
        .map(|v| {
            let s = v
                .as_str()
                .filter(|s| !s.is_empty())
                .ok_or_else(|| anyhow!("{name} requires strings"))?;
            ensure!(seen.insert(s), "Duplicate {name}");
            Ok(s.to_owned())
        })
        .collect()
}

pub fn requested_output_ids(config: &Record, rows: &[Record]) -> Result<Vec<String>> {
    if let Some(value) = at(config, "outputs.requested_outputs").filter(|v| !v.is_null()) {
        return output_list(value);
    }
    let mut result = vec!["metrics", "models", "figures"];
    let registered = |assay: &str| {
        config["inputs"]
            .get(format!("{assay}_path"))
            .is_some_and(|v| !v.is_null())
            || at(config, &format!("assays.{assay}.input_level")).is_some_and(|v| !v.is_null())
            || rows.iter().any(|r| r["assay"] == assay)
    };
    if registered("ymaze") {
        result.push("correlations");
    }
    if registered("nort") {
        result.push("within_group_tests");
    }
    if at(config, "statistics.subgroup_analysis_plan").is_some_and(|v| !v.is_null()) {
        result.push("subgroup_inference");
    }
    if at(config, "outputs.report.enabled")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        result.push("report");
    }
    Ok(result.into_iter().map(str::to_owned).collect())
}
