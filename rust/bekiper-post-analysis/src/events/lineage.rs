//! Registered scientific policies and retained upstream lineage.
use super::*;

pub(super) fn evidence_number(evidence: &Value, name: &str) -> Result<Option<f64>> {
    if evidence[name].is_null() {
        Ok(None)
    } else {
        Ok(Some(nonnegative(&evidence[name], name, None)?))
    }
}

pub fn complete_landmark_roles(roles: &Record) -> bool {
    let mut columns = BTreeSet::new();
    for name in ["head_center", "tail_base"] {
        let Some(mapping) = roles[name].as_object() else {
            return false;
        };
        if mapping.keys().map(String::as_str).collect::<BTreeSet<_>>()
            != ["x", "y"].into_iter().collect()
        {
            return false;
        }
        for value in mapping.values() {
            let Some(column) = value.as_str() else {
                return false;
            };
            if column.trim().is_empty() || !columns.insert(column) {
                return false;
            }
        }
    }
    true
}

pub fn ymaze_provenance(row: &Record, landmark_roles: &Record) -> Result<Record> {
    ymaze_provenance_with_policy(
        row,
        landmark_roles,
        &bekiper_feature::tolerance::TemporalPolicy::ymaze_default(),
    )
}

pub fn ymaze_provenance_with_policy(
    row: &Record,
    landmark_roles: &Record,
    policy: &bekiper_feature::tolerance::TemporalPolicy,
) -> Result<Record> {
    ymaze_provenance_registered(row, landmark_roles, policy, GEOMETRY, 0.0)
}

/// `geometry` is the plan's registered threshold geometry and
/// `registered_zone_gap_s` its registered arm-sequence gap bridge. Radial plans
/// register 0; a zone gap equal to the registered value is policy, not mitigation.
pub fn ymaze_provenance_registered(
    row: &Record,
    landmark_roles: &Record,
    policy: &bekiper_feature::tolerance::TemporalPolicy,
    geometry: &str,
    registered_zone_gap_s: f64,
) -> Result<Record> {
    let empty = json!({});
    let evidence = row
        .get("scoring_provenance")
        .filter(|v| !v.is_null())
        .unwrap_or(&empty);
    let object = evidence
        .as_object()
        .context("scoring_provenance must be an object")?;
    if object.is_empty() {
        return Ok(json!({"status":"unresolved","reasons":["missing_scoring_provenance"]}));
    }
    let mut incompatible = Vec::new();
    if !evidence["geometry_version"].is_null() && evidence["geometry_version"] != geometry {
        incompatible.push("different_geometry");
    }
    if (evidence["apply_dwell_filter"] == true) != (policy.min_duration_seconds > 0.0)
        || evidence_number(evidence, "minimum_dwell_seconds")?.unwrap_or(0.0)
            != policy.min_duration_seconds
    {
        incompatible.push("incompatible_dwell_filter");
    }
    if evidence["requires_both_landmarks_same_arm"] == false {
        incompatible.push("single_landmark_or_proxy");
    }
    let mitigated = evidence["arm_membership_hysteresis"] == true
        || evidence_number(evidence, "zone_max_gap_seconds")?.unwrap_or(0.0)
            != registered_zone_gap_s
        || evidence_number(evidence, "pose_gap_fill_seconds")?.unwrap_or(0.0) > 0.0;
    if let Some(fraction) = evidence_number(evidence, "proximal_threshold")?
        && Some(fraction * 100.0) != row["threshold_pct"].as_f64()
    {
        incompatible.push("threshold_identity_mismatch");
    }
    if !incompatible.is_empty() {
        return Ok(
            json!({"status":"requires_regeneration_or_registered_amendment","reasons":incompatible}),
        );
    }
    if mitigated {
        return Ok(
            json!({"status":"requires_event_set_audit","reasons":["historical_temporal_mitigation"]}),
        );
    }
    let checks = BTreeMap::from([
        ("geometry", evidence["geometry_version"] == geometry),
        (
            "dual_landmarks",
            evidence["requires_both_landmarks_same_arm"] == true,
        ),
        (
            if policy.min_duration_seconds == 0.0 {
                "no_dwell"
            } else {
                "dwell_policy"
            },
            evidence["apply_dwell_filter"] == (policy.min_duration_seconds > 0.0)
                && evidence_number(evidence, "minimum_dwell_seconds")?
                    == Some(policy.min_duration_seconds)
                && (policy.min_duration_seconds == 0.0
                    || (same_value(&evidence["temporal_policy"], &serde_json::to_value(policy)?)
                        && evidence["temporal_policy_version"] == "sample_support_v1")),
        ),
        (
            "landmark_roles",
            complete_landmark_roles(landmark_roles)
                && same_value(&evidence["head_columns"], &landmark_roles["head_center"])
                && same_value(&evidence["tail_columns"], &landmark_roles["tail_base"]),
        ),
        (
            "version",
            [
                row.get("upstream_revision"),
                evidence.get("code_revision"),
                evidence.get("entry_implementation_version"),
            ]
            .into_iter()
            .flatten()
            .any(|v| v.as_str().is_some_and(|s| !s.trim().is_empty())),
        ),
        (
            "no_temporal_mitigation",
            evidence["arm_membership_hysteresis"] == false
                && evidence_number(evidence, "zone_max_gap_seconds")?
                    == Some(registered_zone_gap_s)
                && evidence_number(evidence, "pose_gap_fill_seconds")? == Some(0.0),
        ),
        ("transition_rule", evidence["entry_rule"] == ENTRY_RULE),
    ]);
    Ok(
        json!({"status":if checks.values().all(|v| *v) {"verified_clarified_entry_definition"} else {"unresolved"},
        "reasons":checks.iter().filter(|(_, passed)| !**passed).map(|(name, _)| *name).collect::<Vec<_>>(),"evidence":evidence}),
    )
}

/// Documents rule-definition verification only, never independent event-set
/// verification from the mere presence of candidate intervals or annotations.
pub fn assay_provenance(row: &Record, config: &Record) -> Result<Record> {
    let mut expected = if row["assay"] == "oft" {
        json!({"arena_cm":[40,40,34],"center_cm":[24,24],"border_cm":8,"entry_landmark":"body_centroid",
            "minimum_dwell_seconds":1,"time_origin":"first_valid_detection","analysis_duration_s":300})
    } else {
        json!({"zone_extension_m":bekiper_feature::heuristic::combined::exploration::DEFAULT_NOSE_MAX_DISTANCE,"body_centroid_outside_object_mask":true,"candidate_orientation_or_contact":true,
            "angle_aggregation":"bout_median","angle_max_degrees":bekiper_feature::heuristic::combined::exploration::IMEN_MAX_ANGLE_DEG,"merge_gap_seconds":IMEN_MAX_GAP_SECONDS,"merge_gap_comparison":"<=",
            "merge_before_duration":true,"minimum_dwell_seconds":IMEN_MIN_BOUT_SECONDS,"duration_comparison":">=","speed_filter_mode":"disabled",
             "behaviour_exclusions":["climbing","sitting_on_object","grooming","rearing"],"object_directed_exception":true})
    };
    // Native measurements must not acquire verification for a different
    // registered gap policy merely because their duration/speed limits match.
    if row["assay"] == "nort" {
        // Optional plan key `assays.nort.zone_extension_m` (nose reach, m); the
        // default is the Imen reach, which equals the generic nose distance.
        if let Some(reach) = config["assays"]["nort"].get("zone_extension_m") {
            expected["zone_extension_m"] = reach.clone();
        }
        if let Some(value) = config["assays"]["nort"].get("bout_policy") {
            let policy: bekiper_feature::heuristic::combined::exploration::NortBoutPolicy =
                serde_json::from_value(value.clone())?;
            policy.validate().map_err(anyhow::Error::msg)?;
            expected["minimum_dwell_seconds"] = json!(policy.min_bout_seconds);
            expected["merge_gap_seconds"] = json!(policy.max_gap_seconds);
            expected["merge_gap_comparison"] =
                json!(if policy.strict_boundaries { "<" } else { "<=" });
            expected["speed_filter_mode"] = json!(policy.speed_filter_mode);
            expected["angle_max_degrees"] = json!(policy.max_angle_deg);
            if policy.whisker_enabled {
                // Whisker-fan hits union into observation; geometry (length, age) is
                // recorded by the native `whisker_fan` provenance, not re-derived here.
                expected["bout_contact_policy"] = json!(
                    bekiper_feature::heuristic::combined::exploration::BOUT_CONTACT_POLICY_WHISKER
                );
            }
            if policy.speed_filter_mode
                != bekiper_feature::heuristic::combined::exploration::BoutFilterMode::Disabled
            {
                expected["diagnostic_nose_speed_reference_m_per_s"] = json!(policy.max_nose_speed);
                expected["speed_comparison"] =
                    json!(if policy.strict_boundaries { "<" } else { "<=" });
            }
        }
        for (path, field) in [
            (
                "assays.nort.merged_gap_duration_policy",
                "gap_duration_policy",
            ),
            ("assays.nort.merged_gap_median_policy", "gap_median_policy"),
        ] {
            if let Some(policy) = crate::inputs::choice(config, path, "D07") {
                expected[field] = policy.clone();
            }
        }
    }
    let empty = json!({});
    let evidence = row
        .get("scoring_provenance")
        .filter(|v| !v.is_null())
        .unwrap_or(&empty);
    let audit = row
        .get("_rule_audit")
        .filter(|v| !v.is_null())
        .unwrap_or(&empty);
    ensure!(
        evidence.is_object() && audit.is_object(),
        "Rule provenance and audit must be objects"
    );
    let approved = matches!(
        audit["origin"].as_str(),
        Some("existing_analysis_plan" | "researcher_decision")
    ) || (audit["origin"] == "development_fixture"
        && config["run_mode"] == "synthetic"
        && config["synthetic_fixture"] == true);
    let nonempty = |value: &Value| value.as_str().is_some_and(|s| !s.trim().is_empty());
    let mut checks = Map::new();
    checks.insert(
        "rule_audit".into(),
        json!(
            approved
                && nonempty(&audit["approval_or_plan_reference"])
                && nonempty(&audit["rationale"])
                && audit["rules"].is_object()
                && nonempty(&audit["code_revision"])
        ),
    );
    checks.insert(
        "version".into(),
        json!(
            nonempty(&evidence["code_revision"])
                && evidence["code_revision"] == audit["code_revision"]
        ),
    );
    for (name, expected) in expected.as_object().context("rule definition object")? {
        checks.insert(
            name.clone(),
            json!(
                same_value(&evidence[name], expected)
                    && same_value(&audit["rules"][name], expected)
            ),
        );
    }
    let empty_files = Vec::new();
    let files = match row.get("_rule_evidence_files") {
        None | Some(Value::Null) => &empty_files,
        Some(value) => value
            .as_array()
            .context("Rule evidence files must be a list")?,
    };
    let available: BTreeSet<_> = files.iter().filter_map(|f| f["role"].as_str()).collect();
    let mut required: BTreeSet<_> = [
        "rule_audit",
        "implementation",
        "geometry",
        "landmarks",
        "window",
    ]
    .into_iter()
    .collect();
    if row["assay"] == "nort" {
        required.insert("annotations");
    }
    let sources_valid = required.is_subset(&available)
        && files.iter().all(|f| {
            f["sha256"].as_str().is_some_and(|s| {
                s.len() == 64
                    && s.bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            }) && f["bytes"].as_u64().is_some_and(|n| n > 0)
                && nonempty(&f["name"])
        });
    checks.insert("supporting_sources".into(), json!(sources_valid));
    let verified = checks.values().all(|v| v == &Value::Bool(true));
    let reasons: Vec<_> = checks
        .iter()
        .filter(|(_, passed)| **passed != Value::Bool(true))
        .map(|(name, _)| name.clone())
        .collect();
    Ok(
        json!({"status":if verified {"verified_source_rule_definition"} else {"upstream_unverified"},
        "reasons":reasons,"level":"documented_rule_audit_not_event_set_audit","checks":checks,
        "evidence_files":files,"audit_reference":audit["approval_or_plan_reference"],"evidence":evidence}),
    )
}
