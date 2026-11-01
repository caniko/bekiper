//! Canonical column names, cell decoding and exact identifiers.
use super::*;

const ALIASES: &[(&str, &str)] = &[
    ("animal", "animal_id"),
    ("Animal ID", "animal_id"),
    ("5XFAD_status", "five_xfad"),
    ("5XFAD", "five_xfad"),
    ("HCAR1_status", "hcar1"),
    ("HCAR1", "hcar1"),
    ("total_mean_speed", "mean_speed_m_per_s"),
    ("center_duration_s", "center_time_s"),
    ("latency_to_center_s", "latency_to_first_center_entry_s"),
    ("latency_to_center", "latency_to_first_center_entry_s"),
    ("novel_s", "TN"),
    ("familiar_s", "TF"),
    ("pct_novel", "percent_novel"),
    ("log_ratio", "LR"),
    ("total_duration_s", "analysis_duration_s"),
];

fn alias(name: &str) -> &str {
    ALIASES
        .iter()
        .find(|(alias, _)| *alias == name)
        .map_or(name, |(_, target)| *target)
}

pub(super) fn numeric(name: &str) -> bool {
    [
        "TA",
        "TB",
        "TN",
        "TF",
        "DI",
        "RI",
        "LR",
        "object_bias",
        "side_bias",
        "percent_novel",
        "threshold_pct",
        "proximal_threshold",
        "center_entries",
        "fecal_boli_count",
        "total_entries",
        "valid_entries",
        "alternations",
        "standard_alternations",
        "neutral_triads",
        "possible_alternations",
        "total_arm_entries",
        "valid_arm_entries",
        "n_valid_entries",
        "n_standard_alternations",
        "n_neutral_triads",
        "n_possible_alternations",
        "raw_total_arm_entries",
        "valid_total_arm_entries",
        "removed_short_entries",
        "raw_to_scored_entry_difference",
        "dwell_rejected_entries",
        "total_triads",
        "usable_triads",
        "correct_alternations",
        "incorrect_alternations",
        "raw_arm_A_entries",
        "raw_arm_B_entries",
        "raw_arm_C_entries",
        "valid_arm_A_entries",
        "valid_arm_B_entries",
        "valid_arm_C_entries",
        "valid_bouts",
        "removed_duration_bouts",
        "removed_speed_bouts",
        "removed_angle_bouts",
    ]
    .contains(&name)
        || ["_s", "_m", "_pct", "_m_per_s"]
            .iter()
            .any(|suffix| name.ends_with(suffix))
}

pub fn normalize(row: &Record, mapping: &Record) -> Result<Record> {
    normalize_owned(row.clone(), mapping)
}

/// Explicit plan value dictionaries, shared by table and native-session joins.
pub fn apply_session_value_mappings(row: &mut Record, mappings: &Record) -> Result<()> {
    if mappings.is_null() {
        return Ok(());
    }
    let mappings = mappings
        .as_object()
        .context("session_key_mapping must be an object")?;
    for (field, mapping) in mappings {
        ensure!(
            [
                "assay",
                "session_id",
                "phase",
                "timepoint",
                "group",
                "sex",
                "treatment",
                "hcar1",
                "five_xfad",
                "age_treatment_window",
            ]
            .contains(&field.as_str())
                && mapping.is_object(),
            "session_key_mapping requires session or biological-factor value dictionaries"
        );
        ensure!(
            mapping
                .as_object()
                .unwrap()
                .values()
                .all(|value| value.as_str().is_some_and(|s| !s.trim().is_empty())),
            "session_key_mapping targets must be nonempty strings"
        );
        if let Some(value) = row
            .get(field)
            .filter(|value| value.is_string() || value.is_number())
        {
            let source = canonical_string(value)?;
            if let Some(mapped) = mapping.get(&source) {
                row[field] = mapped.clone();
            }
        }
    }
    Ok(())
}

pub(super) fn normalize_owned(row: Record, mapping: &Record) -> Result<Record> {
    let Value::Object(row) = row else {
        bail!("Table row must be an object")
    };
    ensure!(
        mapping.is_null() || mapping.is_object(),
        "Column mapping must be an object"
    );
    let mut result = Map::new();
    for (source, value) in row {
        let target = match mapping.get(&source) {
            Some(value) => value
                .as_str()
                .filter(|s| !s.trim().is_empty())
                .ok_or_else(|| anyhow!("Column mapping targets must be strings"))?,
            None => alias(&source),
        };
        ensure!(
            !result.contains_key(target),
            "Column alias collision for {target}"
        );
        result.insert(target.to_owned(), normalize_cell(value, target)?);
    }
    Ok(Value::Object(result))
}

fn normalize_cell(mut value: Value, field: &str) -> Result<Value> {
    if value.is_null() || value.as_str().is_some_and(|s| s.trim().is_empty()) {
        return Ok(Value::Null);
    }
    if numeric(field) {
        if let Some(text) = value.as_str() {
            value = numeric_text(text, field)?;
        }
        ensure!(
            value.as_f64().is_some_and(f64::is_finite),
            "{field}: requires a finite number"
        );
    } else if [
        "excluded",
        "passes_15s_exploration_threshold",
        "low_quality",
    ]
    .contains(&field)
    {
        if let Some(text) = value.as_str() {
            value = Value::Bool(
                text.trim()
                    .to_lowercase()
                    .parse()
                    .map_err(|_| anyhow!("{field}: must be boolean"))?,
            );
        }
        ensure!(value.is_boolean(), "{field}: must be boolean");
    } else if [
        "scoring_provenance",
        "qc_flags",
        "missing_reasons",
        "coordinate_samples",
        "candidate_segments",
        "center_intervals",
    ]
    .contains(&field)
    {
        if let Some(text) = value.as_str() {
            value = json::from_str(text)?;
        }
    } else if ["valid_arm_entry_sequence", "raw_arm_entry_sequence"].contains(&field) {
        if let Some(text) = value.as_str() {
            value = if text.starts_with('[') {
                json::from_str(text)?
            } else if text.chars().any(char::is_whitespace) {
                json!(text.split_whitespace().collect::<Vec<_>>())
            } else {
                json!(text.chars().map(|c| c.to_string()).collect::<Vec<_>>())
            };
        }
    } else if [
        "animal_id",
        "session_id",
        "pair_id",
        "group",
        "five_xfad",
        "hcar1",
        "treatment",
    ]
    .contains(&field)
    {
        value =
            Value::String(canonical_string(&value).with_context(|| format!("Invalid {field}"))?);
    }
    Ok(value)
}

pub(super) fn numeric_text(text: &str, field: &str) -> Result<Value> {
    let text = text.trim();
    // Integral exports need not take a lossy floating-point round trip.
    if let Ok(value) = text.parse::<u64>() {
        return Ok(json!(value));
    }
    if let Ok(value) = text.parse::<i64>() {
        return Ok(json!(value));
    }
    let value = text
        .parse::<f64>()
        .with_context(|| format!("{field}: invalid numeric value"))?;
    ensure!(value.is_finite(), "{field}: nonfinite supplied value");
    Ok(json!(value))
}

pub(super) fn canonical_string(value: &Value) -> Result<String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::Number(n) if n.is_i64() || n.is_u64() => Ok(n.to_string()),
        Value::Number(n) => {
            let value = n.as_f64().context("Invalid numeric identifier")?;
            ensure!(
                value.is_finite() && value.abs() <= 9_007_199_254_740_991.0,
                "Numeric identifier exceeds exact floating-point range; supply a string or integer"
            );
            Ok(if value.fract() == 0.0 {
                format!("{value:.0}")
            } else {
                n.to_string()
            })
        }
        _ => bail!("Identifiers and factors must be strings or numbers"),
    }
}

pub fn session_key(row: &Record) -> Result<SessionKey> {
    ensure!(row.is_object(), "Session must be an object");
    for field in KEY {
        ensure!(
            row.get(field).is_some_and(|v| !v.is_null()
                && !v.as_str().is_some_and(|s| s.trim().is_empty())
                && (v.is_string() || v.is_number())),
            "Session key requires {KEY:?}; map assay/timepoint/phase explicitly"
        );
    }
    let assay = row["assay"].as_str().unwrap_or_default();
    ensure!(
        crate::contracts::assays_in_report_order().contains(&assay),
        "Canonical assay must be oft, ymaze or nort"
    );
    ensure!(
        matches!(row["timepoint"].as_str(), Some("pre" | "post")),
        "Canonical timepoint must be pre or post"
    );
    ensure!(
        if assay == "nort" {
            matches!(row["phase"].as_str(), Some("training" | "test"))
        } else {
            row["phase"] == "session"
        },
        "Invalid canonical phase for {assay}"
    );
    Ok([
        canonical_string(&row[KEY[0]])?,
        canonical_string(&row[KEY[1]])?,
        canonical_string(&row[KEY[2]])?,
        canonical_string(&row[KEY[3]])?,
        canonical_string(&row[KEY[4]])?,
    ])
}
