//! Categorical point populations and missingness.
use super::*;

pub(super) fn object_panel(
    source: &[&Record],
    roles: [(&str, &str); 2],
    label: &str,
    policy: &PlotPolicy<'_>,
) -> Result<Record> {
    let mut points = Vec::new();
    let mut ledger = Vec::new();
    let mut missing_count = 0;
    for row in source {
        let identity = Value::Array(KEY.iter().map(|key| row[*key].clone()).collect()).to_string();
        for (order, (field, role)) in roles.iter().enumerate() {
            if let Some(mut point) = point_for(row, field, policy, &mut ledger, &mut missing_count)?
            {
                point["category"] = format!("{}:{role}", string(row, "group")?).into();
                point["object_role"] = (*role).into();
                point["source_endpoint"] = (*field).into();
                point["pair_series"] = true.into();
                point["plot_pair_id"] = identity.clone().into();
                point["pair_order"] = order.into();
                points.push(point);
            }
        }
    }
    finish_categorical(
        "exploration_s",
        label,
        None,
        points,
        ledger,
        missing_count,
        policy,
    )
}

pub(super) fn categorical_panel(
    source: &[&Record],
    field: &str,
    label: &str,
    reference: Option<f64>,
    policy: &PlotPolicy<'_>,
) -> Result<Record> {
    let mut points = Vec::new();
    let mut ledger = Vec::new();
    let mut missing_count = 0;
    for row in source {
        if let Some(point) = point_for(row, field, policy, &mut ledger, &mut missing_count)? {
            points.push(point);
        }
    }
    finish_categorical(
        field,
        label,
        reference,
        points,
        ledger,
        missing_count,
        policy,
    )
}

pub(super) fn finish_categorical(
    field: &str,
    label: &str,
    reference: Option<f64>,
    mut points: Vec<Record>,
    ledger: Vec<Record>,
    missing_count: usize,
    policy: &PlotPolicy<'_>,
) -> Result<Record> {
    points.sort_by_key(|point| {
        (
            point["category"].as_str().unwrap_or("").to_owned(),
            point["animal_id"].as_str().unwrap_or("").to_owned(),
            point["session_id"].as_str().unwrap_or("").to_owned(),
        )
    });
    let mut categories: BTreeMap<String, Vec<&Record>> = BTreeMap::new();
    let mut identities = BTreeSet::new();
    for point in &points {
        let category = string(point, "category")?;
        ensure!(
            identities.insert((
                category.to_owned(),
                string(point, "animal_id")?.to_owned(),
                if policy.population == "recorded_descriptive" {
                    format!("{}:{}", point["session_id"], point["analysis_id"])
                } else {
                    String::new()
                }
            )),
            "Plot category contains repeated animal replicates; register pairing/aggregation first"
        );
        categories.entry(category.into()).or_default().push(point);
    }
    let mut categories = categories.into_iter().collect::<Vec<_>>();
    // Registered series order takes precedence over lexical category order:
    // pre precedes post, and familiar precedes novel within each group.
    categories.sort_by_key(|(category, points)| {
        (
            points[0]["group"].as_str().unwrap_or("").to_owned(),
            points[0]["pair_order"].as_u64().unwrap_or(0),
            category.clone(),
        )
    });
    let summaries = categories
        .iter()
        .map(|(category, values)| summary(category, values, policy))
        .collect::<Result<Vec<_>>>()?;
    let complete = !points.is_empty()
        && missing_count == 0
        && summaries
            .iter()
            .all(|summary| summary["status"] == "available");
    let n_animals = points
        .iter()
        .map(|point| string(point, "animal_id"))
        .collect::<Result<BTreeSet<_>>>()?
        .len();
    Ok(
        json!({"endpoint":field,"label":label,"y_label":endpoint_label(field),"reference_line":reference,"kind":"categorical","points":points,
        "summaries":summaries,"n_animals":n_animals,"status":if complete {"rendered"} else if n_animals==0 {"unavailable_or_blocked"} else {"partial"},
        "missing_observations":missing_count,"eligibility_ledger":ledger,"population_policy":policy.population,"summary_uncertainty":policy.uncertainty}),
    )
}
