//! Pairwise QC scatter populations.
use super::*;

pub(super) fn scatter_panel(
    source: &[&Record],
    x: &str,
    y: &str,
    policy: &PlotPolicy<'_>,
) -> Result<Record> {
    let mut points = Vec::new();
    let mut ledger = Vec::new();
    let mut missing_count = 0;
    let mut animals = BTreeSet::new();
    for row in source {
        let mut row_ledger = Vec::new();
        if let Some(mut point) = point_for(row, y, policy, &mut row_ledger, &mut missing_count)? {
            if let Some(value) = number(row, x)? {
                ensure!(
                    animals.insert(string(row, "animal_id")?.to_owned()),
                    "Scatter panel contains repeated animal replicates"
                );
                point["x"] = value.into();
                points.push(point);
            } else {
                missing_count += 1;
                if let Some(entry) = row_ledger.last_mut() {
                    entry["reason"] = missing_reason(row, x).into();
                }
            }
        }
        ledger.extend(row_ledger);
    }
    points.sort_by_key(|point| point["animal_id"].as_str().unwrap_or("").to_owned());
    Ok(
        json!({"endpoint":y,"x_endpoint":x,"label":format!("{} vs {}",endpoint_label(y),endpoint_label(x)),"kind":"scatter","reference_line":if y=="DI" {Some(0.0)} else {None},
        "x_label":endpoint_label(x),"y_label":endpoint_label(y),"points":points,"summaries":[],"n_animals":animals.len(),
        "status":if animals.is_empty() {"unavailable_or_blocked"} else if missing_count>0 {"partial"} else {"rendered"},
        "missing_observations":missing_count,"eligibility_ledger":ledger,"population_policy":policy.population}),
    )
}
