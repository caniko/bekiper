//! Per-animal summary figures: each recorded criteria endpoint of one animal
//! against its group's recorded values, pre and post, for every protocol step
//! (from the experiment descriptors; Imen: habituation, T1, T2, Y-maze). Rust selects the recorded values; the Python
//! backend (`bekiper-inspect animal-summary`) draws them. Nothing is recomputed.
use anyhow::Result;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, path::Path};

use super::{
    Cancel, PackageArgs,
    index::{self, Recording},
    run_job_shards, sha256_file,
};

/// Must match `ANIMAL_SUMMARY_VERSION` in `bekiper_inspect/plot/animal.py`.
pub(crate) const ANIMAL_SUMMARY_VERSION: &str = "1";

/// Endpoints per (assay, step) panel row.
const PER_STEP: usize = 4;


fn finite(row: &Value, key: &str) -> Option<f64> {
    row[key].as_f64().filter(|v| v.is_finite())
}

/// Remove entries of `dir` that belong to none of `keep` (`<name>` or `<name>.<ext>`), so files
/// of earlier runs are never sealed with this one.
pub(crate) fn prune_stale<'a>(dir: &Path, keep: impl IntoIterator<Item = &'a str> + Clone) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let kept = keep.clone().into_iter().any(|k| {
            name == k
                || name
                    .strip_prefix(k)
                    .is_some_and(|rest| rest.starts_with('.'))
        });
        if !kept {
            let path = entry.path();
            let _ = if path.is_dir() {
                fs::remove_dir_all(path)
            } else {
                fs::remove_file(path)
            };
        }
    }
}

/// One job per animal; panels only where the animal has a recorded value.
pub(crate) fn jobs(
    animals: &[String],
    metrics: &BTreeMap<String, Value>,
    groups: &BTreeMap<String, String>,
    presentation: &bekiper_core::experiment::ResolvedPresentation,
) -> BTreeMap<String, Value> {
    // Panel rows in protocol order: every step of every experiment.
    let steps: Vec<_> = presentation
        .experiments
        .iter()
        .flat_map(|e| e.steps.iter().map(move |s| (e, s)))
        .collect();
    let on_step = |row: &Value, assay: &str, step: &str| {
        row["assay"] == assay && index::row_step(presentation, row) == Some(step)
    };
    let mut by_animal: BTreeMap<String, Vec<&Value>> = BTreeMap::new();
    for row in metrics.values() {
        if let Some(animal) = index::text(&row["animal_id"]) {
            by_animal.entry(animal).or_default().push(row);
        }
    }
    let mut out = BTreeMap::new();
    for animal in animals {
        let rows = by_animal.get(animal).map(Vec::as_slice).unwrap_or_default();
        let group = rows.iter().find_map(|r| index::text(&r["group"]));
        let mut panels = Vec::new();
        for (experiment, step) in &steps {
            let (assay, endpoints) = (&experiment.key, &experiment.endpoints);
            let mine: Vec<&&Value> = rows
                .iter()
                .filter(|r| on_step(r, assay, &step.id))
                .collect();
            let mut taken = 0;
            for endpoint in endpoints {
                if taken == PER_STEP {
                    break;
                }
                let by_time = |source: &[&&Value]| -> Value {
                    let mut values = serde_json::Map::new();
                    for row in source {
                        if let (Some(t), Some(v)) =
                            (index::text(&row["timepoint"]), finite(row, endpoint))
                        {
                            values.insert(t, json!(v));
                        }
                    }
                    Value::Object(values)
                };
                let animal_values = by_time(&mine);
                if animal_values.as_object().is_none_or(|m| m.is_empty()) {
                    continue;
                }
                let mut cohort: BTreeMap<String, Vec<f64>> = BTreeMap::new();
                for row in metrics.values().filter(|r| {
                    on_step(r, assay, &step.id)
                        && group.is_some()
                        && index::text(&r["group"]) == group
                }) {
                    if let (Some(t), Some(v)) =
                        (index::text(&row["timepoint"]), finite(row, endpoint))
                    {
                        cohort.entry(t).or_default().push(v);
                    }
                }
                for values in cohort.values_mut() {
                    values.sort_by(f64::total_cmp);
                }
                panels.push(json!({
                    "assay": assay,
                    "step": step.label,
                    "endpoint": endpoint,
                    "animal": animal_values,
                    "group": cohort,
                }));
                taken += 1;
            }
        }
        let group_label = index::group_label(rows.first().copied(), groups);
        out.insert(
            animal.clone(),
            json!({"title": format!("Animal {animal}"), "subtitle": group_label, "panels": panels}),
        );
    }
    out
}

/// Render (or reuse) every animal's summary; returns `animal -> {path, width, height}`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn render(
    args: &PackageArgs,
    recordings: &[Recording],
    metrics: &BTreeMap<String, Value>,
    groups: &BTreeMap<String, String>,
    presentation: &bekiper_core::experiment::ResolvedPresentation,
    mode: themekit::Mode,
    workers: usize,
    cancel: &Cancel,
) -> Result<BTreeMap<String, Value>> {
    let mut animals: Vec<String> = recordings.iter().filter_map(|r| r.animal.clone()).collect();
    animals.sort();
    animals.dedup();
    let slugs: Vec<String> = animals.iter().map(|a| index::slug(a)).collect();
    prune_stale(
        &args.output.join("figures/animals"),
        slugs.iter().map(String::as_str),
    );
    let mut pending = Vec::new();
    let mut outputs = BTreeMap::new();
    for (animal, mut job) in jobs(&animals, metrics, groups, presentation) {
        let mut hasher = Sha256::new();
        hasher.update(ANIMAL_SUMMARY_VERSION);
        hasher.update(mode.as_str());
        hasher.update(serde_json::to_vec(&job)?);
        let key = format!("{:x}", hasher.finalize());
        let relative = format!("figures/animals/{}", index::slug(&animal));
        let out = args.output.join(&relative);
        let fresh = fs::read(out.with_extension("json"))
            .ok()
            .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
            .is_some_and(|s| {
                s["key"] == json!(key)
                    && sha256_file(&out.with_extension("webp")).ok().as_deref()
                        == s["webp_sha256"].as_str()
            });
        if !fresh || args.preview.is_some() {
            job["key"] = json!(key);
            job["out"] = json!(out);
            job["theme"] = json!(mode.as_str());
            pending.push(job);
        }
        outputs.insert(animal, (relative, out));
    }
    run_job_shards(args, "animal-summary", pending, workers, cancel)?;
    let mut figures = BTreeMap::new();
    for (animal, (relative, out)) in outputs {
        if let Some(sidecar) = fs::read(out.with_extension("json"))
            .ok()
            .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
        {
            figures.insert(
                animal,
                json!({"path": format!("{relative}.webp"), "width": sidecar["width"], "height": sidecar["height"]}),
            );
        }
    }
    Ok(figures)
}
