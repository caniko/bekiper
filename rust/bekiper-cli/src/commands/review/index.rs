//! Trial discovery, metric joins, recording grouping and composite headers.
//! Steps, contexts and labels come from the experiment descriptors
//! (`bekiper_core::experiment`), resolved once per package by [`present`].
use anyhow::{Context, Result};
use bekiper_core::experiment::{
    self, CorpusFacts, Placement, ResolvedExperiment, ResolvedPresentation, TrialFacts,
};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

pub(crate) struct Trial {
    /// Directory name under `data/` (the recorded analysis ID).
    pub id: String,
    pub dir: PathBuf,
    /// Experiment key (`oft`, `nort`, `ymaze`, ...; an unknown task type keeps its name).
    pub assay: String,
    /// `trial_context.json` `experiment.task_type`.
    pub task_type: String,
    /// Detection recipe (`nort_training`, `oul_update`, ...).
    pub recipe: Option<String>,
    /// Source video; recordings group their scoring variants by it.
    pub video: String,
    pub variant: Option<String>,
    /// Recorded post-analysis metrics row, when the cohort run has one.
    pub row: Option<Value>,
    /// `trial_context.json` `session_context` (animal, phase_part, stage, ...).
    pub session: Value,
    /// Feature kinds and perimeter labels of the inspection manifest.
    pub kinds: BTreeSet<String>,
    pub perimeters: Vec<String>,
    /// Where the presentation places the trial; set by [`present`].
    pub placement: Option<Placement>,
}

impl Trial {
    pub fn descriptor(&self) -> &'static experiment::ExperimentDescriptor {
        experiment::descriptor_for(&self.task_type)
    }

    /// Context label of the scoring variant (`45%`, `0% (no dwell)`, `recorded`).
    pub fn variant_label(&self) -> String {
        self.placement.as_ref().map_or_else(
            || self.descriptor().context_label(self.variant.as_deref()),
            |p| p.context.clone(),
        )
    }

    fn principal(&self) -> bool {
        self.placement.as_ref().is_some_and(|p| p.principal)
    }

    fn facts(&self, videos: bool) -> TrialFacts {
        let field = |key: &str| self.row.as_ref().and_then(|r| text(&r[key]));
        TrialFacts {
            id: self.id.clone(),
            task_type: self.task_type.clone(),
            recipe: self.recipe.clone(),
            phase: field("phase"),
            stage: text(&self.session["stage"]),
            timepoint: field("timepoint"),
            phase_part: text(&self.session["phase_part"]),
            variant: self.variant.clone(),
            analysis_role: None,
            feature_kinds: self.kinds.clone(),
            perimeters: self.perimeters.clone(),
            has_video: videos,
            metrics_keys: row_keys(self.row.as_ref()),
        }
    }
}

fn row_keys(row: Option<&Value>) -> BTreeSet<String> {
    row.and_then(Value::as_object)
        .map(|r| r.keys().cloned().collect())
        .unwrap_or_default()
}

/// Facts of a metrics row whose trial directory is not packaged (animal summaries
/// still place it).
fn row_facts(id: &str, row: &Value) -> TrialFacts {
    let field = |key: &str| text(&row[key]);
    let assay = field("assay").unwrap_or_default();
    TrialFacts {
        id: id.into(),
        task_type: experiment::descriptor(&assay).map_or(assay, |d| d.task_type.clone()),
        phase: field("phase"),
        timepoint: field("timepoint"),
        variant: field("variant"),
        analysis_role: field("analysis_role"),
        metrics_keys: row_keys(Some(row)),
        ..TrialFacts::default()
    }
}

/// Resolves the package presentation from every trial and metrics row, and
/// places each packaged trial.
pub(crate) fn present(
    recordings: &mut [Recording],
    metrics: &BTreeMap<String, Value>,
    review: &bekiper_ingress::config::ReviewConfig,
    videos: bool,
) -> Result<ResolvedPresentation> {
    let mut facts: Vec<TrialFacts> = recordings
        .iter()
        .flat_map(|r| &r.trials)
        .map(|t| t.facts(videos))
        .collect();
    let packaged: BTreeSet<String> = facts.iter().map(|f| f.id.clone()).collect();
    facts.extend(
        metrics
            .iter()
            .filter(|(id, _)| !packaged.contains(*id))
            .map(|(id, row)| row_facts(id, row)),
    );
    let presentation = experiment::resolve(
        &CorpusFacts { trials: facts },
        &review.presentation_overrides(),
    )?;
    for trial in recordings.iter_mut().flat_map(|r| r.trials.iter_mut()) {
        trial.placement = presentation.trials.get(&trial.id).cloned();
    }
    Ok(presentation)
}

/// The resolved experiment of an assay key.
pub(crate) fn experiment<'a>(
    presentation: &'a ResolvedPresentation,
    assay: &str,
) -> Option<&'a ResolvedExperiment> {
    presentation.experiments.iter().find(|e| e.key == assay)
}

/// Criteria endpoints of an assay, in order (`[review.primary_endpoints]`, else
/// the descriptor's).
pub(crate) fn criteria(presentation: &ResolvedPresentation, assay: &str) -> Vec<String> {
    experiment(presentation, assay).map_or_else(Vec::new, |e| e.endpoints.clone())
}

pub(crate) struct Recording {
    pub id: String,
    pub assay: String,
    pub animal: Option<String>,
    pub trials: Vec<Trial>,
    /// Package-relative composite path without extension.
    pub figure: String,
    pub sidecar: Option<Value>,
    pub failed: Option<String>,
    /// Why a preview sampled this recording (empty in a full build).
    pub reasons: Vec<String>,
    /// Scored inspection video, when rendered: `{path, poster, width, height, rows}`.
    pub video: Option<Value>,
}

impl Recording {
    pub fn cost(&self) -> usize {
        self.trials.len() * self.trials[0].descriptor().render_cost
    }

    /// The first variant's recorded row (OFT/NORT have exactly one).
    pub fn row(&self) -> Option<&Value> {
        self.trials.iter().find_map(|t| t.row.as_ref())
    }

    /// The principal-context trial (Y-maze: the 45 % variant), else the first.
    pub fn principal(&self) -> &Trial {
        self.trials
            .iter()
            .find(|t| t.principal())
            .unwrap_or(&self.trials[0])
    }

    /// The row a preview stratifies and ranks on: the principal context's.
    pub fn primary_row(&self) -> Option<&Value> {
        self.trials
            .iter()
            .find(|t| t.principal())
            .and_then(|t| t.row.as_ref())
            .or_else(|| self.row())
    }

    fn placement(&self) -> Option<&Placement> {
        self.principal().placement.as_ref()
    }

    /// Animal-page order: section (e.g. the NORT protocol, then the Y-maze), then
    /// timepoint (pre before post), then step.
    pub fn protocol_key(&self) -> ((u32, u8, u32), String) {
        let key = self
            .placement()
            .map_or((u32::MAX, 2, u32::MAX), Placement::protocol_key);
        (key, self.id.clone())
    }

    /// Animal-page section heading (`Pre-treatment: NORT protocol`).
    pub fn heading(&self) -> String {
        self.placement()
            .map_or_else(|| "Other recordings".into(), |p| p.heading.clone())
    }

    /// Object configuration: `A = familiar (D) · B = variable (U)`.
    pub fn objects(&self) -> Option<String> {
        let row = self.primary_row()?;
        self.trials[0]
            .descriptor()
            .objects
            .as_ref()?
            .describe(|field| text(&row[field]))
    }

    /// Readable session label: `Pre · Habituation (OFT) · batch B`, `Post · T2 test`.
    pub fn session_label(&self) -> String {
        self.placement()
            .map_or_else(|| self.assay.clone(), |p| p.session_label.clone())
    }
}

/// The protocol step of a recorded metrics row; only principal-context rows
/// (Y-maze: the 45 % variant) carry a step.
pub(crate) fn row_step<'a>(presentation: &'a ResolvedPresentation, row: &Value) -> Option<&'a str> {
    let placement = presentation.trials.get(row["analysis_id"].as_str()?)?;
    placement
        .principal
        .then_some(placement.step.as_deref())
        .flatten()
}

/// Perimeter labels and feature kinds of an inspection manifest (empty when absent).
fn manifest_facts(dir: &Path) -> (BTreeSet<String>, Vec<String>) {
    let manifest: toml::Table = inputs(dir)
        .ok()
        .and_then(|i| {
            i.into_iter()
                .find(|p| p.to_string_lossy().ends_with(".toml"))
        })
        .and_then(|p| fs::read_to_string(p).ok())
        .and_then(|s| toml::from_str(&s).ok())
        .unwrap_or_default();
    let strings = |table: &str, key: &str| -> Vec<String> {
        manifest
            .get(table)
            .and_then(toml::Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|item| item.get(key)?.as_str().map(str::to_string))
            .collect()
    };
    (
        strings("feature_inspections", "kind").into_iter().collect(),
        strings("perimeters", "label"),
    )
}

/// Every trial directory with an inspection manifest, sorted by ID.
pub(crate) fn discover(data: &Path, only: &[String]) -> Result<Vec<Trial>> {
    let mut trials = Vec::new();
    let mut entries: Vec<_> = fs::read_dir(data)?.collect::<std::io::Result<_>>()?;
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        let dir = entry.path();
        let id = entry.file_name().to_string_lossy().into_owned();
        if id.starts_with('_') || !dir.join("trial_context.json").is_file() {
            continue;
        }
        if !only.is_empty()
            && !only
                .iter()
                .any(|o| *o == id || id.starts_with(&format!("{o}__")))
        {
            continue;
        }
        if inputs(&dir).map(|i| i.is_empty()).unwrap_or(true) {
            continue;
        }
        let context: Value = serde_json::from_slice(&fs::read(dir.join("trial_context.json"))?)
            .with_context(|| format!("{id}/trial_context.json"))?;
        let experiment = &context["experiment"];
        let task_type = experiment["task_type"].as_str().unwrap_or("").to_string();
        let (kinds, perimeters) = manifest_facts(&dir);
        trials.push(Trial {
            assay: experiment::experiment_key(&task_type),
            task_type,
            recipe: text(&experiment["detection"]["recipe"]),
            video: experiment["video"].as_str().unwrap_or(&id).into(),
            variant: id.split_once("__").map(|(_, v)| v.to_string()),
            session: context["session_context"].clone(),
            kinds,
            perimeters,
            placement: None,
            id,
            dir,
            row: None,
        });
    }
    Ok(trials)
}

/// The manifest and evaluated table that a composite is drawn from.
pub(crate) fn inputs(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut found: Vec<PathBuf> = fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name().is_some_and(|n| {
                let n = n.to_string_lossy();
                n.ends_with("_inspection.toml") || n.ends_with("_evaluated.parquet")
            })
        })
        .collect();
    found.sort();
    Ok(if found.len() == 2 { found } else { Vec::new() })
}

/// Identity of a post-analysis context directory: the recorded
/// `primary_context.id`, else (older runs) `arm<pct>` from the recorded
/// threshold or the legacy `primary<pct>` directory name. Directory names only
/// locate outputs; descriptors name contexts by id.
pub(crate) fn context_id(dir: &Path) -> String {
    let recorded = fs::read(dir.join("run_manifest.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
        .map(|manifest| manifest["resolved_config"]["primary_context"].clone());
    if let Some(record) = recorded {
        if let Some(id) = record["id"].as_str() {
            return id.to_string();
        }
        if let Some(pct) = record["threshold_pct"].as_u64() {
            return format!("arm{pct}");
        }
    }
    let name = dir
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    match name.strip_prefix("primary") {
        Some(pct) if !pct.is_empty() && pct.bytes().all(|b| b.is_ascii_digit()) => {
            format!("arm{pct}")
        }
        _ => name,
    }
}

pub(crate) fn contexts(data: &Path) -> Result<Vec<PathBuf>> {
    let mut found = Vec::new();
    let Ok(runs) = fs::read_dir(data.join("_post_analysis")) else {
        return Ok(found);
    };
    let mut runs: Vec<PathBuf> = runs.filter_map(|e| e.ok().map(|e| e.path())).collect();
    runs.sort();
    // The newest completed cohort run wins.
    if let Some(run) = runs
        .iter()
        .rev()
        .find(|r| r.join("run_manifest.json").is_file())
    {
        let mut children: Vec<PathBuf> = fs::read_dir(run)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.join("_internal/metrics.json").is_file())
            .collect();
        // The principal cohort context (Imen: primary45) first: it carries the
        // experiments without threshold contexts and the full vector originals.
        let principal: Vec<&str> = experiment::descriptors()
            .iter()
            .filter_map(|d| d.contexts.as_ref()?.principal_cohort.as_deref())
            .collect();
        children.sort_by_key(|p| {
            let name = p
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            (!principal.contains(&context_id(p).as_str()), name)
        });
        found = children;
    }
    Ok(found)
}

/// `analysis_id -> recorded metrics row`, first context wins.
pub(crate) fn metrics(contexts: &[PathBuf]) -> Result<BTreeMap<String, Value>> {
    let mut rows = BTreeMap::new();
    for context in contexts {
        let records: Vec<Value> =
            serde_json::from_slice(&fs::read(context.join("_internal/metrics.json"))?)?;
        for record in records {
            if let Some(id) = record["analysis_id"].as_str() {
                rows.entry(id.to_string()).or_insert(record);
            }
        }
    }
    Ok(rows)
}

pub(crate) fn recordings(trials: Vec<Trial>, metrics: &BTreeMap<String, Value>) -> Vec<Recording> {
    let mut grouped: BTreeMap<(String, String), Vec<Trial>> = BTreeMap::new();
    for mut trial in trials {
        trial.row = metrics.get(&trial.id).cloned();
        grouped
            .entry((trial.assay.clone(), trial.video.clone()))
            .or_default()
            .push(trial);
    }
    grouped
        .into_values()
        .map(|mut trials| {
            trials.sort_by_key(|t| t.descriptor().context_order(t.variant.as_deref()));
            let id = trials[0]
                .id
                .split("__")
                .next()
                .unwrap_or(&trials[0].id)
                .to_string();
            let animal = trials
                .iter()
                .find_map(|t| text(&t.row.as_ref()?["animal_id"]))
                .or_else(|| trials.iter().find_map(|t| text(&t.session["animal"])));
            let folder = animal.as_deref().map_or("unassigned".to_string(), slug);
            Recording {
                figure: format!("figures/trials/{folder}/{id}"),
                assay: trials[0].assay.clone(),
                id,
                animal,
                trials,
                sidecar: None,
                failed: None,
                reasons: Vec::new(),
                video: None,
            }
        })
        .collect()
}

pub(crate) fn text(value: &Value) -> Option<String> {
    match value {
        Value::String(s) if !s.is_empty() => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

/// Distinct names must map to distinct slugs (`1-2` and `1_2` would share a file).
pub(crate) fn ensure_unique_slugs<'a>(
    kind: &str,
    names: impl IntoIterator<Item = &'a str>,
) -> Result<()> {
    let mut seen = BTreeMap::new();
    for name in names {
        if let Some(other) = seen.insert(slug(name), name)
            && other != name
        {
            anyhow::bail!(
                "{kind} {other:?} and {name:?} share the output name {:?}",
                slug(name)
            );
        }
    }
    Ok(())
}

pub(crate) fn slug(value: &str) -> String {
    value
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

/// `code -> "G5 · AD · HCAR1-WT · lactate"` from the plan's group mapping.
pub(crate) fn group_names(plan: Option<&Path>) -> Result<BTreeMap<String, String>> {
    let Some(plan) = plan else {
        return Ok(BTreeMap::new());
    };
    let table: toml::Table = toml::from_str(&fs::read_to_string(plan)?)?;
    let inputs = table.get("inputs").and_then(|v| v.as_table());
    let mapping = inputs
        .and_then(|i| i.get("group_mapping"))
        .and_then(|v| v.as_table());
    let codes = inputs
        .and_then(|i| i.get("session_key_mapping"))
        .and_then(|v| v.get("group"))
        .and_then(|v| v.as_table());
    let mut names = BTreeMap::new();
    if let (Some(mapping), Some(codes)) = (mapping, codes) {
        for (code, group) in codes {
            let Some(group) = group.as_str() else {
                continue;
            };
            let factors = mapping.get(group).and_then(|v| v.as_table());
            let part = |key: &str| {
                factors
                    .and_then(|f| f.get(key))
                    .and_then(|v| v.as_str())
                    .unwrap_or("?")
            };
            names.insert(
                code.clone(),
                format!(
                    "{group} · {} · HCAR1-{} · {}",
                    part("five_xfad"),
                    part("hcar1"),
                    part("treatment")
                ),
            );
        }
    }
    Ok(names)
}

pub(crate) fn group_label(row: Option<&Value>, groups: &BTreeMap<String, String>) -> String {
    let code = row.and_then(|r| text(&r["group"]));
    match code {
        Some(code) => groups
            .get(&code)
            .cloned()
            .unwrap_or(format!("Group {code}")),
        None => "Unrecorded group".into(),
    }
}

/// QC flags shown as badges: (field, label, value that is good).
pub(crate) const FLAGS: &[(&str, &str, bool)] = &[
    ("duration_within_expected", "360 ± 10 s", true),
    ("window_complete_flag", "300 s window complete", true),
    (
        "passes_15s_exploration_threshold",
        "≥15 s exploration",
        true,
    ),
    ("low_valid_entries_flag", "low valid entries", false),
    ("low_usable_triads_flag", "low usable triads", false),
    ("excluded", "excluded", false),
];

pub(crate) fn flags(row: Option<&Value>) -> BTreeMap<String, bool> {
    let mut out = BTreeMap::new();
    if let Some(row) = row {
        for (field, label, good) in FLAGS {
            // Pass/fail checks always show; warnings only when raised.
            match row[*field].as_bool() {
                Some(value) if *good => {
                    out.insert(label.to_string(), value);
                }
                Some(true) => {
                    out.insert(label.to_string(), false);
                }
                _ => {}
            }
        }
    }
    out
}

pub(crate) fn header(
    recording: &Recording,
    groups: &BTreeMap<String, String>,
    criteria: &[String],
) -> Value {
    let row = recording.primary_row();
    let field = |key: &str| row.and_then(|r| text(&r[key]));
    let session = recording.session_label();
    let animal = recording
        .animal
        .clone()
        .unwrap_or_else(|| "unassigned".into());
    let mut endpoints = serde_json::Map::new();
    if recording.trials.len() > 1 {
        // Y-maze: one line per variant with its first criteria endpoints; the rest
        // are on the animal page.
        for trial in &recording.trials {
            let Some(trial_row) = trial.row.as_ref() else {
                continue;
            };
            let values: serde_json::Map<String, Value> = criteria
                .iter()
                .take(4)
                .map(|key| (key.clone(), trial_row[key.as_str()].clone()))
                .collect();
            endpoints.insert(trial.variant_label(), Value::Object(values));
        }
    } else if let Some(row) = row {
        for key in criteria.iter().take(10) {
            endpoints.insert(key.clone(), row[key.as_str()].clone());
        }
    }
    json!({
        "title": format!("Animal {animal} · {session} · {}", recording.id),
        "identity": [group_label(row, groups), field("sex"), field("age").map(|a| format!("age {a} d"))],
        "endpoints": endpoints,
        "flags": flags(row),
        "trial_id": recording.id,
    })
}

/// Session stratum without the scoring variant: `nort · post · test`.
fn stratum(recording: &Recording) -> Option<String> {
    let row = recording.primary_row()?;
    let variant = row["variant"].as_str().unwrap_or("");
    let label = row["presentation_label"].as_str()?;
    Some(
        label
            .split(" · ")
            .filter(|part| *part != "recorded" && *part != variant)
            .collect::<Vec<_>>()
            .join(" · "),
    )
}

/// Evenly spaced ranks: 3 picks give min, median and max.
fn quantile_picks(len: usize, picks: usize) -> Vec<(usize, String)> {
    if len == 0 || picks == 0 {
        return Vec::new();
    }
    if picks == 1 {
        return vec![(len / 2, "median".into())];
    }
    (0..picks)
        .map(|k| {
            let rank = ((k * (len - 1)) as f64 / (picks - 1) as f64).round() as usize;
            let name = match k {
                0 => "min".to_string(),
                k if k == picks - 1 => "max".to_string(),
                k if 2 * k == picks - 1 => "median".to_string(),
                k => format!("p{}", 100 * k / (picks - 1)),
            };
            (rank, name)
        })
        .collect()
}

/// Deterministic preview sample of one assay from recorded values only: per
/// session stratum the min/median/max of the first criteria endpoint with
/// values there, plus up to three recordings per edge case. Each pick carries
/// its reasons.
pub(crate) fn preview_sample(
    recordings: Vec<Recording>,
    assay: &str,
    per_stratum: usize,
    criteria: &[String],
    failed_before: &dyn Fn(&Recording) -> bool,
) -> Vec<Recording> {
    let pool: Vec<Recording> = recordings
        .into_iter()
        .filter(|r| r.assay == assay)
        .collect();
    let value = |r: &Recording, key: &str| {
        r.primary_row()
            .and_then(|row| row[key].as_f64())
            .filter(|v| v.is_finite())
    };
    let mut strata: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, recording) in pool.iter().enumerate() {
        if let Some(name) = stratum(recording) {
            strata.entry(name).or_default().push(i);
        }
    }
    let mut reasons: BTreeMap<usize, Vec<String>> = BTreeMap::new();
    for (name, members) in &strata {
        let Some(key) = criteria
            .iter()
            .find(|k| members.iter().any(|&i| value(&pool[i], k).is_some()))
        else {
            continue;
        };
        let mut ranked: Vec<(f64, usize)> = members
            .iter()
            .filter_map(|&i| Some((value(&pool[i], key)?, i)))
            .collect();
        ranked.sort_by(|a, b| {
            a.0.total_cmp(&b.0)
                .then_with(|| pool[a.1].id.cmp(&pool[b.1].id))
        });
        for (rank, pick) in quantile_picks(ranked.len(), per_stratum) {
            let reason = format!("{pick} {key} · {name}");
            let entry = reasons.entry(ranked[rank].1).or_default();
            if !entry.contains(&reason) {
                entry.push(reason);
            }
        }
    }
    let lowest = |key: &str| {
        let mut ranked: Vec<(f64, usize)> = (0..pool.len())
            .filter_map(|i| Some((value(&pool[i], key)?, i)))
            .collect();
        ranked.sort_by(|a, b| a.0.total_cmp(&b.0));
        ranked
            .first()
            .map(|(_, i)| *i)
            .into_iter()
            .collect::<Vec<_>>()
    };
    let flagged = |r: &Recording| {
        r.trials
            .iter()
            .any(|t| flags(t.row.as_ref()).values().any(|good| !good))
    };
    let mut edges: Vec<(&str, Vec<usize>)> = vec![
        (
            "QC flag",
            (0..pool.len()).filter(|&i| flagged(&pool[i])).collect(),
        ),
        (
            "no metrics row",
            (0..pool.len())
                .filter(|&i| pool[i].trials.iter().any(|t| t.row.is_none()))
                .collect(),
        ),
        ("shortest analysed duration", lowest("total_duration_s")),
        (
            "failed to render last preview",
            (0..pool.len())
                .filter(|&i| failed_before(&pool[i]))
                .collect(),
        ),
    ];
    for edge in experiment::descriptor(assay)
        .map(|d| d.edge_cases.as_slice())
        .unwrap_or_default()
    {
        edges.push((edge.label.as_str(), lowest(&edge.lowest)));
    }
    for (name, picks) in edges {
        for i in picks.into_iter().take(3) {
            reasons.entry(i).or_default().push(name.into());
        }
    }
    pool.into_iter()
        .enumerate()
        .filter_map(|(i, mut recording)| {
            recording.reasons = reasons.remove(&i)?;
            Some(recording)
        })
        .collect()
}
