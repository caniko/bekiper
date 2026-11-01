//! Corpus inference: which experiments, protocol steps, threshold contexts and
//! blocks a reviewer package shows, from per-trial facts plus overrides. Pure;
//! callers read the facts from disk.
use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use super::{
    Archetype, Capability, Contexts, ExperimentDescriptor, Matcher, Section, VideoKind, descriptor,
    descriptor_for, descriptors, experiment_key,
};
use crate::error::{BekiperError, Result};

/// What one trial directory records. Empty strings count as unrecorded.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TrialFacts {
    /// Analysis ID (directory name under `data/`).
    pub id: String,
    /// `trial_context.json` `experiment.task_type`.
    pub task_type: String,
    pub recipe: Option<String>,
    /// Metrics-row `phase` (NORT `training`/`test`; OFT batch letter).
    pub phase: Option<String>,
    /// Session-context `stage`.
    pub stage: Option<String>,
    /// Metrics-row `timepoint`.
    pub timepoint: Option<String>,
    /// Session-context `phase_part` (`1` = pre, `2` = post).
    pub phase_part: Option<String>,
    /// Scoring variant: the `__<variant>` analysis-ID suffix.
    pub variant: Option<String>,
    pub analysis_role: Option<String>,
    /// Feature `kind`s of the inspection manifest.
    pub feature_kinds: BTreeSet<String>,
    /// Perimeter labels of the inspection manifest.
    pub perimeters: Vec<String>,
    pub has_video: bool,
    /// Keys of the recorded metrics row (empty without one).
    pub metrics_keys: BTreeSet<String>,
}

fn recorded(value: &Option<String>) -> Option<&str> {
    value.as_deref().filter(|v| !v.is_empty())
}

impl TrialFacts {
    /// The variant, else the `analysis_role` suffix (`ymaze__arm45` → `arm45`).
    pub fn variant(&self) -> Option<&str> {
        recorded(&self.variant).or_else(|| {
            recorded(&self.analysis_role)
                .and_then(|r| r.split_once("__"))
                .map(|(_, v)| v)
        })
    }

    /// `pre`/`post` from the metrics row, else the session `phase_part`.
    pub fn timepoint(&self) -> Option<&str> {
        recorded(&self.timepoint).or(match recorded(&self.phase_part) {
            Some("1") => Some("pre"),
            Some("2") => Some("post"),
            _ => None,
        })
    }

    fn field(&self, name: &str) -> Option<&str> {
        match name {
            "recipe" => recorded(&self.recipe),
            "phase" => recorded(&self.phase),
            "stage" => recorded(&self.stage),
            "timepoint" => self.timepoint(),
            "phase_part" => recorded(&self.phase_part),
            _ => None,
        }
    }
}

impl Matcher {
    pub fn matches(&self, trial: &TrialFacts) -> bool {
        [
            ("recipe", &self.recipe),
            ("phase", &self.phase),
            ("stage", &self.stage),
            ("timepoint", &self.timepoint),
            ("phase_part", &self.phase_part),
        ]
        .into_iter()
        .all(|(name, want)| match want.as_deref() {
            None => true,
            Some("") => trial.field(name).is_none(),
            Some(want) => trial.field(name) == Some(want),
        })
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct CorpusFacts {
    pub trials: Vec<TrialFacts>,
}

/// `[review.presentation.<experiment>]`, keyed by descriptor key or id (or the
/// recorded task type of an experiment without a descriptor).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PresentationOverrides {
    pub experiments: BTreeMap<String, ExperimentOverride>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExperimentOverride {
    pub title: Option<String>,
    pub short_name: Option<String>,
    /// Step id → label.
    #[serde(default)]
    pub steps: BTreeMap<String, String>,
    /// The one principal context variant.
    pub principal: Option<String>,
    /// Criteria endpoints in order (replaces the descriptor default).
    pub endpoints: Option<Vec<String>>,
    /// Block ids not shown.
    #[serde(default)]
    pub hide: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ResolvedPresentation {
    /// Experiments in protocol order.
    pub experiments: Vec<ResolvedExperiment>,
    /// Animal-page sections used, in order.
    pub sections: Vec<Section>,
    /// Analysis ID → where the trial sits.
    pub trials: BTreeMap<String, Placement>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ResolvedExperiment {
    pub id: String,
    /// Assay key; the recorded task type for an experiment without a descriptor.
    pub key: String,
    pub title: String,
    pub short_name: String,
    pub archetype: Archetype,
    pub render_cost: usize,
    /// Steps present, in protocol order.
    pub steps: Vec<ResolvedStep>,
    /// Contexts present, in display order.
    pub contexts: Vec<ResolvedContext>,
    pub endpoints: Vec<String>,
    /// Contract block ids shown.
    pub blocks: Vec<String>,
    pub capabilities: BTreeSet<Capability>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ResolvedStep {
    pub id: String,
    pub label: String,
    pub section: String,
    pub order: u32,
    pub video: VideoKind,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ResolvedContext {
    pub variant: Option<String>,
    pub label: String,
    pub principal: bool,
    pub comparison: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Placement {
    /// Experiment key.
    pub experiment: String,
    /// Step id; `None` when no step matched (shown under "Other recordings").
    pub step: Option<String>,
    pub section: String,
    /// Animal-page section heading (`Pre-treatment: NORT protocol`).
    pub heading: String,
    /// `Pre · Habituation (OFT) · batch B`.
    pub session_label: String,
    pub timepoint: Option<String>,
    pub context: String,
    /// The trial is the principal context of its recording.
    pub principal: bool,
    pub video: VideoKind,
    section_order: u32,
    step_order: u32,
}

impl Placement {
    /// Animal-page order: section, timepoint (pre, post, unrecorded), step.
    pub fn protocol_key(&self) -> (u32, u8, u32) {
        let timepoint = match self.timepoint.as_deref() {
            Some("pre") => 0,
            Some("post") => 1,
            _ => 2,
        };
        (self.section_order, timepoint, self.step_order)
    }
}

fn config(message: String) -> BekiperError {
    BekiperError::Config(format!("review.presentation: {message}"))
}

fn section(id: &str) -> &'static Section {
    descriptors()
        .iter()
        .flat_map(|d| &d.sections)
        .find(|s| s.id == id)
        .unwrap_or_else(|| panic!("built-in section {id}"))
}

fn capitalised(text: &str) -> String {
    let mut chars = text.chars();
    chars
        .next()
        .map_or(String::new(), |f| f.to_uppercase().chain(chars).collect())
}

/// `{field}` placeholders from the trial; `None` when one is unrecorded.
fn fill(template: &str, trial: &TrialFacts) -> Option<String> {
    let mut out = template.to_string();
    for name in ["recipe", "phase", "stage", "timepoint", "phase_part"] {
        let slot = format!("{{{name}}}");
        if out.contains(&slot) {
            out = out.replace(&slot, trial.field(name)?);
        }
    }
    Some(out)
}

fn capabilities(d: &ExperimentDescriptor, trials: &[&TrialFacts]) -> BTreeSet<Capability> {
    let kinds = |set: &[&str]| {
        trials
            .iter()
            .any(|t| set.iter().any(|k| t.feature_kinds.contains(*k)))
    };
    let variants: BTreeSet<_> = trials.iter().filter_map(|t| t.variant()).collect();
    [
        (
            Capability::Objects,
            d.objects.as_ref().is_some_and(|o| {
                trials
                    .iter()
                    .any(|t| t.perimeters.iter().any(|p| o.is_object(p)))
            }),
        ),
        (
            Capability::Zones,
            kinds(&["center_periphery", "zone", "quadrant"]),
        ),
        (
            Capability::Arms,
            kinds(&["ymaze_arm_sections", "arm_sequence"]),
        ),
        (Capability::Video, trials.iter().any(|t| t.has_video)),
        (Capability::Contexts, variants.len() > 1),
        (
            Capability::Metrics,
            trials.iter().any(|t| !t.metrics_keys.is_empty()),
        ),
        (
            Capability::Timepoints,
            trials.iter().any(|t| t.timepoint().is_some()),
        ),
    ]
    .into_iter()
    .filter_map(|(c, present)| present.then_some(c))
    .collect()
}

/// Resolves the presentation of a corpus. Override keys, step ids and hidden
/// blocks must name something the descriptor defines.
pub fn resolve(
    corpus: &CorpusFacts,
    overrides: &PresentationOverrides,
) -> Result<ResolvedPresentation> {
    // Experiment key → (descriptor, trials).
    let mut groups: BTreeMap<String, (&'static ExperimentDescriptor, Vec<&TrialFacts>)> =
        BTreeMap::new();
    for trial in &corpus.trials {
        let d = descriptor_for(&trial.task_type);
        let key = experiment_key(&trial.task_type);
        groups.entry(key).or_insert((d, Vec::new())).1.push(trial);
    }
    let mut claimed: BTreeMap<&str, &str> = BTreeMap::new();
    for (name, o) in &overrides.experiments {
        let d = descriptor(name)
            .or_else(|| groups.get(name).map(|(d, _)| *d))
            .ok_or_else(|| config(format!("unknown experiment {name}")))?;
        if descriptor(name).is_some() {
            if let Some(other) = claimed.insert(d.id.as_str(), name) {
                return Err(config(format!(
                    "{other} and {name} override the same experiment"
                )));
            }
        }
        if let Some(step) = o
            .steps
            .keys()
            .find(|s| !d.steps.iter().any(|x| &x.id == *s))
        {
            return Err(config(format!("{name}: unknown step {step}")));
        }
        if let Some(block) = o
            .hide
            .iter()
            .find(|b| !d.blocks.iter().any(|x| &x.id == *b))
        {
            return Err(config(format!("{name}: unknown block {block}")));
        }
    }
    let present: BTreeSet<&str> = groups.values().map(|(d, _)| d.id.as_str()).collect();
    let mut ordered: Vec<_> = groups.into_iter().collect();
    ordered.sort_by(|(a, (da, _)), (b, (db, _))| (da.order, a).cmp(&(db.order, b)));

    let no_contexts = Contexts::default();
    let mut experiments = Vec::new();
    let mut trials = BTreeMap::new();
    let mut sections = BTreeMap::new();
    for (key, (d, members)) in ordered {
        let o = overrides
            .experiments
            .get(&key)
            .or_else(|| overrides.experiments.get(&d.id))
            .cloned()
            .unwrap_or_default();
        let generic = d.task_type.is_empty();
        // The generic fallback is titled by its descriptor, a named unknown task type by its name.
        let name = if generic && key != d.key {
            &key
        } else {
            &d.display_name
        };
        let short = if generic && key != d.key {
            &key
        } else {
            &d.short_name
        };
        let rules = d.contexts.as_ref().unwrap_or(&no_contexts);

        // Contexts: distinct variants in display order, principal chosen.
        let mut variants: Vec<Option<&str>> = members.iter().map(|t| t.variant()).collect();
        variants.sort_by_key(|v| (rules.order_key(*v), *v));
        variants.dedup();
        if let Some(p) = &o.principal {
            if d.contexts.is_none() || !variants.contains(&Some(p.as_str())) {
                return Err(config(format!(
                    "{key}: principal {p} is not a recorded context"
                )));
            }
        }
        let principal: BTreeSet<&str> = match (&o.principal, &d.contexts) {
            (Some(p), _) => BTreeSet::from([p.as_str()]),
            (None, Some(c)) => {
                let named: BTreeSet<&str> = variants
                    .iter()
                    .flatten()
                    .copied()
                    .filter(|v| c.principal.iter().any(|p| p == v))
                    .collect();
                if named.is_empty() {
                    // ponytail: first non-comparison context; name one in overrides when that is wrong.
                    variants
                        .iter()
                        .flatten()
                        .copied()
                        .find(|v| !c.is_comparison(v))
                        .into_iter()
                        .collect()
                } else {
                    named
                }
            }
            (None, None) => BTreeSet::new(),
        };
        let is_principal = |v: Option<&str>| match v {
            _ if d.contexts.is_none() && o.principal.is_none() => true,
            None => variants.iter().all(Option::is_none),
            Some(v) => principal.contains(v),
        };
        let contexts: Vec<ResolvedContext> = variants
            .iter()
            .map(|v| ResolvedContext {
                variant: v.map(str::to_string),
                label: rules.label(*v),
                principal: is_principal(*v),
                comparison: v.is_some_and(|v| rules.is_comparison(v)),
            })
            .collect();

        let mut steps: BTreeMap<String, ResolvedStep> = BTreeMap::new();
        for trial in &members {
            let step = d.steps.iter().find(|s| {
                s.requires_experiment
                    .as_deref()
                    .is_none_or(|e| present.contains(e))
                    && (s.matchers.is_empty() || s.matchers.iter().any(|m| m.matches(trial)))
            });
            let mut section = section(step.map_or("other", |s| s.section.as_str())).clone();
            if generic {
                // An experiment without a descriptor gets its own section, titled by its task type.
                section.id.clone_from(&key);
                section.title = o.title.clone().unwrap_or_else(|| name.clone());
            }
            sections.insert((section.order, section.id.clone()), section.clone());
            let label = step.map(|s| o.steps.get(&s.id).unwrap_or(&s.label).clone());
            let timepoint = trial.timepoint();
            let mut parts: Vec<String> = timepoint.map(capitalised).into_iter().collect();
            parts.push(label.clone().unwrap_or_else(|| "Other".into()));
            match step {
                Some(s) => parts.extend(s.suffix.as_deref().and_then(|t| fill(t, trial))),
                None => parts.push(if generic && key == d.key {
                    name.clone()
                } else {
                    key.clone()
                }),
            }
            if let (Some(s), Some(label)) = (step, &label) {
                steps.entry(s.id.clone()).or_insert(ResolvedStep {
                    id: s.id.clone(),
                    label: label.clone(),
                    section: section.id.clone(),
                    order: s.order,
                    video: s.video,
                });
            }
            let variant = trial.variant();
            trials.insert(
                trial.id.clone(),
                Placement {
                    experiment: key.clone(),
                    step: step.map(|s| s.id.clone()),
                    section: section.id.clone(),
                    heading: section.heading(timepoint),
                    session_label: parts.join(" · "),
                    timepoint: timepoint.map(str::to_string),
                    context: rules.label(variant),
                    principal: is_principal(variant),
                    video: step.map_or_else(VideoKind::default, |s| s.video),
                    section_order: section.order,
                    step_order: step.map_or(u32::MAX, |s| s.order),
                },
            );
        }
        let mut steps: Vec<ResolvedStep> = steps.into_values().collect();
        steps.sort_by_key(|s| {
            let order = sections.keys().find(|(_, id)| *id == s.section);
            (order.map_or(u32::MAX, |(o, _)| *o), s.order)
        });

        let capabilities = capabilities(d, &members);
        let blocks = d
            .blocks
            .iter()
            .filter(|b| b.requires.iter().all(|c| capabilities.contains(c)))
            .filter(|b| !o.hide.contains(&b.id))
            .map(|b| b.id.clone())
            .collect();
        experiments.push(ResolvedExperiment {
            id: d.id.clone(),
            title: o.title.clone().unwrap_or_else(|| name.clone()),
            short_name: o.short_name.clone().unwrap_or_else(|| short.clone()),
            key,
            archetype: d.archetype,
            render_cost: d.render_cost,
            steps,
            contexts,
            endpoints: o.endpoints.clone().unwrap_or_else(|| d.criteria()),
            blocks,
            capabilities,
        });
    }
    Ok(ResolvedPresentation {
        experiments,
        sections: sections.into_values().collect(),
        trials,
    })
}
