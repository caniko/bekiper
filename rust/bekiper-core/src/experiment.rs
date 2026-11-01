//! Experiment descriptors: the single source of how each task type is presented
//! (protocol steps, threshold contexts, objects, endpoints, presentation blocks).
//!
//! Built-in descriptors live in `experiments/*.toml` and are embedded at compile
//! time. Unknown task types resolve to the `generic` descriptor.
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

mod contract;
mod presentation;
#[cfg(test)]
mod tests;

pub use contract::*;
pub use presentation::*;

const SOURCES: [(&str, &str); 5] = [
    ("open_field", include_str!("../experiments/open_field.toml")),
    (
        "object_recognition",
        include_str!("../experiments/object_recognition.toml"),
    ),
    ("y_maze", include_str!("../experiments/y_maze.toml")),
    (
        "cheeseboard",
        include_str!("../experiments/cheeseboard.toml"),
    ),
    ("generic", include_str!("../experiments/generic.toml")),
];

/// Presentation archetype: which family of blocks and visuals an experiment uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Archetype {
    ObjectExploration,
    ZoneOccupancy,
    ChoiceSequence,
    GoalSearch,
    Generic,
}

impl Archetype {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ObjectExploration => "object_exploration",
            Self::ZoneOccupancy => "zone_occupancy",
            Self::ChoiceSequence => "choice_sequence",
            Self::GoalSearch => "goal_search",
            Self::Generic => "generic",
        }
    }
}

/// What the inspection video of a step shows.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VideoKind {
    /// Scored masks over the recording (`bekiper inspect video --scored`).
    #[default]
    Scored,
    /// Object-contact qualia (`--qualia`).
    Qualia,
}

/// Data a block needs before it is shown.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    /// Object or goal outlines (descriptor `objects`) on a perimeter.
    Objects,
    /// Zone features: centre/periphery, zones or quadrants.
    Zones,
    /// Arm features: arm sections or an arm sequence.
    Arms,
    /// A source video.
    Video,
    /// More than one threshold context.
    Contexts,
    /// A recorded metrics row.
    Metrics,
    /// A recorded pre/post timepoint.
    Timepoints,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExperimentDescriptor {
    pub id: String,
    /// Assay key used by metrics rows, post-analysis, `[review.primary_endpoints]`
    /// and package paths (`oft`, `nort`, `ymaze`).
    pub key: String,
    /// Canonical [`crate::types::TaskType`] string; empty for `generic`.
    pub task_type: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    pub display_name: String,
    pub short_name: String,
    pub archetype: Archetype,
    /// Order across experiments (assay pages, navigation).
    pub order: u32,
    /// Composite render cost per trial, relative to a single-channel trial.
    #[serde(default = "one")]
    pub render_cost: usize,
    /// Experiment-level exploratory figure ids.
    #[serde(default)]
    pub figures: Vec<String>,
    #[serde(default)]
    pub sections: Vec<Section>,
    #[serde(default)]
    pub steps: Vec<Step>,
    #[serde(default)]
    pub contexts: Option<Contexts>,
    #[serde(default)]
    pub objects: Option<Objects>,
    #[serde(default)]
    pub features: Features,
    #[serde(default)]
    pub edge_cases: Vec<EdgeCase>,
    #[serde(default)]
    pub post_analysis: Option<PostAnalysis>,
    #[serde(default)]
    pub endpoints: Vec<Endpoint>,
    #[serde(default)]
    pub blocks: Vec<BlockRule>,
    #[serde(default)]
    pub prose: Prose,
}

fn one() -> usize {
    1
}

/// A group of steps shown together on the animal page.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Section {
    pub id: String,
    pub title: String,
    pub order: u32,
    /// One section per timepoint (`Pre-treatment: <title>`).
    #[serde(default)]
    pub by_timepoint: bool,
}

impl Section {
    /// Animal-page heading of this section at a timepoint.
    pub fn heading(&self, timepoint: Option<&str>) -> String {
        match (self.by_timepoint, timepoint) {
            (false, _) => self.title.clone(),
            (true, Some("pre")) => format!("Pre-treatment: {}", self.title),
            (true, Some("post")) => format!("Post-treatment: {}", self.title),
            (true, _) => format!("{} (timepoint unrecorded)", self.title),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Step {
    pub id: String,
    pub label: String,
    /// Section id, possibly owned by another descriptor.
    pub section: String,
    /// Order within a section and timepoint.
    pub order: u32,
    #[serde(default)]
    pub video: VideoKind,
    /// The step applies only when the corpus also has this experiment id.
    #[serde(default)]
    pub requires_experiment: Option<String>,
    /// Session-label suffix; `{field}` is a [`Matcher`] field, dropped when unrecorded.
    #[serde(default)]
    pub suffix: Option<String>,
    /// Alternatives; empty matches every trial. Steps are tried in file order.
    #[serde(default, rename = "match")]
    pub matchers: Vec<Matcher>,
}

/// All set fields must equal the trial's value; an empty value means unrecorded.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Matcher {
    pub recipe: Option<String>,
    pub phase: Option<String>,
    pub stage: Option<String>,
    pub timepoint: Option<String>,
    pub phase_part: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Contexts {
    /// Principal variants; the first non-comparison context when none is present.
    pub principal: Vec<String>,
    /// Post-analysis cohort context that carries the experiments without contexts.
    pub principal_cohort: Option<String>,
    #[serde(default)]
    pub rules: Vec<ContextRule>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContextRule {
    pub pattern: String,
    pub label: Option<String>,
    pub rank: Option<u8>,
    pub comparison: Option<bool>,
}

impl ContextRule {
    /// The `{n}` digits when the pattern matches (`Some("")` without `{n}`).
    fn capture<'a>(&self, variant: &'a str) -> Option<&'a str> {
        let (pattern, any_suffix) = match self.pattern.strip_suffix('*') {
            Some(p) => (p, true),
            None => (self.pattern.as_str(), false),
        };
        let Some((head, tail)) = pattern.split_once("{n}") else {
            return if any_suffix {
                variant.starts_with(pattern).then_some("")
            } else {
                (variant == pattern).then_some("")
            };
        };
        let rest = variant.strip_prefix(head)?;
        let digits = rest.len() - rest.trim_start_matches(|c: char| c.is_ascii_digit()).len();
        let after = &rest[digits..];
        let tail_ok = if any_suffix {
            after.starts_with(tail)
        } else {
            after == tail
        };
        (digits > 0 && tail_ok).then_some(&rest[..digits])
    }
}

impl Contexts {
    /// Display label of a variant (`arm45` → `45%`; none → `recorded`).
    pub fn label(&self, variant: Option<&str>) -> String {
        let Some(variant) = variant else {
            return "recorded".into();
        };
        self.rules
            .iter()
            .find_map(|rule| {
                let n = rule.capture(variant)?;
                Some(rule.label.as_ref()?.replace("{n}", n))
            })
            .unwrap_or_else(|| variant.to_string())
    }

    /// Sort key: rank, then the name with `{n}` zero-padded (`arm25` < `arm100`).
    pub fn order_key(&self, variant: Option<&str>) -> (u8, String) {
        let Some(variant) = variant else {
            return (0, String::new());
        };
        let rank = self
            .rules
            .iter()
            .find_map(|rule| rule.capture(variant).and(rule.rank))
            .unwrap_or(1);
        let padded = self
            .rules
            .iter()
            .find_map(|rule| {
                let n = rule.capture(variant).filter(|n| !n.is_empty())?;
                let head = rule.pattern.split_once("{n}")?.0;
                Some(format!(
                    "{head}{:05}{}",
                    n.parse::<u64>().ok()?,
                    &variant[head.len() + n.len()..]
                ))
            })
            .unwrap_or_else(|| variant.to_string());
        (rank, padded)
    }

    /// Whether the experiment has named variants to compare (an empty rule list
    /// is only a declaration that its principal is the bare analysis).
    pub fn varies(&self) -> bool {
        !self.rules.is_empty()
    }

    pub fn is_comparison(&self, variant: &str) -> bool {
        self.rules
            .iter()
            .find_map(|rule| rule.capture(variant).and(rule.comparison))
            .unwrap_or(false)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Objects {
    /// Metrics object slots (`A`, `B`).
    #[serde(default)]
    pub slots: Vec<String>,
    /// Metrics field of a slot's role, `{slot}` replaced.
    pub label_field: Option<String>,
    /// Metrics field of a slot's placement, `{slot}` replaced.
    pub placement_field: Option<String>,
    /// Perimeter labels that are objects or goals.
    #[serde(default)]
    pub roles: Vec<String>,
    /// Perimeter labels with this prefix are objects too (`object_1`).
    pub perimeter_prefix: Option<String>,
}

impl Objects {
    pub fn is_object(&self, perimeter: &str) -> bool {
        self.roles
            .iter()
            .any(|r| r.eq_ignore_ascii_case(perimeter.trim()))
            || self
                .perimeter_prefix
                .as_deref()
                .is_some_and(|p| perimeter.starts_with(p))
    }

    /// Object configuration from a metrics row: `A = familiar (D) · B = variable (U)`.
    pub fn describe(&self, field: impl Fn(&str) -> Option<String>) -> Option<String> {
        let label_field = self.label_field.as_deref()?;
        let parts: Vec<String> = self
            .slots
            .iter()
            .filter_map(|slot| {
                let label = field(&label_field.replace("{slot}", slot))?;
                let place = self
                    .placement_field
                    .as_deref()
                    .and_then(|f| field(&f.replace("{slot}", slot)));
                Some(match place {
                    Some(place) => format!("{slot} = {label} ({place})"),
                    None => format!("{slot} = {label}"),
                })
            })
            .collect();
        (!parts.is_empty()).then(|| parts.join(" · "))
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Features {
    /// Feature kinds drawn as composite overview panels.
    #[serde(default)]
    pub panels: Vec<String>,
    /// Feature labels never drawn as overview panels.
    #[serde(default)]
    pub panel_exclude_labels: Vec<String>,
    /// Parameters-page preview figures.
    #[serde(default)]
    pub previews: Vec<Preview>,
    /// Heuristic name prefixes drawn as diagnostic channels (not scoring inputs).
    #[serde(default)]
    pub diagnostic_prefixes: Vec<String>,
    /// Feature label prefixes folded into another composite row, with the reason.
    #[serde(default)]
    pub folded: Vec<Folded>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Folded {
    pub prefix: String,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Preview {
    pub kind: String,
    pub title: String,
}

/// Extra preview pick: the recordings with the lowest recorded value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EdgeCase {
    pub label: String,
    pub lowest: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PostAnalysis {
    /// Post-analysis assay id; also the `figures/<assay>/` export folder.
    pub assay: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Endpoint {
    pub name: String,
    /// Short display label (`DI`, `SA%`).
    pub label: Option<String>,
    /// Post-analysis family: `primary`, `secondary` or `score`.
    pub family: Option<String>,
    /// Default model family; the endpoint is a registered model `<assay>.<name>`.
    pub model: Option<String>,
    /// In the default review criteria list.
    #[serde(default)]
    pub criteria: bool,
    /// Exploratory figure ids that plot this endpoint.
    #[serde(default)]
    pub figures: Vec<String>,
    /// Dotted path in `result.toml` `detailed`.
    pub detail: Option<String>,
    /// Restricts per-trial values to one step id.
    pub step: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlockRule {
    /// Contract block id.
    pub id: String,
    #[serde(default)]
    pub requires: Vec<Capability>,
}

/// Reading-guide fragments.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Prose {
    /// Reading-guide sentence on the assay page's contexts.
    pub assay_page: Option<String>,
    /// Reading-guide sentence on multi-variant composites.
    pub composite: Option<String>,
    /// What the contexts vary (`arm-entry threshold`).
    pub threshold: Option<String>,
    /// Assay-page note of an experiment with contexts.
    pub contexts_note: Option<String>,
}

impl ExperimentDescriptor {
    pub fn parse(source: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(source)
    }

    pub fn matches_task_type(&self, task_type: &str) -> bool {
        !self.task_type.is_empty()
            && (self.task_type == task_type || self.aliases.iter().any(|a| a == task_type))
    }

    /// Endpoint names of a post-analysis family, in order.
    pub fn family(&self, family: &str) -> Vec<&str> {
        self.endpoints
            .iter()
            .filter(|e| e.family.as_deref() == Some(family))
            .map(|e| e.name.as_str())
            .collect()
    }

    /// Endpoints fitted as registered models, in order.
    pub fn modelled(&self) -> Vec<&str> {
        self.endpoints
            .iter()
            .filter(|e| e.model.is_some())
            .map(|e| e.name.as_str())
            .collect()
    }

    /// Display label of a scoring variant (`recorded` without one).
    pub fn context_label(&self, variant: Option<&str>) -> String {
        self.contexts.clone().unwrap_or_default().label(variant)
    }

    /// Display order of a scoring variant.
    pub fn context_order(&self, variant: Option<&str>) -> (u8, String) {
        self.contexts.clone().unwrap_or_default().order_key(variant)
    }

    /// Default review criteria endpoints, in order.
    pub fn criteria(&self) -> Vec<String> {
        self.endpoints
            .iter()
            .filter(|e| e.criteria)
            .map(|e| e.name.clone())
            .collect()
    }
}

/// Every built-in descriptor in experiment order; `generic` is last.
pub fn descriptors() -> &'static [ExperimentDescriptor] {
    static ALL: OnceLock<Vec<ExperimentDescriptor>> = OnceLock::new();
    ALL.get_or_init(|| {
        let mut all: Vec<ExperimentDescriptor> = SOURCES
            .iter()
            .map(|(name, source)| {
                ExperimentDescriptor::parse(source)
                    .unwrap_or_else(|e| panic!("built-in descriptor {name}: {e}"))
            })
            .collect();
        all.sort_by_key(|d| d.order);
        all
    })
}

/// The descriptor of a task type (or alias), else `generic`.
pub fn descriptor_for(task_type: &str) -> &'static ExperimentDescriptor {
    let all = descriptors();
    all.iter()
        .find(|d| d.matches_task_type(task_type))
        .unwrap_or_else(|| generic())
}

/// A descriptor by id or assay key.
pub fn descriptor(name: &str) -> Option<&'static ExperimentDescriptor> {
    descriptors().iter().find(|d| d.id == name || d.key == name)
}

pub fn generic() -> &'static ExperimentDescriptor {
    descriptor("generic").expect("built-in generic descriptor")
}

/// Experiment key of a task type: the descriptor's assay key, else the recorded
/// task type itself (never a silent `other`).
pub fn experiment_key(task_type: &str) -> String {
    let d = descriptor_for(task_type);
    if d.task_type.is_empty() && !task_type.is_empty() {
        task_type.to_string()
    } else {
        d.key.clone()
    }
}

/// Descriptors that the post-analysis scores, in experiment order.
pub fn post_analysis_descriptors() -> impl Iterator<Item = &'static ExperimentDescriptor> {
    descriptors().iter().filter(|d| d.post_analysis.is_some())
}
