//! Versioned boundary shared by adapters, computation, plotting and reporting.
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// JSON records retain upstream fields and audit lineage without lossy schema
/// projection. Adapters validate their required canonical key before scoring.
pub type Record = Value;

pub const SCHEMA_VERSION: &str = "2.0";
pub const PLOTTING_SCHEMA_VERSION: &str = "1.0";
/// Conservative concurrency for independent cohort contexts.
pub const DEFAULT_PRIMARY_ANALYSIS_WORKERS: usize = 1;
pub const KEY: [&str; 5] = ["animal_id", "assay", "session_id", "timepoint", "phase"];
pub const FACTORS: [&str; 3] = ["five_xfad", "hcar1", "treatment"];
/// Endpoint lists come from the `bekiper-core` experiment descriptors
/// (`experiments/*.toml`); `descriptor_parity` below pins the Imen values.
fn family_of(assay: &str, family: &str) -> Vec<&'static str> {
    bekiper_core::experiment::descriptor(assay)
        .unwrap_or_else(|| panic!("missing experiment descriptor {assay}"))
        .family(family)
}

macro_rules! family_list {
    ($name:ident, $assay:literal, $family:literal) => {
        pub fn $name() -> &'static [&'static str] {
            static LIST: std::sync::OnceLock<Vec<&'static str>> = std::sync::OnceLock::new();
            LIST.get_or_init(|| family_of($assay, $family))
        }
    };
}
family_list!(oft_primary_fields, "oft", "primary");
family_list!(oft_secondary_fields, "oft", "secondary");
family_list!(ymaze_score_fields, "ymaze", "score");
family_list!(nort_primary_fields, "nort", "primary");
family_list!(nort_secondary_fields, "nort", "secondary");

// Legacy aliases kept only because the pipeline.rs import still names them
// (that file is owned by another session); delete once it stops importing them.
#[doc(hidden)]
#[deprecated(note = "empty placeholder; use the descriptor-derived *_fields() accessors")]
pub const OFT_PRIMARY: [&str; 0] = [];
#[doc(hidden)]
#[deprecated(note = "empty placeholder; use the descriptor-derived *_fields() accessors")]
pub const OFT_SECONDARY: [&str; 0] = [];
#[doc(hidden)]
#[deprecated(note = "empty placeholder; use the descriptor-derived *_fields() accessors")]
pub const YM_SCORES: [&str; 0] = [];

/// Order in which assays are walked wherever it reaches an output (rows,
/// figures, adapter ledger); descriptors iterate oft, nort, ymaze instead.
const REPORT_ORDER: [&str; 3] = ["oft", "ymaze", "nort"];

/// Post-analysis assay ids from the experiment descriptors, in report order
/// (unlisted future assays follow).
pub fn assays_in_report_order() -> &'static [&'static str] {
    static ASSAYS: std::sync::OnceLock<Vec<&'static str>> = std::sync::OnceLock::new();
    ASSAYS.get_or_init(|| {
        let mut assays: Vec<&'static str> = bekiper_core::experiment::post_analysis_descriptors()
            .filter_map(|d| d.post_analysis.as_ref())
            .map(|p| p.assay.as_str())
            .collect();
        assays.sort_by_key(|a| {
            REPORT_ORDER
                .iter()
                .position(|r| r == a)
                .unwrap_or(usize::MAX)
        });
        assays
    })
}

#[derive(Debug, Clone, Default)]
pub struct RunOptions {
    pub ymaze_dwell: bekiper_feature::tolerance::DwellOverrides,
    pub output: Option<PathBuf>,
    pub requested_outputs: Option<Vec<String>>,
    pub sensitivity_thresholds: Option<Vec<u8>>,
    pub primary_thresholds: Option<Vec<u8>>,
    /// One variant of one assay per context (`execution.contexts`); the other
    /// assays take their principal variant. Legacy `primary_thresholds` expand
    /// to Y-maze contexts.
    pub contexts: Option<Vec<ContextSpec>>,
    pub primary_workers: Option<usize>,
    /// Total figure-render processes of this publication (`--workers` of the renderer).
    pub render_workers: Option<usize>,
}

/// `execution.contexts` entry: the assay that varies and its variant.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextSpec {
    pub assay: String,
    pub variant: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CompletionStatus {
    Complete,
    Partial,
    Blocked,
    Unavailable,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Artifact {
    /// Portable path relative to the published run directory.
    pub path: String,
    pub sha256: String,
    pub bytes: u64,
}

pub fn at<'a>(record: &'a Record, path: &str) -> &'a Value {
    path.split('.').fold(record, |value, key| &value[key])
}

pub fn missing(row: &mut Record, field: &str, reason: &str) {
    row[field] = Value::Null;
    if !row["missing_reasons"].is_object() {
        row["missing_reasons"] = serde_json::json!({});
    }
    row["missing_reasons"][field] = reason.into();
}

#[cfg(test)]
mod descriptor_parity {
    use super::*;
    use bekiper_core::experiment::{descriptor, post_analysis_descriptors};

    const OFT_PRIMARY: [&str; 6] = [
        "total_distance_m",
        "mean_speed_m_per_s",
        "center_time_s",
        "center_time_pct",
        "center_entries",
        "latency_to_first_center_entry_s",
    ];
    const OFT_SECONDARY: [&str; 6] = [
        "periphery_time_pct",
        "freezing_s",
        "grooming_s",
        "rearing_s",
        "immobility_s",
        "fecal_boli_count",
    ];
    const YM_SCORES: [&str; 3] = [
        "spontaneous_alternation_pct",
        "spontaneous_alternation_pct_standard",
        "neutral_triads_pct",
    ];

    #[test]
    fn oft_lists() {
        assert_eq!(oft_primary_fields(), OFT_PRIMARY);
        assert_eq!(oft_secondary_fields(), OFT_SECONDARY);
        assert_eq!(descriptor("oft").unwrap().modelled(), OFT_PRIMARY);
    }

    #[test]
    fn ymaze_lists() {
        assert_eq!(ymaze_score_fields(), YM_SCORES);
        assert_eq!(descriptor("ymaze").unwrap().modelled(), [YM_SCORES[0]]);
    }

    #[test]
    fn nort_lists() {
        assert_eq!(nort_primary_fields(), ["DI"]);
        assert_eq!(
            nort_secondary_fields(),
            ["RI", "LR", "object_bias", "side_bias"]
        );
        let nort = descriptor("nort").unwrap();
        assert_eq!(nort.modelled(), ["DI"]);
        // Secondary endpoints must not leak into the review criteria.
        assert_eq!(nort.criteria(), ["DI"]);
    }

    #[test]
    fn plotting_subgroup_endpoints() {
        // plotting.rs subgroup panels: oft primary, first y-maze score, nort DI.
        assert_eq!(&ymaze_score_fields()[..1], ["spontaneous_alternation_pct"]);
    }

    #[test]
    fn contracts_mirror_descriptors() {
        let assays: Vec<_> = post_analysis_descriptors()
            .map(|d| d.post_analysis.as_ref().unwrap().assay.as_str())
            .collect();
        assert_eq!(assays, ["oft", "nort", "ymaze"]);
        // The scorer dispatch in metrics.rs stays code: oft, ymaze, nort.
        assert_eq!(assays_in_report_order(), ["oft", "ymaze", "nort"]);
        let mut dispatched = ["oft", "ymaze", "nort"];
        dispatched.sort_unstable();
        let mut described = assays;
        described.sort_unstable();
        assert_eq!(described, dispatched);
    }

    #[test]
    fn figure_folder_follows_report_order() {
        assert_eq!(
            crate::export::relative_path("figure_nort_ymaze_x.png"),
            "figures/ymaze/figure_nort_ymaze_x.png"
        );
    }
}
