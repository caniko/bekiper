//! Explicit, provenance-preserving adapters for canonical session records.
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Read;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail, ensure};
use polars::prelude::*;
use serde_json::{Map, Value, json};

use crate::{Record, RunOptions};

#[path = "inputs/json.rs"]
mod json;
#[path = "inputs/snapshot.rs"]
mod snapshot;

pub const KEY: [&str; 5] = ["animal_id", "assay", "session_id", "timepoint", "phase"];
pub const FACTORS: [&str; 3] = ["five_xfad", "hcar1", "treatment"];
pub const OUTPUT_IDS: [&str; 9] = [
    "metrics",
    "models",
    "correlations",
    "within_group_tests",
    "figures",
    "subgroup_inference",
    "spatial_location_bias",
    "report",
    "report_pdf",
];
const METADATA_FIELDS: [&str; 10] = [
    "five_xfad",
    "hcar1",
    "treatment",
    "group",
    "sex",
    "age_treatment_window",
    "excluded",
    "exclusion_reason",
    "pair_id",
    "animal_id",
];
pub type SessionKey = [String; 5];
const CONFIG_FINGERPRINT: &str = "_adapter_config_fingerprint";

pub fn load_json(path: &Path) -> Result<Value> {
    snapshot::read_pinned(path, |file| {
        json::from_reader(file).with_context(|| format!("invalid JSON in {}", path.display()))
    })
    .map(|(value, _)| value)
}

pub fn fingerprint(path: &Path) -> Result<Record> {
    snapshot::fingerprint_file(&mut File::open(path)?, path)
}

mod config;
use config::unique_strings;
pub(crate) use config::{SubgroupPlan, validated_subgroup_plan};
pub use config::{
    allowed_thresholds, choice, context_threshold_pct, inherit_threshold_geometry,
    requested_output_ids, resolve_primary_runs, resolve_ymaze_dwell, resolved_config,
    subgroup_plan, validate_sensitivity_thresholds, validate_thresholds_in,
};

mod tables;
pub use tables::read_table;
use tables::read_table_pinned;

mod ods;
use ods::decode_ods;
pub use ods::{read_ods, read_ods_source};

mod normalize;
pub use normalize::{apply_session_value_mappings, normalize, session_key};
use normalize::{canonical_string, normalize_owned, numeric_text};

mod metadata;
pub use metadata::validate_metadata;

mod cohort;
pub use cohort::load_inputs;
