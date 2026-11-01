//! Timestamped event reconstruction. Input candidates do not establish upstream
//! geometry/contact verification; that requires separately pinned rule evidence.
use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context, Result, ensure};
use serde_json::{Map, Value, json};

use crate::contracts::{KEY, Record};
use crate::metrics::{nonnegative, same_value};
use bekiper_feature::heuristic::combined::exploration::{
    IMEN_MAX_GAP_SECONDS, IMEN_MIN_BOUT_SECONDS,
};

pub const GEOMETRY: &str = "imen_radial_vertex_span_v1";
/// Owner decision 2026-10-28: p% cross-line from the arm start edge to the distal line.
pub const GEOMETRY_ARM_START: &str = "imen_arm_start_crossline_v1";
pub const ENTRY_RULE: &str = "dual_landmark_false_to_true_v1";

mod window;
use window::{finite, text};
pub use window::{reconstruction_window, validate_window};

mod roles;
pub use roles::{nort_object_roles, nort_physical_sides};

mod geometry;
use geometry::polygon;
pub use geometry::polygon_contains;

mod ymaze;
pub use ymaze::{
    ymaze_coordinate_entries, ymaze_coordinate_entries_with_policy,
    ymaze_coordinate_entries_with_zone_gap,
};

mod lineage;
pub use lineage::{
    assay_provenance, complete_landmark_roles, ymaze_provenance, ymaze_provenance_registered,
    ymaze_provenance_with_policy,
};

mod nort;
pub use nort::{nort_bouts, nort_bouts_with_policy};

mod oft;
pub use oft::oft_zone_intervals;
