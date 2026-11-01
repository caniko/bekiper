use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use bekiper_core::error::{BekiperError, Result};
use bekiper_core::inspection::{PerimeterSpec, PerimetersFile};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::{
    PerimeterSource, ReferencePoint, VideoContext, load_change_references, load_perimeter_by_id,
    load_source_reference, read_metadata_csv, read_video_map,
};

#[cfg(test)]
use crate::parse_rectangle_csv;

pub const CANONICAL_DIR: &str = "bekiper";
pub const SCHEMA_VERSION: u32 = 1;

/// Facts captured once at ingestion, never inferred from later review/save state.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IngestionProvenance {
    pub imported: bool,
    pub reviewed_externally: bool,
    pub ingested_unix_s: u64,
    pub validation_request_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FileFingerprint {
    pub path: String,
    pub len: u64,
    pub modified_unix_s: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StoreManifest {
    pub schema_version: u32,
    pub imported_unix_s: u64,
    pub source_layout: String,
    #[serde(default)]
    pub external_fingerprints: Vec<FileFingerprint>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CanonicalSession {
    pub key: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub video_number: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub animal: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stage: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase_part: Option<u32>,
    pub perimeter_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_perimeter_id: Option<String>,
    #[serde(default)]
    pub coordinate_space: CoordinateSpace,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_image: Option<String>,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CoordinateSpace {
    #[default]
    Source,
    Target,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReferenceDocument {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<ReferencePointDocument>,
    #[serde(default)]
    pub targets: BTreeMap<String, ReferencePointDocument>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReferencePointDocument {
    pub x: f64,
    pub y: f64,
    pub image_name: String,
}

impl From<ReferencePoint> for ReferencePointDocument {
    fn from(point: ReferencePoint) -> Self {
        Self {
            x: point.x,
            y: point.y,
            image_name: point.image_name,
        }
    }
}

impl From<ReferencePointDocument> for ReferencePoint {
    fn from(point: ReferencePointDocument) -> Self {
        Self {
            x: point.x,
            y: point.y,
            image_name: point.image_name,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SyncState {
    #[serde(default)]
    pub legacy_fingerprints: Vec<FileFingerprint>,
    #[serde(default)]
    pub canonical_fingerprints: Vec<FileFingerprint>,
    pub synced_unix_s: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncMode {
    TwoWay,
    ImportOnly,
    ExportOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SyncOptions {
    pub mode: SyncMode,
    pub dry_run: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncAction {
    Import,
    Export,
    Noop,
    Conflict,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncReport {
    pub action: SyncAction,
    pub canonical_exists: bool,
    pub legacy_changed: bool,
    pub canonical_changed: bool,
    pub conflict_path: Option<PathBuf>,
    pub dry_run: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoreStatus {
    pub canonical_exists: bool,
    pub legacy_changed: bool,
    pub canonical_changed: bool,
    pub preferred_source: &'static str,
    pub conflicts: Vec<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct AnnotationGroupKey {
    pub source_perimeter_id: String,
    pub target_image: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CanonicalCorrectionKind {
    SourceShared,
    IndividualTarget,
    GroupTarget,
    LegacyTarget,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AnnotationGroupInfo {
    pub key: AnnotationGroupKey,
    pub members: Vec<CanonicalSession>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalAnnotationStore {
    pub annotation_dir: PathBuf,
    pub root: PathBuf,
}
