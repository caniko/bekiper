use std::collections::{HashMap, HashSet};
use std::io::Read;
use std::path::{Path, PathBuf};

use bekiper_core::error::{BekiperError, Result};
use indexmap::IndexMap;
use roxmltree::Document;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use zip::ZipArchive;

use super::cell::CellValue;
use super::trial::TrialResult;
use crate::config::{AggregateMetadataConfig, MissingMetadataPolicy};

pub(super) const METADATA_COLUMNS: &[&str] = &[
    "animal",
    "group",
    "group_name",
    "sex",
    "hcar1",
    "age_group",
    "treatment_onset",
    "five_xfad",
    "age_group_nr",
    "start_age_days",
    "start_age_months",
    "end_age_days",
    "end_age_months",
    "treatment",
    "round",
    "comments",
];

const HEADER_MAP: &[(&str, &str)] = &[
    ("Animal ID", "animal"),
    ("Group", "group"),
    ("Group name", "group_name"),
    ("Sex", "sex"),
    ("HCAR1", "hcar1"),
    ("Age group", "age_group"),
    ("Treatment onset", "treatment_onset"),
    ("5XFAD", "five_xfad"),
    ("Age group nr", "age_group_nr"),
    ("Start age (days)", "start_age_days"),
    ("Start age (months)", "start_age_months"),
    ("End age (days)", "end_age_days"),
    ("End age (months)", "end_age_months"),
    ("Treatment", "treatment"),
    ("Round", "round"),
    ("All comments", "comments"),
];

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct AggregateMetadata {
    pub path: PathBuf,
    pub rows: IndexMap<u32, IndexMap<String, CellValue>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct MetadataExclusion {
    pub animal: u32,
    pub reason: String,
}

include!("metadata/table.rs");

include!("metadata/trial_join.rs");

include!("metadata/xlsx_directory.rs");

include!("metadata/xlsx_rows.rs");

include!("metadata/workbook_source.rs");

include!("metadata/xlsx_cells.rs");

#[cfg(test)]
#[path = "metadata/tests.rs"]
mod tests;
