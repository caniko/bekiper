use std::path::{Path, PathBuf};

use bekiper_core::error::{BekiperError, Result};
use bekiper_core::inspection::{IngressSessionContext, PerimeterSpec};
use serde_json::{Value, json};

use crate::canonical::{
    AnnotationGroupInfo, CanonicalAnnotationStore, CanonicalCorrectionKind, CanonicalSession,
    CoordinateSpace, ReferenceDocument as CanonicalReferenceDocument,
};
use crate::{
    PerimeterSource, ReferencePoint, VideoContext, canonicalize_vertices, extract_video_context,
    extract_video_number, parse_coco_json, parse_radial_maze, translate_perimeters,
};

const PNG_EXTENSION: &str = ".png";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerimeterAssetFamily {
    RectangleCsv,
    CircleCsv,
    CocoJson,
    RadialMaze,
    Canonical,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RectangleCsvRow {
    pub label: String,
    pub bbox_x: f64,
    pub bbox_y: f64,
    pub bbox_width: f64,
    pub bbox_height: f64,
    pub extras: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RectangleCsvAsset {
    pub path: PathBuf,
    pub header: Option<Vec<String>>,
    pub rows: Vec<RectangleCsvRow>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircleCsvKind {
    EdgePoint,
    CenterPoint,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CircleCsvRow {
    pub label: String,
    pub center_x: f64,
    pub center_y: f64,
    pub radius: f64,
    pub extras: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CircleCsvAsset {
    pub path: PathBuf,
    pub kind: CircleCsvKind,
    pub header: Option<Vec<String>>,
    pub rows: Vec<CircleCsvRow>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CocoJsonAsset {
    pub path: PathBuf,
    pub document: Value,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ArmLineRow {
    pub label: String,
    pub x1: f64,
    pub y1: f64,
    pub x2: f64,
    pub y2: f64,
    pub extras: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RadialMazeAsset {
    pub dir: PathBuf,
    pub arm_lines: Vec<ArmLineRow>,
    pub center_triangle: Value,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EditablePerimeterAsset {
    RectangleCsv {
        perimeter_id: String,
        asset: RectangleCsvAsset,
    },
    CircleCsv {
        perimeter_id: String,
        asset: CircleCsvAsset,
    },
    CocoJson {
        perimeter_id: String,
        asset: CocoJsonAsset,
    },
    RadialMaze {
        perimeter_id: String,
        asset: RadialMazeAsset,
    },
    Canonical {
        perimeter_id: String,
        store: CanonicalAnnotationStore,
        session_key: String,
        session: CanonicalSession,
        specs: Vec<PerimeterSpec>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReferenceRow {
    pub label: String,
    pub x: f64,
    pub y: f64,
    pub image_name: String,
    pub extras: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReferenceDocument {
    pub path: PathBuf,
    pub rows: Vec<ReferenceRow>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetadataDocument {
    pub path: PathBuf,
    pub headers: Vec<String>,
    pub rows: Vec<Vec<String>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VideoMapDocument {
    pub path: PathBuf,
    pub headers: Vec<String>,
    pub rows: Vec<Vec<String>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SessionMapping {
    Metadata {
        row_index: usize,
        source: PerimeterSource,
    },
    VideoMap {
        row_index: usize,
        perimeter_id: String,
    },
    Fallback {
        perimeter_id: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerimeterSelectionSource {
    MetadataPerimeter,
    MetadataChangeReference,
    VideoMap,
    Fallback,
}

impl PerimeterSelectionSource {
    pub fn label(self) -> &'static str {
        match self {
            Self::MetadataPerimeter => "Perimeter",
            Self::MetadataChangeReference => "ChangeReference",
            Self::VideoMap => "video_map.csv",
            Self::Fallback => "fallback asset",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetReferenceState {
    Unavailable,
    SavedRow,
    FallbackPreview,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnnotationCorrectionKind {
    SourceShared,
    IndividualTarget,
    GroupTarget,
    LegacyTarget,
}

impl AnnotationCorrectionKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::SourceShared => "Source shared",
            Self::IndividualTarget => "Individual correction",
            Self::GroupTarget => "Group correction",
            Self::LegacyTarget => "Legacy target correction",
        }
    }
}

impl From<CanonicalCorrectionKind> for AnnotationCorrectionKind {
    fn from(value: CanonicalCorrectionKind) -> Self {
        match value {
            CanonicalCorrectionKind::SourceShared => Self::SourceShared,
            CanonicalCorrectionKind::IndividualTarget => Self::IndividualTarget,
            CanonicalCorrectionKind::GroupTarget => Self::GroupTarget,
            CanonicalCorrectionKind::LegacyTarget => Self::LegacyTarget,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnnotationGroupSummary {
    pub source_perimeter_id: String,
    pub target_image: String,
    pub member_count: usize,
    pub member_labels: Vec<String>,
    pub correction_kind: AnnotationCorrectionKind,
}

impl TargetReferenceState {
    pub fn label(self) -> &'static str {
        match self {
            Self::Unavailable => "unavailable",
            Self::SavedRow => "saved row",
            Self::FallbackPreview => "synthetic fallback point",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AnnotationSession {
    pub annotation_dir: PathBuf,
    pub plugin_dir: PathBuf,
    pub video_path: PathBuf,
    pub context: Option<IngressSessionContext>,
    pub video_context: Option<VideoContext>,
    pub mapping: SessionMapping,
    pub perimeter_id: String,
    pub asset: EditablePerimeterAsset,
    pub resolved_specs: Vec<PerimeterSpec>,
    pub current_target_image: Option<String>,
    pub metadata: Option<MetadataDocument>,
    pub video_map: Option<VideoMapDocument>,
    pub source_reference: Option<ReferenceDocument>,
    pub change_reference: Option<ReferenceDocument>,
    pub current_target_reference_row: Option<usize>,
    pending_geometry: Option<(GeometryWriteScope, Vec<PerimeterSpec>)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GeometryWriteScope {
    CurrentVideo,
    EntireGroup,
    SharedSource,
}

impl RectangleCsvAsset {
    pub fn load(path: &Path) -> Result<Self> {
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(false)
            .flexible(true)
            .from_path(path)
            .map_err(|e| BekiperError::Data(format!("failed to read {}: {e}", path.display())))?;

        let mut header = None;
        let mut rows = Vec::new();

        for record in reader.records() {
            let record =
                record.map_err(|e| BekiperError::Data(format!("{}: {e}", path.display())))?;
            if record.len() < 5 {
                continue;
            }

            if record[1].trim().parse::<f64>().is_err() {
                header = Some(record.iter().map(ToOwned::to_owned).collect());
                continue;
            }

            rows.push(RectangleCsvRow {
                label: record[0].to_string(),
                bbox_x: record[1]
                    .trim()
                    .parse()
                    .map_err(|e| BekiperError::Data(format!("{}: bbox_x: {e}", path.display())))?,
                bbox_y: record[2]
                    .trim()
                    .parse()
                    .map_err(|e| BekiperError::Data(format!("{}: bbox_y: {e}", path.display())))?,
                bbox_width: record[3].trim().parse().map_err(|e| {
                    BekiperError::Data(format!("{}: bbox_width: {e}", path.display()))
                })?,
                bbox_height: record[4].trim().parse().map_err(|e| {
                    BekiperError::Data(format!("{}: bbox_height: {e}", path.display()))
                })?,
                extras: record.iter().skip(5).map(ToOwned::to_owned).collect(),
            });
        }

        if rows.is_empty() {
            return Err(BekiperError::Data(format!(
                "{}: no rectangle rows found",
                path.display()
            )));
        }

        Ok(Self {
            path: path.to_path_buf(),
            header,
            rows,
        })
    }

    pub fn to_specs(&self) -> Vec<PerimeterSpec> {
        self.rows
            .iter()
            .map(|row| PerimeterSpec::Rectangle {
                label: row.label.clone(),
                center_x: row.bbox_x + row.bbox_width / 2.0,
                center_y: row.bbox_y + row.bbox_height / 2.0,
                width: row.bbox_width,
                height: row.bbox_height,
            })
            .collect()
    }

    pub fn replace_from_specs(&mut self, specs: &[PerimeterSpec]) -> Result<()> {
        if specs.is_empty() {
            return Err(BekiperError::Data(
                "cannot replace rectangle CSV with no specs".into(),
            ));
        }

        let template = self.rows.first().cloned();
        self.rows = specs
            .iter()
            .enumerate()
            .map(|(index, spec)| {
                let rect = bounding_rect_from_spec(spec);
                let extras = self
                    .rows
                    .get(index)
                    .or(template.as_ref())
                    .map(|row| row.extras.clone())
                    .unwrap_or_default();
                RectangleCsvRow {
                    label: rect.0,
                    bbox_x: rect.1,
                    bbox_y: rect.2,
                    bbox_width: rect.3,
                    bbox_height: rect.4,
                    extras,
                }
            })
            .collect();

        Ok(())
    }

    pub fn source_image_name(&self) -> Option<&str> {
        self.rows
            .first()
            .and_then(|row| row.extras.first())
            .map(String::as_str)
    }

    pub fn write(&self) -> Result<()> {
        let mut writer = csv::WriterBuilder::new()
            .has_headers(false)
            .from_path(&self.path)
            .map_err(|e| {
                BekiperError::Data(format!("failed to write {}: {e}", self.path.display()))
            })?;

        if let Some(header) = &self.header {
            writer
                .write_record(header)
                .map_err(|e| BekiperError::Data(format!("{}: {e}", self.path.display())))?;
        }

        for row in &self.rows {
            let mut record = vec![
                row.label.clone(),
                format_number(row.bbox_x),
                format_number(row.bbox_y),
                format_number(row.bbox_width),
                format_number(row.bbox_height),
            ];
            record.extend(row.extras.clone());
            writer
                .write_record(&record)
                .map_err(|e| BekiperError::Data(format!("{}: {e}", self.path.display())))?;
        }

        writer
            .flush()
            .map_err(|e| BekiperError::Data(format!("{}: {e}", self.path.display())))
    }
}
