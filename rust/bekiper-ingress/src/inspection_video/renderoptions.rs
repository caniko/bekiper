use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};

use image::RgbImage;
use polars::prelude::*;

use bekiper_core::error::{BekiperError, Result};
use bekiper_core::inspection::{
    CoordinateColumnSpec, FeatureInspection, InspectionManifest, PerimeterSpec,
};
use bekiper_core::shape::Expandable;
use bekiper_core::types::Label;
use bekiper_perimeter::polygon::Polygon;
use bekiper_perimeter::rectangle::Rectangle;
use bekiper_perimeter::triangle::Triangle;

use crate::overlay::{
    draw_arrow_alpha, draw_circle_alpha, draw_crosshair_alpha, draw_line_alpha, draw_perimeter,
    draw_polyline_alpha, draw_wedge_alpha, fill_rect_alpha, parse_color, scale_perimeter,
};

const DEFAULT_FFMPEG_PRESET: &str = "slower";
const DEFAULT_VIDEO_POINT_COLOR: &str = "cyan";
const DEFAULT_VIDEO_TRUE_COLOR: &str = "lime";
const DEFAULT_VIDEO_FALSE_COLOR: &str = "red";
const DEFAULT_VIDEO_PERIMETER_COLOR: &str = "b";
const DEFAULT_LINEWIDTH: f64 = 1.5;
const DEFAULT_QUIVER_ALPHA: f32 = 0.60;
const DEFAULT_CRF: u8 = 23;

#[derive(Debug, Clone)]
pub struct RenderOptions {
    pub output: Option<PathBuf>,
    pub output_dir: Option<PathBuf>,
    pub codec: String,
    pub heuristic: Option<String>,
    pub feature: Option<String>,
    pub all_features: bool,
    pub config_path: Option<PathBuf>,
    pub ffmpeg_preset: Option<String>,
    /// Decode this file instead of the manifest's recorded `video_path`.
    pub source_video: Option<PathBuf>,
    /// Constant rate factor for CRF encoders; `None` uses 23.
    pub crf: Option<u8>,
    /// Overview plus scored-heuristic outline colours and a mask timeline strip.
    pub scored: bool,
    /// NORT qualia overview: detection-field overlay plus per-object channel timeline.
    pub qualia: bool,
    /// Integer frame upscale for qualia videos; `None` makes them at least 640 px wide.
    pub upscale: Option<u32>,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            output: None,
            output_dir: None,
            codec: "h264".into(),
            heuristic: None,
            feature: None,
            all_features: false,
            config_path: None,
            ffmpeg_preset: None,
            source_video: None,
            crf: None,
            scored: false,
            qualia: false,
            upscale: None,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct VideoTheme {
    pub ffmpeg_preset: String,
    pub video_point_color: String,
    pub video_true_color: String,
    pub video_false_color: String,
    pub video_perimeter_color: String,
    pub linewidth: f64,
    pub quiver_alpha: f32,
}

include!("renderoptions/theme.rs");

include!("renderoptions/loaded_inspection.rs");

include!("renderoptions/job_state.rs");

include!("renderoptions/encoder.rs");

include!("renderoptions/entrypoints.rs");
