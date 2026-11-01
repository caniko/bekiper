pub use bekiper_annotate_protocol::ArenaCenterRule;
pub use bekiper_feature::zone::DEFAULT_VIDEO_CENTER_FRACTION;
use serde::{Deserialize, Serialize};

// ── Detection pipeline defaults ───────────────────────────────────────────
// All values below are expressed as fractions of the image so they scale
// with resolution. Overridable via the `recipe` section of a `ScanConfig`
// in `bekiper-ingress/src/config.rs`.

/// Grayscale-histogram percentile used as the "floor" (background)
/// estimate. 75 means the 75th percentile of pixel intensities.
pub const DEFAULT_FLOOR_PERCENTILE: f64 = 75.0;

/// Multiplier applied to the floor value to derive the binary threshold.
/// Objects darker than `floor * threshold_factor` are detected.
pub const DEFAULT_THRESHOLD_FACTOR: f64 = 0.85;

/// Morphological closing kernel size, as a fraction of the image
/// diagonal. Fills small holes in detected objects.
pub const DEFAULT_CLOSE_KERNEL_FRAC: f64 = 0.007;

/// Minimum blob area (fraction of image area) for a blob to be considered
/// the "mouse" candidate during removal.
pub const DEFAULT_MOUSE_MIN_AREA_FRAC: f64 = 0.02;

/// Dilation radius (fraction of image diagonal) applied around the mouse
/// candidate to cover tail and shadow pixels.
pub const DEFAULT_MOUSE_DILATE_FRAC: f64 = 0.03;

/// Minimum candidate-object area as a fraction of total image area.
pub const DEFAULT_AREA_MIN_FRAC: f64 = 0.005;

/// Maximum candidate-object area as a fraction of total image area.
pub const DEFAULT_AREA_MAX_FRAC: f64 = 0.05;

/// Full detection pipeline configuration.
///
/// All spatial parameters are expressed as fractions of image area or diagonal
/// so the algorithm works identically on any resolution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionPipelineConfig {
    /// Number of objects expected in the scene.
    pub expected_count: usize,
    /// Threshold configuration.
    #[serde(default)]
    pub threshold: ThresholdConfig,
    /// Mouse (large-blob) removal configuration.
    #[serde(default)]
    pub mouse_removal: MouseRemovalConfig,
    /// Area filter bounds (as fractions of image area).
    #[serde(default)]
    pub area_filter: AreaFilterConfig,
    /// How to assign semantic labels to detected objects.
    pub label_strategy: LabelStrategy,
}

/// How to threshold the grayscale image to isolate dark objects.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThresholdConfig {
    /// Percentile of grayscale histogram used as "floor" estimate.
    pub floor_percentile: f64,
    /// Fraction of the floor value used as the binary threshold.
    /// Objects darker than `floor * threshold_factor` are detected.
    pub threshold_factor: f64,
    /// Morphological close kernel size as fraction of image diagonal.
    pub close_kernel_frac: f64,
}

impl Default for ThresholdConfig {
    fn default() -> Self {
        Self {
            floor_percentile: DEFAULT_FLOOR_PERCENTILE,
            threshold_factor: DEFAULT_THRESHOLD_FACTOR,
            close_kernel_frac: DEFAULT_CLOSE_KERNEL_FRAC,
        }
    }
}

/// Configuration for removing the mouse (largest dark blob) from the scene.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MouseRemovalConfig {
    /// Minimum blob area (as fraction of image area) to be considered "mouse".
    pub min_area_frac: f64,
    /// Dilation radius (as fraction of image diagonal) to cover tail/shadow.
    pub dilate_frac: f64,
}

impl Default for MouseRemovalConfig {
    fn default() -> Self {
        Self {
            min_area_frac: DEFAULT_MOUSE_MIN_AREA_FRAC,
            dilate_frac: DEFAULT_MOUSE_DILATE_FRAC,
        }
    }
}

/// Area bounds for candidate objects, as fractions of total image area.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AreaFilterConfig {
    pub min_frac: f64,
    pub max_frac: f64,
}

impl Default for AreaFilterConfig {
    fn default() -> Self {
        Self {
            min_frac: DEFAULT_AREA_MIN_FRAC,
            max_frac: DEFAULT_AREA_MAX_FRAC,
        }
    }
}

/// Strategy for assigning semantic labels to detected objects.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "method")]
pub enum LabelStrategy {
    /// Assign labels based on diagonal position (sum of x+y coordinates).
    /// First label goes to the object closest to top-left.
    #[serde(rename = "diagonal")]
    Diagonal { labels: Vec<String> },
    /// Assign labels sorted by position: top-to-bottom, then left-to-right.
    #[serde(rename = "positional")]
    Positional { labels: Vec<String> },
    /// Assign a single label to the largest detected object.
    #[serde(rename = "largest")]
    Largest { label: String },
}

pub fn default_arena_sample_positions() -> Vec<f64> {
    vec![0.10, 0.50, 0.90]
}

pub const DEFAULT_ARENA_SEARCH_MARGIN_FRACTION: f64 = 0.05;
/// Cropped NORT recordings may put enclosure edges inside the OFT margin.
pub const DEFAULT_NORT_ARENA_SEARCH_MARGIN_FRACTION: f64 = 0.0;
pub const DEFAULT_ARENA_SMOOTHING_FRACTION: f64 = 0.01;
pub const DEFAULT_ARENA_MIN_SIDE_FRACTION: f64 = 0.45;
pub const DEFAULT_ARENA_MAX_ASPECT_ERROR: f64 = 0.20;
pub const DEFAULT_ARENA_MAX_CENTER_SHIFT_FRACTION: f64 = 0.03;
pub const DEFAULT_ARENA_MAX_SIDE_SHIFT_FRACTION: f64 = 0.05;
pub const DEFAULT_ARENA_MIN_EDGE_SCORE: f64 = 1.50;

/// Deterministic Open Field arena calibration settings.
///
/// This config is intentionally separate from object detection recipes: it
/// estimates the static arena boundary from image edges and is allowed under
/// `--no-auto-detect` because it does not detect animals or objects.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(from = "ArenaCalibrationFile")]
pub struct ArenaCalibrationConfig {
    /// Relative video positions to sample, in `[0.0, 1.0]`.
    pub sample_positions: Vec<f64>,
    /// Fraction of image width/height excluded around the outer edge.
    pub search_margin_fraction: f64,
    /// Moving-average smoothing window as a fraction of image extent.
    pub smoothing_fraction: f64,
    /// Minimum accepted arena side as a fraction of the shorter image side.
    pub min_side_fraction: f64,
    /// Maximum `abs(width / height - 1.0)`.
    pub max_aspect_error: f64,
    /// Maximum center disagreement across sampled frames as image fraction.
    pub max_center_shift_fraction: f64,
    /// Maximum side-length disagreement across sampled frames as image fraction.
    pub max_side_shift_fraction: f64,
    /// Minimum edge score relative to the mean projection score.
    pub min_edge_score: f64,
    /// Optional physical center zone dimensions. Converted to pixels using the
    /// configured `arena_side` scale.
    #[serde(default)]
    pub center_zone_meters: Option<(f64, f64)>,
    /// Explicit manual, fixed-dimension, wall-inset, or video-relative convention.
    /// Absent uses legacy dimensions if supplied, otherwise manual outlines.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub center_zone_rule: Option<ArenaCenterRule>,
    /// Top-level keys written in the parsed config (provenance only; not serialized).
    #[serde(skip)]
    pub configured: Vec<&'static str>,
}

/// Mirror with optional fields so [`ArenaCalibrationConfig::configured`] records
/// which keys the file actually set (a key equal to its default still counts).
#[derive(Deserialize)]
struct ArenaCalibrationFile {
    sample_positions: Option<Vec<f64>>,
    search_margin_fraction: Option<f64>,
    smoothing_fraction: Option<f64>,
    min_side_fraction: Option<f64>,
    max_aspect_error: Option<f64>,
    max_center_shift_fraction: Option<f64>,
    max_side_shift_fraction: Option<f64>,
    min_edge_score: Option<f64>,
    center_zone_meters: Option<(f64, f64)>,
    center_zone_rule: Option<ArenaCenterRule>,
}

impl From<ArenaCalibrationFile> for ArenaCalibrationConfig {
    fn from(f: ArenaCalibrationFile) -> Self {
        let d = Self::default();
        let configured = [
            ("sample_positions", f.sample_positions.is_some()),
            ("search_margin_fraction", f.search_margin_fraction.is_some()),
            ("smoothing_fraction", f.smoothing_fraction.is_some()),
            ("min_side_fraction", f.min_side_fraction.is_some()),
            ("max_aspect_error", f.max_aspect_error.is_some()),
            (
                "max_center_shift_fraction",
                f.max_center_shift_fraction.is_some(),
            ),
            (
                "max_side_shift_fraction",
                f.max_side_shift_fraction.is_some(),
            ),
            ("min_edge_score", f.min_edge_score.is_some()),
            ("center_zone_meters", f.center_zone_meters.is_some()),
            ("center_zone_rule", f.center_zone_rule.is_some()),
        ]
        .into_iter()
        .filter_map(|(k, set)| set.then_some(k))
        .collect();
        Self {
            sample_positions: f.sample_positions.unwrap_or(d.sample_positions),
            search_margin_fraction: f.search_margin_fraction.unwrap_or(d.search_margin_fraction),
            smoothing_fraction: f.smoothing_fraction.unwrap_or(d.smoothing_fraction),
            min_side_fraction: f.min_side_fraction.unwrap_or(d.min_side_fraction),
            max_aspect_error: f.max_aspect_error.unwrap_or(d.max_aspect_error),
            max_center_shift_fraction: f
                .max_center_shift_fraction
                .unwrap_or(d.max_center_shift_fraction),
            max_side_shift_fraction: f
                .max_side_shift_fraction
                .unwrap_or(d.max_side_shift_fraction),
            min_edge_score: f.min_edge_score.unwrap_or(d.min_edge_score),
            center_zone_meters: f.center_zone_meters,
            center_zone_rule: f.center_zone_rule,
            configured,
        }
    }
}

impl Default for ArenaCalibrationConfig {
    fn default() -> Self {
        Self {
            sample_positions: default_arena_sample_positions(),
            search_margin_fraction: DEFAULT_ARENA_SEARCH_MARGIN_FRACTION,
            smoothing_fraction: DEFAULT_ARENA_SMOOTHING_FRACTION,
            min_side_fraction: DEFAULT_ARENA_MIN_SIDE_FRACTION,
            max_aspect_error: DEFAULT_ARENA_MAX_ASPECT_ERROR,
            max_center_shift_fraction: DEFAULT_ARENA_MAX_CENTER_SHIFT_FRACTION,
            max_side_shift_fraction: DEFAULT_ARENA_MAX_SIDE_SHIFT_FRACTION,
            min_edge_score: DEFAULT_ARENA_MIN_EDGE_SCORE,
            center_zone_meters: None,
            center_zone_rule: None,
            configured: Vec::new(),
        }
    }
}

impl ArenaCalibrationConfig {
    pub fn effective_center_rule(&self) -> ArenaCenterRule {
        let rule = self.center_zone_rule.clone().unwrap_or_else(|| {
            self.center_zone_meters.map_or(
                ArenaCenterRule::default(),
                |(width_meters, height_meters)| ArenaCenterRule::Dimensions {
                    width_meters,
                    height_meters,
                },
            )
        });
        match rule {
            ArenaCenterRule::VideoRelative {
                width_fraction,
                height_fraction,
            } => ArenaCenterRule::VideoRelative {
                width_fraction: Some(width_fraction.unwrap_or(DEFAULT_VIDEO_CENTER_FRACTION)),
                height_fraction: Some(height_fraction.unwrap_or(DEFAULT_VIDEO_CENTER_FRACTION)),
            },
            _ => rule,
        }
    }

    pub fn set_center_rule(&mut self, rule: ArenaCenterRule) {
        self.center_zone_rule = Some(rule);
        self.center_zone_meters = None;
        self.configured.push("center_zone_rule");
    }

    /// Image-space NORT proposals search up to the border of cropped recordings.
    pub fn nort_default() -> Self {
        Self {
            search_margin_fraction: DEFAULT_NORT_ARENA_SEARCH_MARGIN_FRACTION,
            ..Self::default()
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.sample_positions.len() > 16
            || self
                .sample_positions
                .iter()
                .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        {
            return Err(
                "arena sample_positions must contain at most 16 finite positions in [0, 1]".into(),
            );
        }
        for (name, value, limit) in [
            ("search_margin_fraction", self.search_margin_fraction, 0.33),
            ("smoothing_fraction", self.smoothing_fraction, 0.25),
            ("min_side_fraction", self.min_side_fraction, 1.0),
            (
                "max_center_shift_fraction",
                self.max_center_shift_fraction,
                1.0,
            ),
            ("max_side_shift_fraction", self.max_side_shift_fraction, 1.0),
        ] {
            if !value.is_finite() || !(0.0..=limit).contains(&value) {
                return Err(format!("arena {name} must be finite and in [0, {limit}]"));
            }
        }
        if !self.max_aspect_error.is_finite()
            || self.max_aspect_error < 0.0
            || !self.min_edge_score.is_finite()
            || self.min_edge_score <= 0.0
        {
            return Err(
                "arena aspect tolerance must be finite/nonnegative and edge score finite/positive"
                    .into(),
            );
        }
        if self
            .center_zone_meters
            .is_some_and(|(w, h)| !w.is_finite() || !h.is_finite() || w <= 0.0 || h <= 0.0)
        {
            return Err("arena center_zone_meters must be finite and positive".into());
        }
        let valid = |v: f64| v.is_finite() && v > 0.0;
        match self.effective_center_rule() {
            ArenaCenterRule::Manual {} => {}
            ArenaCenterRule::VideoRelative {
                width_fraction,
                height_fraction,
            } => {
                if ![width_fraction, height_fraction]
                    .into_iter()
                    .all(|v| v.is_some_and(|v| v.is_finite() && v > 0.0 && v < 1.0))
                {
                    return Err("video-relative center fractions must be finite and strictly between 0 and 1".into());
                }
            }
            ArenaCenterRule::Dimensions {
                width_meters,
                height_meters,
            } if valid(width_meters) && valid(height_meters) => {}
            ArenaCenterRule::WallInset { inset_meters } if valid(inset_meters) => {}
            ArenaCenterRule::FrameCentered {
                arena_width_meters,
                arena_height_meters,
                center_width_meters,
                center_height_meters,
                ..
            } if [
                arena_width_meters,
                arena_height_meters,
                center_width_meters,
                center_height_meters,
            ]
            .into_iter()
            .all(valid)
                && center_width_meters < arena_width_meters
                && center_height_meters < arena_height_meters => {}
            _ => return Err("arena center rule values must be finite positive metres".into()),
        }
        Ok(())
    }

    /// Constructor/request overrides > environment > task TOML > defaults.
    pub fn resolve_from(
        file: Option<Self>,
        explicit: Option<Self>,
        lookup: impl Fn(&str) -> Option<String>,
    ) -> Result<Self, String> {
        if let Some(explicit) = explicit {
            explicit.validate()?;
            return Ok(explicit);
        }
        let mut config = file.unwrap_or_default();
        macro_rules! scalar {
            ($field:ident, $env:literal) => {
                if let Some(raw) = lookup($env) {
                    config.$field = raw.parse().map_err(|_| format!("invalid {}", $env))?;
                }
            };
        }
        scalar!(
            search_margin_fraction,
            "BEKIPER_ARENA_SEARCH_MARGIN_FRACTION"
        );
        scalar!(smoothing_fraction, "BEKIPER_ARENA_SMOOTHING_FRACTION");
        scalar!(min_side_fraction, "BEKIPER_ARENA_MIN_SIDE_FRACTION");
        scalar!(max_aspect_error, "BEKIPER_ARENA_MAX_ASPECT_ERROR");
        scalar!(
            max_center_shift_fraction,
            "BEKIPER_ARENA_MAX_CENTER_SHIFT_FRACTION"
        );
        scalar!(
            max_side_shift_fraction,
            "BEKIPER_ARENA_MAX_SIDE_SHIFT_FRACTION"
        );
        scalar!(min_edge_score, "BEKIPER_ARENA_MIN_EDGE_SCORE");
        let numbers = |raw: &str| {
            raw.split(',')
                .map(|v| {
                    v.trim()
                        .parse::<f64>()
                        .map_err(|_| "invalid arena numeric list".to_string())
                })
                .collect::<Result<Vec<_>, _>>()
        };
        if let Some(raw) = lookup("BEKIPER_ARENA_SAMPLE_POSITIONS") {
            config.sample_positions = numbers(&raw)?;
        }
        if let Some(raw) = lookup("BEKIPER_ARENA_CENTER_RULE") {
            let rule = serde_json::from_str(&raw)
                .map_err(|error| format!("invalid BEKIPER_ARENA_CENTER_RULE: {error}"))?;
            config.set_center_rule(rule);
        } else if let Some(raw) = lookup("BEKIPER_ARENA_CENTER_ZONE_METERS") {
            let values = numbers(&raw)?;
            if values.len() != 2 {
                return Err("BEKIPER_ARENA_CENTER_ZONE_METERS requires width,height".into());
            }
            config.center_zone_meters = Some((values[0], values[1]));
            config.center_zone_rule = None;
        }
        config.validate()?;
        Ok(config)
    }
}

#[cfg(test)]
#[path = "config/arena_layering_tests.rs"]
mod arena_layering_tests;
