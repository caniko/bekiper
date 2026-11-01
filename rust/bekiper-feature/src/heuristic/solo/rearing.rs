use polars::prelude::*;
use serde::{Deserialize, Serialize};

use bekiper_core::inspection::HeuristicDecompositionSpec;
use bekiper_math::geometry::euclidean_distance_expr;
use bekiper_math::velocity::velocity_expr;

use crate::heuristic::Heuristic;
use crate::heuristic::{coordinate_label_from_xy, decomposition_component, diag_column_name};

use super::{ForepawRule, Forepaws, forepaw_proximity_expr, speed_per_frame};

// ── Rearing defaults ──────────────────────────────────────────────────────
// These are the canonical defaults used throughout the workspace. The task
// layer (`bekiper-behaviour`) re-exports them so user-facing documentation
// stays close to the task, and the ingress layer overlays TOML values via
// `TaskParamsOverrides::rearing`.

/// Maximum centroid speed (m/s) for rearing. Animals that locomote faster
/// than this are not classified as rearing.
pub const DEFAULT_MAX_SPEED_MPS: f64 = 0.01;

/// Maximum nose-to-tail distance as a fraction of the animal's baseline
/// body length. Below this ratio the body is considered compact (upright).
pub const DEFAULT_MAX_BODY_LENGTH_RATIO: f64 = 0.7;

/// Absolute maximum nose-to-tail distance (m) during rearing.
pub const DEFAULT_MAX_BODY_LENGTH: f64 = 0.05;

/// Forepaw-to-head proximity (m) required to classify forepaws as
/// "elevated" when the optional forepaw rule is enabled.
pub const DEFAULT_FOREPAW_PROXIMITY: f64 = 0.03;

/// Detects rearing: mouse stands on hind legs with forepaws off the floor.
///
/// In top-view tracking, rearing manifests as:
/// - The body becomes more compact (nose-to-tail distance decreases)
/// - The animal is relatively stationary (low speed)
/// - Duration ≥ 0.5s (spec requirement)
///
/// With full skeleton (13 kp + forepaws), we can also check forepaw
/// elevation relative to body center. This heuristic supports both modes.
///
/// The temporal requirement (≥0.5s) is applied via tolerance after detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RearingHeuristic {
    /// Nose coordinate columns.
    pub nose_x: String,
    pub nose_y: String,
    /// Tail base coordinate columns.
    pub tail_base_x: String,
    pub tail_base_y: String,
    /// Body centroid columns (for speed).
    pub centroid_x: String,
    pub centroid_y: String,
    /// Maximum centroid speed (m/frame) for rearing. Animal must be relatively still.
    pub max_speed: f64,
    /// Maximum nose-to-tail distance ratio. During rearing the body shortens.
    /// Expressed as fraction of the baseline (running median) body length.
    /// When body length drops below this fraction of the median, it may indicate rearing.
    pub max_body_length_ratio: f64,
    /// Absolute maximum nose-to-tail distance (meters). If body length is below
    /// this value AND speed is low, mark as potential rearing.
    pub max_body_length: f64,
    /// Optional: forepaw columns. If present, check that forepaws are closer
    /// to the head midpoint than usual (indicating elevation).
    pub forepaws: Option<Forepaws>,
}

impl RearingHeuristic {
    /// Create with default parameters for top-view tracking at given FPS.
    pub fn new(fps: f64) -> Self {
        Self {
            nose_x: "nose_x".into(),
            nose_y: "nose_y".into(),
            tail_base_x: "tail_base_x".into(),
            tail_base_y: "tail_base_y".into(),
            centroid_x: "mouse_center_x".into(),
            centroid_y: "mouse_center_y".into(),
            max_speed: speed_per_frame(DEFAULT_MAX_SPEED_MPS, fps),
            max_body_length_ratio: DEFAULT_MAX_BODY_LENGTH_RATIO,
            max_body_length: DEFAULT_MAX_BODY_LENGTH,
            forepaws: None,
        }
    }

    /// Enable forepaw detection (requires full skeleton with forepaw columns).
    pub fn with_forepaws(mut self) -> Self {
        self.forepaws = Some(Forepaws::default());
        self
    }

    fn low_speed_col(&self) -> String {
        diag_column_name(self.name(), "low_speed")
    }

    fn compact_body_col(&self) -> String {
        diag_column_name(self.name(), "compact_body")
    }

    fn forepaws_col(&self) -> String {
        diag_column_name(self.name(), "forepaws_near_head")
    }
}

impl Heuristic for RearingHeuristic {
    fn name(&self) -> &str {
        "rearing"
    }

    fn evaluate(&self, lf: LazyFrame) -> LazyFrame {
        // Body length: nose-to-tail distance
        let body_length = euclidean_distance_expr(
            &self.nose_x,
            &self.nose_y,
            &self.tail_base_x,
            &self.tail_base_y,
        );

        // Speed check
        let speed = velocity_expr(&self.centroid_x, &self.centroid_y);
        let is_slow = speed.lt_eq(lit(self.max_speed)).fill_null(lit(false));

        // Body is compact (short nose-to-tail)
        let is_compact = body_length
            .lt_eq(lit(self.max_body_length))
            .fill_null(lit(false));

        let mut rearing_expr = is_slow.clone().and(is_compact.clone());
        let mut columns = vec![
            is_slow.alias(self.low_speed_col()),
            is_compact.alias(self.compact_body_col()),
        ];

        // If forepaw columns are available, both must be near head.
        if let Some(fp) = &self.forepaws {
            let near = forepaw_proximity_expr(
                fp,
                &self.nose_x,
                &self.nose_y,
                DEFAULT_FOREPAW_PROXIMITY,
                ForepawRule::Both,
            );
            rearing_expr = rearing_expr.and(near.clone());
            columns.push(near.fill_null(lit(false)).alias(self.forepaws_col()));
        }

        columns.push(rearing_expr.fill_null(lit(false)).alias(self.name()));
        lf.with_columns(columns)
    }

    fn decomposition(&self) -> Option<HeuristicDecompositionSpec> {
        let perimeter = None;
        let mut components = vec![
            decomposition_component(
                "low_speed",
                self.low_speed_col(),
                vec![coordinate_label_from_xy(&self.centroid_x)],
                perimeter.clone(),
            ),
            decomposition_component(
                "compact_body",
                self.compact_body_col(),
                vec![coordinate_label_from_xy(&self.nose_x)],
                perimeter.clone(),
            ),
        ];
        if self.forepaws.is_some() {
            components.push(decomposition_component(
                "forepaws_near_head",
                self.forepaws_col(),
                vec![coordinate_label_from_xy(&self.nose_x)],
                perimeter.clone(),
            ));
        }
        Some(HeuristicDecompositionSpec {
            heuristic_name: self.name().to_string(),
            result_column: self.name().to_string(),
            coordinate_labels: vec![coordinate_label_from_xy(&self.nose_x)],
            perimeter_label: None,
            components,
            show_final_result: true,
        })
    }
}
