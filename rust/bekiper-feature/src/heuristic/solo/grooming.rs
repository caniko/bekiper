use polars::prelude::*;
use serde::{Deserialize, Serialize};

use bekiper_core::inspection::HeuristicDecompositionSpec;
use bekiper_math::geometry::euclidean_distance_expr;
use bekiper_math::velocity::velocity_expr;

use crate::heuristic::Heuristic;
use crate::heuristic::{coordinate_label_from_xy, decomposition_component, diag_column_name};

use super::{ForepawRule, Forepaws, forepaw_proximity_expr, speed_per_frame};

// ── Grooming defaults ─────────────────────────────────────────────────────
// Canonical defaults; task layer re-exports and ingress layer overrides
// via `TaskParamsOverrides::grooming`.

/// Maximum centroid speed (m/s) during grooming. Grooming is nearly
/// stationary behaviour.
pub const DEFAULT_MAX_SPEED_MPS: f64 = 0.005;

/// Maximum nose-to-tail distance (m) during grooming.
pub const DEFAULT_MAX_BODY_LENGTH: f64 = 0.08;

/// Minimum nose-to-tail distance (m) during grooming. Distinguishes a
/// folded grooming posture from a fully compacted rearing posture.
pub const DEFAULT_MIN_BODY_LENGTH: f64 = 0.04;

/// Forepaw-to-nose proximity (m) for the optional forepaw rule.
pub const DEFAULT_FOREPAW_PROXIMITY: f64 = 0.02;

/// Detects grooming: mouse is stationary and in a curled posture
/// with forepaws near the face.
///
/// In top-view tracking, grooming manifests as:
/// - Very low speed (essentially immobile)
/// - Body is moderately compact (curled posture, not fully stretched)
/// - Duration ≥ 1.0s (spec requirement)
///
/// With full skeleton (forepaw columns), we can also verify forepaw
/// proximity to the face region for higher specificity.
///
/// The temporal requirement (≥1.0s) is applied via tolerance after detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroomingHeuristic {
    pub nose_x: String,
    pub nose_y: String,
    pub tail_base_x: String,
    pub tail_base_y: String,
    pub centroid_x: String,
    pub centroid_y: String,
    /// Maximum speed (m/frame) during grooming. Must be very low.
    pub max_speed: f64,
    /// Maximum nose-to-tail distance (meters). Grooming = curled posture.
    pub max_body_length: f64,
    /// Minimum nose-to-tail distance (meters). Distinguishes grooming from rearing
    /// (rearing is more compact than grooming).
    pub min_body_length: f64,
    /// Optional forepaw columns for higher specificity.
    pub forepaws: Option<Forepaws>,
}

impl GroomingHeuristic {
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
            max_body_length: DEFAULT_MAX_BODY_LENGTH,
            min_body_length: DEFAULT_MIN_BODY_LENGTH,
            forepaws: None,
        }
    }

    /// Enable forepaw detection (requires full skeleton).
    pub fn with_forepaws(mut self) -> Self {
        self.forepaws = Some(Forepaws::default());
        self
    }

    fn low_speed_col(&self) -> String {
        diag_column_name(self.name(), "low_speed")
    }

    fn body_length_col(&self) -> String {
        diag_column_name(self.name(), "body_length_in_range")
    }

    fn forepaws_col(&self) -> String {
        diag_column_name(self.name(), "forepaws_near_face")
    }
}

impl Heuristic for GroomingHeuristic {
    fn name(&self) -> &str {
        "grooming"
    }

    fn evaluate(&self, lf: LazyFrame) -> LazyFrame {
        let body_length = euclidean_distance_expr(
            &self.nose_x,
            &self.nose_y,
            &self.tail_base_x,
            &self.tail_base_y,
        );

        let speed = velocity_expr(&self.centroid_x, &self.centroid_y);
        let is_slow = speed.lt_eq(lit(self.max_speed)).fill_null(lit(false));

        // Body length in grooming range (compact but not as compact as rearing)
        let in_range = body_length
            .clone()
            .gt_eq(lit(self.min_body_length))
            .and(body_length.lt_eq(lit(self.max_body_length)))
            .fill_null(lit(false));

        let mut grooming_expr = is_slow.clone().and(in_range.clone());
        let mut columns = vec![
            is_slow.alias(self.low_speed_col()),
            in_range.alias(self.body_length_col()),
        ];

        // If forepaw columns available, at least one must be near the head.
        if let Some(fp) = &self.forepaws {
            let near = forepaw_proximity_expr(
                fp,
                &self.nose_x,
                &self.nose_y,
                DEFAULT_FOREPAW_PROXIMITY,
                ForepawRule::Either,
            );
            grooming_expr = grooming_expr.and(near.clone());
            columns.push(near.fill_null(lit(false)).alias(self.forepaws_col()));
        }

        columns.push(grooming_expr.fill_null(lit(false)).alias(self.name()));
        lf.with_columns(columns)
    }

    fn decomposition(&self) -> Option<HeuristicDecompositionSpec> {
        let mut components = vec![
            decomposition_component(
                "low_speed",
                self.low_speed_col(),
                vec![coordinate_label_from_xy(&self.centroid_x)],
                None,
            ),
            decomposition_component(
                "body_length_in_range",
                self.body_length_col(),
                vec![coordinate_label_from_xy(&self.nose_x)],
                None,
            ),
        ];
        if self.forepaws.is_some() {
            components.push(decomposition_component(
                "forepaws_near_face",
                self.forepaws_col(),
                vec![coordinate_label_from_xy(&self.nose_x)],
                None,
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
