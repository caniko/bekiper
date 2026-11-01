pub mod body_proximity;
pub mod freezing;
pub mod grooming;
pub mod olfaction;
pub mod periphery;
pub mod rearing;
pub mod whisker_config;
pub mod whiskers;
pub mod zone_confinement;

use polars::prelude::*;
use serde::{Deserialize, Serialize};

use bekiper_math::geometry::euclidean_distance_expr;

/// Forepaw coordinate columns (left + right), shared by rearing and grooming heuristics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Forepaws {
    pub left_x: String,
    pub left_y: String,
    pub right_x: String,
    pub right_y: String,
}

impl Default for Forepaws {
    fn default() -> Self {
        Self {
            left_x: "left_forepaw_x".into(),
            left_y: "left_forepaw_y".into(),
            right_x: "right_forepaw_x".into(),
            right_y: "right_forepaw_y".into(),
        }
    }
}

/// How to combine left/right forepaw proximity checks.
pub enum ForepawRule {
    /// Both forepaws must be within threshold.
    Both,
    /// At least one forepaw must be within threshold.
    Either,
}

/// Build a boolean expression that checks forepaw proximity to a reference point.
pub fn forepaw_proximity_expr(
    forepaws: &Forepaws,
    ref_x: &str,
    ref_y: &str,
    threshold: f64,
    rule: ForepawRule,
) -> Expr {
    let left = euclidean_distance_expr(&forepaws.left_x, &forepaws.left_y, ref_x, ref_y);
    let right = euclidean_distance_expr(&forepaws.right_x, &forepaws.right_y, ref_x, ref_y);
    let left_near = left.lt_eq(lit(threshold));
    let right_near = right.lt_eq(lit(threshold));
    match rule {
        ForepawRule::Both => left_near.and(right_near),
        ForepawRule::Either => left_near.or(right_near),
    }
    .fill_null(lit(false))
}

/// Convert a speed threshold from m/s to m/frame.
#[inline]
pub fn speed_per_frame(speed_ms: f64, fps: f64) -> f64 {
    speed_ms / fps
}
