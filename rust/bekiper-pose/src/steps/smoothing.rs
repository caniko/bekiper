use polars::prelude::*;

use crate::pipeline::PipelineStep;
use crate::skeleton::Skeleton;
use bekiper_math::statistics::{interpolate_expr, median_filter_expr};

/// Apply rolling median + interpolation smoothing to coordinate columns.
pub struct MedianSmoother {
    pub window_size: usize,
    pub columns: Vec<String>,
}

include!("smoothing/median.rs");

include!("smoothing/kalman_step.rs");

include!("smoothing/skeleton.rs");

include!("smoothing/kalman_filter.rs");

include!("smoothing/linear_algebra.rs");

include!("smoothing/skeleton_tests.rs");
