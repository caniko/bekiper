impl PipelineStep for MedianSmoother {
    fn name(&self) -> &str {
        "median_smoother"
    }

    fn apply(&self, lf: LazyFrame) -> LazyFrame {
        let mut lf = lf;
        for col_name in &self.columns {
            // First interpolate nulls, then apply median filter.
            lf = lf.with_column(interpolate_expr(col_name).alias(col_name));
            lf = lf.with_column(median_filter_expr(col_name, self.window_size).alias(col_name));
        }
        lf
    }
}

#[derive(Debug, Clone)]
pub struct KalmanSmootherColumns {
    pub label: String,
    pub x_col: String,
    pub y_col: String,
    pub likelihood_col: Option<String>,
}

/// Confidence-weighted constant-velocity Kalman + RTS smoother.
///
/// Research basis: Lightning Pose introduced an Ensemble Kalman Smoother
/// for animal pose post-processing (Biderman, Whiteway et al., Nature
/// Methods 2024, <https://doi.org/10.1038/s41592-024-02319-1>; reference
/// implementation: <https://github.com/paninski-lab/eks>). This is the
/// single-estimate variant used by Bekiper: likelihood controls observation
/// noise, while the backward Rauch-Tung-Striebel pass uses future frames to
/// repair short gaps and suppress implausible jitter. Full EKS can be added
/// later when multiple prediction streams are available.
pub struct ConfidenceKalmanSmoother {
    pub columns: Vec<KalmanSmootherColumns>,
    pub process_noise: f64,
    pub measurement_noise: f64,
    pub min_likelihood: f64,
    pub max_gap_frames: usize,
}

/// Skeleton-aware confidence Kalman + RTS smoother.
///
/// The first pass runs the same per-keypoint confidence-weighted RTS smoother.
/// The second pass projects short-gap interpolated keypoints back toward
/// model-owned bone lengths learned from trusted frames.
pub struct SkeletonKinematicSmoother {
    pub skeleton: Skeleton,
    pub columns: Vec<KalmanSmootherColumns>,
    pub process_noise: f64,
    pub measurement_noise: f64,
    pub min_likelihood: f64,
    pub max_gap_frames: usize,
    pub constraint_iterations: usize,
    pub constraint_strength: f64,
}
