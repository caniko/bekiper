/// Compute `snout_dx`/`snout_dy` = `nose - head` (head direction vector) from the
/// configured `nose_coords`/`head_coords`; defaults are the `nose`/`center_ear` labels.
struct SnoutDirectionStep {
    nose: bekiper_core::types::CoordinateColumns,
    head: bekiper_core::types::CoordinateColumns,
}

impl Default for SnoutDirectionStep {
    fn default() -> Self {
        use bekiper_core::types::CoordinateColumns as C;
        Self {
            nose: C::new("nose_x", "nose_y"),
            head: C::new("center_ear_x", "center_ear_y"),
        }
    }
}

impl bekiper_pose::pipeline::PipelineStep for SnoutDirectionStep {
    fn name(&self) -> &str {
        "snout_direction"
    }
    fn apply(&self, lf: polars::prelude::LazyFrame) -> polars::prelude::LazyFrame {
        use polars::prelude::*;
        lf.with_columns([
            (col(&*self.nose.x) - col(&*self.head.x)).alias("snout_dx"),
            (col(&*self.nose.y) - col(&*self.head.y)).alias("snout_dy"),
        ])
    }
}

/// Compute `left_ear_dx`/`left_ear_dy` and `right_ear_dx`/`right_ear_dy`
/// as direction vectors from `center_ear` through each ear. Needed by
/// `WhiskerHeuristic` for ray-based whisker interaction detection.
struct EarDirectionStep;

impl bekiper_pose::pipeline::PipelineStep for EarDirectionStep {
    fn name(&self) -> &str {
        "ear_direction"
    }
    fn apply(&self, lf: polars::prelude::LazyFrame) -> polars::prelude::LazyFrame {
        use polars::prelude::*;
        lf.with_columns([
            (col("left_ear_x") - col("center_ear_x")).alias("left_ear_dx"),
            (col("left_ear_y") - col("center_ear_y")).alias("left_ear_dy"),
            (col("right_ear_x") - col("center_ear_x")).alias("right_ear_dx"),
            (col("right_ear_y") - col("center_ear_y")).alias("right_ear_dy"),
        ])
    }
}
