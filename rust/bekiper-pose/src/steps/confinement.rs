use polars::prelude::*;

use crate::pipeline::PipelineStep;
use bekiper_core::shape::Shape;
use bekiper_core::types::CoordinateColumns;
use bekiper_perimeter::perimeter::Perimeter;

/// Null coordinate columns where the reference point is outside the enclosure.
///
/// Preserves all rows (and thus frame indices / timing). Coordinates outside
/// the enclosure become null and are later filled by interpolation.
/// This replaces the old `filter_confined` approach which dropped rows and
/// broke frame-based timing calculations.
pub struct ConfinementNuller<S: Shape + Clone + 'static> {
    pub perimeter: Perimeter<S>,
    /// Reference coordinate pair to test against the perimeter.
    pub reference: CoordinateColumns,
    /// All coordinate columns to null when outside (x and y for every keypoint).
    pub columns_to_null: Vec<String>,
}

impl<S: Shape + Clone + Send + Sync + 'static> PipelineStep for ConfinementNuller<S> {
    fn name(&self) -> &str {
        "confinement_nuller"
    }

    fn apply(&self, lf: LazyFrame) -> LazyFrame {
        self.perimeter
            .null_outside(lf, &self.reference, &self.columns_to_null)
    }
}

/// Filter rows to only those where the animal is inside the trial enclosure.
///
/// **Warning**: This drops rows, changing frame indices and breaking timing.
/// Prefer `ConfinementNuller` for the augment pipeline. This variant is kept
/// for cases where row removal is intentional (e.g. post-analysis cropping).
pub struct ConfinementFilter<S: Shape + Clone + 'static> {
    pub perimeter: Perimeter<S>,
    pub coords: CoordinateColumns,
}

impl<S: Shape + Clone + Send + Sync + 'static> PipelineStep for ConfinementFilter<S> {
    fn name(&self) -> &str {
        "confinement_filter"
    }

    fn apply(&self, lf: LazyFrame) -> LazyFrame {
        self.perimeter.filter_confined(lf, &self.coords)
    }
}
