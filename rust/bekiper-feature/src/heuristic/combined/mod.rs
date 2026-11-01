pub mod exploration;
pub mod merge;
pub mod temporal_comparison;

use polars::prelude::*;

use bekiper_core::inspection::HeuristicDecompositionSpec;
use bekiper_math::discrete::all_of;

use crate::heuristic::Heuristic;

/// A combined heuristic that chains a primary (solo) heuristic with
/// zero or more helper heuristics. The final result is the AND
/// of all constituent heuristic columns.
pub struct CombinedHeuristic {
    pub name: String,
    pub primary: Box<dyn Heuristic>,
    pub helpers: Vec<Box<dyn Heuristic>>,
    pub decomposition: Option<HeuristicDecompositionSpec>,
}

impl CombinedHeuristic {
    pub fn new(
        name: impl Into<String>,
        primary: impl Heuristic + 'static,
        helpers: Vec<Box<dyn Heuristic>>,
    ) -> Self {
        Self {
            name: name.into(),
            primary: Box::new(primary),
            helpers,
            decomposition: None,
        }
    }

    pub fn with_decomposition(mut self, decomposition: HeuristicDecompositionSpec) -> Self {
        self.decomposition = Some(decomposition);
        self
    }
}

impl Heuristic for CombinedHeuristic {
    fn name(&self) -> &str {
        &self.name
    }

    fn evaluate(&self, lf: LazyFrame) -> LazyFrame {
        let mut lf = self.primary.evaluate(lf);

        for helper in &self.helpers {
            lf = helper.evaluate(lf);
        }

        let mut component_cols = vec![col(self.primary.result_column())];
        for helper in &self.helpers {
            component_cols.push(col(helper.result_column()));
        }

        lf.with_column(all_of(component_cols).alias(self.name.as_str()))
    }

    fn decomposition(&self) -> Option<HeuristicDecompositionSpec> {
        let mut spec = self.decomposition.clone()?;
        if let Some(primary) = self.primary.decomposition() {
            for component in &mut spec.components {
                if let Some(source) = primary
                    .components
                    .iter()
                    .find(|source| source.result_column == component.result_column)
                {
                    component.buffer_meters = source.buffer_meters;
                    component
                        .ray_origin_label
                        .clone_from(&source.ray_origin_label);
                    component
                        .ray_direction_x_column
                        .clone_from(&source.ray_direction_x_column);
                    component
                        .ray_direction_y_column
                        .clone_from(&source.ray_direction_y_column);
                    component
                        .ray_target_x_column
                        .clone_from(&source.ray_target_x_column);
                    component
                        .ray_target_y_column
                        .clone_from(&source.ray_target_y_column);
                    component.value_column.clone_from(&source.value_column);
                    component.value_unit.clone_from(&source.value_unit);
                    component
                        .evaluated_perimeter
                        .clone_from(&source.evaluated_perimeter);
                    component
                        .buffer_perimeter
                        .clone_from(&source.buffer_perimeter);
                }
            }
        }
        Some(spec)
    }
}
