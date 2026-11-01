#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use bekiper_behaviour::TaskDetail;
    use bekiper_behaviour::object_recognition::NortResult;
    use bekiper_behaviour::open_field::OftResult;
    use bekiper_behaviour::radial_arm::{DEFAULT_PROXIMAL_THRESHOLD, YMazeResult};
    use bekiper_core::inspection::{
        CoordinateColumnSpec, FeatureInspection, HeuristicDecompositionSpec, HeuristicMeta,
        PerimeterSpec, QuadrantSpec,
    };
    use bekiper_core::scale::ScaleDerivationInfo;
    use bekiper_core::types::{Label, TaskType};

    use super::{
        ProximitySpec, RayCastSpec, collect_feature_inspections, collect_task_detailed_inspections,
        collect_ymaze_arm_section_inspections,
    };

    include!("tests/coord_cols.rs");
    include!("tests/spatial_heuristic_with_matching_perimeter_becomes_zone.rs");
    include!("tests/ray_cast_skipped_when_origin_label_missing.rs");
    include!("tests/story_keys_for_object_targeted_heuristics.rs");
}
