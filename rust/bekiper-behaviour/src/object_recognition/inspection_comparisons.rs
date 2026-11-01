impl ObjectRecognition {
    fn persist_inspection_comparisons(&self, df: &mut DataFrame, timing: &BoutTiming) {
        if !self.enable_qualia {
            return;
        }
        for perimeter in self.actual_object_perimeters() {
            for merge_name in self.sorted_merge_rule_names() {
                let rule = &self.merge_rules[merge_name];
                let expand = |names: &[String]| {
                    names
                        .iter()
                        .map(|n| format!("{n}_{}", perimeter.label))
                        .collect()
                };
                let rule = MergeRule {
                    require: expand(&rule.require),
                    any_of: expand(&rule.any_of),
                    exclude: expand(&rule.exclude),
                };
                let states = bekiper_feature::heuristic::combined::temporal_comparison::temporal_fusion_states(
                    df, &rule, timing, self.min_bout_seconds, self.max_gap_seconds, self.gap_density_threshold)
                    .expect("valid native fusion comparison inputs");
                for (name, mask) in [
                    "raw_fusion",
                    "tolerance_after_fusion",
                    "tolerance_within_channels",
                    "tolerance_both",
                ]
                .iter()
                .zip(states.iter())
                {
                    persist_boolean_mask(
                        df,
                        &diag_column_name(
                            &format!("fusion_comparison_{merge_name}_{}", perimeter.label),
                            name,
                        ),
                        mask,
                    );
                }
            }
        }
    }

    fn inspection_decompositions(
        &self,
        heuristics: &[Box<dyn Heuristic>],
    ) -> Vec<HeuristicDecompositionSpec> {
        let mut specs = crate::decompositions_from_heuristics(heuristics);
        if !self.enable_qualia {
            return specs;
        }
        for perimeter in self.actual_object_perimeters() {
            for merge_name in self.sorted_merge_rule_names() {
                if self.merge_rules[merge_name].is_empty() {
                    continue;
                }
                let name = format!("fusion_comparison_{merge_name}_{}", perimeter.label);
                specs.push(HeuristicDecompositionSpec {
                    heuristic_name: name.clone(),
                    result_column: diag_column_name(&name, "tolerance_both"),
                    coordinate_labels: vec![coordinate_label_from_xy(&self.params.head_coords.x)],
                    perimeter_label: Some(perimeter.label.clone()),
                    components: [
                        "raw_fusion",
                        "tolerance_after_fusion",
                        "tolerance_within_channels",
                        "tolerance_both",
                    ]
                    .iter()
                    .map(|state| {
                        decomposition_component(
                            format!(
                                "{state} (bout ≥{}s; gap ≤{}s; density {})",
                                self.min_bout_seconds,
                                self.max_gap_seconds,
                                self.gap_density_threshold
                            ),
                            diag_column_name(&name, state),
                            vec![coordinate_label_from_xy(&self.params.head_coords.x)],
                            Some(perimeter.label.clone()),
                        )
                        .with_geometry(&self.effective_perimeter(perimeter).shape, &perimeter.label)
                    })
                    .collect(),
                    show_final_result: false,
                });
            }
        }
        specs
    }
}
