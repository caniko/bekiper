// Multi-variant NORT scoring (design: docs/src/pipeline/multi-parameter-runs.md).
//
// Variants that differ only in nose/contact distance, head angle, nose speed,
// bout limits, whisker length and (without qualia) footprint erosion share one
// recording evaluation: the object geometry is computed once
// (`SharedObjectGeometry`: critical distances, nearest-point heading, the fan at
// `L_max`) and each variant's exploration/fan/merge columns are rebuilt on top
// of the group's shared frame. Everything after evaluation runs per variant
// through the unchanged `finish_analysis`.
use bekiper_feature::heuristic::combined::exploration::SharedObjectGeometry;

impl ObjectRecognition {
    /// Everything the group's shared frame depends on; `None` when the variant
    /// cannot take the fast path.
    fn shared_evaluation_key(&self) -> Option<String> {
        if self.object_perimeters.is_empty() {
            return None;
        }
        let mut params = serde_json::to_value(&self.params).ok()?;
        params.as_object_mut()?.remove("min_dwell_seconds");
        let merge: Vec<_> = self
            .sorted_merge_rule_names()
            .into_iter()
            .map(|name| (name, serde_json::to_string(&self.merge_rules[name]).ok()))
            .collect();
        Some(format!(
            "{params}|{:?}|{:?}|{:?}|{}|{}|{:?}|{merge:?}|{:?}|{:?}",
            self.object_perimeters,
            self.center_perimeter,
            self.protocol,
            self.exclude_posture,
            self.enable_qualia,
            (
                self.qualia_body_proximity_distance,
                self.qualia_whisker_distance,
                self.qualia_whisker_angle,
                self.qualia_olfaction_distance,
                self.qualia_olfaction_angle,
            ),
            // Erosion also moves the qualia perimeters; whisker length moves
            // only columns that are re-evaluated per variant.
            self.enable_qualia
                .then_some(self.perimeter_erosion_meters.to_bits()),
            self.whisker_fan
                .map(|fan| (fan.apex_fraction.to_bits(), fan.half_angle_deg.to_bits())),
        ))
    }

    /// Heuristics whose columns depend on the swept parameters.
    fn swept_heuristic_names(&self) -> std::collections::HashSet<String> {
        let mut names = std::collections::HashSet::new();
        for perimeter in self.actual_object_perimeters() {
            let label = &perimeter.label;
            names.insert(label.to_string());
            if self.enable_qualia {
                names.insert(format!("whisker_fan_{label}"));
                for merge in self.sorted_merge_rule_names() {
                    names.insert(format!("{merge}_{label}"));
                }
            }
        }
        names
    }

    /// Analyse K variants of one recording. Result `k` is identical to
    /// `variants[k].analyze_detailed(lf.clone(), timing)`; variants whose
    /// shared-evaluation keys match share one evaluation, others run the
    /// legacy path.
    pub fn analyze_variants(
        variants: &[&ObjectRecognition],
        lf: LazyFrame,
        timing: &FrameTiming,
    ) -> Vec<(crate::AnalysisResult, Option<TaskDetail>)> {
        let mut groups: Vec<(Option<String>, Vec<usize>)> = Vec::new();
        for (i, variant) in variants.iter().enumerate() {
            let key = variant.shared_evaluation_key();
            match groups.iter_mut().find(|(k, _)| key.is_some() && *k == key) {
                Some((_, members)) => members.push(i),
                None => groups.push((key, vec![i])),
            }
        }
        let mut out: Vec<Option<(crate::AnalysisResult, Option<TaskDetail>)>> =
            (0..variants.len()).map(|_| None).collect();
        for (_, members) in groups {
            if members.len() == 1 {
                let i = members[0];
                out[i] = Some(variants[i].analyze_detailed(lf.clone(), timing));
                continue;
            }
            let input = lf.clone().collect().expect("failed to collect NORT poses");
            let base = variants[members[0]];
            let l_max = members
                .iter()
                .filter_map(|&i| variants[i].whisker_fan)
                .map(|fan| fan.length_m)
                .fold(f64::NEG_INFINITY, f64::max);
            let fan_max: Option<WhiskerFanParams> = base.whisker_fan.map(|fan| WhiskerFanParams {
                length_m: l_max,
                ..fan
            });
            let shared: HashMap<String, std::sync::Arc<SharedObjectGeometry>> = base
                .actual_object_perimeters()
                .into_iter()
                .map(|perimeter| {
                    let geometry = SharedObjectGeometry::build(
                        &perimeter.shape,
                        &input,
                        &base.params.centroid_coords,
                        &base.params.nose_coords,
                        &base.params.head_coords,
                        fan_max,
                    )
                    .expect("failed to build shared NORT geometry");
                    (perimeter.label.to_string(), std::sync::Arc::new(geometry))
                })
                .collect();
            let with_shared = |variant: &ObjectRecognition| {
                let mut clone = variant.clone();
                clone.shared_geometry = shared.clone();
                clone
            };
            let base_df = with_shared(base)
                .heuristics()
                .iter()
                .fold(input.clone().lazy(), |lf, h| h.evaluate(lf))
                .collect()
                .expect("failed to collect NORT results");
            for (position, &i) in members.iter().enumerate() {
                let variant = variants[i];
                let df = if position == 0 {
                    base_df.clone()
                } else {
                    let names = variant.swept_heuristic_names();
                    with_shared(variant)
                        .heuristics()
                        .into_iter()
                        .filter(|h| names.contains(h.name()))
                        .fold(base_df.clone().lazy(), |lf, h| h.evaluate(lf))
                        .collect()
                        .expect("failed to collect NORT results")
                };
                let heuristics = variant.heuristics();
                let decompositions = variant.inspection_decompositions(&heuristics);
                let inner = variant.finish_analysis(heuristics, df, timing);
                let (result, summaries) = variant.with_summaries(inner, timing);
                out[i] = Some(variant.detailed_from(decompositions, result, summaries));
            }
        }
        out.into_iter()
            .map(|r| r.expect("every variant analysed"))
            .collect()
    }
}

#[cfg(test)]
mod variant_key_tests {
    use super::*;
    use bekiper_feature::heuristic::combined::merge::MergeRule;
    use bekiper_perimeter::circle::Circle;

    #[test]
    fn no_objects_means_no_shared_evaluation() {
        assert!(
            ObjectRecognition::new(vec![], 30.0)
                .shared_evaluation_key()
                .is_none()
        );
    }

    #[test]
    fn qualia_sweeps_fan_and_merge_columns_per_object() {
        let mut task = ObjectRecognition::new(
            vec![Perimeter::new(
                "a",
                ObjectShape::Circle(Circle::new(0.0, 0.0, 0.02)),
            )],
            30.0,
        );
        task.enable_qualia = true;
        task.merge_rules
            .insert("close".into(), MergeRule::default());
        let names = task.swept_heuristic_names();
        for want in ["a", "whisker_fan_a", "close_a"] {
            assert!(names.contains(want), "{names:?}");
        }
    }
}
