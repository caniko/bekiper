// Multi-variant OFT scoring (design: docs/src/pipeline/multi-parameter-runs.md).
//
// Variants that differ only in the centre-zone size (`peripheral_band`), the
// freezing displacement threshold and `min_dwell_seconds` share the window
// selection and one heuristic evaluation. The swept columns are rebuilt from
// critical values (centre side `s*`, rolling displacement) with the legacy
// scalar predicate inside the float band; everything after evaluation (dwell
// selection, summaries) runs per variant through the unchanged `finish`.
use bekiper_feature::heuristic::diag_column_name;
use bekiper_perimeter::critical::centered_square_critical;

impl OpenFieldTest {
    fn detailed_from_result(
        &self,
        decompositions: Vec<bekiper_core::inspection::HeuristicDecompositionSpec>,
        result: OftResult,
        timing: &FrameTiming,
    ) -> (crate::AnalysisResult, Option<TaskDetail>) {
        let col_names = [
            "in_center",
            "in_periphery",
            "freezing",
            "rearing",
            "grooming",
        ];
        let mut summaries = crate::summaries_from_df(
            result
                .evaluated_df
                .as_ref()
                .expect("evaluated_df is always Some after full_analyze"),
            &col_names,
            timing,
        );
        let seconds = [
            result.center_summary.duration_seconds,
            result.periphery_summary.duration_seconds,
            result.freezing_seconds,
            result.rearing_seconds,
            result.grooming_seconds,
        ];
        for (summary, seconds) in summaries.iter_mut().zip(seconds) {
            summary.seconds = seconds;
        }
        let analysis = crate::AnalysisResult {
            task_type: self.task_type(),
            summaries,
            decompositions,
            evaluated_df: result.evaluated_df.clone(),
        };
        (analysis, Some(TaskDetail::OpenField(result)))
    }

    /// Everything the shared evaluation depends on. `None` when the variant
    /// cannot take the fast path (non-square centre).
    fn shared_evaluation_key(&self) -> Option<String> {
        let centre = &self.center.shape;
        if centre.width != centre.height {
            return None;
        }
        let mut params = serde_json::to_value(&self.params).ok()?;
        let map = params.as_object_mut()?;
        for swept in ["min_dwell_seconds", "freezing_displacement_threshold"] {
            map.remove(swept);
        }
        Some(format!(
            "{params}|{:?}|{:?}|{:?}|{:?}",
            self.arena.shape,
            (centre.center_x, centre.center_y),
            (
                self.rearing_max_speed,
                self.rearing_max_body_length_ratio,
                self.rearing_max_body_length,
                self.rearing_forepaw_proximity,
                self.grooming_max_speed,
                self.grooming_max_body_length,
                self.grooming_min_body_length,
                self.grooming_forepaw_proximity,
            ),
            (&self.arena.label, &self.center.label),
        ))
    }

    /// Analyse K variants of one recording. Result `k` is identical to
    /// `variants[k].analyze_detailed(lf.clone(), timing)`; variants that differ
    /// only in the swept parameters share one evaluation, others run the
    /// legacy path.
    pub fn analyze_variants(
        variants: &[&OpenFieldTest],
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
            let base = variants[members[0]];
            let window = base.select_window(lf.clone(), timing);
            let heuristics = base.heuristics();
            let base_df = heuristics
                .iter()
                .fold(window.selected.clone().lazy(), |lf, h| h.evaluate(lf))
                .collect()
                .expect("failed to collect OFT results");
            let shared = SharedOft::new(base, &window, base_df);
            for i in members {
                let variant = variants[i];
                let df = shared.variant_frame(variant);
                let decompositions = crate::decompositions_from_heuristics(&variant.heuristics());
                let result = variant.finish(window.clone(), df);
                out[i] = Some(variant.detailed_from_result(decompositions, result, timing));
            }
        }
        out.into_iter()
            .map(|r| r.expect("every variant analysed"))
            .collect()
    }
}

/// Shared per-recording columns of an OFT variant group.
struct SharedOft {
    base_df: DataFrame,
    xs: Float64Chunked,
    ys: Float64Chunked,
    /// Smallest centre side containing each frame's centroid (NaN = missing).
    centre_critical: Vec<f64>,
    in_arena: BooleanChunked,
    rolling: Float64Chunked,
}

impl SharedOft {
    fn new(base: &OpenFieldTest, window: &OftWindowSelection, base_df: DataFrame) -> Self {
        let coords = &base.params.centroid_coords;
        let xs = base_df.column(&coords.x).unwrap().f64().unwrap().clone();
        let ys = base_df.column(&coords.y).unwrap().f64().unwrap().clone();
        let centre = (base.center.shape.center_x, base.center.shape.center_y);
        let centre_critical = xs
            .into_iter()
            .zip(&ys)
            .map(|(x, y)| match (x, y) {
                (Some(x), Some(y)) => centered_square_critical(centre, x, y),
                _ => f64::NAN,
            })
            .collect();
        let freezing = FreezingHeuristic::new(
            &coords.x,
            &coords.y,
            base.params.freezing_window_seconds,
            base.params.freezing_displacement_threshold,
            base.params.fps,
        );
        let rolling = window
            .selected
            .clone()
            .lazy()
            .select([freezing.rolling_displacement_expr().alias("rolling")])
            .collect()
            .expect("failed to collect OFT rolling displacement");
        let in_arena = base_df
            .column(&diag_column_name("in_periphery", "in_arena"))
            .unwrap()
            .bool()
            .unwrap()
            .clone();
        Self {
            base_df,
            xs,
            ys,
            centre_critical,
            in_arena,
            rolling: rolling.column("rolling").unwrap().f64().unwrap().clone(),
        }
    }

    fn variant_frame(&self, variant: &OpenFieldTest) -> DataFrame {
        use bekiper_core::shape::Shape;
        let side = variant.center.shape.width;
        let in_center: Vec<bool> = (0..self.xs.len())
            .map(|i| match (self.xs.get(i), self.ys.get(i)) {
                (Some(x), Some(y)) => {
                    let c = self.centre_critical[i];
                    if !c.is_finite() || (c - side).abs() <= bekiper_math::sweep::band_eps(c, side)
                    {
                        variant.center.shape.contains(x, y)
                    } else {
                        c <= side
                    }
                }
                _ => false,
            })
            .collect();
        let periphery: Vec<bool> = in_center
            .iter()
            .enumerate()
            .map(|(i, &c)| self.in_arena.get(i).unwrap_or(false) && !c)
            .collect();
        let threshold = variant.params.freezing_displacement_threshold;
        let frozen: Vec<bool> = self
            .rolling
            .into_iter()
            .map(|r| r.is_some_and(|r| r <= threshold))
            .collect();
        let mut df = self.base_df.clone();
        let mut put = |name: String, values: &[bool]| {
            df.with_column(BooleanChunked::from_slice(name.as_str().into(), values).into_column())
                .unwrap();
        };
        put(diag_column_name("in_center", "in_zone"), &in_center);
        put("in_center".into(), &in_center);
        put(diag_column_name("in_periphery", "in_center"), &in_center);
        put("in_periphery".into(), &periphery);
        put(
            diag_column_name("freezing", "thresholded_rolling_displacement"),
            &frozen,
        );
        put("freezing".into(), &frozen);
        df
    }
}
