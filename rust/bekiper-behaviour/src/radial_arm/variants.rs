// Multi-variant Y-maze scoring (design: docs/src/pipeline/multi-parameter-runs.md).
//
// Variants that share the maze, landmarks, geometry convention and effective
// pose gap-fill differ only in `p`, dwell/gap, hysteresis, occupancy and start
// mode. Zone membership and the per-point critical `p*` are computed once per
// group; each variant's `in_arm_X` / `in_center` columns compare `p <= p*`
// (legacy scalar inside the float band, and for non-convex cross-line arms).
// Dwell, gap bridging, hysteresis and sequence extraction run per variant
// through the unchanged legacy stage, so state-machine mitigations need no
// special casing.

impl YMaze {
    fn detailed_from(
        &self,
        decompositions: Vec<bekiper_core::inspection::HeuristicDecompositionSpec>,
        result: YMazeResult,
        timing: &bekiper_core::video::FrameTiming,
    ) -> (crate::AnalysisResult, Option<TaskDetail>) {
        let mut col_names: Vec<String> = self.arm_col_names();
        col_names.push("in_center".to_string());
        col_names.push("freezing".to_string());
        let col_refs: Vec<&str> = col_names.iter().map(|s| s.as_str()).collect();
        let summaries = crate::summaries_from_df(
            result
                .evaluated_df
                .as_ref()
                .expect("evaluated_df is always Some after full_analyze"),
            &col_refs,
            timing,
        );
        let analysis = crate::AnalysisResult {
            task_type: self.task_type(),
            summaries,
            decompositions,
            evaluated_df: result.evaluated_df.clone(),
        };
        (analysis, Some(TaskDetail::YMaze(result)))
    }

    /// Pose gap-fill (frames) actually applied to the mitigated frame.
    fn effective_gap_fill_frames(&self) -> u32 {
        let principal = self.uses_principal_entry_rule();
        ttl_frames_from_seconds(
            if principal {
                0.0
            } else {
                self.pose_gap_fill_seconds
            },
            self.params.fps,
        )
    }

    /// Everything the group's shared geometry depends on.
    fn shared_evaluation_key(&self) -> Option<String> {
        self.validate_temporal_policy().ok()?;
        let mut params = serde_json::to_value(&self.params).ok()?;
        params.as_object_mut()?.remove("min_dwell_seconds");
        Some(format!(
            "{params}|{:?}|{:?}|{}",
            self.maze,
            self.threshold_geometry,
            self.effective_gap_fill_frames(),
        ))
    }

    /// Analyse K variants of one recording. Result `k` is identical to
    /// `variants[k].analyze_detailed(lf.clone(), timing)`; variants with equal
    /// shared-evaluation keys share zone geometry, the rest run the legacy path.
    pub fn analyze_variants(
        variants: &[&YMaze],
        lf: LazyFrame,
        timing: &bekiper_core::video::FrameTiming,
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
            let raw_df = lf
                .clone()
                .collect()
                .expect("failed to collect Y-maze poses");
            let fill = base.effective_gap_fill_frames();
            let mitigated_df = if fill > 0 {
                let columns = base.pose_gap_fill_columns();
                let refs = columns.iter().map(String::as_str).collect::<Vec<_>>();
                apply_ttl_forward_fill(raw_df.clone().lazy(), &refs, fill)
                    .collect()
                    .expect("failed to collect Y-maze poses")
            } else {
                raw_df.clone()
            };
            let (raw, mitigated) = (
                YMazeShared::build(base, &raw_df),
                YMazeShared::build(base, &mitigated_df),
            );
            let pair = YMazeSharedPair {
                raw: std::sync::Arc::new(raw),
                mitigated: std::sync::Arc::new(mitigated),
            };
            let input = raw_df.lazy();
            for i in members {
                let variant = variants[i];
                let decompositions = crate::decompositions_from_heuristics(&variant.heuristics());
                let result = variant.full_analyze_with(input.clone(), timing, Some(&pair));
                out[i] = Some(variant.detailed_from(decompositions, result, timing));
            }
        }
        out.into_iter()
            .map(|r| r.expect("every variant analysed"))
            .collect()
    }
}
