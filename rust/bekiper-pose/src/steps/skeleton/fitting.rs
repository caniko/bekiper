impl SkeletonRepairStep {
    pub fn fit(&self, df: &DataFrame) -> Result<FittedSkeletonRepair> {
        let label_order = self.skeleton.labels();
        let label_to_index: HashMap<String, usize> = label_order
            .iter()
            .enumerate()
            .map(|(idx, label)| (label.clone(), idx))
            .collect();
        let coords = {
            let mut collected = Vec::with_capacity(label_order.len());
            for label in &label_order {
                let x_name = format!("{label}_x");
                let y_name = format!("{label}_y");
                let xs = df
                    .column(&x_name)?
                    .f64()
                    .map_err(|_| BekiperError::Data(format!("{x_name} must be Float64")))?
                    .into_iter()
                    .collect();
                let ys = df
                    .column(&y_name)?
                    .f64()
                    .map_err(|_| BekiperError::Data(format!("{y_name} must be Float64")))?
                    .into_iter()
                    .collect();
                collected.push(LabelCoordinates { xs, ys });
            }
            collected
        };

        let mut node_degrees = vec![0usize; label_order.len()];
        let mut baselines = Vec::new();

        for edge in &self.skeleton.edges {
            let source_idx = *label_to_index.get(&edge.source).ok_or_else(|| {
                BekiperError::Data(format!(
                    "skeleton edge source '{}' missing from nodes",
                    edge.source
                ))
            })?;
            let target_idx = *label_to_index.get(&edge.target).ok_or_else(|| {
                BekiperError::Data(format!(
                    "skeleton edge target '{}' missing from nodes",
                    edge.target
                ))
            })?;

            node_degrees[source_idx] += 1;
            node_degrees[target_idx] += 1;

            let mut lengths = Vec::new();
            for row_index in 0..df.height() {
                let Some((sx, sy)) = pair_at(&coords[source_idx], row_index) else {
                    continue;
                };
                let Some((tx, ty)) = pair_at(&coords[target_idx], row_index) else {
                    continue;
                };
                lengths.push(euclidean_distance((sx, sy), (tx, ty)));
            }

            if lengths.len() < self.min_baseline_frames {
                continue;
            }

            let median = median(&lengths).ok_or_else(|| {
                BekiperError::Data(format!(
                    "cannot compute median baseline for {} -> {}",
                    edge.source, edge.target
                ))
            })?;
            let mad = median_absolute_deviation(&lengths, median).unwrap_or(0.0);

            baselines.push(FittedEdgeBaseline {
                source_idx,
                target_idx,
                median,
                mad,
            });
        }

        Ok(FittedSkeletonRepair {
            label_order,
            label_to_index,
            node_degrees,
            baselines,
            relative_tolerance: self.relative_tolerance,
            mad_multiplier: self.mad_multiplier,
        })
    }
}
