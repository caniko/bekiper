impl FittedSkeletonRepair {
    pub fn label_index(&self, label: &str) -> Option<usize> {
        self.label_to_index.get(label).copied()
    }

    pub fn label_name(&self, label_index: usize) -> &str {
        &self.label_order[label_index]
    }

    pub fn evaluate(&self, df: &DataFrame) -> Result<SkeletonViolationEvaluation> {
        let coords = self.collect_coordinates(df)?;
        let mut flags = vec![vec![false; df.height()]; self.label_order.len()];

        #[allow(clippy::needless_range_loop)]
        for row_index in 0..df.height() {
            for label_idx in self.evaluate_row(&coords, row_index) {
                flags[label_idx][row_index] = true;
            }
        }

        Ok(SkeletonViolationEvaluation::new(
            self.label_order.clone(),
            flags,
        ))
    }

    pub fn row_coordinates(
        &self,
        df: &DataFrame,
        row_index: usize,
    ) -> Result<Vec<Option<(f64, f64)>>> {
        let coords = self.collect_coordinates(df)?;
        Ok(self.row_from_coords(&coords, row_index))
    }

    pub fn evaluate_frame(&self, row_coords: &[Option<(f64, f64)>]) -> BTreeSet<usize> {
        let mut incident_violations = vec![0usize; self.label_order.len()];

        for edge in &self.baselines {
            let Some((sx, sy)) = row_coords[edge.source_idx] else {
                continue;
            };
            let Some((tx, ty)) = row_coords[edge.target_idx] else {
                continue;
            };

            let distance = euclidean_distance((sx, sy), (tx, ty));
            let threshold =
                (self.relative_tolerance * edge.median).max(self.mad_multiplier * edge.mad);
            if (distance - edge.median).abs() > threshold {
                incident_violations[edge.source_idx] += 1;
                incident_violations[edge.target_idx] += 1;
            }
        }

        incident_violations
            .into_iter()
            .enumerate()
            .filter_map(|(label_idx, count)| {
                let degree = self.node_degrees[label_idx];
                if degree == 0 {
                    return None;
                }

                let required = if degree == 1 { 1 } else { 2 };
                (count >= required).then_some(label_idx)
            })
            .collect()
    }

    pub fn apply(
        &self,
        mut df: DataFrame,
        evaluation: &SkeletonViolationEvaluation,
        redetected_rows: &BTreeSet<usize>,
    ) -> Result<DataFrame> {
        for (label_idx, label) in self.label_order.iter().enumerate() {
            let x_name = format!("{label}_x");
            let y_name = format!("{label}_y");

            let x_col = df
                .column(&x_name)?
                .f64()
                .map_err(|_| BekiperError::Data(format!("{x_name} must be Float64")))?;
            let y_col = df
                .column(&y_name)?
                .f64()
                .map_err(|_| BekiperError::Data(format!("{y_name} must be Float64")))?;

            let x_values: Vec<Option<f64>> = x_col
                .into_iter()
                .enumerate()
                .map(|(row_index, value)| {
                    if evaluation.flags[label_idx][row_index] {
                        None
                    } else {
                        value
                    }
                })
                .collect();
            let y_values: Vec<Option<f64>> = y_col
                .into_iter()
                .enumerate()
                .map(|(row_index, value)| {
                    if evaluation.flags[label_idx][row_index] {
                        None
                    } else {
                        value
                    }
                })
                .collect();

            df.with_column(Column::new(x_name.into(), x_values))?;
            df.with_column(Column::new(y_name.into(), y_values))?;
            df.with_column(Column::new(
                format!("{label}_skeleton_violation").into(),
                evaluation.flags[label_idx].clone(),
            ))?;
        }

        df.with_column(Column::new(
            "skeleton_violation_count".into(),
            evaluation.counts.clone(),
        ))?;
        df.with_column(Column::new(
            "skeleton_redetected".into(),
            (0..df.height())
                .map(|row_index| redetected_rows.contains(&row_index))
                .collect::<Vec<bool>>(),
        ))?; // coverage: unreachable — the column is built with exactly df.height() rows

        Ok(df)
    }

    fn collect_coordinates(&self, df: &DataFrame) -> Result<Vec<LabelCoordinates>> {
        let mut coords = Vec::with_capacity(self.label_order.len());

        for label in &self.label_order {
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
            coords.push(LabelCoordinates { xs, ys });
        }

        Ok(coords)
    }

    fn row_from_coords(
        &self,
        coords: &[LabelCoordinates],
        row_index: usize,
    ) -> Vec<Option<(f64, f64)>> {
        coords
            .iter()
            .map(
                |label_coords| match (label_coords.xs[row_index], label_coords.ys[row_index]) {
                    (Some(x), Some(y)) => Some((x, y)),
                    _ => None,
                },
            )
            .collect()
    }

    fn evaluate_row(&self, coords: &[LabelCoordinates], row_index: usize) -> BTreeSet<usize> {
        let row = self.row_from_coords(coords, row_index);
        self.evaluate_frame(&row)
    }
}
