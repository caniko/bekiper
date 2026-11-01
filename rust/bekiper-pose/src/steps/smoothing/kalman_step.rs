impl PipelineStep for ConfidenceKalmanSmoother {
    fn name(&self) -> &str {
        "confidence_kalman_smoother"
    }

    fn apply(&self, lf: LazyFrame) -> LazyFrame {
        let mut df = lf
            .collect()
            .expect("confidence_kalman_smoother: collect failed");

        for cols in &self.columns {
            let (x_values, y_values, likelihoods) = read_coordinate_series(&df, cols)
                .unwrap_or_else(|e| panic!("confidence_kalman_smoother {}: {e}", cols.label));
            let observations_x = observations_for_axis(&x_values, &y_values, &likelihoods, true);
            let observations_y = observations_for_axis(&x_values, &y_values, &likelihoods, false);
            let smooth_x = smooth_axis(
                &observations_x,
                self.process_noise,
                self.measurement_noise,
                self.min_likelihood,
                self.max_gap_frames,
            );
            let smooth_y = smooth_axis(
                &observations_y,
                self.process_noise,
                self.measurement_noise,
                self.min_likelihood,
                self.max_gap_frames,
            );

            df.with_column(Column::new(cols.x_col.clone().into(), smooth_x))
                .expect("confidence_kalman_smoother: write x column failed");
            df.with_column(Column::new(cols.y_col.clone().into(), smooth_y))
                .expect("confidence_kalman_smoother: write y column failed");
        }

        df.lazy()
    }
}
