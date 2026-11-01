impl PipelineStep for SkeletonKinematicSmoother {
    fn name(&self) -> &str {
        "skeleton_kinematic_smoother"
    }

    fn apply(&self, lf: LazyFrame) -> LazyFrame {
        let df = lf
            .collect()
            .expect("skeleton_kinematic_smoother: collect failed");
        let trusted = trusted_masks(&df, &self.columns, self.min_likelihood)
            .expect("skeleton_kinematic_smoother: failed to read trusted masks");
        let baselines =
            fit_skeleton_edge_lengths(&df, &self.skeleton, &self.columns, self.min_likelihood)
                .expect("skeleton_kinematic_smoother: failed to fit skeleton baselines");

        let mut smoothed = ConfidenceKalmanSmoother {
            columns: self.columns.clone(),
            process_noise: self.process_noise,
            measurement_noise: self.measurement_noise,
            min_likelihood: self.min_likelihood,
            max_gap_frames: self.max_gap_frames,
        }
        .apply(df.lazy())
        .collect()
        .expect("skeleton_kinematic_smoother: Kalman pass failed");

        apply_skeleton_constraints(
            &mut smoothed,
            &self.skeleton,
            &self.columns,
            &trusted,
            &baselines,
            self.max_gap_frames,
            self.constraint_iterations.max(1),
            self.constraint_strength.clamp(0.0, 1.0),
        )
        .expect("skeleton_kinematic_smoother: constraint pass failed");

        smoothed.lazy()
    }
}

#[derive(Debug, Clone)]
struct TrustedMask {
    label: String,
    trusted: Vec<bool>,
}

type Vec2 = [f64; 2];
type Mat2 = [[f64; 2]; 2];
type CoordinateSeries = (Vec<Option<f64>>, Vec<Option<f64>>, Vec<Option<f64>>);

fn read_coordinate_series(
    df: &DataFrame,
    cols: &KalmanSmootherColumns,
) -> PolarsResult<CoordinateSeries> {
    let x = df
        .column(&cols.x_col)?
        .f64()?
        .into_iter()
        .collect::<Vec<_>>();
    let y = df
        .column(&cols.y_col)?
        .f64()?
        .into_iter()
        .collect::<Vec<_>>();
    let likelihood = match cols
        .likelihood_col
        .as_ref()
        .and_then(|name| df.column(name).ok())
    {
        Some(series) => series.f64()?.into_iter().collect::<Vec<_>>(),
        None => vec![Some(1.0); df.height()],
    };
    Ok((x, y, likelihood))
}

fn trusted_masks(
    df: &DataFrame,
    columns: &[KalmanSmootherColumns],
    min_likelihood: f64,
) -> PolarsResult<Vec<TrustedMask>> {
    let mut masks = Vec::with_capacity(columns.len());
    for cols in columns {
        let (x_values, y_values, likelihoods) = read_coordinate_series(df, cols)?;
        let trusted = x_values
            .iter()
            .zip(y_values.iter())
            .zip(likelihoods.iter())
            .map(|((x, y), likelihood)| {
                x.is_some() && y.is_some() && likelihood.unwrap_or(0.0) >= min_likelihood
            })
            .collect();
        masks.push(TrustedMask {
            label: cols.label.clone(),
            trusted,
        });
    }
    Ok(masks)
}

fn fit_skeleton_edge_lengths(
    df: &DataFrame,
    skeleton: &Skeleton,
    columns: &[KalmanSmootherColumns],
    min_likelihood: f64,
) -> PolarsResult<std::collections::HashMap<(String, String), f64>> {
    let column_map: std::collections::HashMap<&str, &KalmanSmootherColumns> = columns
        .iter()
        .map(|cols| (cols.label.as_str(), cols))
        .collect();
    let mut baselines = std::collections::HashMap::new();

    for edge in &skeleton.edges {
        let (Some(source), Some(target)) = (
            column_map.get(edge.source.as_str()),
            column_map.get(edge.target.as_str()),
        ) else {
            continue;
        };
        let (sx, sy, sl) = read_coordinate_series(df, source)?;
        let (tx, ty, tl) = read_coordinate_series(df, target)?;
        let mut lengths = Vec::new();
        for row in 0..df.height() {
            let trusted = sl[row].unwrap_or(0.0) >= min_likelihood
                && tl[row].unwrap_or(0.0) >= min_likelihood;
            if !trusted {
                continue;
            }
            let (Some(sx), Some(sy), Some(tx), Some(ty)) = (sx[row], sy[row], tx[row], ty[row])
            else {
                continue;
            };
            lengths.push(distance((sx, sy), (tx, ty)));
        }
        if let Some(median) = median_f64(&lengths) {
            baselines.insert((edge.source.clone(), edge.target.clone()), median);
        }
    }

    Ok(baselines)
}

#[allow(clippy::too_many_arguments)]
fn apply_skeleton_constraints(
    df: &mut DataFrame,
    skeleton: &Skeleton,
    columns: &[KalmanSmootherColumns],
    trusted: &[TrustedMask],
    baselines: &std::collections::HashMap<(String, String), f64>,
    max_gap_frames: usize,
    iterations: usize,
    strength: f64,
) -> PolarsResult<()> {
    let mut series = read_all_positions(df, columns)?;
    let col_index: std::collections::HashMap<&str, usize> = columns
        .iter()
        .enumerate()
        .map(|(idx, cols)| (cols.label.as_str(), idx))
        .collect();
    let trusted_index: std::collections::HashMap<&str, &[bool]> = trusted
        .iter()
        .map(|mask| (mask.label.as_str(), mask.trusted.as_slice()))
        .collect();

    for _ in 0..iterations {
        for edge in &skeleton.edges {
            let Some(length) = baselines.get(&(edge.source.clone(), edge.target.clone())) else {
                continue;
            };
            let (Some(source_idx), Some(target_idx)) = (
                col_index.get(edge.source.as_str()).copied(),
                col_index.get(edge.target.as_str()).copied(),
            ) else {
                continue; // coverage: unreachable — baselines only hold edges whose labels are in `columns`
            };
            let (Some(source_trusted), Some(target_trusted)) = (
                trusted_index.get(edge.source.as_str()).copied(),
                trusted_index.get(edge.target.as_str()).copied(),
            ) else {
                continue; // coverage: unreachable — `trusted` masks are built from the same `columns`
            };

            for row in 0..df.height() {
                let source_short_gap = is_short_gap(source_trusted, row, max_gap_frames);
                let target_short_gap = is_short_gap(target_trusted, row, max_gap_frames);
                if source_short_gap == target_short_gap {
                    continue;
                }

                if source_short_gap {
                    project_endpoint(&mut series, source_idx, target_idx, row, *length, strength);
                } else if target_short_gap {
                    project_endpoint(&mut series, target_idx, source_idx, row, *length, strength);
                }
            }
        }
    }

    write_all_positions(df, columns, &series)
}

#[derive(Clone)]
struct PositionSeries {
    xs: Vec<Option<f64>>,
    ys: Vec<Option<f64>>,
}

fn read_all_positions(
    df: &DataFrame,
    columns: &[KalmanSmootherColumns],
) -> PolarsResult<Vec<PositionSeries>> {
    columns
        .iter()
        .map(|cols| {
            let xs = df.column(&cols.x_col)?.f64()?.into_iter().collect();
            let ys = df.column(&cols.y_col)?.f64()?.into_iter().collect();
            Ok(PositionSeries { xs, ys })
        })
        .collect()
}

fn write_all_positions(
    df: &mut DataFrame,
    columns: &[KalmanSmootherColumns],
    series: &[PositionSeries],
) -> PolarsResult<()> {
    for (cols, values) in columns.iter().zip(series.iter()) {
        df.with_column(Column::new(cols.x_col.clone().into(), values.xs.clone()))?;
        df.with_column(Column::new(cols.y_col.clone().into(), values.ys.clone()))?;
    }
    Ok(())
}

fn project_endpoint(
    series: &mut [PositionSeries],
    moving_idx: usize,
    anchor_idx: usize,
    row: usize,
    length: f64,
    strength: f64,
) {
    let (Some(mx), Some(my), Some(ax), Some(ay)) = (
        series[moving_idx].xs[row],
        series[moving_idx].ys[row],
        series[anchor_idx].xs[row],
        series[anchor_idx].ys[row],
    ) else {
        return;
    };
    let dx = mx - ax;
    let dy = my - ay;
    let current = (dx * dx + dy * dy).sqrt();
    if current <= 1e-9 || length <= 1e-9 {
        return;
    }
    let target_x = ax + dx / current * length;
    let target_y = ay + dy / current * length;
    series[moving_idx].xs[row] = Some(mx + (target_x - mx) * strength);
    series[moving_idx].ys[row] = Some(my + (target_y - my) * strength);
}

fn is_short_gap(trusted: &[bool], row: usize, max_gap_frames: usize) -> bool {
    if trusted[row] {
        return false;
    }

    let mut start = row;
    while start > 0 && !trusted[start - 1] {
        start -= 1;
    }
    let mut end = row + 1;
    while end < trusted.len() && !trusted[end] {
        end += 1;
    }

    start > 0 && end < trusted.len() && (end - start) <= max_gap_frames
}

fn distance(a: (f64, f64), b: (f64, f64)) -> f64 {
    let dx = b.0 - a.0;
    let dy = b.1 - a.1;
    (dx * dx + dy * dy).sqrt()
}

fn median_f64(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        Some((values[mid - 1] + values[mid]) / 2.0)
    } else {
        Some(values[mid])
    }
}
