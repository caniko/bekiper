/// Find the first frame where any exploration column is true (pre-tolerance).
fn find_first_exploration_frame(df: &DataFrame, raw_cols: &[String]) -> Option<usize> {
    let mut first = None;
    for col_name in raw_cols {
        if let Ok(series) = df.column(col_name)
            && let Ok(bools) = series.bool()
        {
            for (i, val) in bools.into_iter().enumerate() {
                if val == Some(true) {
                    first = Some(match first {
                        Some(prev) => std::cmp::min(prev, i),
                        None => i,
                    });
                    break;
                }
            }
        }
    }
    first
}

fn select_analysis_window(
    mode: NortWindowMode,
    total_frames: usize,
    analysis_window_seconds: f64,
    fps: f64,
    first_approach: Option<usize>,
) -> WindowSelection {
    if mode == NortWindowMode::FullAnalysis {
        return WindowSelection {
            mode: NortWindowMode::FullAnalysis,
            start_frame: 0,
            end_frame_exclusive: total_frames,
        };
    }

    if !analysis_window_seconds.is_finite() || analysis_window_seconds <= 0.0 {
        return WindowSelection {
            mode: NortWindowMode::ConfiguredFullTrial,
            start_frame: 0,
            end_frame_exclusive: total_frames,
        };
    }

    let window_frames = (analysis_window_seconds * fps) as usize;
    if window_frames == 0 || total_frames <= window_frames {
        return WindowSelection {
            mode: NortWindowMode::FullTrialShortTrial,
            start_frame: 0,
            end_frame_exclusive: total_frames,
        };
    }

    match first_approach {
        Some(start_frame) => {
            let anchored_end = start_frame.saturating_add(window_frames);
            if anchored_end <= total_frames {
                WindowSelection {
                    mode: NortWindowMode::AnchoredFirstApproach,
                    start_frame,
                    end_frame_exclusive: anchored_end,
                }
            } else {
                WindowSelection {
                    mode: NortWindowMode::BackshiftedLateApproach,
                    start_frame: total_frames.saturating_sub(window_frames),
                    end_frame_exclusive: total_frames,
                }
            }
        }
        None => WindowSelection {
            mode: NortWindowMode::LeadingNoApproach,
            start_frame: 0,
            end_frame_exclusive: window_frames,
        },
    }
}

fn summary_from_mask(
    name: String,
    mask: &BooleanChunked,
    timing: &FrameTiming,
) -> HeuristicSummary {
    let true_frames = mask.sum().unwrap_or(0) as u64;
    let total_frames = mask.len() as u64;
    let mask_values: Vec<bool> = mask
        .into_iter()
        .map(|value| value.unwrap_or(false))
        .collect();

    HeuristicSummary {
        name,
        true_frames,
        total_frames,
        seconds: timing.bool_frames_to_seconds(&mask_values),
    }
}

fn zone_toleranced_mask(
    df: &DataFrame,
    zone_col: &str,
    min_dwell_frames: u32,
) -> Option<BooleanChunked> {
    let raw_bools = df.column(zone_col).ok()?.bool().ok()?;
    Some(apply_tolerance_series(raw_bools, min_dwell_frames, 0))
}

fn toleranced_freezing_mask(df: &DataFrame, fps: f64) -> Option<BooleanChunked> {
    let raw_bools = df.column("freezing").ok()?.bool().ok()?;
    Some(apply_tolerance_series_with_density(
        raw_bools,
        fps.round() as u32,
        0,
        0.0,
    ))
}

/// When multiple objects' raw exploration columns are true on the same frame,
/// keep only the nearest object (by nose-to-boundary distance) and zero out the rest.
fn resolve_overlapping_explorations(
    df: &mut DataFrame,
    perimeters: &[Perimeter<ObjectShape>],
    raw_cols: &[String],
) {
    let n = df.height();
    let nose_x = match df.column("nose_x").and_then(|s| s.f64().cloned()) {
        Ok(v) => v,
        Err(_) => return,
    };
    let nose_y = match df.column("nose_y").and_then(|s| s.f64().cloned()) {
        Ok(v) => v,
        Err(_) => return,
    };

    let mut resolved: Vec<Vec<bool>> = raw_cols
        .iter()
        .map(|col_name| {
            df.column(col_name)
                .ok()
                .and_then(|s| s.bool().ok())
                .map(|b| b.into_iter().map(|v| v.unwrap_or(false)).collect())
                .unwrap_or_else(|| vec![false; n])
        })
        .collect();

    #[allow(clippy::needless_range_loop)]
    for frame in 0..n {
        let active: Vec<usize> = (0..perimeters.len())
            .filter(|&i| resolved[i][frame])
            .collect();

        if active.len() > 1 {
            let (nx, ny) = match (nose_x.get(frame), nose_y.get(frame)) {
                (Some(x), Some(y)) => (x, y),
                _ => continue,
            };
            // Find nearest object by nose-to-boundary distance so enclosing
            // context shapes cannot steal frames from the true target.
            let nearest = active
                .iter()
                .copied()
                .min_by(|&a, &b| {
                    let closest_a = perimeters[a].shape.closest_point(nx, ny);
                    let closest_b = perimeters[b].shape.closest_point(nx, ny);
                    let da = (nx - closest_a.0).powi(2) + (ny - closest_a.1).powi(2);
                    let db = (nx - closest_b.0).powi(2) + (ny - closest_b.1).powi(2);
                    da.total_cmp(&db)
                })
                // SAFETY: active.len() > 1 is checked above, so min_by always returns Some
                .unwrap();
            for &i in &active {
                if i != nearest {
                    resolved[i][frame] = false;
                }
            }
        }
    }

    // Write resolved columns back.
    for (i, col_name) in raw_cols.iter().enumerate() {
        let new_col = BooleanChunked::from_slice(col_name.into(), &resolved[i]);
        let _ = df.with_column(new_col.into_column());
    }
}

fn is_context_label(label: &str) -> bool {
    matches!(label, "arena" | "center_zone")
}

fn compute_toleranced_seconds(ctx: &ObjectRecognitionMetricContext<'_>, column: &str) -> f64 {
    let filters = if is_combined_exploration_raw_column(column) {
        ctx.task.bout_filter_spec(ctx.df, column) // coverage: unreachable — only qualia diagnostic columns reach this helper
    } else {
        BoutFilterSpec::unfiltered()
    };
    object_exploration_with_timing(
        ctx.df,
        column,
        ctx.clock,
        ctx.task.min_bout_seconds,
        ctx.task.max_gap_seconds,
        ctx.task.gap_density_threshold,
        filters,
    )
    .0
}

fn add_object_recognition_zone_columns(
    df: DataFrame,
    arena: Option<&Perimeter<ObjectShape>>,
    coords: &CoordinateColumns,
) -> PolarsResult<DataFrame> {
    let Some(arena) = arena else {
        return Ok(df);
    };

    let bb = arena.shape.bounding_box();
    let zones = generate_grid_zones(
        (bb.min_x + bb.max_x) / 2.0,
        (bb.min_y + bb.max_y) / 2.0,
        bb.max_x - bb.min_x,
        bb.max_y - bb.min_y,
        OBJECT_RECOGNITION_QUADRANT_ROWS,
        OBJECT_RECOGNITION_QUADRANT_COLS,
    );

    let in_arena = arena.confinement_mask(coords).fill_null(lit(false));
    let clipped: Vec<_> = zones
        .iter()
        .map(|(label, _)| {
            col(label)
                .and(in_arena.clone())
                .fill_null(lit(false))
                .alias(label)
        })
        .collect();
    add_grid_zone_columns(df.lazy(), &zones, coords)
        .with_columns(clipped)
        .collect()
}

impl ObjectRecognitionMetricUnit for ObservationTotalsUnit {
    fn compute(
        &self,
        ctx: &ObjectRecognitionMetricContext<'_>,
        metrics: &mut ObjectRecognitionMetrics,
    ) {
        let total_seconds = ctx.observation_seconds.iter().sum();
        let total_bouts = ctx.observation_bouts.iter().sum();
        metrics.observation_totals = Some(ObservationTotals {
            object_labels: ctx.object_labels.clone(),
            observation_seconds: ctx.observation_seconds.clone(),
            observation_bouts: ctx.observation_bouts.clone(),
            total_seconds,
            total_bouts,
        });
    }
}

impl ObjectRecognitionMetricUnit for PairwiseObservationUnit {
    fn compute(
        &self,
        ctx: &ObjectRecognitionMetricContext<'_>,
        metrics: &mut ObjectRecognitionMetrics,
    ) {
        let mut pairs = Vec::new();
        for lhs in 0..ctx.object_labels.len() {
            for rhs in (lhs + 1)..ctx.object_labels.len() {
                pairs.push(PairwiseObservationMetric {
                    lhs_label: ctx.object_labels[lhs].clone(),
                    rhs_label: ctx.object_labels[rhs].clone(),
                    absolute_delta_seconds: (ctx.observation_seconds[lhs]
                        - ctx.observation_seconds[rhs])
                        .abs(),
                });
            }
        }
        metrics.pairwise_observations = pairs;
    }
}

impl ObjectRecognitionMetricUnit for NoveltyIndexUnit {
    fn compute(
        &self,
        ctx: &ObjectRecognitionMetricContext<'_>,
        metrics: &mut ObjectRecognitionMetrics,
    ) {
        let role_index = |role: &str| {
            let mut matches = ctx
                .object_labels
                .iter()
                .enumerate()
                .filter(|(_, label)| label.trim().eq_ignore_ascii_case(role));
            let (index, _) = matches.next()?;
            matches.next().is_none().then_some(index)
        };
        if ctx.task.protocol.computes_novelty_indices()
            && let (Some(familiar), Some(novel)) = (role_index("familiar"), role_index("novel"))
        {
            metrics.novelty_indices = Some(NortIndices::compute(
                ctx.observation_seconds[novel],
                ctx.observation_seconds[familiar],
                ctx.observation_bouts[novel],
                ctx.observation_bouts[familiar],
            ));
        }
    }
}

#[test]
fn novelty_indices_follow_labels_not_annotation_order() {
    let task = ObjectRecognition::new(Vec::new(), 30.0);
    let df = DataFrame::default();
    for reversed in [false, true] {
        let mut labels = vec!["familiar".to_string(), "novel".to_string()];
        let mut seconds = vec![10.0, 20.0];
        let mut bouts = vec![1, 2];
        if reversed {
            labels.reverse();
            seconds.reverse();
            bouts.reverse();
        }
        let mut ctx = ObjectRecognitionMetricContext {
            task: &task,
            df: &df,
            clock: &BoutTiming::cfr(30.0, 0).unwrap(),
            object_labels: labels,
            observation_seconds: seconds,
            observation_bouts: bouts,
            arena: None,
        };
        let mut metrics = ObjectRecognitionMetrics::default();
        NoveltyIndexUnit.compute(&ctx, &mut metrics);
        let indices = metrics.novelty_indices.unwrap();
        assert_eq!(indices.familiar_seconds, 10.0);
        assert_eq!(indices.novel_seconds, 20.0);
        ctx.object_labels = vec!["familiar".into(), "familiar".into()];
        let mut metrics = ObjectRecognitionMetrics::default();
        NoveltyIndexUnit.compute(&ctx, &mut metrics);
        assert!(metrics.novelty_indices.is_none());
    }
}

impl ObjectRecognitionMetricUnit for QualiaBreakdownUnit {
    fn compute(
        &self,
        ctx: &ObjectRecognitionMetricContext<'_>,
        metrics: &mut ObjectRecognitionMetrics,
    ) {
        let mut rollups = Vec::with_capacity(ctx.object_labels.len());
        for label in &ctx.object_labels {
            let body_name = format!("body_proximity_{label}");
            let olfaction_name = format!("olfaction_{label}");
            let whisker_name = format!("whisker_interaction_{label}");

            rollups.push(QualiaDiagnosticRollup {
                label: label.clone(),
                body_center_ear_seconds: compute_toleranced_seconds(
                    ctx,
                    &diag_column_name(&body_name, "center_ear_near"),
                ),
                body_tail_base_seconds: compute_toleranced_seconds(
                    ctx,
                    &diag_column_name(&body_name, "tail_base_near"),
                ),
                body_total_seconds: compute_toleranced_seconds(ctx, &body_name),
                olfaction_nose_seconds: compute_toleranced_seconds(
                    ctx,
                    &diag_column_name(&olfaction_name, "nose_near"),
                ),
                olfaction_ray_seconds: compute_toleranced_seconds(
                    ctx,
                    &diag_column_name(&olfaction_name, "ray_hits_object"),
                ),
                olfaction_total_seconds: compute_toleranced_seconds(ctx, &olfaction_name),
                whisker_left_proximity_seconds: compute_toleranced_seconds(
                    ctx,
                    &diag_column_name(&whisker_name, "left_ear_near"),
                ),
                whisker_left_fov_seconds: compute_toleranced_seconds(
                    ctx,
                    &diag_column_name(&whisker_name, "left_ray_hits_object"),
                ),
                whisker_right_proximity_seconds: compute_toleranced_seconds(
                    ctx,
                    &diag_column_name(&whisker_name, "right_ear_near"),
                ),
                whisker_right_fov_seconds: compute_toleranced_seconds(
                    ctx,
                    &diag_column_name(&whisker_name, "right_ray_hits_object"),
                ),
                whisker_total_seconds: compute_toleranced_seconds(ctx, &whisker_name),
            });
        }
        metrics.qualia_rollups = rollups;
    }
}
