impl Heuristic for DualBodyPartZoneHeuristic {
    fn name(&self) -> &str {
        &self.zone_name
    }

    fn evaluate(&self, lf: LazyFrame) -> LazyFrame {
        let shared = self.shared.as_ref().map(|(s, arm, p)| (&s.arms[*arm], *p));
        let (head_in, tail_in) = match shared {
            Some((arm, _)) => (
                bool_lit("head_in", &arm.head_in),
                bool_lit("tail_in", &arm.tail_in),
            ),
            None => (
                self.perimeter
                    .confinement_mask(&self.head_coords)
                    .fill_null(lit(false)),
                self.perimeter
                    .confinement_mask(&self.tail_coords)
                    .fill_null(lit(false)),
            ),
        };
        let mut expr = head_in.clone().and(tail_in.clone());
        let mut columns = vec![
            head_in.alias(diag_column_name(self.name(), "head_in_zone")),
            tail_in.alias(diag_column_name(self.name(), "tail_in_zone")),
        ];

        if let Some(threshold) = self.threshold {
            let (head_past, tail_past) = match shared {
                Some((arm, p)) => (
                    bool_lit("head_past", &arm.past_values(true, &threshold, p)),
                    bool_lit("tail_past", &arm.past_values(false, &threshold, p)),
                ),
                None => (
                    threshold.past(&self.head_coords).fill_null(lit(false)),
                    threshold.past(&self.tail_coords).fill_null(lit(false)),
                ),
            };

            expr = expr.and(head_past.clone()).and(tail_past.clone());
            columns.push(head_past.alias(diag_column_name(self.name(), "head_past_threshold")));
            columns.push(tail_past.alias(diag_column_name(self.name(), "tail_past_threshold")));
        }

        // The final arm-membership latch, when enabled, is applied after the
        // lazy frame is collected so these raw predicate columns remain
        // frame-accurate for inspection. `self.name()` therefore still starts
        // from the legacy per-frame mask here and may be rewritten later.
        let _ = self.arm_membership_hysteresis;
        columns.push(expr.fill_null(lit(false)).alias(self.name()));
        lf.with_columns(columns)
    }

    fn decomposition(&self) -> Option<HeuristicDecompositionSpec> {
        let mut components = vec![
            decomposition_component(
                "head_in_zone",
                diag_column_name(self.name(), "head_in_zone"),
                vec![coordinate_label_from_xy(&self.head_coords.x)],
                Some(self.perimeter.label.clone()),
            ),
            decomposition_component(
                "tail_in_zone",
                diag_column_name(self.name(), "tail_in_zone"),
                vec![coordinate_label_from_xy(&self.tail_coords.x)],
                Some(self.perimeter.label.clone()),
            ),
        ];
        if self.threshold.is_some() {
            components.push(decomposition_component(
                "head_past_threshold",
                diag_column_name(self.name(), "head_past_threshold"),
                vec![coordinate_label_from_xy(&self.head_coords.x)],
                None,
            ));
            components.push(decomposition_component(
                "tail_past_threshold",
                diag_column_name(self.name(), "tail_past_threshold"),
                vec![coordinate_label_from_xy(&self.tail_coords.x)],
                None,
            ));
        }
        for component in &mut components {
            if component.perimeter_label.is_some() {
                *component = component
                    .clone()
                    .with_geometry(&self.perimeter.shape, &self.perimeter.label);
            }
        }
        Some(HeuristicDecompositionSpec {
            heuristic_name: self.zone_name.clone(),
            result_column: self.zone_name.clone(),
            coordinate_labels: vec![coordinate_label_from_xy(&self.head_coords.x)],
            perimeter_label: Some(self.perimeter.label.clone()),
            components,
            show_final_result: true,
        })
    }
}

/// Heuristic for the Y-maze center zone.
///
/// Center = center polygon OR the proximal region of any arm
/// (head within arm polygon but closer to center than the proximal threshold).
struct YMazeCenterHeuristic {
    center: Perimeter<Polygon>,
    arm_polygons: Vec<Polygon>,
    head_coords: CoordinateColumns,
    /// Per-arm threshold predicate (the one feeding `in_arm_X`, `zone_sequence`,
    /// `arm_sequence` and spontaneous alternation). Head positions inside an
    /// arm polygon but before this line remain in the center zone.
    proximal: Vec<ArmThreshold>,
    /// Per-recording geometry of a variant group and the proximal fraction.
    shared: Option<std::sync::Arc<YMazeShared>>,
    p: f64,
}

impl Heuristic for YMazeCenterHeuristic {
    fn name(&self) -> &str {
        "in_center"
    }

    fn evaluate(&self, lf: LazyFrame) -> LazyFrame {
        // Start with head inside center polygon
        let center_polygon = match &self.shared {
            Some(shared) => bool_lit("center_polygon", &shared.center_polygon),
            None => self
                .center
                .confinement_mask(&self.head_coords)
                .fill_null(lit(false)),
        };
        let mut expr = center_polygon.clone();
        let mut columns =
            vec![center_polygon.alias(diag_column_name(self.name(), "center_polygon"))];

        // Add proximal arm regions: head inside arm polygon and still closer
        // than `nearest + proximal_threshold * (farthest - nearest)`.
        for (idx, (arm, threshold)) in self.arm_polygons.iter().zip(&self.proximal).enumerate() {
            let in_proximal = match &self.shared {
                Some(shared) => {
                    let arm = &shared.arms[idx];
                    let before = arm.before_values(threshold, self.p);
                    let values: Vec<bool> = arm
                        .head_in
                        .iter()
                        .zip(before)
                        .map(|(inside, before)| *inside && before)
                        .collect();
                    bool_lit("in_proximal", &values)
                }
                None => {
                    let arm_perimeter = Perimeter::new("", arm.clone());
                    let in_arm = arm_perimeter.confinement_mask(&self.head_coords);
                    in_arm
                        .and(threshold.before(&self.head_coords))
                        .fill_null(lit(false))
                }
            };
            expr = expr.or(in_proximal.clone());
            let arm_label = ARM_LABELS.get(idx).copied().unwrap_or("arm");
            columns.push(in_proximal.alias(diag_column_name(
                self.name(),
                &format!("proximal_arm_{arm_label}"),
            )));
        }

        columns.push(expr.fill_null(lit(false)).alias(self.name()));
        lf.with_columns(columns)
    }

    fn decomposition(&self) -> Option<HeuristicDecompositionSpec> {
        let mut components = vec![
            decomposition_component(
                "center_polygon",
                diag_column_name(self.name(), "center_polygon"),
                vec![coordinate_label_from_xy(&self.head_coords.x)],
                Some(self.center.label.clone()),
            )
            .with_geometry(&self.center.shape, &self.center.label),
        ];
        for idx in 0..self.arm_polygons.len() {
            let arm_label = ARM_LABELS.get(idx).copied().unwrap_or("arm");
            components.push(
                decomposition_component(
                    format!("proximal_arm_{arm_label}"),
                    diag_column_name(self.name(), &format!("proximal_arm_{arm_label}")),
                    vec![coordinate_label_from_xy(&self.head_coords.x)],
                    Some(format!("arm_{arm_label}")),
                )
                .with_geometry(&self.arm_polygons[idx], &format!("arm_{arm_label}")),
            );
        }
        Some(HeuristicDecompositionSpec {
            heuristic_name: self.name().to_string(),
            result_column: self.name().to_string(),
            coordinate_labels: vec![coordinate_label_from_xy(&self.head_coords.x)],
            perimeter_label: None,
            components,
            show_final_result: true,
        })
    }
}

/// Return the first frame where every given landmark has both coordinates
/// detected (non-null); 0 when a column is missing or no frame qualifies.
fn first_detection_start_index(df: &DataFrame, landmarks: &[&CoordinateColumns]) -> usize {
    let columns: Option<Vec<_>> = landmarks
        .iter()
        .flat_map(|c| [&c.x, &c.y])
        .map(|name| df.column(name).ok()?.f64().ok().cloned())
        .collect();
    columns
        .and_then(|columns| (0..df.height()).find(|&i| columns.iter().all(|c| c.get(i).is_some())))
        .unwrap_or(0)
}

/// Trim a DataFrame by an already-computed start index.
fn trim_to_start_index(df: &DataFrame, start: usize) -> DataFrame {
    match start {
        0 => df.clone(),
        _ => df.slice(start as i64, df.height() - start),
    }
}

fn zone_truth_values(df: &DataFrame, col_name: &str) -> Vec<bool> {
    df.column(col_name)
        .ok()
        .and_then(|s| s.bool().ok())
        .map(|b| b.into_iter().map(|v| v.unwrap_or(false)).collect())
        .unwrap_or_default()
}

fn coordinate_present_values(df: &DataFrame, coords: &CoordinateColumns) -> Vec<bool> {
    let xs: Vec<bool> = df
        .column(&coords.x)
        .ok()
        .and_then(|s| s.f64().ok())
        .map(|x| x.into_iter().map(|v| v.is_some()).collect())
        .unwrap_or_default();
    let ys: Vec<bool> = df
        .column(&coords.y)
        .ok()
        .and_then(|s| s.f64().ok())
        .map(|y| y.into_iter().map(|v| v.is_some()).collect())
        .unwrap_or_default();

    xs.into_iter().zip(ys).map(|(x, y)| x && y).collect()
}

fn latched_arm_membership(
    head_in: &[bool],
    tail_in: &[bool],
    head_past: &[bool],
    tail_past: &[bool],
    pose_present: &[bool],
) -> Vec<bool> {
    debug_assert_eq!(head_in.len(), tail_in.len());
    debug_assert_eq!(head_in.len(), head_past.len());
    debug_assert_eq!(head_in.len(), tail_past.len());
    debug_assert_eq!(head_in.len(), pose_present.len());

    let mut latched = false;
    let mut values = Vec::with_capacity(head_in.len());

    for idx in 0..head_in.len() {
        if !pose_present[idx] {
            values.push(latched);
            continue;
        }

        let both_in_polygon = head_in[idx] && tail_in[idx];
        if !both_in_polygon {
            latched = false;
            values.push(false);
            continue;
        }

        let both_past_threshold = head_past[idx] && tail_past[idx];
        let both_back_behind_threshold = !head_past[idx] && !tail_past[idx];
        if both_past_threshold {
            latched = true;
        } else if both_back_behind_threshold {
            latched = false;
        }
        values.push(latched);
    }

    values
}

/// Full Y-maze analysis result with zone summaries and all spec measurements.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct YMazeResult {
    /// Evidence generated with this result. Absent on historical results;
    /// absence must never be interpreted as a verified scoring definition.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scoring_provenance: Option<serde_json::Value>,
    /// Alternation metrics (arm sequence, correct/incorrect counts, etc.).
    pub analysis: YMazeAnalysis,
    /// Per-zone summaries (arms + center).
    #[serde(default)]
    pub zone_summaries: Vec<ZoneSummary>,
    /// Per-area motion summaries (arms + center).
    #[serde(default)]
    pub area_motion_summaries: Vec<ZoneMotionSummary>,
    /// Sum of time spent across arms only.
    #[serde(default)]
    pub seconds_in_arms: f64,
    /// Sum of time spent across arms and center.
    #[serde(default)]
    pub seconds_in_arms_and_center: f64,
    /// Total freezing duration (seconds).
    pub freezing_seconds: f64,
    /// Total path distance across all zones (meters).
    pub total_distance_meters: f64,
    /// Mean locomotion speed (m/s).
    pub total_mean_speed: f64,
    /// Total raw trial duration before any Y-maze start-window trim (seconds).
    #[serde(default)]
    pub total_duration_seconds: f64,
    /// Duration after applying the configured Y-maze analysis start window (seconds).
    #[serde(default)]
    pub analysis_duration_seconds: f64,
    /// Fraction of each arm polygon's span used in
    /// `nearest + threshold * (farthest - nearest)` for the arm-entry line.
    #[serde(default)]
    pub proximal_threshold: f64,
    /// Configured Y-maze analysis start mode.
    #[serde(default)]
    pub start_mode: YMazeStartMode,
    /// Maximum contiguous `"none"` run length bridged during zone-sequence
    /// extraction, in seconds.
    #[serde(default)]
    pub zone_max_gap_seconds: f64,
    /// Forward-fill TTL applied to head/tail coordinate null runs before
    /// Y-maze zone heuristics, in seconds.
    #[serde(default)]
    pub pose_gap_fill_seconds: f64,
    /// Whether `in_arm_X` exit hysteresis was enabled for this analysis.
    #[serde(default = "default_arm_membership_hysteresis")]
    pub arm_membership_hysteresis: bool,
    /// The evaluated DataFrame; present after `full_analyze`, skipped in serde.
    #[serde(skip)]
    pub evaluated_df: Option<DataFrame>,
}

fn default_arm_membership_hysteresis() -> bool {
    true
}
