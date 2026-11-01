impl BehaviouralTask for OpenFieldTest {
    fn task_type(&self) -> TaskType {
        TaskType::OpenField
    }

    fn effective_parameters(&self) -> Option<serde_json::Value> {
        let mut value = self.params.effective_parameters();
        value["rearing"] = serde_json::json!({
            "max_speed": self.rearing_max_speed,
            "max_body_length_ratio": self.rearing_max_body_length_ratio,
            "max_body_length": self.rearing_max_body_length,
            "forepaw_proximity": self.rearing_forepaw_proximity,
        });
        value["grooming"] = serde_json::json!({
            "max_speed": self.grooming_max_speed,
            "max_body_length": self.grooming_max_body_length,
            "min_body_length": self.grooming_min_body_length,
            "forepaw_proximity": self.grooming_forepaw_proximity,
        });
        Some(value)
    }

    fn heuristics(&self) -> Vec<Box<dyn Heuristic>> {
        vec![
            Box::new(ZoneConfinementHeuristic {
                zone_name: "in_center".into(),
                perimeter: self.center.clone(),
                coords: self.params.centroid_coords.clone(),
            }),
            Box::new(PeripheryHeuristic {
                arena: self.arena.clone(),
                center: self.center.clone(),
                coords: self.params.centroid_coords.clone(),
            }),
            Box::new(FreezingHeuristic::new(
                &self.params.centroid_coords.x,
                &self.params.centroid_coords.y,
                self.params.freezing_window_seconds,
                self.params.freezing_displacement_threshold,
                self.params.fps,
            )),
            Box::new({
                let mut r = RearingHeuristic::new(self.params.fps);
                r.nose_x = self.params.nose_coords.x.clone();
                r.nose_y = self.params.nose_coords.y.clone();
                r.tail_base_x = self.params.tail_coords.x.clone();
                r.tail_base_y = self.params.tail_coords.y.clone();
                r.centroid_x = self.params.centroid_coords.x.clone();
                r.centroid_y = self.params.centroid_coords.y.clone();
                r.max_speed = bekiper_feature::heuristic::solo::speed_per_frame(
                    self.rearing_max_speed,
                    self.params.fps,
                );
                r.max_body_length_ratio = self.rearing_max_body_length_ratio;
                r.max_body_length = self.rearing_max_body_length;
                r
            }),
            Box::new({
                let mut g = GroomingHeuristic::new(self.params.fps);
                g.nose_x = self.params.nose_coords.x.clone();
                g.nose_y = self.params.nose_coords.y.clone();
                g.tail_base_x = self.params.tail_coords.x.clone();
                g.tail_base_y = self.params.tail_coords.y.clone();
                g.centroid_x = self.params.centroid_coords.x.clone();
                g.centroid_y = self.params.centroid_coords.y.clone();
                g.max_speed = bekiper_feature::heuristic::solo::speed_per_frame(
                    self.grooming_max_speed,
                    self.params.fps,
                );
                g.max_body_length = self.grooming_max_body_length;
                g.min_body_length = self.grooming_min_body_length;
                g
            }),
        ]
    }

    fn quadrants(&self) -> Vec<QuadrantSpec> {
        let rect = &self.arena.shape;
        let half_w = rect.width / 2.0;
        let half_h = rect.height / 2.0;
        vec![QuadrantSpec {
            label: "open_field_grid".to_string(),
            x_min: rect.center_x - half_w,
            y_min: rect.center_y - half_h,
            x_max: rect.center_x + half_w,
            y_max: rect.center_y + half_h,
            rows: OPEN_FIELD_QUADRANT_ROWS,
            cols: OPEN_FIELD_QUADRANT_COLS,
        }]
    }

    fn analyze_detailed(
        &self,
        lf: LazyFrame,
        timing: &bekiper_core::video::FrameTiming,
    ) -> (crate::AnalysisResult, Option<TaskDetail>) {
        let heuristics = self.heuristics();
        let decompositions = crate::decompositions_from_heuristics(&heuristics);
        let result = self.full_analyze_with_timing(lf, timing);
        self.detailed_from_result(decompositions, result, timing)
    }
}

/// Full OFT analysis result.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OftResult {
    /// Actual support and requested window; incomplete recordings remain QC flagged.
    #[serde(default)]
    pub window: OftWindow,
    /// Zone summary for the center region.
    pub center_summary: ZoneSummary,
    /// Zone summary for the peripheral region.
    pub periphery_summary: ZoneSummary,
    /// Quadrant occupancy/motion summaries for the 4x4 arena grid.
    #[serde(default)]
    pub quadrant_summaries: Vec<ZoneSummary>,
    /// Quadrant motion summaries for the 4x4 arena grid.
    #[serde(default)]
    pub quadrant_motion_summaries: Vec<ZoneMotionSummary>,
    /// Motion summary for the center region.
    #[serde(default)]
    pub center_motion_summary: ZoneMotionSummary,
    /// Motion summary for the peripheral region.
    #[serde(default)]
    pub periphery_motion_summary: ZoneMotionSummary,
    /// Contiguous occupancy islands in the center zone.
    #[serde(default)]
    pub center_motion_islands: Vec<MotionIslandSummary>,
    /// Contiguous occupancy islands in the periphery zone.
    #[serde(default)]
    pub periphery_motion_islands: Vec<MotionIslandSummary>,
    /// Percentage of trial time spent in the center.
    pub center_time_pct: f64,
    /// Percentage of trial time spent in the periphery.
    pub periphery_time_pct: f64,
    /// Freezing (seconds): immobility ≥1.0 s, excluding grooming and rearing.
    pub freezing_seconds: f64,
    /// Immobility (seconds): every stationary bout ≥1.0 s.
    #[serde(default)]
    pub immobility_seconds: f64,
    /// Immobility as percentage of total trial time.
    pub immobility_pct: f64,
    /// Rearing duration (forepaws off floor ≥0.5s).
    pub rearing_seconds: f64,
    /// Grooming duration (≥1.0s).
    pub grooming_seconds: f64,
    /// Total path distance (meters).
    pub total_distance_meters: f64,
    /// Mean locomotion speed (m/s).
    pub total_mean_speed: f64,
    /// Total trial duration (seconds).
    pub total_duration_seconds: f64,
    /// The evaluated DataFrame; present after `full_analyze`, skipped in serde.
    #[serde(skip)]
    pub evaluated_df: Option<DataFrame>,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct OftWindow {
    pub source_start_frame: usize,
    pub source_end_frame: usize,
    pub start_seconds: f64,
    pub end_seconds: f64,
    pub requested_seconds: f64,
    pub complete: bool,
    pub policy: String,
}

/// Contiguous occupancy island for a zone.
///
/// The interval is stored as half-open frame bounds: `start_frame..end_frame`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MotionIslandSummary {
    /// First frame in the occupied island.
    pub start_frame: usize,
    /// One-past-the-last frame in the occupied island.
    pub end_frame: usize,
    /// Start time in seconds.
    pub start_seconds: f64,
    /// Island duration in seconds.
    pub duration_seconds: f64,
    /// Path distance covered during the island.
    pub distance_meters: f64,
    /// Mean speed during the island.
    pub mean_speed_mps: f64,
}

fn motion_islands_from_mask(
    df: &DataFrame,
    mask: &BooleanChunked,
    timing: &BoutTiming,
    x_col: &str,
    y_col: &str,
) -> Vec<MotionIslandSummary> {
    let mut islands = Vec::new();
    let mut start: Option<usize> = None;

    for idx in 0..mask.len() {
        match (mask.get(idx).unwrap_or(false), start) {
            (true, None) => start = Some(idx),
            (false, Some(start_idx)) => {
                islands.push(build_motion_island_summary(
                    df,
                    timing,
                    start_idx..idx,
                    x_col,
                    y_col,
                ));
                start = None;
            }
            _ => {}
        }
    }

    if let Some(start_idx) = start {
        islands.push(build_motion_island_summary(
            df,
            timing,
            start_idx..mask.len(),
            x_col,
            y_col,
        ));
    }

    islands
}

fn build_motion_island_summary(
    df: &DataFrame,
    timing: &BoutTiming,
    frames: std::ops::Range<usize>,
    x_col: &str,
    y_col: &str,
) -> MotionIslandSummary {
    let (start_frame, end_frame) = (frames.start, frames.end);
    let duration_seconds = timing.duration(start_frame, end_frame);
    let start_seconds = timing.boundary(start_frame);
    let distance_meters = path_distance(
        &df.slice(start_frame as i64, frames.len()),
        x_col,
        y_col,
        None,
    );
    let mean_speed_mps = if duration_seconds > 0.0 {
        distance_meters / duration_seconds
    } else {
        0.0 // coverage: unreachable — islands span at least one positive-duration sample
    };

    MotionIslandSummary {
        start_frame: source_frame_boundary(df, start_frame),
        end_frame: source_frame_boundary(df, end_frame),
        start_seconds,
        duration_seconds,
        distance_meters,
        mean_speed_mps,
    }
}

/// Frame columns survive window selection. The final half-open boundary is one
/// past the last source key, never a selected-row ordinal.
fn source_frame_boundary(df: &DataFrame, index: usize) -> usize {
    if df.height() == 0 {
        return 0;
    }
    let row = index.min(df.height() - 1);
    let keys = df
        .column("frame")
        .expect("OFT retains original frame keys")
        .slice(row as i64, 1)
        .cast(&DataType::UInt64)
        .expect("invalid frame keys");
    let key = keys
        .u64()
        .unwrap()
        .get(0)
        .expect("OFT source frame keys must be non-null");
    let boundary = if index < df.height() {
        key
    } else {
        key.checked_add(1).expect("source frame overflow")
    };
    usize::try_from(boundary).expect("source frame overflow")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::too_many_arguments)]
    fn test_df(
        centroid_x: &[f64],
        centroid_y: &[f64],
        head_x: &[f64],
        head_y: &[f64],
        nose_x: &[f64],
        nose_y: &[f64],
        tail_x: &[f64],
        tail_y: &[f64],
    ) -> DataFrame {
        let cols = vec![
            Float64Chunked::from_slice("mouse_center_x".into(), centroid_x).into_column(),
            Float64Chunked::from_slice("mouse_center_y".into(), centroid_y).into_column(),
            Float64Chunked::from_slice("center_ear_x".into(), head_x).into_column(),
            Float64Chunked::from_slice("center_ear_y".into(), head_y).into_column(),
            Float64Chunked::from_slice("nose_x".into(), nose_x).into_column(),
            Float64Chunked::from_slice("nose_y".into(), nose_y).into_column(),
            Float64Chunked::from_slice("tail_base_x".into(), tail_x).into_column(),
            Float64Chunked::from_slice("tail_base_y".into(), tail_y).into_column(),
        ];
        DataFrame::new(centroid_x.len(), cols).unwrap()
    }

    #[test]
    fn full_analyze_adds_quadrant_and_island_summaries() {
        let task = OpenFieldTest::standard((0.0, 0.0), 2.0, 0.5, 1.0);
        let df = test_df(
            &[0.1, 0.2, 0.7, 0.8, 0.1, 0.7],
            &[0.1, 0.2, 0.1, 0.1, 0.1, 0.7],
            &[0.1, 0.2, 0.7, 0.8, 0.1, 0.7],
            &[0.1, 0.2, 0.1, 0.1, 0.1, 0.7],
            &[0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
            &[0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
            &[-0.8, -0.8, -0.8, -0.8, -0.8, -0.8],
            &[-0.8, -0.8, -0.8, -0.8, -0.8, -0.8],
        );

        let result = task.full_analyze(df.lazy());

        assert_eq!(result.quadrant_summaries.len(), 16);
        assert_eq!(result.quadrant_motion_summaries.len(), 16);
        assert!(
            result
                .quadrant_summaries
                .iter()
                .any(|summary| summary.zone_name == "q_2_2" && summary.duration_seconds > 0.0)
        );
        assert!(result.center_motion_summary.total_displacement_meters > 0.0);
        assert!(result.periphery_motion_summary.total_displacement_meters > 0.0);
        assert_eq!(result.center_motion_islands.len(), 2);
        assert_eq!(result.periphery_motion_islands.len(), 2);
        assert_eq!(result.center_motion_islands[0].start_frame, 0);
        assert_eq!(result.center_motion_islands[0].end_frame, 2);
        assert!((result.center_motion_islands[0].duration_seconds - 2.0).abs() < 1e-10);
        assert_eq!(result.periphery_motion_islands[0].start_frame, 2);
        assert_eq!(result.periphery_motion_islands[0].end_frame, 4);
    }

    #[test]
    fn analyze_detailed_returns_typed_quadrant_data() {
        let task = OpenFieldTest::standard((0.0, 0.0), 2.0, 0.5, 1.0);
        let df = test_df(
            &[0.1, 0.2, 0.7, 0.8, 0.1, 0.7],
            &[0.1, 0.2, 0.1, 0.1, 0.1, 0.7],
            &[0.1, 0.2, 0.7, 0.8, 0.1, 0.7],
            &[0.1, 0.2, 0.1, 0.1, 0.1, 0.7],
            &[0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
            &[0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
            &[-0.8, -0.8, -0.8, -0.8, -0.8, -0.8],
            &[-0.8, -0.8, -0.8, -0.8, -0.8, -0.8],
        );
        let timing = FrameTiming::cfr(1.0);

        let (analysis, detail) = task.analyze_detailed(df.lazy(), &timing);
        assert_eq!(analysis.task_type, TaskType::OpenField);
        assert!(matches!(detail, Some(TaskDetail::OpenField(_))));
        assert!(
            analysis
                .summaries
                .iter()
                .any(|summary| summary.name == "in_center")
        );
    }
}
