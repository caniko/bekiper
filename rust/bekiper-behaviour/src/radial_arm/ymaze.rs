use polars::prelude::*;

use bekiper_core::inspection::HeuristicDecompositionSpec;
use bekiper_core::types::{CoordinateColumns, TaskType};
use bekiper_feature::heuristic::solo::freezing::FreezingHeuristic;
use bekiper_feature::heuristic::{
    Heuristic, coordinate_label_from_xy, decomposition_component, diag_column_name,
};
use bekiper_feature::tolerance::{BoutTiming, TemporalBoutOps, TemporalPolicy};
use bekiper_feature::zone::{ZoneOccupancy, ZoneSummary, zone_summary_from_mask};
use bekiper_math::geometry::path_distance;
use bekiper_perimeter::perimeter::Perimeter;
use bekiper_perimeter::polygon::Polygon;
use bekiper_perimeter::radial_maze::{ArmCrossline, RadialMaze};

use crate::BehaviouralTask;
use crate::TaskDetail;
use crate::motion_detail::{ZoneMotionSummary, motion_summary_from_mask};
use crate::params::TaskParams;
use crate::ymaze::{
    YMazeAnalysis, analyze_alternations, arm_preference, extract_zone_sequence,
    permutation_alternation_distribution, reduce_to_arm_sequence, spontaneous_alternation_pct,
    triad_distributions,
};

/// Fixed arm labels for the Y-maze protocol. Hardcoded by design:
/// the Y-maze schema has exactly three arms labelled A/B/C.
const ARM_LABELS: [&str; 3] = ["A", "B", "C"];

// ── Y-maze zone-threshold defaults ────────────────────────────────────────
// Both values are overridable via `TaskParamsOverrides::ymaze` in
// `bekiper-ingress/src/config.rs`.

/// Fraction of an arm polygon's radial span that defines the proximal boundary.
///
/// The span is measured from the center polygon centroid to the arm polygon's
/// nearest and farthest vertices, then evaluated as
/// `nearest + threshold * (farthest - nearest)`. For an `in_arm_X = true`
/// frame, both head and tail must be inside the arm polygon and at or beyond
/// that threshold line. Default 0.25 means 25% of the arm polygon's span past
/// its nearest vertex, not 25% of the centroid-to-tip distance.
pub const DEFAULT_PROXIMAL_THRESHOLD: f64 = 0.25;
/// Registered principal fractions (25/45/65%); the single owner of this set.
/// With the archived radial geometry they select the dual-landmark entry rule
/// and uninterrupted dwell. Override per task with `ymaze.principal_thresholds`.
pub const DEFAULT_PRINCIPAL_THRESHOLDS: [f64; 3] = [0.25, 0.45, 0.65];
pub const DEFAULT_ZONE_MAX_GAP_SECONDS: f64 = 0.1;
pub const DEFAULT_POSE_GAP_FILL_SECONDS: f64 = 0.2;

/// A fixed geometry identity, not a choice between competing conventions.
/// Configs may assert it explicitly; unknown conventions are rejected by serde.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum YMazeThresholdGeometry {
    #[default]
    #[serde(rename = "imen_radial_vertex_span_v1")]
    ImenRadialVertexSpanV1,
    /// Owner decision 2026-10-28: p% is a cross-line between the arm's start
    /// edge (0%) and its annotated distal line (100%). Dwell, gap, fill and
    /// hysteresis come only from the configured task params.
    #[serde(rename = "imen_arm_start_crossline_v1")]
    ArmStartCrosslineV1,
}

/// Per-arm threshold predicate shared by `in_arm_X` and the centre zone.
#[derive(Debug, Clone, Copy)]
pub(crate) enum ArmThreshold {
    /// Squared distance from the centre centroid at or beyond `min_distance_sq`.
    Radial {
        center: (f64, f64),
        min_distance_sq: f64,
    },
    /// Distal side of the arm-start cross-line (see `RadialMaze::arm_frame`).
    Crossline(ArmCrossline),
}

impl ArmThreshold {
    fn past(&self, c: &CoordinateColumns) -> Expr {
        match *self {
            Self::Radial {
                center: (cx, cy),
                min_distance_sq,
            } => {
                let dx = col(&c.x) - lit(cx);
                let dy = col(&c.y) - lit(cy);
                (dx.clone() * dx + dy.clone() * dy).gt_eq(lit(min_distance_sq))
            }
            Self::Crossline(line) if line.p <= 0.0 => lit(true),
            // Same arithmetic as `ArmCrossline::signed_side`.
            Self::Crossline(line) => ((col(&c.x) - lit(line.p1.0)) * lit(line.normal.0)
                + (col(&c.y) - lit(line.p1.1)) * lit(line.normal.1))
            .gt_eq(lit(0.0)),
        }
    }

    /// Scalar mirror of [`Self::past`] with polars' float comparison semantics
    /// (NaN is the greatest value and equals itself).
    fn past_scalar(&self, x: f64, y: f64) -> bool {
        match *self {
            Self::Radial {
                center: (cx, cy),
                min_distance_sq,
            } => {
                let (dx, dy) = (x - cx, y - cy);
                polars_ge(dx * dx + dy * dy, min_distance_sq)
            }
            Self::Crossline(line) if line.p <= 0.0 => true,
            Self::Crossline(line) => polars_ge(
                (x - line.p1.0) * line.normal.0 + (y - line.p1.1) * line.normal.1,
                0.0,
            ),
        }
    }

    /// Scalar mirror of [`Self::before`].
    fn before_scalar(&self, x: f64, y: f64) -> bool {
        match *self {
            Self::Radial {
                center: (cx, cy),
                min_distance_sq,
            } => {
                let (dx, dy) = (x - cx, y - cy);
                let v = dx * dx + dy * dy;
                !v.is_nan() && (min_distance_sq.is_nan() || v < min_distance_sq)
            }
            Self::Crossline(_) => !self.past_scalar(x, y),
        }
    }

    /// Complement used by the centre zone's proximal arm region.
    fn before(&self, c: &CoordinateColumns) -> Expr {
        match *self {
            Self::Radial {
                center: (cx, cy),
                min_distance_sq,
            } => {
                let dx = col(&c.x) - lit(cx);
                let dy = col(&c.y) - lit(cy);
                (dx.clone() * dx + dy.clone() * dy).lt(lit(min_distance_sq))
            }
            Self::Crossline(_) => self.past(c).not(),
        }
    }
}

/// Choice of where the Y-maze recording window starts.
///
/// The criteria spec phrases this as "DO not remove 10s from video → only
/// start recording from when the animal is detected" — readers have
/// historically split on whether this means "keep the full video" or
/// "trim the leading no-animal frames". This enum makes the decision
/// explicit per project.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum YMazeStartMode {
    /// Begin analysis at frame 0 of the video; do not trim.
    VideoStart,
    /// Skip leading frames until the animal's head keypoint is first
    /// detected (non-null pose).
    #[default]
    FirstDetection,
}

/// Y-maze / radial arm maze task.
///
/// Tracks which arms the animal visits and computes exploration patterns.
/// The clarified principal 25/45/65% analyses count the false-to-true
/// transition of the head-center AND tail-base predicate after at least one
/// second of uninterrupted membership by default. `temporal_policy` overrides it.
/// Other thresholds retain the explicitly configured historical/comparison policy.
///
/// Zone thresholds (expressed as fractions of each arm polygon's radial span):
/// - `proximal_threshold` (default 0.25): `nearest + threshold * (farthest - nearest)`;
///   `in_arm_X = true` requires both head and tail inside the arm polygon and
///   past this line
/// - The center zone includes the center polygon plus the proximal region
///   (0% – `proximal_threshold`) of each arm.
pub struct YMaze {
    /// Explicit seconds-based policy. None uses the inclusive one-second default
    /// for principal thresholds and legacy fields for historical comparisons.
    pub temporal_policy: Option<TemporalPolicy>,
    /// Threshold geometry; default keeps archived radial scoring.
    pub threshold_geometry: YMazeThresholdGeometry,
    pub params: TaskParams,
    pub maze: Perimeter<RadialMaze>,
    /// Fraction of an arm polygon's radial span that defines the arm-entry line.
    ///
    /// For each arm, the line is placed at
    /// `nearest_vertex_dist + proximal_threshold * (farthest_vertex_dist - nearest_vertex_dist)`,
    /// where distances are measured from the center centroid to the arm
    /// polygon's nearest and farthest vertices. An `in_arm_X = true` frame
    /// requires both head and tail to be inside the arm polygon and at or
    /// beyond that line. This threshold participates in `zone_sequence`,
    /// `arm_sequence`, and spontaneous-alternation analysis.
    pub proximal_threshold: f64,
    /// Fractions treated as principal analyses; default [`DEFAULT_PRINCIPAL_THRESHOLDS`].
    pub principal_thresholds: Vec<f64>,
    /// Where the recording window begins. Default: `FirstDetection`.
    pub start_mode: YMazeStartMode,
    /// Whether the ≥`min_dwell_seconds` filter is applied to the arm/center
    /// zone masks before extracting the spontaneous-alternation arm sequence.
    ///
    /// The arm-threshold criterion (head and tail both inside the arm polygon
    /// and past the proximal threshold line) is itself a strict gate, and the
    /// extra dwell filter can remove genuine brief arm entries. Projects that
    /// trust the arm-threshold crossing as the entry definition can set this
    /// to `false` to score SA from the fully mitigated zone sequence without
    /// the extra dwell gate; the parallel `raw_arm_sequence` still exposes the
    /// pre-mitigation crossings for analyst comparison. This historical option
    /// is ignored for principal 25/45/65% analyses unless an explicit
    /// `temporal_policy` is supplied (that policy takes precedence).
    pub apply_dwell_filter: bool,
    /// When true, `in_arm_X` latches once head and tail are both inside the
    /// arm polygon and past the proximal threshold, and only releases once
    /// both are back behind the threshold or either body part leaves the
    /// polygon. Default: `true`.
    pub arm_membership_hysteresis: bool,
    /// Maximum length of a contiguous run of `"none"` frames that should be
    /// treated as part of the previous bout instead of resetting the
    /// zone-sequence state. Default 0.1 s ≈ 3 frames @ 30 fps. Set to 0.0 to
    /// reproduce the legacy reset-on-every-none behaviour.
    pub zone_max_gap_seconds: f64,
    /// Forward-fill short runs of null in head/tail coordinate columns by
    /// this many seconds before zone heuristics run. Filters brief
    /// likelihood-drop frames out of the Y-maze zone input without papering
    /// over genuine tracking loss. Default 0.2 s ≈ 6 frames @ 30 fps. Set to
    /// 0.0 to disable.
    pub pose_gap_fill_seconds: f64,
    /// Frames counted by zone outputs; entries and SA keep the dwell.
    pub zone_occupancy: ZoneOccupancy,
}

impl YMaze {
    /// Principal analyses use the clarified dual-landmark spatial predicate.
    /// No fuzzy threshold matching: the registered fractions are exact values.
    /// Only the archived radial geometry forces the principal policy; the
    /// cross-line geometry takes dwell/gap/fill/hysteresis from configuration.
    pub fn uses_principal_entry_rule(&self) -> bool {
        self.threshold_geometry == YMazeThresholdGeometry::ImenRadialVertexSpanV1
            && self.principal_thresholds.contains(&self.proximal_threshold)
    }

    /// Create a Y-maze task with the given maze perimeter and video FPS,
    /// using the default proximal threshold.
    pub fn new(maze: Perimeter<RadialMaze>, fps: f64) -> Self {
        Self {
            temporal_policy: None,
            threshold_geometry: YMazeThresholdGeometry::default(),
            params: TaskParams::new(fps),
            maze,
            proximal_threshold: DEFAULT_PROXIMAL_THRESHOLD,
            principal_thresholds: DEFAULT_PRINCIPAL_THRESHOLDS.to_vec(),
            start_mode: YMazeStartMode::default(),
            apply_dwell_filter: true,
            arm_membership_hysteresis: true,
            zone_max_gap_seconds: DEFAULT_ZONE_MAX_GAP_SECONDS,
            pose_gap_fill_seconds: DEFAULT_POSE_GAP_FILL_SECONDS,
            zone_occupancy: ZoneOccupancy::default(),
        }
    }

    fn n_arms(&self) -> usize {
        self.maze.shape.arms.len().min(ARM_LABELS.len())
    }

    pub fn effective_temporal_policy(&self) -> TemporalPolicy {
        self.temporal_policy.unwrap_or_else(|| {
            if self.uses_principal_entry_rule() {
                TemporalPolicy::ymaze_default()
            } else {
                TemporalPolicy {
                    min_duration_seconds: if self.apply_dwell_filter {
                        f64::from(self.params.min_dwell_frames()) / self.params.fps
                    } else {
                        0.0
                    },
                    ..TemporalPolicy::default()
                }
            }
        })
    }

    pub fn validate_temporal_policy(&self) -> Result<(), String> {
        self.arm_thresholds()?;
        let policy = self.effective_temporal_policy();
        if self.uses_principal_entry_rule() {
            policy.validate_uninterrupted()
        } else {
            policy.validate()
        }
    }

    /// The factory and annotation preview use the same parameter resolution.
    pub fn apply_overrides(&mut self, ov: &dyn crate::TaskOverrides) -> Result<(), String> {
        if let Some(v) = ov.ymaze_temporal_policy() {
            self.temporal_policy = Some(v);
        }
        if let Some(v) = ov.ymaze_threshold_geometry() {
            self.threshold_geometry = v;
        }
        if let Some(v) = ov.ymaze_proximal_threshold() {
            self.proximal_threshold = v;
        }
        if let Some(v) = ov.ymaze_principal_thresholds() {
            self.principal_thresholds = v;
        }
        if let Some(v) = ov.ymaze_start_mode() {
            self.start_mode = v;
        }
        if let Some(v) = ov.ymaze_apply_dwell_filter() {
            self.apply_dwell_filter = v;
        }
        if let Some(v) = ov.ymaze_arm_membership_hysteresis() {
            self.arm_membership_hysteresis = v;
        }
        if let Some(v) = ov.ymaze_zone_max_gap_seconds() {
            self.zone_max_gap_seconds = v;
        }
        if let Some(v) = ov.ymaze_pose_gap_fill_seconds() {
            self.pose_gap_fill_seconds = v;
        }
        if let Some(v) = ov.ymaze_zone_occupancy() {
            self.zone_occupancy = v;
        }
        self.validate_temporal_policy()
    }

    /// Radial keeps `threshold_radii_squared`; cross-line records each arm's
    /// start edge, distal line and the p% line endpoints instead.
    fn crossline_or_radial_geometry(&self, mut provenance: serde_json::Value) -> serde_json::Value {
        let fields = provenance.as_object_mut().expect("provenance object");
        match self.threshold_geometry {
            YMazeThresholdGeometry::ImenRadialVertexSpanV1 => {
                fields.insert(
                    "threshold_radii_squared".into(),
                    serde_json::json!(self.arm_proximal_distances_sq()),
                );
            }
            YMazeThresholdGeometry::ArmStartCrosslineV1 => {
                let frames: Vec<_> = (0..self.n_arms())
                    .map(|i| self.maze.shape.arm_frame(i).expect("validated arm frame"))
                    .collect();
                fields.insert(
                    "crossline_endpoints".into(),
                    serde_json::json!(
                        frames
                            .iter()
                            .map(|f| {
                                let l = f
                                    .crossline(self.proximal_threshold)
                                    .expect("validated cross-line");
                                serde_json::json!({"p1": l.p1, "p2": l.p2})
                            })
                            .collect::<Vec<_>>()
                    ),
                );
                fields.insert("arm_frames".into(), serde_json::json!(frames));
            }
        }
        provenance
    }

    fn arm_col_names(&self) -> Vec<String> {
        (0..self.n_arms())
            .map(|i| format!("in_arm_{}", ARM_LABELS[i]))
            .collect()
    }

    fn pose_gap_fill_columns(&self) -> [String; 4] {
        pose_gap_fill_columns(&self.params)
    }

    /// Compute the squared center-centroid distance of a per-arm threshold line.
    ///
    /// For each arm polygon, compute the distance from the center centroid to
    /// every arm vertex, then take `nearest = min(vertex_distances)` and
    /// `farthest = max(vertex_distances)`. The threshold line sits at:
    ///
    /// `nearest + threshold * (farthest - nearest)`
    ///
    /// This is the arm polygon's radial span measured outward from its nearest
    /// vertex, not a fraction of the centroid-to-tip distance.
    pub fn arm_threshold_distances_sq(&self, threshold: f64) -> Vec<f64> {
        let (cx, cy) = self.maze.shape.center.centroid();
        self.maze
            .shape
            .arms
            .iter()
            .map(|arm| {
                let dists: Vec<f64> = arm
                    .vertices
                    .iter()
                    .map(|&(x, y)| ((x - cx).powi(2) + (y - cy).powi(2)).sqrt())
                    .collect();
                let nearest = dists.iter().cloned().fold(f64::INFINITY, f64::min);
                let farthest = dists.iter().cloned().fold(0.0_f64, f64::max);
                let arm_length = farthest - nearest;
                let threshold_dist = nearest + threshold * arm_length;
                threshold_dist * threshold_dist // squared for efficient comparison
            })
            .collect()
    }

    fn arm_proximal_distances_sq(&self) -> Vec<f64> {
        self.arm_threshold_distances_sq(self.proximal_threshold)
    }

    /// Per-arm threshold predicates for the configured geometry. Errors when a
    /// cross-line arm has no unique start edge or `p` is outside [0, 1].
    fn arm_thresholds(&self) -> Result<Vec<ArmThreshold>, String> {
        match self.threshold_geometry {
            YMazeThresholdGeometry::ImenRadialVertexSpanV1 => {
                let center = self.maze.shape.center.centroid();
                Ok(self
                    .arm_proximal_distances_sq()
                    .into_iter()
                    .take(self.n_arms())
                    .map(|min_distance_sq| ArmThreshold::Radial {
                        center,
                        min_distance_sq,
                    })
                    .collect())
            }
            YMazeThresholdGeometry::ArmStartCrosslineV1 => {
                if !(0.0..=1.0).contains(&self.proximal_threshold) {
                    return Err("cross-line threshold must be within [0, 1]".into());
                }
                (0..self.n_arms())
                    .map(|i| {
                        Ok(ArmThreshold::Crossline(
                            self.maze
                                .shape
                                .arm_frame(i)?
                                .crossline(self.proximal_threshold)?,
                        ))
                    })
                    .collect()
            }
        }
    }

    fn heuristics_with_hysteresis(
        &self,
        arm_membership_hysteresis: bool,
    ) -> Vec<Box<dyn Heuristic>> {
        self.heuristics_shared(arm_membership_hysteresis, None)
    }

    fn heuristics_shared(
        &self,
        arm_membership_hysteresis: bool,
        shared: Option<&std::sync::Arc<YMazeShared>>,
    ) -> Vec<Box<dyn Heuristic>> {
        let mut heuristics: Vec<Box<dyn Heuristic>> = Vec::new();
        let proximal = self
            .arm_thresholds()
            .expect("Y-maze geometry is validated before analysis");

        // Dual-body-part zone confinement for each arm. `in_arm_X = true`
        // requires both head and tail inside the arm polygon and at or beyond
        // `nearest + proximal_threshold * (farthest - nearest)`. These zones
        // feed `zone_sequence`, `arm_sequence`, and spontaneous alternation.
        for (i, label) in ARM_LABELS.iter().enumerate().take(self.n_arms()) {
            heuristics.push(Box::new(
                DualBodyPartZoneHeuristic::new(
                    format!("in_arm_{label}"),
                    Perimeter::new(format!("arm_{label}"), self.maze.shape.arms[i].clone()),
                    self.params.head_coords.clone(),
                    self.params.tail_coords.clone(),
                    Some(proximal[i]),
                )
                .with_hysteresis(arm_membership_hysteresis)
                .with_shared(shared.cloned(), i, self.proximal_threshold),
            ));
        }

        // Center zone: head in center polygon OR head in the part of an arm
        // closer than `nearest + proximal_threshold * (farthest - nearest)`.
        heuristics.push(Box::new(YMazeCenterHeuristic {
            center: Perimeter::new("center", self.maze.shape.center.clone()),
            arm_polygons: self.maze.shape.arms[..self.n_arms()].to_vec(),
            head_coords: self.params.head_coords.clone(),
            proximal,
            shared: shared.cloned(),
            p: self.proximal_threshold,
        }));

        // Freezing
        heuristics.push(Box::new(FreezingHeuristic::new(
            &self.params.head_coords.x,
            &self.params.head_coords.y,
            self.params.freezing_window_seconds,
            self.params.freezing_displacement_threshold,
            self.params.fps,
        )));

        heuristics
    }

    fn apply_arm_membership_hysteresis(&self, df: &mut DataFrame) {
        if !self.arm_membership_hysteresis {
            return; // coverage: unreachable — the only caller checks the flag first
        }

        let head_present = coordinate_present_values(df, &self.params.head_coords);
        let tail_present = coordinate_present_values(df, &self.params.tail_coords);
        let pose_present: Vec<bool> = head_present
            .iter()
            .zip(&tail_present)
            .map(|(head, tail)| *head && *tail)
            .collect();

        for label in ARM_LABELS.iter().take(self.n_arms()) {
            let zone_name = format!("in_arm_{label}");
            let head_in = zone_truth_values(df, &diag_column_name(&zone_name, "head_in_zone"));
            let tail_in = zone_truth_values(df, &diag_column_name(&zone_name, "tail_in_zone"));
            let head_past =
                zone_truth_values(df, &diag_column_name(&zone_name, "head_past_threshold"));
            let tail_past =
                zone_truth_values(df, &diag_column_name(&zone_name, "tail_past_threshold"));
            let latched =
                latched_arm_membership(&head_in, &tail_in, &head_past, &tail_past, &pose_present);
            df.with_column(Series::new(zone_name.into(), latched).into_column())
                .expect("replace latched in-arm column");
        }
    }

    /// Run full Y-maze analysis with zone summaries and all spec measurements.
    pub fn full_analyze(&self, lf: LazyFrame) -> YMazeResult {
        self.full_analyze_timed(lf, &bekiper_core::video::FrameTiming::cfr(self.params.fps))
    }

    pub fn full_analyze_timed(
        &self,
        lf: LazyFrame,
        timing: &bekiper_core::video::FrameTiming,
    ) -> YMazeResult {
        self.full_analyze_with(lf, timing, None)
    }

    /// `shared` carries per-recording zone geometry of a variant group; it must
    /// have been built from the same input frame (`YMazeShared::build`).
    fn full_analyze_with(
        &self,
        lf: LazyFrame,
        timing: &bekiper_core::video::FrameTiming,
        shared: Option<&YMazeSharedPair>,
    ) -> YMazeResult {
        self.validate_temporal_policy()
            .expect("invalid Y-maze temporal policy");
        let temporal_policy = self.effective_temporal_policy();
        let principal = self.uses_principal_entry_rule();
        let crossline = self.threshold_geometry == YMazeThresholdGeometry::ArmStartCrosslineV1;
        let apply_dwell = temporal_policy.min_duration_seconds > 0.0;
        // Historical mitigations change predicate transitions. Preserve them
        // for comparison analyses, never silently apply them to principal entries.
        let gap_fill_seconds = if principal {
            0.0
        } else {
            self.pose_gap_fill_seconds
        };
        let gap_seconds = if principal {
            0.0
        } else {
            self.zone_max_gap_seconds
        };
        let hysteresis = !principal && self.arm_membership_hysteresis;
        let gap_fill_frames = ttl_frames_from_seconds(gap_fill_seconds, self.params.fps);
        let pose_gap_fill_columns = self.pose_gap_fill_columns();
        let pose_gap_fill_refs = pose_gap_fill_columns
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        let raw_lf = lf.clone();
        let mitigated_lf = if gap_fill_frames > 0 {
            apply_ttl_forward_fill(lf, &pose_gap_fill_refs, gap_fill_frames)
        } else {
            lf
        };
        let heuristics = self.heuristics_shared(hysteresis, shared.map(|pair| &pair.mitigated));
        let mut mitigated_df = collect_with_heuristics(mitigated_lf, &heuristics);
        if hysteresis {
            self.apply_arm_membership_hysteresis(&mut mitigated_df);
        }

        let raw_heuristics = self.heuristics_shared(false, shared.map(|pair| &pair.raw));
        let raw_df = collect_with_heuristics(raw_lf, &raw_heuristics);
        let bout_timing =
            BoutTiming::frames(timing, raw_df.height()).expect("invalid Y-maze frame timing");
        let total_duration = bout_timing.duration(0, raw_df.height());

        let trim_start = match self.start_mode {
            YMazeStartMode::FirstDetection => first_detection_start_index(
                &raw_df,
                &[&self.params.head_coords, &self.params.tail_coords],
            ),
            YMazeStartMode::VideoStart => 0,
        };
        let mut mitigated_df = trim_to_start_index(&mitigated_df, trim_start);
        let raw_df = trim_to_start_index(&raw_df, trim_start);
        let bout_timing = bout_timing.trimmed(trim_start);

        let arm_col_names = self.arm_col_names();
        let zone_max_gap_frames = (gap_seconds * self.params.fps).round().max(0.0) as u32;
        let n_arms = self.n_arms();

        let mut selections = std::collections::BTreeMap::new();
        for name in arm_col_names
            .iter()
            .map(String::as_str)
            .chain(std::iter::once("in_center"))
        {
            let selected = mitigated_df
                .select_temporal_bouts(name, &temporal_policy, &bout_timing)
                .expect("select Y-maze bouts");
            mitigated_df
                .with_column(
                    Series::new(
                        format!("{name}_before_dwell").into(),
                        zone_truth_values(&mitigated_df, name),
                    )
                    .into_column(),
                )
                .expect("retain pre-dwell predicate");
            mitigated_df
                .with_column(Series::new(name.into(), &selected.mask).into_column())
                .expect("apply selected Y-maze mask");
            selections.insert(name.to_string(), selected);
        }

        // Zone summaries for arms + center, over the configured occupancy.
        let occupancy = |name: &str| match self.zone_occupancy {
            ZoneOccupancy::DwellFiltered => name.to_string(),
            ZoneOccupancy::Raw => format!("{name}_before_dwell"),
        };
        let zone_summaries: Vec<ZoneSummary> = arm_col_names
            .iter()
            .map(String::as_str)
            .chain(std::iter::once("in_center"))
            .map(|col_name| {
                zone_summary_from_mask(
                    &mitigated_df,
                    col_name,
                    &self.params.head_coords.x,
                    &self.params.head_coords.y,
                    mitigated_df
                        .column(&occupancy(col_name))
                        .unwrap()
                        .bool()
                        .unwrap(),
                    &bout_timing,
                )
            })
            .collect();
        // Extract two parallel zone sequences:
        // - configured comparison or principal instantaneous sequence -> valid counts
        // - raw            -> drives `raw_*` counts (every threshold crossing)
        let zone_cols: Vec<&str> = arm_col_names
            .iter()
            .map(|s| s.as_str())
            .chain(std::iter::once("in_center"))
            .collect();

        // When `apply_dwell_filter` is false, the headline SA sequence is
        // built directly from the mitigated arm-threshold crossings. The
        // truly raw `raw_*` view remains independent and may therefore differ
        // from the headline even with dwell filtering disabled.
        let mut zone_values_owned: Vec<(String, Vec<bool>)> = arm_col_names
            .iter()
            .enumerate()
            .map(|(idx, col_name)| {
                (
                    ARM_LABELS[idx].to_string(),
                    zone_truth_values(&mitigated_df, col_name.as_str()),
                )
            })
            .collect();
        zone_values_owned.push((
            "Center".to_string(),
            zone_truth_values(&mitigated_df, "in_center"),
        ));

        let zone_values: Vec<(&str, Vec<bool>)> = zone_values_owned
            .iter()
            .map(|(label, bools)| (label.as_str(), bools.clone()))
            .collect();

        let zone_sequence =
            extract_zone_sequence(&zone_values, mitigated_df.height(), zone_max_gap_frames);
        let arm_labels = &ARM_LABELS[..n_arms];
        let arm_sequence = reduce_to_arm_sequence(&zone_sequence, arm_labels);

        let mut raw_zone_values_owned: Vec<(String, Vec<bool>)> = arm_col_names
            .iter()
            .enumerate()
            .map(|(idx, col_name)| {
                (
                    ARM_LABELS[idx].to_string(),
                    zone_truth_values(&raw_df, col_name.as_str()),
                )
            })
            .collect();
        raw_zone_values_owned.push((
            "Center".to_string(),
            zone_truth_values(&raw_df, "in_center"),
        ));

        let raw_zone_values: Vec<(&str, Vec<bool>)> = raw_zone_values_owned
            .iter()
            .map(|(label, bools)| (label.as_str(), bools.clone()))
            .collect();
        let raw_zone_sequence = extract_zone_sequence(&raw_zone_values, raw_df.height(), 0);
        let raw_arm_sequence = reduce_to_arm_sequence(&raw_zone_sequence, arm_labels);

        let (correct, incorrect, neutral) = analyze_alternations(&arm_sequence);
        let permutation_distribution = permutation_alternation_distribution(&arm_sequence);
        let distributions = triad_distributions(&arm_sequence);
        let spontaneous_pct = spontaneous_alternation_pct(correct, incorrect);

        // Parallel SA view from the truly raw arm sequence — pose gap-fill,
        // arm-membership hysteresis, and zone-gap bridging are all bypassed
        // so analysts can compare the mitigated headline against the original
        // arm-threshold crossings.
        let (raw_correct, raw_incorrect, raw_neutral) = analyze_alternations(&raw_arm_sequence);
        let raw_spontaneous_pct = spontaneous_alternation_pct(raw_correct, raw_incorrect);

        let mut arm_entries = [0u32; 3];
        let mut arm_durations = [0.0f64; 3];
        for (i, summary) in zone_summaries.iter().take(n_arms).enumerate() {
            arm_entries[i] = summary.entries;
            arm_durations[i] = summary.duration_seconds;
        }

        let center_entries = zone_summaries
            .iter()
            .find(|s| s.zone_name == "in_center")
            .map(|s| s.entries)
            .unwrap_or(0);

        let pref = arm_preference(&arm_durations);
        let seconds_in_arms = arm_durations.iter().take(n_arms).sum::<f64>();
        let seconds_in_arms_and_center = seconds_in_arms
            + zone_summaries
                .iter()
                .find(|s| s.zone_name == "in_center")
                .map(|s| s.duration_seconds)
                .unwrap_or(0.0);
        let area_motion_summaries: Vec<ZoneMotionSummary> = zone_cols
            .iter()
            .map(|&zone_col| {
                motion_summary_from_mask(
                    &mitigated_df,
                    zone_col,
                    mitigated_df
                        .column(&occupancy(zone_col))
                        .unwrap()
                        .bool()
                        .unwrap(),
                    &self.params.centroid_coords.x,
                    &self.params.centroid_coords.y,
                    self.params.fps,
                    "freezing",
                )
            })
            .collect();

        let freezing_seconds =
            bout_timing.mask_seconds(&zone_truth_values(&mitigated_df, "freezing"));

        let analysis_duration = bout_timing.duration(0, mitigated_df.height());
        // Whole analysed path, including frames outside every zone predicate
        // (e.g. head past the threshold but tail not); same landmark as zones.
        let total_distance = path_distance(
            &mitigated_df,
            &self.params.head_coords.x,
            &self.params.head_coords.y,
            None,
        );
        let total_mean_speed = if analysis_duration > 0.0 {
            total_distance / analysis_duration
        } else {
            0.0
        };

        YMazeResult {
            analysis: YMazeAnalysis {
                zone_sequence,
                first_arm_entered: arm_sequence.first().cloned(),
                last_arm_entered: arm_sequence.last().cloned(),
                arm_sequence,
                raw_arm_sequence,
                arm_entries,
                center_entries,
                correct_alternations: correct,
                incorrect_alternations: incorrect,
                neutral_triads: neutral,
                spontaneous_alternation_pct: spontaneous_pct,
                arm_preference: pref,
                permutation_distribution: permutation_distribution.clone(),
                sum_of_permutation_distribution: permutation_distribution.total(),
                correct_triad_distribution: distributions.correct,
                incorrect_triad_distribution: distributions.incorrect,
                neutral_triad_distribution: distributions.neutral,
                raw_correct_alternations: raw_correct,
                raw_incorrect_alternations: raw_incorrect,
                raw_neutral_triads: raw_neutral,
                raw_spontaneous_alternation_pct: raw_spontaneous_pct,
            },
            zone_summaries,
            area_motion_summaries,
            seconds_in_arms,
            seconds_in_arms_and_center,
            freezing_seconds,
            total_distance_meters: total_distance,
            total_mean_speed,
            total_duration_seconds: total_duration,
            analysis_duration_seconds: analysis_duration,
            proximal_threshold: self.proximal_threshold,
            start_mode: self.start_mode,
            arm_membership_hysteresis: hysteresis,
            zone_max_gap_seconds: gap_seconds,
            pose_gap_fill_seconds: gap_fill_seconds,
            scoring_provenance: Some(self.crossline_or_radial_geometry(serde_json::json!({
                "geometry_version": self.threshold_geometry,
                "center_definition": "arithmetic_mean_of_canonical_center_vertices",
                "center": self.maze.shape.center.centroid(),
                "center_polygon": self.maze.shape.center,
                "proximal_threshold": self.proximal_threshold,
                "package_version": env!("CARGO_PKG_VERSION"),
                "entry_implementation_version": if crossline { "bekiper_ymaze_arm_crossline_2026_10_28_v1" } else { "bekiper_ymaze_principal_2026_10_06_v2" },
                "arm_polygons": self.maze.shape.arms,
                "head_columns": self.params.head_coords,
                "tail_columns": self.params.tail_coords,
                "requires_both_landmarks_same_arm": true,
                "entry_rule": if principal || crossline { "dual_landmark_false_to_true_v1" } else { "configured_comparison_v1" },
                "entry_frame_origin": "original_input_frame_index",
                "analysis_start_frame": trim_start,
                "analysis_start_timestamp_s": bout_timing.boundary(0),
                "sample_end_timestamp_s": bout_timing.boundary(bout_timing.len()),
                "entry_frames": arm_col_names.iter().map(|name| {
                    let mask = zone_truth_values(&mitigated_df, name);
                    let frames: Vec<usize> = mask.iter().enumerate()
                        .filter(|(i, state)| **state && (*i == 0 || !mask[*i - 1]))
                        .map(|(i, _)| i + trim_start).collect();
                    (name.clone(), frames)
                }).collect::<std::collections::BTreeMap<_, _>>(),
                "apply_dwell_filter": apply_dwell,
                "zone_occupancy": self.zone_occupancy,
                "temporal_policy": temporal_policy,
                "temporal_policy_version": "sample_support_v1",
                "temporal_sample_support": "half_open_forward_interval_final_nominal_fps",
                "dwell_decision": if self.temporal_policy.is_some() { "explicit_configurable_dwell_2026_10_06" } else { "legacy_default" },
                "raw_entry_frames": selections.iter().filter(|(name, _)| name.as_str() != "in_center").map(|(name, selected)| (name, selected.raw_spans.iter().map(|s| s.start + trim_start).collect::<Vec<_>>())).collect::<std::collections::BTreeMap<_, _>>(),
                "dwell_bouts": selections.iter().map(|(name, selected)| (name, selected.bouts.iter().map(|b| serde_json::json!({
                    "start_frame": b.start + trim_start, "end_frame_exclusive": b.end + trim_start,
                    "crossing_timestamp_s": bout_timing.boundary(b.start), "duration_s": b.duration,
                    "accepted": b.accepted, "rejection_reason": if b.accepted { None } else { Some("minimum_dwell") },
                    "qualification_frame": b.qualification_index.map(|i| i + trim_start),
                    "qualification_timestamp_s": b.qualification_index.map(|i| bout_timing.boundary(i + usize::from(apply_dwell)))
                })).collect::<Vec<_>>())).collect::<std::collections::BTreeMap<_, _>>(),
                "arm_membership_hysteresis": hysteresis,
                "zone_max_gap_seconds": gap_seconds,
                "pose_gap_fill_seconds": gap_fill_seconds,
                "minimum_dwell_seconds": temporal_policy.min_duration_seconds,
                "requested_apply_dwell_filter": self.apply_dwell_filter,
                "requested_arm_membership_hysteresis": self.arm_membership_hysteresis,
                "requested_zone_max_gap_seconds": self.zone_max_gap_seconds,
                "requested_pose_gap_fill_seconds": self.pose_gap_fill_seconds,
                "principal_amendment": if crossline { "owner_decision_2026_10_28" } else { "researcher_clarification_2026_10_04" },
            }))),
            evaluated_df: Some(mitigated_df),
        }
    }
}

fn pose_gap_fill_columns(params: &TaskParams) -> [String; 4] {
    [
        params.head_coords.x.clone(),
        params.head_coords.y.clone(),
        params.tail_coords.x.clone(),
        params.tail_coords.y.clone(),
    ]
}

fn ttl_frames_from_seconds(seconds: f64, fps: f64) -> u32 {
    (seconds * fps).round().max(0.0) as u32
}

fn apply_ttl_forward_fill(lf: LazyFrame, columns: &[&str], ttl_frames: u32) -> LazyFrame {
    if ttl_frames == 0 || columns.is_empty() {
        return lf;
    }

    let exprs = columns
        .iter()
        .map(|column_name| {
            col(*column_name)
                .fill_null_with_strategy(FillNullStrategy::Forward(Some(ttl_frames as IdxSize)))
                .alias(*column_name)
        })
        .collect::<Vec<_>>();
    lf.with_columns(exprs)
}

fn collect_with_heuristics(lf: LazyFrame, heuristics: &[Box<dyn Heuristic>]) -> DataFrame {
    heuristics
        .iter()
        .fold(lf, |lf, heuristic| heuristic.evaluate(lf))
        .collect()
        .expect("failed to collect Y-maze results")
}

include!("ymaze/task_adapter.rs");

/// Heuristic: true when BOTH head and tail are inside the zone and past an
/// optional radial threshold line.
///
/// In the Y-maze arm case, the line is computed from the
/// center centroid as
/// `nearest_vertex_dist + threshold * (farthest_vertex_dist - nearest_vertex_dist)`.
/// An `in_arm_X` frame therefore requires head and tail both inside the arm
/// polygon and at or beyond that radial boundary. Principal analyses count
/// predicate transitions, optionally qualified by an explicit temporal policy.
struct DualBodyPartZoneHeuristic {
    zone_name: String,
    perimeter: Perimeter<Polygon>,
    head_coords: CoordinateColumns,
    tail_coords: CoordinateColumns,
    /// Threshold predicate both head and tail must satisfy; `None` disables it.
    /// Radial: squared centre-centroid distance from
    /// `nearest_vertex_dist + threshold * (farthest_vertex_dist - nearest_vertex_dist)`.
    /// Cross-line: distal side of the p% arm-start line.
    threshold: Option<ArmThreshold>,
    /// Whether the parent task's post-collect latch should rewrite this zone
    /// for the current analysis chain. Diagnostic columns remain raw either
    /// way.
    arm_membership_hysteresis: bool,
    /// Per-recording geometry of a variant group, the arm index and `p`.
    shared: Option<(std::sync::Arc<YMazeShared>, usize, f64)>,
}

impl DualBodyPartZoneHeuristic {
    fn new(
        zone_name: String,
        perimeter: Perimeter<Polygon>,
        head_coords: CoordinateColumns,
        tail_coords: CoordinateColumns,
        threshold: Option<ArmThreshold>,
    ) -> Self {
        Self {
            zone_name,
            perimeter,
            head_coords,
            tail_coords,
            threshold,
            arm_membership_hysteresis: true,
            shared: None,
        }
    }

    fn with_shared(
        mut self,
        shared: Option<std::sync::Arc<YMazeShared>>,
        arm: usize,
        p: f64,
    ) -> Self {
        self.shared = shared.map(|shared| (shared, arm, p));
        self
    }

    fn with_hysteresis(mut self, arm_membership_hysteresis: bool) -> Self {
        self.arm_membership_hysteresis = arm_membership_hysteresis;
        self
    }
}

include!("ymaze/threshold_tests.rs");

include!("ymaze/dwell_tests.rs");

include!("ymaze/start_tests.rs");

include!("ymaze/crossline_tests.rs");

/// Polars float `>=`: NaN is the greatest value and equals itself.
fn polars_ge(a: f64, b: f64) -> bool {
    a.is_nan() || (!b.is_nan() && a >= b)
}

include!("ymaze/shared.rs");
