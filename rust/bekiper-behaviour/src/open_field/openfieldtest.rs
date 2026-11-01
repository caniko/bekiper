use polars::prelude::*;

use bekiper_core::inspection::QuadrantSpec;
use bekiper_core::types::TaskType;
use bekiper_core::video::FrameTiming;
use bekiper_feature::heuristic::Heuristic;
use bekiper_feature::heuristic::solo::freezing::FreezingHeuristic;
use bekiper_feature::heuristic::solo::grooming::GroomingHeuristic;
use bekiper_feature::heuristic::solo::periphery::PeripheryHeuristic;
use bekiper_feature::heuristic::solo::rearing::RearingHeuristic;
use bekiper_feature::heuristic::solo::zone_confinement::ZoneConfinementHeuristic;
use bekiper_feature::quadrant::{add_grid_zone_columns, generate_grid_zones};
use bekiper_feature::tolerance::{BoutTiming, TemporalPolicy};
use bekiper_feature::zone::{ZoneSummary, zone_summary_from_mask};
use bekiper_math::geometry::path_distance;
use bekiper_perimeter::perimeter::Perimeter;
use bekiper_perimeter::rectangle::Rectangle;

use crate::BehaviouralTask;
use crate::TaskDetail;
use crate::motion_detail::{ZoneMotionSummary, motion_summary_from_mask};
use crate::params::{
    DEFAULT_OFT_FREEZING_MIN_BOUT_S, DEFAULT_OFT_GROOMING_MIN_BOUT_S,
    DEFAULT_OFT_REARING_MIN_BOUT_S, TaskParams,
};

// ── Rearing / grooming defaults ───────────────────────────────────────────
// Re-exported from `bekiper-feature` so task-level doc discoverability
// stays close to the task, while the physical constants live next to the
// algorithm. Overridable at runtime via `TaskParamsOverrides::rearing`
// and `TaskParamsOverrides::grooming` in `bekiper-ingress/src/config.rs`.

pub use bekiper_feature::heuristic::solo::grooming::{
    DEFAULT_FOREPAW_PROXIMITY as DEFAULT_GROOMING_FOREPAW_PROXIMITY,
    DEFAULT_MAX_BODY_LENGTH as DEFAULT_GROOMING_MAX_BODY_LENGTH,
    DEFAULT_MAX_SPEED_MPS as DEFAULT_GROOMING_MAX_SPEED,
    DEFAULT_MIN_BODY_LENGTH as DEFAULT_GROOMING_MIN_BODY_LENGTH,
};
pub use bekiper_feature::heuristic::solo::rearing::{
    DEFAULT_FOREPAW_PROXIMITY as DEFAULT_REARING_FOREPAW_PROXIMITY,
    DEFAULT_MAX_BODY_LENGTH as DEFAULT_REARING_MAX_BODY_LENGTH,
    DEFAULT_MAX_BODY_LENGTH_RATIO as DEFAULT_REARING_MAX_BODY_LENGTH_RATIO,
    DEFAULT_MAX_SPEED_MPS as DEFAULT_REARING_MAX_SPEED,
};

/// Open Field Test task.
///
/// Arena is a square (e.g. 40×40 cm). Zones:
/// - Center: inner rectangle (e.g. 24×24 cm)
/// - Periphery: outer band (e.g. 8 cm from each wall)
///
/// The periphery is the full arena minus the center.
pub struct OpenFieldTest {
    pub params: TaskParams,
    /// Full arena perimeter.
    pub arena: Perimeter<Rectangle>,
    /// Center zone perimeter.
    pub center: Perimeter<Rectangle>,
    /// Rearing heuristic: max centroid speed (m/s). Default: 0.01.
    pub rearing_max_speed: f64,
    /// Rearing heuristic: max nose-to-tail ratio. Default: 0.7.
    pub rearing_max_body_length_ratio: f64,
    /// Rearing heuristic: absolute max nose-to-tail distance (m). Default: 0.05.
    pub rearing_max_body_length: f64,
    /// Rearing heuristic: forepaw-to-head proximity (m). Default: 0.03.
    pub rearing_forepaw_proximity: f64,
    /// Grooming heuristic: max speed (m/s). Default: 0.005.
    pub grooming_max_speed: f64,
    /// Grooming heuristic: max nose-to-tail distance (m). Default: 0.08.
    pub grooming_max_body_length: f64,
    /// Grooming heuristic: min nose-to-tail distance (m). Default: 0.04.
    pub grooming_min_body_length: f64,
    /// Grooming heuristic: forepaw-to-nose proximity (m). Default: 0.02.
    pub grooming_forepaw_proximity: f64,
}

impl OpenFieldTest {
    /// Create an OFT with standard dimensions.
    ///
    /// `arena_center`: (x, y) center of arena in meters.
    /// `arena_side`: arena side length in meters (e.g. 0.40 for 40cm).
    /// `peripheral_band`: width of peripheral zone in meters (e.g. 0.08 for 8cm).
    ///
    /// # Panics
    /// Panics if `arena_side` is not positive or if `peripheral_band` is not
    /// smaller than half of `arena_side` (center zone would be zero or negative).
    pub fn standard(
        arena_center: (f64, f64),
        arena_side: f64,
        peripheral_band: f64,
        fps: f64,
    ) -> Self {
        assert!(
            arena_side > 0.0,
            "arena_side must be positive, got {arena_side}"
        );
        assert!(
            peripheral_band < arena_side / 2.0,
            "peripheral_band ({peripheral_band}) must be less than half of arena_side ({arena_side})"
        );
        let center_side = arena_side - 2.0 * peripheral_band;
        let arena = Perimeter::new(
            "arena",
            Rectangle::new(arena_center.0, arena_center.1, arena_side, arena_side),
        );
        let center = Perimeter::new(
            "in_center",
            Rectangle::new(arena_center.0, arena_center.1, center_side, center_side),
        );

        Self {
            params: TaskParams::new(fps),
            arena,
            center,
            rearing_max_speed: DEFAULT_REARING_MAX_SPEED,
            rearing_max_body_length_ratio: DEFAULT_REARING_MAX_BODY_LENGTH_RATIO,
            rearing_max_body_length: DEFAULT_REARING_MAX_BODY_LENGTH,
            rearing_forepaw_proximity: DEFAULT_REARING_FOREPAW_PROXIMITY,
            grooming_max_speed: DEFAULT_GROOMING_MAX_SPEED,
            grooming_max_body_length: DEFAULT_GROOMING_MAX_BODY_LENGTH,
            grooming_min_body_length: DEFAULT_GROOMING_MIN_BODY_LENGTH,
            grooming_forepaw_proximity: DEFAULT_GROOMING_FOREPAW_PROXIMITY,
        }
    }

    /// Run full OFT analysis with zone summaries and all spec measurements.
    pub fn full_analyze(&self, lf: LazyFrame) -> OftResult {
        self.full_analyze_with_timing(lf, &FrameTiming::cfr(self.params.fps))
    }

    fn full_analyze_with_timing(&self, lf: LazyFrame, timing: &FrameTiming) -> OftResult {
        let window = self.select_window(lf, timing);
        let heuristics = self.heuristics();
        let evaluated = heuristics
            .iter()
            .fold(window.selected.clone().lazy(), |lf, h| h.evaluate(lf));
        let df = evaluated.collect().expect("failed to collect OFT results");
        self.finish(window, df)
    }

    /// Collect the poses and select the analysed window. Depends only on
    /// `oft_window_seconds`, the centroid columns and the timing.
    fn select_window(&self, lf: LazyFrame, timing: &FrameTiming) -> OftWindowSelection {
        let mut source = lf.collect().expect("failed to collect OFT poses");
        if source.column("frame").is_err() {
            source
                .with_column(
                    Series::new(
                        "frame".into(),
                        (0..source.height() as u64).collect::<Vec<_>>(),
                    )
                    .into_column(),
                )
                .unwrap();
        }
        let source_timing =
            BoutTiming::frames(timing, source.height()).expect("invalid OFT timing");
        let window_seconds = self.params.oft_window_seconds;
        assert!(
            window_seconds.is_finite() && window_seconds >= 0.0,
            "invalid OFT window"
        );
        let xs = source
            .column(&self.params.centroid_coords.x)
            .unwrap()
            .f64()
            .unwrap();
        let ys = source
            .column(&self.params.centroid_coords.y)
            .unwrap()
            .f64()
            .unwrap();
        let start = if window_seconds > 0.0 {
            (0..source.height())
                .find(|&i| {
                    xs.get(i).is_some_and(f64::is_finite) && ys.get(i).is_some_and(f64::is_finite)
                })
                .unwrap_or(source.height())
        } else {
            0
        };
        let selected_timing = if window_seconds > 0.0 {
            source_timing.window(start, source_timing.boundary(start) + window_seconds)
        } else {
            source_timing.sliced(start, source.height())
        };
        let selected = source.slice(start as i64, selected_timing.len());
        OftWindowSelection {
            source,
            start,
            source_timing,
            selected_timing,
            selected,
            window_seconds,
        }
    }

    /// Everything after heuristic evaluation: dwell/bout selection, zone and
    /// motion summaries. `df` holds the evaluated heuristic columns.
    fn finish(&self, window: OftWindowSelection, mut df: DataFrame) -> OftResult {
        let OftWindowSelection {
            source,
            start,
            source_timing,
            selected_timing,
            window_seconds,
            ..
        } = window;
        // Keep raw zone predicates for diagnostics; every scored consumer shares
        // the same seconds-based selected mask, with no frame-threshold rounding.
        let select = |df: &mut DataFrame, label: &str, raw: BooleanChunked, minimum: f64| {
            let policy = TemporalPolicy {
                min_duration_seconds: minimum,
                ..Default::default()
            };
            let selection = policy.filter_series(&raw, &selected_timing).unwrap();
            df.with_column(raw.with_name(format!("{label}_raw").into()).into_column())
                .unwrap();
            df.with_column(BooleanChunked::from_slice(label.into(), &selection.mask).into_column())
                .unwrap();
        };
        let bool_column = |df: &DataFrame, label: &str| -> BooleanChunked {
            df.column(label).unwrap().bool().unwrap().clone()
        };
        // Immobility = every stationary bout ≥1 s; the stationary heuristic is
        // published under `freezing`.
        let stationary = bool_column(&df, "freezing");
        for (label, minimum) in [
            ("in_center", self.params.min_dwell_seconds),
            ("in_periphery", self.params.min_dwell_seconds),
            ("rearing", DEFAULT_OFT_REARING_MIN_BOUT_S),
            ("grooming", DEFAULT_OFT_GROOMING_MIN_BOUT_S),
        ] {
            let raw = bool_column(&df, label);
            select(&mut df, label, raw, minimum);
        }
        // ponytail: criteria freezing excludes grooming/rearing, but those are
        // speed + body-length proxies that also match a crouched, frozen mouse.
        // Freezing reports immobility until the proxies are validated against
        // hand-scored video; then re-add `stationary & !(rearing | grooming)`.
        for label in ["immobility", "freezing"] {
            select(
                &mut df,
                label,
                stationary.clone(),
                DEFAULT_OFT_FREEZING_MIN_BOUT_S,
            );
        }
        df.with_column(
            Series::new(
                "sample_duration_s".into(),
                (0..df.height())
                    .map(|i| selected_timing.duration(i, i + 1))
                    .collect::<Vec<_>>(),
            )
            .into_column(),
        )
        .unwrap();
        let (center_summary, center_motion_summary) =
            self.selected_zone(&df, "in_center", &selected_timing);
        let (periphery_summary, periphery_motion_summary) =
            self.selected_zone(&df, "in_periphery", &selected_timing);
        let islands = |label: &str| {
            motion_islands_from_mask(
                &df,
                df.column(label).unwrap().bool().unwrap(),
                &selected_timing,
                &self.params.centroid_coords.x,
                &self.params.centroid_coords.y,
            )
        };
        let center_motion_islands = islands("in_center");
        let periphery_motion_islands = islands("in_periphery");
        let (quadrant_summaries, quadrant_motion_summaries) =
            self.quadrant_summaries(&df, &selected_timing);
        let seconds = |label: &str| {
            selected_timing.mask_seconds(
                &df.column(label)
                    .unwrap()
                    .bool()
                    .unwrap()
                    .into_iter()
                    .map(|v| v.unwrap_or(false))
                    .collect::<Vec<_>>(),
            )
        };
        let freezing_seconds = seconds("freezing");
        let immobility_seconds = seconds("immobility");
        let rearing_seconds = seconds("rearing");
        let grooming_seconds = seconds("grooming");
        let total_duration = selected_timing.duration(0, df.height());
        let total_distance = path_distance(
            &df,
            &self.params.centroid_coords.x,
            &self.params.centroid_coords.y,
            None,
        );
        let total_mean_speed = if total_duration > 0.0 {
            total_distance / total_duration
        } else {
            0.0
        };

        // Owner decision 2026-10-22: % of the analysed duration, so a short
        // recording keeps centre + periphery comparable; `complete` flags it.
        let denominator = total_duration;
        let center_time_pct = if denominator > 0.0 {
            100.0 * center_summary.duration_seconds / denominator
        } else {
            0.0
        };

        let periphery_time_pct = if denominator > 0.0 {
            100.0 * periphery_summary.duration_seconds / denominator
        } else {
            0.0
        };

        let immobility_pct = if total_duration > 0.0 {
            100.0 * immobility_seconds / total_duration
        } else {
            0.0
        };

        OftResult {
            window: OftWindow {
                source_start_frame: source_frame_boundary(&source, start),
                source_end_frame: source_frame_boundary(&source, start + df.height()),
                start_seconds: source_timing.boundary(start),
                end_seconds: selected_timing.boundary(df.height()),
                requested_seconds: window_seconds,
                complete: window_seconds == 0.0 || total_duration >= window_seconds,
                policy: if window_seconds > 0.0 {
                    "first_finite_centroid_half_open_v1"
                } else {
                    "full_video_half_open_v1"
                }
                .into(),
            },
            center_summary,
            periphery_summary,
            quadrant_summaries,
            quadrant_motion_summaries,
            center_motion_summary,
            periphery_motion_summary,
            center_motion_islands,
            periphery_motion_islands,
            center_time_pct,
            periphery_time_pct,
            freezing_seconds,
            immobility_seconds,
            immobility_pct,
            rearing_seconds,
            grooming_seconds,
            total_distance_meters: total_distance,
            total_mean_speed,
            total_duration_seconds: total_duration,
            evaluated_df: Some(df),
        }
    }

    fn quadrant_summaries(
        &self,
        df: &DataFrame,
        timing: &BoutTiming,
    ) -> (Vec<ZoneSummary>, Vec<ZoneMotionSummary>) {
        let rect = &self.arena.shape;
        let zones = generate_grid_zones(
            rect.center_x,
            rect.center_y,
            rect.width,
            rect.height,
            OPEN_FIELD_QUADRANT_ROWS as usize,
            OPEN_FIELD_QUADRANT_COLS as usize,
        );

        let mut df_with_quadrants =
            add_grid_zone_columns(df.clone().lazy(), &zones, &self.params.centroid_coords)
                .collect()
                .expect("failed to collect OFT quadrant results");

        let policy = TemporalPolicy {
            min_duration_seconds: self.params.min_dwell_seconds,
            ..Default::default()
        };
        zones
            .iter()
            .map(|(label, _)| {
                let raw = df_with_quadrants.column(label).unwrap().bool().unwrap();
                let selected = policy.filter_series(raw, timing).unwrap();
                df_with_quadrants
                    .with_column(
                        BooleanChunked::from_slice(label.clone().into(), &selected.mask)
                            .into_column(),
                    )
                    .unwrap();
                let (summary, motion) = self.selected_zone(&df_with_quadrants, label, timing);
                (summary, motion)
            })
            .unzip()
    }

    fn selected_zone(
        &self,
        df: &DataFrame,
        label: &str,
        timing: &BoutTiming,
    ) -> (ZoneSummary, ZoneMotionSummary) {
        let mask = df.column(label).unwrap().bool().unwrap();
        let coords = &self.params.centroid_coords;
        let summary = zone_summary_from_mask(df, label, &coords.x, &coords.y, mask, timing);
        let mut motion = motion_summary_from_mask(
            df,
            label,
            mask,
            &coords.x,
            &coords.y,
            self.params.fps,
            "freezing",
        );
        motion.total_displacement_meters = summary.distance_meters;
        let freezing = df.column("freezing").unwrap().bool().unwrap();
        motion.freezing_seconds = timing.mask_seconds(
            &(0..df.height())
                .map(|i| mask.get(i).unwrap_or(false) && freezing.get(i).unwrap_or(false))
                .collect::<Vec<_>>(),
        );
        (summary, motion)
    }
}

/// Window-selected poses shared by every variant with equal window inputs.
#[derive(Clone)]
pub struct OftWindowSelection {
    source: DataFrame,
    start: usize,
    source_timing: BoutTiming,
    selected_timing: BoutTiming,
    selected: DataFrame,
    window_seconds: f64,
}

/// Default grid dimensions for the Open Field quadrant overlay.
/// Matches the 4x4 convention from the legacy Trial inspector.
const OPEN_FIELD_QUADRANT_ROWS: u32 = 4;
const OPEN_FIELD_QUADRANT_COLS: u32 = 4;
