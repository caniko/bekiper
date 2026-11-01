use polars::prelude::*;
use serde::{Deserialize, Serialize};

use bekiper_core::inspection::{HeuristicDecompositionSpec, PerimeterSpec};
use bekiper_core::types::{CoordinateColumns, TaskType};
use bekiper_core::video::FrameTiming;
use bekiper_feature::heuristic::{
    Heuristic, coordinate_label_from_xy, decomposition_component, diag_column_name,
};
use bekiper_feature::zone::{ZoneSummary, zone_summary_timed};
use bekiper_math::discrete::{apply_tolerance_series, count_bouts};
use bekiper_math::geometry::path_distance;
use bekiper_perimeter::circle::Circle;
use bekiper_perimeter::perimeter::Perimeter;
use bekiper_perimeter::polygon::Polygon;
use bekiper_perimeter::radial_maze::RadialMaze;
use bekiper_perimeter::rectangle::Rectangle;
use bekiper_perimeter::triangle::Triangle;

use crate::{BehaviouralTask, TaskDetail, summaries_from_df};

const START_AREA_COLUMN: &str = "in_start_area";
const ENCLOSURE_COLUMN: &str = "in_enclosure";
const REWARD_MIN_DWELL_SECONDS: f64 = 0.5;

fn reward_column_name(index: usize) -> String {
    format!("in_reward_{index}")
}

fn bool_column_as_chunked(df: &DataFrame, col_name: &str) -> Option<BooleanChunked> {
    df.column(col_name)
        .ok()
        .and_then(|series| series.bool().ok())
        .cloned()
}

fn toleranced_mask(
    df: &DataFrame,
    col_name: &str,
    min_dwell_frames: u32,
) -> Option<BooleanChunked> {
    let raw = bool_column_as_chunked(df, col_name)?;
    Some(apply_tolerance_series(&raw, min_dwell_frames, 0))
}

fn chunked_or(lhs: &BooleanChunked, rhs: &BooleanChunked) -> BooleanChunked {
    lhs | rhs
}

fn chunked_to_vec(mask: &BooleanChunked) -> Vec<bool> {
    mask.into_iter()
        .map(|value| value.unwrap_or(false))
        .collect()
}

fn coordinate_at(
    df: &DataFrame,
    x_col: &str,
    y_col: &str,
    frame_index: usize,
) -> Option<(f64, f64)> {
    let xs = df.column(x_col).ok()?.f64().ok()?;
    let ys = df.column(y_col).ok()?.f64().ok()?;
    Some((xs.get(frame_index)?, ys.get(frame_index)?))
}

fn path_distance_between_frames(
    df: &DataFrame,
    x_col: &str,
    y_col: &str,
    start_frame: usize,
    end_frame_inclusive: usize,
) -> Option<f64> {
    if end_frame_inclusive <= start_frame {
        return Some(0.0);
    }

    let xs = df.column(x_col).ok()?.f64().ok()?;
    let ys = df.column(y_col).ok()?.f64().ok()?;
    let mut prev = (xs.get(start_frame)?, ys.get(start_frame)?);
    let mut total = 0.0;

    for frame in (start_frame + 1)..=end_frame_inclusive {
        let current = (xs.get(frame)?, ys.get(frame)?);
        let dx = current.0 - prev.0;
        let dy = current.1 - prev.1;
        total += (dx * dx + dy * dy).sqrt();
        prev = current;
    }

    Some(total)
}

fn combined_mask(masks: &[BooleanChunked]) -> Option<BooleanChunked> {
    let mut iter = masks.iter();
    let first = iter.next()?.clone();
    Some(iter.fold(first, |acc, mask| chunked_or(&acc, mask)))
}

fn spec_confinement_expr(spec: &PerimeterSpec, coords: &CoordinateColumns) -> Expr {
    match spec {
        PerimeterSpec::Circle {
            label,
            center_x,
            center_y,
            radius,
        } => Perimeter::new(label.clone(), Circle::new(*center_x, *center_y, *radius))
            .confinement_mask(coords),
        PerimeterSpec::Rectangle {
            label,
            center_x,
            center_y,
            width,
            height,
        } => Perimeter::new(
            label.clone(),
            Rectangle::new(*center_x, *center_y, *width, *height),
        )
        .confinement_mask(coords),
        PerimeterSpec::Polygon { label, vertices } => {
            Perimeter::new(label.clone(), Polygon::new(vertices.clone())).confinement_mask(coords)
        }
        PerimeterSpec::RadialMaze {
            label,
            center_vertices,
            arms,
        } => Perimeter::new(
            label.clone(),
            RadialMaze::new(
                Polygon::new(center_vertices.clone()),
                arms.iter().cloned().map(Polygon::new).collect(),
            ),
        )
        .confinement_mask(coords),
        PerimeterSpec::Triangle { label, vertices } => Perimeter::new(
            label.clone(),
            Triangle::new(vertices[0], vertices[1], vertices[2]),
        )
        .confinement_mask(coords),
    }
}

fn first_exit_frame(mask: &BooleanChunked) -> Option<usize> {
    let mut was_inside = false;
    for (frame_idx, value) in mask.into_iter().enumerate() {
        let inside = value.unwrap_or(false);
        if inside {
            was_inside = true;
        } else if was_inside {
            return Some(frame_idx);
        }
    }
    None
}

/// Cheeseboard / reward tracing task.
///
/// Animals search for baited wells on a board with multiple locations.
pub struct Cheeseboard {
    pub start_area: Option<PerimeterSpec>,
    pub enclosure: Perimeter<Circle>,
    pub reward_locations: Vec<Perimeter<Circle>>,
}

/// Typed detailed result for cheeseboard / reward tracing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheeseboardResult {
    /// Label of the enclosure perimeter.
    pub enclosure_label: String,
    /// Labels of the reward wells, in analysis order.
    pub reward_labels: Vec<String>,
    /// Summary for the enclosure perimeter.
    pub enclosure_summary: ZoneSummary,
    /// Per-reward summaries.
    pub reward_summaries: Vec<ZoneSummary>,
    /// First frame after leaving the explicit start area.
    pub start_area_exit_frame: Option<usize>,
    /// Elapsed seconds for `start_area_exit_frame`.
    pub start_area_exit_seconds: Option<f64>,
    /// First frame that enters any reward area after the start-area exit.
    pub reward_arrival_frame: Option<usize>,
    /// Elapsed seconds for `reward_arrival_frame`.
    pub reward_arrival_seconds: Option<f64>,
    /// Legacy-style latency to the first reward after the start-area exit.
    pub seconds_to_find_reward: Option<f64>,
    /// Seconds spent inside any reward area, after tolerance filtering.
    pub seconds_in_reward_area: f64,
    /// Number of distinct reward bouts across the union of reward areas.
    pub reward_entries: u32,
    /// Total path distance until the first reward arrival.
    pub search_path_distance_meters: Option<f64>,
    /// Legacy-style start-to-reward displacement alias.
    ///
    /// This currently matches `search_path_distance_meters`: the total path
    /// traversed between the explicit start-area exit and the first reward
    /// arrival.
    pub start_to_reward_displacement_meters: Option<f64>,
    /// Direct distance from the start proxy to the first reward arrival.
    pub search_direct_distance_meters: Option<f64>,
    /// Mean speed along the search segment before the first reward.
    pub search_mean_speed_mps: Option<f64>,
    /// Path/direct-distance ratio for the search segment.
    pub search_tortuosity: Option<f64>,
    /// Total path distance across the full trial.
    pub total_distance_meters: f64,
    /// Total mean speed across the full trial.
    pub total_mean_speed_mps: f64,
    /// Full trial duration.
    pub total_duration_seconds: f64,
    /// Evaluated DataFrame including the emitted boolean columns.
    #[serde(skip)]
    pub evaluated_df: Option<DataFrame>,
}

impl Cheeseboard {
    fn heuristic_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        if self.start_area.is_some() {
            names.push(START_AREA_COLUMN.to_string());
        }
        names.push(ENCLOSURE_COLUMN.to_string());
        names.extend(
            self.reward_locations
                .iter()
                .enumerate()
                .map(|(i, _)| reward_column_name(i)),
        );
        names
    }

    fn collect_evaluated_df(&self, lf: LazyFrame) -> DataFrame {
        let heuristics = self.heuristics();
        heuristics
            .iter()
            .fold(lf, |lf, h| h.evaluate(lf))
            .collect()
            .expect("failed to collect cheeseboard results")
    }

    pub fn full_analyze(&self, lf: LazyFrame, timing: &FrameTiming) -> CheeseboardResult {
        let df = self.collect_evaluated_df(lf);
        let min_dwell_frames = (timing.fps * REWARD_MIN_DWELL_SECONDS).round().max(1.0) as u32;

        let mut enclosure_summary = zone_summary_timed(
            &df,
            ENCLOSURE_COLUMN,
            "mouse_center_x",
            "mouse_center_y",
            timing,
            min_dwell_frames,
        );
        enclosure_summary.zone_name = self.enclosure.label.clone();

        let mut reward_summaries: Vec<ZoneSummary> = self
            .reward_locations
            .iter()
            .enumerate()
            .map(|(i, reward)| {
                let mut summary = zone_summary_timed(
                    &df,
                    &reward_column_name(i),
                    "mouse_center_x",
                    "mouse_center_y",
                    timing,
                    min_dwell_frames,
                );
                summary.zone_name = reward.label.clone();
                summary
            })
            .collect();

        let start_area_mask = self.start_area.as_ref().map(|_| {
            toleranced_mask(&df, START_AREA_COLUMN, min_dwell_frames).unwrap_or_else(|| {
                BooleanChunked::full(START_AREA_COLUMN.into(), false, df.height())
            })
        });
        let start_area_exit_frame = start_area_mask.as_ref().and_then(first_exit_frame);
        let start_area_exit_seconds =
            start_area_exit_frame.map(|frame| timing.frame_to_elapsed(frame));

        let reward_masks: Vec<BooleanChunked> = self
            .reward_locations
            .iter()
            .enumerate()
            .filter_map(|(i, _)| toleranced_mask(&df, &reward_column_name(i), min_dwell_frames))
            .collect();

        let reward_union_mask = combined_mask(&reward_masks);
        let reward_union_mask = reward_union_mask
            .unwrap_or_else(|| BooleanChunked::full("reward_union".into(), false, df.height()));
        let reward_arrival_frame = start_area_exit_frame.and_then(|start_frame| {
            reward_union_mask
                .clone()
                .into_iter()
                .enumerate()
                .find(|(frame_idx, value)| *frame_idx > start_frame && value.unwrap_or(false))
                .map(|(frame_idx, _)| frame_idx)
        });
        let reward_arrival_seconds =
            reward_arrival_frame.map(|frame| timing.frame_to_elapsed(frame));

        let seconds_in_reward_area =
            timing.bool_frames_to_seconds(&chunked_to_vec(&reward_union_mask));
        let reward_entries = count_bouts(&reward_union_mask);

        let search_path_distance_meters = match (start_area_exit_frame, reward_arrival_frame) {
            (Some(start_frame), Some(reward_frame)) => {
                reward_frame.checked_sub(1).and_then(|end_frame| {
                    path_distance_between_frames(
                        &df,
                        "mouse_center_x",
                        "mouse_center_y",
                        start_frame,
                        end_frame,
                    )
                })
            }
            _ => None,
        };

        let search_duration_seconds = match (start_area_exit_seconds, reward_arrival_seconds) {
            (Some(start_seconds), Some(reward_seconds)) => {
                Some((reward_seconds - start_seconds).max(0.0))
            }
            _ => None,
        };

        let search_mean_speed_mps = match (search_path_distance_meters, search_duration_seconds) {
            (Some(distance), Some(duration)) if duration > 0.0 => Some(distance / duration),
            _ => None,
        };

        let search_direct_distance_meters = match (start_area_exit_frame, reward_arrival_frame) {
            (Some(start_frame), Some(reward_frame)) => {
                reward_frame.checked_sub(1).and_then(|end_frame| {
                    coordinate_at(&df, "mouse_center_x", "mouse_center_y", start_frame)
                        .zip(coordinate_at(
                            &df,
                            "mouse_center_x",
                            "mouse_center_y",
                            end_frame,
                        ))
                        .map(|((start_x, start_y), (end_x, end_y))| {
                            let dx = end_x - start_x;
                            let dy = end_y - start_y;
                            (dx * dx + dy * dy).sqrt()
                        })
                })
            }
            _ => None,
        };

        let search_tortuosity = match (search_path_distance_meters, search_direct_distance_meters) {
            (Some(path), Some(direct)) if direct > 0.0 => Some(path / direct),
            _ => None,
        };

        let total_distance_meters = path_distance(&df, "mouse_center_x", "mouse_center_y", None);
        let total_duration_seconds = if timing.fps > 0.0 {
            df.height() as f64 / timing.fps
        } else {
            0.0
        };
        let total_mean_speed_mps = if total_duration_seconds > 0.0 {
            total_distance_meters / total_duration_seconds
        } else {
            0.0
        };

        let seconds_to_find_reward = match (start_area_exit_seconds, reward_arrival_seconds) {
            (Some(start_seconds), Some(reward_seconds)) if reward_seconds >= start_seconds => {
                Some(reward_seconds - start_seconds)
            }
            _ => None,
        };

        CheeseboardResult {
            enclosure_label: self.enclosure.label.clone(),
            reward_labels: self
                .reward_locations
                .iter()
                .map(|reward| reward.label.clone())
                .collect(),
            enclosure_summary,
            reward_summaries: {
                reward_summaries.shrink_to_fit();
                reward_summaries
            },
            start_area_exit_frame,
            start_area_exit_seconds,
            reward_arrival_frame,
            reward_arrival_seconds,
            seconds_to_find_reward,
            seconds_in_reward_area,
            reward_entries,
            search_path_distance_meters,
            start_to_reward_displacement_meters: search_path_distance_meters,
            search_direct_distance_meters,
            search_mean_speed_mps,
            search_tortuosity,
            total_distance_meters,
            total_mean_speed_mps,
            total_duration_seconds,
            evaluated_df: Some(df),
        }
    }
}
