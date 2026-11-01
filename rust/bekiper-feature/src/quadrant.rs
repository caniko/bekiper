//! Quadrant (NxN grid) analysis: auto-generate rectangular zones from an arena
//! and compute per-zone summaries using the existing `zone_summary` infrastructure.

use polars::prelude::*;

use bekiper_core::types::CoordinateColumns;
use bekiper_perimeter::perimeter::Perimeter;
use bekiper_perimeter::rectangle::Rectangle;

use crate::zone::{ZoneSummary, zone_summary};

/// Parameters for quadrant analysis.
pub struct QuadrantParams<'a> {
    pub arena_cx: f64,
    pub arena_cy: f64,
    pub arena_w: f64,
    pub arena_h: f64,
    pub rows: usize,
    pub cols: usize,
    pub coords: &'a CoordinateColumns,
    pub fps: f64,
    pub min_dwell_frames: u32,
}

/// Generate an NxN grid of rectangular zones from an arena's bounding box.
///
/// Zones are labeled `"q_{row}_{col}"` (0-indexed, row 0 = top).
/// Returns a list of `(label, Perimeter<Rectangle>)` pairs.
pub fn generate_grid_zones(
    arena_cx: f64,
    arena_cy: f64,
    arena_w: f64,
    arena_h: f64,
    rows: usize,
    cols: usize,
) -> Vec<(String, Perimeter<Rectangle>)> {
    let cell_w = arena_w / cols as f64;
    let cell_h = arena_h / rows as f64;
    let origin_x = arena_cx - arena_w / 2.0;
    let origin_y = arena_cy - arena_h / 2.0;

    let mut zones = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            let cx = origin_x + (c as f64 + 0.5) * cell_w;
            let cy = origin_y + (r as f64 + 0.5) * cell_h;
            let label = format!("q_{r}_{c}");
            zones.push((
                label.clone(),
                Perimeter::new(label, Rectangle::new(cx, cy, cell_w, cell_h)),
            ));
        }
    }
    zones
}

/// Add boolean confinement columns for each grid zone to a LazyFrame.
///
/// Each zone gets a column named by its label, containing `true` where
/// the coordinate pair falls inside that zone.
pub fn add_grid_zone_columns(
    lf: LazyFrame,
    zones: &[(String, Perimeter<Rectangle>)],
    coords: &CoordinateColumns,
) -> LazyFrame {
    let exprs: Vec<Expr> = zones
        .iter()
        .map(|(label, perimeter)| perimeter.confinement_mask(coords).alias(label))
        .collect();
    if exprs.is_empty() {
        return lf;
    }
    lf.with_columns(exprs)
}

/// Run full quadrant analysis: generate zones, add columns, compute summaries.
///
/// Returns per-zone `ZoneSummary` for the given NxN grid over the arena.
pub fn quadrant_analysis(df: &DataFrame, params: &QuadrantParams) -> Vec<ZoneSummary> {
    let zones = generate_grid_zones(
        params.arena_cx,
        params.arena_cy,
        params.arena_w,
        params.arena_h,
        params.rows,
        params.cols,
    );

    // Add zone columns to the DataFrame
    let lf = add_grid_zone_columns(df.clone().lazy(), &zones, params.coords);
    let df_with_zones = lf.collect().expect("quadrant_analysis: collect failed");

    // Compute per-zone summaries
    zones
        .iter()
        .map(|(label, _)| {
            zone_summary(
                &df_with_zones,
                label,
                &params.coords.x,
                &params.coords.y,
                params.fps,
                params.min_dwell_frames,
            )
        })
        .collect()
}
