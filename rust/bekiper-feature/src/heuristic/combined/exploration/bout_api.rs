/// Extract per-object exploration results from a collected DataFrame.
///
/// Returns (exploration_seconds, bout_count, toleranced_mask) for a given object
/// after temporal tolerance (gap bridging + minimum bout filtering).
pub fn object_exploration(
    df: &DataFrame,
    raw_col: &str,
    fps: f64,
    min_bout_seconds: f64,
    max_gap_seconds: f64,
    gap_density_threshold: f64,
) -> (f64, u32, BooleanChunked) {
    object_exploration_with_bout_filters(
        df,
        raw_col,
        fps,
        min_bout_seconds,
        max_gap_seconds,
        gap_density_threshold,
        BoutFilterSpec::unfiltered(),
    )
}

#[derive(Debug, Clone, Copy)]
pub struct BoutFilterSpec<'a> {
    pub strict_boundaries: bool,
    pub speed_mode: BoutFilterMode,
    pub angle_mode: BoutFilterMode,
    pub max_nose_speed: f64,
    pub max_angle_deg: f64,
    pub nose_speed_mps: Option<&'a Float64Chunked>,
    pub angle_deg: Option<&'a Float64Chunked>,
    /// Per-frame nose-contact mask. When provided, a bout that would
    /// otherwise be dropped by the per-bout-median angle filter is
    /// rescued if at least one of its frames had `nose_contact == true`
    /// (the spec's "nose contact may count as exploration even if the
    /// head-orientation angle briefly exceeds 45°" rule).
    pub nose_contact: Option<&'a BooleanChunked>,
}

impl BoutFilterSpec<'_> {
    pub fn unfiltered() -> Self {
        Self {
            strict_boundaries: false,
            speed_mode: BoutFilterMode::Disabled,
            angle_mode: BoutFilterMode::Disabled,
            max_nose_speed: DEFAULT_MAX_NOSE_SPEED,
            max_angle_deg: DEFAULT_MAX_ANGLE_DEG,
            nose_speed_mps: None,
            angle_deg: None,
            nose_contact: None,
        }
    }
}

/// Extract per-object exploration results, then optionally apply final
/// bout-level median acceptance filters to the post-tolerance bouts.
pub fn object_exploration_with_bout_filters(
    df: &DataFrame,
    raw_col: &str,
    fps: f64,
    min_bout_seconds: f64,
    max_gap_seconds: f64,
    gap_density_threshold: f64,
    filters: BoutFilterSpec<'_>,
) -> (f64, u32, BooleanChunked) {
    let timing = BoutTiming::cfr(fps, df.height()).expect("valid exploration FPS");
    object_exploration_with_timing(
        df,
        raw_col,
        &timing,
        min_bout_seconds,
        max_gap_seconds,
        gap_density_threshold,
        filters,
    )
}

/// Timestamp-aware native scoring; CFR callers retain their existing adapter.
pub fn object_exploration_with_timing(
    df: &DataFrame,
    raw_col: &str,
    timing: &BoutTiming,
    min_bout_seconds: f64,
    max_gap_seconds: f64,
    gap_density_threshold: f64,
    filters: BoutFilterSpec<'_>,
) -> (f64, u32, BooleanChunked) {
    if let Ok(series) = df.column(raw_col)
        && let Ok(bools) = series.bool()
    {
        let filtered = filter_selected_bouts(
            bools,
            select_exploration_bouts_for_column(
                df,
                raw_col,
                bools,
                timing,
                min_bout_seconds,
                (max_gap_seconds, filters.strict_boundaries),
                gap_density_threshold,
            ),
            filters,
        )
        .0;
        let values: Vec<_> = filtered.into_iter().map(|v| v.unwrap_or(false)).collect();
        let seconds = timing.mask_seconds(&values);
        let bouts = count_bouts(&filtered);
        (seconds, bouts, filtered)
    } else {
        let empty = BooleanChunked::from_slice(raw_col.into(), &vec![false; df.height()]);
        (0.0, 0, empty)
    }
}

pub fn apply_bout_level_filters(
    raw: &BooleanChunked,
    min_bout_frames: u32,
    max_gap_frames: u32,
    gap_density_threshold: f64,
    filters: BoutFilterSpec<'_>,
) -> BooleanChunked {
    apply_bout_level_filters_with_diagnostics(
        raw,
        min_bout_frames,
        max_gap_frames,
        gap_density_threshold,
        filters,
    )
    .0
}

pub fn apply_bout_level_filters_with_diagnostics(
    raw: &BooleanChunked,
    min_bout_frames: u32,
    max_gap_frames: u32,
    gap_density_threshold: f64,
    filters: BoutFilterSpec<'_>,
) -> (BooleanChunked, BoutFilterDiagnostics) {
    let selected = bekiper_math::discrete::select_series_bouts(
        raw,
        min_bout_frames,
        max_gap_frames,
        gap_density_threshold,
    );
    filter_selected_bouts(raw, selected, filters)
}
