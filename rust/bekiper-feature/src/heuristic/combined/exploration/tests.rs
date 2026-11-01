use super::*;
use bekiper_math::discrete::apply_tolerance_series_with_density;

fn test_filter_spec<'a>(
    speed_mode: BoutFilterMode,
    angle_mode: BoutFilterMode,
    nose_speed_mps: Option<&'a Float64Chunked>,
    angle_deg: Option<&'a Float64Chunked>,
    nose_contact: Option<&'a BooleanChunked>,
) -> BoutFilterSpec<'a> {
    BoutFilterSpec {
        strict_boundaries: false,
        speed_mode,
        angle_mode,
        max_nose_speed: 0.05,
        max_angle_deg: 45.0,
        nose_speed_mps,
        angle_deg,
        nose_contact,
    }
}

fn bools(values: &[bool]) -> BooleanChunked {
    BooleanChunked::from_slice("raw".into(), values)
}

fn floats(values: &[f64]) -> Float64Chunked {
    Float64Chunked::from_slice("metric".into(), values)
}

fn filtered_count(
    raw: &[bool],
    speeds: &[f64],
    angles: &[f64],
    speed_mode: BoutFilterMode,
    angle_mode: BoutFilterMode,
    min_bout_frames: u32,
    max_gap_frames: u32,
) -> u64 {
    let raw = bools(raw);
    let speeds = floats(speeds);
    let angles = floats(angles);
    let filtered = apply_bout_level_filters(
        &raw,
        min_bout_frames,
        max_gap_frames,
        0.0,
        BoutFilterSpec {
            strict_boundaries: false,
            speed_mode,
            angle_mode,
            max_nose_speed: 0.05,
            max_angle_deg: 45.0,
            nose_speed_mps: Some(&speeds),
            angle_deg: Some(&angles),
            nose_contact: None,
        },
    );
    filtered.sum().unwrap_or(0) as u64
}

include!("tests/speed_filters.rs");
include!("tests/timing.rs");
include!("tests/angle_filters.rs");
include!("tests/contact_rescue.rs");
include!("tests/diagnostics.rs");
