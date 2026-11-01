use bekiper_core::inspection::PerimeterSpec;
use image::GrayImage;
use serde::Serialize;
use thiserror::Error;

use crate::config::{ArenaCalibrationConfig, ArenaCenterRule, DEFAULT_VIDEO_CENTER_FRACTION};

#[derive(Debug, Clone, Copy, Serialize)]
pub struct ArenaCandidate {
    pub center_x: f64,
    pub center_y: f64,
    pub width: f64,
    pub height: f64,
    pub edge_score: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ArenaCalibrationDebug {
    pub candidates: Vec<ArenaCandidate>,
    pub messages: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ArenaCalibrationResult {
    pub perimeters: Vec<PerimeterSpec>,
    pub debug: ArenaCalibrationDebug,
}

#[derive(Debug, Error)]
pub enum ArenaCalibrationError {
    #[error("arena calibration requires at least one sampled frame")]
    NoFrames,
    #[error("arena calibration failed: {0}")]
    Invalid(String),
}

pub fn calibrate_arena(
    frames: &[GrayImage],
    arena_side_meters: f64,
    config: &ArenaCalibrationConfig,
) -> Result<ArenaCalibrationResult, ArenaCalibrationError> {
    if !arena_side_meters.is_finite() || arena_side_meters <= 0.0 {
        return Err(ArenaCalibrationError::Invalid(
            "arena_side meters must be finite and positive".into(),
        ));
    }
    let mut result = detect_arena_boundary(frames, config)?;
    if let PerimeterSpec::Rectangle { width, height, .. } = result.perimeters[0]
        && let Some(center) = center_zone_for_arena(
            &result.perimeters[0],
            arena_side_meters / width.min(height),
            &config.effective_center_rule(),
        )?
    {
        result.perimeters.push(center);
    }
    Ok(result)
}

/// Generate an explicitly configured center inside an axis-aligned floor rectangle.
/// Manual mode emits nothing. Existing center outlines are retained by the caller.
/// Video-relative mode uses only pixel geometry; physical rules require a scale.
pub fn center_zone_for_arena(
    arena: &PerimeterSpec,
    meters_per_pixel: impl Into<Option<f64>>,
    rule: &ArenaCenterRule,
) -> Result<Option<PerimeterSpec>, ArenaCalibrationError> {
    let scale = meters_per_pixel.into();
    center_zone_for_arena_xy(arena, scale.map(|mpp| (mpp, mpp)), rule)
}

/// [`center_zone_for_arena`] with separate horizontal/vertical metres/pixel
/// for sources whose pixels are not square.
pub fn center_zone_for_arena_xy(
    arena: &PerimeterSpec,
    meters_per_pixel: Option<(f64, f64)>,
    rule: &ArenaCenterRule,
) -> Result<Option<PerimeterSpec>, ArenaCalibrationError> {
    if matches!(rule, ArenaCenterRule::Manual {}) {
        return Ok(None);
    }
    let invalid = |message: &str| ArenaCalibrationError::Invalid(message.into());
    let PerimeterSpec::Rectangle {
        center_x,
        center_y,
        width,
        height,
        ..
    } = arena
    else {
        return Err(invalid(
            "center generation requires an axis-aligned arena; draw and review the center inside the existing polygon manually",
        ));
    };
    let physical_scale = || {
        meters_per_pixel
            .filter(|(x, y)| [x, y].iter().all(|v| v.is_finite() && **v > 0.0))
            .ok_or_else(|| {
                invalid("physical center generation requires finite positive metres/pixel")
            })
    };
    let (center_w, center_h) = match *rule {
        ArenaCenterRule::Manual {} => return Ok(None), // coverage: unreachable — Manual returned above
        ArenaCenterRule::Dimensions {
            width_meters,
            height_meters,
        }
        | ArenaCenterRule::FrameCentered {
            center_width_meters: width_meters,
            center_height_meters: height_meters,
            ..
        } => {
            let (scale_x, scale_y) = physical_scale()?;
            (width_meters / scale_x, height_meters / scale_y)
        }
        ArenaCenterRule::WallInset { inset_meters } => {
            let (scale_x, scale_y) = physical_scale()?;
            if !inset_meters.is_finite() || inset_meters <= 0.0 {
                return Err(invalid("wall inset must be finite positive metres"));
            }
            (
                *width - 2.0 * inset_meters / scale_x,
                *height - 2.0 * inset_meters / scale_y,
            )
        }
        ArenaCenterRule::VideoRelative {
            width_fraction,
            height_fraction,
        } => (
            *width * width_fraction.unwrap_or(DEFAULT_VIDEO_CENTER_FRACTION),
            *height * height_fraction.unwrap_or(DEFAULT_VIDEO_CENTER_FRACTION),
        ),
    };
    if ![center_w, center_h, *width, *height, *center_x, *center_y]
        .iter()
        .all(|v| v.is_finite())
        || center_w <= 0.0
        || center_h <= 0.0
        || center_w >= *width
        || center_h >= *height
    {
        return Err(invalid(
            "center rule must fit strictly inside arena and leave a periphery",
        ));
    }
    Ok(Some(PerimeterSpec::Rectangle {
        label: "center_zone".into(),
        center_x: *center_x,
        center_y: *center_y,
        width: center_w,
        height: center_h,
    }))
}

/// Pixel `arena` and `center_zone` rectangles for a [`ArenaCenterRule::FrameCentered`]
/// rule, both centered on the video frame center. Other rules generate nothing.
pub fn frame_centered_geometry(
    rule: &ArenaCenterRule,
    frame: (u32, u32),
    meters_per_pixel: f64,
) -> Result<Vec<PerimeterSpec>, ArenaCalibrationError> {
    let ArenaCenterRule::FrameCentered {
        arena_width_meters,
        arena_height_meters,
        ..
    } = *rule
    else {
        return Ok(Vec::new());
    };
    let invalid = |message: String| ArenaCalibrationError::Invalid(message);
    if !meters_per_pixel.is_finite() || meters_per_pixel <= 0.0 || frame.0 == 0 || frame.1 == 0 {
        return Err(invalid(
            "frame-centered geometry requires a frame size and finite positive metres/pixel".into(),
        ));
    }
    let (frame_w, frame_h) = (f64::from(frame.0), f64::from(frame.1));
    let (width, height) = (
        arena_width_meters / meters_per_pixel,
        arena_height_meters / meters_per_pixel,
    );
    // A larger arena than the frame means the scale or frame size is wrong.
    if width > frame_w || height > frame_h {
        return Err(invalid(format!(
            "frame-centered arena ({width:.1} x {height:.1} px) does not fit the {}x{} frame; check the scale",
            frame.0, frame.1
        )));
    }
    let arena = PerimeterSpec::Rectangle {
        label: "arena".into(),
        center_x: frame_w / 2.0,
        center_y: frame_h / 2.0,
        width,
        height,
    };
    let center = center_zone_for_arena(&arena, meters_per_pixel, rule)?
        .expect("frame-centered rule always yields a center");
    Ok(vec![arena, center])
}

/// Arena geometry generated for a [`ArenaCenterRule::FrameCentered`] rule.
#[derive(Debug, Clone)]
pub struct GeneratedArena {
    /// Pixel `arena` and `center_zone` rectangles.
    pub perimeters: Vec<PerimeterSpec>,
    /// Horizontal metres/pixel for this video: the detected box's when found.
    pub meters_per_pixel: f64,
    /// Vertical metres/pixel; differs from `meters_per_pixel` when the
    /// detected box is not square in pixels.
    pub meters_per_pixel_y: f64,
    /// `None` for a detected box; otherwise why the frame center was used.
    pub fallback_reason: Option<String>,
}

/// With `detect_box`, the arena is the box detected in `frames` (its short
/// side defines this video's scale and the center zone sits on its center).
/// Without it, or when no box is found, both rectangles are centered on the
/// frame using `fallback_meters_per_pixel`. Other rules generate nothing.
pub fn generate_arena(
    rule: &ArenaCenterRule,
    calibration: &ArenaCalibrationConfig,
    frame: (u32, u32),
    frames: &[GrayImage],
    fallback_meters_per_pixel: Option<f64>,
) -> Result<Option<GeneratedArena>, ArenaCalibrationError> {
    let ArenaCenterRule::FrameCentered {
        arena_width_meters,
        arena_height_meters,
        detect_box,
        ..
    } = *rule
    else {
        return Ok(None);
    };
    let fallback_reason = if detect_box {
        match detect_arena_boundary(frames, calibration) {
            Ok(detected) => {
                let arena = detected.perimeters[0].clone();
                let PerimeterSpec::Rectangle { width, height, .. } = arena else {
                    unreachable!("box detection yields an axis-aligned rectangle") // coverage: unreachable — detect_arena_boundary builds only a rectangle
                };
                // Each axis has its own scale: non-square source pixels
                // (e.g. a 204 x 210 px box) stay physically correct.
                let (meters_per_pixel, meters_per_pixel_y) =
                    (arena_width_meters / width, arena_height_meters / height);
                let center = center_zone_for_arena_xy(
                    &arena,
                    Some((meters_per_pixel, meters_per_pixel_y)),
                    rule,
                )?
                .expect("frame-centered rule always yields a center");
                return Ok(Some(GeneratedArena {
                    perimeters: vec![arena, center],
                    meters_per_pixel,
                    meters_per_pixel_y,
                    fallback_reason: None,
                }));
            }
            Err(error) => Some(format!("no box detected: {error}")),
        }
    } else {
        Some("box detection disabled".into())
    };
    let meters_per_pixel = fallback_meters_per_pixel.ok_or_else(|| {
        ArenaCalibrationError::Invalid(format!(
            "{}; the frame-center fallback needs the recording's scale",
            fallback_reason.as_deref().unwrap_or_default()
        ))
    })?;
    Ok(Some(GeneratedArena {
        perimeters: frame_centered_geometry(rule, frame, meters_per_pixel)?,
        meters_per_pixel,
        meters_per_pixel_y: meters_per_pixel,
        fallback_reason,
    }))
}

/// Estimate only the image-space boundary. Physical calibration and a virtual
/// center convention belong to the caller; edge detection cannot infer either.
pub fn detect_arena_boundary(
    frames: &[GrayImage],
    config: &ArenaCalibrationConfig,
) -> Result<ArenaCalibrationResult, ArenaCalibrationError> {
    config.validate().map_err(ArenaCalibrationError::Invalid)?;
    if frames.is_empty() {
        return Err(ArenaCalibrationError::NoFrames);
    }
    if frames
        .iter()
        .any(|frame| frame.dimensions() != frames[0].dimensions())
    {
        return Err(ArenaCalibrationError::Invalid(
            "sampled frame dimensions differ".into(),
        ));
    }

    let mut candidates = Vec::new();
    let mut messages = Vec::new();
    for (idx, frame) in frames.iter().enumerate() {
        match detect_candidate(frame, config) {
            Ok(candidate) => candidates.push(candidate),
            Err(error) => messages.push(format!("frame {idx}: {error}")),
        }
    }
    // Every frame yields a candidate or a message, and frames is non-empty.
    if !messages.is_empty() {
        return Err(ArenaCalibrationError::Invalid(messages.join("; ")));
    }

    let base_w = frames[0].width() as f64;
    let base_h = frames[0].height() as f64;
    validate_consensus(&candidates, base_w, base_h, config)?;

    let center_x = median(candidates.iter().map(|c| c.center_x).collect());
    let center_y = median(candidates.iter().map(|c| c.center_y).collect());
    let width = median(candidates.iter().map(|c| c.width).collect());
    let height = median(candidates.iter().map(|c| c.height).collect());

    let perimeters = vec![PerimeterSpec::Rectangle {
        label: "arena".into(),
        center_x,
        center_y,
        width,
        height,
    }];

    Ok(ArenaCalibrationResult {
        perimeters,
        debug: ArenaCalibrationDebug {
            candidates,
            messages,
        },
    })
}

fn detect_candidate(
    image: &GrayImage,
    config: &ArenaCalibrationConfig,
) -> Result<ArenaCandidate, ArenaCalibrationError> {
    let (w, h) = image.dimensions();
    if w < 16 || h < 16 {
        return Err(ArenaCalibrationError::Invalid(
            "frame is too small for arena calibration".into(),
        ));
    }

    let mut x_projection = vec![0.0; w as usize];
    let mut y_projection = vec![0.0; h as usize];
    for y in 1..h {
        for x in 1..w {
            let v = image.get_pixel(x, y).0[0] as f64;
            let left = image.get_pixel(x - 1, y).0[0] as f64;
            let up = image.get_pixel(x, y - 1).0[0] as f64;
            x_projection[x as usize] += (v - left).abs();
            y_projection[y as usize] += (v - up).abs();
        }
    }

    smooth_projection(&mut x_projection, smoothing_radius(w, config));
    smooth_projection(&mut y_projection, smoothing_radius(h, config));

    let x_margin = ((w as f64) * config.search_margin_fraction)
        .round()
        .clamp(1.0, (w / 3).max(1) as f64) as usize;
    let y_margin = ((h as f64) * config.search_margin_fraction)
        .round()
        .clamp(1.0, (h / 3).max(1) as f64) as usize;
    let min_side = (w.min(h) as f64) * config.min_side_fraction;

    let left = strongest_index(&x_projection, x_margin, (w as usize / 2).max(x_margin + 1))?;
    let right = strongest_index(
        &x_projection,
        (w as usize / 2).min(w as usize - x_margin - 1),
        w as usize - x_margin,
    )?; // coverage: unreachable — the margins keep every search range non-empty
    let top = strongest_index(&y_projection, y_margin, (h as usize / 2).max(y_margin + 1))?;
    let bottom = strongest_index(
        &y_projection,
        (h as usize / 2).min(h as usize - y_margin - 1),
        h as usize - y_margin,
    )?; // coverage: unreachable — the margins keep every search range non-empty

    let width = right as f64 - left as f64;
    let height = bottom as f64 - top as f64;
    if width < min_side || height < min_side {
        return Err(ArenaCalibrationError::Invalid(format!(
            "detected arena too small: {width:.1}x{height:.1}px"
        )));
    }
    let aspect_error = (width / height - 1.0).abs();
    if aspect_error > config.max_aspect_error {
        return Err(ArenaCalibrationError::Invalid(format!(
            "arena aspect ratio rejected: width={width:.1}, height={height:.1}, max_error={}",
            config.max_aspect_error
        )));
    }

    let mean_x = mean(&x_projection);
    let mean_y = mean(&y_projection);
    let edge_score = [
        x_projection[left],
        x_projection[right],
        y_projection[top],
        y_projection[bottom],
    ]
    .into_iter()
    .fold(f64::INFINITY, f64::min)
        / mean_x.max(mean_y).max(f64::EPSILON);
    if edge_score < config.min_edge_score {
        return Err(ArenaCalibrationError::Invalid(format!(
            "arena edge score {edge_score:.2} below minimum {}",
            config.min_edge_score
        )));
    }

    Ok(ArenaCandidate {
        center_x: (left as f64 + right as f64) / 2.0,
        center_y: (top as f64 + bottom as f64) / 2.0,
        width,
        height,
        edge_score,
    })
}

fn validate_consensus(
    candidates: &[ArenaCandidate],
    image_w: f64,
    image_h: f64,
    config: &ArenaCalibrationConfig,
) -> Result<(), ArenaCalibrationError> {
    let center_limit = image_w.min(image_h) * config.max_center_shift_fraction;
    let side_limit = image_w.min(image_h) * config.max_side_shift_fraction;
    let center_x = median(candidates.iter().map(|c| c.center_x).collect());
    let center_y = median(candidates.iter().map(|c| c.center_y).collect());
    let width = median(candidates.iter().map(|c| c.width).collect());
    let height = median(candidates.iter().map(|c| c.height).collect());

    for candidate in candidates {
        let center_shift = ((candidate.center_x - center_x).powi(2)
            + (candidate.center_y - center_y).powi(2))
        .sqrt();
        let side_shift = (candidate.width - width)
            .abs()
            .max((candidate.height - height).abs());
        if center_shift > center_limit {
            return Err(ArenaCalibrationError::Invalid(format!(
                "sampled frames disagree on arena center by {center_shift:.1}px; limit {center_limit:.1}px"
            )));
        }
        if side_shift > side_limit {
            return Err(ArenaCalibrationError::Invalid(format!(
                "sampled frames disagree on arena side by {side_shift:.1}px; limit {side_limit:.1}px"
            )));
        }
    }

    Ok(())
}

fn smoothing_radius(axis_len: u32, config: &ArenaCalibrationConfig) -> usize {
    ((axis_len as f64) * config.smoothing_fraction)
        .round()
        .max(1.0) as usize
}

fn smooth_projection(values: &mut [f64], radius: usize) {
    let original = values.to_vec();
    for (idx, value) in values.iter_mut().enumerate() {
        let start = idx.saturating_sub(radius);
        let end = (idx + radius + 1).min(original.len());
        *value = original[start..end].iter().sum::<f64>() / (end - start) as f64;
    }
}

fn strongest_index(
    values: &[f64],
    start: usize,
    end: usize,
) -> Result<usize, ArenaCalibrationError> {
    values
        .get(start..end)
        .and_then(|slice| {
            slice
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(idx, _)| start + idx)
        })
        .ok_or_else(|| ArenaCalibrationError::Invalid("empty arena edge search range".into()))
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len().max(1) as f64
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

#[cfg(test)]
#[path = "arena/tests.rs"]
mod tests;
