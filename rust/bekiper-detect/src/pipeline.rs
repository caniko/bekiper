use std::collections::VecDeque;

use image::GrayImage;
use tracing::info;

use crate::config::DetectionPipelineConfig;

/// A detected blob with its spatial properties.
#[derive(Debug, Clone)]
pub struct DetectedBlob {
    /// Pixel area of the blob.
    pub area: u32,
    /// Bounding box: (x, y, width, height).
    pub bbox: (u32, u32, u32, u32),
    /// Centroid (cx, cy) in pixel coordinates.
    pub centroid: (f64, f64),
    /// The binary mask for this blob (same dimensions as input image).
    pub mask: GrayImage,
}

/// Debug artifacts from the detection pipeline for failure analysis.
#[derive(Debug, Clone)]
pub struct DetectionDebug {
    /// Binary threshold image (before mouse removal).
    pub threshold_image: GrayImage,
    /// Binary image after mouse removal (final mask used for blob extraction).
    pub post_mouse_image: GrayImage,
    /// All candidate blobs before the expected_count truncation.
    pub all_candidates: Vec<DetectedBlob>,
}

/// Result of the detection pipeline: blobs + debug artifacts.
#[derive(Debug, Clone)]
pub struct DetectionOutput {
    pub blobs: Vec<DetectedBlob>,
    pub debug: DetectionDebug,
}

/// Run the full detection pipeline on a grayscale image.
///
/// Steps:
/// 1. Percentile-based threshold (dark objects on light background)
/// 2. Flood-fill border removal (black arena surround)
/// 3. Morphological close (fill gaps)
/// 4. Mouse removal (largest blob + dilation)
/// 5. Connected component extraction with area filtering
/// 6. Take the N smallest blobs (objects are smaller than stray detections)
pub fn detect_dark_objects(gray: &GrayImage, config: &DetectionPipelineConfig) -> DetectionOutput {
    let (w, h) = gray.dimensions();
    let img_area = (w * h) as f64;
    let diag = img_area.sqrt();

    let min_area = (img_area * config.area_filter.min_frac) as u32;
    let max_area = (img_area * config.area_filter.max_frac) as u32;
    let mouse_min_area = (img_area * config.mouse_removal.min_area_frac) as u32;

    info!(
        width = w,
        height = h,
        min_area,
        max_area,
        mouse_min_area,
        "detection parameters"
    );

    // Step 1: Percentile-based threshold
    let floor_val = percentile(gray, config.threshold.floor_percentile);
    let threshold_val = (floor_val as f64 * config.threshold.threshold_factor) as u8;
    info!(floor_val, threshold_val, "threshold values");

    let mut binary = threshold_binary_inv(gray, threshold_val);

    // Step 2: Flood-fill from corners to remove black border
    flood_fill_corners(&mut binary, w, h);

    // Step 3: Morphological close
    let close_k = (diag * config.threshold.close_kernel_frac) as u32;
    let close_k = close_k.max(3) | 1; // ensure odd and >= 3
    morphological_close(&mut binary, close_k);

    let threshold_image = binary.clone();

    // Step 4: Mouse removal
    let components = extract_components(&binary);
    let mouse_components: Vec<&Component> = components
        .iter()
        .filter(|c| c.area >= mouse_min_area)
        .collect();

    if !mouse_components.is_empty() {
        let dilate_r = (diag * config.mouse_removal.dilate_frac) as u32;
        let dilate_r = dilate_r.max(3) | 1;
        info!(
            count = mouse_components.len(),
            dilate_r, "removing mouse regions"
        );

        // Create mouse mask and dilate it
        let mut mouse_mask = GrayImage::new(w, h);
        for comp in &mouse_components {
            for &(x, y) in &comp.pixels {
                mouse_mask.put_pixel(x, y, image::Luma([255]));
            }
        }
        dilate_binary(&mut mouse_mask, dilate_r);

        // Subtract mouse from binary
        for y in 0..h {
            for x in 0..w {
                if mouse_mask.get_pixel(x, y).0[0] > 0 {
                    binary.put_pixel(x, y, image::Luma([0]));
                }
            }
        }
    }

    let post_mouse_image = binary.clone();

    // Step 5: Extract remaining components with area filtering
    let components = extract_components(&binary);
    let mut candidates: Vec<DetectedBlob> = components
        .into_iter()
        .filter(|c| c.area >= min_area && c.area <= max_area)
        .map(|c| component_to_blob(c, w, h))
        .collect();

    info!(
        count = candidates.len(),
        "candidate objects after filtering"
    );

    // Step 6: Take the N smallest (objects are smaller than stray detections)
    candidates.sort_by_key(|b| b.area);
    let all_candidates = candidates.clone();
    candidates.truncate(config.expected_count);

    DetectionOutput {
        blobs: candidates,
        debug: DetectionDebug {
            threshold_image,
            post_mouse_image,
            all_candidates,
        },
    }
}

// ---------------------------------------------------------------------------
// Image processing primitives
// ---------------------------------------------------------------------------

/// Compute the p-th percentile of pixel values (0-100).
fn percentile(img: &GrayImage, p: f64) -> u8 {
    let mut values: Vec<u8> = img.pixels().map(|px| px.0[0]).collect();
    values.sort_unstable();
    let idx = ((p / 100.0) * (values.len() - 1) as f64) as usize;
    values[idx.min(values.len() - 1)]
}

/// Binary inverse threshold: pixels <= threshold become 255, others become 0.
fn threshold_binary_inv(img: &GrayImage, threshold: u8) -> GrayImage {
    let (w, h) = img.dimensions();
    let mut out = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let v = img.get_pixel(x, y).0[0];
            out.put_pixel(x, y, image::Luma([if v <= threshold { 255 } else { 0 }]));
        }
    }
    out
}

/// Flood-fill from all four corners, setting filled pixels to 0.
fn flood_fill_corners(binary: &mut GrayImage, w: u32, h: u32) {
    let corners = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)];
    for (cx, cy) in corners {
        flood_fill(binary, cx, cy, 255, 0);
    }
}

/// BFS flood-fill: replace all connected pixels with value `from` starting at (sx, sy) with `to`.
fn flood_fill(img: &mut GrayImage, sx: u32, sy: u32, from: u8, to: u8) {
    let (w, h) = img.dimensions();
    if img.get_pixel(sx, sy).0[0] != from {
        return;
    }

    let mut queue = VecDeque::new();
    queue.push_back((sx, sy));
    img.put_pixel(sx, sy, image::Luma([to]));

    while let Some((x, y)) = queue.pop_front() {
        for (dx, dy) in [(-1i32, 0), (1, 0), (0, -1i32), (0, 1)] {
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            if nx >= 0 && nx < w as i32 && ny >= 0 && ny < h as i32 {
                let nx = nx as u32;
                let ny = ny as u32;
                if img.get_pixel(nx, ny).0[0] == from {
                    img.put_pixel(nx, ny, image::Luma([to]));
                    queue.push_back((nx, ny));
                }
            }
        }
    }
}

/// Morphological close (dilate then erode) with a circular structuring element.
fn morphological_close(binary: &mut GrayImage, kernel_size: u32) {
    dilate_binary(binary, kernel_size);
    erode_binary(binary, kernel_size);
}

/// Dilate a binary image with a circular structuring element of given diameter.
fn dilate_binary(img: &mut GrayImage, diameter: u32) {
    let (w, h) = img.dimensions();
    let r = diameter as i32 / 2;
    let r_sq = r * r;

    // Build circular structuring element offsets
    let mut offsets = Vec::new();
    for dy in -r..=r {
        for dx in -r..=r {
            if dx * dx + dy * dy <= r_sq {
                offsets.push((dx, dy));
            }
        }
    }

    let original = img.clone();
    for y in 0..h {
        for x in 0..w {
            if original.get_pixel(x, y).0[0] == 255 {
                for &(dx, dy) in &offsets {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx >= 0 && nx < w as i32 && ny >= 0 && ny < h as i32 {
                        img.put_pixel(nx as u32, ny as u32, image::Luma([255]));
                    }
                }
            }
        }
    }
}

/// Erode a binary image with a circular structuring element of given diameter.
fn erode_binary(img: &mut GrayImage, diameter: u32) {
    let (w, h) = img.dimensions();
    let r = diameter as i32 / 2;
    let r_sq = r * r;

    let mut offsets = Vec::new();
    for dy in -r..=r {
        for dx in -r..=r {
            if dx * dx + dy * dy <= r_sq {
                offsets.push((dx, dy));
            }
        }
    }

    let original = img.clone();
    for y in 0..h {
        for x in 0..w {
            if original.get_pixel(x, y).0[0] == 255 {
                let all_set = offsets.iter().all(|&(dx, dy)| {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    nx >= 0
                        && nx < w as i32
                        && ny >= 0
                        && ny < h as i32
                        && original.get_pixel(nx as u32, ny as u32).0[0] == 255
                });
                if !all_set {
                    img.put_pixel(x, y, image::Luma([0]));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Connected component extraction
// ---------------------------------------------------------------------------

/// A connected component (blob) in a binary image.
struct Component {
    area: u32,
    pixels: Vec<(u32, u32)>,
}

/// Extract all connected components (8-connected) from a binary image.
fn extract_components(binary: &GrayImage) -> Vec<Component> {
    let (w, h) = binary.dimensions();
    let mut visited = vec![false; (w * h) as usize];
    let mut components = Vec::new();

    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            if binary.get_pixel(x, y).0[0] == 255 && !visited[idx] {
                let pixels = bfs_component(binary, &mut visited, x, y, w, h);
                let area = pixels.len() as u32;
                components.push(Component { area, pixels });
            }
        }
    }

    components
}

/// BFS to collect all pixels of a connected component.
fn bfs_component(
    binary: &GrayImage,
    visited: &mut [bool],
    sx: u32,
    sy: u32,
    w: u32,
    h: u32,
) -> Vec<(u32, u32)> {
    let mut queue = VecDeque::new();
    let mut pixels = Vec::new();

    let idx = (sy * w + sx) as usize;
    visited[idx] = true;
    queue.push_back((sx, sy));

    while let Some((x, y)) = queue.pop_front() {
        pixels.push((x, y));

        // 8-connected neighbors
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx >= 0 && nx < w as i32 && ny >= 0 && ny < h as i32 {
                    let nx = nx as u32;
                    let ny = ny as u32;
                    let nidx = (ny * w + nx) as usize;
                    if !visited[nidx] && binary.get_pixel(nx, ny).0[0] == 255 {
                        visited[nidx] = true;
                        queue.push_back((nx, ny));
                    }
                }
            }
        }
    }

    pixels
}

/// Convert a Component into a DetectedBlob with computed properties.
fn component_to_blob(comp: Component, w: u32, h: u32) -> DetectedBlob {
    let mut min_x = u32::MAX;
    let mut min_y = u32::MAX;
    let mut max_x = 0u32;
    let mut max_y = 0u32;
    let mut sum_x = 0u64;
    let mut sum_y = 0u64;

    for &(x, y) in &comp.pixels {
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
        sum_x += x as u64;
        sum_y += y as u64;
    }

    let n = comp.pixels.len() as f64;
    let cx = sum_x as f64 / n;
    let cy = sum_y as f64 / n;

    let mut mask = GrayImage::new(w, h);
    for &(x, y) in &comp.pixels {
        mask.put_pixel(x, y, image::Luma([255]));
    }

    DetectedBlob {
        area: comp.area,
        bbox: (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1),
        centroid: (cx, cy),
        mask,
    }
}

#[cfg(test)]
#[path = "pipeline/tests.rs"]
mod tests;
