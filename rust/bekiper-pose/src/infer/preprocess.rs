use image::DynamicImage;
use image::imageops::FilterType;
use ndarray::Array4;
use rayon::prelude::*;

use super::spec::{ModelSpec, Normalization};

/// A rectangular crop region in pixel coordinates.
///
/// When set, frames are cropped to this region before resize+normalize.
/// After inference, predicted keypoints must be offset by `(x, y)` to
/// recover original full-frame pixel coordinates.
#[derive(Debug, Clone, Copy)]
pub struct CropRegion {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl CropRegion {
    /// Create a crop region from a bounding box with padding, clamped to frame dimensions.
    pub fn from_bbox(
        min_x: f64,
        min_y: f64,
        max_x: f64,
        max_y: f64,
        padding: f64,
        frame_width: u32,
        frame_height: u32,
    ) -> Self {
        let x = ((min_x - padding).max(0.0)) as u32;
        let y = ((min_y - padding).max(0.0)) as u32;
        let x2 = ((max_x + padding).min(frame_width as f64)) as u32;
        let y2 = ((max_y + padding).min(frame_height as f64)) as u32;
        Self {
            x,
            y,
            width: x2.saturating_sub(x),
            height: y2.saturating_sub(y),
        }
    }

    /// Crop a `DynamicImage` to this region.
    pub fn crop(&self, image: &DynamicImage) -> DynamicImage {
        image.crop_imm(self.x, self.y, self.width, self.height)
    }
}

/// Preprocess a frame into a model-ready f32 tensor.
///
/// Performs: resize → normalize → reshape to [1, C, H, W] or [1, H, W, C].
///
/// Returns the tensor and the (original_width, original_height) for rescaling
/// predicted coordinates back to the original frame space.
pub fn preprocess_frame(image: &DynamicImage, spec: &ModelSpec) -> (Array4<f32>, (u32, u32)) {
    let original_size = (image.width(), image.height());

    let rgb = ensure_model_size(image, spec);

    let h = spec.input_height as usize;
    let w = spec.input_width as usize;
    let c = spec.input_channels as usize;

    let mut tensor = if spec.channels_first {
        Array4::<f32>::zeros((1, c, h, w))
    } else {
        Array4::<f32>::zeros((1, h, w, c))
    };

    let mut frame_slice = tensor.slice_mut(ndarray::s![0, .., .., ..]);
    fill_frame_slice(&mut frame_slice, rgb.as_raw(), w, c, spec);

    (tensor, original_size)
}

/// Preprocess a batch of frames into a single [N, C, H, W] or [N, H, W, C] tensor.
///
/// If frames are already at model dimensions (e.g. via `VideoDecoder::open_with_resize`),
/// the expensive CPU resize is skipped entirely — only normalization remains.
/// Allocates one batch tensor upfront and fills each frame's slice in parallel.
pub fn preprocess_batch(
    images: &[DynamicImage],
    spec: &ModelSpec,
) -> (Array4<f32>, Vec<(u32, u32)>) {
    let n = images.len();
    let h = spec.input_height as usize;
    let w = spec.input_width as usize;
    let c = spec.input_channels as usize;

    // Single allocation for the entire batch
    let mut batch = if spec.channels_first {
        Array4::<f32>::zeros((n, c, h, w))
    } else {
        Array4::<f32>::zeros((n, h, w, c))
    };

    // Resize only if needed, in parallel (this is a no-op if frames are model-sized)
    let prepared: Vec<_> = images
        .par_iter()
        .map(|image| {
            let orig = (image.width(), image.height());
            let rgb = ensure_model_size(image, spec);
            (rgb, orig)
        })
        .collect();

    let original_sizes: Vec<(u32, u32)> = prepared.iter().map(|(_, orig)| *orig).collect();

    // Fill the batch tensor — split into mutable slices for parallel write
    let frame_slices: Vec<_> = batch.outer_iter_mut().collect();

    frame_slices
        .into_par_iter()
        .zip(prepared.par_iter())
        .for_each(|(mut frame_slice, (rgb, _))| {
            fill_frame_slice(&mut frame_slice, rgb.as_raw(), w, c, spec);
        });

    (batch, original_sizes)
}

/// Resize to model dimensions only if the image isn't already the right size.
/// When using `VideoDecoder::open_with_resize`, frames arrive pre-sized and
/// this becomes a cheap `to_rgb8()` (often a no-op for RGB8 images).
#[inline]
fn ensure_model_size(image: &DynamicImage, spec: &ModelSpec) -> image::RgbImage {
    if image.width() == spec.input_width && image.height() == spec.input_height {
        image.to_rgb8()
    } else {
        image
            .resize_exact(spec.input_width, spec.input_height, FilterType::Triangle)
            .to_rgb8()
    }
}

/// Fill a single frame's tensor slice from raw RGB bytes.
///
/// Uses direct flat-buffer access instead of per-pixel `get_pixel` (which does
/// bounds checking). The 4 match arms hoist the normalization/layout branch
/// outside the hot loop so it becomes a straight arithmetic pipeline.
fn fill_frame_slice(
    slice: &mut ndarray::ArrayViewMut3<'_, f32>,
    raw: &[u8],
    w: usize,
    c: usize,
    spec: &ModelSpec,
) {
    let h = spec.input_height as usize;
    let inv255: f32 = 1.0 / 255.0;

    match (&spec.normalization, spec.channels_first) {
        (Normalization::ZeroOne, true) => {
            for y in 0..h {
                let row = y * w * 3;
                for x in 0..w {
                    let px = row + x * 3;
                    for ch in 0..c.min(3) {
                        slice[[ch, y, x]] = raw[px + ch] as f32 * inv255;
                    }
                }
            }
        }
        (Normalization::ZeroOne, false) => {
            for y in 0..h {
                let row = y * w * 3;
                for x in 0..w {
                    let px = row + x * 3;
                    for ch in 0..c.min(3) {
                        slice[[y, x, ch]] = raw[px + ch] as f32 * inv255;
                    }
                }
            }
        }
        (Normalization::ImageNet { mean, std }, true) => {
            for y in 0..h {
                let row = y * w * 3;
                for x in 0..w {
                    let px = row + x * 3;
                    for ch in 0..c.min(3) {
                        slice[[ch, y, x]] = (raw[px + ch] as f32 * inv255 - mean[ch]) / std[ch];
                    }
                }
            }
        }
        (Normalization::ImageNet { mean, std }, false) => {
            for y in 0..h {
                let row = y * w * 3;
                for x in 0..w {
                    let px = row + x * 3;
                    for ch in 0..c.min(3) {
                        slice[[y, x, ch]] = (raw[px + ch] as f32 * inv255 - mean[ch]) / std[ch];
                    }
                }
            }
        }
    }
}

/// Scale predicted keypoint coordinates from model space back to original frame space.
pub fn rescale_keypoints(
    x: f32,
    y: f32,
    model_width: u32,
    model_height: u32,
    original_width: u32,
    original_height: u32,
) -> (f32, f32) {
    let scale_x = original_width as f32 / model_width as f32;
    let scale_y = original_height as f32 / model_height as f32;
    (x * scale_x, y * scale_y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crop_region_from_bbox_basic() {
        let crop = CropRegion::from_bbox(100.0, 200.0, 500.0, 600.0, 50.0, 1920, 1080);
        assert_eq!(crop.x, 50);
        assert_eq!(crop.y, 150);
        assert_eq!(crop.width, 500);
        assert_eq!(crop.height, 500);
    }

    #[test]
    fn crop_region_clamped_to_frame() {
        let crop = CropRegion::from_bbox(10.0, 10.0, 1900.0, 1060.0, 100.0, 1920, 1080);
        // min_x - padding = -90 → clamped to 0
        assert_eq!(crop.x, 0);
        assert_eq!(crop.y, 0);
        // max_x + padding = 2000 → clamped to 1920
        assert_eq!(crop.width, 1920);
        assert_eq!(crop.height, 1080);
    }

    #[test]
    fn crop_region_zero_padding() {
        let crop = CropRegion::from_bbox(100.0, 200.0, 300.0, 400.0, 0.0, 1920, 1080);
        assert_eq!(crop.x, 100);
        assert_eq!(crop.y, 200);
        assert_eq!(crop.width, 200);
        assert_eq!(crop.height, 200);
    }

    #[test]
    fn crop_image_produces_correct_dimensions() {
        let img = DynamicImage::new_rgb8(1920, 1080);
        let crop = CropRegion {
            x: 100,
            y: 200,
            width: 400,
            height: 300,
        };
        let cropped = crop.crop(&img);
        assert_eq!(cropped.width(), 400);
        assert_eq!(cropped.height(), 300);
    }

    #[test]
    fn crop_then_rescale_recovers_coordinates() {
        // Simulate the full coordinate chain:
        // 1. Crop a 1920x1080 frame to a 600x400 region at (200, 100)
        // 2. Resize crop to 256x256 model input
        // 3. Model predicts keypoint at (128, 128) in model space
        // 4. Rescale to crop space: (128 * 600/256, 128 * 400/256) = (300, 200)
        // 5. Offset by crop origin: (300 + 200, 200 + 100) = (500, 300)
        let crop = CropRegion {
            x: 200,
            y: 100,
            width: 600,
            height: 400,
        };

        let model_x = 128.0f32;
        let model_y = 128.0f32;

        // Step 4: rescale from model space to crop space
        let (crop_x, crop_y) =
            rescale_keypoints(model_x, model_y, 256, 256, crop.width, crop.height);
        assert!((crop_x - 300.0).abs() < 0.01);
        assert!((crop_y - 200.0).abs() < 0.01);

        // Step 5: offset to full-frame space
        let full_x = crop_x + crop.x as f32;
        let full_y = crop_y + crop.y as f32;
        assert!((full_x - 500.0).abs() < 0.01);
        assert!((full_y - 300.0).abs() < 0.01);
    }
}
