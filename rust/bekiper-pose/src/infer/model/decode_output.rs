use super::super::preprocess::rescale_keypoints;
use super::super::spec::{Keypoint, ModelSpec, OutputLayout};
use super::*;
use bekiper_core::error::{BekiperError, Result};
use polars::prelude::*;

/// Decode a batch into the same frame/label columns for CPU and GPU inference.
pub(super) fn decode_dataframe(
    spec: &ModelSpec,
    output: &ndarray::ArrayViewD<'_, f32>,
    original_sizes: &[(u32, u32)],
) -> Result<DataFrame> {
    let n = original_sizes.len();
    let num_labels = spec.labels.len();
    let frames: Vec<i64> = (0..n as i64).collect();
    let mut label_data: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> = (0..num_labels)
        .map(|_| {
            (
                Vec::with_capacity(n),
                Vec::with_capacity(n),
                Vec::with_capacity(n),
            )
        })
        .collect();

    for (i, &orig_size) in original_sizes.iter().enumerate().take(n) {
        let keypoints = decode_output(spec, output, i, orig_size)?;
        for (k, kp) in keypoints.iter().enumerate() {
            label_data[k].0.push(kp.x as f64);
            label_data[k].1.push(kp.y as f64);
            label_data[k].2.push(kp.confidence as f64);
        }
    }

    let mut columns: Vec<Column> = Vec::with_capacity(1 + num_labels * 3);
    columns.push(Column::new("frame".into(), &frames));
    for (k, label) in spec.labels.iter().enumerate() {
        columns.push(Column::new(format!("{label}_x").into(), &label_data[k].0));
        columns.push(Column::new(format!("{label}_y").into(), &label_data[k].1));
        columns.push(Column::new(
            format!("{label}_likelihood").into(),
            &label_data[k].2,
        ));
    }
    DataFrame::new(n, columns).map_err(BekiperError::Polars)
}

/// Decode model output for a single frame within a (possibly batched) output tensor.
pub(super) fn decode_output(
    spec: &ModelSpec,
    output: &ndarray::ArrayViewD<'_, f32>,
    frame_idx: usize,
    (orig_w, orig_h): (u32, u32),
) -> Result<Vec<Keypoint>> {
    let num_keypoints = spec.labels.len();
    let mut keypoints = Vec::with_capacity(num_keypoints);

    match &spec.output_layout {
        OutputLayout::KeypointsFlat => {
            // [N, K, 3]
            for k in 0..num_keypoints {
                let raw_x = output[[frame_idx, k, 0]];
                let raw_y = output[[frame_idx, k, 1]];
                let confidence = output[[frame_idx, k, 2]];
                let (x, y) = rescale_keypoints(
                    raw_x,
                    raw_y,
                    spec.input_width,
                    spec.input_height,
                    orig_w,
                    orig_h,
                );
                keypoints.push(Keypoint { x, y, confidence });
            }
        }
        OutputLayout::Heatmaps => {
            // [N, K, H, W]
            let hm_h = output.shape()[2];
            let hm_w = output.shape()[3];
            for k in 0..num_keypoints {
                let mut max_val = f32::NEG_INFINITY;
                let mut max_y = 0usize;
                let mut max_x = 0usize;
                for y in 0..hm_h {
                    for x in 0..hm_w {
                        let val = output[[frame_idx, k, y, x]];
                        if val > max_val {
                            max_val = val;
                            max_y = y;
                            max_x = x;
                        }
                    }
                }
                let model_x = (max_x as f32 / hm_w as f32) * spec.input_width as f32;
                let model_y = (max_y as f32 / hm_h as f32) * spec.input_height as f32;
                let (x, y) = rescale_keypoints(
                    model_x,
                    model_y,
                    spec.input_width,
                    spec.input_height,
                    orig_w,
                    orig_h,
                );
                keypoints.push(Keypoint {
                    x,
                    y,
                    confidence: max_val,
                });
            }
        }
    }

    Ok(keypoints)
}

impl OnnxPoseModel {
    pub(super) fn empty_dataframe(&self) -> Result<DataFrame> {
        let mut columns: Vec<Column> = Vec::new();
        columns.push(Column::new("frame".into(), Vec::<i64>::new()));
        for label in &self.spec.labels {
            columns.push(Column::new(format!("{label}_x").into(), Vec::<f64>::new()));
            columns.push(Column::new(format!("{label}_y").into(), Vec::<f64>::new()));
            columns.push(Column::new(
                format!("{label}_likelihood").into(),
                Vec::<f64>::new(),
            ));
        }
        DataFrame::new(0, columns).map_err(BekiperError::Polars)
    }
}

/// Read VRAM total and used bytes from ROCm sysfs.
///
/// Scans `/sys/class/drm/card*/device/mem_info_vram_total` and `_used` for all
/// AMD GPUs, returning the first card that has both files (typically the discrete GPU).
pub(super) fn read_vram_usage() -> Option<(u64, u64)> {
    for entry in std::fs::read_dir("/sys/class/drm").ok()? {
        let entry = entry.ok()?;
        let name = entry.file_name();
        let name = name.to_str()?;
        if !name.starts_with("card") || name.contains('-') {
            continue;
        }

        let device_dir = entry.path().join("device");
        let total_path = device_dir.join("mem_info_vram_total");
        let used_path = device_dir.join("mem_info_vram_used");

        if !total_path.exists() || !used_path.exists() {
            continue;
        }

        let total: u64 = std::fs::read_to_string(&total_path)
            .ok()?
            .trim()
            .parse()
            .ok()?;
        let used: u64 = std::fs::read_to_string(&used_path)
            .ok()?
            .trim()
            .parse()
            .ok()?;

        // Skip integrated GPUs (typically < 1GB VRAM)
        if total > 1_000_000_000 {
            return Some((total, used));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(layout: OutputLayout) -> ModelSpec {
        let mut spec = ModelSpec::lightning_pose();
        spec.labels = vec!["nose".into(), "tail_base".into()];
        spec.input_width = 4;
        spec.input_height = 4;
        spec.output_layout = layout;
        spec
    }

    fn assert_columns(frame: &DataFrame, expected: [&[f64]; 6]) {
        let names = [
            "frame",
            "nose_x",
            "nose_y",
            "nose_likelihood",
            "tail_base_x",
            "tail_base_y",
            "tail_base_likelihood",
        ];
        assert_eq!(
            frame
                .get_column_names()
                .iter()
                .map(|name| name.as_str())
                .collect::<Vec<_>>(),
            names
        );
        assert_eq!(frame.column("frame").unwrap().dtype(), &DataType::Int64);
        assert_eq!(
            frame
                .column("frame")
                .unwrap()
                .i64()
                .unwrap()
                .into_no_null_iter()
                .collect::<Vec<_>>(),
            [0, 1]
        );
        for (name, values) in names[1..].iter().zip(expected) {
            let column = frame.column(name).unwrap();
            assert_eq!(column.dtype(), &DataType::Float64);
            assert_eq!(
                column
                    .f64()
                    .unwrap()
                    .into_no_null_iter()
                    .collect::<Vec<_>>(),
                values
            );
        }
    }

    #[test]
    fn flat_batch_preserves_frame_label_order_and_per_frame_scaling() {
        let output = ndarray::array![
            [[1.0, 2.0, 0.5], [3.0, 1.0, 0.25]],
            [[2.0, 3.0, 0.75], [1.0, 0.0, 1.0]],
        ]
        .into_dyn();
        let frame = decode_dataframe(
            &spec(OutputLayout::KeypointsFlat),
            &output.view(),
            &[(8, 12), (20, 8)],
        )
        .unwrap();
        assert_columns(
            &frame,
            [
                &[2.0, 10.0],
                &[6.0, 6.0],
                &[0.5, 0.75],
                &[6.0, 5.0],
                &[3.0, 0.0],
                &[0.25, 1.0],
            ],
        );
    }

    #[test]
    fn heatmap_batch_preserves_peak_confidence_and_original_dimensions() {
        let mut output = ndarray::Array4::<f32>::zeros((2, 2, 2, 2));
        output[[0, 0, 0, 1]] = 0.5;
        output[[0, 1, 1, 0]] = 0.75;
        output[[1, 0, 1, 1]] = 1.0;
        output[[1, 1, 0, 0]] = 0.25;
        let frame = decode_dataframe(
            &spec(OutputLayout::Heatmaps),
            &output.view().into_dyn(),
            &[(8, 12), (20, 8)],
        )
        .unwrap();
        assert_columns(
            &frame,
            [
                &[4.0, 10.0],
                &[0.0, 4.0],
                &[0.5, 1.0],
                &[0.0, 0.0],
                &[6.0, 0.0],
                &[0.75, 0.25],
            ],
        );
    }

    #[test]
    fn batch_dataframe_retains_polars_column_errors() {
        let mut spec = spec(OutputLayout::KeypointsFlat);
        spec.labels[1] = spec.labels[0].clone();
        let output = ndarray::Array3::<f32>::zeros((1, 2, 3)).into_dyn();
        assert!(matches!(
            decode_dataframe(&spec, &output.view(), &[(4, 4)]),
            Err(BekiperError::Polars(_))
        ));
    }
}
