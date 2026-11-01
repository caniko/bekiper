/// Merge all `batch_*.parquet` files in `chunks_dir` into a single output parquet.
fn merge_pose_chunks(chunks_dir: &Path, output_path: &Path) -> Result<()> {
    let mut chunk_paths: Vec<PathBuf> = std::fs::read_dir(chunks_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension().is_some_and(|e| e == "parquet")
                && p.file_stem()
                    .and_then(|s| s.to_str())
                    .is_some_and(|s| s.starts_with("batch_"))
        })
        .collect();

    chunk_paths.sort();

    if chunk_paths.is_empty() {
        return Err(BekiperError::Data("no pose chunks found to merge".into()));
    }

    // Read and vstack all chunks
    let mut combined: Option<polars::prelude::DataFrame> = None;
    for path in &chunk_paths {
        let df = polars::prelude::LazyFrame::scan_parquet(
            path.to_str().unwrap().into(),
            Default::default(),
        )
        .map_err(|e| BekiperError::Data(format!("scan chunk {}: {e}", path.display())))?
        .collect()
        .map_err(|e| BekiperError::Data(format!("collect chunk {}: {e}", path.display())))?;

        combined = Some(match combined {
            None => df,
            Some(mut acc) => {
                acc.vstack_mut(&df)?;
                acc
            }
        });
    }

    let mut df = combined.unwrap();
    bekiper_pose::io::write_parquet(&mut df, output_path)?;

    tracing::info!(
        n_chunks = chunk_paths.len(),
        n_frames = df.height(),
        output = %output_path.display(),
        "merged pose chunks"
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    fn test_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("bekiper_test_checkpoint_{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_chunk(dir: &Path, batch_index: u32, frames: &[i64]) {
        let labels = vec!["nose"];
        let n = frames.len();
        let mut columns: Vec<Column> = Vec::new();
        columns.push(Column::new("frame".into(), frames));
        for label in &labels {
            columns.push(Column::new(format!("{label}_x").into(), &vec![0.0f64; n]));
            columns.push(Column::new(format!("{label}_y").into(), &vec![0.0f64; n]));
            columns.push(Column::new(
                format!("{label}_likelihood").into(),
                &vec![0.99f64; n],
            ));
        }
        let mut df = DataFrame::new(n, columns).unwrap();
        let path = dir.join(format!("batch_{batch_index:04}.parquet"));
        bekiper_pose::io::write_parquet(&mut df, &path).unwrap();
    }

    #[test]
    fn gpu_pose_pipeline_keeps_gpu_inference_with_cpu_decode() {
        assert_eq!(
            select_gpu_pose_pipeline(HwBackend::None, true),
            GpuPosePipeline::CpuDecode
        );
    }

    #[test]
    fn gpu_pose_pipeline_uses_vulkan_decode_when_available() {
        assert_eq!(
            select_gpu_pose_pipeline(HwBackend::Vulkan, true),
            GpuPosePipeline::VulkanDecode
        );
    }

    #[test]
    fn gpu_pose_pipeline_rejects_vaapi_with_gpu_inference() {
        assert!(matches!(
            select_gpu_pose_pipeline(HwBackend::VaApi, true),
            GpuPosePipeline::CpuFallback { .. }
        ));
    }

    #[test]
    fn gpu_pose_pipeline_falls_back_when_model_is_cpu() {
        assert!(matches!(
            select_gpu_pose_pipeline(HwBackend::None, false),
            GpuPosePipeline::CpuFallback { .. }
        ));
        assert!(matches!(
            select_gpu_pose_pipeline(HwBackend::Vulkan, false),
            GpuPosePipeline::CpuFallback { .. }
        ));
    }

    #[test]
    fn gpu_inference_batch_size_keeps_full_batch_shape() {
        assert_eq!(gpu_inference_batch_size(32, 32), 32);
    }

    #[test]
    fn gpu_inference_batch_size_pads_tail_batch_to_configured_shape() {
        assert_eq!(gpu_inference_batch_size(4, 32), 32);
    }

    #[test]
    fn gpu_inference_batch_size_never_truncates_oversized_actual_batch() {
        assert_eq!(gpu_inference_batch_size(40, 32), 40);
    }

    #[test]
    fn gpu_inference_batch_size_preserves_empty_batch() {
        assert_eq!(gpu_inference_batch_size(0, 32), 0);
    }

    #[test]
    fn decoded_frame_within_total_keeps_frames_before_probe_count() {
        assert!(decoded_frame_within_total(5475, 5476));
        assert!(!decoded_frame_within_total(5476, 5476));
        assert!(!decoded_frame_within_total(5486, 5476));
    }

    #[test]
    fn decoded_frame_within_total_does_not_cap_unknown_totals() {
        assert!(decoded_frame_within_total(u64::MAX, 0));
    }

    #[test]
    fn find_resume_point_empty_dir() {
        let dir = test_dir("resume_empty");
        let (frame, count) = find_resume_point(&dir).unwrap();
        assert_eq!(frame, 0);
        assert_eq!(count, 0);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn find_resume_point_nonexistent_dir() {
        let dir = std::env::temp_dir().join("bekiper_test_checkpoint_nonexistent");
        let _ = std::fs::remove_dir_all(&dir);
        let (frame, count) = find_resume_point(&dir).unwrap();
        assert_eq!(frame, 0);
        assert_eq!(count, 0);
    }

    #[test]
    fn find_resume_point_single_chunk() {
        let dir = test_dir("resume_single");
        write_chunk(&dir, 0, &[0, 1, 2, 3]);

        let (frame, count) = find_resume_point(&dir).unwrap();
        assert_eq!(frame, 4); // next frame after max(0,1,2,3)
        assert_eq!(count, 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn find_resume_point_multiple_chunks() {
        let dir = test_dir("resume_multi");
        write_chunk(&dir, 0, &[0, 1, 2, 3]);
        write_chunk(&dir, 1, &[4, 5, 6, 7]);
        write_chunk(&dir, 2, &[8, 9, 10]);

        let (frame, count) = find_resume_point(&dir).unwrap();
        assert_eq!(frame, 11); // next frame after max frame in last chunk
        assert_eq!(count, 3);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn find_resume_point_ignores_non_batch_files() {
        let dir = test_dir("resume_ignore");
        write_chunk(&dir, 0, &[0, 1, 2]);

        // Write a non-batch file
        std::fs::write(dir.join("summary.toml"), b"").unwrap();
        std::fs::write(dir.join("other.parquet"), b"not-a-batch").unwrap();

        let (frame, count) = find_resume_point(&dir).unwrap();
        assert_eq!(frame, 3);
        assert_eq!(count, 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn merge_pose_chunks_combines_all() {
        let dir = test_dir("merge_all");
        write_chunk(&dir, 0, &[0, 1, 2, 3]);
        write_chunk(&dir, 1, &[4, 5, 6, 7]);
        write_chunk(&dir, 2, &[8, 9]);

        let output = dir.join("merged.parquet");
        merge_pose_chunks(&dir, &output).unwrap();

        let df = LazyFrame::scan_parquet(output.to_str().unwrap().into(), Default::default())
            .unwrap()
            .collect()
            .unwrap();

        assert_eq!(df.height(), 10);

        let frames: Vec<i64> = df
            .column("frame")
            .unwrap()
            .i64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(frames, (0..10).collect::<Vec<i64>>());

        // Verify all columns present
        assert!(df.column("nose_x").is_ok());
        assert!(df.column("nose_y").is_ok());
        assert!(df.column("nose_likelihood").is_ok());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn merge_pose_chunks_preserves_order() {
        let dir = test_dir("merge_order");
        // Write chunks out of order on disk but with sequential batch names
        write_chunk(&dir, 1, &[4, 5, 6]);
        write_chunk(&dir, 0, &[0, 1, 2, 3]);

        let output = dir.join("merged.parquet");
        merge_pose_chunks(&dir, &output).unwrap();

        let df = LazyFrame::scan_parquet(output.to_str().unwrap().into(), Default::default())
            .unwrap()
            .collect()
            .unwrap();

        // batch_0000 should come first (sorted by filename), then batch_0001
        let frames: Vec<i64> = df
            .column("frame")
            .unwrap()
            .i64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(frames, vec![0, 1, 2, 3, 4, 5, 6]);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn merge_pose_chunks_empty_dir_errors() {
        let dir = test_dir("merge_empty");
        let output = dir.join("merged.parquet");
        assert!(merge_pose_chunks(&dir, &output).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn offset_keypoint_columns_adds_crop_offset() {
        let n = 3;
        let mut df = DataFrame::new(
            n,
            vec![
                Column::new("frame".into(), &[0i64, 1, 2]),
                Column::new("nose_x".into(), &[10.0f64, 20.0, 30.0]),
                Column::new("nose_y".into(), &[40.0f64, 50.0, 60.0]),
                Column::new("tail_x".into(), &[1.0f64, 2.0, 3.0]),
                Column::new("tail_y".into(), &[4.0f64, 5.0, 6.0]),
                Column::new("nose_likelihood".into(), &[0.9f64, 0.8, 0.7]),
            ],
        )
        .unwrap();

        offset_keypoint_columns(&mut df, 100.0, 200.0).unwrap();

        // _x columns should be offset by 100
        let nose_x: Vec<f64> = df
            .column("nose_x")
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(nose_x, vec![110.0, 120.0, 130.0]);

        let tail_x: Vec<f64> = df
            .column("tail_x")
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(tail_x, vec![101.0, 102.0, 103.0]);

        // _y columns should be offset by 200
        let nose_y: Vec<f64> = df
            .column("nose_y")
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(nose_y, vec![240.0, 250.0, 260.0]);

        let tail_y: Vec<f64> = df
            .column("tail_y")
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(tail_y, vec![204.0, 205.0, 206.0]);

        // frame and likelihood columns should be unchanged
        let frames: Vec<i64> = df
            .column("frame")
            .unwrap()
            .i64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(frames, vec![0, 1, 2]);

        let lik: Vec<f64> = df
            .column("nose_likelihood")
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(lik, vec![0.9, 0.8, 0.7]);
    }

    #[test]
    fn offset_keypoint_columns_zero_offset_is_noop() {
        let n = 2;
        let mut df = DataFrame::new(
            n,
            vec![
                Column::new("frame".into(), &[0i64, 1]),
                Column::new("nose_x".into(), &[10.0f64, 20.0]),
                Column::new("nose_y".into(), &[30.0f64, 40.0]),
            ],
        )
        .unwrap();

        offset_keypoint_columns(&mut df, 0.0, 0.0).unwrap();

        let xs: Vec<f64> = df
            .column("nose_x")
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(xs, vec![10.0, 20.0]);
    }

    #[test]
    fn load_precomputed_pose_data_accepts_parquet() {
        let dir = test_dir("precomputed_parquet");
        let src = dir.join("input.parquet");
        let out = dir.join("output.parquet");

        let mut df = DataFrame::new(
            2,
            vec![
                Column::new("frame".into(), &[0i64, 1]),
                Column::new("nose_x".into(), &[10.0f64, 11.0]),
                Column::new("nose_y".into(), &[20.0f64, 21.0]),
                Column::new("nose_likelihood".into(), &[0.9f64, 0.95]),
            ],
        )
        .unwrap();
        bekiper_pose::io::write_parquet(&mut df, &src).unwrap();

        load_precomputed_pose_data(&src, &out).unwrap();

        let loaded = scan_file(&out).unwrap().collect().unwrap();
        assert_eq!(loaded.height(), 2);
        assert!(loaded.column("nose_x").is_ok());

        std::fs::remove_dir_all(&dir).ok();
    }
}
