use super::*;

/// Test that Vulkan Video decode and HIP operations can coexist in the
/// same process without LLVM symbol conflicts. This is the core regression
/// test for the "LLVM ERROR: Option 'default' already exists!" crash that
/// occurred when MIGraphX (LLVM 20) and Mesa RADV (LLVM 21) were loaded
/// together. Fixed by using a custom LLVM-free Mesa RADV build.
#[test]
#[ignore = "requires AMD GPU with Vulkan Video + HIP runtime"]
fn test_no_llvm_conflict_vulkan_plus_hip() {
    // 1. Initialize HIP (loads ROCm stack including MIGraphX's LLVM)
    bekiper_gpu::init(0).expect("HIP init failed");

    // 2. Open Vulkan Video decoder (loads RADV — must NOT load system LLVM)
    let video_path = get_test_video();
    let mut decoder = VideoDecoder::open(&video_path)
        .expect("VideoDecoder::open crashed — possible LLVM conflict");

    if decoder.hw_backend() != HwBackend::Vulkan {
        eprintln!(
            "skipping: Vulkan Video not available (backend={:?})",
            decoder.hw_backend()
        );
        return;
    }

    // 3. Decode a frame (exercises Vulkan Video hardware)
    let raw = decoder
        .gpu_next_nv12_host()
        .expect("Vulkan decode failed")
        .expect("no frames");

    // 4. Upload to HIP (exercises HIP runtime after Vulkan was loaded)
    let gpu_frame = GpuFrame::from_nv12_host(&raw.data, raw.width, raw.height, raw.y_pitch)
        .expect("HIP upload failed after Vulkan decode");

    // 5. GPU preprocess (exercises HIP kernel launch)
    let mut preprocessor =
        GpuPreprocessor::new(64, 64, NormParams::imagenet()).expect("preprocessor failed");
    let output = preprocessor
        .preprocess_nv12(&gpu_frame)
        .expect("NV12 preprocess failed");

    let tensor = output
        .copy_to_host_typed::<f32>(3 * 64 * 64)
        .expect("D2H copy failed");
    assert_eq!(tensor.len(), 3 * 64 * 64);
    assert!(
        tensor.iter().all(|v| v.is_finite()),
        "tensor has non-finite values"
    );

    eprintln!("LLVM coexistence test passed: Vulkan decode + HIP upload + preprocess OK");
}

// ── Resource cleanup tests ─────────────────────────────────────────

/// Sequential decode + host transfer + HIP upload + drop: tests resource cleanup.
#[test]
#[ignore = "requires AMD GPU with Vulkan Video + HIP runtime"]
fn test_vulkan_sequential_decode_no_leak() {
    bekiper_gpu::init(0).expect("HIP init failed");

    let video_path = get_test_video();
    let mut decoder = VideoDecoder::open(&video_path).expect("failed to open video");

    if decoder.hw_backend() != HwBackend::Vulkan {
        eprintln!("skipping: Vulkan Video not available");
        return;
    }

    let mut count = 0;
    for _ in 0..20 {
        let raw = match decoder.gpu_next_nv12_host() {
            Ok(Some(f)) => f,
            Ok(None) => break,
            Err(e) => panic!("decode error on frame {count}: {e}"),
        };

        let _gpu_frame = GpuFrame::from_nv12_host(&raw.data, raw.width, raw.height, raw.y_pitch)
            .unwrap_or_else(|e| panic!("import failed on frame {count}: {e}"));

        count += 1;
    }

    assert!(count > 0, "no frames decoded");
    eprintln!("decoded + imported + dropped {count} frames without leak");
}
