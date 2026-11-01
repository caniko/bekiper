use std::path::{Path, PathBuf};
use std::process::Command;

use bekiper_gpu::{GpuFrame, GpuPreprocessor, NormParams};
use bekiper_pose::video::{HwBackend, VideoDecoder};

/// Find a test video file from the examples directory.
fn find_test_video() -> Option<PathBuf> {
    let candidates =
        ["examples/imen/NORT videos/NORT (post)/T1/Config A/post.T1.configA.video-2.mp4"];

    for rel in &candidates {
        let p = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(rel);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// Generate a short synthetic test video using ffmpeg's testsrc2 filter.
fn generate_test_video() -> PathBuf {
    let dir = std::env::temp_dir().join("bekiper_test_gpu_pipeline");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.mp4");

    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=25:duration=2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "10",
            path.to_str().unwrap(),
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .expect("ffmpeg must be on PATH to run GPU tests");

    assert!(status.success(), "ffmpeg failed to generate test video");
    path
}

/// Get a test video, preferring real data, falling back to synthetic.
pub(crate) fn get_test_video() -> PathBuf {
    find_test_video().unwrap_or_else(generate_test_video)
}

// ── Decoder selection tests ────────────────────────────────────────

/// Test that VideoDecoder selects a hardware backend (Vulkan or VA-API).
#[test]
#[ignore = "requires AMD GPU with Vulkan Video or VA-API"]
fn test_video_decoder_selects_hw_backend() {
    let video_path = get_test_video();
    let decoder = VideoDecoder::open(&video_path).expect("failed to open video");

    eprintln!("hw_backend: {:?}", decoder.hw_backend());
    eprintln!("hw_accel: {}", decoder.is_hw_accel());
    eprintln!("codec: {}", decoder.info().codec);
    eprintln!(
        "resolution: {}x{}",
        decoder.info().width,
        decoder.info().height
    );

    assert!(
        decoder.is_hw_accel(),
        "expected hardware acceleration, got CPU decode"
    );
}

/// Test that VulkanContext is available when Vulkan backend is selected.
#[test]
#[ignore = "requires AMD GPU with Vulkan Video support"]
fn test_video_decoder_vulkan_context_available() {
    let video_path = get_test_video();
    let decoder = VideoDecoder::open(&video_path).expect("failed to open video");

    if decoder.hw_backend() != HwBackend::Vulkan {
        eprintln!(
            "skipping: Vulkan Video not available (backend={:?})",
            decoder.hw_backend()
        );
        return;
    }

    let vk_ctx = decoder.vk_context().expect("vk_context() returned None");
    eprintln!(
        "VulkanContext: decode_qf={}, graphics_qf={}, extensions={}",
        vk_ctx.decode_queue_family,
        vk_ctx.graphics_queue_family,
        vk_ctx.enabled_device_extensions.len(),
    );
}

/// Test that Vulkan decode produces matching resolution vs CPU probe.
#[test]
#[ignore = "requires AMD GPU with Vulkan Video"]
fn test_vulkan_decode_matches_cpu_resolution() {
    let video_path = get_test_video();

    let decoder_hw = VideoDecoder::open(&video_path).expect("hw open failed");

    if decoder_hw.hw_backend() != HwBackend::Vulkan {
        eprintln!("skipping: Vulkan Video not available");
        return;
    }

    let hw_info = decoder_hw.info().clone();
    let cpu_info = bekiper_pose::video::probe_video_info(&video_path).expect("probe failed");

    assert_eq!(hw_info.width, cpu_info.width);
    assert_eq!(hw_info.height, cpu_info.height);
    assert_eq!(hw_info.total_frames, cpu_info.total_frames);
    assert!((hw_info.fps - cpu_info.fps).abs() < 0.01);

    eprintln!(
        "resolution match: {}x{}, {} frames, {:.2} fps",
        hw_info.width, hw_info.height, hw_info.total_frames, hw_info.fps
    );
}

// ── Vulkan Video decode tests ──────────────────────────────────────

/// Decode a single frame via Vulkan Video and transfer to host NV12.
#[test]
#[ignore = "requires AMD GPU with Vulkan Video"]
fn test_vulkan_decode_single_nv12_host() {
    let video_path = get_test_video();
    let mut decoder = VideoDecoder::open(&video_path).expect("failed to open video");

    if decoder.hw_backend() != HwBackend::Vulkan {
        eprintln!(
            "skipping: Vulkan Video not available (backend={:?})",
            decoder.hw_backend()
        );
        return;
    }

    let frame = decoder
        .gpu_next_nv12_host()
        .expect("decode error")
        .expect("no frames in video");

    assert_eq!(frame.index, 0);
    assert!(frame.width > 0);
    assert!(frame.height > 0);
    assert!(frame.y_pitch >= frame.width);
    assert!(frame.timestamp_seconds >= 0.0);

    let expected_size = (frame.y_pitch * frame.height * 3 / 2) as usize;
    assert_eq!(frame.data.len(), expected_size);

    eprintln!(
        "Vulkan NV12 host frame: {}x{}, y_pitch={}, data_len={}, index={}, ts={:.3}s",
        frame.width,
        frame.height,
        frame.y_pitch,
        frame.data.len(),
        frame.index,
        frame.timestamp_seconds
    );
}

/// Decode a batch of frames via Vulkan Video, transferred to host NV12.
#[test]
#[ignore = "requires AMD GPU with Vulkan Video"]
fn test_vulkan_decode_batch_nv12_host() {
    let video_path = get_test_video();
    let mut decoder = VideoDecoder::open(&video_path).expect("failed to open video");

    if decoder.hw_backend() != HwBackend::Vulkan {
        eprintln!("skipping: Vulkan Video not available");
        return;
    }

    let batch = decoder
        .gpu_next_nv12_host_batch(4)
        .expect("batch decode failed");

    assert!(!batch.is_empty(), "no frames decoded");
    assert!(batch.len() <= 4);

    for (i, frame) in batch.iter().enumerate() {
        assert_eq!(frame.index, i as u64);
        assert!(frame.width > 0);
        assert!(frame.height > 0);
        assert!(!frame.data.is_empty());
    }

    eprintln!("Vulkan NV12 host batch: {} frames", batch.len());
}

// ── Vulkan → host NV12 → HIP upload tests ──────────────────────────

/// Vulkan decode → host NV12 → GpuFrame::from_nv12_host.
#[test]
#[ignore = "requires AMD GPU with Vulkan Video + HIP runtime"]
fn test_vulkan_decode_to_gpu_frame() {
    bekiper_gpu::init(0).expect("HIP init failed");

    let video_path = get_test_video();
    let mut decoder = VideoDecoder::open(&video_path).expect("failed to open video");

    if decoder.hw_backend() != HwBackend::Vulkan {
        eprintln!("skipping: Vulkan Video not available");
        return;
    }

    let raw = decoder
        .gpu_next_nv12_host()
        .expect("decode error")
        .expect("no frames");

    let gpu_frame = GpuFrame::from_nv12_host(&raw.data, raw.width, raw.height, raw.y_pitch)
        .expect("GpuFrame::from_nv12_host failed");

    assert!(!gpu_frame.device_ptr().is_null(), "device ptr is null");
    assert_eq!(gpu_frame.layout.width, raw.width);
    assert_eq!(gpu_frame.layout.height, raw.height);
    assert!(gpu_frame.layout.total_size > 0);
    assert!(gpu_frame.layout.y_pitch > 0);

    eprintln!(
        "GpuFrame from host NV12: {}x{}, total_size={}, y_pitch={}, uv_offset={}",
        gpu_frame.layout.width,
        gpu_frame.layout.height,
        gpu_frame.layout.total_size,
        gpu_frame.layout.y_pitch,
        gpu_frame.layout.uv_offset,
    );
}

// ── Full pipeline tests (decode → preprocess → tensor) ───────────

/// Full pipeline: Vulkan decode → host NV12 → HIP upload → NV12 preprocess → NCHW tensor.
#[test]
#[ignore = "requires AMD GPU with Vulkan Video + HIP runtime"]
fn test_vulkan_full_preprocess_pipeline() {
    bekiper_gpu::init(0).expect("HIP init failed");

    let video_path = get_test_video();
    let mut decoder = VideoDecoder::open(&video_path).expect("failed to open video");

    if decoder.hw_backend() != HwBackend::Vulkan {
        eprintln!("skipping: Vulkan Video not available");
        return;
    }

    let raw = decoder
        .gpu_next_nv12_host()
        .expect("decode error")
        .expect("no frames");

    let gpu_frame = GpuFrame::from_nv12_host(&raw.data, raw.width, raw.height, raw.y_pitch)
        .expect("GpuFrame import failed");

    let dst_w = 256u32;
    let dst_h = 256u32;
    let mut preprocessor = GpuPreprocessor::new(dst_w, dst_h, NormParams::imagenet())
        .expect("preprocessor creation failed");

    let output = preprocessor
        .preprocess_nv12(&gpu_frame)
        .expect("preprocess_nv12 failed");

    let n_floats = (3 * dst_w * dst_h) as usize;
    let tensor = output
        .copy_to_host_typed::<f32>(n_floats)
        .expect("D2H copy failed");

    assert_eq!(tensor.len(), n_floats);

    let min = tensor.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = tensor.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean: f32 = tensor.iter().sum::<f32>() / tensor.len() as f32;

    eprintln!("tensor stats: min={min:.3}, max={max:.3}, mean={mean:.3}");

    assert!(min > -5.0, "min value {min} is suspiciously low");
    assert!(max < 5.0, "max value {max} is suspiciously high");
    assert!(!min.is_nan(), "tensor contains NaN");
    assert!(!max.is_nan(), "tensor contains NaN");

    let ch_size = (dst_w * dst_h) as usize;
    let r_mean: f32 = tensor[..ch_size].iter().sum::<f32>() / ch_size as f32;
    let g_mean: f32 = tensor[ch_size..2 * ch_size].iter().sum::<f32>() / ch_size as f32;
    let b_mean: f32 = tensor[2 * ch_size..].iter().sum::<f32>() / ch_size as f32;

    assert!(r_mean.is_finite());
    assert!(g_mean.is_finite());
    assert!(b_mean.is_finite());

    eprintln!("channel means: R={r_mean:.3}, G={g_mean:.3}, B={b_mean:.3}");
}

/// Batch pipeline: Vulkan decode batch → host NV12 → HIP upload → batch preprocess.
#[test]
#[ignore = "requires AMD GPU with Vulkan Video + HIP runtime"]
fn test_vulkan_batch_preprocess_pipeline() {
    bekiper_gpu::init(0).expect("HIP init failed");

    let video_path = get_test_video();
    let mut decoder = VideoDecoder::open(&video_path).expect("failed to open video");

    if decoder.hw_backend() != HwBackend::Vulkan {
        eprintln!("skipping: Vulkan Video not available");
        return;
    }

    let batch_size = 4;
    let raw_frames = decoder
        .gpu_next_nv12_host_batch(batch_size)
        .expect("batch decode failed");

    assert!(!raw_frames.is_empty(), "no frames decoded");

    let gpu_frames: Vec<GpuFrame> = raw_frames
        .iter()
        .map(|raw| {
            GpuFrame::from_nv12_host(&raw.data, raw.width, raw.height, raw.y_pitch)
                .expect("GpuFrame import failed")
        })
        .collect();

    let dst_w = 256u32;
    let dst_h = 256u32;
    let preprocessor = GpuPreprocessor::new(dst_w, dst_h, NormParams::imagenet())
        .expect("preprocessor creation failed");

    let frame_refs: Vec<&GpuFrame> = gpu_frames.iter().collect();
    let output = preprocessor
        .preprocess_nv12_batch(&frame_refs)
        .expect("batch preprocess failed");

    let n = gpu_frames.len();
    let expected_floats = n * 3 * dst_w as usize * dst_h as usize;
    let tensor = output
        .copy_to_host_typed::<f32>(expected_floats)
        .expect("D2H copy failed");

    assert_eq!(tensor.len(), expected_floats);

    let frame_size = 3 * dst_w as usize * dst_h as usize;
    for i in 0..n {
        let slice = &tensor[i * frame_size..(i + 1) * frame_size];
        let min = slice.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(!min.is_nan(), "frame {i} contains NaN");
        assert!(min > -5.0, "frame {i}: min={min} suspiciously low");
        assert!(max < 5.0, "frame {i}: max={max} suspiciously high");
    }

    eprintln!(
        "batch preprocess: {} frames -> [{}, 3, {}, {}]",
        n, n, dst_h, dst_w
    );
}

// ── from_nv12_host edge-case tests ─────────────────────────────────

/// from_nv12_host rejects buffer that is too small for the declared dimensions.
#[test]
#[ignore = "requires AMD GPU with HIP runtime"]
fn test_gpu_frame_from_nv12_host_rejects_undersized() {
    bekiper_gpu::init(0).expect("HIP init failed");

    let video_path = get_test_video();
    let mut decoder = VideoDecoder::open(&video_path).expect("failed to open video");

    if decoder.hw_backend() != HwBackend::Vulkan {
        eprintln!("skipping: Vulkan Video not available");
        return;
    }

    let raw = decoder
        .gpu_next_nv12_host()
        .expect("decode error")
        .expect("no frames");

    // Pass truncated data — should fail
    let truncated = &raw.data[..raw.data.len() / 2];
    let result = GpuFrame::from_nv12_host(truncated, raw.width, raw.height, raw.y_pitch);
    assert!(result.is_err(), "should reject undersized buffer");
}

/// from_nv12_host with oversized buffer (extra padding) still works.
#[test]
#[ignore = "requires AMD GPU with HIP runtime"]
fn test_gpu_frame_from_nv12_host_accepts_oversized() {
    bekiper_gpu::init(0).expect("HIP init failed");

    let video_path = get_test_video();
    let mut decoder = VideoDecoder::open(&video_path).expect("failed to open video");

    if decoder.hw_backend() != HwBackend::Vulkan {
        eprintln!("skipping: Vulkan Video not available");
        return;
    }

    let raw = decoder
        .gpu_next_nv12_host()
        .expect("decode error")
        .expect("no frames");

    // Append extra padding — should still work
    let mut padded = raw.data.clone();
    padded.extend_from_slice(&[0u8; 4096]);
    let frame = GpuFrame::from_nv12_host(&padded, raw.width, raw.height, raw.y_pitch)
        .expect("from_nv12_host should accept oversized buffer");
    assert!(!frame.device_ptr().is_null());
}

// ── LLVM coexistence integration test ──────────────────────────────
