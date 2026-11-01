#[test]
fn test_norm_params_imagenet() {
    let p = NormParams::imagenet();
    assert!((p.mean[0] - 0.485).abs() < 1e-6);
    assert!((p.std[2] - 0.225).abs() < 1e-6);
}

#[test]
fn test_norm_params_zero_one() {
    let p = NormParams::zero_one();
    assert_eq!(p.mean, [0.0, 0.0, 0.0]);
    assert_eq!(p.std, [1.0, 1.0, 1.0]);
}

/// Test that DeviceBuffer can be allocated and freed without panicking.
/// Requires a GPU — this test is ignored in CI.
#[test]
#[ignore = "requires AMD GPU with HIP runtime"]
fn test_device_buffer_alloc_free() {
    init(0).expect("HIP init failed");
    let buf = DeviceBuffer::alloc(1024).expect("alloc failed");
    assert!(!buf.as_ptr().is_null());
    assert_eq!(buf.size(), 1024);
    // buf dropped here, hipFree called
}

/// Test host-to-device and device-to-host copy round-trip.
#[test]
#[ignore = "requires AMD GPU with HIP runtime"]
fn test_device_buffer_round_trip() {
    init(0).expect("HIP init failed");
    let data: Vec<u8> = (0..=255).collect();
    let buf = DeviceBuffer::alloc(256).expect("alloc failed");
    buf.copy_from_host(&data).expect("H2D copy failed");

    let mut out = vec![0u8; 256];
    buf.copy_to_host(&mut out).expect("D2H copy failed");
    assert_eq!(data, out);
}

/// Test typed copy round-trip with f32.
#[test]
#[ignore = "requires AMD GPU with HIP runtime"]
fn test_device_buffer_f32_round_trip() {
    init(0).expect("HIP init failed");
    let data: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
    let buf = DeviceBuffer::alloc_typed::<f32>(64).expect("alloc failed");

    // Copy via raw bytes
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    buf.copy_from_host(bytes).expect("H2D failed");

    let out = buf.copy_to_host_typed::<f32>(64).expect("D2H failed");
    for (a, b) in data.iter().zip(out.iter()) {
        assert!((a - b).abs() < 1e-6, "mismatch: {a} vs {b}");
    }
}

// ── GpuFrame unit tests (from_nv12_host + Drop) ─────────────────

mod gpu_frame_tests {
    /// from_nv12_host rejects buffers smaller than y_pitch * height * 3/2.
    #[test]
    #[ignore = "requires AMD GPU with HIP runtime"]
    fn test_from_nv12_host_rejects_small_buffer() {
        crate::init(0).expect("HIP init failed");

        let too_small = vec![0u8; 100];
        let result = crate::GpuFrame::from_nv12_host(&too_small, 320, 240, 320);
        assert!(result.is_err(), "should reject undersized buffer");
        let msg = format!("{}", result.err().unwrap());
        assert!(msg.contains("too small"), "error message: {msg}");
    }

    /// from_nv12_host with exact-size buffer succeeds and layout is correct.
    #[test]
    #[ignore = "requires AMD GPU with HIP runtime"]
    fn test_from_nv12_host_layout_correct() {
        crate::init(0).expect("HIP init failed");

        let w = 320u32;
        let h = 240u32;
        let pitch = 320u32;
        let total = (pitch * h * 3 / 2) as usize; // NV12 = 1.5 bytes per pixel
        let data = vec![128u8; total];

        let frame =
            crate::GpuFrame::from_nv12_host(&data, w, h, pitch).expect("from_nv12_host failed");

        assert_eq!(frame.layout.width, w);
        assert_eq!(frame.layout.height, h);
        assert_eq!(frame.layout.y_offset, 0);
        assert_eq!(frame.layout.y_pitch, pitch);
        assert_eq!(frame.layout.uv_offset, pitch * h);
        assert_eq!(frame.layout.uv_pitch, pitch);
        assert_eq!(frame.layout.total_size, total as u64);
        assert!(!frame.device_ptr().is_null());
    }

    /// from_nv12_host with padded pitch (pitch > width) works correctly.
    #[test]
    #[ignore = "requires AMD GPU with HIP runtime"]
    fn test_from_nv12_host_padded_pitch() {
        crate::init(0).expect("HIP init failed");

        let w = 320u32;
        let h = 240u32;
        let pitch = 384u32; // 64-byte aligned, > width
        let total = (pitch * h * 3 / 2) as usize;
        let data = vec![64u8; total];

        let frame = crate::GpuFrame::from_nv12_host(&data, w, h, pitch)
            .expect("from_nv12_host with padded pitch failed");

        assert_eq!(frame.layout.y_pitch, pitch);
        assert_eq!(frame.layout.uv_offset, pitch * h);
        assert_eq!(frame.layout.total_size, total as u64);
    }

    /// GpuFrame::Drop works for the hipMalloc path (from_nv12_host).
    /// Allocate and drop many frames to check for leaks / double-free.
    #[test]
    #[ignore = "requires AMD GPU with HIP runtime"]
    fn test_from_nv12_host_drop_no_crash() {
        crate::init(0).expect("HIP init failed");

        let w = 64u32;
        let h = 48u32;
        let pitch = 64u32;
        let data = vec![0u8; (pitch * h * 3 / 2) as usize];

        for _ in 0..50 {
            let frame =
                crate::GpuFrame::from_nv12_host(&data, w, h, pitch).expect("from_nv12_host failed");
            // GpuFrame::drop should call hipFree (not hipDestroyExternalMemory)
            drop(frame);
        }
    }

    /// Plane pointers are offset correctly from the base device pointer.
    #[test]
    #[ignore = "requires AMD GPU with HIP runtime"]
    fn test_from_nv12_host_plane_pointers() {
        crate::init(0).expect("HIP init failed");

        let w = 320u32;
        let h = 240u32;
        let pitch = 320u32;
        let data = vec![0u8; (pitch * h * 3 / 2) as usize];

        let frame =
            crate::GpuFrame::from_nv12_host(&data, w, h, pitch).expect("from_nv12_host failed");

        let base = frame.device_ptr() as usize;
        let y_ptr = frame.y_plane_ptr() as usize;
        let uv_ptr = frame.uv_plane_ptr() as usize;

        assert_eq!(y_ptr, base, "Y plane should start at base");
        assert_eq!(
            uv_ptr,
            base + (pitch * h) as usize,
            "UV plane offset mismatch"
        );
    }

    /// NV12 host data round-trips correctly through HIP.
    #[test]
    #[ignore = "requires AMD GPU with HIP runtime"]
    fn test_from_nv12_host_data_roundtrip() {
        crate::init(0).expect("HIP init failed");

        let w = 4u32;
        let h = 4u32;
        let pitch = 4u32;
        // NV12: 4x4 Y + 4x2 UV = 24 bytes
        let data: Vec<u8> = (0..24).collect();

        let frame =
            crate::GpuFrame::from_nv12_host(&data, w, h, pitch).expect("from_nv12_host failed");

        // Read back from GPU
        let mut readback = vec![0u8; 24];
        crate::error::check_hip(
            unsafe {
                crate::hip_sys::hipMemcpy(
                    readback.as_mut_ptr() as *mut std::ffi::c_void,
                    frame.device_ptr() as *const std::ffi::c_void,
                    24,
                    crate::hip_sys::hipMemcpyDeviceToHost,
                )
            },
            "hipMemcpy D2H",
        )
        .expect("readback failed");

        assert_eq!(data, readback, "NV12 data should round-trip through HIP");
    }
}

// ── Vulkan unit tests ──────────────────────────────────────────
use crate::*;
