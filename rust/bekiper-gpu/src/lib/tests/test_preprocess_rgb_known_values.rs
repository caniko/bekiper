/// Test RGB preprocessing with known input produces expected normalization.
#[test]
#[ignore = "requires AMD GPU with HIP runtime"]
fn test_preprocess_rgb_known_values() {
    init(0).expect("HIP init failed");
    let mut prep =
        GpuPreprocessor::new(2, 2, NormParams::zero_one()).expect("preprocessor creation failed");

    // Create a 2x2 RGB image: all pixels are (128, 64, 32)
    let rgb: Vec<u8> = vec![128, 64, 32, 128, 64, 32, 128, 64, 32, 128, 64, 32];
    let gpu_rgb = DeviceBuffer::alloc(rgb.len()).expect("alloc");
    gpu_rgb.copy_from_host(&rgb).expect("H2D");

    let result = prep
        .preprocess_rgb(gpu_rgb.as_ptr() as *const u8, 2, 2, 6)
        .expect("preprocess failed");

    // Read back: NCHW [3, 2, 2] = 12 floats
    let out = result.copy_to_host_typed::<f32>(12).expect("D2H");

    // With ZeroOne norm, pixel 128 → 128/255 ≈ 0.502
    let expected_r = 128.0 / 255.0;
    let expected_g = 64.0 / 255.0;
    let expected_b = 32.0 / 255.0;

    // NCHW: first 4 floats = R channel, next 4 = G, last 4 = B
    for (i, &val) in out[..4].iter().enumerate() {
        assert!(
            (val - expected_r).abs() < 0.02,
            "R[{i}]: expected {expected_r}, got {val}",
        );
    }
    for (i, &val) in out[4..8].iter().enumerate() {
        assert!(
            (val - expected_g).abs() < 0.02,
            "G[{}]: expected {expected_g}, got {val}",
            i + 4
        );
    }
    for (i, &val) in out[8..12].iter().enumerate() {
        assert!(
            (val - expected_b).abs() < 0.02,
            "B[{}]: expected {expected_b}, got {val}",
            i + 8
        );
    }
}
use crate::*;
