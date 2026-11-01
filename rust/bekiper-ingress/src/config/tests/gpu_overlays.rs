#[test]
fn gpu_config_defaults_match_pose_layer() {
    let cfg = GpuConfig::default();
    assert_eq!(cfg.device_id, bekiper_pose::infer::DEFAULT_GPU_DEVICE_ID);
    assert_eq!(
        cfg.max_batch_size,
        bekiper_pose::infer::DEFAULT_MAX_BATCH_SIZE
    );
    assert!(
        (cfg.vram_headroom_fraction - bekiper_pose::infer::DEFAULT_VRAM_HEADROOM_FRACTION).abs()
            < 1e-12
    );
    assert_eq!(
        cfg.fallback_batch_size,
        bekiper_pose::infer::DEFAULT_FALLBACK_BATCH_SIZE
    );
}

#[test]
fn gpu_config_to_pose_options_roundtrip() {
    let cfg = GpuConfig {
        device_id: 3,
        max_batch_size: 512,
        vram_headroom_fraction: 0.5,
        fallback_batch_size: 16,
    };
    let opts = cfg.to_pose_options();
    assert_eq!(opts.device_id, 3);
    assert_eq!(opts.max_batch_size, 512);
    assert!((opts.vram_headroom_fraction - 0.5).abs() < 1e-12);
    assert_eq!(opts.fallback_batch_size, 16);
}

#[test]
fn gpu_env_overlay_applies_valid_device_id() {
    let mut cfg = GpuConfig::default();
    let env = HashMap::from([("BEKIPER_GPU_DEVICE", "2")]);
    cfg.apply_env_overrides_from(env_lookup(&env));
    assert_eq!(cfg.device_id, 2);
}

#[test]
fn gpu_env_overlay_ignores_invalid_device_id() {
    let mut cfg = GpuConfig::default();
    let env = HashMap::from([("BEKIPER_GPU_DEVICE", "not-a-number")]);
    cfg.apply_env_overrides_from(env_lookup(&env));
    // Stays at the default 0
    assert_eq!(cfg.device_id, 0);
}

#[test]
fn gpu_env_overlay_applies_valid_max_batch_size() {
    let mut cfg = GpuConfig::default();
    let env = HashMap::from([("BEKIPER_GPU_MAX_BATCH_SIZE", "256")]);
    cfg.apply_env_overrides_from(env_lookup(&env));
    assert_eq!(cfg.max_batch_size, 256);
}

#[test]
fn gpu_env_overlay_rejects_zero_max_batch_size() {
    let mut cfg = GpuConfig::default();
    let env = HashMap::from([("BEKIPER_GPU_MAX_BATCH_SIZE", "0")]);
    cfg.apply_env_overrides_from(env_lookup(&env));
    // Stays at the default (not 0)
    assert_eq!(cfg.max_batch_size, 1024);
}

#[test]
fn gpu_env_overlay_applies_valid_vram_headroom() {
    let mut cfg = GpuConfig::default();
    let env = HashMap::from([("BEKIPER_GPU_VRAM_HEADROOM", "0.5")]);
    cfg.apply_env_overrides_from(env_lookup(&env));
    assert!((cfg.vram_headroom_fraction - 0.5).abs() < 1e-12);
}

#[test]
fn gpu_env_overlay_rejects_vram_headroom_above_one() {
    let mut cfg = GpuConfig::default();
    let env = HashMap::from([("BEKIPER_GPU_VRAM_HEADROOM", "1.5")]);
    cfg.apply_env_overrides_from(env_lookup(&env));
    // Stays at the default 0.75
    assert!((cfg.vram_headroom_fraction - 0.75).abs() < 1e-12);
}

#[test]
fn gpu_env_overlay_rejects_negative_vram_headroom() {
    let mut cfg = GpuConfig::default();
    let env = HashMap::from([("BEKIPER_GPU_VRAM_HEADROOM", "-0.1")]);
    cfg.apply_env_overrides_from(env_lookup(&env));
    assert!((cfg.vram_headroom_fraction - 0.75).abs() < 1e-12);
}

#[test]
fn gpu_env_overlay_rejects_nonsense_vram_headroom() {
    let mut cfg = GpuConfig::default();
    let env = HashMap::from([("BEKIPER_GPU_VRAM_HEADROOM", "high")]);
    cfg.apply_env_overrides_from(env_lookup(&env));
    assert!((cfg.vram_headroom_fraction - 0.75).abs() < 1e-12);
}

#[test]
fn gpu_env_overlay_applies_valid_fallback_batch_size() {
    let mut cfg = GpuConfig::default();
    let env = HashMap::from([("BEKIPER_GPU_FALLBACK_BATCH_SIZE", "8")]);
    cfg.apply_env_overrides_from(env_lookup(&env));
    assert_eq!(cfg.fallback_batch_size, 8);
}

#[test]
fn gpu_env_overlay_rejects_zero_fallback_batch_size() {
    let mut cfg = GpuConfig::default();
    let env = HashMap::from([("BEKIPER_GPU_FALLBACK_BATCH_SIZE", "0")]);
    cfg.apply_env_overrides_from(env_lookup(&env));
    assert_eq!(cfg.fallback_batch_size, 32);
}

#[test]
fn gpu_env_overlay_beats_file_values() {
    // Simulates: file sets device_id=1 and max_batch_size=64;
    // env overlay should win on both.
    let mut cfg = GpuConfig {
        device_id: 1,
        max_batch_size: 64,
        vram_headroom_fraction: 0.9,
        fallback_batch_size: 16,
    };
    let env = HashMap::from([
        ("BEKIPER_GPU_DEVICE", "5"),
        ("BEKIPER_GPU_MAX_BATCH_SIZE", "2048"),
        ("BEKIPER_GPU_VRAM_HEADROOM", "0.6"),
        ("BEKIPER_GPU_FALLBACK_BATCH_SIZE", "48"),
    ]);
    cfg.apply_env_overrides_from(env_lookup(&env));
    assert_eq!(cfg.device_id, 5);
    assert_eq!(cfg.max_batch_size, 2048);
    assert!((cfg.vram_headroom_fraction - 0.6).abs() < 1e-12);
    assert_eq!(cfg.fallback_batch_size, 48);
}

#[test]
fn gpu_env_overlay_empty_env_preserves_config() {
    let mut cfg = GpuConfig {
        device_id: 2,
        max_batch_size: 128,
        vram_headroom_fraction: 0.8,
        fallback_batch_size: 24,
    };
    let env: HashMap<&'static str, &'static str> = HashMap::new();
    cfg.apply_env_overrides_from(env_lookup(&env));
    assert_eq!(cfg.device_id, 2);
    assert_eq!(cfg.max_batch_size, 128);
    assert!((cfg.vram_headroom_fraction - 0.8).abs() < 1e-12);
    assert_eq!(cfg.fallback_batch_size, 24);
}

#[test]
fn gpu_env_overlay_partial_keys_only_touches_specified() {
    let mut cfg = GpuConfig {
        device_id: 0,
        max_batch_size: 1024,
        vram_headroom_fraction: 0.75,
        fallback_batch_size: 32,
    };
    // Only override device_id; everything else stays.
    let env = HashMap::from([("BEKIPER_GPU_DEVICE", "7")]);
    cfg.apply_env_overrides_from(env_lookup(&env));
    assert_eq!(cfg.device_id, 7);
    assert_eq!(cfg.max_batch_size, 1024);
    assert!((cfg.vram_headroom_fraction - 0.75).abs() < 1e-12);
    assert_eq!(cfg.fallback_batch_size, 32);
}
