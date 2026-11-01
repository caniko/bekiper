//! Read-only boundary qualification on externally sampled frames. This emits
//! pixel geometry with cropped-NORT defaults, without inventing physical
//! calibration or center rules. Environment overrides use the normal resolver.
use bekiper_detect::{arena::detect_arena_boundary, config::ArenaCalibrationConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let paths: Vec<_> = std::env::args().skip(1).collect();
    if paths.is_empty() {
        return Err("usage: detect_arena_boundary FRAME.png [FRAME.png ...]".into());
    }
    let frames = paths
        .iter()
        .map(|p| image::open(p).map(|i| i.to_luma8()))
        .collect::<Result<Vec<_>, _>>()?;
    let settings = ArenaCalibrationConfig::resolve_from(
        Some(ArenaCalibrationConfig::nort_default()),
        None,
        |key| std::env::var(key).ok(),
    )?;
    let result = detect_arena_boundary(&frames, &settings)?;
    println!(
        "{}",
        serde_json::to_string_pretty(
            &serde_json::json!({"frames":paths,"settings":settings,"perimeters":result.perimeters,"debug":result.debug})
        )?
    );
    Ok(())
}
