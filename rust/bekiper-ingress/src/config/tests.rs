use std::collections::HashMap;
use std::path::{Path, PathBuf};

use bekiper_core::scale::ScaleConfig;
use bekiper_core::types::{DetectionRecipe, TaskType};
use bekiper_core::video::FpsConfig;

use super::*;

fn config_with_scan_variants(variant_toml: &str) -> IngressConfig {
    toml::from_str(&format!(
        r#"
        [project]
        project_name = "variants"
        data_directory = "."

        [[scan]]
        video_dir = "Y-maze videos"
        task_type = "y_maze"
        meters_per_pixel = {{ method = "arm_length", meters = 0.35 }}

        {variant_toml}
        "#
    ))
    .unwrap()
}

fn imen_nort_config_with_rule(pattern: &str, recipe: DetectionRecipe) -> IngressConfig {
    let recipe_str = recipe.as_str();
    toml::from_str(&format!(
        r#"
        [project]
        project_name = "imen"
        data_directory = "."

        [[ingress_profile]]
        profile = "videomap"

        [[ingress_profile.tasks]]
        video_dir = "NORT videos"
        pose_data_dir = "deeplabcut/poses"
        task_type = "object_recognition"
        meters_per_pixel = {{ method = "arena_side", meters = 0.40 }}
        backend = ["manual_annotation"]
        annotations = {{ dir = "annotations/nort" }}

        [ingress_profile.tasks.recipe_rules]
        "{pattern}" = "{recipe_str}"
        "#
    ))
    .unwrap()
}

fn write_canonical_test_session(
    dir: &Path,
    animal: u32,
    stage: u32,
    phase_part: u32,
    perimeter_id: &str,
    labels: &[&str],
) {
    let root = dir.join("annotations/nort/bekiper");
    std::fs::create_dir_all(root.join("sessions")).unwrap();
    std::fs::create_dir_all(root.join("perimeters")).unwrap();
    std::fs::write(root.join("manifest.toml"), "").unwrap();
    let key = format!("animal_{animal}_stage_{stage}_part_{phase_part}");
    std::fs::write(
        root.join("sessions").join(format!("{key}.toml")),
        format!(
            "key = \"{key}\"\nvideo_number = {animal}\nanimal = {animal}\nstage = {stage}\nphase_part = {phase_part}\nperimeter_id = \"{perimeter_id}\"\ncoordinate_space = \"target\"\n",
        ),
    )
    .unwrap();
    let mut perimeter_toml = String::new();
    for label in labels {
        perimeter_toml.push_str(&format!(
            r#"
            [[perimeters]]
            label = "{label}"
            shape = "polygon"
            vertices = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
            "#
        ));
    }
    std::fs::write(
        root.join("perimeters").join(format!("{perimeter_id}.toml")),
        perimeter_toml,
    )
    .unwrap();
}

fn env_lookup<'a>(
    map: &'a HashMap<&'static str, &'static str>,
) -> impl Fn(&str) -> Option<String> + 'a {
    move |k| map.get(k).map(|v| (*v).to_string())
}

include!("tests/discovery.rs");
include!("tests/videomap.rs");
include!("tests/variants.rs");
include!("tests/gpu_overlays.rs");
include!("tests/pipeline_options.rs");
include!("tests/edge_cases.rs");
