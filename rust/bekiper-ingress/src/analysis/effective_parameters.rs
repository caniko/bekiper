//! `[detailed.effective_parameters]`: every resolved scoring knob (defaults
//! included) under its config key name, with the layer that set it.
//!
//! `values`/`source` are flat dotted-key maps, e.g.
//! `"exploration.min_bout_seconds"`, `"ymaze.temporal_policy.min_duration_seconds"`,
//! `"arena_calibration.center_zone_rule.method"`, `"scale.method"`.
//! Sources: `default | toml | env | cli | derived`.
//!
//! ponytail: `toml` means "present in the experiment's merged `task_params`"
//! (so a nested table with serde defaults counts as configured and pose-label
//! defaults filled at expansion count as toml); `arena_calibration.*` is `toml`
//! when its key was written in the config (even at the built-in default value). Env/CLI layers come from
//! `apply_override_sources`, matching `--set`/`BEKIPER_SET` keys and the dedicated
//! shorthands, which `IngressConfig` records as `AppliedOverride`s too.
use bekiper_core::scale::ScaleDerivationInfo;
use bekiper_core::types::TaskType;
use serde_json::{Map, Value, json};

use crate::config::{AppliedOverride, ExperimentConfig};

/// Nested JSON -> flat dotted keys. Arrays are leaves; nulls and empty tables vanish.
pub(crate) fn flatten(prefix: &str, value: &Value, out: &mut Map<String, Value>) {
    match value {
        Value::Null => {}
        Value::Object(map) => {
            for (key, v) in map {
                let full = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}.{key}")
                };
                flatten(&full, v, out);
            }
        }
        leaf => {
            out.insert(prefix.into(), leaf.clone());
        }
    }
}

fn flat(prefix: &str, value: &impl serde::Serialize) -> Map<String, Value> {
    let mut out = Map::new();
    flatten(
        prefix,
        &serde_json::to_value(value).unwrap_or_default(),
        &mut out,
    );
    out
}

const WHISKER: &str = "exploration.whisker";

/// `task` is the behaviour task's nested `effective_parameters()`; `scale` is the
/// derivation the scorer used (so `+frame_center_fallback` / reuse is visible).
pub(crate) fn effective_parameters(
    experiment: &ExperimentConfig,
    task: &Value,
    scale: &ScaleDerivationInfo,
) -> Value {
    let (mut values, mut source) = (Map::new(), Map::new());
    let mut put = |key: String, value: Value, src: &str| {
        source.insert(key.clone(), src.into());
        values.insert(key, value);
    };
    let configured = flat("", &experiment.task_params);
    let task_flat = flat("", task);
    let aged = configured.contains_key(&format!("{WHISKER}.length_source"));
    for (key, value) in task_flat {
        let derived = key
            .strip_prefix(WHISKER)
            .is_some_and(|k| matches!(k, ".length_m" | ".age_days" | ".length_source"));
        let src = if derived {
            "derived"
        } else if configured.contains_key(&key) {
            "toml"
        } else {
            "default"
        };
        if key == format!("{WHISKER}.length_m") {
            // The config spelling of the same value (`length = { meters = .. }`).
            let src = if aged {
                "derived"
            } else if configured.contains_key(&format!("{WHISKER}.length.meters")) {
                "toml"
            } else {
                "default"
            };
            put(format!("{WHISKER}.length.meters"), value.clone(), src);
        }
        put(key, value, src);
    }

    // The age-table spec is replaced by `{ meters }` at expansion; export the original.
    if let Some(spec) = experiment
        .task_params
        .as_ref()
        .and_then(|p| p.exploration.as_ref())
        .and_then(|e| e.whisker.as_ref())
        .and_then(|w| w.age_table.as_ref())
    {
        for (key, value) in flat(&format!("{WHISKER}.length"), spec) {
            put(key, value, "toml");
        }
    }

    if experiment.task_type != TaskType::YMaze {
        let default = flat(
            "arena_calibration",
            &bekiper_detect::config::ArenaCalibrationConfig::default(),
        );
        let set = experiment
            .detection
            .as_ref()
            .and_then(|d| d.arena_calibration.as_ref());
        for (key, value) in set.map_or_else(|| default.clone(), |c| flat("arena_calibration", c)) {
            // Key presence in the configured table decides `toml`, as for task params.
            let field = key["arena_calibration.".len()..].split('.').next();
            let src = if set.is_some_and(|c| c.configured.iter().any(|k| Some(*k) == field)) {
                "toml"
            } else {
                "default"
            };
            put(key, value, src);
        }
    }
    match serde_json::to_value(&experiment.meters_per_pixel).unwrap_or_default() {
        Value::Object(_) => flat("meters_per_pixel", &experiment.meters_per_pixel)
            .into_iter()
            .for_each(|(k, v)| put(k, v, "toml")),
        value => {
            put("meters_per_pixel.method".into(), "manual".into(), "toml");
            put("meters_per_pixel.value".into(), value, "toml");
        }
    }
    flat("scale", scale)
        .into_iter()
        .for_each(|(k, v)| put(k, v, "derived"));
    // A disabled fan reports only `enabled`; its configured (inert) keys stay recorded.
    for (key, value) in configured {
        if key.starts_with(WHISKER) && !values.contains_key(&key) {
            source.insert(key.clone(), "toml".into());
            values.insert(key, value);
        }
    }
    json!({"values": values, "source": source})
}

/// Upgrade `source` to `env`/`cli` for keys that a `BEKIPER_SET`/`--set` entry
/// wrote for this experiment's task (and variant).
pub(crate) fn apply_override_sources(
    parameters: &mut Value,
    task: TaskType,
    experiment: &str,
    record: &[AppliedOverride],
) {
    let Some(source) = parameters.get_mut("source").and_then(Value::as_object_mut) else {
        return;
    };
    for rec in record {
        let (head, rest) = rec.key.split_once('.').unwrap_or((&rec.key, ""));
        let (name, variant) = head
            .split_once('@')
            .map_or((head, None), |(t, v)| (t, Some(v)));
        let prefix = if let Ok(t) = name.parse::<TaskType>() {
            if t != task || variant.is_some_and(|v| !experiment.ends_with(&format!("__{v}"))) {
                continue;
            }
            rest.to_string()
        } else if let Some((_, tail)) = rec.key.split_once("detection.arena_calibration.") {
            format!("arena_calibration.{tail}")
        } else if let Some(at) = rec.key.find("meters_per_pixel") {
            rec.key[at..].to_string()
        } else {
            continue; // root keys outside scoring parameters
        };
        let nested = format!("{prefix}.");
        for (key, src) in source.iter_mut() {
            if *key == prefix || key.starts_with(&nested) {
                *src = rec.layer.into();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn override_sources_label_scoring_keys_and_ignore_other_roots() {
        let mut none = json!({"values": {}});
        apply_override_sources(&mut none, TaskType::OpenField, "x", &[]);
        assert_eq!(none, json!({"values": {}}));

        let record = |key: &str| AppliedOverride {
            layer: "cli",
            key: key.into(),
            targets: vec![],
            value: toml::Value::Boolean(true),
        };
        let mut parameters = json!({"source": {
            "arena_calibration.a": "default", "meters_per_pixel": "toml", "review.workers": "default"}});
        apply_override_sources(
            &mut parameters,
            TaskType::OpenField,
            "x",
            &[
                record("detection.arena_calibration.a"),
                record("ingress.meters_per_pixel"),
                record("review.workers"),
            ],
        );
        assert_eq!(
            parameters["source"],
            json!({"arena_calibration.a": "cli", "meters_per_pixel": "cli", "review.workers": "default"})
        );
    }

    #[test]
    fn a_derived_whisker_length_without_config_reports_the_default_spelling() {
        let experiment: ExperimentConfig = toml::from_str(
            "name = \"x\"\ntask_type = \"object_recognition\"\nmeters_per_pixel = 0.001\n",
        )
        .unwrap();
        let scale = ScaleDerivationInfo {
            method: "arena_side".into(),
            physical_meters: Some(0.4),
            pixel_measurement: Some(200.0),
            arm_pixel_spans: Vec::new(),
            reference_label: Some("arena".into()),
            endpoints: None,
            meters_per_pixel: 0.002,
            meters_per_pixel_y: None,
        };
        let task = json!({"exploration": {"whisker": {"length_m": 0.0117}}});
        let parameters = effective_parameters(&experiment, &task, &scale);
        assert_eq!(
            parameters["source"]["exploration.whisker.length.meters"],
            "default"
        );
    }

    #[test]
    fn flatten_keeps_arrays_and_drops_nulls() {
        let mut out = Map::new();
        flatten(
            "",
            &json!({"a": {"b": [[0, 1.5]], "c": null, "d": {}}, "e": true}),
            &mut out,
        );
        assert_eq!(Value::Object(out), json!({"a.b": [[0, 1.5]], "e": true}));
    }

    #[test]
    fn arena_key_written_at_default_value_is_toml() {
        let default = bekiper_detect::config::ArenaCalibrationConfig::default();
        let body = format!("min_edge_score = {:?}\n", default.min_edge_score);
        let set: bekiper_detect::config::ArenaCalibrationConfig = toml::from_str(&body).unwrap();
        assert_eq!(set.min_edge_score, default.min_edge_score);
        assert_eq!(set.configured, ["min_edge_score"]);
    }

    #[test]
    fn override_sources_match_task_and_variant() {
        let rec = |key: &str, layer| AppliedOverride {
            layer,
            key: key.into(),
            targets: Vec::new(),
            value: toml::Value::Float(1.0),
        };
        let mut p = json!({"source": {"ymaze.proximal_threshold": "toml", "exploration.min_bout_seconds": "toml",
            "ymaze.start_mode": "default", "meters_per_pixel.meters": "toml"}});
        apply_override_sources(
            &mut p,
            TaskType::YMaze,
            "a_seq1__arm25",
            &[
                rec("ymaze@arm25.ymaze.proximal_threshold", "cli"),
                rec("ymaze@other.ymaze.start_mode", "cli"),
                rec("nort.exploration.min_bout_seconds", "env"),
                rec("ingress_profile[0].tasks[0].meters_per_pixel.meters", "env"),
            ],
        );
        assert_eq!(p["source"]["ymaze.proximal_threshold"], "cli");
        assert_eq!(p["source"]["ymaze.start_mode"], "default");
        assert_eq!(p["source"]["exploration.min_bout_seconds"], "toml");
        assert_eq!(p["source"]["meters_per_pixel.meters"], "env");
    }

    /// Every configured leaf under each expanded experiment's `task_params`,
    /// `detection.arena_calibration` and `meters_per_pixel` must be exported.
    #[test]
    fn imen_configs_export_every_configured_key() {
        use crate::config::IngressConfig;
        use crate::stage::Stage;
        use bekiper_core::inspection::PerimeterSpec;

        let perimeters = |task: TaskType| match task {
            TaskType::YMaze => vec![PerimeterSpec::RadialMaze {
                label: "maze".into(),
                center_vertices: vec![(-0.1, -0.1), (0.1, -0.1), (0.1, 0.1), (-0.1, 0.1)],
                arms: vec![
                    vec![(0.1, -0.1), (1.0, -0.1), (1.0, 0.1), (0.1, 0.1)],
                    vec![(-0.1, 0.1), (0.1, 0.1), (0.1, 1.0), (-0.1, 1.0)],
                    vec![(-1.0, -0.1), (-0.1, -0.1), (-0.1, 0.1), (-1.0, 0.1)],
                ],
            }],
            _ => vec![
                PerimeterSpec::Rectangle {
                    label: "arena".into(),
                    center_x: 0.2,
                    center_y: 0.2,
                    width: 0.4,
                    height: 0.4,
                },
                PerimeterSpec::Circle {
                    label: "object_A".into(),
                    center_x: 0.1,
                    center_y: 0.1,
                    radius: 0.02,
                },
                PerimeterSpec::Circle {
                    label: "object_B".into(),
                    center_x: 0.3,
                    center_y: 0.3,
                    radius: 0.02,
                },
            ],
        };
        let scale = ScaleDerivationInfo {
            method: "arena_side".into(),
            physical_meters: Some(0.4),
            pixel_measurement: Some(200.0),
            arm_pixel_spans: Vec::new(),
            reference_label: Some("arena".into()),
            endpoints: None,
            meters_per_pixel: 0.002,
            meters_per_pixel_y: None,
        };
        let imen = concat!(env!("CARGO_MANIFEST_DIR"), "/../../examples/neo/imen");
        let mut checked = 0;
        for (name, body) in [
            (
                "full",
                std::fs::read_to_string(format!("{imen}/bekiper_config.toml")).unwrap(),
            ),
            (
                "ymaze",
                std::fs::read_to_string(format!("{imen}/bekiper_config.ymaze_only.toml")).unwrap(),
            ),
        ] {
            let root = std::env::temp_dir()
                .join(format!("bekiper_effective_{name}_{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&root);
            for video in [
                "NORT videos/NORT (post)/T1/Config A/post.T1.configA.video-10.mp4",
                "OFT videos/OFT (post)/post.video-1.mp4",
                "Y-maze videos/Y-maze (post)/post.video-1.mp4",
            ] {
                let path = root.join(video);
                std::fs::create_dir_all(path.parent().unwrap()).unwrap();
                std::fs::write(&path, []).unwrap();
                let pose = root
                    .join("deeplabcut/poses")
                    .join(format!("{}.h5", video.strip_suffix(".mp4").unwrap()));
                std::fs::create_dir_all(pose.parent().unwrap()).unwrap();
                std::fs::write(pose, []).unwrap();
            }
            for dir in ["annotations/nort", "annotations/ymaze", "output"] {
                std::fs::create_dir_all(root.join(dir)).unwrap();
            }
            std::fs::write(
                root.join("annotations/nort/metadata.csv"),
                "Phase,PhasePart,TrialNumber,Stage,Animal\nA,2,7,0,1\n",
            )
            .unwrap();
            std::fs::copy(
                format!("{imen}/mice_overview.xlsx"),
                root.join("mice_overview.xlsx"),
            )
            .unwrap();
            let body = body.replace(
                "\n[aggregate.metadata]\npath = \"animal_metadata.xlsx\"\nsheet = \"Updated_Mice-overview\"\nmissing_policy = \"exclude_missing\"\n",
                "\n",
            );
            let path = root.join("bekiper_config.toml");
            std::fs::write(&path, body).unwrap();
            let mut config = IngressConfig::from_file(&path).unwrap();
            config
                .expand_for_validation(&root, &[Stage::Detect])
                .unwrap();
            assert!(!config.experiments.is_empty());

            for experiment in &config.experiments {
                let task = crate::analysis::behaviour::build_task(
                    experiment,
                    &perimeters(experiment.task_type),
                    30.0,
                )
                .unwrap()
                .effective_parameters()
                .unwrap();
                let parameters = effective_parameters(experiment, &task, &scale);
                let (values, source) = (&parameters["values"], &parameters["source"]);
                let mut configured = flat("", &experiment.task_params);
                configured.extend(
                    flat("meters_per_pixel", &experiment.meters_per_pixel)
                        .into_iter()
                        .filter(|_| true),
                );
                if let Some(c) = experiment
                    .detection
                    .as_ref()
                    .and_then(|d| d.arena_calibration.as_ref())
                {
                    configured.extend(flat("arena_calibration", c));
                }
                for (key, want) in &configured {
                    assert_eq!(&values[key], want, "{}: {key}", experiment.name);
                    assert!(
                        matches!(source[key].as_str(), Some("toml" | "derived" | "default")),
                        "{}: {key}",
                        experiment.name
                    );
                    if key.starts_with("exploration.") || key.starts_with("ymaze.") {
                        assert_ne!(source[key], "default", "{}: {key}", experiment.name);
                    }
                    checked += 1;
                }
                assert_eq!(values["scale.method"], "arena_side");
                // Fixed-length and disabled-fan variants have no derived length; the loop above checks them.
                if experiment.task_type == TaskType::ObjectRecognition
                    && configured.contains_key("exploration.whisker.length_source")
                    && values["exploration.whisker.enabled"] == true
                {
                    // The age table is resolved to a length at expansion; both stay visible.
                    assert_eq!(values["exploration.whisker.apex_fraction"], json!(0.9));
                    assert_eq!(values["exploration.whisker.length.method"], "age_table");
                    assert_eq!(source["exploration.whisker.length.method"], "toml");
                    assert_eq!(
                        values["exploration.whisker.length.table"],
                        json!([[0.0, 0.0117], [400.0, 0.0117]])
                    );
                    assert_eq!(source["exploration.whisker.length_m"], "derived");
                    assert!(values["exploration.whisker.age_days"].is_number());
                }
            }
            std::fs::remove_dir_all(&root).ok();
        }
        assert!(checked > 40, "only {checked} keys checked");
    }

    #[test]
    fn parameters_survive_toml() {
        use crate::aggregate::TrialResult;
        let trial = |name: &str, key: &str| TrialResult {
            experiment: name.into(),
            task_type: TaskType::OpenField,
            analysis_role: None,
            fps: 30.0,
            meters_per_pixel: 0.002,
            animal_metadata: Value::Null,
            session_context: Value::Null,
            summaries: Vec::new(),
            detailed: json!({"effective_parameters": {"values": {key: 0.5, "scale.method": "arena_side"},
                "source": {key: "toml", "scale.method": "derived"}}}),
        };
        let toml = bekiper_core::inspection::json_value_to_toml_string(
            &trial("a", "min_dwell_seconds").detailed,
        )
        .unwrap();
        assert!(toml.contains("\"scale.method\" = \"arena_side\""), "{toml}");
    }
}
