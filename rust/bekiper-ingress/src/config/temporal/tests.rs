use super::*;
use bekiper_feature::tolerance::DurationComparison;
use std::collections::HashMap;

#[test]
fn imen_configs_use_arm_start_contexts_with_dwell_only_at_zero() {
    for document in [
        include_str!("../../../../../examples/neo/imen/bekiper_config.toml"),
        include_str!("../../../../../examples/neo/imen/bekiper_config.ymaze_only.toml"),
    ] {
        let config: IngressConfig = toml::from_str(document).unwrap();
        config.validate().unwrap();
        let task = config
            .ingress_profile
            .iter()
            .flat_map(|p| &p.tasks)
            .find(|t| t.task_type == bekiper_core::types::TaskType::YMaze)
            .unwrap();
        let base = task.task_params.as_ref().unwrap();
        // The four Y-maze primaries, as thresholds or as explicit contexts.
        let primaries: Vec<String> = match &config.post_analysis.contexts {
            Some(contexts) => contexts
                .iter()
                .filter(|c| c.assay == "ymaze")
                .map(|c| c.variant.clone())
                .collect(),
            None => config
                .post_analysis
                .ymaze_primary_thresholds
                .clone()
                .unwrap()
                .iter()
                .map(|pct| format!("arm{pct}"))
                .collect(),
        };
        assert_eq!(primaries.len(), 4);
        assert!(primaries.iter().all(|p| p.starts_with("arm")));
        let names: Vec<_> = task.variants.iter().map(|v| v.name.as_str()).collect();
        assert_eq!(names, ["arm0_dwell1s", "arm25", "arm45", "arm65"]);
        for variant in &task.variants {
            let resolved = base.merged(&variant.task_params);
            let ymaze = resolved.ymaze.unwrap();
            let policy = ymaze.temporal_policy.unwrap();
            let dwell = variant.name == "arm0_dwell1s";
            assert_eq!(
                (policy.min_duration_seconds, policy.max_gap_seconds),
                if dwell { (1.0, 1.0) } else { (0.0, 0.0) },
                "{}",
                variant.name
            );
            assert_eq!(policy.duration_comparison, DurationComparison::AtLeast);
            assert_eq!(
                ymaze.threshold_geometry,
                Some(bekiper_behaviour::radial_arm::YMazeThresholdGeometry::ArmStartCrosslineV1)
            );
            assert_eq!(ymaze.apply_dwell_filter, Some(dwell));
            assert_eq!(ymaze.zone_max_gap_seconds, Some(1.0));
            assert_eq!(ymaze.pose_gap_fill_seconds, Some(0.0));
            assert_eq!(ymaze.arm_membership_hysteresis, Some(false));
        }
    }
}

#[test]
fn inherited_temporal_policy_is_validated_at_the_effective_variant_threshold() {
    for entry in ["scan", "ingress_profile.tasks"] {
        let profile = if entry == "scan" {
            ""
        } else {
            "[[ingress_profile]]\nprofile = 'videomap'\n"
        };
        for (base_threshold, variant_threshold, valid) in [
            (0.0, Some(0.45), false),
            (0.0, None, true),
            (0.45, Some(0.0), true),
        ] {
            let threshold = variant_threshold
                .map(|t| format!("proximal_threshold = {t}\n"))
                .unwrap_or_default();
            let document = format!(
                "[project]\nproject_name = 'inheritance'\ndata_directory = '.'\n{profile}\
                    [[{entry}]]\nvideo_dir = 'unread'\ntask_type = 'y_maze'\nmeters_per_pixel = 0.001\nfps = 30.0\n\
                    [{entry}.task_params.ymaze]\nproximal_threshold = {base_threshold}\n\
                    [{entry}.task_params.ymaze.temporal_policy]\nmin_duration_seconds = 0.0\nmax_gap_seconds = 0.1\n\
                    [[{entry}.variants]]\nname = 'inherited'\n[{entry}.variants.ymaze]\n{threshold}"
            );
            let config: IngressConfig = toml::from_str(&document).unwrap();
            assert_eq!(config.validate().is_ok(), valid, "{document}");
        }
    }
}

#[test]
fn partial_dwell_overlay_preserves_effective_variant_policy_inheritance() {
    for entry in ["scan", "ingress_profile.tasks"] {
        let profile = if entry == "scan" {
            ""
        } else {
            "[[ingress_profile]]\nprofile = 'videomap'\n"
        };
        let document = format!(
            "[project]\nproject_name = 'overlay-inheritance'\ndata_directory = '.'\n{profile}\
                [[{entry}]]\nvideo_dir = 'unread'\ntask_type = 'y_maze'\nmeters_per_pixel = 0.001\nfps = 30.0\n\
                [{entry}.task_params.ymaze.temporal_policy]\nmin_duration_seconds = 2.0\nduration_comparison = 'greater_than'\n\
                [[{entry}.variants]]\nname = 'inherited'\n[{entry}.variants.ymaze]\nproximal_threshold = 0.65\n\
                [[{entry}.variants]]\nname = 'explicit'\n[{entry}.variants.ymaze.temporal_policy]\nmin_duration_seconds = 0.5\n"
        );
        for (overrides, inherited, explicit) in [
            (
                DwellOverrides {
                    seconds: Some(0.0),
                    ..Default::default()
                },
                (0.0, DurationComparison::GreaterThan),
                (0.0, DurationComparison::AtLeast),
            ),
            (
                DwellOverrides {
                    comparison: Some(DurationComparison::AtLeast),
                    ..Default::default()
                },
                (2.0, DurationComparison::AtLeast),
                (0.5, DurationComparison::AtLeast),
            ),
        ] {
            let mut config: IngressConfig = toml::from_str(&document).unwrap();
            config.apply_ymaze_dwell_overrides(overrides).unwrap();
            config.validate().unwrap();
            let variants = if entry == "scan" {
                &config.scan[0].variants
            } else {
                &config.ingress_profile[0].tasks[0].variants
            };
            for (variant, (seconds, comparison)) in variants.iter().zip([inherited, explicit]) {
                let policy = variant
                    .task_params
                    .ymaze
                    .as_ref()
                    .unwrap()
                    .temporal_policy
                    .unwrap();
                assert_eq!(policy.min_duration_seconds, seconds, "{}", variant.name);
                assert_eq!(policy.duration_comparison, comparison, "{}", variant.name);
            }
        }
    }
}

#[test]
fn comparison_only_overlay_preserves_one_second_ymaze_default() {
    let mut cfg: IngressConfig = toml::from_str(
        r#"
[project]
project_name = "default-dwell"
data_directory = "."
[[scan]]
video_dir = "unread"
task_type = "y_maze"
meters_per_pixel = 0.001
fps = 30.0
[[scan.variants]]
name = "deep"
[scan.variants.ymaze]
proximal_threshold = 0.65
"#,
    )
    .unwrap();
    cfg.apply_ymaze_dwell_overrides(DwellOverrides {
        comparison: Some(DurationComparison::GreaterThan),
        ..Default::default()
    })
    .unwrap();
    for params in [
        cfg.scan[0].task_params.as_ref().unwrap(),
        &cfg.scan[0].variants[0].task_params,
    ] {
        let policy = params.ymaze.as_ref().unwrap().temporal_policy.unwrap();
        assert_eq!(policy.min_duration_seconds, 1.0);
        assert_eq!(policy.duration_comparison, DurationComparison::GreaterThan);
    }
}

#[test]
fn constructor_beats_environment_and_variant_file_policy() {
    let file =
        std::env::temp_dir().join(format!("bekiper-dwell-config-{}.toml", std::process::id()));
    std::fs::write(
        &file,
        r#"
[project]
project_name = "dwell-test"
data_directory = "."
[[scan]]
video_dir = "unread"
task_type = "y_maze"
meters_per_pixel = 0.001
fps = 30.0
[scan.task_params.ymaze.temporal_policy]
min_duration_seconds = 0.5
duration_comparison = "greater_than"
[[scan.variants]]
name = "deep"
[scan.variants.ymaze]
proximal_threshold = 0.65
[scan.variants.ymaze.temporal_policy]
min_duration_seconds = 2.0
[[scan.variants]]
name = "inherited"
[scan.variants.ymaze]
proximal_threshold = 0.45
"#,
    )
    .unwrap();
    let env = HashMap::from([("BEKIPER_YMAZE_DWELL_SECONDS", "1.0")]);
    let mut cfg = IngressConfig::from_file_with_post_analysis_overrides_from(
        &file,
        &Default::default(),
        |key| env.get(key).map(|v| v.to_string()),
    )
    .unwrap();
    let explicit = DwellOverrides {
        seconds: Some(0.0),
        comparison: Some(DurationComparison::AtLeast),
    };
    let from_constructor =
        IngressConfig::from_file_with_overrides_from(&file, &Default::default(), explicit, |key| {
            match key {
                "BEKIPER_YMAZE_DWELL_SECONDS" | "BEKIPER_YMAZE_DWELL_COMPARISON" => {
                    Some("invalid".into())
                }
                _ => None,
            }
        })
        .unwrap();
    std::fs::remove_file(file).unwrap();
    for variant in &from_constructor.scan[0].variants {
        let policy = variant
            .task_params
            .ymaze
            .as_ref()
            .unwrap()
            .temporal_policy
            .unwrap();
        assert_eq!(policy.min_duration_seconds, 0.0);
        assert_eq!(policy.duration_comparison, DurationComparison::AtLeast);
    }
    let base = cfg.scan[0]
        .task_params
        .as_ref()
        .unwrap()
        .ymaze
        .as_ref()
        .unwrap()
        .temporal_policy
        .unwrap();
    assert_eq!(base.min_duration_seconds, 1.0);
    assert_eq!(base.duration_comparison, DurationComparison::GreaterThan);
    let inherited = cfg.scan[0].variants[1]
        .task_params
        .ymaze
        .as_ref()
        .unwrap()
        .temporal_policy
        .unwrap();
    assert_eq!(
        inherited.duration_comparison,
        DurationComparison::GreaterThan
    );
    assert_eq!(
        cfg.scan[0].variants[0]
            .task_params
            .ymaze
            .as_ref()
            .unwrap()
            .temporal_policy
            .unwrap()
            .min_duration_seconds,
        1.0
    );
    cfg.apply_ymaze_dwell_overrides(DwellOverrides {
        seconds: Some(0.0),
        comparison: Some(DurationComparison::AtLeast),
    })
    .unwrap();
    let variant = cfg.scan[0].variants[0].task_params.ymaze.as_ref().unwrap();
    assert_eq!(variant.proximal_threshold, Some(0.65));
    assert_eq!(variant.temporal_policy.unwrap().min_duration_seconds, 0.0);
    assert_eq!(
        variant.temporal_policy.unwrap().duration_comparison,
        DurationComparison::AtLeast
    );
    cfg.scan[0].variants[0]
        .task_params
        .ymaze
        .as_mut()
        .unwrap()
        .temporal_policy
        .as_mut()
        .unwrap()
        .min_duration_seconds = -1.0;
    assert!(cfg.validate().is_err());
}
