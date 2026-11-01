use std::collections::BTreeSet;

use super::*;
use crate::types::TaskType;

fn trial(id: &str, task: &str) -> TrialFacts {
    TrialFacts {
        id: id.into(),
        task_type: task.into(),
        ..Default::default()
    }
}

fn set(items: &[&str]) -> BTreeSet<String> {
    items.iter().map(|s| s.to_string()).collect()
}

fn some(s: &str) -> Option<String> {
    Some(s.into())
}

fn exp<'a>(r: &'a ResolvedPresentation, key: &str) -> &'a ResolvedExperiment {
    r.experiments.iter().find(|e| e.key == key).unwrap()
}

fn has(e: &ResolvedExperiment, block: &str) -> bool {
    e.blocks.iter().any(|b| b == block)
}

fn resolve_all(trials: Vec<TrialFacts>) -> ResolvedPresentation {
    resolve(&CorpusFacts { trials }, &PresentationOverrides::default()).unwrap()
}

fn ymaze_contexts() -> &'static Contexts {
    descriptor("ymaze").unwrap().contexts.as_ref().unwrap()
}

/// The Neo/Imen corpus: OFT habituation, NORT T1/T2 and the Y-maze variants.
fn imen() -> CorpusFacts {
    let oft = |id: &str, timepoint: &str| TrialFacts {
        timepoint: some(timepoint),
        phase: some("B"),
        metrics_keys: set(&["center_time_pct"]),
        feature_kinds: set(&["center_periphery"]),
        has_video: true,
        ..trial(id, "open_field")
    };
    let nort = |id: &str, phase: Option<&str>, timepoint: &str| TrialFacts {
        phase: phase.map(Into::into),
        timepoint: some(timepoint),
        ..trial(id, "object_recognition")
    };
    let mut trials = vec![
        oft("oft_1", "pre"),
        oft("oft_2", "post"),
        nort("t1_pre", Some("training"), "pre"),
        TrialFacts {
            perimeters: vec!["familiar".into(), "novel".into()],
            ..nort("t2_pre", Some("test"), "pre")
        },
        nort("t2_post", Some("test"), "post"),
        TrialFacts {
            timepoint: None,
            stage: some("2"),
            phase_part: some("1"),
            ..nort("t2_nophase", None, "pre")
        },
    ];
    for v in [
        "arm0_dwell1s",
        "arm25",
        "arm45",
        "arm65",
        "proximal25",
        "proximal45",
        "proximal65",
        "compare_0pct_dwell0s",
    ] {
        trials.push(TrialFacts {
            variant: some(v),
            ..trial(&format!("ym_1__{v}"), "y_maze")
        });
    }
    trials.push(TrialFacts {
        analysis_role: some("ymaze__arm45"),
        ..trial("ym_2", "y_maze")
    });
    CorpusFacts { trials }
}

#[test]
fn descriptors_are_consistent() {
    let all = descriptors();
    let ids: BTreeSet<_> = all.iter().map(|d| &d.id).collect();
    let keys: BTreeSet<_> = all.iter().map(|d| &d.key).collect();
    assert_eq!((all.len(), ids.len(), keys.len()), (5, 5, 5));
    assert_eq!(all.last().unwrap().id, "generic");

    for d in all.iter().filter(|d| d.id != "generic") {
        let task: TaskType = d.task_type.parse().unwrap();
        for alias in &d.aliases {
            assert_eq!(alias.parse::<TaskType>().unwrap(), task, "{alias}");
        }
    }
    // Exhaustive: a new TaskType variant must get a descriptor.
    let _ = |t: TaskType| match t {
        TaskType::ObjectRecognition
        | TaskType::YMaze
        | TaskType::OpenField
        | TaskType::Cheeseboard => (),
    };
    for t in [
        TaskType::ObjectRecognition,
        TaskType::YMaze,
        TaskType::OpenField,
        TaskType::Cheeseboard,
    ] {
        assert_eq!(descriptor_for(t.as_str()).task_type, t.as_str());
        let owners = all
            .iter()
            .filter(|d| d.task_type.parse::<TaskType>() == Ok(t))
            .count();
        assert_eq!(owners, 1, "{t}");
    }

    let sections: BTreeSet<_> = all
        .iter()
        .flat_map(|d| &d.sections)
        .map(|s| &s.id)
        .collect();
    for step in all.iter().flat_map(|d| &d.steps) {
        assert!(sections.contains(&step.section), "{}", step.id);
        if let Some(e) = &step.requires_experiment {
            assert!(ids.contains(e), "{e}");
        }
    }

    assert_eq!(descriptor_for("rotarod").id, "generic");
    assert_eq!(descriptor_for("oul").key, "nort");
    assert_eq!(descriptor_for("radial_arm").key, "ymaze");
}

#[test]
fn contract_matches_descriptors() {
    let source = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../design/review-package/contract.toml"
    ))
    .unwrap();
    let contract = Contract::parse(&source).unwrap();
    contract.check_descriptors(descriptors()).unwrap();
    let archetypes = [
        Archetype::ObjectExploration,
        Archetype::ZoneOccupancy,
        Archetype::ChoiceSequence,
        Archetype::GoalSearch,
        Archetype::Generic,
    ]
    .map(Archetype::as_str);
    assert_eq!(contract.axes["archetype"], archetypes);
}

#[test]
fn contract_rejects_bad_input() {
    let contract_with = |archetype: &str| {
        format!(
            "version = 1\n[axes]\narchetype = [\"generic\"]\n\
             [tokens]\ncollection = \"t\"\nmodes = []\ncolors = []\n\
             [[blocks]]\nid = \"b\"\nname = \"B\"\naxes = [\"archetype\"]\n\
             archetypes = [\"{archetype}\"]\nproperties = {{}}\n"
        )
    };
    assert!(Contract::parse(&contract_with("generic")).is_ok());
    assert!(Contract::parse(&contract_with("nope")).is_err());

    let source = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../design/review-package/contract.toml"
    ))
    .unwrap();
    let contract = Contract::parse(&source).unwrap();
    let rule = |id: &str| BlockRule {
        id: id.into(),
        requires: vec![],
    };
    let mut unknown = descriptor("ymaze").unwrap().clone();
    unknown.blocks.push(rule("nope"));
    assert!(contract.check_descriptors(&[unknown]).is_err());
    let mut misplaced = descriptor("oft").unwrap().clone();
    assert_eq!(misplaced.archetype, Archetype::ZoneOccupancy);
    misplaced.blocks.push(rule("object_role_legend"));
    assert!(contract.check_descriptors(&[misplaced]).is_err());
}

#[test]
fn context_labels_and_order() {
    let c = ymaze_contexts();
    for (variant, label) in [
        (Some("proximal25"), "25%"),
        (Some("arm0_dwell1s"), "0% (1 s dwell)"),
        (Some("arm45"), "45%"),
        (Some("arm0"), "0%"),
        (Some("compare_0pct_dwell0s"), "0% (no dwell)"),
        (Some("compare_x"), "compare_x"),
        (Some("foo"), "foo"),
        (None, "recorded"),
    ] {
        assert_eq!(c.label(variant), label, "{variant:?}");
    }
    let mut variants = vec![
        Some("proximal65"),
        Some("arm100"),
        Some("compare_0pct_dwell0s"),
        None,
        Some("arm45"),
        Some("proximal25"),
        Some("arm0_dwell1s"),
        Some("arm25"),
    ];
    variants.sort_by_key(|v| c.order_key(*v));
    assert_eq!(
        variants,
        [
            None,
            Some("arm0_dwell1s"),
            Some("compare_0pct_dwell0s"),
            Some("arm25"),
            Some("arm45"),
            Some("arm100"),
            Some("proximal25"),
            Some("proximal65"),
        ]
    );
    assert!(c.is_comparison("compare_0pct_dwell0s"));
    assert!(!c.is_comparison("arm0_dwell1s"));
}

#[test]
fn imen_corpus() {
    let r = resolve(&imen(), &PresentationOverrides::default()).unwrap();
    let keys: Vec<_> = r.experiments.iter().map(|e| e.key.as_str()).collect();
    assert_eq!(keys, ["oft", "nort", "ymaze"]);
    let titles: Vec<_> = r.experiments.iter().map(|e| e.title.as_str()).collect();
    assert_eq!(
        titles,
        [
            "Open field (NORT habituation)",
            "Novel object recognition",
            "Y-maze spontaneous alternation"
        ]
    );
    let shorts: Vec<_> = r
        .experiments
        .iter()
        .map(|e| e.short_name.as_str())
        .collect();
    assert_eq!(shorts, ["OFT", "NORT", "Y-maze"]);
    let costs: Vec<_> = r.experiments.iter().map(|e| e.render_cost).collect();
    assert_eq!(costs, [1, 3, 1]);
    let section_ids: Vec<_> = r.sections.iter().map(|s| s.id.as_str()).collect();
    assert_eq!(section_ids, ["nort_protocol", "ymaze"]);

    let p = |id: &str| &r.trials[id];
    let oft1 = p("oft_1");
    assert_eq!(oft1.step.as_deref(), Some("habituation"));
    assert_eq!(oft1.session_label, "Pre · Habituation (OFT) · batch B");
    assert_eq!(oft1.heading, "Pre-treatment: NORT protocol");
    assert_eq!(oft1.video, VideoKind::Scored);
    assert_eq!(p("oft_2").heading, "Post-treatment: NORT protocol");
    assert_eq!(p("t1_pre").session_label, "Pre · T1 training");
    assert_eq!(p("t1_pre").video, VideoKind::Qualia);
    assert_eq!(p("t2_post").session_label, "Post · T2 test");
    assert_eq!(p("t2_nophase").step.as_deref(), Some("t2"));
    assert_eq!(p("t2_nophase").timepoint.as_deref(), Some("pre"));

    // oft_1 < t1_pre < t2_pre < oft_2 < t2_post < every Y-maze trial.
    let order = ["oft_1", "t1_pre", "t2_pre", "oft_2", "t2_post"].map(|id| p(id).protocol_key());
    assert!(order.windows(2).all(|w| w[0] < w[1]), "{order:?}");
    for (id, ym) in r.trials.iter().filter(|(id, _)| id.starts_with("ym_")) {
        assert!(order[4] < ym.protocol_key(), "{id}");
        assert_eq!(
            (ym.session_label.as_str(), ym.heading.as_str()),
            ("Y-maze", "Y-maze")
        );
    }

    let ymaze = exp(&r, "ymaze");
    let labels: Vec<_> = ymaze.contexts.iter().map(|c| c.label.as_str()).collect();
    assert_eq!(
        labels,
        [
            "0% (1 s dwell)",
            "0% (no dwell)",
            "25%",
            "45%",
            "65%",
            "25%",
            "45%",
            "65%"
        ]
    );
    let principal: Vec<_> = ymaze
        .contexts
        .iter()
        .filter(|c| c.principal)
        .map(|c| c.variant.as_deref())
        .collect();
    assert_eq!(principal, [Some("arm45"), Some("proximal45")]);
    let comparison: Vec<_> = ymaze
        .contexts
        .iter()
        .filter(|c| c.comparison)
        .map(|c| c.variant.as_deref())
        .collect();
    assert_eq!(comparison, [Some("compare_0pct_dwell0s")]);
    assert!(p("ym_1__arm45").principal);
    assert!(!p("ym_1__arm25").principal);
    assert!(p("ym_2").principal);
    assert_eq!(p("ym_2").context, "45%");

    assert!(has(ymaze, "threshold_comparison"));
    let (oft, nort) = (exp(&r, "oft"), exp(&r, "nort"));
    assert!(has(oft, "video_embed") && has(oft, "endpoint_table"));
    assert!(has(nort, "object_role_legend") && !has(nort, "video_embed"));
    assert_eq!(nort.endpoints, ["DI"]);
    assert_eq!(
        oft.endpoints,
        [
            "total_distance_m",
            "mean_speed_m_per_s",
            "center_time_s",
            "center_time_pct",
            "center_entries",
            "latency_to_first_center_entry_s"
        ]
    );
}

#[test]
fn unmatched_nort_trial_is_other() {
    let mut corpus = imen();
    corpus.trials.push(TrialFacts {
        phase: some("habituation"),
        timepoint: some("pre"),
        ..trial("odd", "object_recognition")
    });
    let r = resolve(&corpus, &PresentationOverrides::default()).unwrap();
    let odd = &r.trials["odd"];
    assert_eq!(odd.step, None);
    assert_eq!(odd.section, "other");
    assert_eq!(odd.heading, "Other recordings");
    assert_eq!(odd.session_label, "Pre · Other · nort");
}

#[test]
fn oul_corpus() {
    let objects: Vec<String> = (1..=4).map(|i| format!("object_{i}")).collect();
    let r = resolve_all(
        [
            ("oul_1", "oul_training"),
            ("oul_2", "oul_training"),
            ("oul_3", "oul_training"),
            ("oul_4", "oul_update"),
            ("oul_5", "oul_test"),
        ]
        .map(|(id, recipe)| TrialFacts {
            recipe: some(recipe),
            perimeters: objects.clone(),
            ..trial(id, "object_recognition")
        })
        .to_vec(),
    );
    let e = exp(&r, "nort");
    let steps: Vec<_> = e
        .steps
        .iter()
        .map(|s| (s.id.as_str(), s.label.as_str()))
        .collect();
    assert_eq!(
        steps,
        [
            ("oul_training", "Training"),
            ("oul_update", "Update"),
            ("oul_test", "Test")
        ]
    );
    assert_eq!(e.archetype, Archetype::ObjectExploration);
    assert!(has(e, "object_role_legend"));
    assert!(e.capabilities.contains(&Capability::Objects));
    assert_eq!(
        r.trials["oul_4"].heading,
        "OUL protocol (timepoint unrecorded)"
    );
}

#[test]
fn cheeseboard_corpus() {
    let r = resolve_all(vec![TrialFacts {
        perimeters: vec!["reward".into(), "start".into()],
        has_video: true,
        metrics_keys: set(&["seconds_to_find_reward"]),
        ..trial("cb_1", "cheeseboard")
    }]);
    let e = exp(&r, "cheeseboard");
    assert_eq!(e.archetype, Archetype::GoalSearch);
    let steps: Vec<_> = e
        .steps
        .iter()
        .map(|s| (s.id.as_str(), s.label.as_str()))
        .collect();
    assert_eq!(steps, [("search", "Reward search")]);
    assert_eq!(r.trials["cb_1"].heading, "Cheeseboard");
    assert!(has(e, "object_role_legend") && has(e, "video_embed"));
    assert_eq!(
        e.endpoints,
        [
            "seconds_to_find_reward",
            "reward_entries",
            "search_path_distance_meters"
        ]
    );
}

#[test]
fn open_field_only_corpus() {
    let oft = |phase: Option<&str>| TrialFacts {
        timepoint: some("pre"),
        phase: phase.map(Into::into),
        ..trial("oft", "open_field")
    };
    let r = resolve_all(vec![oft(None)]);
    let p = &r.trials["oft"];
    assert_eq!(p.step.as_deref(), Some("open_field"));
    assert_eq!(p.session_label, "Pre · Open field");
    assert_eq!(p.heading, "Pre-treatment: Open field");
    let r = resolve_all(vec![oft(Some("A"))]);
    assert_eq!(r.trials["oft"].session_label, "Pre · Open field · batch A");
}

#[test]
fn ymaze_principal_without_named_variant() {
    let principal_of = |variants: &[Option<&str>]| {
        let r = resolve_all(
            variants
                .iter()
                .enumerate()
                .map(|(i, v)| TrialFacts {
                    variant: v.map(Into::into),
                    ..trial(&format!("ym_{i}"), "y_maze")
                })
                .collect(),
        );
        let e = exp(&r, "ymaze");
        let principal: Vec<_> = e
            .contexts
            .iter()
            .filter(|c| c.principal)
            .map(|c| c.variant.clone())
            .collect();
        (
            principal,
            e.contexts[0].label.clone(),
            has(e, "threshold_comparison"),
        )
    };
    let (p, _, _) = principal_of(&[Some("proximal25"), Some("proximal65")]);
    assert_eq!(p, [some("proximal25")]);
    let (p, _, _) = principal_of(&[Some("compare_0pct_dwell0s"), Some("proximal65")]);
    assert_eq!(p, [some("proximal65")]);
    let (p, label, comparison) = principal_of(&[None]);
    assert_eq!(p, [None]);
    assert_eq!(label, "recorded");
    assert!(!comparison);
}

#[test]
fn unknown_task_is_titled_by_its_task_type() {
    let r = resolve_all(vec![trial("rot_1", "rotarod")]);
    let e = &r.experiments[0];
    assert_eq!(
        (e.id.as_str(), e.key.as_str(), e.title.as_str()),
        ("generic", "rotarod", "rotarod")
    );
    assert_eq!(e.archetype, Archetype::Generic);
    assert_eq!(e.steps[0].id, "recording");
    let p = &r.trials["rot_1"];
    assert_eq!(p.step.as_deref(), Some("recording"));
    assert_eq!(
        (p.heading.as_str(), p.section.as_str()),
        ("rotarod", "rotarod")
    );
    for text in [&e.key, &e.title, &e.short_name, &p.experiment] {
        assert!(!text.contains("other"), "{text}");
    }
}

const OVERRIDES: &str = r#"
[nort]
title = "NOR"
steps = { t1 = "Familiarisation" }
endpoints = ["DI", "pct_novel"]
hide = ["object_role_legend"]
[ymaze]
principal = "arm25"
"#;

#[test]
fn overrides_apply() {
    let o: PresentationOverrides = toml::from_str(OVERRIDES).unwrap();
    let r = resolve(&imen(), &o).unwrap();
    let nort = exp(&r, "nort");
    assert_eq!(nort.title, "NOR");
    assert_eq!(r.trials["t1_pre"].session_label, "Pre · Familiarisation");
    assert_eq!(nort.endpoints, ["DI", "pct_novel"]);
    assert!(!has(nort, "object_role_legend"));
    let principal: Vec<_> = exp(&r, "ymaze")
        .contexts
        .iter()
        .filter(|c| c.principal)
        .map(|c| c.variant.as_deref())
        .collect();
    assert_eq!(principal, [Some("arm25")]);
    assert!(!r.trials["ym_2"].principal);

    let by_id: PresentationOverrides =
        toml::from_str("[object_recognition]\ntitle = \"X\"").unwrap();
    assert_eq!(exp(&resolve(&imen(), &by_id).unwrap(), "nort").title, "X");
}

#[test]
fn overrides_reject_unknown_names() {
    for bad in [
        "[foo]\ntitle = \"x\"",
        "[nort]\nsteps = { zzz = \"x\" }",
        "[nort]\nhide = [\"nope\"]",
    ] {
        let o: PresentationOverrides = toml::from_str(bad).unwrap();
        assert!(resolve(&imen(), &o).is_err(), "{bad}");
    }
    assert!(toml::from_str::<PresentationOverrides>("[nort]\ncolour = 1").is_err());

    let o: PresentationOverrides = toml::from_str("[rotarod]\ntitle = \"Rotarod\"").unwrap();
    let corpus = CorpusFacts {
        trials: vec![trial("rot_1", "rotarod")],
    };
    assert_eq!(
        resolve(&corpus, &o).unwrap().experiments[0].title,
        "Rotarod"
    );
}

/// Literals the post-analysis and review code hold today and phase B replaces.
/// bekiper-core cannot depend on bekiper-post-analysis, so they are copied here.
mod parity {
    use super::*;

    #[test]
    fn post_analysis_endpoints() {
        // rust/bekiper-post-analysis/src/contracts.rs:17
        const OFT_PRIMARY: [&str; 6] = [
            "total_distance_m",
            "mean_speed_m_per_s",
            "center_time_s",
            "center_time_pct",
            "center_entries",
            "latency_to_first_center_entry_s",
        ];
        const OFT_SECONDARY: [&str; 6] = [
            "periphery_time_pct",
            "freezing_s",
            "grooming_s",
            "rearing_s",
            "immobility_s",
            "fecal_boli_count",
        ];
        // contracts.rs:33
        const YM_SCORES: [&str; 3] = [
            "spontaneous_alternation_pct",
            "spontaneous_alternation_pct_standard",
            "neutral_triads_pct",
        ];
        let oft = descriptor("oft").unwrap();
        assert_eq!(oft.family("primary"), OFT_PRIMARY);
        assert_eq!(oft.modelled(), OFT_PRIMARY);
        assert_eq!(oft.family("secondary"), OFT_SECONDARY);
        let ymaze = descriptor("ymaze").unwrap();
        assert_eq!(ymaze.family("score"), YM_SCORES);
        // pipeline/models.rs:12-16, verification.rs:424
        assert_eq!(ymaze.modelled(), [YM_SCORES[0]]);
        assert_eq!(descriptor("nort").unwrap().modelled(), ["DI"]);
        // verification.rs order; export.rs:72,293 and metrics.rs:97 use the same set.
        let assays: Vec<_> = post_analysis_descriptors()
            .map(|d| d.post_analysis.as_ref().unwrap().assay.as_str())
            .collect();
        assert_eq!(assays, ["oft", "nort", "ymaze"]);
    }

    #[test]
    fn nort_secondary_endpoints_stay_out_of_review() {
        // export/categories.rs nort secondary list; no model, never a criterion.
        let nort = descriptor("nort").unwrap();
        assert_eq!(nort.family("primary"), ["DI"]);
        assert_eq!(
            nort.family("secondary"),
            ["RI", "LR", "object_bias", "side_bias"]
        );
        assert_eq!(nort.criteria(), ["DI"]);
        assert_eq!(nort.modelled(), ["DI"]);
    }

    #[test]
    fn review_html_and_parameters() {
        // review/html.rs:12 ASSAYS titles, :126 short()
        let r = resolve(&imen(), &PresentationOverrides::default()).unwrap();
        let named: Vec<_> = r
            .experiments
            .iter()
            .map(|e| (e.key.as_str(), e.title.as_str(), e.short_name.as_str()))
            .collect();
        assert_eq!(
            named,
            [
                ("oft", "Open field (NORT habituation)", "OFT"),
                ("nort", "Novel object recognition", "NORT"),
                ("ymaze", "Y-maze spontaneous alternation", "Y-maze"),
            ]
        );
        // review/parameters.rs:19 PREVIEWS
        let previews: BTreeSet<_> = descriptors()
            .iter()
            .flat_map(|d| &d.features.previews)
            .map(|p| (p.kind.as_str(), p.title.as_str()))
            .collect();
        assert_eq!(
            previews,
            BTreeSet::from([
                ("detection_fields", "Detection fields"),
                ("ymaze_arm_sections", "Arm sections and cross-lines"),
                ("center_periphery", "Centre / periphery"),
            ])
        );
    }

    #[test]
    fn composite_videos_and_index() {
        // python composite.py:301 overview panels
        let panels: BTreeSet<_> = descriptors()
            .iter()
            .flat_map(|d| &d.features.panels)
            .map(String::as_str)
            .collect();
        assert_eq!(
            panels,
            BTreeSet::from([
                "center_periphery",
                "object_discrimination",
                "ymaze_arm_sections",
                "arm_sequence"
            ])
        );
        let nort = descriptor("nort").unwrap();
        assert_eq!(nort.features.panel_exclude_labels, ["preference_summary"]);

        // review/videos.rs:54: qualia only for NORT/OUL.
        for d in descriptors() {
            let want = if d.id == nort.id {
                VideoKind::Qualia
            } else {
                VideoKind::Scored
            };
            assert!(d.steps.iter().all(|s| s.video == want), "{}", d.id);
        }

        // review/index.rs: cost x3 for NORT, edge case at :613.
        for d in descriptors() {
            assert_eq!(
                d.render_cost,
                if d.id == nort.id { 3 } else { 1 },
                "{}",
                d.id
            );
            let edges: Vec<_> = d
                .edge_cases
                .iter()
                .map(|e| (e.label.as_str(), e.lowest.as_str()))
                .collect();
            if d.id == nort.id {
                assert_eq!(edges, [("lowest total exploration", "total_exploration_s")]);
            } else {
                assert!(edges.is_empty(), "{}", d.id);
            }
        }

        // index.rs is_principal / contexts() sort.
        let c = ymaze_contexts();
        assert_eq!(c.principal, ["arm45", "proximal45"]);
        assert_eq!(c.principal_cohort.as_deref(), Some("arm45"));
    }
}

fn ymaze_corpus(variants: &[&str]) -> CorpusFacts {
    CorpusFacts {
        trials: variants
            .iter()
            .map(|v| TrialFacts {
                variant: some(v),
                ..trial(v, "y_maze")
            })
            .collect(),
    }
}

fn overrides(toml: &str) -> PresentationOverrides {
    toml::from_str(toml).unwrap()
}

#[test]
fn override_principal_must_be_a_recorded_context() {
    let err = resolve(&imen(), &overrides("[ymaze]\nprincipal=\"arm99\"")).unwrap_err();
    assert!(err.to_string().contains("arm99"), "{err}");
    // No contexts at all: principal is meaningless.
    assert!(resolve(&imen(), &overrides("[oft]\nprincipal=\"arm25\"")).is_err());
    let ok = resolve(&imen(), &overrides("[ymaze]\nprincipal=\"arm25\"")).unwrap();
    let y = ok.experiments.iter().find(|e| e.key == "ymaze").unwrap();
    assert_eq!(y.contexts.iter().filter(|c| c.principal).count(), 1);
}

#[test]
fn two_overrides_for_one_descriptor_are_rejected() {
    let both = overrides("[nort]\ntitle=\"A\"\n[object_recognition]\ntitle=\"B\"");
    let err = resolve(&imen(), &both).unwrap_err();
    assert!(err.to_string().contains("same experiment"), "{err}");
}

#[test]
fn interleaved_equal_sort_keys_dedup() {
    let r = resolve_all(ymaze_corpus(&["arm045", "arm45", "arm045"]).trials);
    let y = &r.experiments[0];
    let mut v: Vec<_> = y.contexts.iter().map(|c| c.variant.clone()).collect();
    v.sort();
    v.dedup();
    assert_eq!(v.len(), y.contexts.len());
}

#[test]
fn trial_without_task_type_is_titled_by_the_generic_descriptor() {
    let r = resolve_all(vec![trial("x", "")]);
    let e = &r.experiments[0];
    assert_eq!(
        (e.title.as_str(), e.short_name.as_str()),
        (
            "Other experiment",
            descriptor("generic").unwrap().short_name.as_str()
        )
    );
    assert!(r.sections.iter().all(|s| s.title != "generic"));
    // A named unknown task type keeps its name.
    assert_eq!(
        resolve_all(vec![trial("y", "foo")]).experiments[0].title,
        "foo"
    );
}

#[test]
fn principal_falls_back_to_first_non_comparison_context() {
    let r = resolve_all(ymaze_corpus(&["arm0_dwell1s", "arm25", "arm65"]).trials);
    let principal: Vec<_> = r.experiments[0]
        .contexts
        .iter()
        .filter(|c| c.principal)
        .map(|c| c.variant.as_deref().unwrap())
        .collect();
    assert_eq!(principal.len(), 1, "{principal:?}");
}

#[test]
fn archived_comparison_variant_displays_raw() {
    let r = resolve_all(ymaze_corpus(&["compare_0pct_dwell1s"]).trials);
    assert_eq!(r.experiments[0].contexts[0].label, "compare_0pct_dwell1s");
}
