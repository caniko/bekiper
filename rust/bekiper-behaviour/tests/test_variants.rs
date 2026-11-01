//! End-to-end equivalence of the multi-variant entry points against K calls of
//! the legacy single-variant analysis (the oracle), CFR and VFR clocks.
use bekiper_behaviour::open_field::OpenFieldTest;
use bekiper_behaviour::{AnalysisResult, BehaviouralTask, TaskDetail};
use bekiper_core::video::FrameTiming;
use polars::prelude::*;

pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    pub fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

pub type Analysis = (AnalysisResult, Option<TaskDetail>);

pub fn assert_same(label: &str, legacy: &Analysis, fast: &Analysis) {
    assert_eq!(
        serde_json::to_value(&legacy.0).unwrap(),
        serde_json::to_value(&fast.0).unwrap(),
        "{label}: analysis result"
    );
    assert_eq!(
        legacy.1.as_ref().map(TaskDetail::as_json_value),
        fast.1.as_ref().map(TaskDetail::as_json_value),
        "{label}: task detail"
    );
    let (a, b) = (
        legacy.0.evaluated_df.as_ref().unwrap(),
        fast.0.evaluated_df.as_ref().unwrap(),
    );
    assert_eq!(
        a.get_column_names(),
        b.get_column_names(),
        "{label}: columns"
    );
    for name in a.get_column_names() {
        assert!(
            a.column(name)
                .unwrap()
                .equals_missing(b.column(name).unwrap()),
            "{label}: column {name} differs"
        );
    }
}

/// CFR, or timestamps with jittered intervals (VFR).
pub fn timing(rng: &mut Rng, n: usize, vfr: bool) -> FrameTiming {
    if !vfr {
        return FrameTiming::cfr(30.0);
    }
    let mut t = 0.0;
    let timestamps: Vec<f64> = (0..n)
        .map(|_| {
            t += 1.0 / 30.0 * (0.8 + 0.4 * rng.unit());
            t
        })
        .collect();
    FrameTiming {
        fps: 30.0,
        timestamps,
        is_vfr: true,
    }
}

/// Quantised random walk with stationary stretches and missing frames, so many
/// frames sit exactly on candidate thresholds.
fn walk(rng: &mut Rng, n: usize, extent: f64, step: f64) -> (Vec<Option<f64>>, Vec<Option<f64>>) {
    let (mut x, mut y) = (0.0_f64, 0.0_f64);
    let mut hold = 0;
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for _ in 0..n {
        if hold > 0 {
            hold -= 1;
        } else if rng.below(40) == 0 {
            hold = 20 + rng.below(60);
        } else {
            x = (x + step * (rng.below(3) as f64 - 1.0)).clamp(-extent, extent);
            y = (y + step * (rng.below(3) as f64 - 1.0)).clamp(-extent, extent);
        }
        if rng.below(50) == 0 {
            xs.push(None);
            ys.push(None);
        } else {
            xs.push(Some(x));
            ys.push(Some(y));
        }
    }
    (xs, ys)
}

fn offset(v: &[Option<f64>], d: f64) -> Vec<Option<f64>> {
    v.iter().map(|x| x.map(|x| x + d)).collect()
}

fn oft_frame(rng: &mut Rng, n: usize) -> DataFrame {
    let (x, y) = walk(rng, n, 0.2, 0.01);
    df! {
        "mouse_center_x" => &x, "mouse_center_y" => &y,
        "center_ear_x" => &x, "center_ear_y" => &y,
        "nose_x" => offset(&x, 0.02), "nose_y" => &y,
        "tail_base_x" => offset(&x, -0.04), "tail_base_y" => &y,
    }
    .unwrap()
}

#[test]
fn open_field_variants_equal_legacy_calls() {
    for (seed, vfr) in [(1u64, false), (2, true), (3, false), (4, true)] {
        let mut rng = Rng(0x9E3779B97F4A7C15 ^ seed);
        let n = 700;
        let frame = oft_frame(&mut rng, n);
        let timing = timing(&mut rng, n, vfr);
        let mut variants = Vec::new();
        for (band, dwell, freeze) in [
            (0.08, 1.0, 0.005),
            (0.06, 1.0, 0.005),
            (0.10, 0.5, 0.005),
            (0.08, 0.0, 0.01),
            (0.08, 1.5, 0.002),
            (0.04, 1.0, 0.02),
        ] {
            let mut oft = OpenFieldTest::standard((0.0, 0.0), 0.40, band, 30.0);
            oft.params.min_dwell_seconds = dwell;
            oft.params.freezing_displacement_threshold = freeze;
            variants.push(oft);
        }
        // Not sweepable: different freezing window -> legacy path for that variant.
        let mut other = OpenFieldTest::standard((0.0, 0.0), 0.40, 0.08, 30.0);
        other.params.freezing_window_seconds = 0.5;
        variants.push(other);
        let refs: Vec<&OpenFieldTest> = variants.iter().collect();
        let fast = OpenFieldTest::analyze_variants(&refs, frame.clone().lazy(), &timing);
        assert_eq!(fast.len(), variants.len());
        for (k, variant) in variants.iter().enumerate() {
            let legacy = variant.analyze_detailed(frame.clone().lazy(), &timing);
            assert_same(
                &format!("oft seed={seed} vfr={vfr} variant={k}"),
                &legacy,
                &fast[k],
            );
        }
    }
}

// ---------------------------------------------------------------- NORT

use bekiper_behaviour::object_recognition::{NortWindowMode, ObjectRecognition};
use bekiper_feature::heuristic::combined::exploration::BoutFilterMode;
use bekiper_feature::heuristic::solo::whiskers::WhiskerFanParams;
use bekiper_perimeter::object_shape::ObjectShape;
use bekiper_perimeter::perimeter::Perimeter;
use bekiper_perimeter::rectangle::Rectangle;
use bekiper_perimeter::triangle::Triangle;

fn nort_frame(rng: &mut Rng, n: usize) -> DataFrame {
    let (x, y) = walk(rng, n, 0.2, 0.005);
    // Nose stretches pointing in one of eight quantised directions.
    let dirs = [
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (1.0, 1.0),
        (-1.0, 1.0),
        (1.0, -1.0),
        (-1.0, -1.0),
    ];
    let mut d = dirs[0];
    let (mut nx, mut ny, mut ex, mut ey, mut dx, mut dy) =
        (vec![], vec![], vec![], vec![], vec![], vec![]);
    for i in 0..n {
        if i % 25 == 0 {
            d = dirs[rng.below(8)];
        }
        let reach = [0.01, 0.02, 0.03][rng.below(3)];
        match (x[i], y[i]) {
            (Some(x), Some(y)) => {
                let n = (reach * d.0, reach * d.1);
                nx.push(Some(x + n.0));
                ny.push(Some(y + n.1));
                ex.push(Some(x + 0.4 * n.0));
                ey.push(Some(y + 0.4 * n.1));
                dx.push(Some(0.6 * n.0 + if rng.below(5) == 0 { 0.01 } else { 0.0 }));
                dy.push(Some(0.6 * n.1));
            }
            _ => {
                for v in [&mut nx, &mut ny, &mut ex, &mut ey, &mut dx, &mut dy] {
                    v.push(None);
                }
            }
        }
    }
    let ears = |dxy: f64| -> Vec<Option<f64>> { ex.iter().map(|v| v.map(|v| v + dxy)).collect() };
    df! {
        "mouse_center_x" => &x, "mouse_center_y" => &y,
        "nose_x" => &nx, "nose_y" => &ny,
        "center_ear_x" => &ex, "center_ear_y" => &ey,
        "left_ear_x" => ears(-0.005), "left_ear_y" => &ey,
        "right_ear_x" => ears(0.005), "right_ear_y" => &ey,
        "left_ear_dx" => &dx, "left_ear_dy" => &dy,
        "right_ear_dx" => &dx, "right_ear_dy" => &dy,
        "tail_base_x" => offset(&x, -0.04), "tail_base_y" => &y,
        "snout_dx" => &dx, "snout_dy" => &dy,
    }
    .unwrap()
}

fn nort_task(qualia: bool) -> ObjectRecognition {
    let mut task = ObjectRecognition::new(
        vec![
            Perimeter::new(
                "familiar",
                ObjectShape::Rectangle(Rectangle::new(-0.1, 0.0, 0.05, 0.05)),
            ),
            Perimeter::new(
                "novel",
                ObjectShape::Triangle(Triangle::new((0.07, 0.02), (0.14, 0.02), (0.1, 0.09))),
            ),
        ],
        30.0,
    );
    task.window_mode = NortWindowMode::FullAnalysis;
    task.params.min_dwell_seconds = 0.0;
    task.enable_qualia = qualia;
    task
}

fn nort_variants(qualia: bool, whisker: bool, rng: &mut Rng) -> Vec<ObjectRecognition> {
    let mut variants = Vec::new();
    for k in 0..9 {
        let mut task = nort_task(qualia);
        task.nose_max_distance = [0.03, 0.025, 0.04, 0.03][k % 4];
        task.contact_distance = [0.005, 0.01, 0.005][k % 3];
        task.max_angle_deg = [45.0, 30.0, 45.0][k % 3];
        task.max_nose_speed = [0.05, 0.1][k % 2];
        task.min_bout_seconds = [0.25, 0.5, 1.0][k % 3];
        task.max_gap_seconds = [1.0, 0.5][k % 2];
        task.strict_boundaries = k % 5 == 4;
        task.speed_filter_mode = [BoutFilterMode::PerFrame, BoutFilterMode::Disabled][k % 2];
        task.angle_filter_mode = [
            BoutFilterMode::PerFrame,
            BoutFilterMode::PerBoutMedian,
            BoutFilterMode::PerFrame,
        ][k % 3];
        task.perimeter_erosion_meters = if qualia {
            0.01
        } else {
            [0.01, 0.0, 0.02, 0.005][k % 4]
        };
        task.exclude_posture = false;
        if whisker {
            task.whisker_fan = Some(WhiskerFanParams {
                length_m: [0.0117, 0.02, 0.03, 0.015][rng.below(4)],
                ..WhiskerFanParams::default()
            });
        }
        variants.push(task);
    }
    variants
}

fn check_nort(label: &str, qualia: bool, whisker: bool) {
    for (seed, vfr) in [(11u64, false), (12, true)] {
        let mut rng = Rng(0x9E3779B97F4A7C15 ^ seed);
        let n = 600;
        let frame = nort_frame(&mut rng, n);
        let timing = timing(&mut rng, n, vfr);
        let variants = nort_variants(qualia, whisker, &mut rng);
        let refs: Vec<&ObjectRecognition> = variants.iter().collect();
        let fast = ObjectRecognition::analyze_variants(&refs, frame.clone().lazy(), &timing);
        let mut explored = 0.0;
        for (k, variant) in variants.iter().enumerate() {
            let legacy = variant.analyze_detailed(frame.clone().lazy(), &timing);
            if let Some(TaskDetail::ObjectRecognition(r)) = &legacy.1 {
                explored += r.exploration_times.iter().sum::<f64>();
            }
            assert_same(
                &format!("{label} seed={seed} vfr={vfr} variant={k}"),
                &legacy,
                &fast[k],
            );
        }
        assert!(
            explored > 0.0,
            "fixture never explores: the test would be vacuous"
        );
    }
}

#[test]
fn nort_variants_equal_legacy_calls_without_qualia() {
    check_nort("nort", false, false);
}

#[test]
fn nort_variants_equal_legacy_calls_with_whisker_fan() {
    check_nort("nort+whisker", false, true);
}

#[test]
fn nort_variants_equal_legacy_calls_with_qualia_and_whisker_fan() {
    check_nort("nort+qualia+whisker", true, true);
}

// ---------------------------------------------------------------- Y-maze

use bekiper_behaviour::radial_arm::{YMaze, YMazeStartMode, YMazeThresholdGeometry};
use bekiper_feature::tolerance::TemporalPolicy;
use bekiper_feature::zone::ZoneOccupancy;
use bekiper_perimeter::polygon::Polygon;
use bekiper_perimeter::radial_maze::RadialMaze;

fn ymaze_maze(convex: bool) -> Perimeter<RadialMaze> {
    let center = Polygon::new(vec![(-0.1, -0.1), (0.1, -0.1), (0.1, 0.1), (-0.1, 0.1)]);
    // Each arm shares exactly one edge with the centre (cross-line geometry).
    let a_d2 = if convex { (1.0, 0.25) } else { (0.2, 0.0) };
    let arm_a = Polygon::new(vec![(0.1, -0.1), (0.1, 0.1), a_d2, (1.0, -0.25)]);
    let arm_b = Polygon::new(vec![(-0.1, 0.1), (0.1, 0.1), (0.25, 1.0), (-0.25, 1.0)]);
    let arm_c = Polygon::new(vec![(-0.1, -0.1), (-0.1, 0.1), (-1.0, 0.25), (-1.0, -0.25)]);
    Perimeter::new("maze", RadialMaze::new(center, vec![arm_a, arm_b, arm_c]))
}

fn ymaze_frame(rng: &mut Rng, n: usize) -> DataFrame {
    let (hx, hy) = walk(rng, n, 1.0, 0.05);
    let tx = offset(&hx, -0.05);
    df! {
        "center_ear_x" => &hx, "center_ear_y" => &hy,
        "tail_base_x" => &tx, "tail_base_y" => &hy,
        "mouse_center_x" => &hx, "mouse_center_y" => &hy,
    }
    .unwrap()
}

fn ymaze_variants(convex: bool, geometry: YMazeThresholdGeometry) -> Vec<YMaze> {
    let crossline = geometry == YMazeThresholdGeometry::ArmStartCrosslineV1;
    let ps: &[f64] = if crossline {
        &[0.0, 0.25, 0.45, 0.5, 0.65, 1.0, 0.3]
    } else {
        &[0.25, 0.3, 0.45, 0.5, 0.65, 0.1, 0.9]
    };
    let mut variants = Vec::new();
    for (k, &p) in ps.iter().enumerate() {
        let mut task = YMaze::new(ymaze_maze(convex), 30.0);
        task.threshold_geometry = geometry;
        task.proximal_threshold = p;
        task.start_mode = [YMazeStartMode::FirstDetection, YMazeStartMode::VideoStart][k % 2];
        task.zone_occupancy = [ZoneOccupancy::DwellFiltered, ZoneOccupancy::Raw][k % 2];
        let principal = !crossline && [0.25, 0.45, 0.65].contains(&p);
        if !principal {
            task.arm_membership_hysteresis = k % 3 != 0;
            task.zone_max_gap_seconds = [0.1, 0.0, 1.0][k % 3];
            task.apply_dwell_filter = k % 4 != 0;
            task.temporal_policy = match k % 3 {
                0 => None,
                1 => Some(TemporalPolicy {
                    min_duration_seconds: 1.0,
                    max_gap_seconds: 0.5,
                    ..TemporalPolicy::default()
                }),
                _ => Some(TemporalPolicy::default()),
            };
        }
        variants.push(task);
    }
    // A different gap-fill produces a different mitigated frame: separate group.
    let mut other = YMaze::new(ymaze_maze(convex), 30.0);
    other.threshold_geometry = geometry;
    other.proximal_threshold = if crossline { 0.4 } else { 0.35 };
    other.pose_gap_fill_seconds = 0.5;
    variants.push(other);
    variants
}

fn check_ymaze(convex: bool, geometry: YMazeThresholdGeometry) {
    for (seed, vfr) in [(21u64, false), (22, true)] {
        let mut rng = Rng(0x9E3779B97F4A7C15 ^ seed);
        let n = 900;
        let frame = ymaze_frame(&mut rng, n);
        let timing = timing(&mut rng, n, vfr);
        let variants = ymaze_variants(convex, geometry);
        let refs: Vec<&YMaze> = variants.iter().collect();
        let fast = YMaze::analyze_variants(&refs, frame.clone().lazy(), &timing);
        let mut arm_frames = 0;
        for (k, variant) in variants.iter().enumerate() {
            let legacy = variant.analyze_detailed(frame.clone().lazy(), &timing);
            if let Some(TaskDetail::YMaze(r)) = &legacy.1 {
                arm_frames += r.zone_summaries.iter().map(|z| z.entries).sum::<u32>();
            }
            assert_same(
                &format!("ymaze convex={convex} {geometry:?} seed={seed} vfr={vfr} variant={k}"),
                &legacy,
                &fast[k],
            );
        }
        assert!(arm_frames > 0, "fixture never enters a zone: vacuous test");
    }
}

#[test]
fn ymaze_crossline_variants_equal_legacy_calls() {
    check_ymaze(true, YMazeThresholdGeometry::ArmStartCrosslineV1);
}

#[test]
fn ymaze_non_convex_crossline_falls_back_and_matches_legacy() {
    check_ymaze(false, YMazeThresholdGeometry::ArmStartCrosslineV1);
}

#[test]
fn ymaze_radial_variants_equal_legacy_calls() {
    check_ymaze(true, YMazeThresholdGeometry::ImenRadialVertexSpanV1);
}

#[test]
fn open_field_non_square_centre_falls_back_to_legacy() {
    let mut rng = Rng(0x51);
    let n = 200;
    let frame = oft_frame(&mut rng, n);
    let timing = timing(&mut rng, n, false);
    let mut variants = Vec::new();
    for dwell in [0.5, 1.0] {
        let mut oft = OpenFieldTest::standard((0.0, 0.0), 0.40, 0.08, 30.0);
        oft.center.shape.width *= 0.9;
        oft.params.min_dwell_seconds = dwell;
        variants.push(oft);
    }
    let refs: Vec<&OpenFieldTest> = variants.iter().collect();
    let fast = OpenFieldTest::analyze_variants(&refs, frame.clone().lazy(), &timing);
    for (k, variant) in variants.iter().enumerate() {
        let legacy = variant.analyze_detailed(frame.clone().lazy(), &timing);
        assert_same(&format!("oft non-square variant={k}"), &legacy, &fast[k]);
    }
}

#[test]
fn nort_single_variant_group_and_whisker_provenance() {
    let mut rng = Rng(0x77);
    let n = 300;
    let frame = nort_frame(&mut rng, n);
    let timing = timing(&mut rng, n, false);
    let mut task = nort_task(false);
    task.whisker_fan = Some(WhiskerFanParams::default());
    task.whisker_age_days = Some(63.0);
    task.whisker_length_source = Some("unit-test".into());
    let fast = ObjectRecognition::analyze_variants(&[&task], frame.clone().lazy(), &timing);
    let legacy = task.analyze_detailed(frame.lazy(), &timing);
    assert_same("nort single", &legacy, &fast[0]);
    let json = serde_json::to_string(&legacy.1.unwrap().as_json_value()).unwrap();
    assert!(
        json.contains("\"age_days\":63.0") && json.contains("unit-test"),
        "{json}"
    );
}

// ---------------------------------------------------------------- benchmark

/// `cargo test --release -p bekiper-behaviour --test test_variants -- --ignored --nocapture`
/// prints T(K) for K = 1, 4, 16 on a synthetic 9,000-frame recording: legacy
/// (K independent analyses) against the multi-variant entry point.
#[test]
#[ignore = "micro-benchmark"]
fn bench_variant_marginal_cost() {
    use std::time::Instant;
    let n = 9_000;
    let mut rng = Rng(0xB5297A4D);
    let timing = timing(&mut rng, n, false);
    let oft_frame = oft_frame(&mut rng, n);
    let nort_frame = nort_frame(&mut rng, n);
    let ymaze_frame = ymaze_frame(&mut rng, n);

    fn report(name: &str, k: usize, legacy: f64, fast: f64, base: &mut Option<f64>) {
        let t1 = *base.get_or_insert(fast);
        println!(
            "{name:<8} K={k:<2} legacy {legacy:>8.3}s ({:>7.3}s/variant)  multi {fast:>8.3}s  marginal v={:>7.3}s ({:>5.1}% of T(1))",
            legacy / k as f64,
            if k > 1 {
                (fast - t1) / (k - 1) as f64
            } else {
                f64::NAN
            },
            if k > 1 {
                100.0 * (fast - t1) / (k - 1) as f64 / t1
            } else {
                f64::NAN
            },
        );
    }
    let time = |f: &mut dyn FnMut()| {
        let start = Instant::now();
        f();
        start.elapsed().as_secs_f64()
    };

    let mut base = None;
    for k in [1usize, 4, 16] {
        let variants: Vec<OpenFieldTest> = (0..k)
            .map(|i| {
                let mut t =
                    OpenFieldTest::standard((0.0, 0.0), 0.40, 0.04 + 0.005 * i as f64, 30.0);
                t.params.min_dwell_seconds = 0.5 + 0.1 * i as f64;
                t.params.freezing_displacement_threshold = 0.002 + 0.001 * i as f64;
                t
            })
            .collect();
        let refs: Vec<&OpenFieldTest> = variants.iter().collect();
        let legacy = time(&mut || {
            for v in &variants {
                std::hint::black_box(v.analyze_detailed(oft_frame.clone().lazy(), &timing));
            }
        });
        let fast = time(&mut || {
            std::hint::black_box(OpenFieldTest::analyze_variants(
                &refs,
                oft_frame.clone().lazy(),
                &timing,
            ));
        });
        report("oft", k, legacy, fast, &mut base);
    }

    let mut base = None;
    for k in [1usize, 4, 16] {
        let mut variants = nort_variants(false, true, &mut rng);
        while variants.len() < k {
            let mut t = nort_task(false);
            let i = variants.len() as f64;
            t.nose_max_distance = 0.02 + 0.001 * i;
            t.contact_distance = 0.004 + 0.0005 * i;
            t.max_angle_deg = 30.0 + i;
            t.min_bout_seconds = 0.25 + 0.05 * i;
            t.whisker_fan = Some(WhiskerFanParams {
                length_m: 0.01 + 0.001 * i,
                ..WhiskerFanParams::default()
            });
            variants.push(t);
        }
        variants.truncate(k);
        let refs: Vec<&ObjectRecognition> = variants.iter().collect();
        let legacy = time(&mut || {
            for v in &variants {
                std::hint::black_box(v.analyze_detailed(nort_frame.clone().lazy(), &timing));
            }
        });
        let fast = time(&mut || {
            std::hint::black_box(ObjectRecognition::analyze_variants(
                &refs,
                nort_frame.clone().lazy(),
                &timing,
            ));
        });
        report("nort", k, legacy, fast, &mut base);
    }

    let mut base = None;
    for k in [1usize, 4, 16] {
        let variants: Vec<YMaze> = (0..k)
            .map(|i| {
                let mut t = YMaze::new(ymaze_maze(true), 30.0);
                t.threshold_geometry = YMazeThresholdGeometry::ArmStartCrosslineV1;
                t.proximal_threshold = i as f64 / 16.0;
                t.zone_max_gap_seconds = 0.1 * (i % 3) as f64;
                t
            })
            .collect();
        let refs: Vec<&YMaze> = variants.iter().collect();
        let legacy = time(&mut || {
            for v in &variants {
                std::hint::black_box(v.analyze_detailed(ymaze_frame.clone().lazy(), &timing));
            }
        });
        let fast = time(&mut || {
            std::hint::black_box(YMaze::analyze_variants(
                &refs,
                ymaze_frame.clone().lazy(),
                &timing,
            ));
        });
        report("ymaze", k, legacy, fast, &mut base);
    }
}
