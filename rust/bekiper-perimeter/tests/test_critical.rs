//! Property tests: perimeter critical-value kernels against their legacy oracles.
use bekiper_core::shape::{Expandable, Shape};
use bekiper_math::sweep::band_eps;
use bekiper_perimeter::circle::Circle;
use bekiper_perimeter::critical::*;
use bekiper_perimeter::fan::{fan_hit, narrow_hit};
use bekiper_perimeter::object_shape::ObjectShape;
use bekiper_perimeter::polygon::Polygon;
use bekiper_perimeter::radial_maze::ArmFrame;
use bekiper_perimeter::rectangle::Rectangle;
use bekiper_perimeter::triangle::Triangle;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn random_frame(rng: &mut Rng) -> ArmFrame {
    let a = rng.range(0.2, 1.0);
    let b = rng.range(0.2, 1.0);
    let c1 = (1.0, -a);
    let c2 = (1.0, b);
    let len = rng.range(1.0, 5.0);
    let d1 = (
        c1.0 + len + rng.range(-0.5, 0.5),
        c1.1 + rng.range(-1.5, 0.5),
    );
    let d2 = (
        c2.0 + len + rng.range(-0.5, 0.5),
        c2.1 + rng.range(-0.5, 1.5),
    );
    ArmFrame { c1, c2, d1, d2 }
}

fn bilinear(f: &ArmFrame, s: f64, p: f64) -> (f64, f64) {
    let lerp = |c: (f64, f64), d: (f64, f64)| (c.0 + p * (d.0 - c.0), c.1 + p * (d.1 - c.1));
    let (a, b) = (lerp(f.c1, f.d1), lerp(f.c2, f.d2));
    (a.0 + s * (b.0 - a.0), a.1 + s * (b.1 - a.1))
}

#[test]
fn crossline_critical_matches_is_past_on_convex_arms() {
    let mut rng = Rng(0x9E3779B97F4A7C15);
    let mut convex = 0;
    let mut kernel_hits = 0;
    let mut total = 0;
    for _ in 0..3000 {
        let f = random_frame(&mut rng);
        if !f.is_strictly_convex() {
            continue;
        }
        convex += 1;
        for _ in 0..40 {
            // Interior points, plus points exactly on a quad edge and on a line p.
            let (s, p0) = match rng.below(4) {
                0 => (rng.unit(), rng.unit()),
                1 => ([0.0, 1.0][rng.below(2)], rng.unit()),
                2 => (rng.unit(), [0.0, 1.0][rng.below(2)]),
                _ => (rng.unit(), [0.25, 0.45, 0.65][rng.below(3)]),
            };
            let (x, y) = bilinear(&f, s, p0);
            let crit = f.crossline_critical(x, y);
            for p in [
                0.0,
                0.1,
                0.25,
                0.45,
                0.65,
                0.9,
                1.0,
                p0,
                (p0 + 0.01).min(1.0),
            ] {
                let Ok(line) = f.crossline(p) else { continue };
                total += 1;
                let legacy = line.is_past(x, y);
                if let Some(c) = crit
                    && (c - p).abs() > band_eps(c, p)
                    && p > 0.0
                {
                    kernel_hits += 1;
                    assert_eq!(p <= c, legacy, "{f:?} q=({x},{y}) p={p} crit={c}");
                }
            }
        }
    }
    assert!(
        convex > 1000,
        "generator produced too few convex arms: {convex}"
    );
    assert!(
        kernel_hits * 3 > total,
        "kernel rarely applicable: {kernel_hits}/{total}"
    );
}

#[test]
fn non_convex_arms_are_rejected_for_the_kernel() {
    let mut rng = Rng(5);
    let mut rejected = 0;
    for _ in 0..500 {
        let mut f = random_frame(&mut rng);
        std::mem::swap(&mut f.d1, &mut f.d2); // bow-tie
        if !f.is_strictly_convex() {
            rejected += 1;
        }
    }
    assert!(rejected > 450);
    let dent = ArmFrame {
        c1: (1.0, -1.0),
        c2: (1.0, 1.0),
        d2: (0.5, 0.5),
        d1: (5.0, -3.0),
    };
    assert!(!dent.is_strictly_convex());
}

fn random_polygon(rng: &mut Rng) -> Polygon {
    let n = 3 + rng.below(6);
    Polygon::new(
        (0..n)
            .map(|_| (rng.range(-0.06, 0.06), rng.range(-0.06, 0.06)))
            .collect(),
    )
}

fn shapes(rng: &mut Rng) -> ObjectShape {
    match rng.below(4) {
        0 => ObjectShape::Circle(Circle::new(
            rng.range(-0.02, 0.02),
            0.01,
            rng.range(0.01, 0.05),
        )),
        1 => ObjectShape::Rectangle(Rectangle::new(
            0.0,
            0.01,
            rng.range(0.02, 0.1),
            rng.range(0.02, 0.1),
        )),
        2 => ObjectShape::Triangle(Triangle::new(
            (rng.range(-0.05, 0.0), rng.range(-0.05, 0.0)),
            (rng.range(0.02, 0.06), rng.range(-0.05, 0.0)),
            (rng.range(-0.02, 0.03), rng.range(0.02, 0.06)),
        )),
        _ => ObjectShape::Polygon(random_polygon(rng)),
    }
}

#[test]
fn expand_and_erode_critical_match_legacy_contains() {
    let mut rng = Rng(0xD1B54A32D192ED03);
    let (mut expand_checked, mut erode_checked, mut erode_fallback) = (0, 0, 0);
    for _ in 0..600 {
        let shape = shapes(&mut rng);
        let Some(kernel) = OffsetKernel::new(&shape) else {
            continue;
        };
        for _ in 0..40 {
            let (x, y) = (rng.range(-0.12, 0.12), rng.range(-0.12, 0.12));
            let c = kernel.expand_critical(x, y);
            for d in [0.005, 0.01, 0.03, rng.range(0.0, 0.08)] {
                if (c - d).abs() > band_eps(c, d) {
                    expand_checked += 1;
                    assert_eq!(
                        shape.expand(d).contains(x, y),
                        c <= d,
                        "{shape:?} q=({x},{y}) d={d} c={c}"
                    );
                }
            }
            let e_c = kernel.erode_critical(x, y);
            for e in [0.001, 0.01, 0.02, rng.range(0.0005, 0.06)] {
                if !kernel.erosion_applies(e) {
                    erode_fallback += 1;
                    continue;
                }
                if (e_c - e).abs() > band_eps(e_c, e) {
                    erode_checked += 1;
                    assert_eq!(
                        shape.expand(-e).contains(x, y),
                        e <= e_c,
                        "{shape:?} q=({x},{y}) e={e} c={e_c}"
                    );
                }
            }
        }
    }
    assert!(expand_checked > 10_000 && erode_checked > 3_000 && erode_fallback > 0);
}

#[test]
fn oft_centre_critical_matches_rectangle_contains() {
    let mut rng = Rng(11);
    for _ in 0..3000 {
        let c = (rng.range(0.1, 0.3), rng.range(0.1, 0.3));
        let (x, y) = (rng.range(0.0, 0.4), rng.range(0.0, 0.4));
        let crit = centered_square_critical(c, x, y);
        for side in [0.16, 0.24, 0.3, rng.range(0.05, 0.4)] {
            if (crit - side).abs() > band_eps(crit, side) {
                assert_eq!(
                    Rectangle::new(c.0, c.1, side, side).contains(x, y),
                    crit <= side
                );
            }
        }
    }
}

#[test]
fn whisker_narrowing_matches_fan_hit_at_shorter_length() {
    let mut rng = Rng(0x2545F4914F6CDD1D);
    let (mut certain, mut band) = (0, 0);
    for _ in 0..3000 {
        let outline = random_polygon(&mut rng).vertices;
        if outline.len() < 3 {
            continue;
        }
        let apex = (rng.range(-0.08, 0.08), rng.range(-0.08, 0.08));
        let axis = (rng.range(-1.0, 1.0), rng.range(-1.0, 1.0));
        let l_max = 0.03;
        let hit_max = fan_hit(apex, axis, 30.0, l_max, &outline);
        for l in [0.0117, 0.02, 0.03, rng.range(0.001, 0.03)] {
            let legacy = fan_hit(apex, axis, 30.0, l, &outline);
            match narrow_hit(hit_max, l) {
                Some(got) => {
                    certain += 1;
                    assert_eq!(got, legacy, "apex={apex:?} axis={axis:?} l={l}");
                }
                None => band += 1,
            }
        }
    }
    assert!(certain > 6000 && band < certain / 20);
}

fn frame(c1: (f64, f64), c2: (f64, f64), d1: (f64, f64), d2: (f64, f64)) -> ArmFrame {
    ArmFrame { c1, c2, d1, d2 }
}

#[test]
fn crossline_critical_degenerate_and_ambiguous_geometry() {
    // Parallel sides (a = 0): linear root; q = (3, 0) lies halfway along the arm.
    let rect = frame((1.0, -1.0), (1.0, 1.0), (5.0, -1.0), (5.0, 1.0));
    assert_eq!(rect.crossline_critical(3.0, 0.0), Some(0.5));
    // Collinear frame: a = b = 0 with c != 0, and all-equal points: scale 0.
    let line = frame((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0));
    assert_eq!(line.crossline_critical(0.5, 1.0), None);
    let point = frame((1.0, 1.0), (1.0, 1.0), (1.0, 1.0), (1.0, 1.0));
    assert_eq!(point.crossline_critical(2.0, 2.0), None);
    // Quadratic frame: no real root, two distinct roots in [0, 1], a double root.
    let quad = frame((0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (3.0, 1.0));
    assert_eq!(quad.crossline_critical(5.0, 0.0), None);
    assert_eq!(quad.crossline_critical(2.0, 1.0), None);
    assert_eq!(quad.crossline_critical(3.0, 1.0), Some(1.0));
}

#[test]
fn centred_square_critical_propagates_nan() {
    assert!(centered_square_critical((0.0, 0.0), f64::NAN, 1.0).is_nan());
    assert!(centered_square_critical((0.0, 0.0), 1.0, f64::NAN).is_nan());
}

#[test]
fn offset_kernels_reject_degenerate_polygons_and_non_positive_erosion() {
    let segment = ObjectShape::Polygon(Polygon::new(vec![(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]));
    assert!(OffsetKernel::new(&segment).is_none());
    assert!(ExpandKernel::new(&Polygon::new(vec![(0.0, 0.0), (1.0, 1.0)])).is_none());
    let kernel = ErodeKernel::new(&Polygon::new(vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)])).unwrap();
    assert!(!kernel.applies(0.0) && !kernel.applies(f64::NAN) && kernel.applies(0.1));
    assert!(ErodeKernel::new(&Polygon::new(vec![(0.0, 0.0), (1.0, 1.0)])).is_none());
}
