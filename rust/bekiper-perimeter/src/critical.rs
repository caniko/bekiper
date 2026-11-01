//! Critical-value kernels (design 4.1/4.3): per-point parameter values at which a
//! monotone geometric predicate flips. A kernel never replaces the legacy
//! predicate near the boundary; callers re-evaluate the legacy scalar inside the
//! float band (`bekiper_math::sweep::band_eps`). Every kernel here is validated
//! against its legacy oracle in `tests/test_critical.rs`.
use crate::object_shape::ObjectShape;
use crate::polygon::{Polygon, arc_units, convex_hull};
use crate::radial_maze::ArmFrame;
use bekiper_core::shape::Expandable;

fn cross(a: (f64, f64), b: (f64, f64)) -> f64 {
    a.0 * b.1 - a.1 * b.0
}

fn sub(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 - b.0, a.1 - b.1)
}

impl ArmFrame {
    /// The cross-line monotonicity proof (design 4.3) needs a strictly convex
    /// arm quadrilateral `c1, c2, d2, d1`; otherwise callers use the legacy
    /// per-variant predicate.
    pub fn is_strictly_convex(&self) -> bool {
        let q = [self.c1, self.c2, self.d2, self.d1];
        let scale = q
            .iter()
            .fold(0.0_f64, |m, p| m.max(p.0.abs()).max(p.1.abs()))
            .max(1.0);
        let margin = 1e-9 * scale * scale;
        let turns: Vec<f64> = (0..4)
            .map(|i| {
                cross(
                    sub(q[(i + 1) % 4], q[i]),
                    sub(q[(i + 2) % 4], q[(i + 1) % 4]),
                )
            })
            .collect();
        turns.iter().all(|&t| t > margin) || turns.iter().all(|&t| t < -margin)
    }

    /// `p*(q)`: the cross-line fraction whose line passes through `q`, so that
    /// (for `q` inside a strictly convex arm) `crossline(p).is_past(q)` iff
    /// `p <= p*`. `None` when `q` lies outside the quadrilateral's `p` range or
    /// the geometry is degenerate; callers then use the legacy predicate.
    pub fn crossline_critical(&self, x: f64, y: f64) -> Option<f64> {
        let (u, w) = (sub(self.d1, self.c1), sub(self.d2, self.c2));
        let (e, v, r) = (sub(self.c2, self.c1), sub(w, u), sub((x, y), self.c1));
        // f(p) = cross(e + p v, r - p u) = a p^2 + b p + c
        let a = cross(u, v);
        let b = cross(v, r) - cross(e, u);
        let c = cross(e, r);
        let scale = a.abs().max(b.abs()).max(c.abs());
        if !(scale > 0.0) || !scale.is_finite() {
            return None;
        }
        let mut roots = [f64::NAN; 2];
        if a.abs() <= 1e-12 * scale {
            if b == 0.0 {
                return None;
            }
            roots[0] = -c / b;
        } else {
            let disc = b * b - 4.0 * a * c;
            if disc < 0.0 {
                return None;
            }
            let q = -0.5 * (b + b.signum() * disc.sqrt());
            roots[0] = q / a;
            if q != 0.0 {
                roots[1] = c / q;
            }
        }
        let f = |p: f64| {
            cross(
                (e.0 + p * v.0, e.1 + p * v.1),
                (r.0 - p * u.0, r.1 - p * u.1),
            )
        };
        let df = |p: f64| 2.0 * a * p + b;
        let mut best: Option<f64> = None;
        for root in roots.into_iter().filter(|p| p.is_finite()) {
            let mut p = root;
            if df(p) != 0.0 {
                p -= f(p) / df(p); // one Newton step polishes cancellation error
            }
            if (-p).max(p - 1.0) > 1e-9 {
                continue;
            }
            match best {
                Some(b) if (b - p).abs() > 1e-9 => return None, // two roots: not a convex-arm point
                Some(_) => {}
                None => best = Some(p),
            }
        }
        best
    }
}

/// Radial threshold critical value: `p* = (d - near) / (far - near)` where `d`
/// is the distance from the centre; `d^2 >= (near + p (far - near))^2` iff
/// `p <= p*` for `p` in the legitimate range (`near + p span >= 0`).
pub fn radial_critical(center: (f64, f64), near: f64, far: f64, x: f64, y: f64) -> f64 {
    let (dx, dy) = (x - center.0, y - center.1);
    ((dx * dx + dy * dy).sqrt() - near) / (far - near)
}

/// OFT centre: `Rectangle::new(cx, cy, s, s)` contains `q` iff `s >= s*(q)`.
pub fn centered_square_critical(center: (f64, f64), x: f64, y: f64) -> f64 {
    let (dx, dy) = ((x - center.0).abs(), (y - center.1).abs());
    if dx.is_nan() || dy.is_nan() {
        return f64::NAN; // `f64::max` would drop the NaN
    }
    2.0 * dx.max(dy)
}

/// Half-plane form of `Polygon::expand(d)` for `d > 0` (design 4.3):
/// the offset hull is `H (+) d Q` for a fixed convex arc polygon `Q`, so
/// `d*(q) = max_plane (nu . q - h_H(nu)) / h_Q(nu)`.
#[derive(Debug, Clone)]
pub struct ExpandKernel {
    /// `(nu, nu . corner, nu . unit)` per boundary edge.
    planes: Vec<((f64, f64), f64, f64)>,
}

impl ExpandKernel {
    pub fn new(polygon: &Polygon) -> Option<Self> {
        let hull = convex_hull(&polygon.vertices);
        if hull.len() < 3 {
            return None;
        }
        let pts = arc_units(&hull);
        let m = pts.len();
        let unit = |angle: f64| (angle.cos(), angle.sin());
        let at1 = |i: usize| {
            let (corner, angle) = pts[i];
            let u = unit(angle);
            (corner.0 + u.0, corner.1 + u.1)
        };
        let mut planes = Vec::with_capacity(m);
        for i in 0..m {
            let j = (i + 1) % m;
            let edge = sub(at1(j), at1(i));
            let len = edge.0.hypot(edge.1);
            if len <= 1e-12 {
                continue;
            }
            let nu = (edge.1 / len, -edge.0 / len);
            let (corner, angle) = pts[i];
            let u = unit(angle);
            let h_q = nu.0 * u.0 + nu.1 * u.1;
            if h_q <= 1e-6 {
                return None;
            }
            planes.push((nu, nu.0 * corner.0 + nu.1 * corner.1, h_q));
        }
        Some(Self { planes })
    }

    /// Smallest offset `d` whose expansion contains `(x, y)`.
    pub fn critical(&self, x: f64, y: f64) -> f64 {
        self.planes
            .iter()
            .map(|&(nu, h_h, h_q)| (nu.0 * x + nu.1 * y - h_h) / h_q)
            .fold(f64::NEG_INFINITY, f64::max)
    }
}

/// Inward offset: for an unflipped mitred erosion the polygon is the
/// intersection of the hull half-planes shifted by `e`, so `q` is inside iff
/// `e <= e*(q) = min_i (h_i - nu_i . q)`.
#[derive(Debug, Clone)]
pub struct ErodeKernel {
    planes: Vec<((f64, f64), f64)>,
    polygon: Polygon,
}

impl ErodeKernel {
    pub fn new(polygon: &Polygon) -> Option<Self> {
        let hull = convex_hull(&polygon.vertices);
        if hull.len() < 3 {
            return None;
        }
        let n = hull.len();
        let planes = (0..n)
            .filter_map(|i| {
                let (a, b) = (hull[i], hull[(i + 1) % n]);
                let len = (b.0 - a.0).hypot(b.1 - a.1);
                (len > 1e-12).then(|| {
                    let nu = ((b.1 - a.1) / len, (a.0 - b.0) / len);
                    (nu, nu.0 * a.0 + nu.1 * a.1)
                })
            })
            .collect();
        Some(Self {
            planes,
            polygon: Polygon::new(hull),
        })
    }

    /// The kernel is exact only while the legacy erosion has not collapsed.
    pub fn applies(&self, erosion: f64) -> bool {
        if !(erosion > 0.0) {
            return false;
        }
        let eroded = self.polygon.expand(-erosion);
        let first = eroded.vertices.first().copied();
        eroded.vertices.len() == self.planes.len()
            && eroded.vertices.iter().any(|v| Some(*v) != first)
    }

    pub fn critical(&self, x: f64, y: f64) -> f64 {
        self.planes
            .iter()
            .map(|&(nu, h)| h - (nu.0 * x + nu.1 * y))
            .fold(f64::INFINITY, f64::min)
    }
}

/// Offset critical values for any object outline.
#[derive(Debug, Clone)]
pub enum OffsetKernel {
    Circle {
        center: (f64, f64),
        radius: f64,
    },
    Polygon {
        expand: ExpandKernel,
        erode: ErodeKernel,
    },
}

impl OffsetKernel {
    pub fn new(shape: &ObjectShape) -> Option<Self> {
        let polygon_kernel = |p: &Polygon| {
            Some(Self::Polygon {
                expand: ExpandKernel::new(p)?,
                erode: ErodeKernel::new(p)?,
            })
        };
        match shape {
            ObjectShape::Circle(c) => Some(Self::Circle {
                center: (c.center_x, c.center_y),
                radius: c.radius,
            }),
            ObjectShape::Polygon(p) => polygon_kernel(p),
            ObjectShape::Rectangle(r) => polygon_kernel(&Polygon::new(vec![
                (r.center_x - r.width / 2.0, r.center_y - r.height / 2.0),
                (r.center_x + r.width / 2.0, r.center_y - r.height / 2.0),
                (r.center_x + r.width / 2.0, r.center_y + r.height / 2.0),
                (r.center_x - r.width / 2.0, r.center_y + r.height / 2.0),
            ])),
            ObjectShape::Triangle(t) => polygon_kernel(&Polygon::new(t.vertices().to_vec())),
        }
    }

    /// Smallest `d >= 0` offset for which `shape.expand(d)` contains the point.
    pub fn expand_critical(&self, x: f64, y: f64) -> f64 {
        match self {
            Self::Circle { center, radius } => (x - center.0).hypot(y - center.1) - radius,
            Self::Polygon { expand, .. } => expand.critical(x, y),
        }
    }

    /// Whether `erode_critical` is exact for this erosion distance.
    pub fn erosion_applies(&self, erosion: f64) -> bool {
        match self {
            Self::Circle { radius, .. } => erosion > 0.0 && erosion < *radius,
            Self::Polygon { erode, .. } => erode.applies(erosion),
        }
    }

    /// Largest erosion `e` for which `shape.expand(-e)` still contains the point.
    pub fn erode_critical(&self, x: f64, y: f64) -> f64 {
        match self {
            Self::Circle { center, radius } => radius - (x - center.0).hypot(y - center.1),
            Self::Polygon { erode, .. } => erode.critical(x, y),
        }
    }
}
