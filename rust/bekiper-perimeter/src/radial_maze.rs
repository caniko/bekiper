use serde::{Deserialize, Serialize};

use crate::polygon::Polygon;
use bekiper_core::shape::{ClosestPoint, RayIntersectable, RayVisibility, Shape};
use bekiper_core::types::BoundingBox;

/// Arm-start frame: the arm's start edge `(c1, c2)` (0%, shared with the centre
/// polygon) and its annotated distal line `(d1, d2)` (100%), with `d1` the
/// vertex adjacent to `c1` in the arm quadrilateral.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ArmFrame {
    pub c1: (f64, f64),
    pub c2: (f64, f64),
    pub d1: (f64, f64),
    pub d2: (f64, f64),
}

/// The p% cross-line of an [`ArmFrame`]: from `c1 + p(d1-c1)` to `c2 + p(d2-c2)`.
/// A point is past it when it lies on the distal side, boundary inclusive.
/// `p <= 0` is plain arm membership and never rejects a point.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ArmCrossline {
    pub p: f64,
    pub p1: (f64, f64),
    pub p2: (f64, f64),
    /// Normal of the line oriented toward the distal side.
    pub normal: (f64, f64),
}

impl ArmFrame {
    /// Errors when the line collapses to a point (no side to test).
    pub fn crossline(&self, p: f64) -> Result<ArmCrossline, String> {
        let lerp = |c: (f64, f64), d: (f64, f64)| (c.0 + p * (d.0 - c.0), c.1 + p * (d.1 - c.1));
        let (p1, p2) = (lerp(self.c1, self.d1), lerp(self.c2, self.d2));
        let (dx, dy) = (p2.0 - p1.0, p2.1 - p1.1);
        let scale = [self.c1, self.c2, self.d1, self.d2]
            .iter()
            .fold(0.0_f64, |m, q| m.max(q.0.abs()).max(q.1.abs()));
        if dx.hypot(dy) <= 1e-9 * scale.max(1.0) {
            return Err(format!(
                "degenerate cross-line at p={p}: endpoints coincide"
            ));
        }
        let distal = (
            (self.d1.0 + self.d2.0 - self.c1.0 - self.c2.0) / 2.0,
            (self.d1.1 + self.d2.1 - self.c1.1 - self.c2.1) / 2.0,
        );
        let normal = if -dy * distal.0 + dx * distal.1 >= 0.0 {
            (-dy, dx)
        } else {
            (dy, -dx)
        };
        Ok(ArmCrossline { p, p1, p2, normal })
    }
}

impl ArmCrossline {
    /// Signed distance-like value; `>= 0` is the distal side. The Polars
    /// predicate in bekiper-behaviour evaluates the same expression.
    pub fn signed_side(&self, x: f64, y: f64) -> f64 {
        (x - self.p1.0) * self.normal.0 + (y - self.p1.1) * self.normal.1
    }

    pub fn is_past(&self, x: f64, y: f64) -> bool {
        self.p <= 0.0 || self.signed_side(x, y) >= 0.0
    }
}

/// A radial arm maze composed of multiple polygon arms branching from a center.
/// Used for Y-maze and multi-arm maze experiments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RadialMaze {
    pub arms: Vec<Polygon>,
    pub center: Polygon,
}

impl RadialMaze {
    pub fn new(center: Polygon, arms: Vec<Polygon>) -> Self {
        Self { arms, center }
    }

    /// Check which arm (if any) a point is in. Returns arm index.
    pub fn arm_containing(&self, x: f64, y: f64) -> Option<usize> {
        self.arms.iter().position(|arm| arm.contains(x, y))
    }

    /// Start edge and distal line of arm `i`. The start edge is the arm edge
    /// whose two vertices coincide with centre vertices; canonical vertex order
    /// does not keep it positional. Errors unless exactly one such edge exists.
    pub fn arm_frame(&self, i: usize) -> Result<ArmFrame, String> {
        let arm = self.arms.get(i).ok_or_else(|| format!("no arm {i}"))?;
        let v = &arm.vertices;
        if v.len() != 4 {
            return Err(format!(
                "arm {i} must be a quadrilateral, got {} vertices",
                v.len()
            ));
        }
        let scale = v
            .iter()
            .chain(&self.center.vertices)
            .fold(0.0_f64, |m, p| m.max(p.0.abs()).max(p.1.abs()));
        let tol_sq = (1e-9 * scale.max(1.0)).powi(2);
        let on_center = |p: (f64, f64)| {
            self.center
                .vertices
                .iter()
                .any(|c| (c.0 - p.0).powi(2) + (c.1 - p.1).powi(2) <= tol_sq)
        };
        let shared: Vec<usize> = (0..4)
            .filter(|&k| on_center(v[k]) && on_center(v[(k + 1) % 4]))
            .collect();
        let [k] = shared[..] else {
            return Err(format!(
                "arm {i} needs exactly one edge shared with the centre polygon, found {}",
                shared.len()
            ));
        };
        Ok(ArmFrame {
            c1: v[k],
            c2: v[(k + 1) % 4],
            d1: v[(k + 3) % 4],
            d2: v[(k + 2) % 4],
        })
    }

    /// True if the point is in the center region.
    pub fn in_center(&self, x: f64, y: f64) -> bool {
        self.center.contains(x, y)
    }
}

impl Shape for RadialMaze {
    fn inspection_perimeter(&self, label: &str) -> Option<bekiper_core::inspection::PerimeterSpec> {
        Some(bekiper_core::inspection::PerimeterSpec::RadialMaze {
            label: label.into(),
            center_vertices: self.center.vertices.clone(),
            arms: self.arms.iter().map(|p| p.vertices.clone()).collect(),
        })
    }
    fn contains(&self, x: f64, y: f64) -> bool {
        self.center.contains(x, y) || self.arms.iter().any(|arm| arm.contains(x, y))
    }

    fn bounding_box(&self) -> BoundingBox {
        let mut bb = self.center.bounding_box();
        for arm in &self.arms {
            let arm_bb = arm.bounding_box();
            bb.min_x = bb.min_x.min(arm_bb.min_x);
            bb.min_y = bb.min_y.min(arm_bb.min_y);
            bb.max_x = bb.max_x.max(arm_bb.max_x);
            bb.max_y = bb.max_y.max(arm_bb.max_y);
        }
        bb
    }
}

impl ClosestPoint for RadialMaze {
    fn closest_point(&self, x: f64, y: f64) -> (f64, f64) {
        let point = (x, y);
        std::iter::once(self.center.closest_point(x, y))
            .chain(self.arms.iter().map(|arm| arm.closest_point(x, y)))
            .min_by(|a, b| squared_distance(*a, point).total_cmp(&squared_distance(*b, point)))
            .unwrap_or(point)
    }
}

impl RayIntersectable for RadialMaze {
    fn ray_visibility(&self, origin: (f64, f64), direction: (f64, f64)) -> Option<RayVisibility> {
        std::iter::once(&self.center)
            .chain(self.arms.iter())
            .filter_map(|shape| shape.ray_visibility(origin, direction))
            .min_by(
                |a, b| match a.min_angle_radians.total_cmp(&b.min_angle_radians) {
                    std::cmp::Ordering::Equal => squared_distance(a.target_point, origin)
                        .total_cmp(&squared_distance(b.target_point, origin)),
                    ordering => ordering,
                },
            )
    }
}

fn squared_distance(a: (f64, f64), b: (f64, f64)) -> f64 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    dx * dx + dy * dy
}

#[cfg(test)]
mod arm_frame_tests {
    use super::*;

    // Non-parallel flared arm on the right edge of a square centre.
    const C1: (f64, f64) = (1.0, -1.0);
    const C2: (f64, f64) = (1.0, 1.0);
    const D1: (f64, f64) = (5.0, -3.0);
    const D2: (f64, f64) = (4.0, 3.0);

    fn rot(p: (f64, f64), quarter_turns: usize) -> (f64, f64) {
        (0..quarter_turns).fold(p, |(x, y), _| (-y, x))
    }

    fn maze(quarter_turns: usize, reversed: bool) -> RadialMaze {
        let r = |p| rot(p, quarter_turns);
        let center = Polygon::new(vec![r((-1.0, -1.0)), r(C1), r(C2), r((-1.0, 1.0))]);
        let mut arm = vec![r(C1), r(C2), r(D2), r(D1)];
        if reversed {
            arm.reverse();
        }
        RadialMaze::new(center, vec![Polygon::new(arm)])
    }

    #[test]
    fn start_edge_found_under_every_rotation_and_winding() {
        for turns in 0..4 {
            for reversed in [false, true] {
                let f = maze(turns, reversed).arm_frame(0).unwrap();
                let r = |p| rot(p, turns);
                let (a, b) = ((f.c1, f.d1), (f.c2, f.d2));
                // c1/c2 may swap, but each start vertex keeps its own distal neighbour.
                let ok = (a == (r(C1), r(D1)) && b == (r(C2), r(D2)))
                    || (a == (r(C2), r(D2)) && b == (r(C1), r(D1)));
                assert!(ok, "turns={turns} reversed={reversed}: {f:?}");
            }
        }
    }

    #[test]
    fn rv_y5_degenerate_crossline_is_an_error() {
        let f = ArmFrame {
            c1: (1.0, -1.0),
            c2: (1.0, 1.0),
            d1: (5.0, 0.0),
            d2: (5.0, 0.0),
        };
        assert!(f.crossline(0.5).is_ok());
        assert!(f.crossline(1.0).is_err());
        let pinched = ArmFrame {
            c1: (1.0, -1.0),
            c2: (1.0, 1.0),
            d1: (1.0, 1.0),
            d2: (1.0, -1.0),
        };
        assert!(pinched.crossline(0.5).is_err());
    }

    #[test]
    fn crossline_thresholds_and_inclusive_boundary() {
        let f = maze(0, false).arm_frame(0).unwrap();
        let (x, y) = (2.75, 0.0); // exactly on the p = 0.5 line
        let half = f.crossline(0.5).unwrap();
        assert!(half.is_past(x, y), "boundary is inclusive");
        assert!(half.is_past(3.0, 0.0) && !half.is_past(2.5, 0.0));
        // p = 0 never rejects (plain membership); the line is the start edge.
        assert!(f.crossline(0.0).unwrap().is_past(0.0, 0.0));
        // p = 0.25: line at y=0 sits at x = 1 + 0.25*3.5 = 1.875.
        let quarter = f.crossline(0.25).unwrap();
        assert!(quarter.is_past(1.875, 0.0) && !quarter.is_past(1.8, 0.0));
        // p = 1 is the distal line itself: only its own points qualify.
        let full = f.crossline(1.0).unwrap();
        assert!(full.is_past(4.5, 0.0) && !full.is_past(4.4, 0.0));
        assert!(full.is_past(f.d1.0, f.d1.1) && full.is_past(f.d2.0, f.d2.1));
    }

    #[test]
    fn errors_without_a_unique_shared_edge() {
        let center = Polygon::new(vec![(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]);
        let detached = Polygon::new(vec![(2.0, -1.0), (2.0, 1.0), (5.0, 3.0), (5.0, -3.0)]);
        let m = RadialMaze::new(center.clone(), vec![detached]);
        assert!(m.arm_frame(0).is_err());
        // Arm covering two centre edges is ambiguous.
        let wrap = Polygon::new(vec![(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (3.0, 3.0)]);
        assert!(RadialMaze::new(center, vec![wrap]).arm_frame(0).is_err());
        assert!(m.arm_frame(7).is_err());
    }
}
