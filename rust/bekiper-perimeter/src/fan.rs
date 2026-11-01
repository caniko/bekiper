//! Exact whisker-fan geometry: a circular sector (apex, radius, +-half-angle
//! about a direction) against a filled polygon outline.
//!
//! Each outline edge is clipped to the sector (two half-planes and the disc);
//! a non-empty clip is a hit. This is the exact union of "vertex inside the
//! sector", "radius crosses an edge" and "arc crosses an edge", and the apex
//! inside the polygon counts as a hit too. Requires `0 < half_angle < 90`.
use std::f64::consts::PI;

use bekiper_math::confinement::point_in_polygon;

use crate::object_shape::ObjectShape;

/// Which lateral fan hit. `Left` is the fan on the +90 degree rotation of the
/// head axis (ear to nose) in the stored coordinate frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FanSide {
    Left,
    Right,
}

impl FanSide {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Left => "left",
            Self::Right => "right",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FanHit {
    pub side: FanSide,
    /// Distance (m) from the apex to the nearest object point inside the fan.
    pub distance: f64,
}

/// Closed outline of an object shape.
///
/// ponytail: a circle is tessellated into a 128-gon (inscribed, radius error
/// below 0.03%); replace with an analytic disc-sector test if circles matter.
pub fn outline(shape: &ObjectShape) -> Vec<(f64, f64)> {
    match shape {
        ObjectShape::Polygon(p) => p.vertices.clone(),
        ObjectShape::Triangle(t) => t.vertices().to_vec(),
        ObjectShape::Rectangle(r) => {
            let (hw, hh) = (r.width / 2.0, r.height / 2.0);
            vec![
                (r.center_x - hw, r.center_y - hh),
                (r.center_x + hw, r.center_y - hh),
                (r.center_x + hw, r.center_y + hh),
                (r.center_x - hw, r.center_y + hh),
            ]
        }
        ObjectShape::Circle(c) => (0..128)
            .map(|i| {
                let a = 2.0 * PI * f64::from(i) / 128.0;
                (
                    c.center_x + c.radius * a.cos(),
                    c.center_y + c.radius * a.sin(),
                )
            })
            .collect(),
    }
}

/// Nearest hit of the two lateral fans (apex, `axis` = head direction) with
/// the outline, or `None` for a miss or a degenerate (zero-length) axis.
pub fn fan_hit(
    apex: (f64, f64),
    axis: (f64, f64),
    half_angle_deg: f64,
    length: f64,
    outline: &[(f64, f64)],
) -> Option<FanHit> {
    let norm = axis.0.hypot(axis.1);
    if !(norm > 1e-12) || outline.len() < 3 {
        return None;
    }
    let (ux, uy) = (axis.0 / norm, axis.1 / norm);
    let half = half_angle_deg.to_radians();
    let left = sector_distance(apex, (-uy, ux), half, length, outline);
    let right = sector_distance(apex, (uy, -ux), half, length, outline);
    match (left, right) {
        (Some(l), Some(r)) if r < l => Some(FanHit {
            side: FanSide::Right,
            distance: r,
        }),
        (Some(l), _) => Some(FanHit {
            side: FanSide::Left,
            distance: l,
        }),
        (None, Some(r)) => Some(FanHit {
            side: FanSide::Right,
            distance: r,
        }),
        (None, None) => None,
    }
}

/// Hit at a shorter `length` derived from the hit computed at a longer
/// length (design 4.3): the sector is clipped to the radius, so the nearest
/// distance and the nearer side are unchanged and a hit survives iff
/// `distance <= length`. `None` means the distance lies inside the float band
/// around `length`; the caller then evaluates [`fan_hit`] at `length`.
pub fn narrow_hit(hit_max: Option<FanHit>, length: f64) -> Option<Option<FanHit>> {
    let Some(hit) = hit_max else {
        return Some(None);
    };
    let eps = bekiper_math::sweep::band_eps(hit.distance, length);
    if hit.distance < length - eps {
        Some(Some(hit))
    } else if hit.distance > length + eps {
        Some(None)
    } else {
        None
    }
}

fn cross(a: (f64, f64), b: (f64, f64)) -> f64 {
    a.0 * b.1 - a.1 * b.0
}

fn rotate(v: (f64, f64), angle: f64) -> (f64, f64) {
    let (s, c) = angle.sin_cos();
    (v.0 * c - v.1 * s, v.0 * s + v.1 * c)
}

/// Narrow `[lo, hi]` to the `t` where `a + b t >= 0`; false when empty.
fn clip_halfplane(a: f64, b: f64, lo: &mut f64, hi: &mut f64) -> bool {
    if b.abs() <= 1e-15 {
        return a >= -1e-15;
    }
    let t = -a / b;
    if b > 0.0 {
        *lo = lo.max(t);
    } else {
        *hi = hi.min(t);
    }
    true
}

/// Distance from the apex to the nearest outline point inside one sector.
fn sector_distance(
    apex: (f64, f64),
    centre_dir: (f64, f64),
    half: f64,
    length: f64,
    outline: &[(f64, f64)],
) -> Option<f64> {
    if point_in_polygon(apex.0, apex.1, outline) {
        return Some(0.0);
    }
    // Wedge = left of r1 and right of r2 (valid for half < 90 degrees).
    let (r1, r2) = (rotate(centre_dir, -half), rotate(centre_dir, half));
    let mut best: Option<f64> = None;
    for i in 0..outline.len() {
        let (a, b) = (outline[i], outline[(i + 1) % outline.len()]);
        let p = (a.0 - apex.0, a.1 - apex.1);
        let d = (b.0 - a.0, b.1 - a.1);
        let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
        if !clip_halfplane(cross(r1, p), cross(r1, d), &mut lo, &mut hi)
            || !clip_halfplane(cross(p, r2), cross(d, r2), &mut lo, &mut hi)
        {
            continue;
        }
        // |p + t d|^2 <= length^2
        let (qa, qb, qc) = (
            d.0 * d.0 + d.1 * d.1,
            p.0 * d.0 + p.1 * d.1,
            p.0 * p.0 + p.1 * p.1 - length * length,
        );
        if qa <= 1e-24 {
            if qc > 0.0 {
                continue;
            }
        } else {
            let disc = qb * qb - qa * qc;
            if disc < 0.0 {
                continue;
            }
            let root = disc.sqrt();
            lo = lo.max((-qb - root) / qa);
            hi = hi.min((-qb + root) / qa);
        }
        if lo > hi + 1e-12 {
            continue;
        }
        let t = if qa <= 1e-24 {
            0.0
        } else {
            (-qb / qa).clamp(lo, hi.max(lo))
        };
        let dist = (p.0 + t * d.0).hypot(p.1 + t * d.1);
        best = Some(best.map_or(dist, |b| b.min(dist)));
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    const L: f64 = 0.0117;
    fn rect(x0: f64, y0: f64, x1: f64, y1: f64) -> Vec<(f64, f64)> {
        vec![(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    }
    // Apex at the origin, head along +x: left fan opens toward +y, right toward -y.
    fn hit(poly: &[(f64, f64)]) -> Option<FanHit> {
        fan_hit((0.0, 0.0), (1.0, 0.0), 30.0, L, poly)
    }

    #[test]
    fn vertex_inside_sector_hits_left() {
        let h = hit(&rect(-0.001, 0.004, 0.001, 0.006)).unwrap();
        assert_eq!(h.side, FanSide::Left);
        assert!((h.distance - 0.004).abs() < 1e-9);
    }

    #[test]
    fn radius_crossing_without_vertex_inside() {
        // Corners: two beyond 30 degrees, two beyond the length; the bar crosses the 60-degree radius.
        let poly = rect(0.0035, 0.005, 0.0045, 0.012);
        assert!(
            !poly
                .iter()
                .any(|&(x, y)| x.hypot(y) <= L && x.atan2(y).abs() <= 30f64.to_radians())
        );
        assert_eq!(hit(&poly).unwrap().side, FanSide::Left);
    }

    #[test]
    fn arc_crossing_without_vertex_or_radius() {
        // Horizontal bar whose corners are past the arc but whose middle dips inside it.
        let poly = rect(-0.004, 0.0115, 0.004, 0.0125);
        assert!(poly.iter().all(|&(x, y)| x.hypot(y) > L));
        let h = hit(&poly).unwrap();
        assert!((h.distance - 0.0115).abs() < 1e-9);
    }

    #[test]
    fn misses_and_sides() {
        assert!(
            hit(&rect(-0.002, 0.013, 0.002, 0.014)).is_none(),
            "beyond length"
        );
        assert!(
            hit(&rect(0.007, -0.001, 0.009, 0.001)).is_none(),
            "in front, outside +-30 deg"
        );
        assert!(
            hit(&rect(-0.009, -0.001, -0.007, 0.001)).is_none(),
            "behind the head"
        );
        assert_eq!(
            hit(&rect(-0.001, -0.006, 0.001, -0.004)).unwrap().side,
            FanSide::Right
        );
    }

    #[test]
    fn apex_inside_object_and_degenerate_axis() {
        assert_eq!(hit(&rect(-0.01, -0.01, 0.01, 0.01)).unwrap().distance, 0.0);
        assert!(
            fan_hit(
                (0.0, 0.0),
                (0.0, 0.0),
                30.0,
                L,
                &rect(-0.001, 0.004, 0.001, 0.006)
            )
            .is_none()
        );
    }

    #[test]
    fn outline_of_a_polygon_is_its_vertices() {
        let v = rect(0.0, 0.0, 1.0, 1.0);
        let shape = ObjectShape::Polygon(crate::polygon::Polygon::new(v.clone()));
        assert_eq!(outline(&shape), v);
    }

    // 45 degree fans put the left wedge boundary on y = x, so an edge along
    // that diagonal is parallel to a wedge ray.
    fn hit45(poly: &[(f64, f64)]) -> Option<FanHit> {
        fan_hit((0.0, 0.0), (1.0, 0.0), 45.0, L, poly)
    }

    #[test]
    fn edge_along_the_wedge_ray_still_hits() {
        let h = hit45(&[(0.001, 0.001), (0.002, 0.002), (0.0015, 0.0035)]).unwrap();
        assert_eq!(h.side, FanSide::Left);
        assert!((h.distance - 0.001_f64.hypot(0.001)).abs() < 1e-9);
    }

    #[test]
    fn edge_parallel_to_and_outside_the_wedge_ray_misses() {
        assert!(hit45(&[(0.003, 0.001), (0.004, 0.002), (0.0035, 0.0005)]).is_none());
    }

    #[test]
    fn zero_length_edges_hit_only_within_the_radius() {
        let near = [(0.0, 0.005), (0.0, 0.005), (0.01, 0.008), (-0.01, 0.008)];
        assert!((hit(&near).unwrap().distance - 0.005).abs() < 1e-9);
        let far = [(0.0, 0.05), (0.0, 0.05), (0.01, 0.06), (-0.01, 0.06)];
        assert!(hit(&far).is_none());
    }
}
