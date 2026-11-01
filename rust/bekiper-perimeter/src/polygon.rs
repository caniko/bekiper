use serde::{Deserialize, Serialize};

use bekiper_core::shape::{ClosestPoint, Expandable, RayIntersectable, RayVisibility, Shape};
use bekiper_core::types::BoundingBox;
use bekiper_math::confinement::point_in_polygon;
use bekiper_math::geometry::canonicalize_polygon_vertices as canonicalize_vertices;
use bekiper_math::vector::{GEOMETRY_EPSILON, angle_between, ray_line_segment_intersection_point};

/// A general polygon defined by ordered vertices.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Polygon {
    pub vertices: Vec<(f64, f64)>,
}

impl Polygon {
    pub fn new(vertices: Vec<(f64, f64)>) -> Self {
        Self {
            vertices: canonicalize_vertices(vertices),
        }
    }

    /// Compute the centroid of this polygon.
    pub fn centroid(&self) -> (f64, f64) {
        let n = self.vertices.len() as f64;
        let (sx, sy) = self
            .vertices
            .iter()
            .fold((0.0, 0.0), |(ax, ay), &(x, y)| (ax + x, ay + y));
        (sx / n, sy / n)
    }

    /// Get edges as pairs of (start, end) vertices.
    pub fn edges(&self) -> impl Iterator<Item = ((f64, f64), (f64, f64))> + '_ {
        let n = self.vertices.len();
        (0..n).map(move |i| (self.vertices[i], self.vertices[(i + 1) % n]))
    }

    pub fn closest_point(&self, x: f64, y: f64) -> (f64, f64) {
        let point = (x, y);
        self.edges()
            .map(|(start, end)| closest_point_on_segment(point, start, end))
            .min_by(|a, b| squared_distance(*a, point).total_cmp(&squared_distance(*b, point)))
            .unwrap_or(point)
    }
}

fn closest_point_on_segment(point: (f64, f64), start: (f64, f64), end: (f64, f64)) -> (f64, f64) {
    let seg_dx = end.0 - start.0;
    let seg_dy = end.1 - start.1;
    let len_sq = seg_dx * seg_dx + seg_dy * seg_dy;
    if len_sq <= GEOMETRY_EPSILON {
        return start;
    }

    let px = point.0 - start.0;
    let py = point.1 - start.1;
    let t = ((px * seg_dx) + (py * seg_dy)) / len_sq;
    let t = t.clamp(0.0, 1.0);
    (start.0 + t * seg_dx, start.1 + t * seg_dy)
}

fn squared_distance(a: (f64, f64), b: (f64, f64)) -> f64 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    dx * dx + dy * dy
}

pub(crate) fn normalize(vector: (f64, f64)) -> Option<(f64, f64)> {
    let mag = (vector.0 * vector.0 + vector.1 * vector.1).sqrt();
    (mag > GEOMETRY_EPSILON).then_some((vector.0 / mag, vector.1 / mag))
}

impl Shape for Polygon {
    fn inspection_perimeter(&self, label: &str) -> Option<bekiper_core::inspection::PerimeterSpec> {
        Some(bekiper_core::inspection::PerimeterSpec::Polygon {
            label: label.into(),
            vertices: self.vertices.clone(),
        })
    }
    fn contains(&self, x: f64, y: f64) -> bool {
        point_in_polygon(x, y, &self.vertices)
    }

    fn bounding_box(&self) -> BoundingBox {
        let mut bb = BoundingBox {
            min_x: f64::INFINITY,
            min_y: f64::INFINITY,
            max_x: f64::NEG_INFINITY,
            max_y: f64::NEG_INFINITY,
        };
        for &(x, y) in &self.vertices {
            bb.min_x = bb.min_x.min(x);
            bb.min_y = bb.min_y.min(y);
            bb.max_x = bb.max_x.max(x);
            bb.max_y = bb.max_y.max(y);
        }
        bb
    }
}

impl ClosestPoint for Polygon {
    fn closest_point(&self, x: f64, y: f64) -> (f64, f64) {
        Polygon::closest_point(self, x, y)
    }
}

impl<'de> Deserialize<'de> for Polygon {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct PolygonData {
            vertices: Vec<(f64, f64)>,
        }

        let data = PolygonData::deserialize(deserializer)?;
        Ok(Polygon::new(data.vertices))
    }
}

impl Expandable for Polygon {
    type Output = Polygon;

    /// Offset the object's convex hull by `distance` (criteria: "convex hull
    /// offset"). Positive distances move every edge outward along its normal
    /// and round the corners, so each boundary point is exactly `distance`
    /// from the hull; negative distances move edges inward (mitred corners).
    fn expand(&self, distance: f64) -> Polygon {
        let hull = convex_hull(&self.vertices);
        if hull.len() < 3 || distance == 0.0 {
            return Polygon::new(hull);
        }
        let n = hull.len();
        // Outward unit normal of edge i (hull is counter-clockwise).
        let normal = |i: usize| {
            let (a, b) = (hull[i], hull[(i + 1) % n]);
            normalize((b.1 - a.1, a.0 - b.0)).unwrap_or((0.0, 0.0))
        };
        if distance > 0.0 {
            let vertices = arc_units(&hull)
                .into_iter()
                .map(|(corner, angle)| {
                    (
                        corner.0 + distance * angle.cos(),
                        corner.1 + distance * angle.sin(),
                    )
                })
                .collect();
            return Polygon::new(vertices);
        }
        // Inward: intersect consecutive edges shifted by `distance`.
        let vertices: Vec<(f64, f64)> = (0..n)
            .map(|i| {
                let (np, nn) = (normal((i + n - 1) % n), normal(i));
                // Corner point v + distance * (np + nn) / (1 + np.nn) lies on both shifted edges.
                let denom = 1.0 + np.0 * nn.0 + np.1 * nn.1;
                (
                    hull[i].0 + distance * (np.0 + nn.0) / denom,
                    hull[i].1 + distance * (np.1 + nn.1) / denom,
                )
            })
            .collect();
        // ponytail: an erosion past the inradius flips an edge; collapse to the
        // centroid instead of iteratively dropping edges (objects are >> 2 cm).
        let flipped = (0..n).any(|i| {
            let (a, b) = (hull[i], hull[(i + 1) % n]);
            let (c, d) = (vertices[i], vertices[(i + 1) % n]);
            (b.0 - a.0) * (d.0 - c.0) + (b.1 - a.1) * (d.1 - c.1) <= 0.0
        });
        if flipped {
            let centroid = Polygon::new(hull).centroid();
            return Polygon {
                vertices: vec![centroid; 3],
            };
        }
        Polygon::new(vertices)
    }
}

/// Corner-arc points of the positive offset of a counter-clockwise convex
/// hull as `(corner, angle)`; the offset vertex is `corner + d * (cos, sin)`.
/// The angles do not depend on `d`, which the multi-variant kernels rely on.
// ponytail: corner arcs are chords every pi/16 (sag < 0.005 * distance).
pub(crate) fn arc_units(hull: &[(f64, f64)]) -> Vec<((f64, f64), f64)> {
    let n = hull.len();
    let normal = |i: usize| {
        let (a, b) = (hull[i], hull[(i + 1) % n]);
        normalize((b.1 - a.1, a.0 - b.0)).unwrap_or((0.0, 0.0))
    };
    let step = std::f64::consts::PI / 16.0;
    let mut out = Vec::new();
    for (i, &corner) in hull.iter().enumerate() {
        let (incoming, outgoing) = (normal((i + n - 1) % n), normal(i));
        let start = incoming.1.atan2(incoming.0);
        let sweep = (outgoing.1.atan2(outgoing.0) - start).rem_euclid(std::f64::consts::TAU);
        let steps = (sweep / step).ceil().max(1.0) as usize;
        for k in 0..=steps {
            out.push((corner, start + sweep * k as f64 / steps as f64));
        }
    }
    out
}

/// Counter-clockwise convex hull without collinear points (monotone chain).
pub(crate) fn convex_hull(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut sorted = points.to_vec();
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    sorted.dedup();
    if sorted.len() < 3 {
        return sorted;
    }
    let cross = |o: (f64, f64), a: (f64, f64), b: (f64, f64)| {
        (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
    };
    let mut hull: Vec<(f64, f64)> = Vec::with_capacity(2 * sorted.len());
    for pass in [sorted.clone(), sorted.into_iter().rev().collect()] {
        let floor = hull.len();
        for p in pass {
            while hull.len() >= floor + 2
                && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0
            {
                hull.pop();
            }
            hull.push(p);
        }
        hull.pop();
    }
    hull
}

impl RayIntersectable for Polygon {
    fn ray_visibility(&self, origin: (f64, f64), direction: (f64, f64)) -> Option<RayVisibility> {
        if self.contains(origin.0, origin.1) {
            let heading = normalize(direction)?;
            return self
                .edges()
                .filter_map(|(start, end)| {
                    ray_line_segment_intersection_point(origin, heading, start, end).map(|point| {
                        RayVisibility {
                            min_angle_radians: 0.0,
                            target_point: point,
                        }
                    })
                })
                .min_by(|a, b| compare_visibility(origin, a, b));
        }

        let heading = normalize(direction)?;
        self.edges()
            .filter_map(|(start, end)| segment_visibility(origin, heading, start, end))
            .min_by(|a, b| compare_visibility(origin, a, b))
    }
}

fn compare_visibility(
    origin: (f64, f64),
    a: &RayVisibility,
    b: &RayVisibility,
) -> std::cmp::Ordering {
    match a.min_angle_radians.total_cmp(&b.min_angle_radians) {
        std::cmp::Ordering::Equal => squared_distance(a.target_point, origin)
            .total_cmp(&squared_distance(b.target_point, origin)),
        ordering => ordering,
    }
}

fn segment_visibility(
    origin: (f64, f64),
    heading: (f64, f64),
    start: (f64, f64),
    end: (f64, f64),
) -> Option<RayVisibility> {
    if let Some(point) = ray_line_segment_intersection_point(origin, heading, start, end) {
        return Some(RayVisibility {
            min_angle_radians: 0.0,
            target_point: point,
        });
    }

    [start, end]
        .into_iter()
        .filter_map(|point| {
            let vector = (point.0 - origin.0, point.1 - origin.1);
            normalize(vector).map(|unit| RayVisibility {
                min_angle_radians: angle_between(heading, unit),
                target_point: point,
            })
        })
        .min_by(|a, b| compare_visibility(origin, a, b))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_square() -> Polygon {
        Polygon::new(vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    }

    fn concave_l_shape() -> Polygon {
        Polygon::new(vec![
            (0.0, 0.0),
            (2.0, 0.0),
            (2.0, 1.0),
            (1.0, 1.0),
            (1.0, 2.0),
            (0.0, 2.0),
        ])
    }

    #[test]
    fn test_polygon_contains() {
        let sq = unit_square();
        assert!(sq.contains(0.5, 0.5));
        assert!(!sq.contains(1.5, 0.5));
    }

    #[test]
    fn test_polygon_centroid() {
        let sq = unit_square();
        let (cx, cy) = sq.centroid();
        assert!((cx - 0.5).abs() < f64::EPSILON);
        assert!((cy - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_polygon_canonicalizes_vertex_order() {
        let poly = Polygon::new(vec![(1.0, 1.0), (0.0, 1.0), (0.0, 0.0), (1.0, 0.0)]);
        assert_eq!(
            poly.vertices,
            vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        );
    }

    #[test]
    fn test_polygon_deserialize_canonicalizes_vertex_order() {
        let poly: Polygon =
            toml::from_str("vertices = [[1.0, 1.0], [0.0, 1.0], [0.0, 0.0], [1.0, 0.0]]").unwrap();
        assert_eq!(
            poly.vertices,
            vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        );
    }

    #[test]
    fn test_polygon_closest_point() {
        let sq = unit_square();
        assert_eq!(sq.closest_point(2.0, 0.5), (1.0, 0.5));
    }

    #[test]
    fn test_polygon_expand_is_a_true_offset_of_the_convex_hull() {
        let sq = unit_square();
        let expanded = sq.expand(0.1);
        // Edges move the full distance, not d * cos(theta) as a radial push did.
        assert!(expanded.contains(0.5, -0.099));
        assert!(!expanded.contains(0.5, -0.101));
        assert!(expanded.contains(1.099, 0.5));
        // Corners are rounded: 0.101 along the diagonal is outside.
        let diagonal = 0.101 / 2f64.sqrt();
        assert!(!expanded.contains(1.0 + diagonal, 1.0 + diagonal));
        assert!(expanded.contains(1.0 + 0.099 / 2f64.sqrt(), 1.0 + 0.099 / 2f64.sqrt()));
        // Concave notch is filled by the hull.
        assert!(concave_l_shape().expand(0.01).contains(1.5, 1.5));
    }

    #[test]
    fn test_polygon_erosion_moves_each_edge_inward_by_distance() {
        let eroded = unit_square().expand(-0.1);
        assert!(eroded.contains(0.5, 0.101));
        assert!(!eroded.contains(0.5, 0.099));
        assert!(eroded.contains(0.101, 0.101));
        assert!(!unit_square().expand(-0.6).contains(0.5, 0.5));
    }

    #[test]
    fn test_polygon_ray_intersection() {
        let sq = unit_square();
        assert!(sq.ray_intersects((-1.0, 0.5), (1.0, 0.0)));
        assert!(!sq.ray_intersects((-1.0, 2.0), (1.0, 0.0)));
    }

    #[test]
    fn test_polygon_ray_visibility_reports_first_hit_point() {
        let sq = unit_square();
        let visibility = sq.ray_visibility((-1.0, 0.5), (1.0, 0.0)).unwrap();
        assert!(visibility.min_angle_radians.abs() < 1e-10);
        assert!(visibility.target_point.0.abs() < 1e-10);
        assert!((visibility.target_point.1 - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_polygon_ray_visibility_uses_nearest_endpoint_when_heading_misses() {
        let poly = Polygon::new(vec![(1.0, 0.1), (1.2, 0.1), (1.2, 0.3), (1.0, 0.3)]);
        let visibility = poly.ray_visibility((0.0, 0.0), (1.0, 0.0)).unwrap();
        let expected = (0.1_f64).atan2(1.2_f64);
        assert!((visibility.min_angle_radians - expected).abs() < 1e-10);
        assert_eq!(visibility.target_point, (1.2, 0.1));
    }

    #[test]
    fn test_concave_polygon_ray_visibility_reports_inner_edge_hit() {
        let poly = concave_l_shape();
        let visibility = poly.ray_visibility((1.5, 1.5), (-1.0, 0.0)).unwrap();
        assert!(visibility.min_angle_radians.abs() < 1e-10);
        assert!((visibility.target_point.0 - 1.0).abs() < 1e-10);
        assert!((visibility.target_point.1 - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_polygon_ray_visibility_when_origin_is_on_boundary() {
        let sq = unit_square();
        let visibility = sq.ray_visibility((0.0, 0.5), (1.0, 0.0)).unwrap();
        assert!(visibility.min_angle_radians.abs() < 1e-10);
        assert!(visibility.target_point.0.abs() < 1e-10);
        assert!((visibility.target_point.1 - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_polygon_cone_intersection_accepts_near_miss() {
        let poly = Polygon::new(vec![(1.0, 0.1), (1.2, 0.1), (1.2, 0.3), (1.0, 0.3)]);
        assert!(poly.cone_intersects((0.0, 0.0), (1.0, 0.0), 10.0));
    }

    #[test]
    fn test_polygon_cone_intersection_rejects_tight_cone() {
        let poly = Polygon::new(vec![(1.0, 0.1), (1.2, 0.1), (1.2, 0.3), (1.0, 0.3)]);
        assert!(!poly.cone_intersects((0.0, 0.0), (1.0, 0.0), 3.0));
    }
}
