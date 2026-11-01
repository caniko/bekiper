use serde::{Deserialize, Serialize};

use bekiper_core::shape::{ClosestPoint, Expandable, RayIntersectable, RayVisibility, Shape};
use bekiper_core::types::BoundingBox;

use crate::circle::Circle;
use crate::polygon::Polygon;
use crate::rectangle::Rectangle;
use crate::triangle::Triangle;

/// Enum of shapes that can serve as object perimeters in behavioural tasks.
///
/// Wraps all concrete shape types that support containment, expansion, and
/// ray intersection — the operations needed for exploration heuristics.
/// Using an enum avoids trait-object overhead (`Box<dyn>`) while still
/// allowing heterogeneous collections of perimeters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ObjectShape {
    Circle(Circle),
    Rectangle(Rectangle),
    Polygon(Polygon),
    Triangle(Triangle),
}

impl Shape for ObjectShape {
    fn inspection_perimeter(&self, label: &str) -> Option<bekiper_core::inspection::PerimeterSpec> {
        match self {
            Self::Circle(s) => s.inspection_perimeter(label),
            Self::Rectangle(s) => s.inspection_perimeter(label),
            Self::Polygon(s) => s.inspection_perimeter(label),
            Self::Triangle(s) => s.inspection_perimeter(label),
        }
    }
    fn contains(&self, x: f64, y: f64) -> bool {
        match self {
            Self::Circle(s) => s.contains(x, y),
            Self::Rectangle(s) => s.contains(x, y),
            Self::Polygon(s) => s.contains(x, y),
            Self::Triangle(s) => s.contains(x, y),
        }
    }

    fn bounding_box(&self) -> BoundingBox {
        match self {
            Self::Circle(s) => s.bounding_box(),
            Self::Rectangle(s) => s.bounding_box(),
            Self::Polygon(s) => s.bounding_box(),
            Self::Triangle(s) => s.bounding_box(),
        }
    }
}

impl Expandable for ObjectShape {
    type Output = ObjectShape;

    fn expand(&self, distance: f64) -> ObjectShape {
        match self {
            Self::Circle(s) => ObjectShape::Circle(s.expand(distance)),
            // Rectangle and Triangle expand to Polygon
            Self::Rectangle(s) => ObjectShape::Polygon(s.expand(distance)),
            Self::Polygon(s) => ObjectShape::Polygon(s.expand(distance)),
            Self::Triangle(s) => ObjectShape::Polygon(s.expand(distance)),
        }
    }
}

impl ClosestPoint for ObjectShape {
    fn closest_point(&self, x: f64, y: f64) -> (f64, f64) {
        match self {
            Self::Circle(s) => s.closest_point(x, y),
            Self::Rectangle(s) => s.closest_point(x, y),
            Self::Polygon(s) => s.closest_point(x, y),
            Self::Triangle(s) => s.closest_point(x, y),
        }
    }
}

impl RayIntersectable for ObjectShape {
    fn ray_visibility(&self, origin: (f64, f64), direction: (f64, f64)) -> Option<RayVisibility> {
        match self {
            Self::Circle(s) => s.ray_visibility(origin, direction),
            Self::Rectangle(s) => s.ray_visibility(origin, direction),
            Self::Polygon(s) => s.ray_visibility(origin, direction),
            Self::Triangle(s) => s.ray_visibility(origin, direction),
        }
    }
}

impl ObjectShape {
    /// Compute the centroid of this shape.
    pub fn centroid(&self) -> (f64, f64) {
        match self {
            Self::Circle(c) => (c.center_x, c.center_y),
            Self::Rectangle(r) => (r.center_x, r.center_y),
            Self::Polygon(p) => p.centroid(),
            Self::Triangle(t) => {
                let v = t.vertices();
                (
                    (v[0].0 + v[1].0 + v[2].0) / 3.0,
                    (v[0].1 + v[1].1 + v[2].1) / 3.0,
                )
            }
        }
    }
}

impl From<Circle> for ObjectShape {
    fn from(s: Circle) -> Self {
        Self::Circle(s)
    }
}

impl From<Rectangle> for ObjectShape {
    fn from(s: Rectangle) -> Self {
        Self::Rectangle(s)
    }
}

impl From<Polygon> for ObjectShape {
    fn from(s: Polygon) -> Self {
        Self::Polygon(s)
    }
}

impl From<Triangle> for ObjectShape {
    fn from(s: Triangle) -> Self {
        Self::Triangle(s)
    }
}
