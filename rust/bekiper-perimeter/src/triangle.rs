use serde::{Deserialize, Serialize};

use crate::polygon::Polygon;
use bekiper_core::shape::{ClosestPoint, Expandable, RayIntersectable, RayVisibility, Shape};
use bekiper_core::types::BoundingBox;

/// A triangle defined by three vertices. Delegates to `Polygon` — DRY.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Triangle {
    polygon: Polygon,
}

impl Triangle {
    pub fn new(v1: (f64, f64), v2: (f64, f64), v3: (f64, f64)) -> Self {
        Self {
            polygon: Polygon::new(vec![v1, v2, v3]),
        }
    }

    pub fn vertices(&self) -> &[(f64, f64)] {
        &self.polygon.vertices
    }
}

impl Shape for Triangle {
    fn inspection_perimeter(&self, label: &str) -> Option<bekiper_core::inspection::PerimeterSpec> {
        Some(bekiper_core::inspection::PerimeterSpec::Triangle {
            label: label.into(),
            vertices: self.vertices().try_into().ok()?,
        })
    }
    fn contains(&self, x: f64, y: f64) -> bool {
        self.polygon.contains(x, y)
    }

    fn bounding_box(&self) -> BoundingBox {
        self.polygon.bounding_box()
    }
}

impl Expandable for Triangle {
    type Output = Polygon;

    fn expand(&self, distance: f64) -> Polygon {
        self.polygon.expand(distance)
    }
}

impl ClosestPoint for Triangle {
    fn closest_point(&self, x: f64, y: f64) -> (f64, f64) {
        self.polygon.closest_point(x, y)
    }
}

impl RayIntersectable for Triangle {
    fn ray_visibility(&self, origin: (f64, f64), direction: (f64, f64)) -> Option<RayVisibility> {
        self.polygon.ray_visibility(origin, direction)
    }
}
