use serde::{Deserialize, Serialize};

use crate::polygon::Polygon;
use bekiper_core::shape::{ClosestPoint, Expandable, RayIntersectable, RayVisibility, Shape};
use bekiper_core::types::BoundingBox;

/// A rectangle defined by center, width, height, and optional rotation.
/// Delegates to `Polygon` for all geometric operations — DRY.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Rectangle {
    pub center_x: f64,
    pub center_y: f64,
    pub width: f64,
    pub height: f64,
    polygon: Polygon,
}

impl Rectangle {
    pub fn new(center_x: f64, center_y: f64, width: f64, height: f64) -> Self {
        let hw = width / 2.0;
        let hh = height / 2.0;
        let polygon = Polygon::new(vec![
            (center_x - hw, center_y - hh),
            (center_x + hw, center_y - hh),
            (center_x + hw, center_y + hh),
            (center_x - hw, center_y + hh),
        ]);
        Self {
            center_x,
            center_y,
            width,
            height,
            polygon,
        }
    }
}

// All Shape/Expandable/RayIntersectable delegated to inner Polygon — DRY.
impl Shape for Rectangle {
    fn inspection_perimeter(&self, label: &str) -> Option<bekiper_core::inspection::PerimeterSpec> {
        Some(bekiper_core::inspection::PerimeterSpec::Rectangle {
            label: label.into(),
            center_x: self.center_x,
            center_y: self.center_y,
            width: self.width,
            height: self.height,
        })
    }
    fn contains(&self, x: f64, y: f64) -> bool {
        self.polygon.contains(x, y)
    }

    fn bounding_box(&self) -> BoundingBox {
        self.polygon.bounding_box()
    }
}

impl Expandable for Rectangle {
    type Output = Polygon;

    fn expand(&self, distance: f64) -> Polygon {
        self.polygon.expand(distance)
    }
}

impl ClosestPoint for Rectangle {
    fn closest_point(&self, x: f64, y: f64) -> (f64, f64) {
        self.polygon.closest_point(x, y)
    }
}

impl RayIntersectable for Rectangle {
    fn ray_visibility(&self, origin: (f64, f64), direction: (f64, f64)) -> Option<RayVisibility> {
        self.polygon.ray_visibility(origin, direction)
    }
}
