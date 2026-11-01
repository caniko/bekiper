//! Canonical polygon parsing and ray-casting containment.
use super::*;

pub(super) fn polygon(value: &Value) -> Result<Vec<[f64; 2]>> {
    let vertices = value
        .as_array()
        .context("Geometry requires polygon vertices")?;
    let mut points = Vec::new();
    for vertex in vertices {
        let pair = vertex
            .as_array()
            .context("Geometry vertex must be a coordinate pair")?;
        ensure!(pair.len() == 2, "Geometry vertex must be a coordinate pair");
        let point = [
            finite(&pair[0], "geometry x")?,
            finite(&pair[1], "geometry y")?,
        ];
        ensure!(
            !points.contains(&point),
            "Geometry requires unique canonical polygon vertices"
        );
        points.push(point);
    }
    ensure!(
        points.len() >= 3,
        "Geometry requires at least three polygon vertices"
    );
    let area: f64 = points
        .iter()
        .zip(points.iter().cycle().skip(1))
        .take(points.len())
        .map(|(a, b)| a[0] * b[1] - b[0] * a[1])
        .sum();
    ensure!(area.is_finite() && area != 0.0, "Degenerate polygon");
    Ok(points)
}

/// Same strict ray-casting inequalities as bekiper-math::confinement.
pub fn polygon_contains(x: f64, y: f64, vertices: &[[f64; 2]]) -> bool {
    if vertices.is_empty() {
        return false;
    }
    let mut inside = false;
    let mut j = vertices.len() - 1;
    for (i, [xi, yi]) in vertices.iter().copied().enumerate() {
        let [xj, yj] = vertices[j];
        if (yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi {
            inside = !inside;
        }
        j = i;
    }
    inside
}
