use image::{Rgb, RgbImage};

use bekiper_core::inspection::PerimeterSpec;

use super::{draw_line_alpha, draw_polyline_alpha};

#[allow(clippy::too_many_arguments)]
pub fn draw_wedge_alpha(
    img: &mut RgbImage,
    cx: f64,
    cy: f64,
    radius: f64,
    start_deg: f64,
    end_deg: f64,
    color: Rgb<u8>,
    thickness: i32,
    outline_alpha: f32,
    fill_alpha: f32,
) {
    let start = start_deg.to_radians();
    let end = end_deg.to_radians();
    let span = end - start;
    let steps = (radius.abs() * 3.0).max(40.0) as usize;
    let mut points = Vec::with_capacity(steps + 1);
    for i in 0..=steps {
        let t = start + span * (i as f64) / (steps as f64);
        points.push((cx + radius * t.cos(), cy + radius * t.sin()));
    }

    if fill_alpha > 0.0 {
        for &(px, py) in &points {
            draw_line_alpha(img, cx, cy, px, py, color, 1, fill_alpha);
        }
    }

    if let Some(first) = points.first().copied() {
        draw_line_alpha(
            img,
            cx,
            cy,
            first.0,
            first.1,
            color,
            thickness,
            outline_alpha,
        );
    }
    if let Some(last) = points.last().copied() {
        draw_line_alpha(img, cx, cy, last.0, last.1, color, thickness, outline_alpha);
    }
    draw_polyline_alpha(img, &points, color, thickness, false, outline_alpha);
}

pub fn scale_point(x: f64, y: f64, sx: f64, sy: f64) -> (f64, f64) {
    (x * sx, y * sy)
}

pub fn scale_points(points: &[(f64, f64)], sx: f64, sy: f64) -> Vec<(f64, f64)> {
    points
        .iter()
        .map(|&(x, y)| scale_point(x, y, sx, sy))
        .collect()
}

pub fn scale_perimeter(spec: &PerimeterSpec, sx: f64, sy: f64) -> PerimeterSpec {
    match spec {
        PerimeterSpec::Circle {
            label,
            center_x,
            center_y,
            radius,
        } => {
            if sx == sy {
                PerimeterSpec::Circle {
                    label: label.clone(),
                    center_x: center_x * sx,
                    center_y: center_y * sy,
                    radius: radius * sx.abs(),
                }
            } else {
                // Render-only affine outline: a non-uniformly scaled circle is
                // an ellipse. Bound chord error to 0.25 displayed pixels and
                // retain exact cardinal endpoints. Native geometry is untouched.
                let max_radius = radius.abs() * sx.abs().max(sy.abs());
                let steps = ((std::f64::consts::PI * (max_radius / 0.5).sqrt()).ceil() as usize)
                    .max(32)
                    .next_multiple_of(4);
                PerimeterSpec::Polygon {
                    label: label.clone(),
                    vertices: (0..steps)
                        .map(|index| {
                            let angle = std::f64::consts::TAU * index as f64 / steps as f64;
                            (
                                (center_x + radius * angle.cos()) * sx,
                                (center_y + radius * angle.sin()) * sy,
                            )
                        })
                        .collect(),
                }
            }
        }
        PerimeterSpec::Rectangle {
            label,
            center_x,
            center_y,
            width,
            height,
        } => PerimeterSpec::Rectangle {
            label: label.clone(),
            center_x: center_x * sx,
            center_y: center_y * sy,
            width: width * sx,
            height: height * sy,
        },
        PerimeterSpec::Polygon { label, vertices } => PerimeterSpec::Polygon {
            label: label.clone(),
            vertices: scale_points(vertices, sx, sy),
        },
        PerimeterSpec::Triangle { label, vertices } => PerimeterSpec::Triangle {
            label: label.clone(),
            vertices: [
                scale_point(vertices[0].0, vertices[0].1, sx, sy),
                scale_point(vertices[1].0, vertices[1].1, sx, sy),
                scale_point(vertices[2].0, vertices[2].1, sx, sy),
            ],
        },
        PerimeterSpec::RadialMaze {
            label,
            center_vertices,
            arms,
        } => PerimeterSpec::RadialMaze {
            label: label.clone(),
            center_vertices: scale_points(center_vertices, sx, sy),
            arms: arms.iter().map(|arm| scale_points(arm, sx, sy)).collect(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::overlay::draw_perimeter;

    #[test]
    fn circle_geometry_uses_full_frame_transform() {
        let circle = PerimeterSpec::Circle {
            label: "native".into(),
            center_x: 100.0,
            center_y: 100.0,
            radius: 20.0,
        };
        let scaled = scale_perimeter(&circle, 0.25, 0.5);
        let mut image = RgbImage::new(100, 100);
        let color = Rgb([255, 0, 0]);
        draw_perimeter(&mut image, &scaled, color, 1, 1.0);
        for (x, y) in [(20, 50), (30, 50), (25, 40), (25, 60)] {
            assert_eq!(
                *image.get_pixel(x, y),
                color,
                "ellipse boundary at ({x}, {y})"
            );
        }
        assert_eq!(*image.get_pixel(25, 45), Rgb([0, 0, 0]));
        let uniform = scale_perimeter(&circle, 0.5, 0.5);
        assert!(matches!(
            uniform,
            PerimeterSpec::Circle {
                center_x: 50.0,
                center_y: 50.0,
                radius: 10.0,
                ..
            }
        ));
        assert!(matches!(circle, PerimeterSpec::Circle { radius: 20.0, .. }));
    }
}
