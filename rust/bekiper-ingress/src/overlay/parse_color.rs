use image::{DynamicImage, Rgb, RgbImage};

use bekiper_core::inspection::PerimeterSpec;

// ── Colours ─────────────────────────────────────────────────────────────────

pub const RED: Rgb<u8> = Rgb([255, 60, 60]);
pub const CYAN: Rgb<u8> = Rgb([0, 220, 220]);
pub const LIME: Rgb<u8> = Rgb([60, 255, 60]);
pub const YELLOW: Rgb<u8> = Rgb([255, 255, 60]);
pub const MAGENTA: Rgb<u8> = Rgb([255, 60, 255]);
pub const WHITE: Rgb<u8> = Rgb([255, 255, 255]);
pub const BLACK: Rgb<u8> = Rgb([0, 0, 0]);

const ARM_COLORS: [Rgb<u8>; 6] = [CYAN, LIME, YELLOW, MAGENTA, WHITE, RED];

// ── Public API ──────────────────────────────────────────────────────────────

/// Parse a small subset of CSS / matplotlib-style color strings.
///
/// Supports:
/// - `#RRGGBB`
/// - `#RGB`
/// - common named colors used by the inspection theme
/// - matplotlib one-letter aliases like `b`, `g`, `r`, `c`, `m`, `y`, `k`, `w`
pub fn parse_color(spec: &str) -> Rgb<u8> {
    let value = spec.trim().to_ascii_lowercase();

    if let Some(hex) = value.strip_prefix('#') {
        return match hex.len() {
            3 => {
                let chars: Vec<char> = hex.chars().collect();
                let r = u8::from_str_radix(&format!("{0}{0}", chars[0]), 16).ok();
                let g = u8::from_str_radix(&format!("{0}{0}", chars[1]), 16).ok();
                let b = u8::from_str_radix(&format!("{0}{0}", chars[2]), 16).ok();
                match (r, g, b) {
                    (Some(r), Some(g), Some(b)) => Rgb([r, g, b]),
                    _ => WHITE,
                }
            }
            6 => {
                let r = u8::from_str_radix(&hex[0..2], 16).ok();
                let g = u8::from_str_radix(&hex[2..4], 16).ok();
                let b = u8::from_str_radix(&hex[4..6], 16).ok();
                match (r, g, b) {
                    (Some(r), Some(g), Some(b)) => Rgb([r, g, b]),
                    _ => WHITE,
                }
            }
            _ => WHITE,
        };
    }

    match value.as_str() {
        "red" | "r" => RED,
        "lime" | "green" | "g" => LIME,
        "blue" | "b" => Rgb([70, 130, 255]),
        "cyan" | "c" => CYAN,
        "magenta" | "m" => MAGENTA,
        "yellow" | "y" => YELLOW,
        "white" | "w" => WHITE,
        "black" | "k" => BLACK,
        "gray" | "grey" => Rgb([150, 150, 150]),
        "dimgray" | "dimgrey" => Rgb([105, 105, 105]),
        "steelblue" => Rgb([70, 130, 180]),
        "darkorange" => Rgb([255, 140, 0]),
        "dodgerblue" => Rgb([30, 144, 255]),
        "crimson" => Rgb([220, 20, 60]),
        "orange" => Rgb([255, 165, 0]),
        "gold" => Rgb([255, 215, 0]),
        "purple" => Rgb([128, 0, 128]),
        "pink" => Rgb([255, 105, 180]),
        "brown" => Rgb([165, 42, 42]),
        _ => WHITE,
    }
}

/// Draw perimeter annotations onto a video frame.
pub fn draw_perimeters(img: &mut RgbImage, specs: &[PerimeterSpec]) {
    for spec in specs {
        let (color, thickness) = match spec {
            PerimeterSpec::Triangle { .. } => (MAGENTA, 2),
            _ => (CYAN, 2),
        };
        draw_perimeter(img, spec, color, thickness, 1.0);
    }
}

/// Draw one perimeter annotation onto a video frame.
pub fn draw_perimeter(
    img: &mut RgbImage,
    spec: &PerimeterSpec,
    color: Rgb<u8>,
    thickness: i32,
    alpha: f32,
) {
    match spec {
        PerimeterSpec::Circle {
            center_x,
            center_y,
            radius,
            ..
        } => {
            draw_circle_alpha(img, *center_x, *center_y, *radius, color, thickness, alpha);
        }
        PerimeterSpec::Rectangle {
            center_x,
            center_y,
            width,
            height,
            ..
        } => {
            let x0 = center_x - width / 2.0;
            let y0 = center_y - height / 2.0;
            draw_rect_alpha(img, x0, y0, *width, *height, color, thickness, alpha);
        }
        PerimeterSpec::Polygon { vertices, .. } => {
            draw_polygon_alpha(img, vertices, color, thickness, alpha);
        }
        PerimeterSpec::Triangle { vertices, .. } => {
            let verts: Vec<(f64, f64)> = vertices.to_vec();
            draw_polygon_alpha(img, &verts, color, thickness, alpha);
        }
        PerimeterSpec::RadialMaze {
            center_vertices,
            arms,
            ..
        } => {
            for (i, arm) in arms.iter().enumerate() {
                let arm_color = if alpha >= 0.999 {
                    ARM_COLORS[i % ARM_COLORS.len()]
                } else {
                    color
                };
                draw_polyline_alpha(img, arm, arm_color, thickness, false, alpha);
                if let Some(&(x, y)) = arm.first() {
                    draw_crosshair_alpha(img, x, y, 6, arm_color, 1, alpha);
                }
                if let Some(&(x, y)) = arm.last() {
                    draw_crosshair_alpha(img, x, y, 6, arm_color, 1, alpha);
                }
            }
            if !center_vertices.is_empty() {
                draw_polygon_alpha(img, center_vertices, color, thickness, alpha);
            }
        }
    }
}

/// Draw a reference-point crosshair (circle + cross).
pub fn draw_reference_point(img: &mut RgbImage, x: f64, y: f64) {
    draw_circle(img, x, y, 10.0, RED, 2);
    draw_line(img, x - 14.0, y, x + 14.0, y, RED, 2);
    draw_line(img, x, y - 14.0, x, y + 14.0, RED, 2);
}

/// Convert a DynamicImage to RgbImage for drawing.
pub fn to_rgb(img: &DynamicImage) -> RgbImage {
    img.to_rgb8()
}

// ── Drawing primitives ──────────────────────────────────────────────────────

pub(super) fn blend_pixel(img: &mut RgbImage, x: i32, y: i32, color: Rgb<u8>, alpha: f32) {
    if x < 0 || y < 0 || (x as u32) >= img.width() || (y as u32) >= img.height() {
        return;
    }

    if alpha >= 0.999 {
        img.put_pixel(x as u32, y as u32, color);
        return;
    }

    let dst = *img.get_pixel(x as u32, y as u32);
    let a = alpha.clamp(0.0, 1.0);
    let blend =
        |src: u8, dst: u8| -> u8 { ((src as f32 * a) + (dst as f32 * (1.0 - a))).round() as u8 };
    img.put_pixel(
        x as u32,
        y as u32,
        Rgb([
            blend(color[0], dst[0]),
            blend(color[1], dst[1]),
            blend(color[2], dst[2]),
        ]),
    );
}

pub fn draw_line(
    img: &mut RgbImage,
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
    color: Rgb<u8>,
    thickness: i32,
) {
    draw_line_alpha(img, x0, y0, x1, y1, color, thickness, 1.0);
}

#[allow(clippy::too_many_arguments)]
pub fn draw_line_alpha(
    img: &mut RgbImage,
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
    color: Rgb<u8>,
    thickness: i32,
    alpha: f32,
) {
    let (mut x0, mut y0) = (x0.round() as i32, y0.round() as i32);
    let (x1, y1) = (x1.round() as i32, y1.round() as i32);
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;

    let half = thickness.max(1) / 2;
    loop {
        for tx in -half..=half {
            for ty in -half..=half {
                blend_pixel(img, x0 + tx, y0 + ty, color, alpha);
            }
        }
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

pub fn draw_circle(
    img: &mut RgbImage,
    cx: f64,
    cy: f64,
    radius: f64,
    color: Rgb<u8>,
    thickness: i32,
) {
    draw_circle_alpha(img, cx, cy, radius, color, thickness, 1.0);
}

pub fn draw_circle_alpha(
    img: &mut RgbImage,
    cx: f64,
    cy: f64,
    radius: f64,
    color: Rgb<u8>,
    thickness: i32,
    alpha: f32,
) {
    let steps = (radius.abs() * 4.0).max(60.0) as usize;
    for i in 0..steps {
        let a0 = 2.0 * std::f64::consts::PI * (i as f64) / (steps as f64);
        let a1 = 2.0 * std::f64::consts::PI * ((i + 1) as f64) / (steps as f64);
        draw_line_alpha(
            img,
            cx + radius * a0.cos(),
            cy + radius * a0.sin(),
            cx + radius * a1.cos(),
            cy + radius * a1.sin(),
            color,
            thickness,
            alpha,
        );
    }
}

pub fn draw_rect(
    img: &mut RgbImage,
    x: f64,
    y: f64,
    w: f64,
    h: f64,
    color: Rgb<u8>,
    thickness: i32,
) {
    draw_rect_alpha(img, x, y, w, h, color, thickness, 1.0);
}

#[allow(clippy::too_many_arguments)]
pub fn draw_rect_alpha(
    img: &mut RgbImage,
    x: f64,
    y: f64,
    w: f64,
    h: f64,
    color: Rgb<u8>,
    thickness: i32,
    alpha: f32,
) {
    draw_line_alpha(img, x, y, x + w, y, color, thickness, alpha);
    draw_line_alpha(img, x + w, y, x + w, y + h, color, thickness, alpha);
    draw_line_alpha(img, x + w, y + h, x, y + h, color, thickness, alpha);
    draw_line_alpha(img, x, y + h, x, y, color, thickness, alpha);
}

pub fn fill_rect_alpha(
    img: &mut RgbImage,
    x: f64,
    y: f64,
    w: f64,
    h: f64,
    color: Rgb<u8>,
    alpha: f32,
) {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = (x + w).ceil() as i32;
    let y1 = (y + h).ceil() as i32;
    for py in y0..y1 {
        for px in x0..x1 {
            blend_pixel(img, px, py, color, alpha);
        }
    }
}

pub fn draw_polygon(img: &mut RgbImage, vertices: &[(f64, f64)], color: Rgb<u8>, thickness: i32) {
    draw_polygon_alpha(img, vertices, color, thickness, 1.0);
}

pub fn draw_polygon_alpha(
    img: &mut RgbImage,
    vertices: &[(f64, f64)],
    color: Rgb<u8>,
    thickness: i32,
    alpha: f32,
) {
    draw_polyline_alpha(img, vertices, color, thickness, true, alpha);
}

pub fn draw_polyline_alpha(
    img: &mut RgbImage,
    vertices: &[(f64, f64)],
    color: Rgb<u8>,
    thickness: i32,
    close: bool,
    alpha: f32,
) {
    if vertices.len() < 2 {
        return;
    }
    for pair in vertices.windows(2) {
        draw_line_alpha(
            img, pair[0].0, pair[0].1, pair[1].0, pair[1].1, color, thickness, alpha,
        );
    }
    if close && let (Some(first), Some(last)) = (vertices.first(), vertices.last()) {
        draw_line_alpha(
            img, last.0, last.1, first.0, first.1, color, thickness, alpha,
        );
    }
}

pub fn draw_crosshair(
    img: &mut RgbImage,
    x: f64,
    y: f64,
    size: i32,
    color: Rgb<u8>,
    thickness: i32,
) {
    draw_crosshair_alpha(img, x, y, size, color, thickness, 1.0);
}

pub fn draw_crosshair_alpha(
    img: &mut RgbImage,
    x: f64,
    y: f64,
    size: i32,
    color: Rgb<u8>,
    thickness: i32,
    alpha: f32,
) {
    let s = size as f64;
    draw_line_alpha(img, x - s, y, x + s, y, color, thickness, alpha);
    draw_line_alpha(img, x, y - s, x, y + s, color, thickness, alpha);
}

#[allow(clippy::too_many_arguments)]
pub fn draw_arrow_alpha(
    img: &mut RgbImage,
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
    color: Rgb<u8>,
    thickness: i32,
    alpha: f32,
) {
    draw_line_alpha(img, x0, y0, x1, y1, color, thickness, alpha);

    let dx = x1 - x0;
    let dy = y1 - y0;
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1.0 {
        return;
    }
    let ux = dx / len;
    let uy = dy / len;
    let head_len = (len * 0.18).clamp(6.0, 16.0);
    let wing = head_len * 0.45;
    let bx = x1 - ux * head_len;
    let by = y1 - uy * head_len;
    let left_x = bx - uy * wing;
    let left_y = by + ux * wing;
    let right_x = bx + uy * wing;
    let right_y = by - ux * wing;
    draw_line_alpha(img, x1, y1, left_x, left_y, color, thickness, alpha);
    draw_line_alpha(img, x1, y1, right_x, right_y, color, thickness, alpha);
}
