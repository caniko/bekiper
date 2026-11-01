/// Strip geometry in pixels: first-row top, bar height, row pitch, label
/// gutter left of the bars, and the footer (time axis, then legend) below the
/// rows. `unit` is the integer frame upscale and `text` the glyph magnification.
#[derive(Debug, Clone, Copy)]
struct StripLayout {
    top: u32,
    row_height: u32,
    pitch: u32,
    gutter: u32,
    axis: u32,
    footer: u32,
    unit: u32,
    text: u32,
}

const SCORED_LAYOUT: StripLayout = StripLayout {
    top: SCORED_STRIP_TOP,
    row_height: SCORED_ROW_HEIGHT,
    pitch: SCORED_ROW_PITCH,
    gutter: 0,
    axis: 0,
    footer: 0,
    unit: 1,
    text: 1,
};

/// Qualia strip for a frame upscaled by `unit`: glyphs are magnified by
/// `unit` (at least 2, so rows stay 10 px tall), seven label characters fit
/// the gutter, and the footer holds an 8-glyph-row axis plus three legend lines.
fn qualia_layout(unit: u32) -> StripLayout {
    let s = unit.max(2);
    StripLayout {
        top: s,
        row_height: 5 * s,
        pitch: 6 * s,
        gutter: 30 * s,
        axis: 8 * s,
        footer: 8 * s + 22 * s,
        unit,
        text: s,
    }
}

// Frame overlays keep fixed colours across night/day mode (AGENTS.md Theming).
// Cone and fans take three distinct hues that avoid the object blue and red.
const CONE_COLOR: &str = "#E69F00";
const FAN_COLORS: [&str; 2] = ["#3fd18b", "#c299ff"];

/// Colour of an object's outline and strip cells, as in the binary quiver
/// figures: novel red, familiar blue (palette false/true tokens); others take
/// a categorical entry.
fn object_color(label: &str, index: usize) -> image::Rgb<u8> {
    let label = label.to_ascii_lowercase();
    parse_color(if label.contains("novel") {
        themekit::DARK.false_mask
    } else if label.contains("familiar") {
        themekit::DARK.true_mask
    } else {
        themekit::DARK.categorical[(2 + index) % 10]
    })
}

/// 3x5 capitals, five rows of three bits (bit 2 is the left column): A-Z, 0-9.
const GLYPHS: [[u8; 5]; 36] = [
    [2, 5, 7, 5, 5],
    [6, 5, 6, 5, 6],
    [3, 4, 4, 4, 3],
    [6, 5, 5, 5, 6],
    [7, 4, 6, 4, 7],
    [7, 4, 6, 4, 4],
    [3, 4, 5, 5, 3],
    [5, 5, 7, 5, 5],
    [7, 2, 2, 2, 7],
    [1, 1, 1, 5, 2],
    [5, 5, 6, 5, 5],
    [4, 4, 4, 4, 7],
    [5, 7, 7, 5, 5],
    [6, 5, 5, 5, 5],
    [2, 5, 5, 5, 2],
    [6, 5, 6, 4, 4],
    [2, 5, 5, 6, 3],
    [6, 5, 6, 5, 5],
    [3, 4, 2, 1, 6],
    [7, 2, 2, 2, 2],
    [5, 5, 5, 5, 7],
    [5, 5, 5, 5, 2],
    [5, 5, 7, 7, 5],
    [5, 5, 2, 5, 5],
    [5, 5, 2, 2, 2],
    [7, 1, 2, 4, 7],
    [7, 5, 5, 5, 7],
    [2, 6, 2, 2, 7],
    [6, 1, 2, 4, 7],
    [6, 1, 2, 1, 6],
    [5, 5, 7, 1, 1],
    [7, 4, 6, 1, 6],
    [3, 4, 7, 5, 7],
    [7, 1, 2, 2, 2],
    [7, 5, 7, 5, 7],
    [7, 5, 7, 1, 6],
];

fn glyph(c: char) -> [u8; 5] {
    match c.to_ascii_uppercase() {
        c @ 'A'..='Z' => GLYPHS[(c as u8 - b'A') as usize],
        c @ '0'..='9' => GLYPHS[26 + (c as u8 - b'0') as usize],
        '_' => [0, 0, 0, 0, 7],
        '-' => [0, 0, 7, 0, 0],
        _ => [0; 5],
    }
}

/// `text` in 3x5 capitals magnified by `scale`, clipped to the image.
/// ponytail: no font dependency; characters outside A-Z, 0-9, `_` and `-` draw blank.
fn draw_text(img: &mut RgbImage, x: u32, y: u32, text: &str, color: image::Rgb<u8>, scale: u32) {
    for (index, c) in text.chars().enumerate() {
        let left = x + index as u32 * 4 * scale;
        for (row, bits) in glyph(c).into_iter().enumerate() {
            for col in (0..3).filter(|col| bits >> (2 - col) & 1 == 1) {
                for (dx, dy) in (0..scale).flat_map(|dx| (0..scale).map(move |dy| (dx, dy))) {
                    let (px, py) = (left + col * scale + dx, y + row as u32 * scale + dy);
                    if px < img.width() && py < img.height() {
                        img.put_pixel(px, py, color);
                    }
                }
            }
        }
    }
}

/// One strip row per channel per object, grouped by object. A channel whose
/// column is absent from the evaluated frame has no row.
fn qualia_rows(df: &DataFrame, feature: &FeatureInspection) -> Result<Vec<ScoredRow>> {
    let FeatureInspection::DetectionFields {
        object_labels,
        nose_near_columns,
        nose_gate_columns,
        whisker_hit_columns,
        ..
    } = feature
    else {
        return Ok(Vec::new());
    };
    type Channel = Option<(Vec<Option<bool>>, String)>;
    let column = |name: Option<&String>| -> Result<Channel> {
        match name {
            Some(name) if df.column(name).is_ok() => {
                Ok(Some((load_bool_column(df, name)?, name.clone())))
            }
            _ => Ok(None),
        }
    };
    let mut rows = Vec::new();
    for (index, object) in object_labels.iter().enumerate() {
        let letter = (b'A' + (index % 26) as u8) as char;
        // The nose rule fires when the nose is in reach and passes the contact-or-angle gate.
        let nose = match (
            column(nose_near_columns.get(index))?,
            column(nose_gate_columns.get(index))?,
        ) {
            (Some((near, near_name)), Some((gate, gate_name))) => Some((
                near.iter()
                    .zip(&gate)
                    .map(|(a, b)| a.zip(*b).map(|(a, b)| a && b))
                    .collect(),
                format!("{near_name}&{gate_name}"),
            )),
            _ => None,
        };
        let exploring = format!("exploring_{object}");
        let named = |name: String| column(Some(&name));
        let channels = [
            ("NOSE", "nose_rule", nose),
            (
                "WHISK",
                "whisker_fan",
                column(whisker_hit_columns.get(index))?,
            ),
            ("OLF", "olfaction", named(format!("olfaction_{object}"))?),
            (
                "BODY",
                "body_proximity",
                named(format!("body_proximity_{object}"))?,
            ),
            (
                "EXPL",
                "combined_exploration",
                named(bekiper_feature::heuristic::diag_column_name(
                    &exploring,
                    "raw_combined",
                ))?,
            ),
            (
                "BOUT",
                "accepted_bouts",
                named(bekiper_feature::heuristic::diag_column_name(
                    &exploring,
                    "final_toleranced",
                ))?,
            ),
        ];
        for (tag, channel, found) in channels {
            if let Some((mask, result_column)) = found {
                rows.push(ScoredRow {
                    label: format!("{letter} {tag}"),
                    heuristic: format!("{object}:{channel}"),
                    result_column,
                    perimeter: None,
                    color: Some(object_color(object, index)),
                    mask,
                });
            }
        }
    }
    Ok(rows)
}

/// Legend below the axis, one swatch per colour drawn: overlay fields,
/// the true/hit and false cell colours with the playhead, then the A/B objects.
fn draw_qualia_legend(strip: &mut RgbImage, feature: &FeatureInspection, top: u32, s: u32) {
    let FeatureInspection::DetectionFields {
        object_labels,
        whisker_half_angle_deg,
        ..
    } = feature
    else {
        return;
    };
    let ink = parse_color(themekit::DARK.ink);
    // A swatch split into one stripe per colour, then its caption.
    let item =
        |strip: &mut RgbImage, x: &mut u32, y: u32, colors: &[image::Rgb<u8>], text: &str| {
            for (index, color) in colors.iter().enumerate() {
                let stripe = 5 * s / colors.len() as u32;
                for (dx, dy) in (0..stripe).flat_map(|dx| (0..5 * s).map(move |dy| (dx, dy))) {
                    if *x + index as u32 * stripe + dx < strip.width() {
                        strip.put_pixel(*x + index as u32 * stripe + dx, y + dy, *color);
                    }
                }
            }
            draw_text(strip, *x + 6 * s, y, text, ink, s);
            *x += 6 * s + text.len() as u32 * 4 * s + 3 * s;
        };
    let pitch = 7 * s;
    let mut x = s;
    item(strip, &mut x, top, &[parse_color(CONE_COLOR)], "CONE");
    item(strip, &mut x, top, &[ink], "CONTACT");
    if whisker_half_angle_deg.is_some() {
        item(strip, &mut x, top, &[parse_color(FAN_COLORS[0])], "FAN L");
        item(strip, &mut x, top, &[parse_color(FAN_COLORS[1])], "FAN R");
    }
    x = s;
    let objects: Vec<_> = object_labels
        .iter()
        .enumerate()
        .map(|(index, object)| object_color(object, index))
        .collect();
    item(strip, &mut x, top + pitch, &objects, "TRUE OR HIT");
    item(
        strip,
        &mut x,
        top + pitch,
        &[parse_color(themekit::DARK.border)],
        "FALSE",
    );
    item(strip, &mut x, top + pitch, &[ink], "PLAYHEAD");
    x = s;
    for (index, object) in object_labels.iter().enumerate() {
        let text = format!("{} {object}", (b'A' + (index % 26) as u8) as char);
        item(strip, &mut x, top + 2 * pitch, &[objects[index]], &text);
    }
}

/// Vertices of a perimeter outline; empty for mazes.
fn perimeter_vertices(spec: &PerimeterSpec) -> Vec<(f64, f64)> {
    match spec {
        PerimeterSpec::Circle {
            center_x,
            center_y,
            radius,
            ..
        } => (0..48)
            .map(|i| {
                let a = std::f64::consts::TAU * f64::from(i) / 48.0;
                (center_x + radius * a.cos(), center_y + radius * a.sin())
            })
            .collect(),
        PerimeterSpec::Rectangle {
            center_x,
            center_y,
            width,
            height,
            ..
        } => {
            let (hw, hh) = (width / 2.0, height / 2.0);
            vec![
                (center_x - hw, center_y - hh),
                (center_x + hw, center_y - hh),
                (center_x + hw, center_y + hh),
                (center_x - hw, center_y + hh),
            ]
        }
        PerimeterSpec::Polygon { vertices, .. } => vertices.clone(),
        PerimeterSpec::Triangle { vertices, .. } => vertices.to_vec(),
        PerimeterSpec::RadialMaze { .. } => Vec::new(),
    }
}

/// Even-odd fill of a perimeter, pixel centres tested.
/// ponytail: per-pixel scan of the bounding box; objects are a few hundred pixels.
fn fill_perimeter(img: &mut RgbImage, spec: &PerimeterSpec, color: image::Rgb<u8>, alpha: f32) {
    let vertices = perimeter_vertices(spec);
    if vertices.len() < 3 {
        return;
    }
    let fold = |f: fn(f64, f64) -> f64, pick: fn(&(f64, f64)) -> f64, init: f64| {
        vertices.iter().map(pick).fold(init, f)
    };
    let (x0, x1) = (
        fold(f64::min, |v| v.0, f64::MAX).floor().max(0.0) as i32,
        fold(f64::max, |v| v.0, f64::MIN).ceil() as i32,
    );
    let (y0, y1) = (
        fold(f64::min, |v| v.1, f64::MAX).floor().max(0.0) as i32,
        fold(f64::max, |v| v.1, f64::MIN).ceil() as i32,
    );
    for y in y0..=y1.min(img.height() as i32 - 1) {
        for x in x0..=x1.min(img.width() as i32 - 1) {
            let (px, py) = (f64::from(x) + 0.5, f64::from(y) + 0.5);
            let inside = (0..vertices.len()).fold(false, |inside, i| {
                let (a, b) = (vertices[i], vertices[(i + 1) % vertices.len()]);
                inside
                    ^ ((a.1 > py) != (b.1 > py)
                        && px < (b.0 - a.0) * (py - a.1) / (b.1 - a.1) + a.0)
            });
            if inside {
                fill_rect_alpha(img, f64::from(x), f64::from(y), 1.0, 1.0, color, alpha);
            }
        }
    }
}

/// Object outlines plus the nose cone, contact circle and whisker fans of one
/// frame, in pixels. An object whose nose rule or whisker fan fires gets a
/// thick outline and a translucent fill in its colour; active fields fill more
/// strongly than idle ones. Every stroke has a dark halo so it reads on grey.
fn draw_detection_fields(
    img: &mut RgbImage,
    frame_idx: usize,
    bundle: &LoadedInspection,
    state: &JobState,
    feature: &FeatureInspection,
) {
    let FeatureInspection::DetectionFields {
        nose_label,
        head_label,
        cone_half_angle_deg,
        nose_reach_m,
        contact_distance_m,
        whisker_apex_fraction,
        whisker_half_angle_deg,
        whisker_length_m,
        object_labels,
        nose_near_columns,
        nose_gate_columns,
        whisker_hit_columns,
        ..
    } = feature
    else {
        return;
    };
    let hit = |columns: &[String], index: usize| {
        columns
            .get(index)
            .and_then(|column| state.component_at(column, frame_idx))
            .unwrap_or(false)
    };
    let nose_hit = |index| hit(nose_near_columns, index) && hit(nose_gate_columns, index);
    let unit = state.layout.unit;
    let stroke = unit.div_ceil(2) as i32;
    let halo = parse_color(themekit::DARK.background);
    for (index, label) in object_labels.iter().enumerate() {
        let Some(perimeter) = state.perimeters.get(label) else {
            continue;
        };
        let color = object_color(label, index);
        let active = nose_hit(index) || hit(whisker_hit_columns, index);
        let width = if active { 2 * stroke + 1 } else { stroke };
        if active {
            fill_perimeter(img, perimeter, color, 0.25);
        }
        draw_perimeter(img, perimeter, halo, width + 2, 0.8);
        draw_perimeter(img, perimeter, color, width, 1.0);
    }
    let (Some(nose), Some(head)) = (
        state.coord_at(nose_label, frame_idx),
        state.coord_at(head_label, frame_idx),
    ) else {
        return;
    };
    let scale = frame_scale(&bundle.manifest, img.width(), img.height());
    let mpp = bundle.manifest.settings.meters_per_pixel;
    // ponytail: lengths use the x scale; non-square source pixels skew fields by y_aspect.
    let pixels = |metres: f64| metric_length_to_pixels(metres, mpp, scale.sx);
    let axis = (nose.1 - head.1).atan2(nose.0 - head.0).to_degrees();
    let ink = parse_color(themekit::DARK.ink);
    let alpha = |active: bool| if active { 0.45 } else { 0.12 };
    let wedge = |img: &mut RgbImage,
                 apex: (f64, f64),
                 length: f64,
                 centre: f64,
                 half: f64,
                 color,
                 active| {
        let (start, end) = (centre - half, centre + half);
        draw_wedge_alpha(
            img,
            apex.0,
            apex.1,
            length,
            start,
            end,
            halo,
            stroke + 2,
            0.8,
            0.0,
        );
        draw_wedge_alpha(
            img,
            apex.0,
            apex.1,
            length,
            start,
            end,
            color,
            stroke,
            1.0,
            alpha(active),
        );
    };

    let any_nose = (0..object_labels.len()).any(nose_hit);
    wedge(
        img,
        nose,
        pixels(*nose_reach_m),
        axis,
        *cone_half_angle_deg,
        parse_color(CONE_COLOR),
        any_nose,
    );
    let contact = pixels(*contact_distance_m);
    draw_circle_alpha(img, nose.0, nose.1, contact, halo, stroke + 2, 0.8);
    draw_circle_alpha(img, nose.0, nose.1, contact, ink, stroke, 1.0);

    if let (Some(fraction), Some(half), Some(length)) = (
        whisker_apex_fraction,
        whisker_half_angle_deg,
        whisker_length_m,
    ) {
        let apex = (
            head.0 + fraction * (nose.0 - head.0),
            head.1 + fraction * (nose.1 - head.1),
        );
        // The hit column's side is "left", "right" or "both"; a missing side lights both fans.
        let sides: Vec<Option<&str>> = (0..object_labels.len())
            .filter(|&index| hit(whisker_hit_columns, index))
            .map(|index| state.side_at(&whisker_hit_columns[index], frame_idx))
            .collect();
        for (side, turn, color) in [
            ("left", 90.0, FAN_COLORS[0]),
            ("right", -90.0, FAN_COLORS[1]),
        ] {
            let active = sides
                .iter()
                .any(|found| matches!(found, None | Some("both")) || *found == Some(side));
            wedge(
                img,
                apex,
                pixels(*length),
                axis + turn,
                *half,
                parse_color(color),
                active,
            );
        }
        // Whisker apex: a small ink dot on the head-to-nose axis.
        let radius = f64::from(unit) + 1.0;
        draw_circle_alpha(img, apex.0, apex.1, radius, halo, stroke + 2, 0.8);
        draw_circle_alpha(img, apex.0, apex.1, radius, ink, stroke, 1.0);
    }
    draw_line_alpha(img, head.0, head.1, nose.0, nose.1, halo, stroke + 2, 0.8);
    draw_line_alpha(img, head.0, head.1, nose.0, nose.1, ink, stroke, 1.0);
}
