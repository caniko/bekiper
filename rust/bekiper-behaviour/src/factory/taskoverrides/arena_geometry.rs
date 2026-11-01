/// Extract (cx, cy, w, h) from a perimeter matching *target_label* (Rectangle or Polygon bounding box).
fn extract_rect_dims(
    perimeters: &[PerimeterSpec],
    target_label: &str,
) -> Option<(f64, f64, f64, f64)> {
    perimeters.iter().find_map(|p| match p {
        PerimeterSpec::Rectangle {
            label,
            center_x,
            center_y,
            width,
            height,
        } if label == target_label => Some((*center_x, *center_y, *width, *height)),
        PerimeterSpec::Polygon { label, vertices } if label == target_label => {
            let xs: Vec<f64> = vertices.iter().map(|(x, _)| *x).collect();
            let ys: Vec<f64> = vertices.iter().map(|(_, y)| *y).collect();
            let min_x = xs.iter().cloned().reduce(f64::min)?;
            let max_x = xs.iter().cloned().reduce(f64::max)?;
            let min_y = ys.iter().cloned().reduce(f64::min)?;
            let max_y = ys.iter().cloned().reduce(f64::max)?;
            Some((
                (min_x + max_x) / 2.0,
                (min_y + max_y) / 2.0,
                max_x - min_x,
                max_y - min_y,
            ))
        }
        _ => None,
    })
}

/// Extract (cx, cy, w, h) from a perimeter labeled "arena".
fn extract_arena_dims(perimeters: &[PerimeterSpec]) -> Option<(f64, f64, f64, f64)> {
    extract_rect_dims(perimeters, "arena")
}

fn label_matches_any(label: &str, needles: &[&str]) -> bool {
    let normalized = label.to_ascii_lowercase();
    needles.iter().any(|needle| normalized.contains(needle))
}

fn is_start_label(label: &str) -> bool {
    label_matches_any(
        label,
        &["start_area", "start_perimeter", "trial_start", "start"],
    )
}

fn is_enclosure_label(label: &str) -> bool {
    label_matches_any(label, &["enclosure", "board", "arena", "cheeseboard"])
}

fn is_reward_label(label: &str) -> bool {
    label_matches_any(label, &["reward"])
}

fn circle_perimeter(spec: &PerimeterSpec) -> Option<Perimeter<Circle>> {
    match spec {
        PerimeterSpec::Circle {
            label,
            center_x,
            center_y,
            radius,
        } => Some(Perimeter::new(
            label.clone(),
            Circle::new(*center_x, *center_y, *radius),
        )),
        _ => None,
    }
}
