fn pair_at(coords: &LabelCoordinates, row_index: usize) -> Option<(f64, f64)> {
    match (coords.xs[row_index], coords.ys[row_index]) {
        (Some(x), Some(y)) => Some((x, y)),
        _ => None,
    }
}

fn euclidean_distance((x1, y1): (f64, f64), (x2, y2): (f64, f64)) -> f64 {
    let dx = x2 - x1;
    let dy = y2 - y1;
    (dx * dx + dy * dy).sqrt()
}

fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        Some((sorted[mid - 1] + sorted[mid]) / 2.0)
    } else {
        Some(sorted[mid])
    }
}

fn median_absolute_deviation(values: &[f64], median_value: f64) -> Option<f64> {
    let deviations: Vec<f64> = values
        .iter()
        .map(|value| (value - median_value).abs())
        .collect();
    median(&deviations)
}
