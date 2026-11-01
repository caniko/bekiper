fn bool_column(df: &DataFrame, name: &str) -> Option<BooleanChunked> {
    df.column(name).ok()?.bool().ok().cloned()
}

fn count_true_column(df: &DataFrame, name: &str) -> u64 {
    bool_column(df, name)
        .and_then(|mask| mask.sum())
        .map(u64::from)
        .unwrap_or(0)
}

fn count_pose_ok(df: &DataFrame, coords: &CoordinateColumns) -> u64 {
    let Ok(xs) = df.column(&coords.x).and_then(|series| series.f64()) else {
        return 0;
    };
    let Ok(ys) = df.column(&coords.y).and_then(|series| series.f64()) else {
        return 0;
    };
    xs.into_iter()
        .zip(ys)
        .filter(|(x, y)| x.is_some() && y.is_some())
        .count() as u64
}

fn bout_lengths_ms(mask: &BooleanChunked, timing: &BoutTiming) -> Vec<u64> {
    true_ranges(mask)
        .into_iter()
        .take(100)
        .map(|(start, end)| (timing.duration(start, end) * 1000.0).round() as u64)
        .collect()
}
