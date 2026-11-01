fn insert_zone_columns<'a, F, G>(
    row: &mut IndexMap<String, CellValue>,
    prefix: &str,
    entries_column: &str,
    zone_key: &str,
    find_zone: &F,
    find_motion: &G,
) where
    F: Fn(&str) -> Option<&'a Value>,
    G: Fn(&str) -> Option<&'a Value>,
{
    let null = Value::Null;
    let zone = find_zone(zone_key).unwrap_or(&null);
    row.insert(entries_column.into(), pluck(zone, "entries"));
    row.insert(
        format!("{prefix}_duration_s"),
        pluck(zone, "duration_seconds"),
    );
    row.insert(
        format!("{prefix}_distance_m"),
        pluck(zone, "distance_meters"),
    );
    row.insert(format!("{prefix}_mean_speed"), pluck(zone, "mean_speed"));
    row.insert(
        format!("{prefix}_latency_s"),
        pluck(zone, "latency_seconds"),
    );

    let motion = find_motion(zone_key).unwrap_or(&null);
    row.insert(
        format!("{prefix}_motion_distance_m"),
        pluck(motion, "total_displacement_meters"),
    );
    row.insert(
        format!("{prefix}_motion_median_speed"),
        pluck(motion, "median_speed_mps"),
    );
    row.insert(
        format!("{prefix}_motion_median_acceleration"),
        pluck(motion, "median_acceleration_mps2"),
    );
    row.insert(
        format!("{prefix}_motion_freezing_s"),
        pluck(motion, "freezing_seconds"),
    );
}
