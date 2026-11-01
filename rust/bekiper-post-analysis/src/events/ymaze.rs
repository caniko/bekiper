//! Continuous same-arm landmark event reconstruction.
use super::*;

/// Historical zero-dwell compatibility adapter. Current scoring resolves its
/// policy explicitly and calls `ymaze_coordinate_entries_with_policy` with the
/// final sample boundary required by positive dwell.
pub fn ymaze_coordinate_entries(
    samples: &[Record],
    geometry: &Record,
    threshold_pct: u8,
) -> Result<Record> {
    ymaze_coordinate_entries_with_policy(
        samples,
        geometry,
        threshold_pct,
        &Default::default(),
        None,
    )
}

/// One policy and timing adapter shared with native task analysis.
pub fn ymaze_coordinate_entries_with_policy(
    samples: &[Record],
    geometry: &Record,
    threshold_pct: u8,
    policy: &bekiper_feature::tolerance::TemporalPolicy,
    sample_end_s: Option<f64>,
) -> Result<Record> {
    ymaze_coordinate_entries_with_zone_gap(
        samples,
        geometry,
        threshold_pct,
        policy,
        sample_end_s,
        0.0,
    )
}

/// As above, bridging up to `zone_max_gap_s` of no-arm frames between visits
/// to the same arm (cross-line geometry only), like native `extract_zone_sequence`.
pub fn ymaze_coordinate_entries_with_zone_gap(
    samples: &[Record],
    geometry: &Record,
    threshold_pct: u8,
    policy: &bekiper_feature::tolerance::TemporalPolicy,
    sample_end_s: Option<f64>,
    zone_max_gap_s: f64,
) -> Result<Record> {
    use bekiper_feature::tolerance::BoutTiming;
    let crossline = geometry["version"] == GEOMETRY_ARM_START;
    // The archived radial rule requires uninterrupted membership; the
    // cross-line contexts carry their own registered policy (gap allowed).
    if crossline {
        policy.validate()
    } else {
        policy.validate_uninterrupted()
    }
    .map_err(anyhow::Error::msg)?;
    ensure!(
        (geometry["version"] == GEOMETRY || crossline) && geometry["canonical_vertices"] == true,
        "Y-maze geometry needs explicit canonical imen_radial_vertex_span_v1 or imen_arm_start_crossline_v1 vertices"
    );
    ensure!(
        if crossline {
            [0, 25, 45, 65].contains(&threshold_pct)
        } else {
            [25, 45, 65].contains(&threshold_pct)
        },
        "Y-maze requires threshold 25/45/65 (cross-line geometry also 0)"
    );
    let version = if crossline {
        GEOMETRY_ARM_START
    } else {
        GEOMETRY
    };
    let arms = geometry["arms"]
        .as_object()
        .context("Y-maze requires three explicitly labelled arms")?;
    ensure!(
        arms.keys().map(String::as_str).collect::<BTreeSet<_>>()
            == ["A", "B", "C"].into_iter().collect(),
        "Y-maze requires three explicitly labelled arms"
    );
    let center = polygon(&geometry["center"])?;
    let cx = center.iter().map(|v| v[0]).sum::<f64>() / center.len() as f64;
    let cy = center.iter().map(|v| v[1]).sum::<f64>() / center.len() as f64;
    ensure!(cx.is_finite() && cy.is_finite(), "Invalid geometry center");
    let mut polygons = BTreeMap::new();
    let mut radii = BTreeMap::new();
    let mut lines = BTreeMap::new();
    let maze = bekiper_perimeter::radial_maze::RadialMaze::new(
        bekiper_perimeter::polygon::Polygon::new(center.iter().map(|v| (v[0], v[1])).collect()),
        ["A", "B", "C"]
            .into_iter()
            .map(|arm| {
                Ok(bekiper_perimeter::polygon::Polygon::new(
                    polygon(&arms[arm])?.iter().map(|v| (v[0], v[1])).collect(),
                ))
            })
            .collect::<Result<_>>()?,
    );
    for (i, arm) in ["A", "B", "C"].into_iter().enumerate() {
        let points = polygon(&arms[arm])?;
        if crossline {
            // Same predicate as the native task: bekiper-perimeter owns the side test.
            let line = maze
                .arm_frame(i)
                .map_err(anyhow::Error::msg)?
                .crossline(f64::from(threshold_pct) / 100.0)
                .map_err(anyhow::Error::msg)?;
            lines.insert(arm, line);
            polygons.insert(arm, points);
            continue;
        }
        let distances: Vec<_> = points
            .iter()
            .map(|v| (v[0] - cx).hypot(v[1] - cy))
            .collect();
        let min = distances.iter().copied().fold(f64::INFINITY, f64::min);
        let max = distances.iter().copied().fold(0.0, f64::max);
        let radius = min + f64::from(threshold_pct) / 100.0 * (max - min);
        ensure!(
            radius.is_finite() && (radius * radius).is_finite(),
            "Invalid geometry radius"
        );
        radii.insert(arm, radius);
        polygons.insert(arm, points);
    }
    let mut sequence = Vec::new();
    let mut debug = Vec::new();
    let mut previous = [false; 3];
    let mut masks = [Vec::new(), Vec::new(), Vec::new()];
    let mut last_time = f64::NEG_INFINITY;
    for (index, sample) in samples.iter().enumerate() {
        let time = nonnegative(&sample["timestamp_s"], "timestamp_s", None)?;
        ensure!(time > last_time, "Nonmonotonic timestamps");
        last_time = time;
        let mut coordinates = Vec::new();
        let mut present = true;
        for name in [
            "head_center_x",
            "head_center_y",
            "tail_base_x",
            "tail_base_y",
        ] {
            if sample[name].is_null() {
                present = false;
            } else {
                coordinates.push(finite(&sample[name], name)?);
            }
        }
        let mut states = [false; 3];
        for (i, arm) in ["A", "B", "C"].into_iter().enumerate() {
            let (head, tail) = if present {
                let [hx, hy, tx, ty] = [
                    coordinates[0],
                    coordinates[1],
                    coordinates[2],
                    coordinates[3],
                ];
                let inside = |x: f64, y: f64| {
                    polygon_contains(x, y, &polygons[arm])
                        && match lines.get(arm) {
                            Some(line) => line.is_past(x, y),
                            None => (x - cx).powi(2) + (y - cy).powi(2) >= radii[arm].powi(2),
                        }
                };
                (inside(hx, hy), inside(tx, ty))
            } else {
                (false, false)
            };
            states[i] = head && tail;
            masks[i].push(states[i]);
            let entry = states[i] && !previous[i];
            debug.push(json!({"sample_index":index,"timestamp_s":time,"arm":arm,"head_past":head,"tail_past":tail,
                "predicate":states[i],"entry":entry,"reason":if present {None} else {Some("tracking_missing")},
                "threshold_pct":threshold_pct,"geometry_version":version,"entry_rule":ENTRY_RULE}));
            if entry {
                sequence.push(arm);
            }
        }
        ensure!(
            states.iter().filter(|&&state| state).count() <= 1,
            "Overlapping arm geometry: simultaneous arm membership"
        );
        previous = states;
    }
    ensure!(
        zone_max_gap_s.is_finite() && zone_max_gap_s >= 0.0,
        "Invalid zone_max_gap_seconds"
    );
    if crossline && zone_max_gap_s > 0.0 && samples.len() > 1 {
        // Native frame rounding: round(gap * fps) with fps from the sampling span.
        let span = last_time - nonnegative(&samples[0]["timestamp_s"], "timestamp_s", None)?;
        let fps = (samples.len() - 1) as f64 / span;
        let max_gap = (zone_max_gap_s * fps).round() as usize;
        let n = samples.len();
        let mut last: Option<(usize, usize)> = None; // (arm, last occupied frame)
        for f in 0..n {
            let Some(i) = (0..3).find(|&i| masks[i][f]) else {
                continue;
            };
            if let Some((arm, at)) = last
                && arm == i
                && f - at - 1 <= max_gap
            {
                masks[i][at + 1..f].fill(true);
            }
            last = Some((i, f));
        }
    }
    let timing = if policy.min_duration_seconds > 0.0 {
        let timestamps = samples
            .iter()
            .map(|s| s["timestamp_s"].as_f64().context("missing timestamp"))
            .collect::<Result<Vec<_>>>()?;
        BoutTiming::timestamps(&timestamps, sample_end_s.context("Positive Y-maze dwell requires explicit coordinate_sample_end_s (exclusive final sample boundary)")?).map_err(anyhow::Error::msg)?
    } else {
        BoutTiming::cfr(1.0, samples.len()).map_err(anyhow::Error::msg)?
    };
    let mut accepted = Vec::new();
    let mut bouts = Vec::new();
    let mut removed = 0;
    for (i, arm) in ["A", "B", "C"].into_iter().enumerate() {
        let selected = policy
            .filter(&masks[i], &timing)
            .map_err(anyhow::Error::msg)?;
        for bout in selected.bouts {
            if bout.accepted {
                accepted.push((bout.start, arm));
            } else {
                removed += 1;
            }
            let event = &mut debug[bout.start * 3 + i];
            event["raw_entry"] = json!(true);
            event["entry"] = json!(bout.accepted);
            if !bout.accepted {
                event["reason"] = "minimum_dwell".into();
            }
            event["dwell_duration_s"] = if policy.min_duration_seconds > 0.0 {
                json!(bout.duration)
            } else {
                Value::Null
            };
            event["qualification_timestamp_s"] = if policy.min_duration_seconds > 0.0 {
                bout.qualification_index
                    .map(|j| json!(timing.boundary(j + 1)))
                    .unwrap_or(Value::Null)
            } else {
                samples[bout.start]["timestamp_s"].clone()
            };
            bouts.push(json!({"arm":arm,"sample_start":bout.start,"sample_end_exclusive":bout.end,
                "crossing_timestamp_s":samples[bout.start]["timestamp_s"],"duration_s":event["dwell_duration_s"],
                "accepted":bout.accepted,"rejection_reason":event["reason"],"qualification_timestamp_s":event["qualification_timestamp_s"]}));
        }
    }
    accepted.sort_by_key(|(index, arm)| (*index, *arm));
    let mut result = json!({"valid_arm_entry_sequence":accepted.iter().map(|(_, arm)| arm).collect::<Vec<_>>(),
        "raw_arm_entry_sequence":sequence,"removed_short_entries":removed,
        "raw_to_scored_entry_difference":sequence.len() as i64 - accepted.len() as i64,
        "dwell_rejected_entries":removed,"dwell_bouts":bouts,
        "temporal_policy":policy,"event_debug":debug,"threshold_radii":radii,
        "geometry_version":version,"entry_rule":ENTRY_RULE});
    if crossline {
        result["crossline_endpoints"] = json!(
            lines
                .iter()
                .map(|(arm, l)| (*arm, json!({"p1": l.p1, "p2": l.p2})))
                .collect::<BTreeMap<_, _>>()
        );
    }
    Ok(result)
}
