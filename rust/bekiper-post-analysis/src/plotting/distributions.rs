//! Native descriptive distribution payloads; Python only draws these values.

use super::*;

fn quantile(sorted: &[f64], probability: f64) -> f64 {
    let index = probability * (sorted.len() - 1) as f64;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;
    sorted[lower] + (sorted[upper] - sorted[lower]) * (index - lower as f64)
}

pub(super) fn attach(panel: &mut Record) -> Result<()> {
    let points = panel["points"]
        .as_array()
        .context("distribution points")?
        .clone();
    let finite = points
        .iter()
        .filter_map(|p| p["value"].as_f64())
        .collect::<Vec<_>>();
    if finite.is_empty() {
        return Ok(());
    }
    let mut low = finite.iter().copied().fold(f64::INFINITY, f64::min);
    let mut high = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if low == high {
        let span = low.abs().max(1.0) * 0.05;
        low -= span;
        high += span;
    }
    let mut edges = (0..=12)
        .map(|i| low + (high - low) * i as f64 / 12.0)
        .collect::<Vec<_>>();
    if edges.windows(2).any(|w| w[0] >= w[1]) {
        // A few ULPs of spread cannot represent twelve separate bins. Expand
        // the drawing range; retain every recorded value and count unchanged.
        let span = low.abs().max(high.abs()).max(1.0) * 0.05;
        low -= span;
        high += span;
        edges = (0..=12)
            .map(|i| low + (high - low) * i as f64 / 12.0)
            .collect();
    }
    for summary in panel["summaries"]
        .as_array_mut()
        .context("distribution summaries")?
    {
        let members = points
            .iter()
            .filter(|p| p["category"] == summary["category"])
            .collect::<Vec<_>>();
        if members.iter().any(|p| p["censored"] == true) {
            summary["distribution"] = json!({"status":"unavailable:censored"});
            continue;
        }
        let mut values = members
            .iter()
            .filter_map(|p| p["value"].as_f64())
            .collect::<Vec<_>>();
        values.sort_by(f64::total_cmp);
        if values.is_empty() {
            continue;
        }
        let q1 = quantile(&values, 0.25);
        let median = quantile(&values, 0.5);
        let q3 = quantile(&values, 0.75);
        let iqr = q3 - q1;
        let whislo = values
            .iter()
            .copied()
            .find(|v| *v >= q1 - 1.5 * iqr)
            .unwrap_or(q1)
            .min(q1);
        let whishi = values
            .iter()
            .rev()
            .copied()
            .find(|v| *v <= q3 + 1.5 * iqr)
            .unwrap_or(q3)
            .max(q3);
        // Interpolated quartiles can extend beyond adjacent observations in
        // tied populations. A whisker must reach at least its own box edge.
        let mut counts = vec![0_usize; 12];
        for value in &values {
            let bin = (((value - low) / (high - low) * 12.0).floor() as usize).min(11);
            counts[bin] += 1;
        }
        let origin = values[0];
        let mean = origin + values.iter().map(|v| v - origin).sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
            / (values.len().saturating_sub(1).max(1)) as f64;
        let bandwidth = 1.06 * variance.sqrt() * (values.len() as f64).powf(-0.2);
        let density = if values.len() < 2 || bandwidth <= 0.0 || !bandwidth.is_finite() {
            json!({"status":"unavailable:insufficient_variation","x":[],"y":[]})
        } else {
            let min = values[0] - 3.0 * bandwidth;
            let max = values[values.len() - 1] + 3.0 * bandwidth;
            let x = (0..128)
                .map(|i| min + (max - min) * i as f64 / 127.0)
                .collect::<Vec<_>>();
            if x.windows(2).any(|w| w[0] >= w[1]) {
                json!({"status":"unavailable:insufficient_grid_resolution","x":[],"y":[]})
            } else {
                let norm = values.len() as f64 * bandwidth * (2.0 * std::f64::consts::PI).sqrt();
                let y = x
                    .iter()
                    .map(|x| {
                        values
                            .iter()
                            .map(|v| (-0.5 * ((x - v) / bandwidth).powi(2)).exp())
                            .sum::<f64>()
                            / norm
                    })
                    .collect::<Vec<_>>();
                json!({"status":"available","x":x,"y":y,"bandwidth":bandwidth,"method":"Gaussian KDE; normal-reference bandwidth 1.06 SD n^(-1/5); 128 points; ±3 bandwidths"})
            }
        };
        summary["distribution"] = json!({"status":"available","quantile_method":"linear interpolation (n-1)p",
            "box":{"q1":q1,"med":median,"q3":q3,"whislo":whislo,"whishi":whishi,"fliers":values.iter().filter(|v| **v<whislo || **v>whishi).copied().collect::<Vec<_>>()},
            "histogram":{"edges":edges,"counts":counts},"density":density});
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tied_and_sparse_populations_have_ordered_tukey_whiskers() {
        // Independent Tukey-box expectations, including interpolation between
        // tied observations where an adjacent value lies inside the box.
        for (values, expected_box, fliers, has_density) in [
            (
                vec![0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.25, 0.25],
                vec![1.0],
                true,
            ),
            (
                vec![0.0, 1.0, 1.0, 1.0],
                [0.75, 0.75, 1.0, 1.0, 1.0],
                vec![0.0],
                true,
            ),
            (vec![2.0], [2.0; 5], vec![], false),
            (vec![0.0; 4], [0.0; 5], vec![], false),
            (vec![0.0, 0.0, 0.0, 0.0, 20.0], [0.0; 5], vec![20.0], true),
        ] {
            let mut panel = json!({
                "points": values.iter().map(|value| json!({"category":"a", "value":value})).collect::<Vec<_>>(),
                "summaries": [{"category":"a"}]
            });
            attach(&mut panel).unwrap();
            let distribution = &panel["summaries"][0]["distribution"];
            let actual = ["whislo", "q1", "med", "q3", "whishi"]
                .map(|key| distribution["box"][key].as_f64().unwrap());
            assert_eq!(actual, expected_box, "population {values:?}");
            assert_eq!(distribution["box"]["fliers"], json!(fliers));
            assert_eq!(
                distribution["histogram"]["counts"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|count| count.as_u64().unwrap())
                    .sum::<u64>(),
                values.len() as u64
            );
            assert_eq!(
                distribution["density"]["status"] == "available",
                has_density
            );
        }
    }

    #[test]
    fn native_quantiles_counts_and_density_preserve_population() {
        let mut panel = json!({"points":(1..=5).map(|v| json!({"category":"a","value":v,"censored":false})).collect::<Vec<_>>(),
                               "summaries":[{"category":"a"}]});
        attach(&mut panel).unwrap();
        let d = &panel["summaries"][0]["distribution"];
        assert_eq!(d["box"]["q1"], 2.0);
        assert_eq!(d["box"]["med"], 3.0);
        assert_eq!(d["box"]["q3"], 4.0);
        assert_eq!(
            d["histogram"]["counts"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap())
                .sum::<u64>(),
            5
        );
        let x = d["density"]["x"].as_array().unwrap();
        let y = d["density"]["y"].as_array().unwrap();
        let area: f64 = (1..x.len())
            .map(|i| {
                (x[i].as_f64().unwrap() - x[i - 1].as_f64().unwrap())
                    * (y[i].as_f64().unwrap() + y[i - 1].as_f64().unwrap())
                    / 2.0
            })
            .sum();
        assert!((area - 1.0).abs() < 0.005);
    }

    #[test]
    fn nonzero_constants_and_few_ulp_spreads_do_not_emit_unordered_grids() {
        for (values, expected_status) in [
            (
                vec![19.166666666666664; 6],
                "unavailable:insufficient_variation",
            ),
            (
                vec![19.0, 19.0_f64.next_up()],
                "unavailable:insufficient_grid_resolution",
            ),
        ] {
            let mut panel = json!({"points":values.iter().map(|value| json!({"category":"a","value":value})).collect::<Vec<_>>(),"summaries":[{"category":"a"}]});
            attach(&mut panel).unwrap();
            let distribution = &panel["summaries"][0]["distribution"];
            assert_eq!(distribution["density"]["status"], expected_status);
            assert!(distribution["density"]["x"].as_array().unwrap().is_empty());
            let edges = distribution["histogram"]["edges"].as_array().unwrap();
            assert!(
                edges
                    .windows(2)
                    .all(|w| w[0].as_f64().unwrap() < w[1].as_f64().unwrap())
            );
            assert_eq!(
                distribution["histogram"]["counts"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap())
                    .sum::<u64>(),
                values.len() as u64
            );
            assert_eq!(
                panel["points"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|p| p["value"].as_f64().unwrap())
                    .collect::<Vec<_>>(),
                values
            );
        }
    }

    #[test]
    fn categories_without_numeric_members_get_no_distribution() {
        let mut panel = json!({"points":[{"category":"a","value":1.0},{"category":"a","value":2.0},
            {"category":"b","value":null}],"summaries":[{"category":"a"},{"category":"b"}]});
        attach(&mut panel).unwrap();
        assert!(panel["summaries"][0]["distribution"].is_object());
        assert!(panel["summaries"][1].get("distribution").is_none());
    }
}
