fn build_cheeseboard(
    perimeters: &[PerimeterSpec],
    params: TaskParams,
    trial_start_perimeter: Option<&str>,
) -> Result<Cheeseboard> {
    let _ = params; // Cheeseboard doesn't use TaskParams yet (heuristics not implemented)

    let start_area = trial_start_perimeter
        .and_then(|configured| {
            perimeters
                .iter()
                .find(|spec| spec.label().eq_ignore_ascii_case(configured))
                .cloned()
        })
        .or_else(|| {
            perimeters
                .iter()
                .find(|spec| is_start_label(spec.label()))
                .cloned()
        });

    let circles: Vec<Perimeter<Circle>> = perimeters.iter().filter_map(circle_perimeter).collect();
    if circles.is_empty() {
        return Err(BekiperError::Config(
            "cheeseboard task requires at least one Circle perimeter".into(),
        ));
    }

    let enclosure = perimeters
        .iter()
        .find(|spec| is_enclosure_label(spec.label()))
        .and_then(circle_perimeter)
        .or_else(|| {
            circles
                .iter()
                .filter(|perimeter| {
                    !start_area
                        .as_ref()
                        .map(|spec| spec.label() == &perimeter.label)
                        .unwrap_or(false)
                })
                .max_by(|a, b| {
                    a.shape
                        .radius
                        .partial_cmp(&b.shape.radius)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .cloned()
        })
        .unwrap_or_else(|| circles[0].clone());

    let rewards: Vec<Perimeter<Circle>> = {
        let explicit_rewards: Vec<Perimeter<Circle>> = perimeters
            .iter()
            .filter(|spec| is_reward_label(spec.label()))
            .filter_map(circle_perimeter)
            .filter(|perimeter| {
                perimeter.label != enclosure.label
                    && !start_area
                        .as_ref()
                        .map(|spec| spec.label() == &perimeter.label)
                        .unwrap_or(false)
            })
            .collect();

        if explicit_rewards.is_empty() {
            circles
                .iter()
                .filter(|perimeter| {
                    perimeter.label != enclosure.label
                        && !start_area
                            .as_ref()
                            .map(|spec| spec.label() == &perimeter.label)
                            .unwrap_or(false)
                })
                .cloned()
                .collect()
        } else {
            explicit_rewards
        }
    };

    Ok(Cheeseboard {
        start_area,
        enclosure,
        reward_locations: rewards,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_rect_dims_finds_arena() {
        let perimeters = vec![PerimeterSpec::Rectangle {
            label: "arena".into(),
            center_x: 0.5,
            center_y: 0.5,
            width: 1.0,
            height: 0.8,
        }];
        let dims = extract_arena_dims(&perimeters);
        assert_eq!(dims, Some((0.5, 0.5, 1.0, 0.8)));
    }

    #[test]
    fn extract_rect_dims_finds_center_zone() {
        let perimeters = vec![PerimeterSpec::Rectangle {
            label: "center_zone".into(),
            center_x: 0.4,
            center_y: 0.3,
            width: 0.35,
            height: 0.35,
        }];
        // "center_zone" is not found by extract_arena_dims
        assert!(extract_arena_dims(&perimeters).is_none());
        // but is found by extract_rect_dims with the right label
        let dims = extract_rect_dims(&perimeters, "center_zone");
        assert_eq!(dims, Some((0.4, 0.3, 0.35, 0.35)));
    }

    #[test]
    fn extract_rect_dims_polygon_bounding_box() {
        let perimeters = vec![PerimeterSpec::Polygon {
            label: "arena".into(),
            vertices: vec![(0.0, 0.0), (1.0, 0.0), (1.0, 0.8), (0.0, 0.8)],
        }];
        let dims = extract_arena_dims(&perimeters).unwrap();
        assert!((dims.0 - 0.5).abs() < 1e-9); // cx
        assert!((dims.1 - 0.4).abs() < 1e-9); // cy
        assert!((dims.2 - 1.0).abs() < 1e-9); // w
        assert!((dims.3 - 0.8).abs() < 1e-9); // h
    }

    #[test]
    fn build_cheeseboard_prefers_explicit_labels() {
        let perimeters = vec![
            PerimeterSpec::Rectangle {
                label: "start_area".into(),
                center_x: 5.0,
                center_y: 0.0,
                width: 1.0,
                height: 1.0,
            },
            PerimeterSpec::Circle {
                label: "enclosure".into(),
                center_x: 0.0,
                center_y: 0.0,
                radius: 5.0,
            },
            PerimeterSpec::Circle {
                label: "reward_a".into(),
                center_x: 3.0,
                center_y: 0.0,
                radius: 0.5,
            },
            PerimeterSpec::Circle {
                label: "reward_b".into(),
                center_x: -3.0,
                center_y: 0.0,
                radius: 0.5,
            },
        ];

        let task = build_cheeseboard(&perimeters, TaskParams::new(30.0), None)
            .expect("explicit cheeseboard labels should build");

        assert_eq!(
            task.start_area.as_ref().map(|spec| spec.label().as_str()),
            Some("start_area")
        );
        assert_eq!(task.enclosure.label, "enclosure");
        assert_eq!(
            task.reward_locations
                .iter()
                .map(|p| p.label.as_str())
                .collect::<Vec<_>>(),
            vec!["reward_a", "reward_b"]
        );
    }
}
