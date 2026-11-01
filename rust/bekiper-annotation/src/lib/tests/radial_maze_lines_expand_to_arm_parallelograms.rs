#[test]
fn radial_maze_lines_expand_to_arm_parallelograms() {
    let dir = make_temp_dir();
    std::fs::write(
            dir.join("arm-lines.csv"),
            "a,151,161,173,122,img.png,640,480\nb,462,156,438,119,img.png,640,480\nc,294,397,337,397,img.png,640,480\n",
        )
        .unwrap();
    std::fs::write(
            dir.join("center-triangle.json"),
            r#"{"annotations": [{"category_id": 1, "segmentation": [[288,237,309,196,335,236]], "bbox": [288,196,47,41]}], "categories": [{"id": 1, "name": "center"}]}"#,
        )
        .unwrap();

    let specs = parse_radial_maze(&dir).unwrap();
    match &specs[0] {
        PerimeterSpec::RadialMaze {
            center_vertices,
            arms,
            ..
        } => {
            assert_eq!(arms.len(), 3);
            assert!(arms.iter().all(|arm| arm.len() == 4));

            // a matches the similarly angled left center edge.
            assert_has_point(&arms[0], (288.0, 237.0));
            assert_has_point(&arms[0], (309.0, 196.0));
            assert_has_point(&arms[0], (151.0, 161.0));
            assert_has_point(&arms[0], (173.0, 122.0));

            // b matches the right center edge.
            assert_has_point(&arms[1], (309.0, 196.0));
            assert_has_point(&arms[1], (335.0, 236.0));
            assert_has_point(&arms[1], (462.0, 156.0));
            assert_has_point(&arms[1], (438.0, 119.0));

            // c matches the bottom center edge.
            assert_has_point(&arms[2], (288.0, 237.0));
            assert_has_point(&arms[2], (335.0, 236.0));
            assert_has_point(&arms[2], (294.0, 397.0));
            assert_has_point(&arms[2], (337.0, 397.0));

            assert_eq!(center_vertices.len(), 3);
        }
        other => panic!("expected RadialMaze, got {:?}", other),
    }
}

#[test]
fn parse_rectangle_csv_with_header() {
    let dir = make_temp_dir();
    let csv_path = &dir.join("test_with_header.csv");
    std::fs::write(
            csv_path,
            "label_name,bbox_x,bbox_y,bbox_width,bbox_height,image_name,image_width,image_height\nfamiliar,48,49,25,38,100-video.png,224,224\nnovel,152,150,28,27,100-video.png,224,224\n",
        )
        .unwrap();

    let specs = parse_rectangle_csv(csv_path).unwrap();
    assert_eq!(specs.len(), 2);
    match &specs[0] {
        PerimeterSpec::Rectangle { label, .. } => {
            assert_eq!(label.as_str(), "familiar");
        }
        other => panic!("expected Rectangle, got {:?}", other),
    }
}

#[test]
fn scale_calibration_from_plugin_dir() {
    let dir = make_temp_dir();
    let plugin_dir = &dir.join("plugin_files");
    std::fs::create_dir(plugin_dir).unwrap();
    std::fs::write(
        plugin_dir.join("meters_per_pixel-diagonal-0.565685.csv"),
        "mpp,8,218,215,7,A2_training_3.png,224,224\n",
    )
    .unwrap();

    let cals = load_scale_calibrations(plugin_dir).unwrap();
    assert_eq!(cals.len(), 1);
    assert_eq!(cals[0].method, "diagonal");
    assert!((cals[0].physical_meters - 0.565685).abs() < 1e-6);
    // pixel distance = sqrt((215-8)^2 + (7-218)^2) ≈ 295.58
    // mpp ≈ 0.565685 / 295.58 ≈ 0.001914
    assert!((cals[0].meters_per_pixel.0 - 0.001914).abs() < 1e-4);
}

#[test]
fn parse_circle_csv_training_two_objects() {
    let dir = make_temp_dir();
    let csv_path = dir.join("perimeter-circle-1033.csv");
    // Real data from examples/neo/dejana/dlc-data/1033/0.perimeter-circle-1033.csv
    // Each row: index, center_x, center_y, edge_x, edge_y, image, w, h
    std::fs::write(
        &csv_path,
        "2,682,297,722,296,0.video-1033_0-image.jpeg,1062,964\n\
             1,688,639,733,639,0.video-1033_0-image.jpeg,1062,964\n",
    )
    .unwrap();

    let specs = parse_circle_csv(&csv_path).unwrap();
    assert_eq!(specs.len(), 2);

    // First circle: index 2, center=(682, 297), edge=(722, 296), r=40.01
    match &specs[0] {
        PerimeterSpec::Circle {
            label,
            center_x,
            center_y,
            radius,
        } => {
            assert_eq!(label.as_str(), "object_2");
            assert!((center_x - 682.0).abs() < 0.01);
            assert!((center_y - 297.0).abs() < 0.01);
            assert!((*radius - 40.01).abs() < 0.1);
        }
        other => panic!("expected Circle, got {:?}", other),
    }

    // Second circle: index 1, center=(688, 639), edge=(733, 639), r=45
    match &specs[1] {
        PerimeterSpec::Circle {
            label,
            center_x,
            center_y,
            radius,
        } => {
            assert_eq!(label.as_str(), "object_1");
            assert!((center_x - 688.0).abs() < 0.01);
            assert!((center_y - 639.0).abs() < 0.01);
            assert!((*radius - 45.0).abs() < 0.01);
        }
        other => panic!("expected Circle, got {:?}", other),
    }
}

#[test]
fn parse_circle_csv_test_four_objects() {
    let dir = make_temp_dir();
    let csv_path = dir.join("perimeter-circle-1033.csv");
    // Test phase: 4 rows = 4 objects
    std::fs::write(
        &csv_path,
        "1,719,654,755,653,4.video-1033_0-image.jpeg,1062,964\n\
             2,716,288,757,285,4.video-1033_0-image.jpeg,1062,964\n\
             3,341,281,300,280,4.video-1033_0-image.jpeg,1062,964\n\
             4,344,665,308,684,4.video-1033_0-image.jpeg,1062,964\n",
    )
    .unwrap();

    let specs = parse_circle_csv(&csv_path).unwrap();
    assert_eq!(specs.len(), 4);
    assert_eq!(specs[0].label().as_str(), "object_1");
    assert_eq!(specs[1].label().as_str(), "object_2");
    assert_eq!(specs[2].label().as_str(), "object_3");
    assert_eq!(specs[3].label().as_str(), "object_4");
}

#[test]
fn parse_center_csv_single_row() {
    let dir = make_temp_dir();
    let csv_path = dir.join("center-1033_0.csv");
    std::fs::write(
        &csv_path,
        "center,531,466,0.video-1033_0-image.jpeg,1062,964\n",
    )
    .unwrap();

    let center = parse_center_csv(&csv_path).unwrap();
    assert!((center.x - 531.0).abs() < 0.01);
    assert!((center.y - 466.0).abs() < 0.01);
}

#[test]
fn load_perimeter_by_id_finds_circle() {
    let dir = make_temp_dir();
    let plugin_dir = dir.join("plugin_files");
    std::fs::create_dir(&plugin_dir).unwrap();
    std::fs::write(
        plugin_dir.join("perimeter-circle-test_id.csv"),
        "1,100,200,140,200,img.jpeg,640,480\n\
             2,120,180,120,220,img.jpeg,640,480\n",
    )
    .unwrap();

    let specs = load_perimeter_by_id(&plugin_dir, "test_id").unwrap();
    assert_eq!(specs.len(), 2); // 2 rows = 2 circles
    assert!(matches!(&specs[0], PerimeterSpec::Circle { .. }));
    assert!(matches!(&specs[1], PerimeterSpec::Circle { .. }));
}

#[test]
fn load_perimeter_by_id_finds_circle_point() {
    let dir = make_temp_dir();
    let plugin_dir = dir.join("plugin_files");
    std::fs::create_dir(&plugin_dir).unwrap();
    std::fs::write(
        plugin_dir.join("perimeter-circle_point-test_id.csv"),
        "nose,100,200,img.jpeg,640,480\npaw,120,180,img.jpeg,640,480\n",
    )
    .unwrap();

    let specs = load_perimeter_by_id(&plugin_dir, "test_id").unwrap();
    assert_eq!(specs.len(), 2);
    match &specs[0] {
        PerimeterSpec::Circle {
            label,
            center_x,
            center_y,
            radius,
        } => {
            assert_eq!(label.as_str(), "nose");
            assert!((center_x - 100.0).abs() < 0.01);
            assert!((center_y - 200.0).abs() < 0.01);
            assert!((*radius - 0.0).abs() < 0.01);
        }
        other => panic!("expected Circle, got {:?}", other),
    }
}
