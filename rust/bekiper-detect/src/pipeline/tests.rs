use super::*;

fn make_test_image() -> GrayImage {
    // 100x100 image: bright floor (~200), two small dark squares (~80), one large dark blob (~60, "mouse")
    let mut img = GrayImage::from_pixel(100, 100, image::Luma([200]));

    // Object 1: 8x8 square at (15, 15) — area 64
    for y in 15..23 {
        for x in 15..23 {
            img.put_pixel(x, y, image::Luma([80]));
        }
    }

    // Object 2: 8x8 square at (75, 75) — area 64
    for y in 75..83 {
        for x in 75..83 {
            img.put_pixel(x, y, image::Luma([80]));
        }
    }

    // Mouse: 20x20 blob at (50, 50) — area 400
    for y in 40..60 {
        for x in 40..60 {
            img.put_pixel(x, y, image::Luma([60]));
        }
    }

    img
}

#[test]
fn detect_two_objects_in_synthetic_image() {
    let img = make_test_image();
    let config = DetectionPipelineConfig {
        expected_count: 2,
        threshold: crate::config::ThresholdConfig::default(),
        mouse_removal: crate::config::MouseRemovalConfig {
            min_area_frac: 0.02,
            dilate_frac: 0.03,
        },
        area_filter: crate::config::AreaFilterConfig {
            min_frac: 0.003,
            max_frac: 0.02,
        },
        label_strategy: crate::config::LabelStrategy::Diagonal {
            labels: vec!["a".into(), "b".into()],
        },
    };

    let output = detect_dark_objects(&img, &config);
    assert_eq!(output.blobs.len(), 2, "should detect exactly 2 objects");

    // Both should have area ~64
    for blob in &output.blobs {
        assert!(blob.area >= 50 && blob.area <= 80, "area={}", blob.area);
    }
}

#[test]
fn percentile_works() {
    let img = GrayImage::from_fn(10, 10, |x, _| {
        image::Luma([x as u8 * 10]) // 0, 10, 20, ..., 90
    });
    let p50 = percentile(&img, 50.0);
    assert!((40..=50).contains(&p50), "p50={}", p50);
}

#[test]
fn flood_fill_corners_removes_border() {
    // 10x10 binary image, all white except a 4x4 hole in center
    let mut binary = GrayImage::from_pixel(10, 10, image::Luma([255]));
    for y in 3..7 {
        for x in 3..7 {
            binary.put_pixel(x, y, image::Luma([0]));
        }
    }

    // After flood fill from corners, everything connected to corners should be 0
    flood_fill_corners(&mut binary, 10, 10);

    // The center hole was already 0, it should remain 0
    // The border-connected region should now be 0
    let corner_val = binary.get_pixel(0, 0).0[0];
    assert_eq!(corner_val, 0, "corner should be filled to 0");

    // Center was already 0
    let center_val = binary.get_pixel(5, 5).0[0];
    assert_eq!(center_val, 0, "center was already 0");
}

#[test]
fn threshold_binary_inv_works() {
    let img = GrayImage::from_fn(10, 1, |x, _| image::Luma([x as u8 * 25]));
    let binary = threshold_binary_inv(&img, 100);

    // pixels with value <= 100 should be 255
    for x in 0..10 {
        let v = binary.get_pixel(x, 0).0[0];
        let original = x as u8 * 25;
        if original <= 100 {
            assert_eq!(v, 255, "pixel {x} (val={original}) should be 255");
        } else {
            assert_eq!(v, 0, "pixel {x} (val={original}) should be 0");
        }
    }
}

/// Minimal always-on subscriber so `info!` fields are evaluated.
struct Enabled;

impl tracing::Subscriber for Enabled {
    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, _: &tracing::Event<'_>) {}
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}

#[test]
fn objects_touching_an_edge_survive_corner_clearing_and_are_logged() {
    let mut img = make_test_image();
    // A third 8x8 object flush with the left edge, away from the corners.
    for y in 46..54 {
        for x in 0..8 {
            img.put_pixel(x, y, image::Luma([80]));
        }
    }
    let config = DetectionPipelineConfig {
        expected_count: 3,
        threshold: crate::config::ThresholdConfig::default(),
        mouse_removal: crate::config::MouseRemovalConfig {
            min_area_frac: 0.02,
            dilate_frac: 0.03,
        },
        area_filter: crate::config::AreaFilterConfig {
            min_frac: 0.003,
            max_frac: 0.02,
        },
        label_strategy: crate::config::LabelStrategy::Positional {
            labels: vec!["a".into(), "b".into(), "c".into()],
        },
    };
    let output = tracing::subscriber::with_default(Enabled, || detect_dark_objects(&img, &config));
    assert_eq!(output.blobs.len(), 3);
    // Closing erodes the outermost column, but the object is kept.
    assert!(
        output.blobs.iter().any(|blob| blob.bbox.0 <= 1),
        "edge object kept: {:?}",
        output.blobs.iter().map(|b| b.bbox).collect::<Vec<_>>()
    );
}

#[test]
fn components_flush_with_every_border_stay_whole() {
    let mut binary = GrayImage::new(6, 4);
    // An L touching the left and bottom edges, and a lone right-edge pixel.
    for (x, y) in [(0, 1), (0, 2), (0, 3), (1, 3), (5, 0)] {
        binary.put_pixel(x, y, image::Luma([255]));
    }
    let mut areas: Vec<_> = extract_components(&binary).iter().map(|c| c.area).collect();
    areas.sort();
    assert_eq!(areas, vec![1, 4]);
}
