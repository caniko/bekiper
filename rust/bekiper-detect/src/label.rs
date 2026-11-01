use bekiper_core::inspection::PerimeterSpec;

use crate::config::LabelStrategy;
use crate::pipeline::DetectedBlob;

/// Assign labels to detected blobs and convert them to `PerimeterSpec`s.
///
/// Returns an error if the number of blobs doesn't match the strategy's expected count.
pub fn assign_labels(
    blobs: &[DetectedBlob],
    strategy: &LabelStrategy,
) -> Result<Vec<PerimeterSpec>, LabelError> {
    match strategy {
        LabelStrategy::Diagonal { labels } => diagonal_assignment(blobs, labels),
        LabelStrategy::Positional { labels } => positional_assignment(blobs, labels),
        LabelStrategy::Largest { label } => largest_assignment(blobs, label),
    }
}

/// Assign labels based on diagonal position (sum of x+y coordinates).
/// First label goes to the blob with the smallest x+y (closest to top-left).
fn diagonal_assignment(
    blobs: &[DetectedBlob],
    labels: &[String],
) -> Result<Vec<PerimeterSpec>, LabelError> {
    if blobs.len() != labels.len() {
        return Err(LabelError::CountMismatch {
            expected: labels.len(),
            got: blobs.len(),
        });
    }

    let mut indexed: Vec<(usize, f64)> = blobs
        .iter()
        .enumerate()
        .map(|(i, b)| (i, b.centroid.0 + b.centroid.1))
        .collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut result = Vec::with_capacity(labels.len());
    for (idx, (blob_idx, _)) in indexed.iter().enumerate() {
        result.push(blob_to_polygon(&blobs[*blob_idx], &labels[idx]));
    }
    Ok(result)
}

/// Assign labels sorted by position: top-to-bottom, then left-to-right.
fn positional_assignment(
    blobs: &[DetectedBlob],
    labels: &[String],
) -> Result<Vec<PerimeterSpec>, LabelError> {
    if blobs.len() != labels.len() {
        return Err(LabelError::CountMismatch {
            expected: labels.len(),
            got: blobs.len(),
        });
    }

    let mut indexed: Vec<(usize, (f64, f64))> = blobs
        .iter()
        .enumerate()
        .map(|(i, b)| (i, (b.centroid.1, b.centroid.0))) // (y, x)
        .collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut result = Vec::with_capacity(labels.len());
    for (idx, (blob_idx, _)) in indexed.iter().enumerate() {
        result.push(blob_to_polygon(&blobs[*blob_idx], &labels[idx]));
    }
    Ok(result)
}

/// Assign a single label to the largest detected blob.
fn largest_assignment(
    blobs: &[DetectedBlob],
    label: &str,
) -> Result<Vec<PerimeterSpec>, LabelError> {
    if blobs.is_empty() {
        return Err(LabelError::CountMismatch {
            expected: 1,
            got: 0,
        });
    }

    let largest = blobs
        .iter()
        .max_by_key(|b| b.area)
        .ok_or(LabelError::CountMismatch {
            expected: 1,
            got: 0,
        })?;
    Ok(vec![blob_to_polygon(largest, label)])
}

/// Convert a blob's bounding box to a polygon `PerimeterSpec`.
///
/// Uses the bounding rectangle as the polygon vertices (4 corners).
fn blob_to_polygon(blob: &DetectedBlob, label: &str) -> PerimeterSpec {
    let (x, y, w, h) = blob.bbox;
    let x = x as f64;
    let y = y as f64;
    let w = w as f64;
    let h = h as f64;

    PerimeterSpec::Polygon {
        label: label.to_string(),
        vertices: vec![(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
    }
}

#[derive(Debug, thiserror::Error)]
pub enum LabelError {
    #[error("expected {expected} objects, got {got}")]
    CountMismatch { expected: usize, got: usize },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_blob(cx: f64, cy: f64, area: u32) -> DetectedBlob {
        DetectedBlob {
            area,
            bbox: (cx as u32 - 5, cy as u32 - 5, 10, 10),
            centroid: (cx, cy),
            mask: image::GrayImage::new(1, 1),
        }
    }

    #[test]
    fn diagonal_assigns_top_left_first() {
        let blobs = vec![
            make_blob(80.0, 80.0, 100), // far from top-left
            make_blob(20.0, 20.0, 100), // close to top-left
        ];
        let labels = vec!["first".into(), "second".into()];
        let result = diagonal_assignment(&blobs, &labels).unwrap();
        assert_eq!(result[0].label(), "first"); // blob at (20,20) gets "first"
        assert_eq!(result[1].label(), "second"); // blob at (80,80) gets "second"
    }

    #[test]
    fn positional_assigns_top_to_bottom() {
        let blobs = vec![
            make_blob(50.0, 80.0, 100), // bottom
            make_blob(50.0, 20.0, 100), // top
        ];
        let labels = vec!["top".into(), "bottom".into()];
        let result = positional_assignment(&blobs, &labels).unwrap();
        assert_eq!(result[0].label(), "top");
        assert_eq!(result[1].label(), "bottom");
    }

    #[test]
    fn count_mismatch_error() {
        let blobs = vec![make_blob(50.0, 50.0, 100)];
        let labels = vec!["a".into(), "b".into()];
        let result = diagonal_assignment(&blobs, &labels);
        assert!(result.is_err());
    }

    #[test]
    fn largest_picks_biggest() {
        let blobs = vec![
            make_blob(20.0, 20.0, 50),
            make_blob(50.0, 50.0, 200),
            make_blob(80.0, 80.0, 100),
        ];
        let result = largest_assignment(&blobs, "arena").unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].label(), "arena");
        // Should pick the blob with area=200
        assert!(
            matches!(&result[0], PerimeterSpec::Polygon { vertices, .. }
                if (vertices.iter().map(|v| v.0).sum::<f64>() / 4.0 - 50.0).abs() < 1.0),
            "should pick blob at x=50"
        );
    }

    #[test]
    fn assign_labels_dispatches_and_checks_counts() {
        let blobs = vec![make_blob(50.0, 80.0, 300), make_blob(50.0, 20.0, 100)];
        let labels: Vec<String> = vec!["a".into(), "b".into()];
        let positional = LabelStrategy::Positional {
            labels: labels.clone(),
        };
        let result = assign_labels(&blobs, &positional).unwrap();
        assert_eq!(
            (result[0].label().as_str(), result[1].label().as_str()),
            ("a", "b")
        );
        let diagonal = LabelStrategy::Diagonal { labels };
        assert_eq!(assign_labels(&blobs, &diagonal).unwrap()[0].label(), "a");
        let largest = LabelStrategy::Largest {
            label: "arena".into(),
        };
        // The 300 px blob is 10 x 10 at (45, 75).
        assert!(matches!(&assign_labels(&blobs, &largest).unwrap()[0],
            PerimeterSpec::Polygon { vertices, .. } if vertices[0] == (45.0, 75.0) && vertices[2] == (55.0, 85.0)));
        let err = assign_labels(&blobs[..1], &positional).unwrap_err();
        assert_eq!(err.to_string(), "expected 2 objects, got 1");
        let err = assign_labels(&[], &largest).unwrap_err();
        assert_eq!(err.to_string(), "expected 1 objects, got 0");
    }
}
