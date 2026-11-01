use bekiper_core::inspection::PerimeterSpec;
use bekiper_core::types::DetectionRecipe;
use image::GrayImage;

use crate::config::{
    AreaFilterConfig, DetectionPipelineConfig, LabelStrategy, MouseRemovalConfig, ThresholdConfig,
};
use crate::label::{LabelError, assign_labels};
use crate::pipeline::{DetectionDebug, detect_dark_objects};

/// Expand a `DetectionRecipe` into a full pipeline configuration.
pub fn recipe_to_config(recipe: DetectionRecipe) -> DetectionPipelineConfig {
    match recipe {
        DetectionRecipe::NortTraining | DetectionRecipe::NortNovelty => nort_config(
            recipe
                .required_annotation_labels()
                .iter()
                .map(|label| (*label).into())
                .collect(),
        ),
        DetectionRecipe::OulTraining => oul_config(2, vec!["object_1".into(), "object_2".into()]),
        DetectionRecipe::OulUpdate => oul_config(2, vec!["object_1".into(), "object_3".into()]),
        DetectionRecipe::OulTest => oul_config(
            4,
            vec![
                "object_1".into(),
                "object_2".into(),
                "object_3".into(),
                "object_4".into(),
            ],
        ),
        DetectionRecipe::Oft => arena_config("arena"),
        DetectionRecipe::YMaze => arena_config("maze"),
    }
}

/// Result of `run_recipe`: labeled perimeters + debug artifacts (always available).
pub struct RecipeResult {
    pub perimeters: Vec<PerimeterSpec>,
    pub debug: DetectionDebug,
}

/// Run detection for a recipe on a grayscale image, returning labeled perimeters.
pub fn run_recipe(recipe: DetectionRecipe, gray: &GrayImage) -> Result<RecipeResult, DetectError> {
    let config = recipe_to_config(recipe);
    run_with_config(config, recipe, gray)
}

/// Run detection with an explicit pipeline config, returning labeled perimeters.
///
/// Use this when applying user overrides on top of recipe defaults:
/// ```ignore
/// let mut config = recipe_to_config(recipe);
/// config.threshold.floor_percentile = 80.0;
/// run_with_config(config, recipe, &gray)?;
/// ```
pub fn run_with_config(
    config: DetectionPipelineConfig,
    recipe: DetectionRecipe,
    gray: &GrayImage,
) -> Result<RecipeResult, DetectError> {
    let output = detect_dark_objects(gray, &config);

    if output.blobs.len() < config.expected_count {
        return Err(DetectError::InsufficientObjects {
            expected: config.expected_count,
            got: output.blobs.len(),
            recipe,
            debug: output.debug,
        });
    }

    let perimeters =
        assign_labels(&output.blobs, &config.label_strategy).map_err(DetectError::Label)?;
    Ok(RecipeResult {
        perimeters,
        debug: output.debug,
    })
}

fn nort_config(labels: Vec<String>) -> DetectionPipelineConfig {
    DetectionPipelineConfig {
        expected_count: 2,
        threshold: ThresholdConfig::default(),
        mouse_removal: MouseRemovalConfig {
            min_area_frac: 0.02,
            // Small dilation to avoid engulfing nearby small objects.
            dilate_frac: 0.005,
        },
        area_filter: AreaFilterConfig {
            min_frac: 0.005,
            max_frac: 0.05,
        },
        label_strategy: LabelStrategy::Diagonal { labels },
    }
}

fn oul_config(count: usize, labels: Vec<String>) -> DetectionPipelineConfig {
    DetectionPipelineConfig {
        expected_count: count,
        threshold: ThresholdConfig::default(),
        mouse_removal: MouseRemovalConfig {
            min_area_frac: 0.02,
            dilate_frac: 0.005,
        },
        area_filter: AreaFilterConfig {
            min_frac: 0.005,
            max_frac: 0.05,
        },
        label_strategy: LabelStrategy::Positional { labels },
    }
}

fn arena_config(label: &str) -> DetectionPipelineConfig {
    DetectionPipelineConfig {
        expected_count: 1,
        threshold: ThresholdConfig {
            floor_percentile: 75.0,
            threshold_factor: 0.85,
            close_kernel_frac: 0.01,
        },
        mouse_removal: MouseRemovalConfig {
            min_area_frac: 0.005,
            dilate_frac: 0.02,
        },
        area_filter: AreaFilterConfig {
            min_frac: 0.1,
            max_frac: 0.9,
        },
        label_strategy: LabelStrategy::Largest {
            label: label.to_string(),
        },
    }
}

#[derive(Debug, thiserror::Error)]
pub enum DetectError {
    #[error(
        "expected {expected} objects, got {got} for recipe {recipe}. \
         Try a different frame or check that objects are visible."
    )]
    InsufficientObjects {
        expected: usize,
        got: usize,
        recipe: DetectionRecipe,
        debug: DetectionDebug,
    },
    #[error(transparent)]
    Label(#[from] LabelError),
}

impl DetectError {
    /// Extract debug info if available (consumes the error).
    pub fn into_debug(self) -> Option<DetectionDebug> {
        match self {
            DetectError::InsufficientObjects { debug, .. } => Some(debug),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nort_training_config_has_correct_labels() {
        let config = recipe_to_config(DetectionRecipe::NortTraining);
        assert_eq!(config.expected_count, 2);
        assert!(matches!(&config.label_strategy,
            LabelStrategy::Diagonal { labels } if labels == &["variable", "familiar"]));
    }

    #[test]
    fn oul_test_config_has_four_objects() {
        let config = recipe_to_config(DetectionRecipe::OulTest);
        assert_eq!(config.expected_count, 4);
        assert!(matches!(&config.label_strategy,
            LabelStrategy::Positional { labels } if labels.len() == 4));
    }

    #[test]
    fn oft_config_uses_largest() {
        let config = recipe_to_config(DetectionRecipe::Oft);
        assert_eq!(config.expected_count, 1);
        assert!(
            matches!(&config.label_strategy, LabelStrategy::Largest { label } if label == "arena")
        );
    }

    #[test]
    fn run_recipe_on_synthetic_image() {
        // 100x100 bright image with two small dark squares
        let mut img = GrayImage::from_pixel(100, 100, image::Luma([200]));

        // Object at top-left quadrant
        for y in 15..23 {
            for x in 15..23 {
                img.put_pixel(x, y, image::Luma([80]));
            }
        }
        // Object at bottom-right quadrant
        for y in 75..83 {
            for x in 75..83 {
                img.put_pixel(x, y, image::Luma([80]));
            }
        }

        let result = run_recipe(DetectionRecipe::NortTraining, &img).unwrap();
        assert_eq!(result.perimeters.len(), 2);

        // Diagonal: top-left object should get "variable", bottom-right gets "familiar"
        assert_eq!(result.perimeters[0].label(), "variable");
        assert_eq!(result.perimeters[1].label(), "familiar");
    }

    #[test]
    fn oul_and_ymaze_configs_name_their_objects() {
        let labels = |recipe| match recipe_to_config(recipe).label_strategy {
            LabelStrategy::Positional { labels } => labels,
            LabelStrategy::Largest { label } => vec![label],
            LabelStrategy::Diagonal { labels } => labels,
        };
        assert_eq!(
            labels(DetectionRecipe::OulTraining),
            vec!["object_1", "object_2"]
        );
        assert_eq!(
            labels(DetectionRecipe::OulUpdate),
            vec!["object_1", "object_3"]
        );
        assert_eq!(labels(DetectionRecipe::YMaze), vec!["maze"]);
        assert_eq!(
            labels(DetectionRecipe::NortNovelty),
            vec!["novel", "familiar"]
        );
    }

    fn dark_squares(corners: &[(u32, u32)]) -> GrayImage {
        let mut img = GrayImage::from_pixel(100, 100, image::Luma([200]));
        for &(x0, y0) in corners {
            for y in y0..y0 + 8 {
                for x in x0..x0 + 8 {
                    img.put_pixel(x, y, image::Luma([80]));
                }
            }
        }
        img
    }

    #[test]
    fn missing_objects_keep_debug_and_extra_objects_fail_labelling() {
        let err = run_recipe(DetectionRecipe::NortTraining, &dark_squares(&[(15, 15)]))
            .err()
            .unwrap();
        assert!(matches!(
            err,
            DetectError::InsufficientObjects {
                expected: 2,
                got: 1,
                ..
            }
        ));
        assert!(err.to_string().contains("expected 2"), "{err}");
        assert!(err.into_debug().is_some());

        // Extra detections are truncated to the expected count, so only a
        // config whose labels disagree with that count fails labelling.
        let mut config = recipe_to_config(DetectionRecipe::NortTraining);
        config.label_strategy = LabelStrategy::Diagonal {
            labels: vec!["a".into(), "b".into(), "c".into()],
        };
        let image = dark_squares(&[(15, 15), (75, 75), (15, 75)]);
        let err = run_with_config(config, DetectionRecipe::NortTraining, &image)
            .err()
            .unwrap();
        assert!(matches!(
            err,
            DetectError::Label(LabelError::CountMismatch {
                expected: 3,
                got: 2
            })
        ));
        assert!(err.into_debug().is_none());
    }
}
