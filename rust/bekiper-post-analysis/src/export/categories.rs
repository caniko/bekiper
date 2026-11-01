//! Ordered human-export classification. Earlier rules take precedence: QC and
//! imported/recomputed comparisons must not become parameters or provenance.
use serde_json::Value;

use crate::contracts::{
    KEY, nort_primary_fields, nort_secondary_fields, oft_primary_fields, oft_secondary_fields,
    ymaze_score_fields,
};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Category {
    Identity,
    Design,
    Parameters,
    Primary,
    Secondary,
    Measurements,
    Availability,
    Qc,
    Missingness,
    Verification,
    Comparisons,
    Provenance,
    AdditionalFields,
}

impl Category {
    pub fn name(self) -> &'static str {
        match self {
            Self::Identity => "identity",
            Self::Design => "design",
            Self::Parameters => "parameters",
            Self::Primary => "primary",
            Self::Secondary => "secondary",
            Self::Measurements => "measurements",
            Self::Availability => "availability",
            Self::Qc => "qc",
            Self::Missingness => "missingness",
            Self::Verification => "verification",
            Self::Comparisons => "comparisons",
            Self::Provenance => "provenance",
            Self::AdditionalFields => "additional_fields",
        }
    }
}

struct FieldRule {
    category: Category,
    exact: &'static [&'static str],
    prefixes: &'static [&'static str],
    suffixes: &'static [&'static str],
    fragments: &'static [&'static str],
}

impl FieldRule {
    const fn exact(category: Category, exact: &'static [&'static str]) -> Self {
        Self {
            category,
            exact,
            prefixes: &[],
            suffixes: &[],
            fragments: &[],
        }
    }

    fn matches(&self, field: &str) -> bool {
        self.exact.contains(&field)
            || self.prefixes.iter().any(|prefix| field.starts_with(prefix))
            || self.suffixes.iter().any(|suffix| field.ends_with(suffix))
            || self
                .fragments
                .iter()
                .any(|fragment| field.contains(fragment))
    }
}

// First match wins, including names that carry both a QC/comparison prefix and
// a parameter/provenance suffix. Rules share matching behavior, not precedence.
const FIELD_RULES: &[FieldRule] = &[
    FieldRule::exact(Category::Identity, &KEY),
    FieldRule::exact(
        Category::Design,
        &[
            "five_xfad",
            "hcar1",
            "treatment",
            "group",
            "sex",
            "age_treatment_window",
        ],
    ),
    FieldRule::exact(
        Category::Availability,
        &[
            "excluded",
            "exclusion_reason",
            "measurement_available",
            "synthetic_observation",
            "blocked_outputs",
        ],
    ),
    FieldRule {
        suffixes: &["_qc_flags"],
        fragments: &["qc_join"],
        ..FieldRule::exact(
            Category::Qc,
            &[
                "qc_flags",
                "low_quality",
                "low_usable_triads_flag",
                "low_valid_entries_flag",
                "passes_15s_exploration_threshold",
                "protocol_timing_departures",
            ],
        )
    },
    FieldRule {
        suffixes: &["_missing_reasons"],
        ..FieldRule::exact(Category::Missingness, &["missing_reasons"])
    },
    FieldRule::exact(Category::Verification, &["input_verification"]),
    FieldRule {
        prefixes: &["upstream_", "recomputed_"],
        ..FieldRule::exact(Category::Comparisons, &["disagreements"])
    },
    FieldRule {
        fragments: &["provenance"],
        ..FieldRule::exact(
            Category::Provenance,
            &[
                "geometry",
                "coordinate_samples",
                "candidate_segments",
                "center_intervals",
                "native_source",
                "metric_sources",
                "experiment",
            ],
        )
    },
    FieldRule {
        prefixes: &["analysis_"],
        fragments: &["window", "threshold"],
        ..FieldRule::exact(
            Category::Parameters,
            &[
                "first_detection_s",
                "analysis_start_s",
                "source_video_duration_s",
                "entry_rule",
                "threshold_radii",
                "crossline_endpoints",
                "fps",
                "meters_per_pixel",
                "protocol",
                "protocol_metadata",
                "object_configuration",
                "resolved_object_roles",
                "object_A_label",
                "object_B_label",
                "start_mode",
                "geometry_version",
                "pose_gap_fill_seconds",
                "zone_max_gap_seconds",
                "arm_membership_hysteresis",
                "temporal_policy",
            ],
        )
    },
];

const MEASUREMENTS: FieldRule = FieldRule {
    suffixes: &[
        "_s",
        "_m",
        "_pct",
        "_entries",
        "_bouts",
        "_speed",
        "_acceleration",
    ],
    fragments: &["sequence", "bouts", "triads", "exploration"],
    ..FieldRule::exact(
        Category::Measurements,
        &[
            "TA",
            "TB",
            "TN",
            "TF",
            "native_object_measurements",
            "window_labels",
            "raw_to_scored_entry_difference",
        ],
    )
};

pub(super) fn category(assay: &str, field: &str, value: &Value) -> Category {
    if let Some(rule) = FIELD_RULES.iter().find(|rule| rule.matches(field)) {
        return rule.category;
    }
    let primary_secondary: &[(&str, &[&str], &[&str])] = &[
        ("oft", oft_primary_fields(), oft_secondary_fields()),
        (
            "ymaze",
            &ymaze_score_fields()[..1],
            &ymaze_score_fields()[1..],
        ),
        ("nort", nort_primary_fields(), nort_secondary_fields()),
    ];
    if let Some((_, primary, secondary)) =
        primary_secondary.iter().find(|(name, _, _)| *name == assay)
    {
        for (category, names) in [
            (Category::Primary, *primary),
            (Category::Secondary, *secondary),
        ] {
            if names.contains(&field) {
                return category;
            }
        }
    }
    if value.is_number() || MEASUREMENTS.matches(field) {
        Category::Measurements
    } else {
        Category::AdditionalFields
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn identity_names_and_unknown_assays_fall_back_to_value_shape() {
        assert_eq!(category("oft", "animal_id", &json!("a")).name(), "identity");
        let rule = FieldRule::exact(Category::Design, &["sex"]);
        assert!(rule.matches("sex") && !rule.matches("sex_label"));
        assert!(category("rotarod", "DI", &json!(0.5)) == Category::Measurements);
        assert!(category("rotarod", "note", &json!("text")) == Category::AdditionalFields);
    }
}
