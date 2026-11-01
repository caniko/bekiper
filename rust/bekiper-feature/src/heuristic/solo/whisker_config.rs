//! Whisker-fan configuration: serde shape of
//! `[...task_params.exploration.whisker]`, variant merging, env overlay and the
//! pure age-to-length resolution. Ingress only supplies the workbook lookup.
use serde::{Deserialize, Serialize};

use super::whiskers::WhiskerFanParams;

/// Recording-metadata age may differ from the workbook age by this many days
/// before a mismatch warning is raised.
pub const AGE_MISMATCH_TOLERANCE_DAYS: f64 = 1.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgeTableMethod {
    AgeTable,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetersSpec {
    pub meters: f64,
}

/// `{ method = "age_table", source = "mice_overview.xlsx", ... table = [[age_days, meters], ...] }`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgeTableSpec {
    pub method: AgeTableMethod,
    /// Workbook path relative to the config directory.
    pub source: String,
    #[serde(default = "default_sheet")]
    pub sheet: String,
    #[serde(default = "default_animal_column")]
    pub animal_column: String,
    #[serde(default = "default_pre_age_column")]
    pub pre_age_column: String,
    #[serde(default = "default_post_age_column")]
    pub post_age_column: String,
    pub table: Vec<(f64, f64)>,
}

fn default_sheet() -> String {
    "Updated_Mice-overview".into()
}
fn default_animal_column() -> String {
    "Animal ID".into()
}
fn default_pre_age_column() -> String {
    "Start age (days)".into()
}
fn default_post_age_column() -> String {
    "End age (days)".into()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum WhiskerLengthSpec {
    Meters(MetersSpec),
    AgeTable(AgeTableSpec),
}

/// Optional layer; `None` fields inherit. Variants may override any of them.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WhiskerOverrides {
    /// Default false: nose rule only.
    pub enabled: Option<bool>,
    /// Position along the head axis from the inter-ear midpoint (0, the head
    /// midline) to the nose tip (1).
    pub apex_fraction: Option<f64>,
    pub half_angle_deg: Option<f64>,
    pub length: Option<WhiskerLengthSpec>,
    /// Injected by ingress when `length` is an age table (provenance).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub age_days: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub length_source: Option<String>,
    /// The configured age-table spec, kept by ingress after it replaces `length`
    /// with the resolved `{ meters }` (provenance only; never read from TOML).
    #[serde(skip)]
    pub age_table: Option<AgeTableSpec>,
}

impl WhiskerOverrides {
    pub fn merged(&self, other: &Self) -> Self {
        Self {
            enabled: other.enabled.or(self.enabled),
            apex_fraction: other.apex_fraction.or(self.apex_fraction),
            half_angle_deg: other.half_angle_deg.or(self.half_angle_deg),
            length: other.length.clone().or_else(|| self.length.clone()),
            age_days: other.age_days.or(self.age_days),
            length_source: other
                .length_source
                .clone()
                .or_else(|| self.length_source.clone()),
            age_table: other.age_table.clone().or_else(|| self.age_table.clone()),
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled.unwrap_or(false)
    }

    /// Resolved fan parameters; `None` when disabled. An unresolved age table
    /// is an error (ingress resolves it at experiment expansion).
    pub fn params(&self) -> Result<Option<WhiskerFanParams>, String> {
        if !self.is_enabled() {
            return Ok(None);
        }
        let d = WhiskerFanParams::default();
        let length_m = match &self.length {
            None => d.length_m,
            Some(WhiskerLengthSpec::Meters(m)) => m.meters,
            Some(WhiskerLengthSpec::AgeTable(_)) => {
                return Err("whisker length age_table was not resolved for this experiment".into());
            }
        };
        let params = WhiskerFanParams {
            apex_fraction: self.apex_fraction.unwrap_or(d.apex_fraction),
            half_angle_deg: self.half_angle_deg.unwrap_or(d.half_angle_deg),
            length_m,
        };
        params.validate()?;
        Ok(Some(params))
    }
}

/// Scalar knobs from the environment / CLI (explicit values win over env).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct WhiskerEnv {
    pub enabled: Option<bool>,
    pub apex_fraction: Option<f64>,
    pub half_angle_deg: Option<f64>,
    pub length_m: Option<f64>,
}

impl WhiskerEnv {
    pub fn with_env(self, lookup: impl Fn(&str) -> Option<String>) -> Result<Self, String> {
        let number = |explicit: Option<f64>, key: &str| -> Result<Option<f64>, String> {
            match explicit {
                Some(v) => Ok(Some(v)),
                None => lookup(key)
                    .map(|v| {
                        v.parse::<f64>()
                            .ok()
                            .filter(|v| v.is_finite())
                            .ok_or_else(|| format!("{key} must be a finite number"))
                    })
                    .transpose(),
            }
        };
        let enabled = match self.enabled {
            Some(v) => Some(v),
            None => lookup("BEKIPER_NORT_WHISKER_ENABLED")
                .map(|v| match v.as_str() {
                    "true" | "1" => Ok(true),
                    "false" | "0" => Ok(false),
                    _ => Err("BEKIPER_NORT_WHISKER_ENABLED must be true or false".to_string()),
                })
                .transpose()?,
        };
        Ok(Self {
            enabled,
            apex_fraction: number(self.apex_fraction, "BEKIPER_NORT_WHISKER_APEX_FRACTION")?,
            half_angle_deg: number(self.half_angle_deg, "BEKIPER_NORT_WHISKER_HALF_ANGLE_DEG")?,
            length_m: number(self.length_m, "BEKIPER_NORT_WHISKER_LENGTH_M")?,
        })
    }

    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }

    /// Overlay onto one config layer; a set length replaces any age table.
    pub fn apply(&self, target: &mut WhiskerOverrides) {
        target.enabled = self.enabled.or(target.enabled);
        target.apex_fraction = self.apex_fraction.or(target.apex_fraction);
        target.half_angle_deg = self.half_angle_deg.or(target.half_angle_deg);
        if let Some(meters) = self.length_m {
            target.length = Some(WhiskerLengthSpec::Meters(MetersSpec { meters }));
        }
    }
}

impl AgeTableSpec {
    /// Sorted-by-age, finite, positive-length table with at least one anchor.
    pub fn validate(&self) -> Result<(), String> {
        if self.table.is_empty() {
            return Err("whisker age_table needs at least one [age_days, meters] anchor".into());
        }
        let finite = self
            .table
            .iter()
            .all(|&(a, m)| a.is_finite() && a >= 0.0 && m.is_finite() && m > 0.0);
        if !finite || self.table.windows(2).any(|w| w[1].0 <= w[0].0) {
            return Err(
                "whisker age_table anchors must be finite, ages strictly increasing and lengths positive"
                    .into(),
            );
        }
        Ok(())
    }

    /// Linear interpolation of length (m) at `age_days`, clamped at the ends.
    pub fn length_at(&self, age_days: f64) -> f64 {
        let t = &self.table;
        if age_days <= t[0].0 {
            return t[0].1;
        }
        if age_days >= t[t.len() - 1].0 {
            return t[t.len() - 1].1;
        }
        let i = t.partition_point(|&(a, _)| a <= age_days);
        let ((a0, m0), (a1, m1)) = (t[i - 1], t[i]);
        m0 + (m1 - m0) * (age_days - a0) / (a1 - a0)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedWhiskerLength {
    /// `None` when the animal has no age but the table is constant.
    pub age_days: Option<f64>,
    pub length_m: f64,
    /// e.g. `mice_overview.xlsx:Updated_Mice-overview:Start age (days)`.
    pub source: String,
    /// Set when the recording metadata age disagrees with the workbook.
    pub warning: Option<String>,
}

/// Pick the pre (start) or post (end) age of one animal and interpolate.
/// `ages` is the workbook `(start, end)` pair for the animal (`None` = absent).
/// A missing age is an error unless the table is constant, whose length does
/// not depend on age; that case resolves without an age and warns.
pub fn resolve_age_length(
    spec: &AgeTableSpec,
    experiment: &str,
    timepoint: Option<&str>,
    ages: Option<(Option<f64>, Option<f64>)>,
    recorded_age_days: Option<f64>,
) -> Result<ResolvedWhiskerLength, String> {
    spec.validate()?;
    let (column, pick) = match timepoint {
        Some("pre") => (&spec.pre_age_column, ages.map(|a| a.0)),
        Some("post") => (&spec.post_age_column, ages.map(|a| a.1)),
        other => {
            return Err(format!(
                "whisker age_table: experiment {experiment} has no pre/post timepoint ({other:?})"
            ));
        }
    };
    let constant = spec.table.iter().all(|row| row.1 == spec.table[0].1);
    let age_days = pick
        .ok_or_else(|| {
            format!(
                "whisker age_table: experiment {experiment} animal is absent from {}",
                spec.source
            )
        })
        .and_then(|age| {
            age.filter(|a| a.is_finite()).ok_or_else(|| {
                format!(
                    "whisker age_table: experiment {experiment} has no `{column}` value in {}",
                    spec.source
                )
            })
        });
    let age_days = match age_days {
        Ok(age) => age,
        Err(reason) if constant => {
            return Ok(ResolvedWhiskerLength {
                age_days: None,
                length_m: spec.table[0].1,
                source: format!("{}:{}:constant_table", spec.source, spec.sheet),
                warning: Some(format!(
                    "{reason}; the table is constant, so no age is needed"
                )),
            });
        }
        Err(reason) => return Err(reason),
    };
    let warning = recorded_age_days
        .filter(|r| (r - age_days).abs() > AGE_MISMATCH_TOLERANCE_DAYS)
        .map(|r| {
            format!(
                "experiment {experiment}: recording metadata age {r} d differs from {} `{column}` {age_days} d",
                spec.source
            )
        });
    Ok(ResolvedWhiskerLength {
        age_days: Some(age_days),
        length_m: spec.length_at(age_days),
        source: format!("{}:{}:{}", spec.source, spec.sheet, column),
        warning,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(table: &[(f64, f64)]) -> AgeTableSpec {
        AgeTableSpec {
            method: AgeTableMethod::AgeTable,
            source: "mice_overview.xlsx".into(),
            sheet: default_sheet(),
            animal_column: default_animal_column(),
            pre_age_column: default_pre_age_column(),
            post_age_column: default_post_age_column(),
            table: table.to_vec(),
        }
    }

    #[test]
    fn interpolates_and_clamps() {
        let s = spec(&[(100.0, 0.010), (200.0, 0.020)]);
        assert_eq!(s.length_at(50.0), 0.010);
        assert!((s.length_at(150.0) - 0.015).abs() < 1e-12);
        assert_eq!(s.length_at(500.0), 0.020);
        assert_eq!(
            spec(&[(0.0, 0.0117), (400.0, 0.0117)]).length_at(123.0),
            0.0117
        );
    }

    #[test]
    fn rejects_bad_tables() {
        for t in [
            vec![],
            vec![(2.0, 0.01), (1.0, 0.02)],
            vec![(1.0, 0.0)],
            vec![(1.0, 0.01), (1.0, 0.02)],
            vec![(1.0, f64::NAN)],
        ] {
            assert!(spec(&t).validate().is_err(), "{t:?}");
        }
    }

    #[test]
    fn pre_uses_start_post_uses_end_and_errors_name_experiment() {
        let s = spec(&[(100.0, 0.010), (200.0, 0.020)]);
        let ages = Some((Some(100.0), Some(200.0)));
        let pre = resolve_age_length(&s, "e1", Some("pre"), ages, None).unwrap();
        let post = resolve_age_length(&s, "e1", Some("post"), ages, None).unwrap();
        assert_eq!((pre.age_days, pre.length_m), (Some(100.0), 0.010));
        assert_eq!((post.age_days, post.length_m), (Some(200.0), 0.020));
        assert!(post.source.ends_with(":End age (days)"));
        for (tp, ages) in [
            (Some("pre"), None),
            (None, ages),
            (Some("pre"), Some((None, Some(1.0)))),
        ] {
            let err = resolve_age_length(&s, "e9", tp, ages, None).unwrap_err();
            assert!(err.contains("e9"), "{err}");
        }
    }

    #[test]
    fn constant_table_resolves_without_an_age() {
        let flat = spec(&[(0.0, 0.0117), (400.0, 0.0117)]);
        for ages in [None, Some((None, None))] {
            let r = resolve_age_length(&flat, "e7", Some("post"), ages, None).unwrap();
            assert_eq!((r.age_days, r.length_m), (None, 0.0117));
            assert!(r.source.ends_with(":constant_table"));
            assert!(r.warning.unwrap().contains("e7"));
        }
        // A missing timepoint stays an error even for a constant table.
        assert!(resolve_age_length(&flat, "e7", None, None, None).is_err());
    }

    #[test]
    fn mismatch_warns_only_beyond_tolerance() {
        let s = spec(&[(0.0, 0.0117)]);
        let ages = Some((Some(100.0), Some(130.0)));
        assert!(
            resolve_age_length(&s, "e", Some("pre"), ages, Some(100.5))
                .unwrap()
                .warning
                .is_none()
        );
        assert!(
            resolve_age_length(&s, "e", Some("pre"), ages, Some(110.0))
                .unwrap()
                .warning
                .is_some()
        );
    }

    #[test]
    fn merge_and_resolve() {
        let meters = |m| Some(WhiskerLengthSpec::Meters(MetersSpec { meters: m }));
        let o = WhiskerOverrides {
            enabled: Some(true),
            length: meters(0.02),
            ..Default::default()
        };
        assert_eq!(o.params().unwrap().unwrap().length_m, 0.02);
        let v = WhiskerOverrides {
            length: meters(0.03),
            ..Default::default()
        };
        let m = o.merged(&v);
        assert!(m.is_enabled());
        assert_eq!(m.params().unwrap().unwrap().length_m, 0.03);
        assert_eq!(WhiskerOverrides::default().params().unwrap(), None);
        let t = WhiskerOverrides {
            enabled: Some(true),
            length: Some(WhiskerLengthSpec::AgeTable(spec(&[(0.0, 0.0117)]))),
            ..Default::default()
        };
        assert!(
            t.params().is_err(),
            "unresolved age table must not fall back"
        );
        let d = WhiskerOverrides {
            enabled: Some(true),
            ..Default::default()
        };
        let p = d.params().unwrap().unwrap();
        assert_eq!(
            (p.apex_fraction, p.half_angle_deg, p.length_m),
            (0.9, 30.0, 0.0117)
        );
    }

    #[test]
    fn env_overlay_uses_injected_lookup_and_explicit_wins() {
        let lookup = |k: &str| match k {
            "BEKIPER_NORT_WHISKER_ENABLED" => Some("1".to_string()),
            "BEKIPER_NORT_WHISKER_LENGTH_M" => Some("0.02".to_string()),
            _ => None,
        };
        let env = WhiskerEnv {
            length_m: Some(0.03),
            ..Default::default()
        }
        .with_env(lookup)
        .unwrap();
        assert_eq!((env.enabled, env.length_m), (Some(true), Some(0.03)));
        let mut target = WhiskerOverrides::default();
        env.apply(&mut target);
        assert_eq!(target.params().unwrap().unwrap().length_m, 0.03);
        assert!(
            WhiskerEnv::default()
                .with_env(|_| Some("nan".to_string()))
                .is_err()
        );
        let off = WhiskerEnv {
            enabled: Some(false),
            ..Default::default()
        }
        .with_env(lookup)
        .unwrap();
        assert_eq!(off.enabled, Some(false));
    }
}
