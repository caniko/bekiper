//! Validated seconds-based adapters to the shared temporal kernel.
use bekiper_core::video::FrameTiming;
use bekiper_math::temporal::select_bouts;
pub use bekiper_math::temporal::{DurationComparison, TemporalSelection};
use polars::prelude::*;
use serde::{Deserialize, Serialize};

pub const DEFAULT_MIN_DWELL_SECONDS: f64 = 1.0;

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DwellOverrides {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seconds: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub comparison: Option<DurationComparison>,
}

impl DwellOverrides {
    pub fn from_env(lookup: impl Fn(&str) -> Option<String>) -> Result<Self, String> {
        Self::default().with_env(lookup)
    }

    /// Constructor/CLI values are resolved before reading their lower-priority
    /// env keys, so a valid explicit override can replace an invalid env value.
    pub fn with_env(self, lookup: impl Fn(&str) -> Option<String>) -> Result<Self, String> {
        let seconds = if self.seconds.is_some() {
            self.seconds
        } else {
            lookup("BEKIPER_YMAZE_DWELL_SECONDS")
                .map(|s| {
                    s.parse::<f64>()
                        .map_err(|_| "BEKIPER_YMAZE_DWELL_SECONDS must be a number".to_string())
                })
                .transpose()?
        };
        let comparison = if self.comparison.is_some() {
            self.comparison
        } else {
            lookup("BEKIPER_YMAZE_DWELL_COMPARISON")
                .map(|s| s.parse())
                .transpose()?
        };
        let overrides = Self {
            seconds,
            comparison,
        };
        overrides.apply(TemporalPolicy::default()).validate()?;
        Ok(overrides)
    }

    pub fn is_empty(&self) -> bool {
        self.seconds.is_none() && self.comparison.is_none()
    }

    pub fn apply(self, policy: TemporalPolicy) -> TemporalPolicy {
        policy.with_overrides(self.seconds, self.comparison)
    }

    pub fn merged(self, other: Self) -> Self {
        Self {
            seconds: other.seconds.or(self.seconds),
            comparison: other.comparison.or(self.comparison),
        }
    }
}

/// Reusable temporal policy. A zero minimum disables dwell; bridging is separate.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TemporalPolicy {
    pub min_duration_seconds: f64,
    pub duration_comparison: DurationComparison,
    pub max_gap_seconds: f64,
    pub gap_density_threshold: f64,
}

impl TemporalPolicy {
    /// Principal Y-maze default: inclusive one-second uninterrupted membership.
    /// Other consumers retain the unfiltered generic `Default` policy.
    pub fn ymaze_default() -> Self {
        Self {
            min_duration_seconds: DEFAULT_MIN_DWELL_SECONDS,
            ..Self::default()
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("min_duration_seconds", self.min_duration_seconds),
            ("max_gap_seconds", self.max_gap_seconds),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(format!("{name} must be finite and nonnegative"));
            }
        }
        if !self.gap_density_threshold.is_finite()
            || !(0.0..=1.0).contains(&self.gap_density_threshold)
        {
            return Err("gap_density_threshold must be finite and in [0, 1]".into());
        }
        Ok(())
    }

    /// Consumers with strict continuity can use the shared validation rather
    /// than silently disabling a configured gap allowance.
    pub fn validate_uninterrupted(&self) -> Result<(), String> {
        self.validate()?;
        if self.max_gap_seconds != 0.0 {
            return Err(
                "temporal policy requires uninterrupted membership; max_gap_seconds must be zero"
                    .into(),
            );
        }
        Ok(())
    }

    /// Same overlay for constructor/CLI and injected environment lookups.
    pub fn with_overrides(
        self,
        seconds: Option<f64>,
        comparison: Option<DurationComparison>,
    ) -> Self {
        Self {
            min_duration_seconds: seconds.unwrap_or(self.min_duration_seconds),
            duration_comparison: comparison.unwrap_or(self.duration_comparison),
            ..self
        }
    }

    pub fn filter(&self, raw: &[bool], timing: &BoutTiming) -> Result<TemporalSelection, String> {
        self.validate()?;
        if raw.len() != timing.len() {
            return Err("temporal mask and timing lengths differ".into());
        }
        Ok(select_bouts(
            raw,
            |a, b| timing.duration(a, b),
            self.min_duration_seconds,
            self.duration_comparison,
            self.max_gap_seconds,
            self.gap_density_threshold,
        ))
    }

    pub fn filter_series(
        &self,
        raw: &BooleanChunked,
        timing: &BoutTiming,
    ) -> PolarsResult<TemporalSelection> {
        let mask: Vec<_> = raw.into_iter().map(|v| v.unwrap_or(false)).collect();
        self.filter(&mask, timing)
            .map_err(|e| PolarsError::ComputeError(e.into()))
    }
}

/// Half-open sample support: sample i covers [t_i, t_(i+1)). CFR duration uses
/// frame count / FPS directly, without seconds-to-frame rounding. Timestamped
/// streams require an explicit final boundary; missing observations break masks.
#[derive(Debug, Clone)]
pub struct BoutTiming {
    clock: Clock,
}

#[derive(Debug, Clone)]
enum Clock {
    Constant {
        fps: f64,
        samples: usize,
        start_frame: usize,
    },
    Timestamped {
        boundaries: Vec<f64>,
    },
}

impl BoutTiming {
    pub fn cfr(fps: f64, samples: usize) -> Result<Self, String> {
        if !fps.is_finite() || fps <= 0.0 {
            return Err("temporal FPS must be finite and positive".into());
        }
        Ok(Self {
            clock: Clock::Constant {
                fps,
                samples,
                start_frame: 0,
            },
        })
    }

    pub fn timestamps(timestamps: &[f64], end: f64) -> Result<Self, String> {
        let mut boundaries = timestamps.to_vec();
        boundaries.push(end);
        if boundaries.iter().any(|v| !v.is_finite()) || boundaries.windows(2).any(|v| v[1] <= v[0])
        {
            return Err("temporal sample boundaries must be finite and strictly increasing".into());
        }
        Ok(Self {
            clock: Clock::Timestamped { boundaries },
        })
    }

    /// Native frame adapter. The last frame has nominal 1/FPS support, as
    /// video PTS provides starts only. Record that convention in provenance.
    pub fn frames(timing: &FrameTiming, samples: usize) -> Result<Self, String> {
        if timing.timestamps.is_empty() {
            return Self::cfr(timing.fps, samples);
        }
        if timing.timestamps.len() != samples || !timing.fps.is_finite() || timing.fps <= 0.0 {
            return Err(
                "frame timestamps must cover every sample with positive nominal FPS".into(),
            );
        }
        let end = timing.timestamps.last().copied().unwrap_or(0.0) + 1.0 / timing.fps;
        Self::timestamps(&timing.timestamps, end)
    }

    pub fn len(&self) -> usize {
        match &self.clock {
            Clock::Constant { samples, .. } => *samples,
            Clock::Timestamped { boundaries } => boundaries.len() - 1,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn boundary(&self, index: usize) -> f64 {
        match &self.clock {
            Clock::Constant {
                fps, start_frame, ..
            } => (start_frame + index) as f64 / fps,
            Clock::Timestamped { boundaries } => boundaries[index],
        }
    }

    pub fn duration(&self, start: usize, end: usize) -> f64 {
        match &self.clock {
            Clock::Constant { fps, .. } => (end - start) as f64 / fps,
            Clock::Timestamped { .. } => self.boundary(end) - self.boundary(start),
        }
    }

    pub fn trimmed(&self, start: usize) -> Self {
        Self {
            clock: match &self.clock {
                Clock::Constant {
                    fps,
                    samples,
                    start_frame,
                } => Clock::Constant {
                    fps: *fps,
                    samples: samples - start,
                    start_frame: start_frame + start,
                },
                Clock::Timestamped { boundaries } => Clock::Timestamped {
                    boundaries: boundaries[start..].to_vec(),
                },
            },
        }
    }

    /// Retain the original half-open end boundary when selecting a window.
    pub fn sliced(&self, start: usize, end: usize) -> Self {
        assert!(start <= end && end <= self.len());
        Self {
            clock: match &self.clock {
                Clock::Constant {
                    fps, start_frame, ..
                } => Clock::Constant {
                    fps: *fps,
                    samples: end - start,
                    start_frame: start_frame + start,
                },
                Clock::Timestamped { boundaries } => Clock::Timestamped {
                    boundaries: boundaries[start..=end].to_vec(),
                },
            },
        }
    }

    /// Select samples whose starts precede `end_seconds`, clipping the final
    /// sample's half-open support at that exact boundary (including CFR).
    pub fn window(&self, start: usize, end_seconds: f64) -> Self {
        let end = (start..self.len())
            .find(|&i| self.boundary(i) >= end_seconds)
            .unwrap_or(self.len());
        let mut boundaries: Vec<_> = (start..end).map(|i| self.boundary(i)).collect();
        boundaries.push(self.boundary(end).min(end_seconds));
        Self {
            clock: Clock::Timestamped { boundaries },
        }
    }

    pub fn mask_seconds(&self, mask: &[bool]) -> f64 {
        match &self.clock {
            Clock::Constant { fps, .. } => mask.iter().filter(|&&v| v).count() as f64 / fps,
            Clock::Timestamped { .. } => mask
                .iter()
                .enumerate()
                .filter(|(_, v)| **v)
                .map(|(i, _)| self.duration(i, i + 1))
                .sum(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trimming_preserves_original_cfr_crossing_time() {
        let timing = BoutTiming::cfr(4.0, 12).unwrap().trimmed(4);
        assert_eq!(timing.boundary(0), 1.0);
        assert_eq!(timing.boundary(4), 2.0);
        assert_eq!(timing.duration(0, 4), 1.0);
    }

    #[test]
    fn seconds_never_truncate_fractional_frame_limits() {
        let policy = TemporalPolicy {
            min_duration_seconds: 1.5,
            ..Default::default()
        };
        assert!(
            !policy
                .filter(&[true; 37], &BoutTiming::cfr(25.0, 37).unwrap())
                .unwrap()
                .bouts[0]
                .accepted
        );
        assert!(
            policy
                .filter(&[true; 38], &BoutTiming::cfr(25.0, 38).unwrap())
                .unwrap()
                .bouts[0]
                .accepted
        );
    }

    #[test]
    fn variable_intervals_and_final_boundary_control_acceptance() {
        let policy = TemporalPolicy {
            min_duration_seconds: 1.0,
            ..Default::default()
        };
        let timing = BoutTiming::timestamps(&[5.0, 5.25, 5.5, 6.5], 6.75).unwrap();
        let result = policy.filter(&[false, true, true, false], &timing).unwrap();
        assert_eq!(result.bouts[0].duration, 1.25);
        assert_eq!(result.bouts[0].start, 1);
        assert_eq!(result.bouts[0].qualification_index, Some(2));
        assert_eq!(timing.mask_seconds(&result.mask), 1.25);
        assert!(BoutTiming::timestamps(&[0.0, 0.0], 1.0).is_err());
        assert!(BoutTiming::timestamps(&[0.0], 0.0).is_err());
        assert!(
            TemporalPolicy {
                min_duration_seconds: f64::NAN,
                ..policy
            }
            .validate()
            .is_err()
        );
    }
}
