//! Merged object bouts and observed-sample evidence.
use super::*;
use bekiper_feature::heuristic::combined::exploration::{
    BOUT_CONTACT_POLICY, BOUT_CONTACT_POLICY_WHISKER, BoutFilterMode, NortBoutPolicy,
    accepts_bout_angle,
};

#[derive(Clone)]
struct Candidate<'a> {
    value: &'a Value,
    id: &'a str,
    object: &'a str,
    start: f64,
    end: f64,
    rejected: bool,
    behavioral_barrier: bool,
    angles: Vec<f64>,
    speeds: Vec<f64>,
    nose_contact: bool,
    /// Any whisker-fan hit among the observed frames (policy-gated).
    whisker_hit: bool,
}

pub(super) fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(f64::total_cmp);
    let half = samples.len() / 2;
    if samples.len() % 2 == 1 {
        samples[half]
    } else {
        samples[half - 1] / 2.0 + samples[half] / 2.0
    }
}

pub(super) fn observed_samples(
    segment: &Value,
    name: &str,
    maximum: Option<f64>,
) -> Result<Vec<f64>> {
    let values = segment[name]
        .as_array()
        .context("Bout medians require observed samples, not medians-of-medians")?;
    ensure!(
        !values.is_empty(),
        "Bout medians require observed samples, not medians-of-medians"
    );
    values
        .iter()
        .map(|value| nonnegative(value, name, maximum))
        .collect()
}

fn optional_speeds(samples: &Value) -> Result<Vec<f64>> {
    match samples.get("nose_speeds_m_per_s") {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::Array(values)) if values.is_empty() => Ok(Vec::new()),
        Some(_) => observed_samples(samples, "nose_speeds_m_per_s", None),
    }
}

fn observed_contact(samples: &Value, angle_count: usize) -> Result<bool> {
    let Some(contacts) = samples
        .get("nose_contacts")
        .filter(|value| !value.is_null())
    else {
        return Ok(false); // Historical inputs without contact evidence stay angle-only.
    };
    let contacts = contacts
        .as_array()
        .context("nose_contacts must be a boolean sample array")?;
    ensure!(
        contacts.len() == angle_count,
        "nose_contacts must align with angles_degrees"
    );
    let contacts: Vec<_> = contacts
        .iter()
        .map(|value| {
            value
                .as_bool()
                .context("nose_contacts samples must be boolean")
        })
        .collect::<Result<_>>()?;
    Ok(contacts.into_iter().any(|contact| contact))
}

/// `whisker_hit` flag or an aligned `whisker_contacts` boolean sample array.
fn observed_whisker(samples: &Value, angle_count: usize) -> Result<bool> {
    if let Some(hit) = samples.get("whisker_hit").filter(|v| !v.is_null()) {
        return hit.as_bool().context("whisker_hit must be boolean");
    }
    let Some(contacts) = samples.get("whisker_contacts").filter(|v| !v.is_null()) else {
        return Ok(false);
    };
    let contacts = contacts
        .as_array()
        .context("whisker_contacts must be a boolean sample array")?;
    ensure!(
        contacts.len() == angle_count,
        "whisker_contacts must align with angles_degrees"
    );
    contacts.iter().try_fold(false, |any, value| {
        Ok(any
            | value
                .as_bool()
                .context("whisker_contacts samples must be boolean")?)
    })
}

/// Merge only same-object components without consuming another object's
/// interval or a rejected behavioral interval as a distraction/gap.
pub fn nort_bouts(segments: &[Record], gap_duration: &str, gap_medians: &str) -> Result<Record> {
    nort_bouts_with_policy(segments, gap_duration, gap_medians, &NortBoutPolicy::imen())
}

pub fn nort_bouts_with_policy(
    segments: &[Record],
    gap_duration: &str,
    gap_medians: &str,
    policy: &NortBoutPolicy,
) -> Result<Record> {
    policy.validate().map_err(anyhow::Error::msg)?;
    ensure!(
        policy.speed_filter_mode != BoutFilterMode::PerFrame,
        "standalone bout reconstruction requires disabled or per_bout_median speed mode"
    );
    ensure!(
        matches!(gap_duration, "include_gaps" | "exclude_gaps")
            && matches!(gap_medians, "observed_samples_only" | "include_gaps"),
        "D07 requires supported explicit duration and median gap accounting"
    );
    let mut identifiers = BTreeSet::new();
    let mut candidates = Vec::new();
    let mut raw = Vec::new();
    for segment in segments {
        ensure!(segment.is_object(), "Candidate segment must be an object");
        let id = text(segment, "segment_id")?;
        let object = text(segment, "object_id")?;
        let start = nonnegative(&segment["start_s"], "start_s", None)?;
        let end = nonnegative(&segment["end_s"], "end_s", None)?;
        ensure!(
            end > start && identifiers.insert(id),
            "Invalid/duplicate candidate interval"
        );
        let eligible = segment["candidate_eligible"]
            .as_bool()
            .context("Candidate eligibility must be an explicit boolean")?;
        let behavioral_barrier = matches!(
            segment["behaviour"].as_str(),
            Some("climbing" | "sitting_on_object" | "grooming" | "rearing")
        ) && segment["object_directed"] != true;
        let mut reasons = Vec::new();
        if !eligible {
            reasons.push("upstream_candidate_rejection");
        }
        if behavioral_barrier {
            reasons.push("behaviour_not_object_directed");
        }
        let angles = observed_samples(segment, "angles_degrees", Some(180.0))?;
        // Speed is descriptive under the October 7 instruction. Its absence
        // cannot reject an otherwise valid candidate or bout.
        let speeds = optional_speeds(segment)?;
        let nose_contact = observed_contact(segment, angles.len())?;
        let whisker_hit = policy.whisker_enabled && observed_whisker(segment, angles.len())?;
        let mut debug = segment.clone();
        debug["stage"] = "candidate".into();
        debug["rejection_reasons"] = json!(reasons);
        debug["valid"] = json!(reasons.is_empty());
        raw.push(debug);
        candidates.push(Candidate {
            value: segment,
            id,
            object,
            start,
            end,
            rejected: !reasons.is_empty(),
            behavioral_barrier,
            angles,
            speeds,
            nose_contact,
            whisker_hit,
        });
    }
    let mut groups: BTreeMap<&str, Vec<&Candidate<'_>>> = BTreeMap::new();
    for candidate in &candidates {
        groups.entry(candidate.object).or_default().push(candidate);
    }
    let mut accepted: Vec<_> = candidates.iter().filter(|c| !c.rejected).collect();
    accepted.sort_by(|a, b| a.start.total_cmp(&b.start));
    for (i, first) in accepted.iter().enumerate() {
        ensure!(
            !candidates
                .iter()
                .any(|b| b.behavioral_barrier && b.start < first.end && b.end > first.start),
            "Overlapping behavioural exclusion and accepted candidate require upstream resolution (D07)"
        );
        for second in &accepted[i + 1..] {
            if second.start >= first.end {
                break;
            }
            ensure!(
                first.object == second.object,
                "Simultaneous different-object candidates need upstream registered arbitration (D07)"
            );
        }
    }
    let mut times: BTreeMap<&str, f64> = BTreeMap::new();
    let mut merged_debug = Vec::new();
    for (object, mut group) in groups {
        group.sort_by(|a, b| a.start.total_cmp(&b.start));
        let mut prior_end = f64::NEG_INFINITY;
        for candidate in &group {
            ensure!(
                candidate.start >= prior_end,
                "Overlapping same-object candidate intervals"
            );
            prior_end = candidate.end;
        }
        let barriers_between = |previous_end: f64, start: f64| -> BTreeSet<&str> {
            candidates
                .iter()
                .filter(|b| {
                    (b.rejected || b.object != object) && b.start < start && b.end > previous_end
                })
                .map(|b| b.id)
                .collect()
        };
        let accepted: Vec<_> = group.into_iter().filter(|c| !c.rejected).collect();
        let batches = bekiper_math::temporal::partition_adjacent(&accepted, |previous, next| {
            (if policy.strict_boundaries {
                next.start - previous.end < policy.max_gap_seconds
            } else {
                next.start - previous.end <= policy.max_gap_seconds
            }) && barriers_between(previous.end, next.start).is_empty()
        });
        // A candidate for this object was observed, even if all were rejected.
        // An object absent from the input is never assigned an invented zero.
        times.insert(object, 0.0);
        let mut previous_end = f64::NEG_INFINITY;
        for batch in batches {
            let first = batch.first().context("empty merged bout")?;
            let last = batch.last().context("empty merged bout")?;
            let barriers: Vec<_> = barriers_between(previous_end, first.start)
                .into_iter()
                .collect();
            previous_end = last.end;
            let observed: f64 = batch.iter().map(|c| c.end - c.start).sum();
            let span = last.end - first.start;
            let duration = if gap_duration == "include_gaps" {
                span
            } else {
                observed
            };
            let mut angles: Vec<_> = batch
                .iter()
                .flat_map(|c| c.angles.iter().copied())
                .collect();
            let mut speeds: Vec<_> = batch
                .iter()
                .flat_map(|c| c.speeds.iter().copied())
                .collect();
            let mut nose_contact = batch.iter().any(|c| c.nose_contact || c.whisker_hit);
            let mut gap_evidence = Vec::new();
            if gap_medians == "include_gaps" {
                for pair in batch.windows(2).filter(|pair| pair[1].start > pair[0].end) {
                    let gap = pair[1]
                        .value
                        .get("gap_before")
                        .filter(|value| value.is_object())
                        .context(
                            "gap_before samples are required for gap-inclusive bout medians",
                        )?;
                    ensure!(
                        gap["start_s"].as_f64() == Some(pair[0].end)
                            && gap["end_s"].as_f64() == Some(pair[1].start),
                        "gap_before must cover exactly the preceding merged gap"
                    );
                    let gap_angles = observed_samples(gap, "angles_degrees", Some(180.0))
                        .context("gap_before requires observed angle samples")?;
                    nose_contact |= observed_contact(gap, gap_angles.len())
                        .context("gap_before contact samples are invalid")?;
                    if policy.whisker_enabled {
                        nose_contact |= observed_whisker(gap, gap_angles.len())
                            .context("gap_before whisker samples are invalid")?;
                    }
                    speeds.extend(
                        optional_speeds(gap).context("gap_before speed samples are invalid")?,
                    );
                    angles.extend(gap_angles);
                    gap_evidence.push(gap);
                }
            }
            let angle = median(angles);
            let speed = (!speeds.is_empty()).then(|| median(speeds));
            let angle_ok = accepts_bout_angle(Some(angle), policy.max_angle_deg, nose_contact);
            let contact_rescue = angle > policy.max_angle_deg && angle_ok;
            let mut reasons = Vec::new();
            if !bekiper_math::temporal::DurationComparison::AtLeast
                .accepts(duration, policy.min_bout_seconds)
            {
                reasons.push("duration");
            }
            if !angle_ok {
                reasons.push("angle");
            }
            if policy.speed_filter_mode != BoutFilterMode::Disabled
                && !speed.is_some_and(|v| {
                    if policy.strict_boundaries {
                        v < policy.max_nose_speed
                    } else {
                        v <= policy.max_nose_speed
                    }
                })
            {
                reasons.push("speed");
            }
            ensure!(
                duration.is_finite() && observed.is_finite(),
                "Merged bout duration overflow"
            );
            merged_debug.push(json!({"stage":"merged","object_id":object,"component_ids":batch.iter().map(|c| c.id).collect::<Vec<_>>(),
                "components":batch.iter().map(|c| c.value).collect::<Vec<_>>(),"merge_barrier_ids":barriers,
                "start_s":first.start,"end_s":last.end,"duration_s":duration,"observed_duration_s":observed,"gap_s":span-observed,
                "median_angle_degrees":angle,"median_nose_speed_m_per_s":speed,"gap_duration_policy":gap_duration,
                "gap_median_policy":gap_medians,"speed_filter_mode":policy.speed_filter_mode,
                "gap_evidence":gap_evidence,"bout_contact_policy":if policy.whisker_enabled {BOUT_CONTACT_POLICY_WHISKER} else {BOUT_CONTACT_POLICY},
                "contact_override_applied":contact_rescue,
                "minimum_dwell_seconds":policy.min_bout_seconds,"duration_comparison":">=",
                "merge_gap_seconds":policy.max_gap_seconds,"merge_gap_comparison":if policy.strict_boundaries {"<"} else {"<="},
                "speed_comparison":if policy.strict_boundaries {"<"} else {"<="},
                "policy_reference":if *policy == NortBoutPolicy::imen() {"researcher_instruction_2026-10-07"} else {"configured_nort_bout_policy_v1"},
                "rejection_reasons":reasons,"valid":reasons.is_empty()}));
            if reasons.is_empty() {
                let time = times.get_mut(object).context("missing object time")?;
                *time += duration;
                ensure!(time.is_finite(), "Object exploration duration overflow");
            }
        }
    }
    // The session-wide barriers imply non-overlapping accepted merged spans.
    // Assert the invariant so later merge-policy changes cannot double-count.
    let mut valid: Vec<_> = merged_debug.iter().filter(|b| b["valid"] == true).collect();
    valid.sort_by(|a, b| {
        a["start_s"]
            .as_f64()
            .unwrap_or(0.0)
            .total_cmp(&b["start_s"].as_f64().unwrap_or(0.0))
    });
    for pair in valid.windows(2) {
        ensure!(
            pair[0]["end_s"].as_f64() <= pair[1]["start_s"].as_f64(),
            "Merged exploration intervals overlap across objects"
        );
    }
    let count = |reason: &str| {
        merged_debug
            .iter()
            .filter(|b| {
                b["rejection_reasons"]
                    .as_array()
                    .is_some_and(|r| r.iter().any(|v| v == reason))
            })
            .count()
    };
    let valid_bouts = valid.len();
    let removed_duration = count("duration");
    let removed_speed = count("speed");
    let removed_angle = count("angle");
    let contact_rescues = merged_debug
        .iter()
        .filter(|bout| bout["contact_override_applied"] == true && bout["valid"] == true)
        .count();
    raw.extend(merged_debug);
    Ok(
        json!({"object_exploration_s":times,"bout_debug":raw,"valid_bouts":valid_bouts,
        "removed_duration_bouts":removed_duration,"removed_speed_bouts":removed_speed,"removed_angle_bouts":removed_angle,
        "bouts_rescued_by_contact_override":contact_rescues}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_recorded_whisker_hit_flag_must_be_boolean() {
        assert!(observed_whisker(&json!({"whisker_hit": true}), 0).unwrap());
        assert!(observed_whisker(&json!({"whisker_hit": "yes"}), 0).is_err());
    }
}
