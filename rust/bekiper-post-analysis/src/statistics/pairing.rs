//! Explicit pre/post pairing and unmatched-pair ledger.
use super::*;

/// Pair explicit pre/post observations by animal/assay/phase/threshold/pair ID,
/// independent of row order. The ledger retains missing or rejected pairs.
pub fn pairing(
    rows: &[Record],
    endpoint: &str,
    requirement: Option<&str>,
) -> Result<(Vec<Record>, Vec<Record>)> {
    validate_requirement(requirement)?;
    let mut groups: BTreeMap<String, BTreeMap<String, &Record>> = BTreeMap::new();
    for row in rows {
        let timepoint = identity(row, "timepoint")?;
        ensure!(
            matches!(timepoint, "pre" | "post"),
            "Pairing requires canonical pre/post timepoints"
        );
        let threshold = if row["threshold_pct"].is_null() {
            Value::Null
        } else {
            json!(number(row, "threshold_pct")?)
        };
        let pair_id = if row["pair_id"].is_null() {
            "primary"
        } else {
            identity(row, "pair_id")?
        };
        let key = serde_json::to_string(&json!([
            identity(row, "animal_id")?,
            identity(row, "assay")?,
            identity(row, "phase")?,
            threshold,
            pair_id
        ]))?; // coverage: unreachable — serializing an in-memory JSON value cannot fail
        identity(row, "session_id")?;
        ensure!(
            groups
                .entry(key)
                .or_default()
                .insert(timepoint.into(), row)
                .is_none(),
            "Multiple sessions per animal/timepoint; explicit pair_id or aggregation plan required"
        );
    }
    let mut paired = Vec::new();
    let mut ledger = Vec::new();
    for (key, parts) in groups {
        let key: Value = serde_json::from_str(&key)?;
        let pre = parts.get("pre").copied();
        let post = parts.get("post").copied();
        let mut reason = None;
        if let (Some(pre), Some(post)) = (pre, post) {
            for field in FACTORS.into_iter().chain(["group", "sex"]) {
                ensure!(
                    pre[field] == post[field],
                    "Inconsistent paired animal metadata for {field}"
                );
            }
            if excluded(pre)? || excluded(post)? {
                reason = Some("excluded:paired_metadata".into());
            } else if let Some(value) = crate::genotype::eligibility_reason(pre)
                .or_else(|| crate::genotype::eligibility_reason(post))
            {
                reason = Some(value);
            } else if let Some(value) = verification_reason(pre, requirement) {
                reason =
                    Some(value.replacen("upstream_unverified:", "upstream_unverified:pre:", 1));
            } else if let Some(value) = verification_reason(post, requirement) {
                reason =
                    Some(value.replacen("upstream_unverified:", "upstream_unverified:post:", 1));
            } else if pre[endpoint].is_null() || post[endpoint].is_null() {
                reason = Some("missing_pair_endpoint".into());
            }
        } else {
            reason = Some("unmatched_prepost".into());
        }
        ledger.push(json!({"animal_id":key[0], "assay":key[1], "phase":key[2], "threshold_pct":key[3],
            "pair_id":key[4], "endpoint":endpoint, "reason":reason, "verification_requirement":requirement,
            "pre_session_id":pre.map(|row| &row["session_id"]), "post_session_id":post.map(|row| &row["session_id"]),
            "pre_input_verification_status":pre.map(verification_status),
            "post_input_verification_status":post.map(verification_status),
            "pre_genotype_eligibility":pre.map(|row|&row["genotype_eligibility"]),
            "post_genotype_eligibility":post.map(|row|&row["genotype_eligibility"]),
            "pre_endpoint_missing":pre.map(|row|row[endpoint].is_null()),
            "post_endpoint_missing":post.map(|row|row[endpoint].is_null()),
            "qc_excluded":false}));
        if reason.is_none() {
            let pre = pre.context("eligible pair requires pre observation")?;
            let post = post.context("eligible pair requires post observation")?;
            let mut result = post.clone();
            let difference = number(post, endpoint)? - number(pre, endpoint)?;
            ensure!(
                difference.is_finite(),
                "Paired endpoint difference overflow"
            );
            result["pre_outcome"] = pre[endpoint].clone();
            result["paired_difference"] = difference.into();
            if key[1] == "ymaze" && endpoint == "spontaneous_alternation_pct" {
                result["DeltaSA"] = difference.into();
            }
            result["pre_session_id"] = pre["session_id"].clone();
            result["pre_latency_right_censored"] =
                json!(optional_flag(pre, "latency_right_censored")?);
            paired.push(result);
        }
    }
    Ok((paired, ledger))
}
