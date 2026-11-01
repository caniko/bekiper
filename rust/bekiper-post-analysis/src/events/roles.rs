//! Exact-session NORT object identities.
use super::*;

pub(super) fn key(row: &Value) -> Result<Vec<String>> {
    crate::inputs::session_key(row)?;
    KEY.iter()
        .map(|field| Ok(text(row, field)?.to_owned()))
        .collect()
}

pub fn nort_object_roles(row: &Record, mapping: &Record) -> Result<Option<Record>> {
    ensure!(mapping.is_object(), "NORT object roles must be a mapping");
    let empty = Vec::new();
    let sessions = match mapping.get("sessions") {
        None => &empty,
        Some(value) => value
            .as_array()
            .context("NORT object roles sessions must be a list")?,
    };
    let mut indexed = BTreeMap::new();
    for assignment in sessions {
        ensure!(
            indexed.insert(key(assignment)?, assignment).is_none(),
            "Duplicate NORT object roles session assignment"
        );
    }
    let assignment = indexed.get(&key(row)?).copied();
    let roles = match assignment {
        Some(assignment) => &assignment["roles"],
        None => &mapping[text(row, "phase")?],
    };
    if roles.is_null() {
        return Ok(None);
    }
    let fields: BTreeSet<_> = if row["phase"] == "training" {
        ["TA", "TB"]
    } else {
        ["TN", "TF"]
    }
    .into_iter()
    .collect();
    let roles_map = roles
        .as_object()
        .context("NORT object roles require exact phase assignments")?;
    ensure!(
        roles_map
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>()
            == fields,
        "NORT object roles require exact, distinct nonempty phase assignments"
    );
    let objects: BTreeSet<_> = roles_map
        .values()
        .map(|v| {
            let id = v
                .as_str()
                .context("NORT object ID must be a nonempty string")?;
            ensure!(
                !id.trim().is_empty(),
                "NORT object ID must be a nonempty string"
            );
            Ok(id)
        })
        .collect::<Result<_>>()?;
    ensure!(
        objects.len() == 2,
        "NORT object roles require two distinct objects"
    );
    let configuration = &row["object_configuration"];
    if configuration.is_null() || configuration.as_str().is_some_and(|s| s.is_empty()) {
        return Ok(None);
    }
    if let Some(assignment) = assignment
        && let Some(expected) = assignment.get("object_configuration")
    {
        ensure!(
            same_value(expected, configuration),
            "NORT object roles conflict with session object configuration"
        );
    }
    if configuration.is_object() {
        let available = configuration["objects"]
            .as_object()
            .context("NORT object configuration requires objects")?;
        ensure!(
            objects.iter().all(|id| available.contains_key(*id)),
            "NORT object roles reference absent configuration objects"
        );
    }
    Ok(Some(roles.clone()))
}

/// Physical side assignments must be exact-session evidence, never phase defaults.
pub fn nort_physical_sides(row: &Record, mapping: &Record) -> Result<Option<Record>> {
    ensure!(mapping.is_object(), "NORT physical sides must be a mapping");
    let sessions = mapping
        .get("sessions")
        .map(|v| {
            v.as_array()
                .context("NORT physical sides sessions must be a list")
        })
        .transpose()?;
    let mut indexed = BTreeMap::new();
    for assignment in sessions.into_iter().flatten() {
        ensure!(
            indexed.insert(key(assignment)?, assignment).is_none(),
            "Duplicate NORT side-mapping session"
        );
    }
    let Some(assignment) = indexed.get(&key(row)?) else {
        return Ok(None);
    };
    let Some(sides) = assignment.get("physical_sides").filter(|v| !v.is_null()) else {
        return Ok(None);
    };
    ensure!(
        (sides["TA"] == "left" && sides["TB"] == "right")
            || (sides["TA"] == "right" && sides["TB"] == "left"),
        "Physical sides require distinct TA/TB left/right assignments"
    );
    if let Some(expected) = assignment.get("recording_id") {
        ensure!(
            expected == &row["recording_id"],
            "Physical side mapping recording identity conflicts"
        );
    }
    if let Some(expected) = assignment.get("object_configuration") {
        ensure!(
            same_value(expected, &row["object_configuration"]),
            "Physical side mapping object configuration conflicts"
        );
    }
    Ok(Some(sides.clone()))
}
