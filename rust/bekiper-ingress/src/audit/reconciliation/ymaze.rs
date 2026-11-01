use super::*;

pub(super) struct Mappings {
    animals: BTreeMap<u32, Record>,
    geometry: BTreeMap<(String, u32, u32), Record>,
}

pub(super) fn load(path: &Path) -> anyhow::Result<Mappings> {
    let mut animals = BTreeMap::new();
    for row in inputs::read_ods_source(path, "animal")? {
        let id = number(&row["Animal"], "ODS Animal")?;
        ensure!(
            animals.insert(id, row).is_none(),
            "duplicate Y-maze animal mapping: {id}"
        );
    }
    let mut geometry = BTreeMap::new();
    for cohort in ["A", "B"] {
        let mut part = None;
        for row in inputs::read_ods_source(path, cohort)? {
            if !row["Part"].is_null() {
                part = Some(number(&row["Part"], "ODS Part")?);
            }
            let trial = row
                .get("TrialNumber")
                .filter(|v| !v.is_null())
                .or_else(|| row.get("Number").filter(|v| !v.is_null()));
            let Some(trial) = trial else { continue };
            let key = (
                cohort.into(),
                part.context("Y-maze geometry needs explicit Part")?,
                number(trial, "ODS trial")?,
            );
            ensure!(
                geometry.insert(key, row).is_none(),
                "duplicate Y-maze geometry mapping"
            );
        }
    }
    Ok(Mappings { animals, geometry })
}

impl Mappings {
    pub(super) fn resolve(&self, animal: Option<u32>, part: Option<u32>) -> anyhow::Result<Record> {
        let Some(row) = animal.and_then(|id| self.animals.get(&id)) else {
            return Ok(json!({"status":"animal_mapping_missing"}));
        };
        let Some(part @ (1 | 2)) = part else {
            return Ok(json!({"status":"phase_part_unresolved"}));
        };
        let cohort = text(&row["Phase"], "Y-maze cohort Phase")?;
        let trial = number(
            &row[format!("PhasePart-{part}-TrialNumber")],
            "Y-maze trial",
        )?;
        let geometry = self.geometry.get(&(cohort.into(), part, trial));
        Ok(
            json!({"animal_sheet_row":row["source_row"],"phase":cohort,"phase_part":part,"trial_number":trial,
            "geometry_sheet_row":geometry.map(|r|&r["source_row"]),"geometry_source_fields":geometry,
            "status":if geometry.is_some() {"explicit_mapping_present_geometry_review_unverified"} else {"geometry_row_missing"}}),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn animal_and_cohort_bind_the_explicit_trial_instead_of_the_video_number() {
        let mappings = Mappings {
            animals: BTreeMap::from([(
                67,
                json!({"Animal":67,"Phase":"B","PhasePart-1-TrialNumber":9,"PhasePart-2-TrialNumber":4,"source_row":68}),
            )]),
            geometry: BTreeMap::from([(
                ("B".into(), 2, 4),
                json!({"source_row":42,"Part":2,"TrialNumber":4,"ChangeReference":"arena-B"}),
            )]),
        };
        let mapped = mappings.resolve(Some(67), Some(2)).unwrap();
        assert_eq!(mapped["trial_number"], 4);
        assert_eq!(mapped["animal_sheet_row"], 68);
        assert_eq!(mapped["geometry_sheet_row"], 42);
        assert_eq!(
            mapped["geometry_source_fields"]["ChangeReference"],
            "arena-B"
        );
        assert_eq!(
            mappings.resolve(Some(67), Some(1)).unwrap()["status"],
            "geometry_row_missing"
        );
        assert_eq!(
            mappings.resolve(Some(66), Some(2)).unwrap()["status"],
            "animal_mapping_missing"
        );
        assert_eq!(
            mappings.resolve(Some(67), Some(3)).unwrap()["status"],
            "phase_part_unresolved"
        );
    }
}
