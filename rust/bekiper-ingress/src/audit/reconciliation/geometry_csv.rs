//! Literal geometry-binding rows. Biology is always supplied by the workbook.
use super::*;
use bekiper_annotation::VideoContext;

pub(super) struct GeometryRows(BTreeMap<(u32, u32, u32), Record>);

pub(super) fn load(path: &Path) -> anyhow::Result<GeometryRows> {
    let mut reader = csv::Reader::from_path(path)?;
    let headers = reader.headers()?.clone();
    let names: Vec<_> = headers
        .iter()
        .map(|h| h.trim_start_matches('\u{feff}'))
        .collect();
    ensure!(
        names.iter().all(|h| !h.is_empty() && *h != "source_row")
            && names.iter().collect::<BTreeSet<_>>().len() == names.len(),
        "geometry CSV has missing or duplicate headers"
    );
    let column = |name: &str| {
        names
            .iter()
            .position(|h| *h == name)
            .with_context(|| format!("geometry CSV is missing {name}"))
    };
    let animal_column = column("Animal")?;
    let stage_column = column("Stage")?;
    let part_column = column("PhasePart")?;
    let phase_column = column("Phase")?;
    let mut rows = BTreeMap::new();
    let mut part = None;
    let mut cohort = None;
    for (index, cells) in reader.records().enumerate() {
        let cells = cells?;
        let source_row = cells.position().map_or(index as u64 + 2, |p| p.line());
        if !cells[part_column].trim().is_empty() {
            part = Some((
                number(&json!(cells[part_column]), "CSV PhasePart")?,
                source_row,
            ));
        }
        if !cells[phase_column].trim().is_empty() {
            cohort = Some((cells[phase_column].trim().to_string(), source_row));
        }
        if cells[animal_column].trim().is_empty() {
            continue;
        }
        let mut row: Record = json!({});
        for (header, cell) in names.iter().zip(&cells) {
            row[*header] = cell.into();
        }
        row["source_row"] = json!(source_row);
        let (phase_part, part_source_row) =
            part.context("CSV PhasePart needs an explicit preceding value")?;
        let (phase, phase_source_row) = cohort
            .as_ref()
            .context("CSV Phase needs an explicit preceding value")?;
        row["binding_identity"] = json!({"phase_part":phase_part,"phase_part_source_row":part_source_row,"cohort_phase":phase,"cohort_phase_source_row":phase_source_row});
        let key = (
            number(&json!(cells[animal_column]), "CSV Animal")?,
            number(&json!(cells[stage_column]), "CSV Stage")?,
            phase_part,
        );
        ensure!(
            rows.insert(key, row).is_none(),
            "duplicate geometry CSV animal/stage/part: {key:?}"
        );
    }
    Ok(GeometryRows(rows))
}

impl GeometryRows {
    pub(super) fn resolve(
        &self,
        animal: Option<u32>,
        video: Option<&VideoContext>,
    ) -> anyhow::Result<Record> {
        let Some((animal, video)) = animal.zip(video) else {
            return Ok(json!({"status":"identity_unresolved"}));
        };
        let row = self.0.get(&(animal, video.stage, video.phase_part));
        Ok(
            json!({"status":if row.is_some() {"source_binding_present_geometry_review_unverified"} else {"recording_binding_missing"},
            "source_fields":row,"animal":animal,"stage":video.stage,"phase_part":video.phase_part}),
        )
    }
}
