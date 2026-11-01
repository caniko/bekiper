//! Row-space checks use the same scaled design and retained singular components.
use super::*;

pub(super) fn estimable(
    vector: &DVector<f64>,
    scales: &[f64],
    v_t: &DMatrix<f64>,
    retained: &[usize],
    n: usize,
) -> bool {
    if retained.len() == vector.len() {
        return true;
    }
    let scaled =
        DVector::from_iterator(vector.len(), vector.iter().zip(scales).map(|(c, s)| c / s));
    let mut remainder = scaled.clone();
    for &k in retained {
        let component = v_t.row(k).transpose();
        remainder -= &component * component.dot(&scaled);
    }
    remainder.norm() <= 64.0 * f64::EPSILON * n.max(vector.len()) as f64 * scaled.norm()
}

pub(super) fn cell_counts(rows: &[Record], levels: &Record) -> Result<Record> {
    let mut cells = Vec::new();
    for f in categorical_levels(levels, "five_xfad")? {
        for h in categorical_levels(levels, "hcar1")? {
            for t in categorical_levels(levels, "treatment")? {
                let animals: Vec<_> = rows
                    .iter()
                    .filter(|r| r["five_xfad"] == f && r["hcar1"] == h && r["treatment"] == t)
                    .map(|r| r["animal_id"].clone())
                    .collect();
                cells.push(json!({"five_xfad":f,"hcar1":h,"treatment":t,"n_animals":animals.len(),"animal_ids":animals}));
            }
        }
    }
    Ok(json!(cells))
}
