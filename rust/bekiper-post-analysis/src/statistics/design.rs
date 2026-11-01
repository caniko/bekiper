//! Full factorial design and contrast identities.
use super::*;

pub fn validate_contrast_mapping(groups: &Record, roles: &Record) -> Result<()> {
    for role in ["ad_level", "hcar1_wt", "hcar1_ko", "lactate", "vehicle"] {
        identity(roles, role).context("Explicit factor semantics for planned contrast required")?;
    }
    ensure!(
        roles["hcar1_wt"] != roles["hcar1_ko"] && roles["lactate"] != roles["vehicle"],
        "Planned contrast roles require distinct HCAR1 and treatment levels"
    );
    for (group, hcar, treatment) in [
        ("G5", "hcar1_wt", "lactate"),
        ("G6", "hcar1_wt", "vehicle"),
        ("G1", "hcar1_ko", "lactate"),
        ("G2", "hcar1_ko", "vehicle"),
    ] {
        ensure!(
            groups[group].is_object()
                && groups[group]["five_xfad"] == roles["ad_level"]
                && groups[group]["hcar1"] == roles[hcar]
                && groups[group]["treatment"] == roles[treatment],
            "Planned contrast role mismatch for {group}"
        );
    }
    Ok(())
}

pub(super) fn categorical_levels(levels: &Record, name: &str) -> Result<Vec<String>> {
    let values = levels[name]
        .as_array()
        .with_context(|| format!("Register ordered categorical levels/reference for {name}"))?;
    ensure!(
        values.len() >= 2,
        "Categorical {name} requires at least two registered levels"
    );
    let mut seen = BTreeSet::new();
    values
        .iter()
        .map(|value| {
            let label = value
                .as_str()
                .filter(|label| !label.is_empty())
                .context("Categorical levels require nonempty strings")?;
            ensure!(seen.insert(label), "Duplicate categorical level for {name}");
            Ok(label.to_owned())
        })
        .collect()
}

#[derive(Clone)]
pub(super) enum Term {
    Intercept,
    Interaction(Vec<(String, String)>),
    Numeric(String),
    Categorical(String, String),
}

pub(super) fn model_terms(
    levels: &Record,
    covariates: &[String],
) -> Result<(Vec<Term>, Vec<String>)> {
    let factors: Vec<_> = FACTORS
        .iter()
        .map(|factor| categorical_levels(levels, factor))
        .collect::<Result<_>>()?;
    let mut terms = vec![Term::Intercept];
    let mut names = vec!["Intercept".into()];
    for order in 1..=3 {
        for mask in 1_u32..8 {
            if mask.count_ones() != order {
                continue;
            }
            let mut products = vec![Vec::new()];
            for (index, factor) in FACTORS.iter().enumerate() {
                if mask & (1 << index) == 0 {
                    continue;
                }
                products = products
                    .into_iter()
                    .flat_map(|product| {
                        factors[index][1..].iter().map(move |level| {
                            let mut term = product.clone();
                            term.push(((*factor).to_owned(), level.clone()));
                            term
                        })
                    })
                    .collect();
            }
            for product in products {
                names.push(
                    product
                        .iter()
                        .map(|(factor, level)| format!("{factor}[{level}]"))
                        .collect::<Vec<_>>()
                        .join(":"),
                );
                terms.push(Term::Interaction(product));
            }
        }
    }
    let mut seen = BTreeSet::new();
    for covariate in covariates {
        ensure!(
            !covariate.is_empty()
                && !FACTORS.contains(&covariate.as_str())
                && covariate != "Intercept"
                && seen.insert(covariate),
            "Covariates must be unique additional fields"
        );
        if levels.get(covariate).is_some() {
            for level in categorical_levels(levels, covariate)?.into_iter().skip(1) {
                names.push(format!("{covariate}[{level}]"));
                terms.push(Term::Categorical(covariate.clone(), level));
            }
        } else {
            names.push(covariate.clone());
            terms.push(Term::Numeric(covariate.clone()));
        }
    }
    Ok((terms, names))
}

/// Full interactions are built from every ordered categorical level. Extra
/// covariates are numeric unless their ordered categorical levels are explicit.
pub fn design(
    rows: &[Record],
    levels: &Record,
    covariates: &[String],
) -> Result<(DMatrix<f64>, Vec<String>)> {
    let (terms, names) = model_terms(levels, covariates)?;
    for row in rows {
        for factor in FACTORS.iter().copied().chain(
            covariates
                .iter()
                .filter(|name| levels.get(*name).is_some())
                .map(String::as_str),
        ) {
            let declared = categorical_levels(levels, factor)?;
            ensure!(
                declared.iter().any(|level| row[factor] == *level),
                "Unknown registered categorical level for {factor}"
            );
        }
    }
    let mut values = Vec::with_capacity(rows.len() * terms.len());
    for row in rows {
        for term in &terms {
            values.push(match term {
                Term::Intercept => 1.0,
                Term::Interaction(parts) => {
                    f64::from(parts.iter().all(|(factor, level)| row[factor] == *level))
                }
                Term::Numeric(name) => number(row, name)?,
                Term::Categorical(name, level) => f64::from(row[name] == *level),
            });
        }
    }
    Ok((
        DMatrix::from_row_slice(rows.len(), terms.len(), &values),
        names,
    ))
}
