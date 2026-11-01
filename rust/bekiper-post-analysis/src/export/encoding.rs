//! Lossless JSON-to-TOML representation, including explicit null and unsigned
//! integer locations. Field classification and package layout are independent.
use std::{fs, path::Path};

use anyhow::{Context, Result};
use serde::Serialize;
use serde_json::Value;

pub(super) fn pointer(parent: &str, key: &str) -> String {
    format!("{parent}/{}", key.replace('~', "~0").replace('/', "~1"))
}

/// TOML cannot represent JSON null or unsigned integers above i64::MAX. Their
/// exact locations are declared as JSON pointers, allowing lossless recovery.
pub(super) fn encode(
    value: &Value,
    location: &str,
    null_state: &str,
    nulls: &mut Vec<String>,
    integers: &mut Vec<String>,
) -> Result<toml::Value> {
    Ok(match value {
        Value::Null => {
            nulls.push(location.into());
            toml::Value::Table(toml::Table::from_iter([(
                "state".into(),
                toml::Value::String(null_state.into()),
            )]))
        }
        Value::Bool(value) => toml::Value::Boolean(*value),
        Value::String(value) => toml::Value::String(value.clone()),
        Value::Number(value) => {
            if let Some(value) = value.as_i64() {
                toml::Value::Integer(value)
            } else if value.as_u64().is_some() {
                integers.push(location.into());
                toml::Value::String(value.to_string())
            } else {
                toml::Value::Float(value.as_f64().context("metric number must be finite")?)
            }
        }
        Value::Array(values) => toml::Value::Array(
            values
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    encode(
                        value,
                        &pointer(location, &index.to_string()),
                        null_state,
                        nulls,
                        integers,
                    )
                })
                .collect::<Result<_>>()?,
        ),
        Value::Object(values) => toml::Value::Table(
            values
                .iter()
                .map(|(key, value)| {
                    Ok((
                        key.clone(),
                        encode(value, &pointer(location, key), null_state, nulls, integers)?,
                    ))
                })
                .collect::<Result<_>>()?,
        ),
    })
}

pub(super) fn write_toml(path: &Path, description: &str, value: &impl Serialize) -> Result<()> {
    let text = format!(
        "# {description}\n# Nulls are explicit state tables. Session null_fields (parameters: representation.null_fields) give exact JSON-pointer locations.\n# large_integer_fields identifies exact decimal strings beyond TOML's signed integer range.\n\n{}",
        toml::to_string_pretty(value)?
    );
    fs::write(path, text)?;
    Ok(())
}
