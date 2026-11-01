//! Pinned JSON, CSV and Parquet table decoding.
use super::*;

pub fn read_table(path: &Path, sheet: Option<&str>) -> Result<Vec<Record>> {
    read_table_pinned(path, sheet).map(|(rows, _)| rows)
}

pub(super) fn read_table_pinned(path: &Path, sheet: Option<&str>) -> Result<(Vec<Record>, Record)> {
    snapshot::read_pinned(path, |file| decode_table(file, path, sheet))
}

pub(super) fn decode_table(
    file: &mut File,
    path: &Path,
    sheet: Option<&str>,
) -> Result<Vec<Record>> {
    match path.extension().and_then(|s| s.to_str()) {
        Some("json") => {
            let Value::Array(rows) = json::from_reader(file)? else {
                bail!("JSON input must be an array of session objects");
            };
            ensure!(
                rows.iter().all(Value::is_object),
                "JSON input must contain session objects"
            );
            Ok(rows)
        }
        Some("csv") => {
            let mut reader = csv::Reader::from_reader(file);
            let headers = reader.headers()?.clone();
            let distinct: BTreeSet<_> = headers.iter().collect();
            ensure!(
                !headers.is_empty()
                    && distinct.len() == headers.len()
                    && headers.iter().all(|h| !h.trim().is_empty()),
                "Missing or duplicate CSV headers"
            );
            reader
                .records()
                .map(|row| {
                    let row = row?;
                    Ok(Value::Object(
                        headers
                            .iter()
                            .zip(row.iter())
                            .map(|(h, v)| (h.to_owned(), Value::String(v.to_owned())))
                            .collect(),
                    ))
                })
                .collect()
        }
        Some("parquet") => {
            let frame = ParquetReader::new(file.try_clone()?).finish()?;
            (0..frame.height())
                .map(|index| {
                    let mut row = Map::new();
                    for column in frame.columns() {
                        row.insert(
                            column.name().to_string(),
                            parquet_value(column.get(index)?)?,
                        );
                    }
                    Ok(Value::Object(row))
                })
                .collect()
        }
        Some("ods") => decode_ods(file, sheet),
        other => {
            bail!("Unsupported table extension {other:?}; export normalized metadata to CSV/JSON")
        }
    }
}

pub(super) fn parquet_value(value: AnyValue<'_>) -> Result<Value> {
    Ok(match value {
        AnyValue::Null => Value::Null,
        AnyValue::Boolean(v) => json!(v),
        AnyValue::String(v) => json!(v),
        AnyValue::StringOwned(v) => json!(v.as_str()),
        AnyValue::Int8(v) => json!(v),
        AnyValue::Int16(v) => json!(v),
        AnyValue::Int32(v) => json!(v),
        AnyValue::Int64(v) => json!(v),
        AnyValue::UInt8(v) => json!(v),
        AnyValue::UInt16(v) => json!(v),
        AnyValue::UInt32(v) => json!(v),
        AnyValue::UInt64(v) => json!(v),
        AnyValue::Float32(v) => {
            ensure!(v.is_finite(), "nonfinite supplied value");
            json!(v)
        }
        AnyValue::Float64(v) => {
            ensure!(v.is_finite(), "nonfinite supplied value");
            json!(v)
        }
        AnyValue::List(v) => Value::Array(
            (0..v.len())
                .map(|i| parquet_value(v.get(i)?))
                .collect::<Result<_>>()?,
        ),
        value @ AnyValue::Struct(..) => parquet_value(value.into_static())?,
        AnyValue::StructOwned(payload) => {
            let (values, fields) = *payload;
            let mut object = Map::new();
            for (field, value) in fields.iter().zip(values) {
                let name = field.name().to_string();
                ensure!(
                    !object.contains_key(&name),
                    "Duplicate Parquet struct field: {name}"
                );
                object.insert(name, parquet_value(value)?);
            }
            Value::Object(object)
        }
        v => bail!("Unsupported Parquet cell type: {:?}", v.dtype()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Narrow unsigned columns need polars dtype features this crate does not
    // enable; the cell mapping itself is still exact.
    #[test]
    fn narrow_unsigned_cells_map_to_json_integers() {
        assert_eq!(parquet_value(AnyValue::UInt8(8)).unwrap(), json!(8));
        assert_eq!(parquet_value(AnyValue::UInt16(16)).unwrap(), json!(16));
    }

    // Parquet decoding yields owned cells; borrowed cells from in-memory
    // columns must map identically.
    #[test]
    fn borrowed_string_and_struct_cells_map_like_owned_cells() {
        assert_eq!(parquet_value(AnyValue::String("WT")).unwrap(), json!("WT"));
        let fields = [Series::new("x".into(), [1i64])];
        let column = StructChunked::from_series("nested".into(), 1, fields.iter())
            .unwrap()
            .into_series();
        let cell = column.get(0).unwrap();
        assert!(matches!(cell, AnyValue::Struct(..)));
        assert_eq!(parquet_value(cell).unwrap(), json!({"x":1}));
    }
}
