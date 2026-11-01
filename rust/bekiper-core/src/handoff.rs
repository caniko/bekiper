//! Deterministic TOML handoffs. Absent optional fields are omitted; integer and
//! array values must be representable without rounding or inventing missing data.
use crate::error::{BekiperError, Result};
use serde::{Serialize, de::DeserializeOwned};

pub fn to_string<T: Serialize + ?Sized>(value: &T) -> Result<String> {
    let value = serde_json::to_value(value).map_err(|e| BekiperError::Data(e.to_string()))?;
    toml::to_string_pretty(&convert(&value)?).map_err(|e| BekiperError::Data(e.to_string()))
}

pub fn from_str<T: DeserializeOwned>(text: &str) -> Result<T> {
    let value: toml::Value = toml::from_str(text).map_err(|e| BekiperError::Data(e.to_string()))?;
    let value = serde_json::to_value(value).map_err(|e| BekiperError::Data(e.to_string()))?;
    serde_json::from_value(value).map_err(|e| BekiperError::Data(e.to_string()))
}

fn convert(value: &serde_json::Value) -> Result<toml::Value> {
    use serde_json::Value;
    Ok(match value {
        Value::Null => {
            return Err(BekiperError::Data(
                "null array/root values cannot be represented in a TOML handoff".into(),
            ));
        }
        Value::Bool(value) => toml::Value::Boolean(*value),
        Value::String(value) => toml::Value::String(value.clone()),
        Value::Number(value) => {
            if let Some(integer) = value.as_i64() {
                toml::Value::Integer(integer)
            } else if value.is_f64() {
                toml::Value::Float(
                    value
                        .as_f64()
                        .ok_or_else(|| BekiperError::Data("invalid handoff float".into()))?,
                )
            } else {
                return Err(BekiperError::Data("handoff integer exceeds TOML's signed 64-bit range; encode wide identifiers as strings".into()));
            }
        }
        Value::Array(values) => {
            toml::Value::Array(values.iter().map(convert).collect::<Result<_>>()?)
        }
        Value::Object(values) => {
            let mut table = toml::map::Map::new();
            // Sort regardless of serde_json's preserve_order feature.
            let mut keys: Vec<_> = values.keys().collect();
            keys.sort();
            for key in keys {
                if !values[key].is_null() {
                    table.insert(key.clone(), convert(&values[key])?);
                }
            }
            toml::Value::Table(table)
        }
    })
}

/// Small captured binary assets use hexadecimal strings rather than one boxed
/// TOML/serde value per byte. Legacy byte arrays remain readable.
pub mod bytes_hex {
    use serde::{Deserialize, Deserializer, Serializer, de::Error};

    pub fn serialize<S: Serializer>(bytes: &[u8], serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&encode(bytes))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<u8>, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Encoded {
            Hex(String),
            Legacy(Vec<u8>),
        }
        match Encoded::deserialize(deserializer)? {
            Encoded::Legacy(bytes) => Ok(bytes),
            Encoded::Hex(text) => decode(&text).map_err(D::Error::custom),
        }
    }

    pub(super) fn encode(bytes: &[u8]) -> String {
        const DIGITS: &[u8] = b"0123456789abcdef";
        let mut text = String::with_capacity(bytes.len() * 2);
        for &byte in bytes {
            text.push(DIGITS[(byte >> 4) as usize] as char);
            text.push(DIGITS[(byte & 15) as usize] as char);
        }
        text
    }

    fn decode(text: &str) -> Result<Vec<u8>, &'static str> {
        if !text.len().is_multiple_of(2) || !text.is_ascii() {
            return Err("invalid captured hexadecimal bytes");
        }
        text.as_bytes()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| {
                fn digit(byte: u8) -> Result<u8, &'static str> {
                    match byte {
                        b'0'..=b'9' => Ok(byte - b'0'),
                        b'a'..=b'f' => Ok(byte - b'a' + 10),
                        _ => Err("invalid captured hexadecimal bytes"),
                    }
                }
                Ok(digit(pair[0])? * 16 + digit(pair[1])?)
            })
            .collect()
    }
}

pub mod optional_bytes_hex {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(
        bytes: &Option<Vec<u8>>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        match bytes {
            Some(bytes) => serializer.serialize_some(&super::bytes_hex::encode(bytes)),
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<Vec<u8>>, D::Error> {
        #[derive(Deserialize)]
        struct Bytes(#[serde(with = "super::bytes_hex")] Vec<u8>);
        Ok(Option::<Bytes>::deserialize(deserializer)?.map(|bytes| bytes.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_wide_string_ids_and_missing_fields_without_rounding() {
        let value = serde_json::json!({"receipt_id":"ffffffffffffffffffffffffffffffff", "optional":null, "trials":[{"analysis_id":"a", "seconds":0.5}]});
        let text = to_string(&value).unwrap();
        assert!(text.contains("[[trials]]"));
        let restored: serde_json::Value = from_str(&text).unwrap();
        assert_eq!(restored["receipt_id"], value["receipt_id"]);
        assert_eq!(restored["trials"], value["trials"]);
        assert!(to_string(&serde_json::json!({"bytes":u64::MAX})).is_err());
        assert!(to_string(&serde_json::json!({"points":[null]})).is_err());
    }
}
