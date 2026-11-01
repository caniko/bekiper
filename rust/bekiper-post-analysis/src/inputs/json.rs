//! Strict JSON ingestion: duplicate object keys must never shadow evidence or
//! authoritative metadata. Missing numbers remain null and nonfinite values fail.
use std::{fmt, io::Read};

use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Number, Value};

struct StrictValue(Value);

impl<'de> Deserialize<'de> for StrictValue {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(StrictVisitor).map(Self)
    }
}

struct StrictVisitor;

impl<'de> Visitor<'de> for StrictVisitor {
    type Value = Value;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("JSON with unique object keys and finite numbers")
    }

    fn visit_bool<E: de::Error>(self, value: bool) -> Result<Value, E> {
        Ok(Value::Bool(value))
    }

    fn visit_i64<E: de::Error>(self, value: i64) -> Result<Value, E> {
        Ok(Value::Number(value.into()))
    }

    fn visit_u64<E: de::Error>(self, value: u64) -> Result<Value, E> {
        Ok(Value::Number(value.into()))
    }

    fn visit_f64<E: de::Error>(self, value: f64) -> Result<Value, E> {
        Number::from_f64(value)
            .map(Value::Number)
            .ok_or_else(|| E::custom("nonfinite supplied JSON value"))
    }

    fn visit_str<E: de::Error>(self, value: &str) -> Result<Value, E> {
        Ok(Value::String(value.to_owned()))
    }

    fn visit_string<E: de::Error>(self, value: String) -> Result<Value, E> {
        Ok(Value::String(value))
    }

    fn visit_unit<E: de::Error>(self) -> Result<Value, E> {
        Ok(Value::Null)
    }

    fn visit_none<E: de::Error>(self) -> Result<Value, E> {
        Ok(Value::Null)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<Value, A::Error> {
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element::<StrictValue>()? {
            values.push(value.0);
        }
        Ok(Value::Array(values))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut object: A) -> Result<Value, A::Error> {
        let mut values = Map::new();
        while let Some(key) = object.next_key::<String>()? {
            if values.contains_key(&key) {
                return Err(de::Error::custom(format!("duplicate JSON key: {key}")));
            }
            values.insert(key, object.next_value::<StrictValue>()?.0);
        }
        Ok(Value::Object(values))
    }
}

pub(super) fn from_reader(reader: impl Read) -> serde_json::Result<Value> {
    serde_json::from_reader::<_, StrictValue>(reader).map(|value| value.0)
}

pub(super) fn from_str(text: &str) -> serde_json::Result<Value> {
    serde_json::from_str::<StrictValue>(text).map(|value| value.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::de::{IntoDeserializer, value::Error};

    struct NoneDeserializer;

    impl<'de> Deserializer<'de> for NoneDeserializer {
        type Error = Error;

        fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
            visitor.visit_none()
        }

        serde::forward_to_deserialize_any! {
            bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string bytes
            byte_buf option unit unit_struct newtype_struct seq tuple tuple_struct map
            struct enum identifier ignored_any
        }
    }

    // serde_json only drives the borrowed-string and unit visitors; other
    // deserializers must still map owned strings and none exactly or fail typed.
    #[test]
    fn owned_strings_none_and_unsupported_kinds_are_strict() {
        let owned = StrictValue::deserialize(IntoDeserializer::<Error>::into_deserializer(
            String::from("WT"),
        ))
        .unwrap();
        assert_eq!(owned.0, Value::String("WT".into()));
        let none = StrictValue::deserialize(NoneDeserializer).unwrap();
        assert_eq!(none.0, Value::Null);
        let bytes = serde::de::value::BytesDeserializer::<Error>::new(b"raw");
        let error = StrictValue::deserialize(bytes).err().unwrap().to_string();
        assert!(error.contains("JSON with unique object keys and finite numbers"));
    }
}
