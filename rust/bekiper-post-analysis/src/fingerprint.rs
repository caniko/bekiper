//! Shared byte accounting for opened inputs, published artifacts and bounded
//! archive members. Callers own seeking, file identity and their record schema.
use std::io::Read;

use anyhow::{Context, Result};
use serde_json::json;
use sha2::{Digest, Sha256};

use crate::Record;

pub(crate) struct Fingerprint {
    pub bytes: u64,
    pub sha256: String,
}

impl Fingerprint {
    pub fn named(self, name: impl AsRef<str>) -> Record {
        json!({"name":name.as_ref(),"bytes":self.bytes,"sha256":self.sha256})
    }
}

pub(crate) fn reader(reader: &mut impl Read, overflow_context: &str) -> Result<Fingerprint> {
    let mut hash = Sha256::new();
    let mut bytes = 0u64;
    let mut buffer = [0u8; 65_536];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        bytes = bytes
            .checked_add(count as u64)
            .with_context(|| overflow_context.to_owned())?;
        hash.update(&buffer[..count]);
    }
    Ok(Fingerprint {
        bytes,
        sha256: format!("{:x}", hash.finalize()),
    })
}
