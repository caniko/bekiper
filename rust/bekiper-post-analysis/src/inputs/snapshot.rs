//! Decode and fingerprint the same opened file, detecting in-place changes
//! around parsing. Replacing a pathname cannot switch the decoder to new bytes.
use std::{
    fs::File,
    io::{Seek, SeekFrom},
    path::Path,
};

use anyhow::{Context, Result, ensure};

use crate::Record;

pub(super) fn fingerprint_file(file: &mut File, path: &Path) -> Result<Record> {
    file.seek(SeekFrom::Start(0))?;
    Ok(crate::fingerprint::reader(file, "input size overflow")?
        .named(path.file_name().unwrap_or_default().to_string_lossy()))
}

pub(super) fn read_pinned<T>(
    path: &Path,
    decode: impl FnOnce(&mut File) -> Result<T>,
) -> Result<(T, Record)> {
    let mut file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let before = fingerprint_file(&mut file, path)?;
    file.seek(SeekFrom::Start(0))?;
    let value = decode(&mut file)?;
    let after = fingerprint_file(&mut file, path)?;
    ensure!(
        before == after,
        "Input changed during decoding: {}",
        path.display()
    );
    Ok((value, before))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use sha2::{Digest, Sha256};

    #[test]
    fn changing_the_open_file_during_decode_invalidates_the_fingerprint() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("input.json");
        std::fs::write(&path, b"[1]").unwrap();
        let result = read_pinned(&path, |file| {
            let parsed: serde_json::Value = serde_json::from_reader(file)?;
            std::fs::write(&path, b"[2]")?;
            Ok(parsed)
        });
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("changed during decoding")
        );
    }

    #[test]
    fn replacing_a_path_cannot_switch_the_opened_decoder_to_new_bytes() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("input.json");
        let replacement = directory.path().join("replacement.json");
        std::fs::write(&path, b"[1]").unwrap();
        std::fs::write(&replacement, b"[2]").unwrap();
        let (parsed, fingerprint) = read_pinned(&path, |file| {
            std::fs::rename(&replacement, &path)?;
            Ok(serde_json::from_reader::<_, serde_json::Value>(file)?)
        })
        .unwrap();
        assert_eq!(parsed, json!([1]));
        assert_eq!(
            fingerprint["sha256"],
            format!("{:x}", Sha256::digest(b"[1]"))
        );
    }
}
