//! The build and installed verifier use the same component-ordered source digest.
use sha2::{Digest, Sha256};
use std::{
    fs, io,
    path::{Path, PathBuf},
};

pub(crate) fn digest(crate_root: &Path, source_lock: &[u8]) -> io::Result<String> {
    let mut files = Vec::new();
    collect(crate_root, Path::new("src"), &mut files)?;
    for dependency in ["bekiper-math", "bekiper-feature"] {
        let directory = PathBuf::from(format!("../{dependency}"));
        collect(crate_root, &directory.join("src"), &mut files)?;
        files.push(directory.join("Cargo.toml"));
    }
    files.extend([
        "Cargo.toml".into(),
        "build.rs".into(),
        "../../Cargo.lock".into(),
    ]);
    files.sort();
    let mut digest = Sha256::new();
    for file in files {
        digest.update(file.to_string_lossy().as_bytes());
        digest.update([0]);
        if file == Path::new("../../Cargo.lock") {
            digest.update(source_lock);
        } else {
            digest.update(fs::read(crate_root.join(file))?);
        }
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn collect(root: &Path, directory: &Path, files: &mut Vec<PathBuf>) -> io::Result<()> {
    for entry in fs::read_dir(root.join(directory))? {
        let entry = entry?;
        let path = directory.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            collect(root, &path, files)?;
        } else if path.extension().is_some_and(|e| e == "rs") {
            files.push(path);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_binds_engine_sources_and_the_supplied_lock() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let first = digest(root, b"lock-a").unwrap();
        assert_eq!(first.len(), 64);
        assert_eq!(first, digest(root, b"lock-a").unwrap());
        assert_ne!(first, digest(root, b"lock-b").unwrap());
        assert!(digest(&root.join("missing"), b"lock-a").is_err());
    }
}
