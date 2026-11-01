//! SHA256SUMS inventory, zip delivery and verification.
use anyhow::{Context, Result, bail, ensure};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    io::{Read, Write},
    path::{Component, Path, PathBuf},
};
use zip::{CompressionMethod, ZipArchive, ZipWriter, write::SimpleFileOptions};

pub(crate) const SUMS: &str = "SHA256SUMS";

/// Package files excluding operational logs and the inventory itself.
fn members(root: &Path) -> Result<Vec<String>> {
    fn walk(root: &Path, dir: &Path, out: &mut Vec<String>) -> Result<()> {
        for entry in fs::read_dir(dir)? {
            let path = entry?.path();
            let relative = path
                .strip_prefix(root)?
                .to_string_lossy()
                .replace('\\', "/");
            if relative == SUMS || relative == "logs" || relative == "review_state.json" {
                continue;
            }
            if path.is_dir() {
                walk(root, &path, out)?;
            } else if !relative.ends_with(".tmp.webp") {
                out.push(relative);
            }
        }
        Ok(())
    }
    let mut out = Vec::new();
    walk(root, root, &mut out)?;
    out.sort();
    Ok(out)
}

fn digest(reader: &mut impl Read) -> Result<String> {
    let mut hasher = Sha256::new();
    std::io::copy(reader, &mut hasher)?;
    Ok(format!("{:x}", hasher.finalize()))
}

/// Write `SHA256SUMS` (`sha256sum -c` format) and return the member count.
pub(crate) fn write_sums(root: &Path) -> Result<usize> {
    let mut text = String::new();
    let files = members(root)?;
    for name in &files {
        text.push_str(&format!(
            "{}  {name}\n",
            digest(&mut fs::File::open(root.join(name))?)?
        ));
    }
    fs::write(root.join(SUMS), text)?;
    Ok(files.len())
}

fn parse_sums(text: &str) -> Result<BTreeMap<String, String>> {
    let mut sums = BTreeMap::new();
    for line in text.lines() {
        let (hash, name) = line.split_once("  ").context("malformed SHA256SUMS line")?;
        ensure!(safe(name), "unsafe path in SHA256SUMS: {name}");
        sums.insert(name.to_string(), hash.to_string());
    }
    Ok(sums)
}

/// Relative, normal components only (no absolute paths or `..`).
fn safe(name: &str) -> bool {
    let path = Path::new(name);
    !name.is_empty() && path.components().all(|c| matches!(c, Component::Normal(_)))
}

pub(crate) fn verify_dir(root: &Path) -> Result<usize> {
    let sums = parse_sums(&fs::read_to_string(root.join(SUMS))?)?;
    let files = members(root)?;
    ensure!(
        files.iter().eq(sums.keys()),
        "package members differ from SHA256SUMS"
    );
    for (name, hash) in &sums {
        let actual = digest(&mut fs::File::open(root.join(name))?)?;
        if &actual != hash {
            bail!("checksum mismatch: {name}");
        }
    }
    Ok(sums.len())
}

fn stored(name: &str) -> bool {
    [".webp", ".png", ".pdf", ".ods", ".zip"]
        .iter()
        .any(|e| name.ends_with(e))
}

/// Write `<root>.zip` (images and PDFs stored, text deflated) plus its `.sha256`.
pub(crate) fn write_zip(root: &Path) -> Result<PathBuf> {
    let prefix = root
        .file_name()
        .context("package name")?
        .to_string_lossy()
        .into_owned();
    let zip_path = root.with_extension("zip");
    let tmp = root.with_extension("zip.tmp");
    let mut zip = ZipWriter::new(fs::File::create(&tmp)?);
    let mut names = members(root)?;
    names.push(SUMS.into());
    for name in &names {
        let method = if stored(name) {
            CompressionMethod::Stored
        } else {
            CompressionMethod::Deflated
        };
        let options = SimpleFileOptions::default()
            .compression_method(method)
            .large_file(fs::metadata(root.join(name))?.len() >= u32::MAX as u64);
        zip.start_file(format!("{prefix}/{name}"), options)?;
        std::io::copy(&mut fs::File::open(root.join(name))?, &mut zip)?;
    }
    zip.finish()?.flush()?;
    fs::rename(&tmp, &zip_path)?;
    let hash = digest(&mut fs::File::open(&zip_path)?)?;
    fs::write(
        zip_path.with_extension("zip.sha256"),
        format!(
            "{hash}  {}\n",
            zip_path.file_name().context("zip name")?.to_string_lossy()
        ),
    )?;
    Ok(zip_path)
}

/// Every zip member must be listed in its SHA256SUMS with a matching hash.
pub(crate) fn verify_zip(path: &Path) -> Result<usize> {
    let mut archive = ZipArchive::new(fs::File::open(path)?)?;
    let prefix = {
        let first = archive.by_index(0)?.name().to_string();
        first.split('/').next().context("zip prefix")?.to_string()
    };
    let mut text = String::new();
    archive
        .by_name(&format!("{prefix}/{SUMS}"))
        .context("zip has no SHA256SUMS")?
        .read_to_string(&mut text)?;
    let sums = parse_sums(&text)?;
    ensure!(
        archive.len() == sums.len() + 1,
        "zip members differ from SHA256SUMS"
    );
    for (name, hash) in &sums {
        let mut entry = archive
            .by_name(&format!("{prefix}/{name}"))
            .with_context(|| format!("zip is missing {name}"))?;
        if &digest(&mut entry)? != hash {
            bail!("checksum mismatch in zip: {name}");
        }
    }
    Ok(sums.len())
}
