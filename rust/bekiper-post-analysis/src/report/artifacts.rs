//! Report input records, immutable writes and artifact verification.
use super::*;

pub(super) fn generator_source_hash() -> String {
    let mut hash = Sha256::new();
    for (name, source) in [
        ("report.rs", include_str!("../report.rs")),
        ("report/artifacts.rs", include_str!("artifacts.rs")),
        ("report/checks.rs", include_str!("checks.rs")),
        ("report/compiler.rs", include_str!("compiler.rs")),
        ("report/figures.rs", include_str!("figures.rs")),
        ("report/format.rs", include_str!("format.rs")),
        ("report/inclusion.rs", include_str!("inclusion.rs")),
        ("report/markdown.rs", include_str!("markdown.rs")),
        ("report/models.rs", include_str!("models.rs")),
        ("report/provenance.rs", include_str!("provenance.rs")),
        ("report/scope.rs", include_str!("scope.rs")),
        ("report/supplementary.rs", include_str!("supplementary.rs")),
        ("report/tex.rs", include_str!("tex.rs")),
    ] {
        hash.update(name.as_bytes());
        hash.update([0]);
        hash.update(source.as_bytes());
    }
    format!("{:x}", hash.finalize())
}

pub(super) fn config_input_hash(manifest: &Value) -> &Value {
    let historical = &manifest["config_fingerprint"]["sha256"];
    if !historical.is_null() {
        return historical;
    }
    manifest["inputs"]
        .as_array()
        .and_then(|inputs| {
            inputs
                .iter()
                .find(|input| input["role"] == "post_analysis_plan")
        })
        .map(|input| &input["sha256"])
        .unwrap_or(&Value::Null)
}

pub(super) fn basename(value: &str) -> String {
    Path::new(value)
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "Unavailable".into())
}

pub(super) fn read_rows(directory: &Path, name: &str) -> Result<Vec<Record>> {
    let path = crate::export::input_path(directory, name);
    if !path.try_exists()? {
        return Ok(Vec::new());
    }
    let value: Value = serde_json::from_slice(&fs::read(path)?)
        .with_context(|| format!("invalid report input {name}"))?;
    value
        .as_array()
        .cloned()
        .with_context(|| format!("report input {name} must be an array"))
}

pub(super) fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .with_context(|| {
            format!(
                "could not create fresh report artifact {}",
                basename(&path.to_string_lossy())
            )
        })?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

pub(super) fn validate_pdf(path: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("missing PDF {}", basename(&path.to_string_lossy())))?;
    ensure!(
        metadata.is_file() && !metadata.file_type().is_symlink(),
        "PDF must be a regular, non-symlink file"
    );
    let mut signature = [0; 5];
    fs::File::open(path)?.read_exact(&mut signature)?;
    ensure!(
        &signature == b"%PDF-" && metadata.len() > 64,
        "compiler/renderer did not produce a PDF"
    );
    Ok(())
}

pub(super) fn fingerprint(path: &Path) -> Result<Value> {
    Ok(
        crate::fingerprint::reader(&mut fs::File::open(path)?, "report artifact size overflow")?
            .named(basename(&path.to_string_lossy())),
    )
}

pub(super) fn record_fingerprint(role: &str, value: &Value) -> Result<Value> {
    let bytes = serde_json::to_vec(value)?;
    Ok(
        json!({"role":role,"encoding":"serde_json_serialization_of_in_memory_input",
        "bytes":bytes.len(),"sha256":format!("{:x}",Sha256::digest(&bytes))}),
    )
}
