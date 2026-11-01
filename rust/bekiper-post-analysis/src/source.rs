//! Package-embedded scientific references and build identity work outside Git.
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

pub const SOURCE_RECIPE: &str =
    include_str!("../../../behavioral_post_analysis_harness/reference/BEHAVIORAL_RECIPE.md");
pub const YMAZE_AMENDMENT: &str =
    include_str!("../../../behavioral_post_analysis_harness/reference/YMAZE_AMENDMENT.json");

pub fn embedded_fingerprint(name: &str, content: &str) -> Value {
    json!({"name": name, "sha256": format!("{:x}", Sha256::digest(content.as_bytes())), "bytes": content.len(), "origin": "package_embedded"})
}

pub fn code_identity() -> Value {
    let revision = env!("BEKIPER_BUILD_REVISION");
    // "true"/"false" from git status, anything else (e.g. "unknown") is null.
    let dirty = env!("BEKIPER_BUILD_DIRTY")
        .parse::<bool>()
        .map_or(Value::Null, Value::Bool);
    json!({"package": env!("CARGO_PKG_NAME"), "version": env!("CARGO_PKG_VERSION"),
        "revision": if revision == "unknown" { None } else { Some(revision) }, "dirty": dirty,
        "source_sha256": env!("BEKIPER_ENGINE_SHA256"),
        "source_lock_sha256": env!("BEKIPER_SOURCE_LOCK_SHA256"),
        "dependency_lock_sha256": env!("BEKIPER_DEPENDENCY_LOCK_SHA256"),
        "identity_origin": "embedded_at_build"})
}
