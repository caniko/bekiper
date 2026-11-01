use sha2::{Digest, Sha256};
use std::{env, fs, path::Path, process::Command};
#[path = "src/source_tree.rs"]
mod source_tree;

fn main() {
    println!("cargo:rerun-if-env-changed=BEKIPER_CODE_REVISION");
    println!("cargo:rerun-if-env-changed=BEKIPER_CODE_DIRTY");
    println!("cargo:rerun-if-env-changed=BEKIPER_SOURCE_CARGO_LOCK");
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed=../../Cargo.lock");
    println!("cargo:rerun-if-changed=../../.git/index");
    println!(
        "cargo:rerun-if-changed=../../behavioral_post_analysis_harness/reference/BEHAVIORAL_RECIPE.md"
    );
    println!(
        "cargo:rerun-if-changed=../../behavioral_post_analysis_harness/reference/YMAZE_AMENDMENT.json"
    );
    let git = |args: &[&str]| {
        Command::new("git")
            .args(args)
            .output()
            .ok()
            .filter(|output| output.status.success())
            .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
    };
    let revision = env::var("BEKIPER_CODE_REVISION")
        .ok()
        .or_else(|| git(&["rev-parse", "HEAD"]))
        .unwrap_or_else(|| "unknown".into());
    let dirty = env::var("BEKIPER_CODE_DIRTY")
        .ok()
        .or_else(|| git(&["status", "--porcelain"]).map(|text| (!text.is_empty()).to_string()))
        .unwrap_or_else(|| "unknown".into());
    // Cargo prunes unused workspace members from filtered package builds.
    // Bind source identity to the original lock, and record the effective lock
    // separately so installed runs retain both identities.
    let source_lock = env::var_os("BEKIPER_SOURCE_CARGO_LOCK")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| "../../Cargo.lock".into());
    println!("cargo:rerun-if-changed={}", source_lock.display());
    let source_lock_bytes = fs::read(&source_lock).expect("source workspace lock");
    let dependency_lock_bytes = fs::read("../../Cargo.lock").expect("effective dependency lock");
    // Scientific duration selection now lives in shared workspace crates.
    for crate_name in ["bekiper-math", "bekiper-feature"] {
        let directory = format!("../{crate_name}");
        println!("cargo:rerun-if-changed={directory}/src");
        println!("cargo:rerun-if-changed={directory}/Cargo.toml");
    }
    let hash =
        source_tree::digest(Path::new("."), &source_lock_bytes).expect("native source identity");
    println!("cargo:rustc-env=BEKIPER_BUILD_REVISION={revision}");
    println!("cargo:rustc-env=BEKIPER_BUILD_DIRTY={dirty}");
    println!(
        "cargo:rustc-env=BEKIPER_SOURCE_LOCK_SHA256={:x}",
        Sha256::digest(&source_lock_bytes)
    );
    println!(
        "cargo:rustc-env=BEKIPER_DEPENDENCY_LOCK_SHA256={:x}",
        Sha256::digest(&dependency_lock_bytes)
    );
    println!("cargo:rustc-env=BEKIPER_ENGINE_SHA256={hash}");
}
