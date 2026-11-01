use sha2::{Digest, Sha256};
use std::{env, fs, path::Path};

fn hash_source(path: &Path, root: &Path, hash: &mut Sha256) {
    println!("cargo:rerun-if-changed={}", path.display());
    if path.is_dir() {
        let mut entries: Vec<_> = fs::read_dir(path)
            .expect("preparation directory")
            .map(|entry| entry.expect("preparation entry").path())
            .collect();
        entries.sort();
        for entry in entries {
            if entry.is_dir()
                || entry.extension().is_some_and(|ext| ext == "rs")
                || entry
                    .file_name()
                    .is_some_and(|name| name == "Cargo.toml" || name == "Cargo.lock")
            {
                hash_source(&entry, root, hash);
            }
        }
    } else {
        hash.update(
            path.strip_prefix(root)
                .expect("workspace source")
                .to_string_lossy()
                .as_bytes(),
        );
        hash.update([0]);
        let bytes = fs::read(path).expect("preparation source");
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(bytes);
    }
}

fn main() {
    let manifest = env::var("CARGO_MANIFEST_DIR").expect("manifest");
    let root = Path::new(&manifest)
        .parent()
        .expect("rust")
        .parent()
        .expect("workspace");
    let mut hash = Sha256::new();
    for feature in [
        "CARGO_FEATURE_POSE_GPU",
        "CARGO_FEATURE_POSE_INFER",
        "CARGO_FEATURE_POSE_VIDEO",
        "CARGO_FEATURE_POSE_HDF5",
    ] {
        println!("cargo:rerun-if-env-changed={feature}");
        hash.update(feature.as_bytes());
        hash.update([u8::from(env::var_os(feature).is_some())]);
    }
    // A conservative dependency closure, including adapters that select inputs.
    // Any change here requires recapture; scoring/presentation-only crates can
    // use explicit target-bound qualification without relabeling the producer.
    for part in [
        "Cargo.toml",
        "Cargo.lock",
        "rust/bekiper-core/src",
        "rust/bekiper-core/Cargo.toml",
        "rust/bekiper-pose/src",
        "rust/bekiper-pose/Cargo.toml",
        "rust/bekiper-perimeter/src",
        "rust/bekiper-perimeter/Cargo.toml",
        "rust/bekiper-detect/src",
        "rust/bekiper-detect/Cargo.toml",
        "rust/bekiper-math/src",
        "rust/bekiper-math/Cargo.toml",
        "rust/bekiper-feature/src",
        "rust/bekiper-feature/Cargo.toml",
        "rust/bekiper-ingress/Cargo.toml",
        "rust/bekiper-ingress/build.rs",
        "rust/bekiper-ingress/src/config",
        "rust/bekiper-ingress/src/pose_import.rs",
        "rust/bekiper-ingress/src/analysis",
        "rust/bekiper-ingress/src/preparation.rs",
        "rust/bekiper-ingress/src/workflow/augment",
        "rust/bekiper-ingress/src/workflow/pose",
        "rust/bekiper-ingress/src/workflow/pose.rs",
        "rust/bekiper-ingress/src/workflow/inputs.rs",
    ] {
        hash_source(&root.join(part), root, &mut hash);
    }
    println!(
        "cargo:rustc-env=BEKIPER_PREPARATION_VERSION={:x}",
        hash.finalize()
    );
    let mut reconciliation = Sha256::new();
    for part in [
        "Cargo.toml",
        "Cargo.lock",
        "rust/bekiper-ingress/build.rs",
        "rust/bekiper-ingress/Cargo.toml",
        "rust/bekiper-ingress/src/audit/population.rs",
        "rust/bekiper-ingress/src/audit/reconciliation.rs",
        "rust/bekiper-ingress/src/audit/reconciliation",
        "rust/bekiper-ingress/src/aggregate/metadata.rs",
        "rust/bekiper-annotation/src",
        "rust/bekiper-post-analysis/src",
        "rust/bekiper-core/src/handoff.rs",
    ] {
        hash_source(&root.join(part), root, &mut reconciliation);
    }
    println!(
        "cargo:rustc-env=BEKIPER_RECONCILIATION_VERSION={:x}",
        reconciliation.finalize()
    );
}
