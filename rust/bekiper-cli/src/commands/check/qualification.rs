//! Fixed, sequential qualification gates over one clean, exact Git revision.
use anyhow::{Context, Result, ensure};
use bekiper_post_analysis::{inputs, pipeline};
use clap::ValueEnum;
use serde_json::{Value, json};
use std::{
    fs::{self, File},
    io::Read,
    path::Path,
    process::Command,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum QualificationProfile {
    Native,
    Packages,
}

struct Gate {
    name: &'static str,
    program: &'static str,
    args: Vec<String>,
}

struct Cancellation {
    flag: Arc<AtomicBool>,
    #[cfg(unix)]
    signals: Vec<signal_hook::SigId>,
}

impl Cancellation {
    fn new() -> Result<Self> {
        let mut result = Self {
            flag: Arc::new(AtomicBool::new(false)),
            #[cfg(unix)]
            signals: Vec::new(),
        };
        #[cfg(unix)]
        for signal in [signal_hook::consts::SIGINT, signal_hook::consts::SIGTERM] {
            result.signals.push(signal_hook::flag::register(
                signal,
                Arc::clone(&result.flag),
            )?);
        }
        Ok(result)
    }
}

impl Drop for Cancellation {
    fn drop(&mut self) {
        #[cfg(unix)]
        for signal in self.signals.drain(..) {
            signal_hook::low_level::unregister(signal);
        }
    }
}

fn gates(profile: QualificationProfile, output: &Path) -> Vec<Gate> {
    let gate = |name, program, args: Vec<&str>| Gate {
        name,
        program,
        args: args.into_iter().map(str::to_owned).collect(),
    };
    match profile {
        QualificationProfile::Native => {
            let packages = [
                "bekiper-post-analysis",
                "bekiper-ingress",
                "bekiper-cli",
                "bekiper-registry",
            ];
            let selectors: Vec<_> = packages.iter().flat_map(|p| ["-p", *p]).collect();
            let mut tests = vec!["test"];
            tests.extend(&selectors);
            tests.extend(["--lib", "--tests", "--", "--test-threads=1"]);
            let mut clippy = vec!["clippy"];
            clippy.extend(&selectors);
            clippy.extend(["--all-targets", "--", "-D", "warnings"]);
            let mut docs = vec!["doc"];
            docs.extend(&selectors);
            docs.push("--no-deps");
            let mut book = gate("mdbook", "mdbook", vec!["build", "docs", "--dest-dir"]);
            book.args
                .push(output.join("book").to_string_lossy().into_owned());
            vec![
                gate("tests", "cargo", tests),
                gate("clippy", "cargo", clippy),
                gate("rustdoc", "cargo", docs),
                gate(
                    "binaries",
                    "cargo",
                    vec![
                        "build",
                        "-p",
                        "bekiper-post-analysis",
                        "-p",
                        "bekiper-cli",
                        "-p",
                        "bekiper-registry",
                        "--bins",
                    ],
                ),
                book,
                gate("format", "cargo", vec!["fmt", "--all", "--check"]),
                gate("diff", "git", vec!["diff", "--check"]),
                gate(
                    "renderer-tests",
                    "python",
                    vec![
                        "-m",
                        "pytest",
                        "python/bekiper-cli/tests/test_post_analysis_renderer.py",
                        "python/bekiper-cli/tests/test_native_distributions.py",
                        "-q",
                    ],
                ),
                gate(
                    "renderer-lint",
                    "ruff",
                    vec![
                        "check",
                        "python/bekiper-cli/bekiper_cli/post_analysis/render_bundle.py",
                        "python/bekiper-cli/tests/test_post_analysis_renderer.py",
                    ],
                ),
                gate(
                    "renderer-format",
                    "ruff",
                    vec![
                        "format",
                        "--check",
                        "python/bekiper-cli/bekiper_cli/post_analysis/render_bundle.py",
                        "python/bekiper-cli/tests/test_post_analysis_renderer.py",
                    ],
                ),
            ]
        }
        QualificationProfile::Packages => [
            ("registry", ".#bekiper-registry", false),
            ("website", ".#website", false),
            (
                "inspection-runtime",
                ".#checks.x86_64-linux.inspection-runtime",
                true,
            ),
            (
                "post-analysis-runtime",
                ".#checks.x86_64-linux.post-analysis-runtime",
                true,
            ),
        ]
        .into_iter()
        .map(|(name, target, tests)| {
            let mut result = gate(
                name,
                "canix",
                vec![
                    "cache",
                    "binary",
                    "build",
                    target,
                    "--no-push",
                    "--max-jobs",
                    "1",
                    "--cores",
                    "2",
                    "--output",
                    "json",
                ],
            );
            if tests {
                result.args.push("--include-tests".into());
            }
            result
        })
        .collect(),
    }
}

fn git(checkout: &Path, args: &[&str]) -> Result<Vec<u8>> {
    let result = Command::new("git")
        .current_dir(checkout)
        .args(args)
        .output()?;
    ensure!(
        result.status.success(),
        "Git identity failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    Ok(result.stdout)
}

fn source_identity(checkout: &Path, revision: &str) -> Result<Value> {
    let head = String::from_utf8(git(checkout, &["rev-parse", "HEAD"])?)?
        .trim()
        .to_owned();
    ensure!(
        head == revision,
        "qualification checkout HEAD differs from requested exact revision"
    );
    ensure!(
        git(
            checkout,
            &["status", "--porcelain=v1", "--untracked-files=all"]
        )?
        .is_empty(),
        "qualification requires a clean source checkout"
    );
    let files = git(checkout, &["ls-files", "-z"])?;
    // Reuse the streaming byte hasher; frame each pathname and file digest unambiguously.
    let mut entries = Vec::new();
    for name in files.split(|b| *b == 0).filter(|n| !n.is_empty()) {
        let name = std::str::from_utf8(name)?;
        let path = checkout.join(name);
        let metadata = fs::symlink_metadata(&path)?;
        ensure!(
            metadata.is_file() || metadata.file_type().is_symlink(),
            "unsupported qualification source member {name}"
        );
        entries.push(if metadata.file_type().is_symlink() {
            json!({"path":name,"target":fs::read_link(path)?})
        } else {
            json!({"path":name,"fingerprint":inputs::fingerprint(&path)?})
        });
    }
    Ok(json!({"revision":head,"files":entries}))
}

fn save(path: &Path, receipt: &Value) -> Result<()> {
    let temporary = path.with_extension("pending.json");
    pipeline::write_json(&temporary, receipt)?;
    fs::rename(temporary, path)?;
    Ok(())
}

pub fn qualify(
    checkout: &Path,
    revision: &str,
    profile: QualificationProfile,
    output: &Path,
) -> Result<Value> {
    let checkout = checkout.canonicalize()?;
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
        .canonicalize()?;
    let output = parent.join(
        output
            .file_name()
            .context("qualification output directory")?,
    );
    ensure!(
        !output.starts_with(&checkout) && !checkout.starts_with(&output),
        "qualification outputs must be outside the source checkout"
    );
    fs::create_dir(&output).context("qualification evidence requires a fresh output directory")?;
    let receipt_path = output.join("receipt.json");
    let mut receipt = json!({"schema_version":1,"profile":format!("{profile:?}").to_ascii_lowercase(),"checkout":checkout,
        "revision":revision,"status":"running","gates":[],"analysis_admitted":false,"atlas_activation":false});
    receipt["driver"] = inputs::fingerprint(&std::env::current_exe()?)?;
    save(&receipt_path, &receipt)?;
    let run = (|| -> Result<()> {
        let cancellation = Cancellation::new()?;
        let identity = source_identity(&checkout, revision)?;
        pipeline::write_json(&output.join("source_identity.json"), &identity)?;
        receipt["source_identity"] = inputs::fingerprint(&output.join("source_identity.json"))?;
        save(&receipt_path, &receipt)?;
        for gate in gates(profile, &output) {
            ensure!(
                source_identity(&checkout, revision)? == identity,
                "source changed before gate {}",
                gate.name
            );
            let stdout = output.join(format!("{}.stdout.log", gate.name));
            let stderr = output.join(format!("{}.stderr.log", gate.name));
            receipt["gates"].as_array_mut().context("gate attempts")?.push(json!({"name":gate.name,"program":gate.program,"args":gate.args,"status":"running"}));
            save(&receipt_path, &receipt)?;
            eprintln!("qualification: {}", gate.name);
            let mut command = Command::new(gate.program);
            command
                .args(&gate.args)
                .current_dir(&checkout)
                .env_remove("RUSTC_WRAPPER")
                .env("CARGO_BUILD_JOBS", "2")
                .env("RUSTDOCFLAGS", "-D warnings")
                .env("PYTHONPATH", checkout.join("python/bekiper-cli"));
            let status = pipeline::run_child(&mut command, &stdout, &stderr, &|| {
                cancellation.flag.load(Ordering::Relaxed)
            })
            .with_context(|| format!("qualification gate {}", gate.name))?;
            let unchanged =
                source_identity(&checkout, revision).is_ok_and(|after| after == identity);
            let attempt = receipt["gates"]
                .as_array_mut()
                .context("gate attempts")?
                .last_mut()
                .context("current gate")?;
            attempt["exit_status"] = json!(status.code());
            attempt["source_unchanged"] = unchanged.into();
            attempt["stdout"] = inputs::fingerprint(&stdout)?;
            attempt["stderr"] = inputs::fingerprint(&stderr)?;
            let mut accepted = status.success() && unchanged;
            if matches!(profile, QualificationProfile::Packages) && status.success() {
                let response = canix_response(&stdout)?;
                accepted &= response["ok"] == true
                    && response["data"]["results"].as_array().is_some_and(|r| {
                        !r.is_empty()
                            && r.iter()
                                .all(|p| p["path"].as_str().is_some_and(|p| Path::new(p).exists()))
                    });
                attempt["response"] = response;
            }
            attempt["status"] = if accepted { "passed" } else { "failed" }.into();
            save(&receipt_path, &receipt)?;
            ensure!(
                accepted,
                "gate {} failed; see {}",
                gate.name,
                output.display()
            );
        }
        Ok(())
    })();
    receipt["status"] = if run.is_ok() { "passed" } else { "failed" }.into();
    if let Err(error) = &run {
        receipt["error"] = format!("{error:#}").into();
        for gate in receipt["gates"]
            .as_array_mut()
            .context("qualification gates")?
        {
            if gate["status"] == "running" {
                gate["status"] = "failed".into();
                gate["error"] = format!("{error:#}").into();
                let name = gate["name"].as_str().context("gate name")?.to_owned();
                for stream in ["stdout", "stderr"] {
                    let log = output.join(format!("{name}.{stream}.log"));
                    if log.is_file() {
                        gate[stream] = inputs::fingerprint(&log)?;
                    }
                }
            }
        }
    }
    save(&receipt_path, &receipt)?;
    run?;
    Ok(receipt)
}

fn canix_response(path: &Path) -> Result<Value> {
    let mut text = String::new();
    File::open(path)?
        .take(16 * 1024 * 1024 + 1)
        .read_to_string(&mut text)?;
    ensure!(text.len() <= 16 * 1024 * 1024, "oversized Canix response");
    for (index, character) in text.char_indices() {
        if character != '{' {
            continue;
        }
        let mut records = serde_json::Deserializer::from_str(&text[index..]).into_iter::<Value>();
        if let Some(Ok(record)) = records.next()
            && record.get("schemaVersion").is_some()
            && record.get("ok").is_some()
        {
            return Ok(record);
        }
    }
    anyhow::bail!("package gate has no structured Canix response")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn dirty_or_wrong_revision_source_never_enters_a_gate_and_retains_failure() {
        let parent = tempfile::tempdir().unwrap();
        let repo = parent.path().join("repo");
        fs::create_dir(&repo).unwrap();
        git(&repo, &["init", "-q"]).unwrap();
        fs::write(repo.join("source"), "first").unwrap();
        git(&repo, &["add", "source"]).unwrap();
        // Construct a labelled fixture object without operator identity/signing hooks.
        let tree = String::from_utf8(git(&repo, &["write-tree"]).unwrap()).unwrap();
        let object = format!(
            "tree {}\nauthor Synthetic fixture <fixture@example.invalid> 0 +0000\ncommitter Synthetic fixture <fixture@example.invalid> 0 +0000\n\nSynthetic qualification source fixture\n",
            tree.trim()
        );
        let mut child = Command::new("git")
            .current_dir(&repo)
            .args(["hash-object", "-w", "-t", "commit", "--stdin"])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(object.as_bytes())
            .unwrap();
        let result = child.wait_with_output().unwrap();
        assert!(result.status.success());
        let head = String::from_utf8(result.stdout).unwrap();
        git(&repo, &["update-ref", "HEAD", head.trim()]).unwrap();
        let revision = String::from_utf8(git(&repo, &["rev-parse", "HEAD"]).unwrap())
            .unwrap()
            .trim()
            .to_string();
        assert!(source_identity(&repo, &revision).is_ok());
        assert!(source_identity(&repo, "different").is_err());
        fs::write(repo.join("source"), "changed").unwrap();
        let out = parent.path().join("evidence");
        assert!(qualify(&repo, &revision, QualificationProfile::Native, &out).is_err());
        let receipt = inputs::load_json(&out.join("receipt.json")).unwrap();
        assert_eq!(receipt["status"], "failed");
        assert_eq!(receipt["gates"], json!([]));
        assert!(qualify(&repo, &revision, QualificationProfile::Native, &out).is_err());
    }

    #[test]
    fn fixed_package_profile_requires_canix_admission_and_structured_results() {
        let gates = gates(QualificationProfile::Packages, Path::new("evidence"));
        assert_eq!(gates.len(), 4);
        assert!(gates.iter().all(|g| {
            g.program == "canix"
                && g.args
                    .starts_with(&["cache".into(), "binary".into(), "build".into()])
                && g.args.contains(&"--no-push".into())
                && g.args.contains(&"--max-jobs".into())
        }));
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("stdout");
        fs::write(
            &path,
            "prelude {invalid}\n{\"schemaVersion\":1,\"ok\":false}\ntrailer",
        )
        .unwrap();
        assert_eq!(canix_response(&path).unwrap()["ok"], false);
        fs::write(&path, "command exited 0 without a response").unwrap();
        assert!(canix_response(&path).is_err());
    }
}
