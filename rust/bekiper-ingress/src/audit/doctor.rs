//! Audit doctor responsibilities.
use super::*;

pub fn doctor_report(config_path: Option<&Path>) -> AuditReport {
    let mut report = AuditReport::new("doctor", config_path.map(Path::to_path_buf));
    report.extend(doctor_diagnostics(config_path));
    report
}

pub(super) fn doctor_diagnostics(config_path: Option<&Path>) -> Vec<AuditDiagnostic> {
    let mut diagnostics = Vec::new();

    let in_nix_shell = std::env::var("IN_NIX_SHELL")
        .ok()
        .is_some_and(|v| !v.is_empty());
    diagnostics.push(if in_nix_shell {
        AuditDiagnostic::new(
            AuditSeverity::Ok,
            "environment",
            "running inside a nix shell",
        )
    } else {
        AuditDiagnostic::new(
            AuditSeverity::Warning,
            "environment",
            "IN_NIX_SHELL is not set",
        )
        .cause("runtime libraries are normally provided by `nix develop`")
        .fix_hint("enter the project devshell before GPU/Python analysis")
        .validation_command("nix develop -c bekiper doctor")
    });

    match std::env::var("ORT_DYLIB_PATH") {
        Ok(path) if Path::new(&path).exists() => diagnostics.push(
            AuditDiagnostic::new(AuditSeverity::Ok, "environment", "ORT_DYLIB_PATH exists")
                .artifact_path(path),
        ),
        Ok(path) => diagnostics.push(
            AuditDiagnostic::new(
                AuditSeverity::Error,
                "environment",
                "ORT_DYLIB_PATH points to a missing file",
            )
            .artifact_path(path)
            .cause("ONNX Runtime is loaded dynamically by the pose model")
            .fix_hint("enter the nix devshell or fix ORT_DYLIB_PATH")
            .validation_command("nix develop -c bekiper doctor"),
        ),
        Err(_) => diagnostics.push(
            AuditDiagnostic::new(
                AuditSeverity::Error,
                "environment",
                "ORT_DYLIB_PATH is unset",
            )
            .cause("ONNX Runtime is required before pose model loading")
            .fix_hint("enter the nix devshell")
            .validation_command("nix develop -c bekiper doctor"),
        ),
    }

    for var in ["HSA_OVERRIDE_GFX_VERSION", "PYTORCH_ROCM_ARCH"] {
        diagnostics.push(match std::env::var(var) {
            Ok(value) if !value.is_empty() => {
                AuditDiagnostic::new(AuditSeverity::Ok, "environment", format!("{var}={value}"))
            }
            _ => AuditDiagnostic::new(
                AuditSeverity::Warning,
                "environment",
                format!("{var} is unset"),
            )
            .cause("ROCm GPU runs on gfx1100 need the devshell GPU environment")
            .fix_hint("enter the nix devshell before GPU analysis")
            .validation_command("nix develop -c bekiper doctor"),
        });
    }

    diagnostics.push(python_import_diagnostic("cv2"));
    diagnostics.push(python_import_diagnostic("torch"));
    diagnostics.push(python_import_diagnostic("bekiper_auto_detect"));

    if let Some(path) = config_path {
        diagnostics.extend(pose_model_diagnostics(path));
        diagnostics.extend(output_safety_diagnostics(path));
    }

    diagnostics
}

pub(super) fn python_import_diagnostic(module: &str) -> AuditDiagnostic {
    let code = format!("import {module}; print('ok')");
    match Command::new("python3").arg("-c").arg(&code).output() {
        Ok(output) if output.status.success() => AuditDiagnostic::new(
            AuditSeverity::Ok,
            "python",
            format!("python3 imports {module}"),
        ),
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            AuditDiagnostic::new(
                AuditSeverity::Error,
                "python",
                format!("python3 cannot import {module}"),
            )
            .cause(stderr_tail(&stderr))
            .fix_hint("refresh the devshell Python venv")
            .validation_command(format!("nix develop -c python3 -c \"import {module}\""))
        }
        Err(error) => {
            AuditDiagnostic::new(AuditSeverity::Error, "python", "failed to execute python3")
                .cause(error.to_string())
                .fix_hint("enter the nix devshell so python3 resolves to the project venv")
                .validation_command(
                    "nix develop -c python3 -c \"import cv2; import bekiper_auto_detect\"",
                )
        }
    }
}

pub(super) fn pose_model_diagnostics(config_path: &Path) -> Vec<AuditDiagnostic> {
    let mut diagnostics = Vec::new();
    let base_dir = config_path.parent().unwrap_or(Path::new("."));
    let Ok(mut config) = IngressConfig::from_file(config_path) else {
        return diagnostics;
    };
    if config
        .expand_for_validation(base_dir, &[Stage::Pose])
        .is_err()
    {
        return diagnostics;
    }

    let mut seen = HashSet::new();
    for experiment in &config.experiments {
        let Some(pose_model) = experiment.pose_model.as_ref() else {
            continue;
        };
        let resolved = pose_model.resolved_against(base_dir);
        if !seen.insert((resolved.path.clone(), resolved.spec.clone())) {
            continue;
        }
        diagnostics.push(path_readiness_diagnostic(
            "pose model",
            &resolved.path,
            "pose inference requires the ONNX model file",
            format!("bekiper doctor -c {}", config_path.display()),
        ));
        if let PoseModelPreset::Custom(spec) = &resolved.spec {
            diagnostics.push(path_readiness_diagnostic(
                "pose model",
                spec,
                "pose inference requires the model spec JSON",
                format!("bekiper doctor -c {}", config_path.display()),
            ));
        } else {
            #[cfg(feature = "pose-gpu")]
            match resolve_model_spec(&resolved.spec) {
                Ok(_) => diagnostics.push(AuditDiagnostic::new(
                    AuditSeverity::Ok,
                    "pose model",
                    "pose model preset spec resolves",
                )),
                Err(error) => diagnostics.push(
                    AuditDiagnostic::new(
                        AuditSeverity::Error,
                        "pose model",
                        "pose model spec does not resolve",
                    )
                    .cause(error.to_string())
                    .fix_hint("fix the pose_model.spec value")
                    .validation_command(format!("bekiper doctor -c {}", config_path.display())),
                ),
            }
            #[cfg(not(feature = "pose-gpu"))]
            diagnostics.push(
                AuditDiagnostic::new(
                    AuditSeverity::Skipped,
                    "pose model",
                    "pose model preset spec validation requires the pose-gpu feature",
                )
                .validation_command("cargo check -p bekiper-cli --features full"),
            );
        }

        let cache_path = resolved.path.with_extension("mxr");
        if cache_path.exists() {
            diagnostics.push(
                AuditDiagnostic::new(
                    AuditSeverity::Ok,
                    "pose model",
                    "MIGraphX compiled cache exists",
                )
                .artifact_path(cache_path),
            );
        } else if let Some(parent) = cache_path.parent() {
            diagnostics.push(if parent.exists() {
                AuditDiagnostic::new(
                    AuditSeverity::Warning,
                    "pose model",
                    "MIGraphX compiled cache is not present; first GPU run may compile",
                )
                .artifact_path(cache_path)
                .fix_hint("run a short GPU pose smoke to warm the model cache")
            } else {
                AuditDiagnostic::new(
                    AuditSeverity::Warning,
                    "pose model",
                    "MIGraphX cache parent directory does not exist",
                )
                .artifact_path(parent)
                .fix_hint("restore the model directory before GPU pose inference")
            });
        }
    }
    diagnostics
}

pub(super) fn output_safety_diagnostics(config_path: &Path) -> Vec<AuditDiagnostic> {
    let base_dir = config_path.parent().unwrap_or(Path::new("."));
    let Ok(config) = IngressConfig::from_file(config_path) else {
        return Vec::new();
    };
    let output_dir = base_dir.join(
        config
            .project
            .output_dir
            .as_deref()
            .unwrap_or(Path::new("output")),
    );
    let severity = if output_dir.exists() {
        AuditSeverity::Ok
    } else {
        AuditSeverity::Skipped
    };
    vec![
        AuditDiagnostic::new(
            severity,
            "output",
            if output_dir.exists() {
                "project output directory exists; analysis will resume from checkpoints where fresh"
            } else {
                "project output directory does not exist yet; analysis would create it"
            },
        )
        .artifact_path(output_dir)
        .validation_command(format!(
            "bekiper run -c {} --dry-run",
            config_path.display()
        )),
    ]
}
