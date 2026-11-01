use std::net::SocketAddr;
use std::path::PathBuf;

use anyhow::Context;
use bekiper_annotation::canonical::{CanonicalAnnotationStore, SyncAction, SyncMode, SyncOptions};
use clap::{Args, Subcommand};

#[derive(Args)]
pub struct AnnotateArgs {
    #[command(subcommand)]
    pub command: AnnotateCommand,
}

#[derive(Subcommand)]
pub enum AnnotateCommand {
    /// Launch the local annotation GUI.
    #[cfg(feature = "gui")]
    Gui(AnnotateGuiArgs),

    /// Serve a single dataset over token-authenticated WebTransport.
    Serve(AnnotateServeArgs),

    /// Connect to a remote `bekiper annotate serve` over WebTransport.
    Connect(AnnotateConnectArgs),

    /// Synchronize legacy annotation files with the canonical bekiper store.
    Sync(AnnotateSyncArgs),

    /// Show canonical annotation store status.
    Status(AnnotateStatusArgs),
}

#[derive(Args)]
pub struct AnnotateGuiArgs {
    /// Path to the project bekiper_config.toml.
    pub config: PathBuf,
}

#[derive(Args)]
pub struct AnnotateSyncArgs {
    /// Annotation directory containing metadata/video_map and plugin_files.
    pub annotation_dir: PathBuf,

    /// Only import legacy files into the canonical store.
    #[arg(long, conflicts_with = "export_only")]
    pub import_only: bool,

    /// Only export canonical files back to legacy-compatible files.
    #[arg(long, conflicts_with = "import_only")]
    pub export_only: bool,

    /// Report the planned action without writing files.
    #[arg(long)]
    pub dry_run: bool,

    /// Imported annotations were reviewed outside Bekiper before this ingestion.
    #[arg(long, conflicts_with = "export_only")]
    pub reviewed_externally: bool,
}

#[derive(Args)]
pub struct AnnotateStatusArgs {
    /// Annotation directory containing metadata/video_map and plugin_files.
    pub annotation_dir: PathBuf,
}

#[derive(Args)]
pub struct AnnotateServeArgs {
    /// Path to the project bekiper_config.toml.
    pub config: PathBuf,

    /// UDP socket address to bind for WebTransport over QUIC.
    #[arg(long, default_value = "0.0.0.0:8443")]
    pub bind: SocketAddr,

    /// PEM certificate path for production TLS.
    #[arg(long)]
    pub cert_path: Option<PathBuf>,

    /// PEM private key path for production TLS.
    #[arg(long)]
    pub key_path: Option<PathBuf>,

    /// Generate an ephemeral self-signed certificate for local development.
    #[arg(long, default_value_t = false)]
    pub self_signed_dev: bool,

    /// Read the bearer token from this file; overrides auto-generation.
    #[arg(long)]
    pub token_file: Option<PathBuf>,

    /// Pre-encode AV1 GOP cache before accepting sessions: pass a trial id or `all`.
    #[arg(long)]
    pub prewarm_av1_cache: Option<String>,
}

#[derive(Args)]
pub struct AnnotateConnectArgs {
    /// Server URL, e.g. `https://bekiper.candee.baby:8443`.
    pub url: String,

    /// Pre-obtained bearer token: either a Rauthy OIDC access token or the
    /// legacy `bka_` static token printed by `bekiper annotate serve`.
    ///
    /// Optional: when omitted, supply OIDC options to authenticate. For shells,
    /// prefer the `BEKIPER_ANNOTATE_TOKEN` env var to avoid leaking the token
    /// via `ps`.
    #[arg(long, env = "BEKIPER_ANNOTATE_TOKEN")]
    pub token: Option<String>,

    /// Rauthy OIDC issuer base URL (e.g. `https://id.tartanoglu.com/auth/v1`).
    /// With `--oidc-client-id`, authenticates via the OAuth device flow when no
    /// `--token` or password-grant credentials are given.
    #[arg(long, env = "BEKIPER_OIDC_ISSUER")]
    pub oidc_issuer: Option<String>,

    /// Public OIDC client id registered in Rauthy for the annotate app.
    #[arg(long, env = "BEKIPER_OIDC_CLIENT_ID")]
    pub oidc_client_id: Option<String>,

    /// Rauthy username/email for a headless password-grant smoke test.
    #[arg(long, env = "BEKIPER_OIDC_USERNAME", requires = "oidc_password_file")]
    pub oidc_username: Option<String>,

    /// File containing the Rauthy password for a headless password-grant smoke test.
    #[arg(long, env = "BEKIPER_OIDC_PASSWORD_FILE", requires = "oidc_username")]
    pub oidc_password_file: Option<PathBuf>,

    /// Accept self-signed certificates. Dev only; never enable against
    /// production servers.
    #[arg(long, default_value_t = false)]
    pub insecure_self_signed: bool,

    /// Drive the trait-level connect handshake and exit without opening the
    /// GUI. Useful for CI smoke tests and connection probes.
    #[arg(long, default_value_t = false)]
    pub smoke_test: bool,
}

pub fn run(args: AnnotateArgs) -> anyhow::Result<()> {
    match args.command {
        #[cfg(feature = "gui")]
        AnnotateCommand::Gui(args) => bekiper_annotate::run(args.config)
            .map_err(|e| anyhow::anyhow!("bekiper-annotate GUI runtime: {e:?}")),
        AnnotateCommand::Serve(args) => {
            init_server_tracing();
            bekiper_annotate::server::run_blocking(bekiper_annotate::server::ServeOptions {
                config: args.config,
                bind: args.bind,
                cert_path: args.cert_path,
                key_path: args.key_path,
                self_signed_dev: args.self_signed_dev,
                token_file: args.token_file,
                prewarm_av1_cache: args.prewarm_av1_cache,
            })
        }
        #[cfg(feature = "annotate-client")]
        AnnotateCommand::Connect(args) => {
            init_server_tracing();
            run_connect(args)
        }
        #[cfg(not(feature = "annotate-client"))]
        AnnotateCommand::Connect(_) => {
            anyhow::bail!(
                "annotate connect requires the `annotate-client` feature; rebuild with --features annotate-client"
            )
        }
        AnnotateCommand::Sync(args) => run_sync(args),
        AnnotateCommand::Status(args) => run_status(args),
    }
}

fn run_sync(args: AnnotateSyncArgs) -> anyhow::Result<()> {
    let mode = if args.import_only {
        SyncMode::ImportOnly
    } else if args.export_only {
        SyncMode::ExportOnly
    } else {
        SyncMode::TwoWay
    };
    let report = CanonicalAnnotationStore::sync_with_review(
        &args.annotation_dir,
        SyncOptions {
            mode,
            dry_run: args.dry_run,
        },
        args.reviewed_externally,
    )
    .with_context(|| {
        format!(
            "failed to synchronize annotation directory {}",
            args.annotation_dir.display()
        )
    })?;
    println!(
        "{}{}: {}",
        if args.dry_run { "dry-run " } else { "" },
        action_label(report.action),
        args.annotation_dir.display()
    );
    println!(
        "canonical: {}, legacy changed: {}, canonical changed: {}",
        report.canonical_exists, report.legacy_changed, report.canonical_changed
    );
    if let Some(path) = report.conflict_path {
        println!("conflict report: {}", path.display());
    }
    Ok(())
}

fn run_status(args: AnnotateStatusArgs) -> anyhow::Result<()> {
    let status = CanonicalAnnotationStore::status(&args.annotation_dir).with_context(|| {
        format!(
            "failed to inspect annotation directory {}",
            args.annotation_dir.display()
        )
    })?;
    println!("annotation dir: {}", args.annotation_dir.display());
    println!("canonical store: {}", status.canonical_exists);
    println!("preferred source: {}", status.preferred_source);
    println!("legacy changed: {}", status.legacy_changed);
    println!("canonical changed: {}", status.canonical_changed);
    println!("conflicts: {}", status.conflicts.len());
    for path in status.conflicts {
        println!("  {}", path.display());
    }
    Ok(())
}

fn action_label(action: SyncAction) -> &'static str {
    match action {
        SyncAction::Import => "import",
        SyncAction::Export => "export",
        SyncAction::Noop => "noop",
        SyncAction::Conflict => "conflict",
    }
}

#[cfg(feature = "annotate-client")]
fn run_connect(args: AnnotateConnectArgs) -> anyhow::Result<()> {
    use bekiper_annotate::backend::{
        AnnotateBackend, AnnotationStore, DeviceFlowConfig, NetworkBackend, NetworkBackendConfig,
        PasswordGrantConfig, obtain_token_via_device_flow, obtain_token_via_password_grant,
    };
    use std::sync::Arc;
    use url::Url;

    let url =
        Url::parse(&args.url).map_err(|e| anyhow::anyhow!("invalid --url '{}': {e}", args.url))?;
    let config = NetworkBackendConfig {
        insecure_self_signed: args.insecure_self_signed,
        ..NetworkBackendConfig::default()
    };
    tracing::info!(
        url = %args.url,
        insecure_self_signed = args.insecure_self_signed,
        smoke_test = args.smoke_test,
        "annotate client connecting"
    );

    // Keep this runtime alive for the lifetime of the GUI session below:
    // NetworkBackend spawns its reader and writer tasks on it during
    // `connect`, and dropping the runtime would terminate them.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    // Resolve the bearer token: an explicit --token (OIDC access token or the
    // legacy bka_ static token) wins; otherwise authenticate interactively via
    // the OAuth device flow against the configured Rauthy issuer.
    let token = runtime.block_on(async {
        if let Some(token) = args.token.as_deref().filter(|t| !t.is_empty()) {
            return Ok::<String, anyhow::Error>(token.to_string());
        }
        if let (Some(issuer), Some(client_id), Some(username), Some(password_file)) = (
            args.oidc_issuer.as_deref(),
            args.oidc_client_id.as_deref(),
            args.oidc_username.as_deref(),
            args.oidc_password_file.as_deref(),
        ) {
            let password = std::fs::read_to_string(password_file).with_context(|| {
                format!(
                    "failed to read OIDC password file {}",
                    password_file.display()
                )
            })?;
            let password = password.trim_end_matches(['\r', '\n']).to_string();
            if password.is_empty() {
                anyhow::bail!("OIDC password file {} is empty", password_file.display());
            }
            let cfg = PasswordGrantConfig::new(issuer, client_id, username, password);
            let tokens = obtain_token_via_password_grant(&cfg).await?;
            return Ok(tokens.access_token);
        }
        match (args.oidc_issuer.as_deref(), args.oidc_client_id.as_deref()) {
            (Some(issuer), Some(client_id)) => {
                let cfg = DeviceFlowConfig::new(issuer, client_id);
                let tokens = obtain_token_via_device_flow(&cfg).await?;
                Ok(tokens.access_token)
            }
            _ => anyhow::bail!(
                "no credential supplied: pass --token (or BEKIPER_ANNOTATE_TOKEN), or \
                 --oidc-issuer and --oidc-client-id to authenticate via the device flow"
            ),
        }
    })?;

    let backend: Arc<dyn AnnotateBackend> = runtime
        .block_on(async {
            let backend = NetworkBackend::connect(url.clone(), token.clone(), config).await?;
            let (dataset_id, dataset_name, server_version, capabilities, client_id) =
                backend.server_info_snapshot().await;
            println!(
                "connected: dataset={dataset_name} ({dataset_id}) server={server_version} client_id={client_id}"
            );
            println!("capabilities: {}", capabilities.join(", "));

            if args.smoke_test {
                match backend.list_trials().await {
                    Ok(entries) => println!("list_trials: {} entries", entries.len()),
                    Err(e) => println!("list_trials failed: {e}"),
                }
                match backend.list_sessions().await {
                    Ok(entries) => println!("list_sessions: {} entries", entries.len()),
                    Err(e) => println!("list_sessions failed: {e}"),
                }
                match backend.list_groups().await {
                    Ok((groups, last_modified)) => println!(
                        "list_groups: {} groups (last_modified={last_modified})",
                        groups.len()
                    ),
                    Err(e) => println!("list_groups failed: {e}"),
                }
            }

            Ok::<Arc<dyn AnnotateBackend>, bekiper_annotate::backend::BackendError>(Arc::new(
                backend,
            ))
        })
        .map_err(|e| anyhow::anyhow!("connect: {e}"))?;

    if args.smoke_test {
        return Ok(());
    }

    #[cfg(feature = "gui")]
    {
        // Hand the backend off to the Iced GUI. Note that iced spawns its own
        // tokio runtime for `Task::perform`; the backend's pre-existing
        // reader/writer tasks remain on the runtime created above, which is
        // kept alive by being held in this stack frame for the duration of
        // `run_with_backend`.
        let _runtime_keep_alive = runtime;
        bekiper_annotate::run_with_backend(backend)
            .map_err(|e| anyhow::anyhow!("bekiper-annotate GUI runtime: {e:?}"))
    }
    #[cfg(not(feature = "gui"))]
    {
        let _runtime_keep_alive = runtime;
        let _backend_keep_alive = backend;
        anyhow::bail!("the `gui` feature is not enabled; rerun the smoke test with --smoke-test")
    }
}

fn init_server_tracing() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .try_init();
}

#[cfg(test)]
mod sync_tests {
    use super::*;
    use bekiper_core::inspection::PerimeterSpec;

    fn temp_annotation_dir(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "bekiper_cli_annotate_sync_{label}_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("plugin_files")).unwrap();
        std::fs::write(
            dir.join("plugin_files/perimeter-rectangle-training_1.csv"),
            "object,10,20,30,40,image.png,100,100\n",
        )
        .unwrap();
        std::fs::write(
            dir.join("video_map.csv"),
            "video_number,perimeter_id\n1,training_1\n",
        )
        .unwrap();
        dir
    }

    fn temp_rereferenced_annotation_dir(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "bekiper_cli_annotate_sync_{label}_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("plugin_files")).unwrap();
        std::fs::write(
            dir.join("plugin_files/perimeter-rectangle-training_1.csv"),
            "familiar,48,49,25,38,A2_training_1.png,232,216\n",
        )
        .unwrap();
        std::fs::write(
            dir.join("plugin_files/reference-training_1.csv"),
            "reference,100,107,A2_training_1.png,232,216\n",
        )
        .unwrap();
        std::fs::write(
            dir.join("plugin_files/change_reference-training_1.csv"),
            "reference,100,109,A1_training_1.png,232,216\n",
        )
        .unwrap();
        std::fs::write(
            dir.join("metadata.csv"),
            "Phase,PhasePart,TrialNumber,Stage,Animal,Apparatus,Perimeter,ChangeReference,ChangeReferenceImageName\nA,2,47,1,1,1,,training_1,A1_training_1\n",
        )
        .unwrap();
        dir
    }

    #[test]
    fn sync_dry_run_reports_without_creating_store() {
        let dir = temp_annotation_dir("dry_run");
        run_sync(AnnotateSyncArgs {
            annotation_dir: dir.clone(),
            import_only: false,
            export_only: false,
            dry_run: true,
            reviewed_externally: false,
        })
        .unwrap();
        assert!(!dir.join("bekiper/manifest.toml").exists());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn sync_import_then_noop_succeeds() {
        let dir = temp_annotation_dir("import_noop");
        run_sync(AnnotateSyncArgs {
            annotation_dir: dir.clone(),
            import_only: false,
            export_only: false,
            dry_run: false,
            reviewed_externally: false,
        })
        .unwrap();
        assert!(dir.join("bekiper/manifest.toml").exists());

        run_sync(AnnotateSyncArgs {
            annotation_dir: dir.clone(),
            import_only: false,
            export_only: false,
            dry_run: false,
            reviewed_externally: false,
        })
        .unwrap();
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn sync_export_only_exports_target_specific_corrections() {
        let dir = temp_rereferenced_annotation_dir("target_export");
        let store = CanonicalAnnotationStore::import_legacy(&dir).unwrap();
        store
            .materialize_target_perimeter(
                "metadata_row_2_animal_1_stage_1_part_2",
                &[PerimeterSpec::Rectangle {
                    label: "familiar".into(),
                    center_x: 70.0,
                    center_y: 80.0,
                    width: 30.0,
                    height: 40.0,
                }],
            )
            .unwrap();

        run_sync(AnnotateSyncArgs {
            annotation_dir: dir.clone(),
            import_only: false,
            export_only: true,
            dry_run: false,
            reviewed_externally: false,
        })
        .unwrap();

        assert!(
            dir.join("plugin_files/perimeter-rectangle-training_1__target__A1_training_1__session__metadata_row_2_animal_1_stage_1_part_2.csv")
                .exists()
        );
        let metadata = std::fs::read_to_string(dir.join("metadata.csv")).unwrap();
        assert!(metadata.contains(
            "training_1__target__A1_training_1__session__metadata_row_2_animal_1_stage_1_part_2"
        ));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn serve_help_lists_token_file_flag() {
        let mut command =
            <AnnotateServeArgs as clap::Args>::augment_args(clap::Command::new("serve"));
        let mut help = Vec::new();
        command.write_long_help(&mut help).unwrap();
        let help = String::from_utf8(help).unwrap();

        assert!(help.contains("--token-file <TOKEN_FILE>"), "{help}");
    }

    #[test]
    fn connect_help_lists_headless_oidc_password_flags() {
        let mut command =
            <AnnotateConnectArgs as clap::Args>::augment_args(clap::Command::new("connect"));
        let mut help = Vec::new();
        command.write_long_help(&mut help).unwrap();
        let help = String::from_utf8(help).unwrap();

        assert!(help.contains("--oidc-username <OIDC_USERNAME>"), "{help}");
        assert!(
            help.contains("--oidc-password-file <OIDC_PASSWORD_FILE>"),
            "{help}"
        );
    }
}
