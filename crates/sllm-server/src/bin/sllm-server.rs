use std::collections::BTreeMap;
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::Read;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const MAX_TLS_PEM_BYTES: usize = 1024 * 1024;

use sllm_core::{
    GEMMA4_RECOMMENDED_CONTEXT_TOKENS, KvCacheEncoding, KvCacheSelectionRequest,
    QWEN35_MOE_MODEL_FINGERPRINT, QWEN35_RECOMMENDED_CONTEXT_TOKENS, ReviewedModelLock,
    builtin_reviewed_model_lock, read_derived_gguf_lock, resolve_kv_cache_selection,
};
use sllm_server::{
    ChatGenerationBackendV1, CheckpointStartupConfigV1, ContextWindowStartupConfigV1,
    CredentialStoreV1, DraftStartupConfigV1, Gemma4BackendConfigV1, Gemma4ChatBackendV1,
    KvCacheExplicitSourceV1, KvCacheSelectionReportV1, ModelLifecycleConfigV1,
    ModelLifecycleDescriptorV1, ModelLifecycleLoadedV1, ModelLifecycleRegistryV1,
    ModelRegistryEntryV1, ModelRegistryV1, Phase41ProductionConfigV1, PrefixCacheStartupConfigV1,
    ProductionShutdownAuditV1, QwenAdapterArtifactConfigV1, QwenAdapterCatalogConfigV1,
    QwenBackendConfigV1, QwenChatBackendV1, ResumableStoreV1, SchedulerConfigV1, SchedulerV1,
    ServerConfigV1, ServerLifecycleStateV1, ServerLifecycleV1, ServerMetricsV1,
    build_dynamic_router_v1, build_router_v1, dynamic_model_plan_digest_preflight,
    qwen_adapter_catalog_identity_preflight, read_model_manifest_v1,
};

fn main() -> ExitCode {
    match parse_args().and_then(run) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("sllm-server: {error}");
            ExitCode::FAILURE
        }
    }
}

#[derive(Debug)]
struct Config {
    models: Option<PathBuf>,
    gguf: PathBuf,
    derived_lock: PathBuf,
    device_index: u32,
    target: String,
    listen: SocketAddr,
    model: String,
    api_key_env: Option<String>,
    api_key_file: Option<PathBuf>,
    cors_origins: Vec<String>,
    metrics: bool,
    resumable_sse: bool,
    replay_sessions: usize,
    replay_events: usize,
    tls_cert: Option<PathBuf>,
    tls_key: Option<PathBuf>,
    openwebui_compatibility: bool,
    queue_capacity: usize,
    event_capacity: usize,
    request_timeout: Duration,
    completion_timeout: Duration,
    shutdown_timeout: Duration,
    context_length: Option<u32>,
    kv_cache_encoding: Option<KvCacheEncoding>,
    phase41: Phase41ProductionConfigV1,
}

fn parse_args() -> Result<Config, String> {
    parse_args_from(env::args().skip(1))
}

fn parse_args_from<I>(args: I) -> Result<Config, String>
where
    I: IntoIterator,
    I::Item: Into<String>,
{
    let mut raw = args.into_iter().map(Into::into);
    let mut values = BTreeMap::new();
    while let Some(flag) = raw.next() {
        if flag == "--help" || flag == "-h" {
            return Err(usage().to_owned());
        }
        if !flag.starts_with("--") || values.contains_key(&flag) {
            return Err(format!("invalid or duplicate argument {flag}\n{}", usage()));
        }
        let value = raw
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        values.insert(flag, value);
    }
    let models = values.remove("--models").map(PathBuf::from);
    if models.as_ref().is_some_and(|path| !path.is_absolute()) {
        return Err("--models manifest path must be absolute".to_owned());
    }
    let legacy_gguf = values.remove("--gguf");
    let legacy_derived_lock = values.remove("--derived-lock");
    let legacy_device_index = values.remove("--device-index");
    let legacy_target = values.remove("--target");
    let requested_model = values.remove("--model");
    let (gguf, derived_lock, device_index, target, model) = if models.is_some() {
        if legacy_gguf.is_some()
            || legacy_derived_lock.is_some()
            || legacy_device_index.is_some()
            || legacy_target.is_some()
            || requested_model.is_some()
        {
            return Err(
                "--models is mutually exclusive with --gguf, --derived-lock, --device-index, --target, and --model"
                    .to_owned(),
            );
        }
        // Dynamic manifests supply these values per alias.  Keep the legacy
        // fields populated for the shared Config shape; the dynamic startup
        // branch must never consume them.
        (
            PathBuf::new(),
            PathBuf::new(),
            0,
            String::new(),
            "dynamic".to_owned(),
        )
    } else {
        let gguf = PathBuf::from(
            legacy_gguf.ok_or_else(|| "missing required argument --gguf".to_owned())?,
        );
        let derived_lock = PathBuf::from(
            legacy_derived_lock
                .ok_or_else(|| "missing required argument --derived-lock".to_owned())?,
        );
        let device_index = parse_value(
            &legacy_device_index
                .ok_or_else(|| "missing required argument --device-index".to_owned())?,
            "device index",
        )?;
        let target =
            legacy_target.ok_or_else(|| "missing required argument --target".to_owned())?;
        let model = requested_model.unwrap_or_else(|| "qwen3.5-4b".to_owned());
        (gguf, derived_lock, device_index, target, model)
    };
    let listen = parse_value(
        &values
            .remove("--listen")
            .unwrap_or_else(|| "127.0.0.1:8080".to_owned()),
        "listen address",
    )?;
    let api_key_env = values.remove("--api-key-env");
    let api_key_file = values.remove("--api-key-file").map(PathBuf::from);
    if api_key_env.is_some() && api_key_file.is_some() {
        return Err("--api-key-env and --api-key-file are mutually exclusive".to_owned());
    }
    let cors_origins = values
        .remove("--cors-origins")
        .map(|value| value.split(',').map(str::to_owned).collect::<Vec<_>>())
        .unwrap_or_default();
    if cors_origins.iter().any(String::is_empty) {
        return Err("--cors-origins must be a comma-separated list of exact origins".to_owned());
    }
    let metrics = parse_default(&mut values, "--metrics", false)?;
    let resumable_sse = parse_default(&mut values, "--resumable-sse", false)?;
    let replay_sessions = parse_default(&mut values, "--replay-sessions", 64_usize)?;
    let replay_events = parse_default(&mut values, "--replay-events", 256_usize)?;
    let tls_cert = values.remove("--tls-cert").map(PathBuf::from);
    let tls_key = values.remove("--tls-key").map(PathBuf::from);
    if tls_cert.is_some() != tls_key.is_some() {
        return Err("--tls-cert and --tls-key must be configured together".to_owned());
    }
    let compatibility_profile = values
        .remove("--compatibility-profile")
        .unwrap_or_else(|| "strict".to_owned());
    let openwebui_compatibility = match compatibility_profile.as_str() {
        "strict" => false,
        "openwebui" => true,
        _ => {
            return Err(format!(
                "compatibility profile must be strict or openwebui: {compatibility_profile}"
            ));
        }
    };
    let queue_capacity = parse_default(&mut values, "--queue-capacity", 8_usize)?;
    let event_capacity = parse_default(&mut values, "--event-capacity", 16_usize)?;
    let request_timeout = Duration::from_secs(parse_default(
        &mut values,
        "--request-timeout-seconds",
        300_u64,
    )?);
    let completion_timeout = Duration::from_secs(parse_default(
        &mut values,
        "--completion-timeout-seconds",
        120_u64,
    )?);
    let shutdown_timeout = Duration::from_secs(parse_default(
        &mut values,
        "--shutdown-timeout-seconds",
        30_u64,
    )?);
    let context_length = values
        .remove("--context-length")
        .map(|value| parse_value::<u32>(&value, "context length"))
        .transpose()?;
    if context_length == Some(0) {
        return Err("context length must be nonzero".to_owned());
    }
    let kv_cache_encoding = values
        .remove("--kv-cache-encoding")
        .map(|value| parse_kv_cache_encoding(&value))
        .transpose()?;

    let prefix_cache_mode = values
        .remove("--prefix-cache")
        .unwrap_or_else(|| "disabled".to_owned());
    let prefix_cache = match prefix_cache_mode.as_str() {
        "disabled" => {
            reject_disabled_options(
                &values,
                &[
                    "--prefix-cache-max-entries",
                    "--prefix-cache-max-tokens",
                    "--prefix-cache-max-resident-bytes",
                ],
                "--prefix-cache",
            )?;
            PrefixCacheStartupConfigV1::Disabled
        }
        "enabled" => PrefixCacheStartupConfigV1::Enabled {
            max_entries: parse_value(
                &take_required(&mut values, "--prefix-cache-max-entries")?,
                "prefix cache max entries",
            )?,
            max_logical_tokens: parse_value(
                &take_required(&mut values, "--prefix-cache-max-tokens")?,
                "prefix cache max tokens",
            )?,
            max_resident_bytes: parse_value(
                &take_required(&mut values, "--prefix-cache-max-resident-bytes")?,
                "prefix cache max resident bytes",
            )?,
        },
        value => {
            return Err(format!(
                "--prefix-cache must be disabled or enabled: {value}"
            ));
        }
    };
    let context_policy = values
        .remove("--context-policy")
        .unwrap_or_else(|| "disabled".to_owned());
    let context_window = match context_policy.as_str() {
        "disabled" => {
            reject_disabled_options(
                &values,
                &["--context-keep-prefix", "--context-keep-recent"],
                "--context-policy",
            )?;
            ContextWindowStartupConfigV1::Disabled
        }
        "keep-prefix-recent-v1" => ContextWindowStartupConfigV1::KeepPrefixRecentV1 {
            keep_prefix: parse_value(
                &take_required(&mut values, "--context-keep-prefix")?,
                "context keep-prefix",
            )?,
            keep_recent: parse_value(
                &take_required(&mut values, "--context-keep-recent")?,
                "context keep-recent",
            )?,
        },
        value => {
            return Err(format!(
                "--context-policy must be disabled or keep-prefix-recent-v1: {value}"
            ));
        }
    };
    let checkpoint_mode = values
        .remove("--checkpoint")
        .unwrap_or_else(|| "disabled".to_owned());
    let checkpoint = match checkpoint_mode.as_str() {
        "disabled" => {
            reject_disabled_options(
                &values,
                &[
                    "--checkpoint-directory",
                    "--checkpoint-quota-bytes",
                    "--checkpoint-load",
                    "--checkpoint-save",
                ],
                "--checkpoint",
            )?;
            CheckpointStartupConfigV1::Disabled
        }
        "enabled" => CheckpointStartupConfigV1::Enabled {
            directory: PathBuf::from(take_required(&mut values, "--checkpoint-directory")?),
            quota_bytes: parse_value(
                &take_required(&mut values, "--checkpoint-quota-bytes")?,
                "checkpoint quota bytes",
            )?,
            load_name: values.remove("--checkpoint-load"),
            save_name: values.remove("--checkpoint-save"),
        },
        value => return Err(format!("--checkpoint must be disabled or enabled: {value}")),
    };
    let draft_mode = values
        .remove("--draft")
        .unwrap_or_else(|| "disabled".to_owned());
    let draft = match draft_mode.as_str() {
        "disabled" | "mtp-auto" => {
            reject_disabled_options(
                &values,
                &[
                    "--draft-ngram-order",
                    "--draft-width",
                    "--draft-model-identity",
                    "--draft-tokenizer-identity",
                    "--draft-vocabulary-size",
                ],
                "--draft",
            )?;
            if draft_mode == "disabled" {
                DraftStartupConfigV1::Disabled
            } else {
                DraftStartupConfigV1::MtpAuto
            }
        }
        "ngram" => DraftStartupConfigV1::Ngram {
            order: parse_value(
                &take_required(&mut values, "--draft-ngram-order")?,
                "draft ngram order",
            )?,
            width: parse_value(&take_required(&mut values, "--draft-width")?, "draft width")?,
        },
        "external" => DraftStartupConfigV1::External {
            model_identity: take_required(&mut values, "--draft-model-identity")?,
            tokenizer_identity: take_required(&mut values, "--draft-tokenizer-identity")?,
            vocabulary_size: parse_value(
                &take_required(&mut values, "--draft-vocabulary-size")?,
                "draft vocabulary size",
            )?,
            width: parse_value(&take_required(&mut values, "--draft-width")?, "draft width")?,
        },
        value => {
            return Err(format!(
                "--draft must be disabled, mtp-auto, ngram, or external: {value}"
            ));
        }
    };
    let phase41 = Phase41ProductionConfigV1 {
        prefix_cache,
        context_window,
        checkpoint,
        draft,
    };
    phase41
        .validate()
        .map_err(|error| format!("Phase41 configuration invalid: {error}"))?;
    if let Some(flag) = values.keys().next() {
        return Err(format!("unknown argument {flag}\n{}", usage()));
    }
    Ok(Config {
        models,
        gguf,
        derived_lock,
        device_index,
        target,
        listen,
        model,
        api_key_env,
        api_key_file,
        cors_origins,
        metrics,
        resumable_sse,
        replay_sessions,
        replay_events,
        tls_cert,
        tls_key,
        openwebui_compatibility,
        queue_capacity,
        event_capacity,
        request_timeout,
        completion_timeout,
        shutdown_timeout,
        context_length,
        kv_cache_encoding,
        phase41,
    })
}

fn run(config: Config) -> Result<(), String> {
    config
        .phase41
        .validate_startup()
        .map_err(|error| format!("Phase41 startup configuration invalid: {error}"))?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|error| format!("Tokio runtime construction failed: {error}"))?;
    runtime.block_on(async move {
        let credentials = if let Some(path) = config.api_key_file.as_deref() {
            CredentialStoreV1::from_key_file(path)
                .map_err(|error| format!("API key file validation failed: {error}"))?
        } else if let Some(name) = config.api_key_env.as_deref() {
            let token = env::var(name)
                .map_err(|_| format!("API key environment variable {name} is absent"))?;
            CredentialStoreV1::from_user_key(token)
                .map_err(|error| format!("API key validation failed: {error}"))?
        } else {
            CredentialStoreV1::open()
        };
        let tls = match (config.tls_cert.as_deref(), config.tls_key.as_deref()) {
            (Some(cert), Some(key)) => {
                let cert = read_tls_cert_file(cert)?;
                let key = read_private_key_file(key)?;
                Some(
                    axum_server::tls_rustls::RustlsConfig::from_pem(cert, key)
                        .await
                        .map_err(|error| format!("TLS certificate/key validation failed: {error}"))?,
                )
            }
            (None, None) => None,
            _ => unreachable!("TLS certificate/key pairing was checked during parsing"),
        };
        if let Some(manifest_path) = config.models.clone() {
            return run_dynamic_manifest(config, manifest_path, credentials, tls).await;
        }
        let lifecycle = ServerLifecycleV1::new(ServerLifecycleStateV1::Loading);
        let derived = read_derived_gguf_lock(&config.derived_lock)
            .map_err(|error| format!("derived GGUF lock validation failed: {error}"))?;
        let gguf_moe = derived.semantic_model_id.starts_with("qwen35moe:");
        let reviewed = if gguf_moe {
            None
        } else {
            Some(
                builtin_reviewed_model_lock(&derived.source_lock_fingerprints)
                    .map_err(|error| format!("built-in model lock resolution failed: {error}"))?,
            )
        };
        let (backend, startup_kv_selection) = if gguf_moe {
            let (kv_cache_resolved_selection, kv_cache_selection) = resolve_server_kv_selection(
                config.kv_cache_encoding,
                &config.target,
                QWEN35_MOE_MODEL_FINGERPRINT,
                false,
                KvCacheExplicitSourceV1::Process,
            )?;
            let kv_cache_encoding = kv_cache_resolved_selection.resolved();
            let backend_config = QwenBackendConfigV1 {
                gguf_path: config.gguf,
                derived_lock_path: config.derived_lock,
                device_index: config.device_index,
                target: config.target,
                completion_timeout: config.completion_timeout,
                shutdown_timeout: config.shutdown_timeout,
                context_length: config
                    .context_length
                    .unwrap_or(QWEN35_RECOMMENDED_CONTEXT_TOKENS as u32),
                kv_cache_encoding,
                kv_cache_resolved_selection: Some(kv_cache_resolved_selection),
                kv_cache_selection: Some(kv_cache_selection.clone()),
                phase41: config.phase41.clone(),
                adapter_catalog: None,
            };
            (
                ActiveBackend::Qwen(Arc::new(
                    QwenChatBackendV1::open(backend_config)
                        .map_err(|error| error.to_string())?,
                )),
                kv_cache_selection,
            )
        } else {
            match reviewed.expect("non-MoE GGUF resolved a reviewed lock") {
                ReviewedModelLock::Qwen35(lock) => {
                    let (kv_cache_resolved_selection, kv_cache_selection) =
                        resolve_server_kv_selection(
                        config.kv_cache_encoding,
                        &config.target,
                        lock.fingerprint(),
                        true,
                        KvCacheExplicitSourceV1::Process,
                        )?;
                    let kv_cache_encoding = kv_cache_resolved_selection.resolved();
                    let backend_config = QwenBackendConfigV1 {
                        gguf_path: config.gguf,
                        derived_lock_path: config.derived_lock,
                        device_index: config.device_index,
                        target: config.target,
                        completion_timeout: config.completion_timeout,
                        shutdown_timeout: config.shutdown_timeout,
                        context_length: config
                            .context_length
                            .unwrap_or(QWEN35_RECOMMENDED_CONTEXT_TOKENS as u32),
                        kv_cache_encoding,
                        kv_cache_resolved_selection: Some(kv_cache_resolved_selection),
                        kv_cache_selection: Some(kv_cache_selection.clone()),
                        phase41: config.phase41.clone(),
                        adapter_catalog: None,
                    };
                    (
                        ActiveBackend::Qwen(Arc::new(
                            QwenChatBackendV1::open(backend_config)
                                .map_err(|error| error.to_string())?,
                        )),
                        kv_cache_selection,
                    )
                }
                ReviewedModelLock::Gemma4(_) => {
                    if config
                        .kv_cache_encoding
                        .is_some_and(|encoding| encoding != KvCacheEncoding::Fp16)
                    {
                        return Err(
                            "--kv-cache-encoding applies to Qwen; Gemma uses its fixed recipe"
                                .to_owned(),
                        );
                    }
                    let kv_cache_selection = KvCacheSelectionReportV1 {
                        requested: config
                            .kv_cache_encoding
                            .map(KvCacheEncoding::canonical_name)
                            .unwrap_or("auto")
                            .to_owned(),
                        resolved: KvCacheEncoding::Fp8E4M3FnStatic
                            .canonical_name()
                            .to_owned(),
                        selection_source: "model-recipe-explicit".to_owned(),
                        reason: "Gemma uses its fixed reviewed KV recipe".to_owned(),
                        physical_variant: None,
                        descriptor_id: None,
                        policy_version: sllm_core::KV_CACHE_SELECTION_POLICY_VERSION_V1,
                    };
                    let backend_config = Gemma4BackendConfigV1 {
                        gguf_path: config.gguf,
                        derived_lock_path: config.derived_lock,
                        device_index: config.device_index,
                        target: config.target,
                        completion_timeout: config.completion_timeout,
                        shutdown_timeout: config.shutdown_timeout,
                        context_length: config
                            .context_length
                            .unwrap_or(GEMMA4_RECOMMENDED_CONTEXT_TOKENS as u32),
                        phase41: config.phase41.clone(),
                    };
                    (
                        ActiveBackend::Gemma(Arc::new(
                            Gemma4ChatBackendV1::open(backend_config)
                                .map_err(|error| error.to_string())?,
                        )),
                        kv_cache_selection,
                    )
                }
            }
        };
        let shutdown_guard = BackendShutdownGuard::new(&backend);
        if let Some(warning) = context_length_warning(
            backend.context_length(),
            backend.recommended_context_tokens(),
        ) {
            eprintln!("{warning}");
        }
        let backend_trait = backend.as_trait();
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| "system clock is before UNIX epoch".to_owned())?
            .as_secs();
        let entry = ModelRegistryEntryV1::new(
            config.model,
            created,
            "sllm",
            backend.model_fingerprint(),
            backend_trait,
        )
        .map_err(|error| error.to_string())?;
        let registry = ModelRegistryV1::new(vec![entry]).map_err(|error| error.to_string())?;
        let model_aliases = registry
            .entries()
            .iter()
            .map(|entry| entry.alias().to_owned())
            .collect::<Vec<_>>();
        let scheduler = SchedulerV1::new(
            SchedulerConfigV1::new(
                config.queue_capacity,
                config.event_capacity,
                config.request_timeout,
            )
            .map_err(|error| error.to_string())?,
        );
        let mut server_config = if config.openwebui_compatibility {
            ServerConfigV1::openwebui_compatible(None)
        } else {
            ServerConfigV1::new(None)
        }
        .map_err(|error| error.to_string())?
        .with_credentials(credentials)
        .with_lifecycle(lifecycle.clone())
        .with_cors_origins(&config.cors_origins)
        .map_err(|error| error.to_string())?;
        if config.metrics {
            server_config = server_config.with_metrics(
                ServerMetricsV1::new(model_aliases).map_err(|error| error.to_string())?,
            );
        }
        if config.resumable_sse {
            server_config = server_config.with_resumable_store(
                ResumableStoreV1::new(config.replay_sessions, config.replay_events)
                    .map_err(|error| format!("resumable SSE configuration failed: {error}"))?,
            );
        }
        let router = build_router_v1(registry, scheduler.clone(), server_config);
        let listener = std::net::TcpListener::bind(config.listen)
            .map_err(|error| format!("listen failed: {error}"))?;
        listener
            .set_nonblocking(true)
            .map_err(|error| format!("listen socket setup failed: {error}"))?;
        let address = listener
            .local_addr()
            .map_err(|error| format!("local address query failed: {error}"))?;
        lifecycle.transition(ServerLifecycleStateV1::Ready);
        println!(
            "{}",
            serde_json::json!({
                "event": "ready",
                "listen": address.to_string(),
                "target": backend.target(),
                "model_fingerprint": backend.model_fingerprint(),
                "compatibility_profile": if config.openwebui_compatibility { "openwebui" } else { "strict" },
                "context_length": backend.context_length(),
                "official_recommended_context_tokens": backend.recommended_context_tokens(),
                "kv_cache_selection": startup_kv_selection,
                "tls": tls.is_some(),
                "authentication": config.api_key_env.is_some() || config.api_key_file.is_some(),
                "metrics": config.metrics,
                "resumable_sse": config.resumable_sse,
                "cors": !config.cors_origins.is_empty(),
            })
        );
        let handle = axum_server::Handle::new();
        let signal_handle = handle.clone();
        let signal_scheduler = scheduler.clone();
        let signal_lifecycle = lifecycle.clone();
        let shutdown_timeout = config.shutdown_timeout;
        let signal_task = tokio::spawn(async move {
            if tokio::signal::ctrl_c().await.is_ok() {
                signal_lifecycle.transition(ServerLifecycleStateV1::Draining);
                signal_scheduler.shutdown();
                signal_handle.graceful_shutdown(Some(shutdown_timeout));
            }
        });
        let serve_result = if let Some(tls) = tls {
            match axum_server::from_tcp_rustls(listener, tls) {
                Ok(server) => {
                    server
                        .handle(handle)
                        .serve(router.into_make_service())
                        .await
                }
                Err(error) => Err(error),
            }
        } else {
            match axum_server::from_tcp(listener) {
                Ok(server) => {
                    server
                        .handle(handle)
                        .serve(router.into_make_service())
                        .await
                }
                Err(error) => Err(error),
            }
        };
        signal_task.abort();
        lifecycle.transition(ServerLifecycleStateV1::Draining);
        scheduler.shutdown();
        if let Err(error) = serve_result {
            lifecycle.transition(ServerLifecycleStateV1::Failed);
            return Err(format!("HTTP server failed: {error}"));
        }
        let report = shutdown_guard
            .shutdown()
            .map_err(|error| error.to_string())?;
        lifecycle.transition(ServerLifecycleStateV1::Shutdown);
        println!(
            "{}",
            serde_json::json!({"event": "shutdown_audit", "report": report})
        );
        Ok(())
    })
}

fn read_tls_cert_file(path: &Path) -> Result<Vec<u8>, String> {
    read_tls_file(path, false)
}

fn read_private_key_file(path: &Path) -> Result<Vec<u8>, String> {
    read_tls_file(path, true)
}

fn read_tls_file(path: &Path, private: bool) -> Result<Vec<u8>, String> {
    let kind = if private {
        "private key"
    } else {
        "certificate"
    };
    let metadata = fs::symlink_metadata(path)
        .map_err(|_| format!("TLS {kind} file could not be inspected"))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(format!("TLS {kind} must be a regular, non-symlink file"));
    }
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

        options.custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW);
        if private && metadata.permissions().mode() & 0o077 != 0 {
            return Err("TLS private key permissions are too broad".to_owned());
        }
    }
    let file: File = options
        .open(path)
        .map_err(|_| format!("TLS {kind} file could not be opened"))?;
    let opened = file
        .metadata()
        .map_err(|_| format!("TLS {kind} file could not be inspected"))?;
    if !opened.is_file() {
        return Err(format!("TLS {kind} must be a regular, non-symlink file"));
    }
    #[cfg(unix)]
    if private {
        use std::os::unix::fs::PermissionsExt;
        if opened.permissions().mode() & 0o077 != 0 {
            return Err("TLS private key permissions are too broad".to_owned());
        }
    }
    if opened.len() > MAX_TLS_PEM_BYTES as u64 {
        return Err(format!("TLS {kind} file is too large"));
    }
    let mut bytes = Vec::with_capacity(opened.len() as usize);
    file.take((MAX_TLS_PEM_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|_| format!("TLS {kind} file could not be read"))?;
    if bytes.len() > MAX_TLS_PEM_BYTES {
        return Err(format!("TLS {kind} file is too large"));
    }
    Ok(bytes)
}

fn context_length_warning(
    context_length: u32,
    recommended_context_tokens: u32,
) -> Option<serde_json::Value> {
    (context_length > recommended_context_tokens).then(|| {
        serde_json::json!({
            "event": "warning",
            "kind": "context_length_exceeds_recommended",
            "context_length": context_length,
            "official_recommended_context_tokens": recommended_context_tokens,
            "message": format!(
                "the model's official recommended context is {recommended_context_tokens} tokens; {context_length} was requested and output quality beyond the recommendation is not guaranteed",
            ),
        })
    })
}

/// Start the manifest-backed dynamic model registry.  All locks and identities
/// are admitted before the first backend is opened; the loader then repeats the
/// lock admission immediately before each GPU load.
async fn run_dynamic_manifest(
    config: Config,
    manifest_path: PathBuf,
    credentials: CredentialStoreV1,
    tls: Option<axum_server::tls_rustls::RustlsConfig>,
) -> Result<(), String> {
    let manifest = read_model_manifest_v1(&manifest_path)
        .map_err(|error| format!("model manifest validation failed: {error}"))?;
    let mut descriptors = Vec::with_capacity(manifest.models().len());
    let mut entries = BTreeMap::new();
    let mut resident_quota = 0_u64;
    for entry in manifest.models() {
        let derived = read_derived_gguf_lock(entry.derived_lock()).map_err(|error| {
            format!(
                "model {} derived lock validation failed: {error}",
                entry.alias()
            )
        })?;
        let model_identity = dynamic_model_identity(&derived)?;
        let adapter_identity = dynamic_adapter_identity(entry)?;
        let plan_identity =
            dynamic_model_plan_digest_preflight(entry.gguf(), &derived).map_err(|error| {
                format!(
                    "model {} verified weight-plan preflight failed: {error}",
                    entry.alias()
                )
            })?;
        resident_quota = resident_quota
            .checked_add(entry.declared_resident_bytes())
            .ok_or_else(|| "model manifest resident-byte quota overflowed".to_owned())?;
        descriptors.push(
            ModelLifecycleDescriptorV1::new(
                entry.alias(),
                model_identity,
                plan_identity,
                adapter_identity,
                entry.declared_resident_bytes(),
            )
            .map_err(|error| {
                format!(
                    "model {} lifecycle descriptor invalid: {error:?}",
                    entry.alias()
                )
            })?,
        );
        entries.insert(entry.alias().to_owned(), entry.clone());
    }
    let lifecycle_config = ModelLifecycleConfigV1::new(resident_quota)
        .and_then(|value| value.with_timeouts(config.completion_timeout, config.shutdown_timeout))
        .map_err(|error| format!("dynamic lifecycle configuration invalid: {error:?}"))?;
    let entries = Arc::new(entries);
    let active = Arc::new(std::sync::Mutex::new(
        BTreeMap::<String, Arc<ActiveBackend>>::new(),
    ));
    let load_entries = Arc::clone(&entries);
    let load_active = Arc::clone(&active);
    let load_phase41 = config.phase41.clone();
    let load_kv = config.kv_cache_encoding;
    let load_context_length = config.context_length;
    let load_completion_timeout = config.completion_timeout;
    let load_shutdown_timeout = config.shutdown_timeout;
    let loader = move |descriptor: &ModelLifecycleDescriptorV1| {
        let entry = load_entries
            .get(descriptor.alias())
            .ok_or_else(|| "manifest alias disappeared".to_owned())?;
        let derived = read_derived_gguf_lock(entry.derived_lock())
            .map_err(|error| format!("derived lock validation failed: {error}"))?;
        let model_identity = dynamic_model_identity(&derived)?;
        let plan_identity_before = dynamic_model_plan_digest_preflight(entry.gguf(), &derived)
            .map_err(|error| format!("verified weight-plan preflight failed: {error}"))?;
        if model_identity != descriptor.identity().model_identity()
            || plan_identity_before != descriptor.identity().plan_identity()
        {
            return Err("model identity changed since manifest preflight".to_owned());
        }
        let is_moe = derived.semantic_model_id.starts_with("qwen35moe:");
        let adapter_identity_before = dynamic_adapter_identity(entry)?;
        if adapter_identity_before != descriptor.identity().adapter_identity() {
            return Err("adapter catalog identity changed since manifest preflight".to_owned());
        }
        let adapter_catalog = dynamic_adapter_catalog(entry)?;
        let backend = if is_moe {
            if adapter_catalog.is_some() {
                return Err("MoE models do not support adapter catalogs".to_owned());
            }
            let (requested_kv, explicit_source) =
                effective_server_kv_request(entry.kv_cache_encoding(), load_kv);
            let (kv_cache_resolved_selection, kv_cache_selection) = resolve_server_kv_selection(
                requested_kv,
                entry.target(),
                QWEN35_MOE_MODEL_FINGERPRINT,
                false,
                explicit_source,
            )?;
            let kv_cache_encoding = kv_cache_resolved_selection.resolved();
            ActiveBackend::Qwen(Arc::new(
                QwenChatBackendV1::open(QwenBackendConfigV1 {
                    gguf_path: entry.gguf().to_owned(),
                    derived_lock_path: entry.derived_lock().to_owned(),
                    device_index: entry.device_index(),
                    target: entry.target().to_owned(),
                    completion_timeout: load_completion_timeout,
                    shutdown_timeout: load_shutdown_timeout,
                    context_length: load_context_length
                        .unwrap_or(QWEN35_RECOMMENDED_CONTEXT_TOKENS as u32),
                    kv_cache_encoding,
                    kv_cache_resolved_selection: Some(kv_cache_resolved_selection),
                    kv_cache_selection: Some(kv_cache_selection),
                    phase41: load_phase41.clone(),
                    adapter_catalog: None,
                })
                .map_err(|error| error.to_string())?,
            ))
        } else {
            match builtin_reviewed_model_lock(&derived.source_lock_fingerprints)
                .map_err(|error| format!("reviewed model lock resolution failed: {error}"))?
            {
                ReviewedModelLock::Qwen35(lock) => {
                    let (requested_kv, explicit_source) =
                        effective_server_kv_request(entry.kv_cache_encoding(), load_kv);
                    let (kv_cache_resolved_selection, kv_cache_selection) =
                        resolve_server_kv_selection(
                            requested_kv,
                            entry.target(),
                            lock.fingerprint(),
                            true,
                            explicit_source,
                        )?;
                    let kv_cache_encoding = kv_cache_resolved_selection.resolved();
                    ActiveBackend::Qwen(Arc::new(
                        QwenChatBackendV1::open(QwenBackendConfigV1 {
                            gguf_path: entry.gguf().to_owned(),
                            derived_lock_path: entry.derived_lock().to_owned(),
                            device_index: entry.device_index(),
                            target: entry.target().to_owned(),
                            completion_timeout: load_completion_timeout,
                            shutdown_timeout: load_shutdown_timeout,
                            context_length: load_context_length
                                .unwrap_or(QWEN35_RECOMMENDED_CONTEXT_TOKENS as u32),
                            kv_cache_encoding,
                            kv_cache_resolved_selection: Some(kv_cache_resolved_selection),
                            kv_cache_selection: Some(kv_cache_selection),
                            phase41: load_phase41.clone(),
                            adapter_catalog,
                        })
                        .map_err(|error| error.to_string())?,
                    ))
                }
                ReviewedModelLock::Gemma4(lock) => {
                    if adapter_catalog.is_some() {
                        return Err("Gemma models do not support adapter catalogs".to_owned());
                    }
                    let (requested_kv, explicit_source) =
                        effective_server_kv_request(entry.kv_cache_encoding(), load_kv);
                    let (kv_cache_resolved_selection, _) = resolve_server_kv_selection(
                        requested_kv,
                        entry.target(),
                        lock.fingerprint(),
                        false,
                        explicit_source,
                    )?;
                    let kv_cache_encoding = kv_cache_resolved_selection.resolved();
                    if kv_cache_encoding != KvCacheEncoding::Fp16 {
                        return Err(
                            "--kv-cache-encoding applies to Qwen; Gemma uses its fixed recipe"
                                .to_owned(),
                        );
                    }
                    ActiveBackend::Gemma(Arc::new(
                        Gemma4ChatBackendV1::open(Gemma4BackendConfigV1 {
                            gguf_path: entry.gguf().to_owned(),
                            derived_lock_path: entry.derived_lock().to_owned(),
                            device_index: entry.device_index(),
                            target: entry.target().to_owned(),
                            completion_timeout: load_completion_timeout,
                            shutdown_timeout: load_shutdown_timeout,
                            context_length: load_context_length
                                .unwrap_or(GEMMA4_RECOMMENDED_CONTEXT_TOKENS as u32),
                            phase41: load_phase41.clone(),
                        })
                        .map_err(|error| error.to_string())?,
                    ))
                }
            }
        };
        let backend = Arc::new(backend);
        if backend.plan_digest() != descriptor.identity().plan_identity() {
            return Err(fail_after_backend_open(
                &backend,
                "loaded backend weight-plan identity differs",
            ));
        }
        let backend_adapter_identity = backend
            .adapter_catalog_identity()
            .map_err(|error| fail_after_backend_open(&backend, error.to_string()))?;
        if backend_adapter_identity != descriptor.identity().adapter_identity() {
            return Err(fail_after_backend_open(
                &backend,
                "loaded backend adapter catalog identity differs",
            ));
        }
        let derived_after = read_derived_gguf_lock(entry.derived_lock())
            .map_err(|error| fail_after_backend_open(&backend, error.to_string()))?;
        let model_identity_after = dynamic_model_identity(&derived_after)
            .map_err(|error| fail_after_backend_open(&backend, error))?;
        let plan_identity_after = dynamic_model_plan_digest_preflight(entry.gguf(), &derived_after)
            .map_err(|error| fail_after_backend_open(&backend, error.to_string()))?;
        let adapter_identity_after = dynamic_adapter_identity(entry)
            .map_err(|error| fail_after_backend_open(&backend, error))?;
        if model_identity_after != descriptor.identity().model_identity()
            || plan_identity_after != descriptor.identity().plan_identity()
            || adapter_identity_after != descriptor.identity().adapter_identity()
        {
            return Err(fail_after_backend_open(
                &backend,
                "model, weight-plan, or adapter identity changed during backend load",
            ));
        }
        let resident_bytes = backend.resident_bytes();
        if resident_bytes != descriptor.declared_resident_bytes() {
            return Err(fail_after_backend_open(
                &backend,
                format!(
                    "model {} resident bytes differ from manifest declaration",
                    descriptor.alias()
                ),
            ));
        }
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| fail_after_backend_open(&backend, "system clock is before UNIX epoch"))?
            .as_secs();
        let owner = Arc::new(
            ModelRegistryEntryV1::new(
                descriptor.alias(),
                created,
                "sllm-dynamic",
                backend.model_fingerprint(),
                backend.as_trait(),
            )
            .map_err(|error| fail_after_backend_open(&backend, error.to_string()))?,
        );
        let loaded = ModelLifecycleLoadedV1::new(
            owner,
            resident_bytes,
            descriptor.identity().model_identity().to_owned(),
            descriptor.identity().plan_identity().to_owned(),
            descriptor.identity().adapter_identity().to_owned(),
        )
        .map_err(|error| {
            fail_after_backend_open(
                &backend,
                format!("dynamic loaded identity invalid: {error:?}"),
            )
        })?;
        load_active
            .lock()
            .map_err(|_| {
                fail_after_backend_open(&backend, "dynamic active backend map is poisoned")
            })?
            .insert(descriptor.alias().to_owned(), backend);
        Ok(loaded)
    };
    let shutdown_active = Arc::clone(&active);
    let shutdown = move |loaded: ModelLifecycleLoadedV1| {
        let alias = loaded.owner().alias().to_owned();
        let backend = shutdown_active
            .lock()
            .map_err(|_| "dynamic active backend map is poisoned".to_owned())?
            .remove(&alias)
            .ok_or_else(|| "active backend owner is missing".to_owned())?;
        let report = match backend.shutdown() {
            Ok(report) => report,
            Err(error) => {
                shutdown_active
                    .lock()
                    .map_err(|_| "dynamic active backend map is poisoned".to_owned())?
                    .insert(alias, backend);
                return Err(error.to_string());
            }
        };
        if report.final_current_bytes != 0
            || report.retryable_cleanup != 0
            || report.durable_quarantine != 0
        {
            return Err("backend shutdown audit was not clean".to_owned());
        }
        Ok::<(), String>(())
    };
    let lifecycle =
        ModelLifecycleRegistryV1::new_with_fns(descriptors, loader, shutdown, lifecycle_config)
            .map_err(|error| format!("dynamic lifecycle registry invalid: {error:?}"))?;
    let lifecycle_state = ServerLifecycleV1::new(ServerLifecycleStateV1::Loading);
    for entry in manifest.models().iter().filter(|entry| entry.preload()) {
        if let Err(error) = lifecycle.preload(entry.alias()) {
            let cleanup = unload_dynamic_registry(&lifecycle);
            lifecycle_state.transition(ServerLifecycleStateV1::Failed);
            return Err(match cleanup {
                Some(cleanup) => format!(
                    "preload of model {} failed: {error:?}; cleanup failed: {cleanup}",
                    entry.alias()
                ),
                None => format!("preload of model {} failed: {error:?}", entry.alias()),
            });
        }
    }
    let aliases = lifecycle.configured_aliases();
    let scheduler = SchedulerV1::new(
        SchedulerConfigV1::new(
            config.queue_capacity,
            config.event_capacity,
            config.request_timeout,
        )
        .map_err(|error| error.to_string())?,
    );
    let mut server_config = if config.openwebui_compatibility {
        ServerConfigV1::openwebui_compatible(None)
    } else {
        ServerConfigV1::new(None)
    }
    .map_err(|error| error.to_string())?
    .with_credentials(credentials)
    .with_lifecycle(lifecycle_state.clone())
    .with_cors_origins(&config.cors_origins)
    .map_err(|error| error.to_string())?;
    if config.metrics {
        server_config = server_config.with_metrics(
            ServerMetricsV1::new(aliases.clone()).map_err(|error| error.to_string())?,
        );
    }
    if config.resumable_sse {
        server_config = server_config.with_resumable_store(
            ResumableStoreV1::new(config.replay_sessions, config.replay_events)
                .map_err(|error| format!("resumable SSE configuration failed: {error}"))?,
        );
    }
    let router = build_dynamic_router_v1(lifecycle.clone(), scheduler.clone(), server_config);
    let listener = std::net::TcpListener::bind(config.listen)
        .map_err(|error| format!("listen failed: {error}"))?;
    listener
        .set_nonblocking(true)
        .map_err(|error| format!("listen socket setup failed: {error}"))?;
    let address = listener
        .local_addr()
        .map_err(|error| format!("local address query failed: {error}"))?;
    lifecycle_state.transition(ServerLifecycleStateV1::Ready);
    println!(
        "{}",
        serde_json::json!({
            "event": "ready",
            "listen": address.to_string(),
            "dynamic_models": aliases,
            "compatibility_profile": if config.openwebui_compatibility { "openwebui" } else { "strict" },
            "tls": tls.is_some(),
            "authentication": config.api_key_env.is_some() || config.api_key_file.is_some(),
            "metrics": config.metrics,
            "resumable_sse": config.resumable_sse,
            "cors": !config.cors_origins.is_empty(),
        })
    );
    let handle = axum_server::Handle::new();
    let signal_handle = handle.clone();
    let signal_scheduler = scheduler.clone();
    let signal_lifecycle = lifecycle_state.clone();
    let shutdown_timeout = config.shutdown_timeout;
    let signal_task = tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            signal_lifecycle.transition(ServerLifecycleStateV1::Draining);
            signal_scheduler.shutdown();
            signal_handle.graceful_shutdown(Some(shutdown_timeout));
        }
    });
    let serve_result = if let Some(tls) = tls {
        match axum_server::from_tcp_rustls(listener, tls) {
            Ok(server) => {
                server
                    .handle(handle)
                    .serve(router.into_make_service())
                    .await
            }
            Err(error) => Err(error),
        }
    } else {
        match axum_server::from_tcp(listener) {
            Ok(server) => {
                server
                    .handle(handle)
                    .serve(router.into_make_service())
                    .await
            }
            Err(error) => Err(error),
        }
    };
    signal_task.abort();
    lifecycle_state.transition(ServerLifecycleStateV1::Draining);
    scheduler.shutdown();
    let shutdown_aliases = lifecycle.configured_aliases();
    let shutdown_error = unload_dynamic_registry(&lifecycle);
    let clean = shutdown_error.is_none()
        && lifecycle.loaded_count() == 0
        && active
            .lock()
            .map(|backends| backends.is_empty())
            .unwrap_or(false);
    println!(
        "{}",
        serde_json::json!({
            "event": "shutdown_audit",
            "dynamic": true,
            "aliases": shutdown_aliases,
            "count": manifest.models().len(),
            "clean": clean,
        })
    );
    if let Some(error) = shutdown_error {
        lifecycle_state.transition(ServerLifecycleStateV1::Failed);
        return Err(error);
    }
    if let Err(error) = serve_result {
        lifecycle_state.transition(ServerLifecycleStateV1::Failed);
        return Err(format!("HTTP server failed: {error}"));
    }
    lifecycle_state.transition(ServerLifecycleStateV1::Shutdown);
    Ok(())
}

fn unload_dynamic_registry(lifecycle: &ModelLifecycleRegistryV1) -> Option<String> {
    let mut error = None;
    for alias in lifecycle.configured_aliases() {
        if let Err(failure) = lifecycle.unload(&alias) {
            error = Some(format!(
                "dynamic model {alias} shutdown failed: {failure:?}"
            ));
        }
    }
    error
}

fn dynamic_model_identity(derived: &sllm_core::DerivedGgufLock) -> Result<String, String> {
    if derived.semantic_model_id.starts_with("qwen35moe:") {
        Ok(QWEN35_MOE_MODEL_FINGERPRINT.to_owned())
    } else {
        Ok(
            builtin_reviewed_model_lock(&derived.source_lock_fingerprints)
                .map_err(|error| format!("built-in model lock resolution failed: {error}"))?
                .fingerprint()
                .to_owned(),
        )
    }
}

fn dynamic_adapter_identity(entry: &sllm_server::ModelManifestEntryV1) -> Result<String, String> {
    let catalog = dynamic_adapter_catalog(entry)?;
    let empty = QwenAdapterCatalogConfigV1::default();
    qwen_adapter_catalog_identity_preflight(catalog.as_ref().unwrap_or(&empty))
        .map_err(|error| error.to_string())
}

fn dynamic_adapter_catalog(
    entry: &sllm_server::ModelManifestEntryV1,
) -> Result<Option<QwenAdapterCatalogConfigV1>, String> {
    if entry.adapters().is_empty() && entry.control_vectors().is_empty() {
        return Ok(None);
    }
    Ok(Some(QwenAdapterCatalogConfigV1 {
        lora: entry
            .adapters()
            .iter()
            .map(|artifact| QwenAdapterArtifactConfigV1 {
                alias: artifact.alias().to_owned(),
                lock_path: artifact.lock().to_owned(),
                payload_path: artifact.payload().to_owned(),
            })
            .collect(),
        control_vectors: entry
            .control_vectors()
            .iter()
            .map(|artifact| QwenAdapterArtifactConfigV1 {
                alias: artifact.alias().to_owned(),
                lock_path: artifact.lock().to_owned(),
                payload_path: artifact.payload().to_owned(),
            })
            .collect(),
    }))
}

enum ActiveBackend {
    Qwen(Arc<QwenChatBackendV1>),
    Gemma(Arc<Gemma4ChatBackendV1>),
}

fn fail_after_backend_open(backend: &Arc<ActiveBackend>, reason: impl Into<String>) -> String {
    let reason = reason.into();
    match backend.shutdown() {
        Ok(report)
            if report.final_current_bytes == 0
                && report.retryable_cleanup == 0
                && report.durable_quarantine == 0 =>
        {
            reason
        }
        Ok(report) => format!(
            "{reason}; backend cleanup audit was not clean (current={}, retryable={}, quarantine={})",
            report.final_current_bytes, report.retryable_cleanup, report.durable_quarantine
        ),
        Err(error) => format!("{reason}; backend cleanup failed: {error}"),
    }
}

impl ActiveBackend {
    fn as_trait(&self) -> Arc<dyn ChatGenerationBackendV1> {
        match self {
            Self::Qwen(backend) => backend.clone(),
            Self::Gemma(backend) => backend.clone(),
        }
    }

    fn model_fingerprint(&self) -> &str {
        match self {
            Self::Qwen(backend) => backend.model_fingerprint(),
            Self::Gemma(backend) => backend.model_fingerprint(),
        }
    }

    fn plan_digest(&self) -> &str {
        match self {
            Self::Qwen(backend) => backend.plan_digest(),
            Self::Gemma(backend) => backend.plan_digest(),
        }
    }

    fn target(&self) -> &str {
        match self {
            Self::Qwen(backend) => backend.target(),
            Self::Gemma(backend) => backend.target(),
        }
    }

    fn context_length(&self) -> u32 {
        match self {
            Self::Qwen(backend) => backend.context_length(),
            Self::Gemma(backend) => backend.context_length(),
        }
    }

    fn recommended_context_tokens(&self) -> u32 {
        match self {
            Self::Qwen(backend) => backend.recommended_context_tokens(),
            Self::Gemma(backend) => backend.recommended_context_tokens(),
        }
    }

    fn resident_bytes(&self) -> u64 {
        self.as_trait()
            .observability_snapshot()
            .model_resident
            .current_bytes
    }

    fn adapter_catalog_identity(&self) -> Result<String, String> {
        match self {
            Self::Qwen(backend) => backend
                .adapter_catalog_identity()
                .map_err(|error| error.to_string()),
            Self::Gemma(_) => Ok("adapter:none-v1".to_owned()),
        }
    }

    fn shutdown(&self) -> Result<ProductionShutdownAuditV1, sllm_server::BackendErrorV1> {
        match self {
            Self::Qwen(backend) => backend.shutdown(),
            Self::Gemma(backend) => backend.shutdown(),
        }
    }
}

struct BackendShutdownGuard<'a> {
    backend: &'a ActiveBackend,
    active: bool,
}

impl<'a> BackendShutdownGuard<'a> {
    fn new(backend: &'a ActiveBackend) -> Self {
        Self {
            backend,
            active: true,
        }
    }

    fn shutdown(mut self) -> Result<ProductionShutdownAuditV1, sllm_server::BackendErrorV1> {
        self.active = false;
        self.backend.shutdown()
    }
}

impl Drop for BackendShutdownGuard<'_> {
    fn drop(&mut self) {
        if self.active {
            let _ = self.backend.shutdown();
        }
    }
}

fn take_required(values: &mut BTreeMap<String, String>, flag: &str) -> Result<String, String> {
    values
        .remove(flag)
        .ok_or_else(|| format!("required argument {flag} is absent\n{}", usage()))
}

fn parse_default<T: std::str::FromStr>(
    values: &mut BTreeMap<String, String>,
    flag: &str,
    default: T,
) -> Result<T, String> {
    match values.remove(flag) {
        Some(value) => parse_value(&value, flag),
        None => Ok(default),
    }
}

fn parse_value<T: std::str::FromStr>(value: &str, name: &str) -> Result<T, String> {
    value
        .parse()
        .map_err(|_| format!("invalid {name}: {value}"))
}

fn parse_kv_cache_encoding(value: &str) -> Result<KvCacheEncoding, String> {
    match value {
        "fp16" => Ok(KvCacheEncoding::Fp16),
        "fp8" => Ok(KvCacheEncoding::Fp8E4M3Fn),
        "fp8-static" => Ok(KvCacheEncoding::Fp8E4M3FnStatic),
        "nvfp4" => Ok(KvCacheEncoding::Nvfp4),
        "kv-mxfp8-e4" => Ok(KvCacheEncoding::Mxfp8E4),
        "kv-mxfp8-e5" => Ok(KvCacheEncoding::Mxfp8E5),
        _ => Err(format!(
            "KV cache encoding must be fp16, fp8, fp8-static, nvfp4, kv-mxfp8-e4, or kv-mxfp8-e5: {value}"
        )),
    }
}

fn resolve_server_kv_selection(
    requested: Option<KvCacheEncoding>,
    exact_target: &str,
    model_fingerprint: &str,
    dense_text: bool,
    explicit_source: KvCacheExplicitSourceV1,
) -> Result<(sllm_core::KvCacheSelection, KvCacheSelectionReportV1), String> {
    let selection = resolve_kv_cache_selection(KvCacheSelectionRequest::new(
        requested,
        exact_target,
        model_fingerprint,
        dense_text,
        dense_text,
        true,
        256,
    ))
    .map_err(|error| error.to_string())?;
    Ok((
        selection,
        KvCacheSelectionReportV1::from_core(selection, explicit_source),
    ))
}

fn effective_server_kv_request(
    model_entry: Option<KvCacheEncoding>,
    process: Option<KvCacheEncoding>,
) -> (Option<KvCacheEncoding>, KvCacheExplicitSourceV1) {
    match model_entry {
        Some(encoding) => (Some(encoding), KvCacheExplicitSourceV1::ModelEntry),
        None => (process, KvCacheExplicitSourceV1::Process),
    }
}

fn reject_disabled_options(
    values: &BTreeMap<String, String>,
    options: &[&str],
    mode_flag: &str,
) -> Result<(), String> {
    if let Some(option) = options.iter().find(|option| values.contains_key(**option)) {
        return Err(format!("{option} requires {mode_flag} to be enabled"));
    }
    Ok(())
}

fn usage() -> &'static str {
    "usage: sllm-server (--models PATH | --gguf PATH --derived-lock PATH --device-index N --target GFX) [--listen HOST:PORT] [--model ALIAS] [--api-key-env NAME | --api-key-file PATH] [--cors-origins ORIGIN,...] [--metrics true|false] [--resumable-sse true|false] [--replay-sessions N] [--replay-events N] [--tls-cert PATH --tls-key PATH] [--compatibility-profile strict|openwebui] [--context-length TOKENS] [--kv-cache-encoding fp16|fp8|fp8-static|nvfp4|kv-mxfp8-e4|kv-mxfp8-e5] (default for reviewed Qwen3.5-4B BF16 dense text: kv-mxfp8-e4; rollback: fp16) [--queue-capacity N] [--event-capacity N] [--request-timeout-seconds N] [--completion-timeout-seconds N] [--shutdown-timeout-seconds N] [--prefix-cache disabled|enabled --prefix-cache-max-entries N --prefix-cache-max-tokens N --prefix-cache-max-resident-bytes N] [--context-policy disabled|keep-prefix-recent-v1 --context-keep-prefix N --context-keep-recent N] [--checkpoint disabled|enabled --checkpoint-directory PATH --checkpoint-quota-bytes N [--checkpoint-load NAME] [--checkpoint-save NAME]] [--draft disabled|mtp-auto|ngram|external [--draft-ngram-order N --draft-width N] [--draft-model-identity ID --draft-tokenizer-identity ID --draft-vocabulary-size N --draft-width N]]"
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{
        context_length_warning, effective_server_kv_request, parse_args_from,
        read_private_key_file, read_tls_cert_file, resolve_server_kv_selection, usage,
    };
    use sllm_core::{KvCacheEncoding, QWEN35_4B_FINGERPRINT};
    use sllm_server::{
        CheckpointStartupConfigV1, ContextWindowStartupConfigV1, DraftStartupConfigV1,
        KvCacheExplicitSourceV1, PrefixCacheStartupConfigV1,
    };

    fn base_args(extra: &[&str]) -> Vec<String> {
        let mut args = vec![
            "--gguf",
            "model.gguf",
            "--derived-lock",
            "model.lock.json",
            "--device-index",
            "0",
            "--target",
            "gfx1201",
        ];
        args.extend_from_slice(extra);
        args.into_iter().map(str::to_owned).collect()
    }

    #[test]
    fn context_warning_is_advisory_only_above_the_model_recommendation() {
        assert!(context_length_warning(262_143, 262_144).is_none());
        assert!(context_length_warning(262_144, 262_144).is_none());
        let warning = context_length_warning(1_000_000, 262_144).unwrap();
        assert_eq!(warning["event"], "warning");
        assert_eq!(warning["kind"], "context_length_exceeds_recommended");
        assert_eq!(warning["context_length"], 1_000_000);
        assert_eq!(warning["official_recommended_context_tokens"], 262_144);
    }

    #[test]
    fn phase41_cli_defaults_every_opt_in_disabled() {
        let config = parse_args_from(base_args(&[])).expect("default CLI should parse");
        assert_eq!(config.kv_cache_encoding, None);
        assert!(matches!(
            config.phase41.prefix_cache,
            PrefixCacheStartupConfigV1::Disabled
        ));
        assert!(matches!(
            config.phase41.context_window,
            ContextWindowStartupConfigV1::Disabled
        ));
        assert!(matches!(
            config.phase41.checkpoint,
            CheckpointStartupConfigV1::Disabled
        ));
        assert!(matches!(
            config.phase41.draft,
            DraftStartupConfigV1::Disabled
        ));
    }

    #[test]
    fn kv_cli_preserves_auto_retires_block16_and_accepts_mxfp8_names() {
        let explicit_fp16 = parse_args_from(base_args(&["--kv-cache-encoding", "fp16"]))
            .expect("explicit fp16 should parse");
        assert_eq!(explicit_fp16.kv_cache_encoding, Some(KvCacheEncoding::Fp16));
        for (name, expected) in [
            ("kv-mxfp8-e4", KvCacheEncoding::Mxfp8E4),
            ("kv-mxfp8-e5", KvCacheEncoding::Mxfp8E5),
        ] {
            let config = parse_args_from(base_args(&["--kv-cache-encoding", name]))
                .expect("canonical MXFP8 spelling should parse");
            assert_eq!(config.kv_cache_encoding, Some(expected));
        }
        for alias in [
            "fp8-e4-block16",
            "kv-fp8-e4m3-block16",
            "kv-fp8-e5m2-block16",
            "KV-FP8-E4-BLOCK16",
            "mxfp8-e4",
            "kv-mxfp8-e4-block32",
            "KV-MXFP8-E4",
        ] {
            assert!(
                parse_args_from(base_args(&["--kv-cache-encoding", alias])).is_err(),
                "derived alias must be rejected: {alias}"
            );
        }
        assert!(usage().contains("kv-mxfp8-e4|kv-mxfp8-e5"));
        assert!(!usage().contains("kv-fp8-e4-block16|"));
        assert!(usage().contains("default for reviewed Qwen3.5-4B BF16 dense text: kv-mxfp8-e4"));
    }

    #[test]
    fn server_selection_reports_source_and_fails_closed_outside_exact_scope() {
        let (resolved, auto) = resolve_server_kv_selection(
            None,
            "gfx1201",
            QWEN35_4B_FINGERPRINT,
            true,
            KvCacheExplicitSourceV1::Process,
        )
        .unwrap();
        assert_eq!(resolved.resolved(), KvCacheEncoding::Mxfp8E4);
        assert_eq!(auto.requested, "auto");
        assert_eq!(auto.selection_source, "mxfp8-e4-default");

        assert!(
            resolve_server_kv_selection(
                Some(KvCacheEncoding::Fp8E4M3Block16),
                "gfx1201",
                QWEN35_4B_FINGERPRINT,
                true,
                KvCacheExplicitSourceV1::ModelEntry,
            )
            .is_err()
        );

        assert!(
            resolve_server_kv_selection(
                Some(KvCacheEncoding::Fp8E4M3Block16),
                "gfx942",
                QWEN35_4B_FINGERPRINT,
                true,
                KvCacheExplicitSourceV1::Process,
            )
            .is_err()
        );
        assert!(
            resolve_server_kv_selection(
                Some(KvCacheEncoding::Fp8E4M3Block16),
                "gfx1201",
                QWEN35_4B_FINGERPRINT,
                false,
                KvCacheExplicitSourceV1::Process,
            )
            .is_err()
        );

        for (encoding, target, physical_variant) in [
            (KvCacheEncoding::Mxfp8E4, "gfx1201", "E4M3-OCP"),
            (KvCacheEncoding::Mxfp8E5, "gfx1030", "E5M2-OCP"),
        ] {
            let (resolved, report) = resolve_server_kv_selection(
                Some(encoding),
                target,
                QWEN35_4B_FINGERPRINT,
                true,
                KvCacheExplicitSourceV1::Process,
            )
            .unwrap();
            assert_eq!(resolved.resolved(), encoding);
            assert_eq!(report.selection_source, "process-explicit");
            assert_eq!(report.physical_variant.as_deref(), Some(physical_variant));
            let descriptor_id = format!("{}-v1", encoding.canonical_name());
            assert_eq!(
                report.descriptor_id.as_deref(),
                Some(descriptor_id.as_str())
            );
        }
        for target in ["gfx1030", "gfx1201", "gfx942:sramecc+:xnack-"] {
            let (resolved, report) = resolve_server_kv_selection(
                Some(KvCacheEncoding::Mxfp8E4),
                target,
                QWEN35_4B_FINGERPRINT,
                true,
                KvCacheExplicitSourceV1::Process,
            )
            .unwrap();
            assert_eq!(resolved.resolved(), KvCacheEncoding::Mxfp8E4);
            assert_eq!(report.physical_variant.as_deref(), Some("E4M3-OCP"));
        }
        for (encoding, target, fingerprint, dense) in [
            (
                KvCacheEncoding::Mxfp8E5,
                "gfx1201",
                QWEN35_4B_FINGERPRINT,
                true,
            ),
            (KvCacheEncoding::Mxfp8E4, "gfx1201", "wrong-model", true),
            (
                KvCacheEncoding::Mxfp8E4,
                "gfx1201",
                QWEN35_4B_FINGERPRINT,
                false,
            ),
        ] {
            assert!(
                resolve_server_kv_selection(
                    Some(encoding),
                    target,
                    fingerprint,
                    dense,
                    KvCacheExplicitSourceV1::Process,
                )
                .is_err()
            );
        }
    }

    #[test]
    fn model_entry_kv_encoding_precedes_process_selection() {
        let (requested, source) = effective_server_kv_request(
            Some(KvCacheEncoding::Mxfp8E4),
            Some(KvCacheEncoding::Fp16),
        );
        assert_eq!(requested, Some(KvCacheEncoding::Mxfp8E4));
        assert_eq!(source, KvCacheExplicitSourceV1::ModelEntry);

        let (requested, source) = effective_server_kv_request(None, Some(KvCacheEncoding::Mxfp8E5));
        assert_eq!(requested, Some(KvCacheEncoding::Mxfp8E5));
        assert_eq!(source, KvCacheExplicitSourceV1::Process);
    }

    #[test]
    fn models_manifest_is_an_alternative_to_legacy_source_flags() {
        let config = parse_args_from(["--models", "/etc/sllm/models.json", "--target", "gfx1201"]);
        assert!(
            config.is_err(),
            "manifest and legacy target must be exclusive"
        );

        let config = parse_args_from(["--models", "/etc/sllm/models.json"])
            .expect("manifest-only startup flags should parse");
        assert_eq!(
            config.models.as_deref(),
            Some(std::path::Path::new("/etc/sllm/models.json"))
        );
        assert_eq!(config.model, "dynamic");
    }

    #[test]
    fn models_manifest_rejects_legacy_model_alias() {
        let error = parse_args_from(["--models", "/etc/sllm/models.json", "--model", "qwen"])
            .expect_err("legacy --model must not be accepted with --models");
        assert!(error.contains("mutually exclusive"));
    }

    #[test]
    fn phase41_cli_accepts_documented_boundaries() {
        let config = parse_args_from(base_args(&[
            "--prefix-cache",
            "enabled",
            "--prefix-cache-max-entries",
            "1",
            "--prefix-cache-max-tokens",
            "1",
            "--prefix-cache-max-resident-bytes",
            "1",
            "--context-policy",
            "keep-prefix-recent-v1",
            "--context-keep-prefix",
            "0",
            "--context-keep-recent",
            "1",
            "--checkpoint",
            "enabled",
            "--checkpoint-directory",
            "/var/lib/sllm/checkpoints",
            "--checkpoint-quota-bytes",
            "1",
            "--checkpoint-save",
            "startup",
            "--draft",
            "ngram",
            "--draft-ngram-order",
            "16",
            "--draft-width",
            "8",
        ]))
        .expect("valid Phase41 boundary configuration should parse");
        assert!(matches!(
            config.phase41.prefix_cache,
            PrefixCacheStartupConfigV1::Enabled {
                max_entries: 1,
                max_logical_tokens: 1,
                max_resident_bytes: 1,
            }
        ));
        assert!(matches!(
            config.phase41.context_window,
            ContextWindowStartupConfigV1::KeepPrefixRecentV1 {
                keep_prefix: 0,
                keep_recent: 1,
            }
        ));
        assert!(matches!(
            config.phase41.draft,
            DraftStartupConfigV1::Ngram {
                order: 16,
                width: 8,
            }
        ));

        let upper = parse_args_from(base_args(&[
            "--prefix-cache",
            "enabled",
            "--prefix-cache-max-entries",
            "256",
            "--prefix-cache-max-tokens",
            "1048576",
            "--prefix-cache-max-resident-bytes",
            "1",
            "--context-policy",
            "keep-prefix-recent-v1",
            "--context-keep-prefix",
            "0",
            "--context-keep-recent",
            "0",
        ]));
        assert!(upper.is_err(), "context keep 0/0 must be rejected");

        let upper = parse_args_from(base_args(&[
            "--prefix-cache",
            "enabled",
            "--prefix-cache-max-entries",
            "256",
            "--prefix-cache-max-tokens",
            "1048576",
            "--prefix-cache-max-resident-bytes",
            "1",
            "--context-policy",
            "keep-prefix-recent-v1",
            "--context-keep-prefix",
            "1",
            "--context-keep-recent",
            "0",
        ]))
        .expect("prefix cache upper bounds should parse");
        assert!(matches!(
            upper.phase41.prefix_cache,
            PrefixCacheStartupConfigV1::Enabled {
                max_entries: 256,
                max_logical_tokens: 1_048_576,
                max_resident_bytes: 1,
            }
        ));
    }

    #[test]
    fn phase41_cli_rejects_just_over_limits_and_context_overflow() {
        for args in [
            vec![
                "--prefix-cache",
                "enabled",
                "--prefix-cache-max-entries",
                "257",
                "--prefix-cache-max-tokens",
                "1",
                "--prefix-cache-max-resident-bytes",
                "1",
            ],
            vec![
                "--prefix-cache",
                "enabled",
                "--prefix-cache-max-entries",
                "1",
                "--prefix-cache-max-tokens",
                "1048577",
                "--prefix-cache-max-resident-bytes",
                "1",
            ],
            vec![
                "--context-policy",
                "keep-prefix-recent-v1",
                "--context-keep-prefix",
                "18446744073709551615",
                "--context-keep-recent",
                "1",
            ],
        ] {
            assert!(
                parse_args_from(base_args(&args)).is_err(),
                "expected just-over/overflow limits to be rejected: {args:?}"
            );
        }
    }

    #[test]
    fn phase41_cli_accepts_external_draft_and_checks_width_bounds() {
        let valid = parse_args_from(base_args(&[
            "--draft",
            "external",
            "--draft-model-identity",
            "target-lock",
            "--draft-tokenizer-identity",
            "tokenizer-lock",
            "--draft-vocabulary-size",
            "1",
            "--draft-width",
            "8",
        ]))
        .expect("valid external draft configuration should parse");
        assert!(matches!(
            valid.phase41.draft,
            DraftStartupConfigV1::External {
                vocabulary_size: 1,
                width: 8,
                ..
            }
        ));
        for width in ["0", "9"] {
            assert!(
                parse_args_from(base_args(&[
                    "--draft",
                    "external",
                    "--draft-model-identity",
                    "target-lock",
                    "--draft-tokenizer-identity",
                    "tokenizer-lock",
                    "--draft-vocabulary-size",
                    "1",
                    "--draft-width",
                    width,
                ]))
                .is_err()
            );
        }
    }

    #[test]
    fn phase41_cli_load_validation_is_explicit_and_redacted() {
        let directory = std::env::temp_dir().join(format!(
            "sllm-phase41-cli-missing-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&directory).unwrap();
        let directory_string = directory.to_string_lossy().into_owned();
        let config = parse_args_from(base_args(&[
            "--checkpoint",
            "enabled",
            "--checkpoint-directory",
            &directory_string,
            "--checkpoint-quota-bytes",
            "1",
            "--checkpoint-load",
            "missing",
        ]))
        .expect("checkpoint load option should parse");
        let error = config.phase41.validate_startup().unwrap_err().to_string();
        assert_eq!(error, "configured checkpoint load target is unavailable");
        assert!(!error.contains(&directory_string));
        assert!(!error.contains("missing"));
        fs::remove_dir(directory).unwrap();
    }

    #[test]
    fn phase41_cli_rejects_missing_or_incompatible_options() {
        for args in [
            vec!["--prefix-cache", "enabled"],
            vec![
                "--prefix-cache-max-entries",
                "1",
                "--prefix-cache-max-tokens",
                "1",
                "--prefix-cache-max-resident-bytes",
                "1",
            ],
            vec![
                "--context-policy",
                "keep-prefix-recent-v1",
                "--context-keep-prefix",
                "0",
            ],
            vec![
                "--checkpoint",
                "enabled",
                "--checkpoint-directory",
                "/tmp/checkpoints",
                "--checkpoint-quota-bytes",
                "1",
            ],
            vec![
                "--draft",
                "ngram",
                "--draft-ngram-order",
                "0",
                "--draft-width",
                "1",
            ],
            vec![
                "--draft",
                "external",
                "--draft-model-identity",
                "target",
                "--draft-tokenizer-identity",
                "tokenizer",
                "--draft-vocabulary-size",
                "1",
            ],
        ] {
            assert!(
                parse_args_from(base_args(&args)).is_err(),
                "expected invalid Phase41 options to be rejected: {args:?}"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn tls_files_require_regular_inputs_and_private_key_permissions() {
        use std::os::unix::fs::{PermissionsExt, symlink};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory =
            std::env::temp_dir().join(format!("sllm-phase39-tls-{}-{nonce}", std::process::id()));
        fs::create_dir(&directory).unwrap();
        let cert = directory.join("server.crt");
        let cert_link = directory.join("server-link.crt");
        let key = directory.join("server.key");
        fs::write(&cert, b"test certificate").unwrap();
        symlink(&cert, &cert_link).unwrap();
        fs::write(&key, b"test key").unwrap();
        fs::set_permissions(&key, fs::Permissions::from_mode(0o600)).unwrap();
        assert_eq!(read_tls_cert_file(&cert).unwrap(), b"test certificate");
        assert_eq!(read_private_key_file(&key).unwrap(), b"test key");
        assert!(read_tls_cert_file(&cert_link).is_err());

        fs::set_permissions(&key, fs::Permissions::from_mode(0o640)).unwrap();
        assert_eq!(
            read_private_key_file(&key).unwrap_err(),
            "TLS private key permissions are too broad"
        );
        assert!(read_tls_cert_file(&directory).is_err());

        fs::remove_file(cert).unwrap();
        fs::remove_file(cert_link).unwrap();
        fs::remove_file(key).unwrap();
        fs::remove_dir(directory).unwrap();
    }
}
