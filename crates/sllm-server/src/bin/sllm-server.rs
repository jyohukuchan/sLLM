use std::collections::BTreeMap;
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::Read;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[path = "sllm-server/webui.rs"]
mod sllm_server_webui;

use sllm_server_webui::{DEFAULT_WEBUI_PORT, WebUiProcess};

const MAX_TLS_PEM_BYTES: usize = 1024 * 1024;

use sllm_core::{
    GEMMA4_MOE_MODEL_FINGERPRINT, GEMMA4_RECOMMENDED_CONTEXT_TOKENS, KvCacheEncoding,
    KvCacheSelectionRequest, MINISTRAL3_GRAPH_ORIGINAL_CONTEXT, MINISTRAL3_MODEL_ALIAS,
    MINISTRAL3_MODEL_LOCK_FINGERPRINT, QWEN35_MOE_MODEL_FINGERPRINT,
    QWEN35_RECOMMENDED_CONTEXT_TOKENS, ReviewedModelLock, builtin_reviewed_model_lock,
    read_derived_gguf_lock, resolve_kv_cache_selection,
};
use sllm_hip::Context as HipContext;
use sllm_server::{
    ChatGenerationBackendV1, CheckpointStartupConfigV1, ContextWindowStartupConfigV1,
    CredentialStoreV1, DraftStartupConfigV1, Gemma4BackendConfigV1, Gemma4ChatBackendV1,
    Gemma4MoeBackendConfigV1, Gemma4MoeChatBackendV1, KvCacheExplicitSourceV1,
    KvCacheSelectionReportV1, Ministral3BackendConfigV1, Ministral3ChatBackendV1,
    ModelLibraryDeviceV1, ModelLibraryRegistrationV1, ModelLibraryV1, ModelLifecycleConfigV1,
    ModelLifecycleDescriptorV1, ModelLifecycleLoadedV1, ModelLifecycleRegistryV1,
    ModelRegistryEntryV1, ModelRegistryV1, Phase41ProductionConfigV1, PrefixCacheStartupConfigV1,
    ProductionShutdownAuditV1, QwenAdapterArtifactConfigV1, QwenAdapterCatalogConfigV1,
    QwenBackendConfigV1, QwenChatBackendV1, ResumableStoreV1, SchedulerConfigV1, SchedulerV1,
    ServerConfigV1, ServerLifecycleStateV1, ServerLifecycleV1, ServerMetricsV1,
    build_dynamic_router_v1, build_router_v1, dynamic_model_plan_digest_preflight,
    ministral3_model_plan_preflight_v1, qwen_adapter_catalog_identity_preflight,
    read_model_manifest_v1,
};

#[derive(Clone)]
struct DynamicModelEntryV1 {
    gguf: PathBuf,
    derived_lock: Option<PathBuf>,
    mtp_assistant_gguf_path: Option<PathBuf>,
    mtp_assistant_derived_lock_path: Option<PathBuf>,
    device_index: u32,
    target: String,
    kv_cache_encoding: Option<KvCacheEncoding>,
    adapter_catalog: Option<QwenAdapterCatalogConfigV1>,
}

impl DynamicModelEntryV1 {
    fn from_manifest(entry: &sllm_server::ModelManifestEntryV1) -> Result<Self, String> {
        Ok(Self {
            gguf: entry.gguf().to_owned(),
            derived_lock: Some(entry.derived_lock().to_owned()),
            // The manifest format predates model-library companion pairing.
            // Keep this path target-only; WebUI library registration is the
            // only route that can opt a Gemma target into MTP today.
            mtp_assistant_gguf_path: None,
            mtp_assistant_derived_lock_path: None,
            device_index: entry.device_index(),
            target: entry.target().to_owned(),
            kv_cache_encoding: entry.kv_cache_encoding(),
            adapter_catalog: dynamic_adapter_catalog(entry)?,
        })
    }

    fn from_library(entry: &ModelLibraryRegistrationV1) -> Self {
        Self {
            gguf: entry.gguf_path.clone(),
            derived_lock: entry.derived_lock_path.clone(),
            mtp_assistant_gguf_path: entry.mtp_assistant_gguf_path.clone(),
            mtp_assistant_derived_lock_path: entry.mtp_assistant_derived_lock_path.clone(),
            device_index: entry.device_index,
            target: entry.target.clone(),
            kv_cache_encoding: None,
            adapter_catalog: None,
        }
    }
}

/// Select the draft mode for a dynamically loaded Gemma target.
///
/// A model-library registration is the WebUI's opt-in point for a canonical
/// assistant pair.  Keep the pair invariant here so a partially resolved
/// registration cannot reach backend construction, and force target-only
/// loads to remain explicitly disabled even when a process-wide draft option
/// was supplied for another model type.
fn dynamic_gemma_phase41(
    base: &Phase41ProductionConfigV1,
    assistant_gguf: Option<&PathBuf>,
    assistant_derived_lock: Option<&PathBuf>,
) -> Result<Phase41ProductionConfigV1, String> {
    let paired = match (assistant_gguf, assistant_derived_lock) {
        (None, None) => false,
        (Some(_), Some(_)) => true,
        _ => {
            return Err(
                "Gemma MTP assistant GGUF and derived-lock paths must be configured together"
                    .to_owned(),
            );
        }
    };
    let mut phase41 = base.clone();
    phase41.draft = if paired {
        DraftStartupConfigV1::MtpAuto
    } else {
        DraftStartupConfigV1::Disabled
    };
    Ok(phase41)
}

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
    library_only: bool,
    gguf: PathBuf,
    derived_lock: Option<PathBuf>,
    mtp_assistant_gguf_path: Option<PathBuf>,
    mtp_assistant_derived_lock_path: Option<PathBuf>,
    device_index: u32,
    target: String,
    listen: SocketAddr,
    webui: bool,
    webui_port: u16,
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
    let mtp_assistant_gguf_path = values.remove("--mtp-assistant-gguf").map(PathBuf::from);
    let mtp_assistant_derived_lock_path = values
        .remove("--mtp-assistant-derived-lock")
        .map(PathBuf::from);
    if mtp_assistant_gguf_path.is_some() != mtp_assistant_derived_lock_path.is_some() {
        return Err(
            "--mtp-assistant-gguf and --mtp-assistant-derived-lock must be configured together"
                .to_owned(),
        );
    }
    let legacy_device_index = values.remove("--device-index");
    let legacy_target = values.remove("--target");
    let requested_model = values.remove("--model");
    let has_legacy_source = legacy_gguf.is_some()
        || legacy_derived_lock.is_some()
        || legacy_device_index.is_some()
        || legacy_target.is_some()
        || requested_model.is_some();
    let (library_only, gguf, derived_lock, device_index, target, model) = if models.is_some() {
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
            false,
            PathBuf::new(),
            None,
            0,
            String::new(),
            "dynamic".to_owned(),
        )
    } else if !has_legacy_source {
        (
            true,
            PathBuf::new(),
            None,
            0,
            String::new(),
            "dynamic".to_owned(),
        )
    } else {
        let gguf = PathBuf::from(
            legacy_gguf.ok_or_else(|| "missing required argument --gguf".to_owned())?,
        );
        let derived_lock = legacy_derived_lock.map(PathBuf::from);
        let device_index = parse_value(
            &legacy_device_index
                .ok_or_else(|| "missing required argument --device-index".to_owned())?,
            "device index",
        )?;
        let target =
            legacy_target.ok_or_else(|| "missing required argument --target".to_owned())?;
        let model = requested_model.unwrap_or_else(|| {
            if derived_lock.is_none() {
                MINISTRAL3_MODEL_ALIAS.to_owned()
            } else {
                "qwen3.5-4b".to_owned()
            }
        });
        (false, gguf, derived_lock, device_index, target, model)
    };
    let listen: SocketAddr = parse_value(
        &values
            .remove("--listen")
            .unwrap_or_else(|| "127.0.0.1:8080".to_owned()),
        "listen address",
    )?;
    let webui = parse_default(&mut values, "--webui", true)?;
    let webui_port_specified = values.contains_key("--webui-port");
    let webui_port = parse_default(&mut values, "--webui-port", DEFAULT_WEBUI_PORT)?;
    if !webui && webui_port_specified {
        return Err("--webui-port requires --webui true".to_owned());
    }
    if webui && webui_port == 0 {
        return Err("--webui-port must be nonzero".to_owned());
    }
    if webui && listen.port() == 0 {
        return Err("--listen port 0 cannot be used while WebUI is enabled".to_owned());
    }
    if webui && listen.port() == webui_port {
        return Err("API and WebUI ports must be different".to_owned());
    }
    let api_key_env = values.remove("--api-key-env");
    let api_key_file = values.remove("--api-key-file").map(PathBuf::from);
    if api_key_env.is_some() && api_key_file.is_some() {
        return Err("--api-key-env and --api-key-file are mutually exclusive".to_owned());
    }
    let mut cors_origins = values
        .remove("--cors-origins")
        .map(|value| value.split(',').map(str::to_owned).collect::<Vec<_>>())
        .unwrap_or_default();
    if cors_origins.iter().any(String::is_empty) {
        return Err("--cors-origins must be a comma-separated list of exact origins".to_owned());
    }
    if webui {
        for origin in [
            format!("http://localhost:{webui_port}"),
            format!("http://127.0.0.1:{webui_port}"),
        ] {
            if !cors_origins.contains(&origin) {
                cors_origins.push(origin);
            }
        }
    }
    if cors_origins.len() > 32 {
        return Err("CORS origin count exceeds the bounded limit".to_owned());
    }
    let metrics = values
        .remove("--metrics")
        .map(|value| parse_value(&value, "--metrics"))
        .transpose()?
        .unwrap_or(webui);
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
    if mtp_assistant_gguf_path.is_some() {
        if models.is_some() || library_only {
            return Err(
                "Gemma MTP assistant options require the static --gguf/--derived-lock source"
                    .to_owned(),
            );
        }
        if !matches!(phase41.draft, DraftStartupConfigV1::MtpAuto) {
            return Err("Gemma MTP assistant options require --draft mtp-auto".to_owned());
        }
    }
    phase41
        .validate()
        .map_err(|error| format!("Phase41 configuration invalid: {error}"))?;
    if let Some(flag) = values.keys().next() {
        return Err(format!("unknown argument {flag}\n{}", usage()));
    }
    Ok(Config {
        models,
        library_only,
        gguf,
        derived_lock,
        mtp_assistant_gguf_path,
        mtp_assistant_derived_lock_path,
        device_index,
        target,
        listen,
        webui,
        webui_port,
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
        if config.library_only {
            return run_dynamic_manifest(config, None, credentials, tls).await;
        }
        if let Some(manifest_path) = config.models.clone() {
            return run_dynamic_manifest(config, Some(manifest_path), credentials, tls).await;
        }
        let lifecycle = ServerLifecycleV1::new(ServerLifecycleStateV1::Loading);
        let (backend, startup_kv_selection) = if let Some(derived_lock_path) = config.derived_lock.clone() {
        let derived = read_derived_gguf_lock(&derived_lock_path)
            .map_err(|error| format!("derived GGUF lock validation failed: {error}"))?;
        let gguf_moe = derived.semantic_model_id.starts_with("qwen35moe:");
        let gemma_moe = derived.semantic_model_id.starts_with("gemma4moe:");
        let reviewed = if gguf_moe || gemma_moe {
            None
        } else {
            Some(
                builtin_reviewed_model_lock(&derived.source_lock_fingerprints)
                    .map_err(|error| format!("built-in model lock resolution failed: {error}"))?,
            )
        };
        if config.mtp_assistant_gguf_path.is_some()
            && !matches!(reviewed.as_ref(), Some(ReviewedModelLock::Gemma4(_)))
        {
            return Err(
                "Gemma MTP assistant options require a reviewed dense Gemma 4 target"
                    .to_owned(),
            );
        }
        if gguf_moe {
            let (kv_cache_resolved_selection, kv_cache_selection) = resolve_server_kv_selection(
                config.kv_cache_encoding,
                &config.target,
                QWEN35_MOE_MODEL_FINGERPRINT,
                false,
                KvCacheExplicitSourceV1::Process,
            )?;
            let kv_cache_encoding = kv_cache_resolved_selection.resolved();
            let backend_config = QwenBackendConfigV1 {
                gguf_path: config.gguf.clone(),
                derived_lock_path: derived_lock_path.clone(),
                device_index: config.device_index,
                target: config.target.clone(),
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
        } else if gemma_moe {
            validate_gemma4_moe_kv_request(config.kv_cache_encoding)?;
            validate_gemma4_moe_context_override(config.context_length)?;
            let kv_cache_selection = KvCacheSelectionReportV1 {
                requested: config
                    .kv_cache_encoding
                    .map(KvCacheEncoding::canonical_name)
                    .unwrap_or("auto")
                    .to_owned(),
                resolved: "fp8-static".to_owned(),
                selection_source: "model-recipe-explicit".to_owned(),
                reason: "Gemma 4 MoE uses its fixed reviewed static FP8 KV recipe".to_owned(),
                physical_variant: Some("E4M3-OCP".to_owned()),
                descriptor_id: None,
                policy_version: sllm_core::KV_CACHE_SELECTION_POLICY_VERSION_V1,
            };
            let backend_config = Gemma4MoeBackendConfigV1 {
                gguf_path: config.gguf.clone(),
                derived_lock_path: derived_lock_path.clone(),
                device_index: config.device_index,
                target: config.target.clone(),
                completion_timeout: config.completion_timeout,
                shutdown_timeout: config.shutdown_timeout,
                context_length: config
                    .context_length
                    .unwrap_or(GEMMA4_RECOMMENDED_CONTEXT_TOKENS as u32),
                phase41: config.phase41.clone(),
            };
            (
                ActiveBackend::GemmaMoe(Arc::new(
                    Gemma4MoeChatBackendV1::open(backend_config)
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
                        gguf_path: config.gguf.clone(),
                        derived_lock_path: derived_lock_path.clone(),
                        device_index: config.device_index,
                        target: config.target.clone(),
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
                        gguf_path: config.gguf.clone(),
                        derived_lock_path: derived_lock_path.clone(),
                        mtp_assistant_gguf_path: config.mtp_assistant_gguf_path,
                        mtp_assistant_derived_lock_path: config.mtp_assistant_derived_lock_path,
                        device_index: config.device_index,
                        target: config.target.clone(),
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
                ReviewedModelLock::Ministral3(_) => {
                    return Err(
                        "the official Ministral 3 model must be loaded directly without a derived lock"
                            .to_owned(),
                    );
                }
            }
        }
        } else {
            validate_ministral3_static_config(&config)?;
            let kv_cache_selection = ministral3_kv_selection(config.kv_cache_encoding);
            let backend = Ministral3ChatBackendV1::open(Ministral3BackendConfigV1 {
                gguf_path: config.gguf.clone(),
                device_index: config.device_index,
                target: config.target.clone(),
                completion_timeout: config.completion_timeout,
                shutdown_timeout: config.shutdown_timeout,
                context_length: config
                    .context_length
                    .unwrap_or(MINISTRAL3_GRAPH_ORIGINAL_CONTEXT as u32),
            })
            .map_err(|error| error.to_string())?;
            (
                ActiveBackend::Ministral3(Arc::new(backend)),
                kv_cache_selection,
            )
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
        if let Some(hardware) = detect_model_library_device(config.device_index) {
            server_config = server_config.with_hardware(hardware);
        }
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
        let webui_process = config
            .webui
            .then(|| WebUiProcess::start(config.webui_port, address, tls.is_some()))
            .transpose()?;
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
                "webui": config.webui,
                "webui_url": webui_process.as_ref().map(WebUiProcess::url),
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

fn ministral3_kv_selection(requested: Option<KvCacheEncoding>) -> KvCacheSelectionReportV1 {
    KvCacheSelectionReportV1 {
        requested: requested
            .map(KvCacheEncoding::canonical_name)
            .unwrap_or("auto")
            .to_owned(),
        resolved: KvCacheEncoding::Fp16.canonical_name().to_owned(),
        selection_source: "model-fixed-fp16".to_owned(),
        reason: "Ministral 3 uses its fixed reviewed FP16 KV cache recipe".to_owned(),
        physical_variant: None,
        descriptor_id: None,
        policy_version: sllm_core::KV_CACHE_SELECTION_POLICY_VERSION_V1,
    }
}

fn validate_ministral3_static_config(config: &Config) -> Result<(), String> {
    validate_ministral3_dynamic_config(config.kv_cache_encoding, &config.phase41, None)?;
    if config.mtp_assistant_gguf_path.is_some() || config.mtp_assistant_derived_lock_path.is_some()
    {
        return Err(
            "Ministral 3 is text-only and does not support MTP assistant artifacts".to_owned(),
        );
    }
    Ok(())
}

fn validate_ministral3_dynamic_config(
    requested_kv: Option<KvCacheEncoding>,
    phase41: &Phase41ProductionConfigV1,
    adapter_catalog: Option<&QwenAdapterCatalogConfigV1>,
) -> Result<(), String> {
    if requested_kv.is_some_and(|encoding| encoding != KvCacheEncoding::Fp16) {
        return Err("Ministral 3 supports only the reviewed FP16 KV cache recipe".to_owned());
    }
    if adapter_catalog.is_some() {
        return Err("Ministral 3 does not support adapter catalogs".to_owned());
    }
    if !matches!(phase41.prefix_cache, PrefixCacheStartupConfigV1::Disabled)
        || !matches!(
            phase41.context_window,
            ContextWindowStartupConfigV1::Disabled
        )
        || !matches!(phase41.checkpoint, CheckpointStartupConfigV1::Disabled)
        || !matches!(phase41.draft, DraftStartupConfigV1::Disabled)
    {
        return Err(
            "Ministral 3 currently supports only the baseline Phase41 lifecycle configuration"
                .to_owned(),
        );
    }
    Ok(())
}

fn detect_model_library_device(device_index: u32) -> Option<ModelLibraryDeviceV1> {
    let info = HipContext::query_device(device_index).ok()?;
    let logical_target = info.gcn_arch_name.split(':').next().unwrap_or_default();
    if !matches!(logical_target, "gfx1030" | "gfx1201" | "gfx942") {
        return None;
    }
    Some(ModelLibraryDeviceV1 {
        device_index: info.device_index,
        target: info.gcn_arch_name,
        name: info.name,
        total_memory_bytes: info.total_memory_bytes,
    })
}

fn default_model_library_state_path() -> Result<PathBuf, String> {
    if let Some(path) = env::var_os("XDG_CONFIG_HOME").map(PathBuf::from) {
        if path.is_absolute() {
            return Ok(path.join("sllm/model-library.json"));
        }
    }
    if let Some(path) = env::var_os("HOME").map(PathBuf::from) {
        if path.is_absolute() {
            return Ok(path.join(".config/sllm/model-library.json"));
        }
    }
    env::current_dir()
        .map(|path| path.join(".sllm/model-library.json"))
        .map_err(|_| "model-library state directory could not be resolved".to_owned())
}

fn default_model_library_initial_path() -> Result<PathBuf, String> {
    if let Some(path) = env::var_os("HOME").map(PathBuf::from) {
        if path.is_absolute() && path.is_dir() {
            return fs::canonicalize(path)
                .map_err(|_| "initial model folder could not be resolved".to_owned());
        }
    }
    env::current_dir()
        .and_then(fs::canonicalize)
        .map_err(|_| "initial model folder could not be resolved".to_owned())
}

/// Start the manifest-backed dynamic model registry.  All locks and identities
/// are admitted before the first backend is opened; the loader then repeats the
/// lock admission immediately before each GPU load.
async fn run_dynamic_manifest(
    config: Config,
    manifest_path: Option<PathBuf>,
    credentials: CredentialStoreV1,
    tls: Option<axum_server::tls_rustls::RustlsConfig>,
) -> Result<(), String> {
    let manifest = manifest_path
        .as_ref()
        .map(read_model_manifest_v1)
        .transpose()
        .map_err(|error| format!("model manifest validation failed: {error}"))?;
    let manifest_models = manifest.as_ref().map_or(&[][..], |value| value.models());
    let mut descriptors = Vec::with_capacity(manifest_models.len());
    let mut entries = BTreeMap::new();
    let mut resident_quota = 0_u64;
    for entry in manifest_models {
        let derived = read_derived_gguf_lock(entry.derived_lock()).map_err(|error| {
            format!(
                "model {} derived lock validation failed: {error}",
                entry.alias()
            )
        })?;
        if derived.semantic_model_id.starts_with("gemma4moe:") {
            validate_gemma4_moe_context_override(config.context_length)
                .map_err(|error| format!("model {}: {error}", entry.alias()))?;
        }
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
        entries.insert(
            entry.alias().to_owned(),
            DynamicModelEntryV1::from_manifest(entry)?,
        );
    }
    let library_device = detect_model_library_device(0);
    if let Some(device) = library_device.as_ref() {
        resident_quota = resident_quota.max(device.total_memory_bytes);
    }
    resident_quota = resident_quota.max(1);
    let lifecycle_config = ModelLifecycleConfigV1::new(resident_quota)
        .and_then(|value| value.with_timeouts(config.completion_timeout, config.shutdown_timeout))
        .map_err(|error| format!("dynamic lifecycle configuration invalid: {error:?}"))?;
    let entries = Arc::new(RwLock::new(entries));
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
            .read()
            .map_err(|_| "dynamic model catalog is poisoned".to_owned())?
            .get(descriptor.alias())
            .cloned()
            .ok_or_else(|| "dynamic model alias disappeared".to_owned())?;
        let (derived, model_identity, plan_identity_before, resident_bytes_before) =
            match entry.derived_lock.as_ref() {
                Some(derived_lock) => {
                    let derived = read_derived_gguf_lock(derived_lock)
                        .map_err(|error| format!("derived lock validation failed: {error}"))?;
                    let model_identity = dynamic_model_identity(&derived)?;
                    let plan_identity_before =
                        dynamic_model_plan_digest_preflight(&entry.gguf, &derived).map_err(
                            |error| format!("verified weight-plan preflight failed: {error}"),
                        )?;
                    (Some(derived), model_identity, plan_identity_before, None)
                }
                None => {
                    let (plan_identity_before, resident_bytes_before) =
                        ministral3_model_plan_preflight_v1(&entry.gguf).map_err(|error| {
                            format!("official Ministral 3 weight-plan preflight failed: {error}")
                        })?;
                    (
                        None,
                        MINISTRAL3_MODEL_LOCK_FINGERPRINT.to_owned(),
                        plan_identity_before,
                        Some(resident_bytes_before),
                    )
                }
            };
        if model_identity != descriptor.identity().model_identity()
            || plan_identity_before != descriptor.identity().plan_identity()
            || resident_bytes_before
                .is_some_and(|bytes| bytes != descriptor.declared_resident_bytes())
        {
            return Err(
                "model, weight-plan, or resident-byte identity changed since preflight".to_owned(),
            );
        }
        let is_moe = derived
            .as_ref()
            .is_some_and(|value| value.semantic_model_id.starts_with("qwen35moe:"));
        let is_gemma_moe = derived
            .as_ref()
            .is_some_and(|value| value.semantic_model_id.starts_with("gemma4moe:"));
        let empty_adapter_catalog = QwenAdapterCatalogConfigV1::default();
        let adapter_identity_before = qwen_adapter_catalog_identity_preflight(
            entry
                .adapter_catalog
                .as_ref()
                .unwrap_or(&empty_adapter_catalog),
        )
        .map_err(|error| error.to_string())?;
        if adapter_identity_before != descriptor.identity().adapter_identity() {
            return Err("adapter catalog identity changed since manifest preflight".to_owned());
        }
        let adapter_catalog = entry.adapter_catalog.clone();
        let backend = match derived.as_ref() {
            None => {
                validate_ministral3_dynamic_config(
                    load_kv,
                    &load_phase41,
                    adapter_catalog.as_ref(),
                )?;
                ActiveBackend::Ministral3(Arc::new(
                    Ministral3ChatBackendV1::open(Ministral3BackendConfigV1 {
                        gguf_path: entry.gguf.clone(),
                        device_index: entry.device_index,
                        target: entry.target.clone(),
                        completion_timeout: load_completion_timeout,
                        shutdown_timeout: load_shutdown_timeout,
                        context_length: load_context_length
                            .unwrap_or(MINISTRAL3_GRAPH_ORIGINAL_CONTEXT as u32),
                    })
                    .map_err(|error| error.to_string())?,
                ))
            }
            Some(_) if is_moe => {
                let derived_lock = entry
                    .derived_lock
                    .as_ref()
                    .ok_or_else(|| "derived lock disappeared for MoE dynamic entry".to_owned())?;
                if adapter_catalog.is_some() {
                    return Err("MoE models do not support adapter catalogs".to_owned());
                }
                let (requested_kv, explicit_source) =
                    effective_server_kv_request(entry.kv_cache_encoding, load_kv);
                let (kv_cache_resolved_selection, kv_cache_selection) =
                    resolve_server_kv_selection(
                        requested_kv,
                        &entry.target,
                        QWEN35_MOE_MODEL_FINGERPRINT,
                        false,
                        explicit_source,
                    )?;
                let kv_cache_encoding = kv_cache_resolved_selection.resolved();
                ActiveBackend::Qwen(Arc::new(
                    QwenChatBackendV1::open(QwenBackendConfigV1 {
                        gguf_path: entry.gguf.clone(),
                        derived_lock_path: derived_lock.clone(),
                        device_index: entry.device_index,
                        target: entry.target.clone(),
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
            }
            Some(_) if is_gemma_moe => {
                let derived_lock = entry.derived_lock.as_ref().ok_or_else(|| {
                    "derived lock disappeared for Gemma MoE dynamic entry".to_owned()
                })?;
                if adapter_catalog.is_some() {
                    return Err("Gemma 4 MoE models do not support adapter catalogs".to_owned());
                }
                let (requested_kv, _) =
                    effective_server_kv_request(entry.kv_cache_encoding, load_kv);
                validate_gemma4_moe_kv_request(requested_kv)?;
                ActiveBackend::GemmaMoe(Arc::new(
                    Gemma4MoeChatBackendV1::open(Gemma4MoeBackendConfigV1 {
                        gguf_path: entry.gguf.clone(),
                        derived_lock_path: derived_lock.clone(),
                        device_index: entry.device_index,
                        target: entry.target.clone(),
                        completion_timeout: load_completion_timeout,
                        shutdown_timeout: load_shutdown_timeout,
                        context_length: load_context_length
                            .unwrap_or(GEMMA4_RECOMMENDED_CONTEXT_TOKENS as u32),
                        phase41: load_phase41.clone(),
                    })
                    .map_err(|error| error.to_string())?,
                ))
            }
            Some(derived) => {
                let derived_lock = entry.derived_lock.as_ref().ok_or_else(|| {
                    "derived lock disappeared for reviewed dynamic entry".to_owned()
                })?;
                match builtin_reviewed_model_lock(&derived.source_lock_fingerprints)
                    .map_err(|error| format!("reviewed model lock resolution failed: {error}"))?
                {
                    ReviewedModelLock::Qwen35(lock) => {
                        let (requested_kv, explicit_source) =
                            effective_server_kv_request(entry.kv_cache_encoding, load_kv);
                        let (kv_cache_resolved_selection, kv_cache_selection) =
                            resolve_server_kv_selection(
                                requested_kv,
                                &entry.target,
                                lock.fingerprint(),
                                true,
                                explicit_source,
                            )?;
                        let kv_cache_encoding = kv_cache_resolved_selection.resolved();
                        ActiveBackend::Qwen(Arc::new(
                            QwenChatBackendV1::open(QwenBackendConfigV1 {
                                gguf_path: entry.gguf.clone(),
                                derived_lock_path: derived_lock.clone(),
                                device_index: entry.device_index,
                                target: entry.target.clone(),
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
                            effective_server_kv_request(entry.kv_cache_encoding, load_kv);
                        let (kv_cache_resolved_selection, _) = resolve_server_kv_selection(
                            requested_kv,
                            &entry.target,
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
                        let phase41 = dynamic_gemma_phase41(
                            &load_phase41,
                            entry.mtp_assistant_gguf_path.as_ref(),
                            entry.mtp_assistant_derived_lock_path.as_ref(),
                        )?;
                        ActiveBackend::Gemma(Arc::new(
                            Gemma4ChatBackendV1::open(Gemma4BackendConfigV1 {
                                gguf_path: entry.gguf.clone(),
                                derived_lock_path: derived_lock.clone(),
                                mtp_assistant_gguf_path: entry.mtp_assistant_gguf_path.clone(),
                                mtp_assistant_derived_lock_path: entry
                                    .mtp_assistant_derived_lock_path
                                    .clone(),
                                device_index: entry.device_index,
                                target: entry.target.clone(),
                                completion_timeout: load_completion_timeout,
                                shutdown_timeout: load_shutdown_timeout,
                                context_length: load_context_length
                                    .unwrap_or(GEMMA4_RECOMMENDED_CONTEXT_TOKENS as u32),
                                phase41,
                            })
                            .map_err(|error| error.to_string())?,
                        ))
                    }
                    ReviewedModelLock::Ministral3(_) => {
                        return Err(
                        "the official Ministral 3 model must be loaded directly without a derived lock"
                            .to_owned(),
                    );
                    }
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
        let (model_identity_after, plan_identity_after, resident_bytes_after) = match entry
            .derived_lock
            .as_ref()
        {
            Some(derived_lock) => {
                let derived_after = read_derived_gguf_lock(derived_lock)
                    .map_err(|error| fail_after_backend_open(&backend, error.to_string()))?;
                let model_identity_after = dynamic_model_identity(&derived_after)
                    .map_err(|error| fail_after_backend_open(&backend, error))?;
                let plan_identity_after =
                    dynamic_model_plan_digest_preflight(&entry.gguf, &derived_after)
                        .map_err(|error| fail_after_backend_open(&backend, error.to_string()))?;
                (model_identity_after, plan_identity_after, None)
            }
            None => {
                let (plan_identity_after, resident_bytes_after) =
                    ministral3_model_plan_preflight_v1(&entry.gguf)
                        .map_err(|error| fail_after_backend_open(&backend, error.to_string()))?;
                (
                    MINISTRAL3_MODEL_LOCK_FINGERPRINT.to_owned(),
                    plan_identity_after,
                    Some(resident_bytes_after),
                )
            }
        };
        let empty_adapter_catalog = QwenAdapterCatalogConfigV1::default();
        let adapter_identity_after = qwen_adapter_catalog_identity_preflight(
            entry
                .adapter_catalog
                .as_ref()
                .unwrap_or(&empty_adapter_catalog),
        )
        .map_err(|error| fail_after_backend_open(&backend, error.to_string()))?;
        if model_identity_after != descriptor.identity().model_identity()
            || plan_identity_after != descriptor.identity().plan_identity()
            || resident_bytes_after
                .is_some_and(|bytes| bytes != descriptor.declared_resident_bytes())
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
    for entry in manifest_models.iter().filter(|entry| entry.preload()) {
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

    if config.library_only && !config.listen.ip().is_loopback() {
        return Err("WebUI model-library startup requires a loopback --listen address".to_owned());
    }
    let dynamic_metrics = if config.metrics {
        Some(
            ServerMetricsV1::new_dynamic(lifecycle.configured_aliases())
                .map_err(|error| error.to_string())?,
        )
    } else {
        None
    };
    let model_library = if config.listen.ip().is_loopback() {
        let register_lifecycle = lifecycle.clone();
        let register_entries = Arc::clone(&entries);
        let register_metrics = dynamic_metrics.clone();
        let register = move |registration: ModelLibraryRegistrationV1| {
            if let Some(metrics) = register_metrics.as_ref() {
                metrics
                    .register_model_alias(&registration.alias)
                    .map_err(|error| format!("metrics registration failed: {error}"))?;
            }
            let descriptor = ModelLifecycleDescriptorV1::new(
                &registration.alias,
                &registration.model_identity,
                &registration.plan_identity,
                "adapter:none-v1",
                registration.resident_bytes,
            )
            .map_err(|error| format!("lifecycle descriptor is invalid: {error:?}"))?;
            let runtime_entry = DynamicModelEntryV1::from_library(&registration);
            {
                let mut catalog = register_entries
                    .write()
                    .map_err(|_| "dynamic model catalog is poisoned".to_owned())?;
                if catalog.contains_key(&registration.alias) {
                    return Err("model alias already exists".to_owned());
                }
                catalog.insert(registration.alias.clone(), runtime_entry);
            }
            if let Err(error) = register_lifecycle.register(descriptor) {
                if let Ok(mut catalog) = register_entries.write() {
                    catalog.remove(&registration.alias);
                }
                return Err(format!("lifecycle registration failed: {error:?}"));
            }
            Ok(())
        };
        let unregister_lifecycle = lifecycle.clone();
        let unregister_entries = Arc::clone(&entries);
        let unregister = move |alias: &str| {
            unregister_lifecycle
                .unregister(alias)
                .map_err(|error| format!("lifecycle removal failed: {error:?}"))?;
            unregister_entries
                .write()
                .map_err(|_| "dynamic model catalog is poisoned".to_owned())?
                .remove(alias);
            Ok(())
        };
        let state_path = default_model_library_state_path()?;
        let initial_path = default_model_library_initial_path()?;
        Some(ModelLibraryV1::open(
            state_path,
            initial_path,
            library_device.clone(),
            register,
            unregister,
        )?)
    } else {
        None
    };
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
    if let Some(hardware) = library_device.clone() {
        server_config = server_config.with_hardware(hardware);
    }
    if config.listen.ip().is_loopback() {
        server_config = server_config
            .with_loopback_admin(config.listen)
            .map_err(|error| error.to_string())?;
    }
    if let Some(model_library) = model_library {
        server_config = server_config.with_model_library(model_library);
    }
    if let Some(metrics) = dynamic_metrics {
        server_config = server_config.with_metrics(metrics);
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
    let webui_process = config
        .webui
        .then(|| WebUiProcess::start(config.webui_port, address, tls.is_some()))
        .transpose()?;
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
            "webui": config.webui,
            "webui_url": webui_process.as_ref().map(WebUiProcess::url),
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
            "count": shutdown_aliases.len(),
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
    } else if derived.semantic_model_id.starts_with("gemma4moe:") {
        if derived.semantic_model_id != format!("gemma4moe:{GEMMA4_MOE_MODEL_FINGERPRINT}")
            || derived.source_lock_fingerprints.as_slice() != [GEMMA4_MOE_MODEL_FINGERPRINT]
        {
            return Err("Gemma 4 MoE derived GGUF source identity differs".to_owned());
        }
        Ok(GEMMA4_MOE_MODEL_FINGERPRINT.to_owned())
    } else {
        match builtin_reviewed_model_lock(&derived.source_lock_fingerprints)
            .map_err(|error| format!("built-in model lock resolution failed: {error}"))?
        {
            ReviewedModelLock::Qwen35(lock) => Ok(lock.fingerprint().to_owned()),
            ReviewedModelLock::Gemma4(lock) => Ok(lock.fingerprint().to_owned()),
            ReviewedModelLock::Ministral3(_) => Err(
                "the official Ministral 3 model lock cannot be used as a derived GGUF source"
                    .to_owned(),
            ),
        }
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
    GemmaMoe(Arc<Gemma4MoeChatBackendV1>),
    Ministral3(Arc<Ministral3ChatBackendV1>),
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
            Self::GemmaMoe(backend) => backend.clone(),
            Self::Ministral3(backend) => backend.clone(),
        }
    }

    fn model_fingerprint(&self) -> &str {
        match self {
            Self::Qwen(backend) => backend.model_fingerprint(),
            Self::Gemma(backend) => backend.model_fingerprint(),
            Self::GemmaMoe(backend) => backend.model_fingerprint(),
            Self::Ministral3(backend) => backend.model_fingerprint(),
        }
    }

    fn plan_digest(&self) -> &str {
        match self {
            Self::Qwen(backend) => backend.plan_digest(),
            Self::Gemma(backend) => backend.plan_digest(),
            Self::GemmaMoe(backend) => backend.plan_digest(),
            Self::Ministral3(backend) => backend.plan_digest(),
        }
    }

    fn target(&self) -> &str {
        match self {
            Self::Qwen(backend) => backend.target(),
            Self::Gemma(backend) => backend.target(),
            Self::GemmaMoe(backend) => backend.target(),
            Self::Ministral3(backend) => backend.target(),
        }
    }

    fn context_length(&self) -> u32 {
        match self {
            Self::Qwen(backend) => backend.context_length(),
            Self::Gemma(backend) => backend.context_length(),
            Self::GemmaMoe(backend) => backend.context_length(),
            Self::Ministral3(backend) => backend.context_length(),
        }
    }

    fn recommended_context_tokens(&self) -> u32 {
        match self {
            Self::Qwen(backend) => backend.recommended_context_tokens(),
            Self::Gemma(backend) => backend.recommended_context_tokens(),
            Self::GemmaMoe(backend) => backend.recommended_context_tokens(),
            Self::Ministral3(backend) => backend.recommended_context_tokens(),
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
            Self::Gemma(_) | Self::GemmaMoe(_) | Self::Ministral3(_) => {
                Ok("adapter:none-v1".to_owned())
            }
        }
    }

    fn shutdown(&self) -> Result<ProductionShutdownAuditV1, sllm_server::BackendErrorV1> {
        match self {
            Self::Qwen(backend) => backend.shutdown(),
            Self::Gemma(backend) => backend.shutdown(),
            Self::GemmaMoe(backend) => backend.shutdown(),
            Self::Ministral3(backend) => backend.shutdown(),
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

fn validate_gemma4_moe_kv_request(requested: Option<KvCacheEncoding>) -> Result<(), String> {
    if requested.is_some_and(|encoding| encoding != KvCacheEncoding::Fp8E4M3FnStatic) {
        return Err(
            "Gemma 4 MoE uses its fixed static FP8 E4M3 KV recipe; omit --kv-cache-encoding (auto) or use --kv-cache-encoding fp8-static"
                .to_owned(),
        );
    }
    Ok(())
}

fn validate_gemma4_moe_context_override(context: Option<u32>) -> Result<(), String> {
    if context.is_some_and(|value| value > GEMMA4_RECOMMENDED_CONTEXT_TOKENS as u32) {
        return Err(format!(
            "Gemma 4 MoE does not support a context length above {} tokens",
            GEMMA4_RECOMMENDED_CONTEXT_TOKENS
        ));
    }
    Ok(())
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
    "usage: sllm-server [--models PATH | --gguf PATH [--derived-lock PATH] --device-index N --target GFX [--mtp-assistant-gguf PATH --mtp-assistant-derived-lock PATH --draft mtp-auto]] [--listen HOST:PORT] [--webui true|false] [--webui-port PORT] [--model ALIAS] [--api-key-env NAME | --api-key-file PATH] [--cors-origins ORIGIN,...] [--metrics true|false] [--resumable-sse true|false] [--replay-sessions N] [--replay-events N] [--tls-cert PATH --tls-key PATH] [--compatibility-profile strict|openwebui] [--context-length TOKENS] [--kv-cache-encoding fp16|fp8|fp8-static|nvfp4|kv-mxfp8-e4|kv-mxfp8-e5] (Qwen default: kv-mxfp8-e4; Gemma 4 MoE: auto or fp8-static only; direct official Ministral 3: FP16 only; FP16 rollback applies to Qwen) [--queue-capacity N] [--event-capacity N] [--request-timeout-seconds N] [--completion-timeout-seconds N] [--shutdown-timeout-seconds N] [--prefix-cache disabled|enabled --prefix-cache-max-entries N --prefix-cache-max-tokens N --prefix-cache-max-resident-bytes N] [--context-policy disabled|keep-prefix-recent-v1 --context-keep-prefix N --context-keep-recent N] [--checkpoint disabled|enabled --checkpoint-directory PATH --checkpoint-quota-bytes N [--checkpoint-load NAME] [--checkpoint-save NAME]] [--draft disabled|mtp-auto|ngram|external [--draft-ngram-order N --draft-width N] [--draft-model-identity ID --draft-tokenizer-identity ID --draft-vocabulary-size N --draft-width N]]"
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{
        DEFAULT_WEBUI_PORT, DynamicModelEntryV1, context_length_warning, dynamic_gemma_phase41,
        dynamic_model_identity, effective_server_kv_request, parse_args_from,
        read_private_key_file, read_tls_cert_file, resolve_server_kv_selection, usage,
        validate_gemma4_moe_context_override, validate_gemma4_moe_kv_request,
    };
    use sllm_core::{
        DerivedGgufConverter, DerivedGgufLock, DerivedGgufOutput, GEMMA4_MOE_MODEL_FINGERPRINT,
        GEMMA4_RECOMMENDED_CONTEXT_TOKENS, KvCacheEncoding, MINISTRAL3_MODEL_ALIAS,
        QWEN35_4B_FINGERPRINT,
    };
    use sllm_server::{
        CheckpointStartupConfigV1, ContextWindowStartupConfigV1, DraftStartupConfigV1,
        KvCacheExplicitSourceV1, ModelLibraryRegistrationV1, Phase41ProductionConfigV1,
        PrefixCacheStartupConfigV1,
    };
    use std::collections::BTreeMap;

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

    fn synthetic_derived_identity(
        semantic_model_id: &str,
        source_lock_fingerprints: Vec<String>,
    ) -> DerivedGgufLock {
        DerivedGgufLock {
            schema_version: "derived-gguf-lock-v1".to_owned(),
            fingerprint: "sha256:0000000000000000000000000000000000000000000000000000000000000000"
                .to_owned(),
            semantic_model_id: semantic_model_id.to_owned(),
            source_lock_fingerprints,
            converter: DerivedGgufConverter {
                repository: "sllm-test".to_owned(),
                commit: "test".to_owned(),
                arguments: vec!["test".to_owned()],
                effective_config: BTreeMap::new(),
                environment: BTreeMap::new(),
            },
            output: DerivedGgufOutput {
                path: "test.gguf".to_owned(),
                size_bytes: 1,
                sha256: "sha256:0000000000000000000000000000000000000000000000000000000000000000"
                    .to_owned(),
                metadata_sha256:
                    "sha256:0000000000000000000000000000000000000000000000000000000000000000"
                        .to_owned(),
                tensor_catalog_sha256:
                    "sha256:0000000000000000000000000000000000000000000000000000000000000000"
                        .to_owned(),
            },
        }
    }

    #[test]
    fn gemma4moe_dynamic_identity_requires_the_exact_reviewed_source() {
        let semantic = format!("gemma4moe:{GEMMA4_MOE_MODEL_FINGERPRINT}");
        let valid =
            synthetic_derived_identity(&semantic, vec![GEMMA4_MOE_MODEL_FINGERPRINT.to_owned()]);
        assert_eq!(
            dynamic_model_identity(&valid).unwrap(),
            GEMMA4_MOE_MODEL_FINGERPRINT
        );

        let wrong_source =
            synthetic_derived_identity(&semantic, vec![QWEN35_4B_FINGERPRINT.into()]);
        assert!(dynamic_model_identity(&wrong_source).is_err());

        let wrong_semantic = synthetic_derived_identity(
            "gemma4moe:sha256:0000000000000000000000000000000000000000000000000000000000000000",
            vec![GEMMA4_MOE_MODEL_FINGERPRINT.to_owned()],
        );
        assert!(dynamic_model_identity(&wrong_semantic).is_err());
    }

    #[test]
    fn phase41_cli_defaults_every_opt_in_disabled() {
        let config = parse_args_from(base_args(&[])).expect("default CLI should parse");
        assert_eq!(config.kv_cache_encoding, None);
        assert!(config.mtp_assistant_gguf_path.is_none());
        assert!(config.mtp_assistant_derived_lock_path.is_none());
        assert!(config.webui);
        assert_eq!(config.webui_port, DEFAULT_WEBUI_PORT);
        assert!(config.metrics);
        assert!(
            config
                .cors_origins
                .contains(&"http://localhost:65457".to_owned())
        );
        assert!(
            config
                .cors_origins
                .contains(&"http://127.0.0.1:65457".to_owned())
        );
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
    fn static_gemma_mtp_companion_options_require_a_complete_pair_and_mtp_auto() {
        let config = parse_args_from(base_args(&[
            "--draft",
            "mtp-auto",
            "--mtp-assistant-gguf",
            "assistant.gguf",
            "--mtp-assistant-derived-lock",
            "assistant.lock.json",
        ]))
        .expect("static MTP companion pair should parse");
        assert_eq!(
            config.mtp_assistant_gguf_path,
            Some(PathBuf::from("assistant.gguf"))
        );
        assert_eq!(
            config.mtp_assistant_derived_lock_path,
            Some(PathBuf::from("assistant.lock.json"))
        );
        assert!(matches!(
            config.phase41.draft,
            DraftStartupConfigV1::MtpAuto
        ));
        assert!(usage().contains("--mtp-assistant-gguf PATH"));
        assert!(usage().contains("--mtp-assistant-derived-lock PATH"));

        for arguments in [
            base_args(&["--mtp-assistant-gguf", "assistant.gguf"]),
            base_args(&["--mtp-assistant-derived-lock", "assistant.lock.json"]),
            base_args(&[
                "--mtp-assistant-gguf",
                "assistant.gguf",
                "--mtp-assistant-derived-lock",
                "assistant.lock.json",
            ]),
            base_args(&[
                "--draft",
                "ngram",
                "--draft-ngram-order",
                "2",
                "--draft-width",
                "1",
                "--mtp-assistant-gguf",
                "assistant.gguf",
                "--mtp-assistant-derived-lock",
                "assistant.lock.json",
            ]),
        ] {
            assert!(
                parse_args_from(arguments).is_err(),
                "invalid static MTP companion options must be rejected"
            );
        }
    }

    #[test]
    fn static_gemma_mtp_companion_options_are_not_available_to_dynamic_sources() {
        let pair = [
            "--mtp-assistant-gguf",
            "assistant.gguf",
            "--mtp-assistant-derived-lock",
            "assistant.lock.json",
            "--draft",
            "mtp-auto",
        ];
        let error = parse_args_from(pair)
            .expect_err("library-only startup must reject static MTP companion options");
        assert!(error.contains("static --gguf/--derived-lock"));

        let mut manifest = vec!["--models", "/etc/sllm/models.json"];
        manifest.extend_from_slice(&pair);
        let error = parse_args_from(manifest)
            .expect_err("manifest startup must reject static MTP companion options");
        assert!(error.contains("static --gguf/--derived-lock"));
    }

    #[test]
    fn model_library_gemma_pair_reaches_dynamic_entry_and_enables_mtp() {
        let registration = ModelLibraryRegistrationV1 {
            alias: "gemma4".to_owned(),
            gguf_path: PathBuf::from("/models/gemma4.gguf"),
            derived_lock_path: Some(PathBuf::from("/models/gemma4.lock.json")),
            architecture: "gemma4".to_owned(),
            model_identity: "target".to_owned(),
            plan_identity: "plan".to_owned(),
            resident_bytes: 1,
            device_index: 0,
            target: "gfx1201".to_owned(),
            mtp_assistant_gguf_path: Some(PathBuf::from("/models/gemma4-mtp.gguf")),
            mtp_assistant_derived_lock_path: Some(PathBuf::from("/models/gemma4-mtp.lock.json")),
            mtp_assistant_identity: Some("assistant".to_owned()),
            mtp_semantic_pair_identity: Some("pair".to_owned()),
        };
        let entry = DynamicModelEntryV1::from_library(&registration);
        assert_eq!(
            entry.mtp_assistant_gguf_path,
            registration.mtp_assistant_gguf_path
        );
        assert_eq!(
            entry.mtp_assistant_derived_lock_path,
            registration.mtp_assistant_derived_lock_path
        );
        let phase41 = dynamic_gemma_phase41(
            &Phase41ProductionConfigV1::default(),
            entry.mtp_assistant_gguf_path.as_ref(),
            entry.mtp_assistant_derived_lock_path.as_ref(),
        )
        .expect("complete assistant pair should select MTP auto");
        assert!(matches!(phase41.draft, DraftStartupConfigV1::MtpAuto));
    }

    #[test]
    fn dynamic_gemma_without_pair_disables_mtp_and_rejects_partial_pair() {
        let base = Phase41ProductionConfigV1 {
            draft: DraftStartupConfigV1::MtpAuto,
            ..Phase41ProductionConfigV1::default()
        };
        let target_only = dynamic_gemma_phase41(&base, None, None)
            .expect("target-only dynamic Gemma load should be valid");
        assert!(matches!(target_only.draft, DraftStartupConfigV1::Disabled));
        assert!(
            dynamic_gemma_phase41(&base, Some(&PathBuf::from("assistant.gguf")), None).is_err()
        );
    }

    #[test]
    fn webui_cli_supports_headless_and_custom_port_without_duplicate_cors() {
        let headless = parse_args_from(base_args(&["--webui", "false"]))
            .expect("headless server should parse");
        assert!(!headless.webui);
        assert!(!headless.metrics);
        assert!(headless.cors_origins.is_empty());

        let custom = parse_args_from(base_args(&[
            "--webui",
            "true",
            "--webui-port",
            "32123",
            "--cors-origins",
            "http://localhost:32123",
            "--metrics",
            "false",
        ]))
        .expect("custom WebUI port should parse");
        assert_eq!(custom.webui_port, 32_123);
        assert!(!custom.metrics, "an explicit metrics setting must win");
        assert_eq!(
            custom
                .cors_origins
                .iter()
                .filter(|origin| origin.as_str() == "http://localhost:32123")
                .count(),
            1
        );
        assert!(
            custom
                .cors_origins
                .contains(&"http://127.0.0.1:32123".to_owned())
        );
    }

    #[test]
    fn webui_cli_rejects_ambiguous_or_unusable_ports() {
        for arguments in [
            vec!["--webui", "false", "--webui-port", "32123"],
            vec!["--webui-port", "0"],
            vec!["--listen", "127.0.0.1:0"],
            vec!["--listen", "127.0.0.1:65457"],
        ] {
            assert!(parse_args_from(base_args(&arguments)).is_err());
        }
        assert!(usage().contains("[--webui true|false] [--webui-port PORT]"));
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
        assert!(usage().contains("Qwen default: kv-mxfp8-e4"));
        assert!(usage().contains("Gemma 4 MoE: auto or fp8-static only"));
    }

    #[test]
    fn gemma4moe_kv_policy_accepts_only_auto_or_canonical_static_fp8() {
        assert!(validate_gemma4_moe_kv_request(None).is_ok());
        assert!(validate_gemma4_moe_kv_request(Some(KvCacheEncoding::Fp8E4M3FnStatic)).is_ok());
        for encoding in [
            KvCacheEncoding::Fp16,
            KvCacheEncoding::Fp8E4M3Fn,
            KvCacheEncoding::Nvfp4,
            KvCacheEncoding::Mxfp8E4,
            KvCacheEncoding::Mxfp8E5,
        ] {
            let error = validate_gemma4_moe_kv_request(Some(encoding)).unwrap_err();
            assert!(error.contains("fp8-static"));
        }
    }

    #[test]
    fn gemma4moe_context_override_is_bounded_at_startup() {
        assert!(validate_gemma4_moe_context_override(None).is_ok());
        assert!(
            validate_gemma4_moe_context_override(Some(GEMMA4_RECOMMENDED_CONTEXT_TOKENS as u32))
                .is_ok()
        );
        assert!(
            validate_gemma4_moe_context_override(Some(
                GEMMA4_RECOMMENDED_CONTEXT_TOKENS as u32 + 1
            ))
            .is_err()
        );
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
    fn direct_official_ministral_source_omits_optional_derived_lock() {
        let config = parse_args_from([
            "--gguf",
            "/models/Ministral-3-3B-Instruct-2512-BF16.gguf",
            "--device-index",
            "0",
            "--target",
            "gfx1201",
        ])
        .expect("direct official Ministral source should parse");
        assert_eq!(config.derived_lock, None);
        assert_eq!(config.model, MINISTRAL3_MODEL_ALIAS);

        let derived = parse_args_from(base_args(&[])).expect("derived source should still parse");
        assert_eq!(derived.derived_lock, Some(PathBuf::from("model.lock.json")));
        assert_eq!(derived.model, "qwen3.5-4b");
        assert!(usage().contains("[--derived-lock PATH]"));
    }

    #[test]
    fn model_library_direct_registration_propagates_no_derived_lock() {
        let registration = ModelLibraryRegistrationV1 {
            alias: MINISTRAL3_MODEL_ALIAS.to_owned(),
            gguf_path: PathBuf::from("/models/Ministral-3-3B-Instruct-2512-BF16.gguf"),
            derived_lock_path: None,
            architecture: "mistral3".to_owned(),
            model_identity: "model".to_owned(),
            plan_identity: "plan".to_owned(),
            resident_bytes: 1,
            device_index: 0,
            target: "gfx1201".to_owned(),
            mtp_assistant_gguf_path: None,
            mtp_assistant_derived_lock_path: None,
            mtp_assistant_identity: None,
            mtp_semantic_pair_identity: None,
        };
        let entry = DynamicModelEntryV1::from_library(&registration);
        assert_eq!(entry.derived_lock, None);
        assert_eq!(entry.gguf, registration.gguf_path);
    }

    #[test]
    fn no_model_source_starts_in_webui_library_mode() {
        let config = parse_args_from(std::iter::empty::<String>())
            .expect("model source is selected later from the loopback WebUI");
        assert!(config.library_only);
        assert!(config.models.is_none());
        assert!(config.listen.ip().is_loopback());

        let partial = parse_args_from(["--gguf", "model.gguf"]);
        assert!(
            partial.is_err(),
            "partial legacy model flags remain invalid"
        );
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
