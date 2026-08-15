use std::collections::BTreeMap;
use std::env;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use sllm_core::{ReviewedModelLock, read_reviewed_model_lock};
use sllm_server::{
    ChatGenerationBackendV1, Gemma4BackendConfigV1, Gemma4ChatBackendV1, ModelRegistryEntryV1,
    ModelRegistryV1, ProductionShutdownAuditV1, QwenBackendConfigV1, QwenChatBackendV1,
    SchedulerConfigV1, SchedulerV1, ServerConfigV1, build_router_v1,
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
    lock: PathBuf,
    cache: PathBuf,
    device_index: u32,
    target: String,
    listen: SocketAddr,
    model: String,
    api_key_env: Option<String>,
    openwebui_compatibility: bool,
    queue_capacity: usize,
    event_capacity: usize,
    request_timeout: Duration,
    completion_timeout: Duration,
    shutdown_timeout: Duration,
    fp8_manifest: Option<PathBuf>,
    fp8_artifact: Option<PathBuf>,
    fp8_provider: Option<String>,
}

fn parse_args() -> Result<Config, String> {
    let mut raw = env::args().skip(1);
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
    let lock = PathBuf::from(take_required(&mut values, "--lock")?);
    let cache = PathBuf::from(take_required(&mut values, "--cache")?);
    let device_index = parse_value(
        &take_required(&mut values, "--device-index")?,
        "device index",
    )?;
    let target = take_required(&mut values, "--target")?;
    let fp8_manifest = values.remove("--fp8-manifest").map(PathBuf::from);
    let fp8_artifact = values.remove("--fp8-artifact").map(PathBuf::from);
    let fp8_provider = values.remove("--fp8-provider");
    let listen = parse_value(
        &values
            .remove("--listen")
            .unwrap_or_else(|| "127.0.0.1:8080".to_owned()),
        "listen address",
    )?;
    let model = values
        .remove("--model")
        .unwrap_or_else(|| "qwen3.5-4b".to_owned());
    let api_key_env = values.remove("--api-key-env");
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
    if let Some(flag) = values.keys().next() {
        return Err(format!("unknown argument {flag}\n{}", usage()));
    }
    Ok(Config {
        lock,
        cache,
        device_index,
        target,
        listen,
        model,
        api_key_env,
        openwebui_compatibility,
        queue_capacity,
        event_capacity,
        request_timeout,
        completion_timeout,
        shutdown_timeout,
        fp8_manifest,
        fp8_artifact,
        fp8_provider,
    })
}

fn run(config: Config) -> Result<(), String> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|error| format!("Tokio runtime construction failed: {error}"))?;
    runtime.block_on(async move {
        let reviewed = read_reviewed_model_lock(&config.lock)
            .map_err(|error| format!("model lock validation failed: {error}"))?;
        let backend = match reviewed {
            ReviewedModelLock::Qwen35(_) => ActiveBackend::Qwen(Arc::new(
                QwenChatBackendV1::open(QwenBackendConfigV1 {
                    lock_path: config.lock,
                    cache_path: config.cache,
                    device_index: config.device_index,
                    target: config.target,
                    completion_timeout: config.completion_timeout,
                    shutdown_timeout: config.shutdown_timeout,
                    fp8_manifest_path: config.fp8_manifest,
                    fp8_artifact_path: config.fp8_artifact,
                    fp8_provider: config.fp8_provider,
                })
                .map_err(|error| error.to_string())?,
            )),
            ReviewedModelLock::Gemma4(_) => {
                if config.fp8_manifest.is_some()
                    || config.fp8_artifact.is_some()
                    || config.fp8_provider.is_some()
                {
                    return Err("Gemma 4 server profile supports locked BF16 weights only".to_owned());
                }
                ActiveBackend::Gemma(Arc::new(
                    Gemma4ChatBackendV1::open(Gemma4BackendConfigV1 {
                        lock_path: config.lock,
                        cache_path: config.cache,
                        device_index: config.device_index,
                        target: config.target,
                        completion_timeout: config.completion_timeout,
                        shutdown_timeout: config.shutdown_timeout,
                    })
                    .map_err(|error| error.to_string())?,
                ))
            }
        };
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
        let scheduler = SchedulerV1::new(
            SchedulerConfigV1::new(
                config.queue_capacity,
                config.event_capacity,
                config.request_timeout,
            )
            .map_err(|error| error.to_string())?,
        );
        let bearer = config
            .api_key_env
            .as_deref()
            .map(|name| {
                env::var(name).map_err(|_| format!("API key environment variable {name} is absent"))
            })
            .transpose()?;
        let server_config = if config.openwebui_compatibility {
            ServerConfigV1::openwebui_compatible(bearer)
        } else {
            ServerConfigV1::new(bearer)
        }
        .map_err(|error| error.to_string())?;
        let router = build_router_v1(
            registry,
            scheduler.clone(),
            server_config,
        );
        let listener = tokio::net::TcpListener::bind(config.listen)
            .await
            .map_err(|error| format!("listen failed: {error}"))?;
        let address = listener
            .local_addr()
            .map_err(|error| format!("local address query failed: {error}"))?;
        println!(
            "{}",
            serde_json::json!({
                "event": "ready",
                "listen": address.to_string(),
                "target": backend.target(),
                "model_fingerprint": backend.model_fingerprint(),
                "compatibility_profile": if config.openwebui_compatibility { "openwebui" } else { "strict" },
            })
        );
        axum::serve(listener, router)
            .with_graceful_shutdown(async {
                let _ = tokio::signal::ctrl_c().await;
            })
            .await
            .map_err(|error| format!("HTTP server failed: {error}"))?;
        scheduler.shutdown();
        let report = backend.shutdown().map_err(|error| error.to_string())?;
        println!(
            "{}",
            serde_json::json!({"event": "shutdown_audit", "report": report})
        );
        Ok(())
    })
}

enum ActiveBackend {
    Qwen(Arc<QwenChatBackendV1>),
    Gemma(Arc<Gemma4ChatBackendV1>),
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

    fn target(&self) -> &str {
        match self {
            Self::Qwen(backend) => backend.target(),
            Self::Gemma(backend) => backend.target(),
        }
    }

    fn shutdown(&self) -> Result<ProductionShutdownAuditV1, sllm_server::BackendErrorV1> {
        match self {
            Self::Qwen(backend) => backend.shutdown(),
            Self::Gemma(backend) => backend.shutdown(),
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

fn usage() -> &'static str {
    "usage: sllm-server --lock PATH --cache PATH --device-index N --target GFX [--fp8-manifest PATH --fp8-artifact PATH --fp8-provider native|native-fnuz|emulation|converted-bf16] [--listen HOST:PORT] [--model ALIAS] [--api-key-env NAME] [--compatibility-profile strict|openwebui] [--queue-capacity N] [--event-capacity N] [--request-timeout-seconds N] [--completion-timeout-seconds N] [--shutdown-timeout-seconds N]"
}
