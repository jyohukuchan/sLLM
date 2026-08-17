use std::collections::BTreeMap;
use std::env;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use sllm_core::{
    GEMMA4_RECOMMENDED_CONTEXT_TOKENS, QWEN35_RECOMMENDED_CONTEXT_TOKENS, ReviewedModelLock,
    builtin_reviewed_model_lock, read_derived_gguf_lock,
};
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
    gguf: PathBuf,
    derived_lock: PathBuf,
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
    context_length: Option<u32>,
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
    let gguf = PathBuf::from(take_required(&mut values, "--gguf")?);
    let derived_lock = PathBuf::from(take_required(&mut values, "--derived-lock")?);
    let device_index = parse_value(
        &take_required(&mut values, "--device-index")?,
        "device index",
    )?;
    let target = take_required(&mut values, "--target")?;
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
    let context_length = values
        .remove("--context-length")
        .map(|value| parse_value::<u32>(&value, "context length"))
        .transpose()?;
    if context_length == Some(0) {
        return Err("context length must be nonzero".to_owned());
    }
    if let Some(flag) = values.keys().next() {
        return Err(format!("unknown argument {flag}\n{}", usage()));
    }
    Ok(Config {
        gguf,
        derived_lock,
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
        context_length,
    })
}

fn run(config: Config) -> Result<(), String> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|error| format!("Tokio runtime construction failed: {error}"))?;
    runtime.block_on(async move {
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
        let backend = if gguf_moe {
            ActiveBackend::Qwen(Arc::new(
                QwenChatBackendV1::open(QwenBackendConfigV1 {
                    gguf_path: config.gguf,
                    derived_lock_path: config.derived_lock,
                    device_index: config.device_index,
                    target: config.target,
                    completion_timeout: config.completion_timeout,
                    shutdown_timeout: config.shutdown_timeout,
                    context_length: config
                        .context_length
                        .unwrap_or(QWEN35_RECOMMENDED_CONTEXT_TOKENS as u32),
                })
                .map_err(|error| error.to_string())?,
            ))
        } else {
            match reviewed.expect("non-MoE GGUF resolved a reviewed lock") {
                ReviewedModelLock::Qwen35(_) => ActiveBackend::Qwen(Arc::new(
                    QwenChatBackendV1::open(QwenBackendConfigV1 {
                        gguf_path: config.gguf,
                        derived_lock_path: config.derived_lock,
                        device_index: config.device_index,
                        target: config.target,
                        completion_timeout: config.completion_timeout,
                        shutdown_timeout: config.shutdown_timeout,
                        context_length: config
                            .context_length
                            .unwrap_or(QWEN35_RECOMMENDED_CONTEXT_TOKENS as u32),
                    })
                    .map_err(|error| error.to_string())?,
                )),
                ReviewedModelLock::Gemma4(_) => ActiveBackend::Gemma(Arc::new(
                    Gemma4ChatBackendV1::open(Gemma4BackendConfigV1 {
                        gguf_path: config.gguf,
                        derived_lock_path: config.derived_lock,
                        device_index: config.device_index,
                        target: config.target,
                        completion_timeout: config.completion_timeout,
                        shutdown_timeout: config.shutdown_timeout,
                        context_length: config
                            .context_length
                            .unwrap_or(GEMMA4_RECOMMENDED_CONTEXT_TOKENS as u32),
                    })
                    .map_err(|error| error.to_string())?,
                )),
            }
        };
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
                "context_length": backend.context_length(),
                "official_recommended_context_tokens": backend.recommended_context_tokens(),
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
    "usage: sllm-server --gguf PATH --derived-lock PATH --device-index N --target GFX [--listen HOST:PORT] [--model ALIAS] [--api-key-env NAME] [--compatibility-profile strict|openwebui] [--context-length TOKENS] [--queue-capacity N] [--event-capacity N] [--request-timeout-seconds N] [--completion-timeout-seconds N] [--shutdown-timeout-seconds N]"
}

#[cfg(test)]
mod tests {
    use super::context_length_warning;

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
}
