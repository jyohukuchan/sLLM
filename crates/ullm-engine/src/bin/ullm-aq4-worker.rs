// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

use serde::Serialize;
use std::env;
use std::ffi::OsString;
use std::io::{BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use ullm_engine::aq4_worker_backend::{Qwen35Aq4WorkerBackend, Qwen35Aq4WorkerBackendConfig};
use ullm_engine::served_model::{ServedModelError, WorkerBackendKind, load_served_model};
use ullm_engine::sq8_worker_runtime::run_sq8_worker_process;

const PROCESS_IO_BUFFER_BYTES: usize = 64 * 1024;

#[derive(Debug, PartialEq, Eq)]
struct WorkerArgs {
    engine: PathBuf,
    package: PathBuf,
    device_index: u32,
    layers: String,
}

#[derive(Debug, PartialEq, Eq)]
enum WorkerSource {
    Legacy(WorkerArgs),
    ServedModelManifest(PathBuf),
}

enum CliAction {
    Run(WorkerSource),
    Help,
    Version,
}

fn main() -> ExitCode {
    match parse_cli(env::args_os().skip(1)) {
        Ok(CliAction::Help) => {
            eprintln!(
                "Usage: ullm-aq4-worker [--engine PATH] --package PATH [--device-index N] [--layers all|CSV]\n\
                 Gateway form: --artifact AQ4_PACKAGE --package COMPAT_PATH [extra options]\n\
                 Manifest mode: ullm-aq4-worker --served-model-manifest PATH\n\
                 Reads ullm.worker.v1 commands from stdin and writes events to stdout.\n\
                 Compatibility mode invokes the AQ4 engine CLI once per request.\n\
                 AQ4 manifest activation requires the resident backend and currently fails closed."
            );
            ExitCode::SUCCESS
        }
        Ok(CliAction::Version) => {
            eprintln!("ullm-aq4-worker {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        Ok(CliAction::Run(args)) => run_worker(args),
        Err(error) => {
            write_process_log("error", "cli_failed", Some("invalid_cli"), Some(&error));
            ExitCode::FAILURE
        }
    }
}

fn run_worker(source: WorkerSource) -> ExitCode {
    let args = match source {
        WorkerSource::Legacy(args) => {
            ensure_aq4_profile_defaults();
            Ok(args)
        }
        WorkerSource::ServedModelManifest(path) => load_served_model(path)
            .and_then(|model| {
                let current_exe =
                    env::current_exe().map_err(|error| ServedModelError(error.to_string()))?;
                model.worker_startup(WorkerBackendKind::Aq4, &current_exe)
            })
            .and_then(|_| {
                // Manifest identity covers this worker binary, not the legacy sibling
                // `ullm-engine` child. Activation remains disabled until AQ4 execution
                // is resident in this process and covered by the same identity.
                Err(ServedModelError(
                    "AQ4 manifest activation requires the resident backend".into(),
                ))
            }),
    };
    let args = match args {
        Ok(args) => args,
        Err(_) => {
            write_process_log("error", "manifest_failed", Some("invalid_manifest"), None);
            return ExitCode::FAILURE;
        }
    };
    let config = Qwen35Aq4WorkerBackendConfig::new(args.engine, args.package)
        .map(|config| config.with_device_index(args.device_index))
        .and_then(|config| config.with_layers(args.layers));
    let config = match config {
        Ok(config) => config,
        Err(error) => {
            write_process_log("error", "cli_failed", Some("invalid_cli"), Some(&error));
            return ExitCode::FAILURE;
        }
    };
    let input = BufReader::with_capacity(PROCESS_IO_BUFFER_BYTES, std::io::stdin());
    let output = BufWriter::with_capacity(PROCESS_IO_BUFFER_BYTES, std::io::stdout());
    match run_sq8_worker_process(input, output, move || Qwen35Aq4WorkerBackend::load(config)) {
        Ok(_) => {
            write_process_log("info", "process_stopped", None, None);
            ExitCode::SUCCESS
        }
        Err(error) => {
            write_process_log(
                "error",
                "process_failed",
                Some("process_failed"),
                Some(&error),
            );
            ExitCode::FAILURE
        }
    }
}

fn parse_cli(args: impl IntoIterator<Item = OsString>) -> Result<CliAction, String> {
    let args = args.into_iter().collect::<Vec<_>>();
    if args == [OsString::from("--help")] {
        return Ok(CliAction::Help);
    }
    if args == [OsString::from("--version")] {
        return Ok(CliAction::Version);
    }
    if args.iter().any(|value| value == "--served-model-manifest") {
        if args.len() != 2 || args[0] != "--served-model-manifest" || args[1].is_empty() {
            return Err("manifest mode and legacy options are mutually exclusive".into());
        }
        return Ok(CliAction::Run(WorkerSource::ServedModelManifest(
            PathBuf::from(&args[1]),
        )));
    }
    let mut engine = None;
    let mut artifact = None;
    let mut package = None;
    let mut device_index = 1_u32;
    let mut layers = "all".to_string();
    let mut index = 0;
    while index < args.len() {
        let option = args[index]
            .to_str()
            .ok_or_else(|| "AQ4 worker option is not valid UTF-8".to_string())?;
        index += 1;
        let value = args
            .get(index)
            .ok_or_else(|| format!("AQ4 worker option {option} is missing its value"))?;
        match option {
            "--engine" if engine.is_none() => engine = Some(PathBuf::from(value)),
            "--artifact" if artifact.is_none() => artifact = Some(PathBuf::from(value)),
            "--package" if package.is_none() => package = Some(PathBuf::from(value)),
            "--device-index" => {
                device_index = value
                    .to_str()
                    .ok_or_else(|| "AQ4 device index is not valid UTF-8".to_string())?
                    .parse()
                    .map_err(|_| "AQ4 device index must be an unsigned integer".to_string())?;
            }
            "--layers" => {
                layers = value
                    .to_str()
                    .filter(|value| !value.is_empty())
                    .ok_or_else(|| "AQ4 layers must be nonempty UTF-8".to_string())?
                    .to_string();
            }
            "--engine" | "--artifact" | "--package" => {
                return Err(format!("AQ4 worker option {option} was provided twice"));
            }
            _ => return Err(format!("AQ4 worker received unknown option {option}")),
        }
        index += 1;
    }
    Ok(CliAction::Run(WorkerSource::Legacy(WorkerArgs {
        engine: engine.map_or_else(default_engine_path, Ok)?,
        package: artifact
            .or(package)
            .ok_or_else(|| "AQ4 worker --artifact or --package is required".to_string())?,
        device_index,
        layers,
    })))
}

fn default_engine_path() -> Result<PathBuf, String> {
    env::current_exe()
        .map(|path| path.with_file_name("ullm-engine"))
        .map_err(|error| format!("failed to resolve AQ4 worker executable: {error}"))
}

fn ensure_aq4_profile_defaults() {
    let zero_sha256 = "0".repeat(64);
    for (name, value) in [
        ("ULLM_MODEL_ID", "ullm-qwen3.5-9b-aq4"),
        ("ULLM_MODEL_REVISION", "aq4-cli-compat-v0.1"),
        ("ULLM_ARTIFACT_CONTENT_SHA256", zero_sha256.as_str()),
        ("ULLM_PACKAGE_MANIFEST_SHA256", zero_sha256.as_str()),
        ("ULLM_DEVICE", "gfx1201"),
        ("ULLM_EXECUTION_PROFILE", "rdna4_aq4_cli_compat"),
        ("ULLM_MODEL_CONTEXT_LENGTH", "4096"),
        ("ULLM_MAX_NEW_TOKENS", "512"),
        ("ULLM_VOCAB_SIZE", "248320"),
        ("ULLM_EOS_TOKEN_IDS", "248044,248046"),
        ("ULLM_TOP_K", "1"),
    ] {
        if env::var_os(name).is_none() {
            // SAFETY: no worker reader, writer, or inference thread exists yet.
            unsafe { env::set_var(name, value) };
        }
    }
}

#[derive(Serialize)]
struct ProcessLog<'a> {
    schema_version: &'static str,
    level: &'static str,
    event: &'static str,
    phase: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    error_code: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    detail: Option<&'a str>,
}

fn write_process_log(
    level: &'static str,
    event: &'static str,
    error_code: Option<&'static str>,
    detail: Option<&str>,
) {
    let record = ProcessLog {
        schema_version: "ullm.worker.log.v1",
        level,
        event,
        phase: "process",
        error_code,
        detail,
    };
    let mut stderr = std::io::stderr().lock();
    let _ = serde_json::to_writer(&mut stderr, &record);
    let _ = stderr.write_all(b"\n");
    let _ = stderr.flush();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(values: &[&str]) -> Vec<OsString> {
        values.iter().map(OsString::from).collect()
    }

    #[test]
    fn cli_accepts_required_paths_and_optional_device_and_layers() {
        let CliAction::Run(parsed) = parse_cli(args(&[
            "--engine",
            "/engine",
            "--package",
            "/package",
            "--device-index",
            "7",
            "--layers",
            "0,1",
        ]))
        .unwrap() else {
            panic!("expected run action");
        };
        assert_eq!(
            parsed,
            WorkerSource::Legacy(WorkerArgs {
                engine: PathBuf::from("/engine"),
                package: PathBuf::from("/package"),
                device_index: 7,
                layers: "0,1".to_string(),
            })
        );
    }

    #[test]
    fn cli_rejects_missing_and_duplicate_required_options() {
        for values in [
            vec![],
            vec!["--engine", "/a", "--engine", "/b", "--package", "/p"],
        ] {
            assert!(parse_cli(args(&values)).is_err(), "{values:?}");
        }
    }

    #[test]
    fn cli_accepts_gateway_artifact_form_and_prefers_artifact() {
        let CliAction::Run(parsed) = parse_cli(args(&[
            "--engine",
            "/engine",
            "--artifact",
            "/aq4",
            "--package",
            "/compat",
        ]))
        .unwrap() else {
            panic!("expected run action");
        };
        let WorkerSource::Legacy(parsed) = parsed else {
            panic!("expected legacy source")
        };
        assert_eq!(parsed.package, PathBuf::from("/aq4"));
    }

    #[test]
    fn cli_accepts_manifest_as_an_exclusive_source() {
        let CliAction::Run(parsed) =
            parse_cli(args(&["--served-model-manifest", "/served-model.json"])).unwrap()
        else {
            panic!("expected run action");
        };
        assert_eq!(
            parsed,
            WorkerSource::ServedModelManifest(PathBuf::from("/served-model.json"))
        );
        assert!(
            parse_cli(args(&[
                "--served-model-manifest",
                "/served-model.json",
                "--device-index",
                "7"
            ]))
            .is_err()
        );
        assert!(
            parse_cli(args(&[
                "--served-model-manifest",
                "/served-model.json",
                "--package",
                "/package"
            ]))
            .is_err()
        );
    }

    #[test]
    fn manifest_failure_log_omits_sensitive_detail() {
        let value = serde_json::to_value(ProcessLog {
            schema_version: "ullm.worker.log.v1",
            level: "error",
            event: "manifest_failed",
            phase: "process",
            error_code: Some("invalid_manifest"),
            detail: None,
        })
        .unwrap();
        assert_eq!(value["error_code"], "invalid_manifest");
        assert!(value.get("detail").is_none());
    }
}
