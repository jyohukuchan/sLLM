// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

use serde::Serialize;
use std::env;
use std::ffi::OsString;
use std::io::{BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use ullm_engine::served_model::{WorkerBackendKind, load_served_model};
use ullm_engine::sq8_worker_backend::{Qwen3Sq8WorkerBackend, Qwen3Sq8WorkerBackendConfig};
use ullm_engine::sq8_worker_protocol::configured_worker_profile;
use ullm_engine::sq8_worker_runtime::run_sq8_worker_process_with_profile;

const PROCESS_IO_BUFFER_BYTES: usize = 64 * 1024;

#[derive(Debug, PartialEq, Eq)]
struct WorkerArgs {
    artifact: PathBuf,
    package: PathBuf,
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
                "Usage: ullm-sq8-worker --artifact PATH --package PATH\n\
                 Manifest mode: ullm-sq8-worker --served-model-manifest PATH\n\
                 Reads ullm.worker.v1/v2 commands from stdin and writes matching events to stdout."
            );
            ExitCode::SUCCESS
        }
        Ok(CliAction::Version) => {
            eprintln!("ullm-sq8-worker {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        Ok(CliAction::Run(args)) => run_worker(args),
        Err(_) => {
            write_process_log("error", "cli_failed", Some("invalid_cli"), None);
            ExitCode::FAILURE
        }
    }
}

fn run_worker(source: WorkerSource) -> ExitCode {
    let startup = match source {
        WorkerSource::Legacy(args) => {
            Ok((args.artifact, args.package, configured_worker_profile()))
        }
        WorkerSource::ServedModelManifest(path) => load_served_model(path)
            .and_then(|model| {
                let current_exe = env::current_exe().map_err(|error| {
                    ullm_engine::served_model::ServedModelError(error.to_string())
                })?;
                model.worker_startup(WorkerBackendKind::Sq8, &current_exe)
            })
            .and_then(|startup| {
                Ok((
                    startup.artifact_dir.expect("SQ8 startup has an artifact"),
                    startup.package_dir,
                    startup.profile.into_worker_profile(),
                ))
            }),
    };
    let (artifact, package, profile) = match startup {
        Ok(startup) => startup,
        Err(_) => {
            write_process_log("error", "manifest_failed", Some("invalid_manifest"), None);
            return ExitCode::FAILURE;
        }
    };
    let config = match Qwen3Sq8WorkerBackendConfig::new(artifact, package) {
        Ok(config) => config,
        Err(_) => {
            write_process_log("error", "cli_failed", Some("invalid_cli"), None);
            return ExitCode::FAILURE;
        }
    };
    let input = BufReader::with_capacity(PROCESS_IO_BUFFER_BYTES, std::io::stdin());
    let output = BufWriter::with_capacity(PROCESS_IO_BUFFER_BYTES, std::io::stdout());
    match run_sq8_worker_process_with_profile(input, output, profile, move || {
        Qwen3Sq8WorkerBackend::load(config)
    }) {
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

    let mut artifact = None;
    let mut package = None;
    let mut served_model_manifest = None;
    let mut index = 0;
    while index < args.len() {
        let option = args[index]
            .to_str()
            .ok_or_else(|| "worker option name is not valid UTF-8".to_string())?;
        let target = match option {
            "--artifact" => &mut artifact,
            "--package" => &mut package,
            "--served-model-manifest" => &mut served_model_manifest,
            _ => return Err("worker received an unknown option".into()),
        };
        if target.is_some() {
            return Err("worker option was provided more than once".into());
        }
        index += 1;
        let value = args
            .get(index)
            .ok_or_else(|| "worker option is missing its path".to_string())?;
        if value.is_empty() {
            return Err("worker path must be nonempty".into());
        }
        *target = Some(PathBuf::from(value));
        index += 1;
    }

    if let Some(path) = served_model_manifest {
        if artifact.is_some() || package.is_some() {
            return Err("manifest mode and legacy path options are mutually exclusive".into());
        }
        return Ok(CliAction::Run(WorkerSource::ServedModelManifest(path)));
    }
    Ok(CliAction::Run(WorkerSource::Legacy(WorkerArgs {
        artifact: artifact.ok_or_else(|| "worker artifact path is required".to_string())?,
        package: package.ok_or_else(|| "worker package path is required".to_string())?,
    })))
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
    fn cli_requires_exact_artifact_and_package_options_in_either_order() {
        for values in [
            ["--artifact", "/artifact", "--package", "/package"],
            ["--package", "/package", "--artifact", "/artifact"],
        ] {
            let CliAction::Run(parsed) = parse_cli(args(&values)).unwrap() else {
                panic!("expected run action");
            };
            assert_eq!(
                parsed,
                WorkerSource::Legacy(WorkerArgs {
                    artifact: PathBuf::from("/artifact"),
                    package: PathBuf::from("/package")
                })
            );
        }
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
                "--artifact",
                "/artifact",
                "--package",
                "/package"
            ]))
            .is_err()
        );
    }

    #[test]
    fn cli_rejects_missing_duplicate_unknown_and_inline_options() {
        for values in [
            vec![],
            vec!["--artifact", "/artifact"],
            vec!["--package", "/package"],
            vec!["--artifact", "/a", "--artifact", "/b", "--package", "/p"],
            vec!["--artifact=/a", "--package", "/p"],
            vec!["positional", "--artifact", "/a", "--package", "/p"],
        ] {
            assert!(parse_cli(args(&values)).is_err(), "{values:?}");
        }
    }

    #[test]
    fn cli_help_and_version_must_be_the_only_argument() {
        assert!(matches!(
            parse_cli(args(&["--help"])).unwrap(),
            CliAction::Help
        ));
        assert!(matches!(
            parse_cli(args(&["--version"])).unwrap(),
            CliAction::Version
        ));
        assert!(parse_cli(args(&["--help", "--version"])).is_err());
    }

    #[test]
    fn process_log_schema_omits_payload_and_path_fields() {
        let value = serde_json::to_value(ProcessLog {
            schema_version: "ullm.worker.log.v1",
            level: "error",
            event: "process_failed",
            phase: "process",
            error_code: Some("process_failed"),
            detail: Some("startup: invalid configuration"),
        })
        .unwrap();
        assert_eq!(value["schema_version"], "ullm.worker.log.v1");
        assert_eq!(value["error_code"], "process_failed");
        assert!(value.get("artifact").is_none());
        assert!(value.get("package").is_none());
        assert_eq!(value["detail"], "startup: invalid configuration");
        assert!(value.get("prompt_token_ids").is_none());

        let manifest_failure = serde_json::to_value(ProcessLog {
            schema_version: "ullm.worker.log.v1",
            level: "error",
            event: "manifest_failed",
            phase: "process",
            error_code: Some("invalid_manifest"),
            detail: None,
        })
        .unwrap();
        assert!(manifest_failure.get("detail").is_none());
    }
}
