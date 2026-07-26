// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Dedicated manifest-only worker for the Gemma4 E2B resident BF16 backend.

use serde::Serialize;
use std::env;
use std::ffi::OsString;
use std::io::{BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use ullm_engine::gemma4_worker_backend::{
    Gemma4E2bWorkerBackend, validate_gemma4_e2b_served_model,
};
use ullm_engine::served_model::{WorkerBackendKind, load_served_model};
use ullm_engine::worker_runtime::run_worker_process_with_profile;

const PROCESS_IO_BUFFER_BYTES: usize = 64 * 1024;

enum CliAction {
    Run(PathBuf),
    Help,
    Version,
}

fn main() -> ExitCode {
    match parse_cli(env::args_os().skip(1)) {
        Ok(CliAction::Help) => {
            eprintln!(
                "Usage: ullm-gemma4-worker --served-model-manifest PATH\n\\
                 Reads ullm.worker.v1 commands from stdin and writes matching events to stdout.\n\\
                 This worker is manifest-only and accepts only the Gemma4 E2B BF16_0 contract."
            );
            ExitCode::SUCCESS
        }
        Ok(CliAction::Version) => {
            eprintln!("ullm-gemma4-worker {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        Ok(CliAction::Run(manifest)) => run_worker(manifest),
        Err(error) => {
            write_process_log("error", "cli_failed", Some("invalid_cli"), Some(&error));
            ExitCode::FAILURE
        }
    }
}

fn run_worker(manifest_path: PathBuf) -> ExitCode {
    let startup = (|| {
        let model = load_served_model(manifest_path)?;
        validate_gemma4_e2b_served_model(&model)?;
        let current_exe = env::current_exe()
            .map_err(|error| ullm_engine::served_model::ServedModelError(error.to_string()))?;
        model.worker_startup(WorkerBackendKind::Gemma4E2b, &current_exe)
    })();
    let startup = match startup {
        Ok(startup) => startup,
        Err(_) => {
            write_process_log("error", "manifest_failed", Some("invalid_manifest"), None);
            return ExitCode::FAILURE;
        }
    };
    if startup.artifact_dir.is_some() || startup.reasoning.is_some() || startup.profile.top_k != 1 {
        write_process_log("error", "manifest_failed", Some("invalid_manifest"), None);
        return ExitCode::FAILURE;
    }
    let profile = startup.profile.into_worker_profile();
    let package_dir = startup.package_dir;
    let input = BufReader::with_capacity(PROCESS_IO_BUFFER_BYTES, std::io::stdin());
    let output = BufWriter::with_capacity(PROCESS_IO_BUFFER_BYTES, std::io::stdout());
    match run_worker_process_with_profile(input, output, profile.clone(), move || {
        Gemma4E2bWorkerBackend::load(
            package_dir,
            profile.context_length,
            profile.max_new_tokens,
            profile.vocab_size,
            profile.eos_token_ids,
        )
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
    if args.len() == 2 && args[0] == "--served-model-manifest" && !args[1].is_empty() {
        return Ok(CliAction::Run(PathBuf::from(&args[1])));
    }
    Err("Gemma4 worker requires exactly --served-model-manifest PATH".into())
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
    fn cli_accepts_only_manifest_mode() {
        let CliAction::Run(path) =
            parse_cli(args(&["--served-model-manifest", "/sealed/gemma4.json"])).unwrap()
        else {
            panic!("expected manifest mode");
        };
        assert_eq!(path, PathBuf::from("/sealed/gemma4.json"));
        for invalid in [
            vec![],
            vec!["--model-dir", "/model"],
            vec!["--served-model-manifest"],
            vec!["--served-model-manifest", "/manifest", "--extra"],
        ] {
            assert!(parse_cli(args(&invalid)).is_err());
        }
    }
}
