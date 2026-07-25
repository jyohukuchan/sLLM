// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! CPU-only, hash-checked scan of a canonical `SQ8_0` artifact for FNUZ prepack.

use std::env;
use std::path::PathBuf;
use ullm_engine::sq_canonical::read_sq8_canonical_artifact;
use ullm_engine::sq8_fnuz_prepack::{
    SQ8_FNUZ_PREPACK_SCAN_CHUNK_BYTES, scan_sq8_canonical_artifact_for_fnuz_prepack,
};

struct Options {
    artifact: PathBuf,
    chunk_bytes: usize,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("sq8_fnuz_prepack_scan: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let options = parse_options(env::args().skip(1))?;
    let artifact = read_sq8_canonical_artifact(&options.artifact)?;
    let report = scan_sq8_canonical_artifact_for_fnuz_prepack(&artifact, options.chunk_bytes)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&report)
            .map_err(|error| format!("failed to encode scan report as JSON: {error}"))?
    );
    Ok(())
}

fn parse_options(arguments: impl Iterator<Item = String>) -> Result<Options, String> {
    let mut values = arguments;
    let mut artifact = None;
    let mut chunk_bytes = SQ8_FNUZ_PREPACK_SCAN_CHUNK_BYTES;
    while let Some(argument) = values.next() {
        match argument.as_str() {
            "--artifact" => {
                artifact =
                    Some(PathBuf::from(values.next().ok_or_else(|| {
                        "--artifact requires a directory".to_string()
                    })?));
            }
            "--chunk-bytes" => {
                let value = values
                    .next()
                    .ok_or_else(|| "--chunk-bytes requires a positive integer".to_string())?;
                chunk_bytes = value.parse::<usize>().map_err(|error| {
                    format!("--chunk-bytes must be a positive integer, got {value:?}: {error}")
                })?;
                if chunk_bytes == 0 {
                    return Err("--chunk-bytes must be greater than zero".to_string());
                }
            }
            "--help" | "-h" => {
                return Err(
                    "usage: sq8_fnuz_prepack_scan --artifact ARTIFACT_DIR [--chunk-bytes BYTES]"
                        .to_string(),
                );
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    Ok(Options {
        artifact: artifact.ok_or_else(|| "--artifact is required".to_string())?,
        chunk_bytes,
    })
}
