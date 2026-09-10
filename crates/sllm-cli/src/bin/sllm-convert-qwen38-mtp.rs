//! Host-only converter for the reviewed Qwen3.8 MTP companion sidecar.
//!
//! The target artifact remains unchanged.  This command reads the fixed
//! Qwen3.5-27B semantic lock, verifies the Unsloth NVFP4 artifact, and emits
//! an authenticated `manifest.json` plus `payload.safetensors` directory.

use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

use serde_json::json;
use sllm_core::{
    MtpWeightEncoding, convert_qwen38_mtp_quantized_sidecar, parse_model_lock,
    verify_unsloth_qwen38_nvfp4,
};

const USAGE: &str = "usage: sllm-convert-qwen38-mtp --artifact-root ABSOLUTE_DIRECTORY --encoding mxfp8|mxfp6 --output-dir ABSOLUTE_DIRECTORY";

fn main() -> ExitCode {
    match run(env::args_os().skip(1).collect()) {
        Ok(report) => {
            println!("{report}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("sllm-convert-qwen38-mtp: {error}");
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
    }
}

fn run(args: Vec<std::ffi::OsString>) -> Result<String, String> {
    let mut artifact_root = None;
    let mut encoding = None;
    let mut output_dir = None;
    let mut index = 0;
    while index < args.len() {
        let flag = args[index].to_str().ok_or("arguments must be UTF-8")?;
        if flag == "--help" || flag == "-h" {
            return Ok(USAGE.to_owned());
        }
        let value = args
            .get(index + 1)
            .ok_or_else(|| format!("{flag} requires a value"))?;
        match flag {
            "--artifact-root" => artifact_root = Some(PathBuf::from(value)),
            "--encoding" => encoding = Some(value.to_str().ok_or("encoding must be UTF-8")?),
            "--output-dir" => output_dir = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown argument {flag}")),
        }
        index += 2;
    }
    let artifact_root = artifact_root.ok_or("--artifact-root is required")?;
    let output_dir = output_dir.ok_or("--output-dir is required")?;
    if !artifact_root.is_absolute() || !output_dir.is_absolute() {
        return Err("artifact and output paths must be absolute".to_owned());
    }
    let encoding = match encoding.ok_or("--encoding is required")? {
        "mxfp8" | "mxfp8-w8a8-e4m3-block32-e8m0" => MtpWeightEncoding::Mxfp8W8A8Block32E8M0,
        "mxfp6" | "mxfp6-w6a6-e3m2-block32-e8m0" => MtpWeightEncoding::Mxfp6W6A6Block32E8M0,
        value => return Err(format!("unsupported --encoding value {value:?}")),
    };
    if output_dir.exists() {
        return Err("--output-dir already exists; choose a new directory".to_owned());
    }
    let lock = parse_model_lock(include_bytes!(
        "../../../../docs/models/locks/qwen3.5-27b-bf16.json"
    ))
    .map_err(|error| format!("embedded reviewed Qwen3.5-27B lock is invalid: {error}"))?;
    let artifact = verify_unsloth_qwen38_nvfp4(&artifact_root)
        .map_err(|error| format!("Qwen3.8 NVFP4 artifact verification failed: {error}"))?;
    let verified = convert_qwen38_mtp_quantized_sidecar(&lock, &artifact, encoding, &output_dir)
        .map_err(|error| format!("MTP sidecar conversion failed: {error}"))?;
    Ok(json!({
        "kind": "qwen38-mtp-companion",
        "encoding": verified.encoding().manifest_name(),
        "output_dir": output_dir,
        "manifest_fingerprint": verified.manifest_fingerprint(),
        "combined_recipe_digest": verified.combined_recipe_digest(artifact.recipe_digest()),
        "files": ["manifest.json", "payload.safetensors"],
    })
    .to_string())
}
