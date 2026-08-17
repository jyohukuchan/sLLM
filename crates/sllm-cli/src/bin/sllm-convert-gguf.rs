use sllm_core::{
    DerivedGgufConverter, DerivedGgufLock, GgufWritePlan, GgufWriteReport,
    QWEN35_MOE_MODEL_FINGERPRINT, UNSLOTH_GEMMA4_NVFP4_MODEL_SHA256, build_gemma4_nvfp4_gguf_plan,
    build_qwen35_bf16_gguf_plan, build_qwen35_fp8_gguf_plan, build_qwen35_moe_mxfp4_gguf_plan,
    read_model_lock, read_reviewed_model_lock, verify_fp8_sidecar, verify_qwen35_moe_artifact,
    verify_unsloth_gemma4_nvfp4, write_gemma4_nvfp4_gguf, write_qwen35_bf16_gguf,
    write_qwen35_fp8_gguf, write_qwen35_moe_mxfp4_gguf,
};
use std::collections::BTreeMap;
use std::env;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;

const REPOSITORY: &str = "https://github.com/89chin/sLLM";

#[derive(Debug)]
struct Arguments {
    kind: String,
    lock: Option<PathBuf>,
    cache: PathBuf,
    manifest: Option<PathBuf>,
    artifact: Option<PathBuf>,
    output: Option<PathBuf>,
    derived_lock: Option<PathBuf>,
    converter_commit: Option<String>,
    dry_run: bool,
    raw: Vec<String>,
}

fn main() -> ExitCode {
    match run(env::args().skip(1).collect()) {
        Ok(output) => {
            println!("{output}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("sllm-convert-gguf: {error}");
            ExitCode::from(2)
        }
    }
}

fn run(raw: Vec<String>) -> Result<String, String> {
    if raw
        .iter()
        .any(|argument| matches!(argument.as_str(), "--help" | "-h"))
    {
        return Ok(help());
    }
    let arguments = parse_arguments(raw)?;
    match arguments.kind.as_str() {
        "qwen35-bf16" => run_qwen35_bf16(&arguments),
        "qwen35-fp8" => run_qwen35_fp8(&arguments),
        "gemma4-nvfp4" => run_gemma4_nvfp4(&arguments),
        "qwen35moe-mxfp4" => run_qwen35_moe(&arguments),
        kind => Err(format!("unsupported --kind `{kind}`")),
    }
}

fn run_qwen35_bf16(arguments: &Arguments) -> Result<String, String> {
    let lock_path = required_path(&arguments.lock, "--lock")?;
    let lock = read_model_lock(lock_path).map_err(|error| error.to_string())?;
    let cache = lock
        .verify_cache(&arguments.cache)
        .map_err(|error| error.to_string())?;
    let plan = build_qwen35_bf16_gguf_plan(&lock, &cache).map_err(|error| error.to_string())?;
    finish_conversion(
        arguments,
        "qwen35",
        format!("qwen35:{}", lock.fingerprint()),
        vec![lock.fingerprint().to_owned()],
        "source-preserving BF16/F16/F32",
        &plan,
        |output| write_qwen35_bf16_gguf(&lock, &cache, output).map_err(|e| e.to_string()),
    )
}

fn run_qwen35_fp8(arguments: &Arguments) -> Result<String, String> {
    let lock_path = required_path(&arguments.lock, "--lock")?;
    let manifest = required_path(&arguments.manifest, "--manifest")?;
    let artifact_path = required_path(&arguments.artifact, "--artifact")?;
    let lock = read_model_lock(lock_path).map_err(|error| error.to_string())?;
    let cache = lock
        .verify_cache(&arguments.cache)
        .map_err(|error| error.to_string())?;
    let sidecar = verify_fp8_sidecar(manifest, artifact_path, lock_path, &lock)
        .map_err(|error| error.to_string())?;
    let plan =
        build_qwen35_fp8_gguf_plan(&lock, &cache, &sidecar).map_err(|error| error.to_string())?;
    finish_conversion(
        arguments,
        "qwen35",
        format!("qwen35:{}", lock.fingerprint()),
        vec![
            lock.fingerprint().to_owned(),
            sidecar.manifest_fingerprint().to_owned(),
        ],
        "FP8 E4M3FN values with F32 channel scales",
        &plan,
        |output| write_qwen35_fp8_gguf(&lock, &cache, &sidecar, output).map_err(|e| e.to_string()),
    )
}

fn run_gemma4_nvfp4(arguments: &Arguments) -> Result<String, String> {
    let lock_path = required_path(&arguments.lock, "--lock")?;
    let lock = match read_reviewed_model_lock(lock_path).map_err(|error| error.to_string())? {
        sllm_core::ReviewedModelLock::Gemma4(lock) => lock,
        _ => return Err("--lock is not a reviewed Gemma 4 lock".to_owned()),
    };
    let artifact =
        verify_unsloth_gemma4_nvfp4(&arguments.cache).map_err(|error| error.to_string())?;
    let plan = build_gemma4_nvfp4_gguf_plan(&lock, &artifact).map_err(|error| error.to_string())?;
    let fingerprint = format!("sha256:{UNSLOTH_GEMMA4_NVFP4_MODEL_SHA256}");
    finish_conversion(
        arguments,
        "gemma4",
        format!("gemma4:{}", lock.fingerprint()),
        vec![lock.fingerprint().to_owned(), fingerprint],
        "mixed BF16/FP8/NVFP4 lossless standard-block repack",
        &plan,
        |output| write_gemma4_nvfp4_gguf(&lock, &artifact, output).map_err(|e| e.to_string()),
    )
}

fn run_qwen35_moe(arguments: &Arguments) -> Result<String, String> {
    let artifact =
        verify_qwen35_moe_artifact(&arguments.cache).map_err(|error| error.to_string())?;
    let plan = build_qwen35_moe_mxfp4_gguf_plan(&artifact).map_err(|error| error.to_string())?;
    finish_conversion(
        arguments,
        "qwen35moe",
        format!("qwen35moe:{QWEN35_MOE_MODEL_FINGERPRINT}"),
        vec![QWEN35_MOE_MODEL_FINGERPRINT.to_owned()],
        "mixed BF16/MXFP4 lossless standard-block repack",
        &plan,
        |output| write_qwen35_moe_mxfp4_gguf(&artifact, output).map_err(|e| e.to_string()),
    )
}

fn finish_conversion<F>(
    arguments: &Arguments,
    architecture: &str,
    semantic_model_id: String,
    source_fingerprints: Vec<String>,
    tensor_mode: &str,
    plan: &GgufWritePlan,
    write: F,
) -> Result<String, String>
where
    F: FnOnce(&PathBuf) -> Result<GgufWriteReport, String>,
{
    if arguments.dry_run {
        let payload_bytes = plan
            .tensors
            .iter()
            .try_fold(0_u64, |sum, tensor| {
                tensor.byte_length().and_then(|length| {
                    sum.checked_add(length).ok_or_else(|| {
                        sllm_core::GgufError::Invalid("GGUF payload size overflows".to_owned())
                    })
                })
            })
            .map_err(|error| error.to_string())?;
        return serde_json::to_string(&serde_json::json!({
            "result": "PASS",
            "mode": "dry-run",
            "architecture": architecture,
            "source_fingerprints": source_fingerprints,
            "metadata_count": plan.metadata.len(),
            "tensor_count": plan.tensors.len(),
            "payload_bytes": payload_bytes,
        }))
        .map_err(|error| error.to_string());
    }
    let output = required_path(&arguments.output, "--output")?;
    let derived_lock_path = required_path(&arguments.derived_lock, "--derived-lock")?;
    let converter_commit = arguments
        .converter_commit
        .clone()
        .ok_or_else(|| "--converter-commit is required unless --dry-run is used".to_owned())?;
    let report = write(output)?;
    let derived = build_derived_lock(
        arguments,
        semantic_model_id,
        source_fingerprints.clone(),
        architecture,
        tensor_mode,
        converter_commit,
        &report,
    )
    .map_err(|error| error.to_string())?;
    write_new_file(
        derived_lock_path,
        &derived
            .canonical_json()
            .map_err(|error| error.to_string())?,
    )?;
    serde_json::to_string(&serde_json::json!({
        "result": "PASS",
        "mode": "convert",
        "source_fingerprints": source_fingerprints,
        "derived_lock_fingerprint": derived.fingerprint,
        "output": report.output_path,
        "size_bytes": report.size_bytes,
        "sha256": report.sha256,
        "metadata_sha256": report.metadata_sha256,
        "tensor_catalog_sha256": report.tensor_catalog_sha256,
        "tensor_count": report.tensor_count,
    }))
    .map_err(|error| error.to_string())
}

fn build_derived_lock(
    arguments: &Arguments,
    semantic_model_id: String,
    source_fingerprints: Vec<String>,
    architecture: &str,
    tensor_mode: &str,
    converter_commit: String,
    report: &sllm_core::GgufWriteReport,
) -> Result<DerivedGgufLock, sllm_core::GgufError> {
    DerivedGgufLock::new(
        semantic_model_id,
        source_fingerprints,
        DerivedGgufConverter {
            repository: REPOSITORY.to_owned(),
            commit: converter_commit,
            arguments: std::iter::once("sllm-convert-gguf".to_owned())
                .chain(arguments.raw.iter().cloned())
                .collect(),
            effective_config: BTreeMap::from([
                ("architecture".to_owned(), architecture.to_owned()),
                ("format".to_owned(), "GGUF v3 little-endian".to_owned()),
                ("alignment".to_owned(), "32".to_owned()),
                ("tensor_mode".to_owned(), tensor_mode.to_owned()),
            ]),
            environment: BTreeMap::from([
                ("os".to_owned(), env::consts::OS.to_owned()),
                ("arch".to_owned(), env::consts::ARCH.to_owned()),
                (
                    "sllm_version".to_owned(),
                    env!("CARGO_PKG_VERSION").to_owned(),
                ),
            ]),
        },
        report,
    )
}

fn parse_arguments(raw: Vec<String>) -> Result<Arguments, String> {
    let mut kind = None;
    let mut lock = None;
    let mut cache = None;
    let mut manifest = None;
    let mut artifact = None;
    let mut output = None;
    let mut derived_lock = None;
    let mut converter_commit = None;
    let mut dry_run = false;
    let mut index = 0;
    while index < raw.len() {
        let flag = raw[index].as_str();
        if flag == "--dry-run" {
            if dry_run {
                return Err("duplicate --dry-run".to_owned());
            }
            dry_run = true;
            index += 1;
            continue;
        }
        let target = match flag {
            "--kind" => &mut kind,
            "--lock" => &mut lock,
            "--cache" => &mut cache,
            "--manifest" => &mut manifest,
            "--artifact" => &mut artifact,
            "--output" => &mut output,
            "--derived-lock" => &mut derived_lock,
            "--converter-commit" => &mut converter_commit,
            _ => return Err(format!("unknown argument `{flag}`")),
        };
        if target.is_some() {
            return Err(format!("duplicate argument `{flag}`"));
        }
        index += 1;
        let value = raw
            .get(index)
            .ok_or_else(|| format!("missing value for `{flag}`"))?;
        if value.is_empty() || value.starts_with("--") {
            return Err(format!("invalid value for `{flag}`"));
        }
        *target = Some(value.clone());
        index += 1;
    }
    Ok(Arguments {
        kind: kind.unwrap_or_else(|| "qwen35-bf16".to_owned()),
        lock: lock.map(PathBuf::from),
        cache: cache
            .map(PathBuf::from)
            .ok_or_else(|| "--cache is required".to_owned())?,
        manifest: manifest.map(PathBuf::from),
        artifact: artifact.map(PathBuf::from),
        output: output.map(PathBuf::from),
        derived_lock: derived_lock.map(PathBuf::from),
        converter_commit,
        dry_run,
        raw,
    })
}

fn required_path<'a>(path: &'a Option<PathBuf>, flag: &str) -> Result<&'a PathBuf, String> {
    path.as_ref().ok_or_else(|| format!("{flag} is required"))
}

fn write_new_file(path: &PathBuf, bytes: &[u8]) -> Result<(), String> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("{}: {error}", path.display()))?;
    file.write_all(bytes)
        .and_then(|_| file.sync_all())
        .map_err(|error| format!("{}: {error}", path.display()))
}

fn help() -> String {
    "Usage: sllm-convert-gguf --kind qwen35-bf16 --lock PATH --cache PATH --dry-run\n       sllm-convert-gguf --kind qwen35-fp8 --lock PATH --cache PATH --manifest PATH --artifact PATH --dry-run\n       sllm-convert-gguf --kind gemma4-nvfp4 --lock PATH --cache PATH --dry-run\n       sllm-convert-gguf --kind qwen35moe-mxfp4 --cache PATH --dry-run\n       Replace --dry-run with --output PATH --derived-lock PATH --converter-commit SHA40 to write.".to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arguments_are_closed_and_dry_run_has_no_output_requirement() {
        let parsed = parse_arguments(vec![
            "--lock".to_owned(),
            "lock.json".to_owned(),
            "--cache".to_owned(),
            "cache".to_owned(),
            "--dry-run".to_owned(),
        ])
        .expect("valid arguments");
        assert!(parsed.dry_run);
        assert_eq!(parsed.kind, "qwen35-bf16");
        assert!(parse_arguments(vec!["--unknown".to_owned()]).is_err());
        assert!(parse_arguments(vec!["--cache".to_owned()]).is_err());
    }
}
