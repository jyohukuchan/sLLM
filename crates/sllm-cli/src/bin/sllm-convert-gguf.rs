use sllm_core::{
    DerivedGgufConverter, DerivedGgufLock, GEMMA4_MOE_MODEL_FINGERPRINT, GgufWritePlan,
    GgufWriteReport, QWEN35_MOE_MODEL_FINGERPRINT, QwenMxWeightActivationFormat,
    UNSLOTH_GEMMA4_NVFP4_MODEL_SHA256, build_gemma4_moe_nvfp4_gguf_plan,
    build_gemma4_mtp_bf16_gguf_plan, build_gemma4_nvfp4_gguf_plan, build_qwen35_bf16_gguf_plan,
    build_qwen35_fp8_gguf_plan, build_qwen35_moe_mxfp4_gguf_plan,
    build_qwen35_mx_weight_activation_gguf_plan, gemma4_mtp_pair_semantic_id,
    parse_gemma4_mtp_model_lock, read_model_lock, read_reviewed_model_lock, verify_fp8_sidecar,
    verify_gemma4_moe_artifact, verify_qwen35_moe_artifact, verify_unsloth_gemma4_nvfp4,
    write_gemma4_moe_nvfp4_gguf, write_gemma4_mtp_bf16_gguf, write_gemma4_nvfp4_gguf,
    write_qwen35_bf16_gguf, write_qwen35_fp8_gguf, write_qwen35_moe_mxfp4_gguf,
    write_qwen35_mx_weight_activation_gguf,
};
use sllm_tools::{
    AtomicBundleV1, TOOL_JSON_CANONICALIZATION_V1, TOOL_RUN_SCHEMA_VERSION_V1,
    TOOL_RUN_STRUCT_SIZE_V1, ToolFileIdentityV1, ToolIdentityV1, ToolRecipeIdentityV1,
    ToolRunManifestV1, ToolRunStateV1, rust_toolchain_environment, sha256_bytes,
};
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
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
    output_bundle: Option<PathBuf>,
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
        "qwen35-mxfp8-w8a8" => run_qwen35_mx(&arguments, QwenMxWeightActivationFormat::Mxfp8E4m3),
        "qwen35-mxfp6-w6a6" => run_qwen35_mx(&arguments, QwenMxWeightActivationFormat::Mxfp6E3m2),
        "gemma4-nvfp4" => run_gemma4_nvfp4(&arguments),
        "gemma4moe-nvfp4" => run_gemma4_moe_nvfp4(&arguments),
        "gemma4-mtp-bf16" => run_gemma4_mtp_bf16(&arguments),
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

fn run_qwen35_mx(
    arguments: &Arguments,
    format: QwenMxWeightActivationFormat,
) -> Result<String, String> {
    let lock_path = required_path(&arguments.lock, "--lock")?;
    let lock = read_model_lock(lock_path).map_err(|error| error.to_string())?;
    let cache = lock
        .verify_cache(&arguments.cache)
        .map_err(|error| error.to_string())?;
    let plan = build_qwen35_mx_weight_activation_gguf_plan(&lock, &cache, format)
        .map_err(|error| error.to_string())?;
    finish_conversion(
        arguments,
        "qwen35",
        format!("qwen35:{}", lock.fingerprint()),
        vec![lock.fingerprint().to_owned()],
        format.tensor_mode(),
        &plan,
        |output| {
            write_qwen35_mx_weight_activation_gguf(&lock, &cache, format, output)
                .map_err(|error| error.to_string())
        },
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

fn run_gemma4_moe_nvfp4(arguments: &Arguments) -> Result<String, String> {
    let artifact =
        verify_gemma4_moe_artifact(&arguments.cache).map_err(|error| error.to_string())?;
    let plan = build_gemma4_moe_nvfp4_gguf_plan(&artifact).map_err(|error| error.to_string())?;
    finish_conversion(
        arguments,
        "gemma4moe",
        format!("gemma4moe:{GEMMA4_MOE_MODEL_FINGERPRINT}"),
        vec![GEMMA4_MOE_MODEL_FINGERPRINT.to_owned()],
        "mixed BF16/NVFP4 lossless standard-block repack with implicit-unit static FP8 KV",
        &plan,
        |output| write_gemma4_moe_nvfp4_gguf(&artifact, output).map_err(|e| e.to_string()),
    )
}

fn run_gemma4_mtp_bf16(arguments: &Arguments) -> Result<String, String> {
    let lock_path = required_path(&arguments.lock, "--lock")?;
    let target = match read_reviewed_model_lock(lock_path).map_err(|error| error.to_string())? {
        sllm_core::ReviewedModelLock::Gemma4(lock) => lock,
        _ => return Err("--lock is not a reviewed Gemma 4 target lock".to_owned()),
    };
    let assistant_lock = parse_gemma4_mtp_model_lock(include_bytes!(
        "../../../../docs/models/locks/gemma4-12b-it-assistant-bf16.json"
    ))
    .map_err(|error| error.to_string())?;
    let assistant = assistant_lock
        .verify_cache(&arguments.cache, &target)
        .map_err(|error| error.to_string())?;
    let plan = build_gemma4_mtp_bf16_gguf_plan(&assistant_lock, &target, &assistant)
        .map_err(|error| error.to_string())?;
    let semantic_model_id =
        gemma4_mtp_pair_semantic_id(target.fingerprint(), assistant_lock.fingerprint());
    finish_conversion(
        arguments,
        "gemma4mtp",
        semantic_model_id,
        vec![
            target.fingerprint().to_owned(),
            assistant_lock.fingerprint().to_owned(),
        ],
        "BF16 lossless Gemma 4 MTP assistant (target-paired)",
        &plan,
        |output| {
            write_gemma4_mtp_bf16_gguf(&assistant_lock, &target, &assistant, output)
                .map_err(|error| error.to_string())
        },
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
    let converter_commit = arguments
        .converter_commit
        .clone()
        .ok_or_else(|| "--converter-commit is required unless --dry-run is used".to_owned())?;
    if arguments.output_bundle.is_some()
        && (arguments.output.is_some() || arguments.derived_lock.is_some())
    {
        return Err(
            "--output-bundle cannot be combined with --output or --derived-lock".to_owned(),
        );
    }
    if let Some(output_bundle) = &arguments.output_bundle {
        let bundle = AtomicBundleV1::create(output_bundle).map_err(|error| error.to_string())?;
        let staged_output = bundle
            .path("model.gguf")
            .map_err(|error| error.to_string())?;
        let mut report = write(&staged_output)?;
        report.output_path = output_bundle.join("model.gguf");
        let derived = build_derived_lock(
            arguments,
            semantic_model_id,
            source_fingerprints.clone(),
            architecture,
            tensor_mode,
            converter_commit.clone(),
            &report,
        )
        .map_err(|error| error.to_string())?;
        let derived_bytes = derived
            .canonical_json()
            .map_err(|error| error.to_string())?;
        let staged_lock = bundle
            .write_bytes("model.derived-lock.json", &derived_bytes)
            .map_err(|error| error.to_string())?;
        let manifest = conversion_manifest(
            arguments,
            converter_commit,
            architecture,
            tensor_mode,
            &staged_output,
            &staged_lock,
            plan.tensors.len(),
        )?;
        bundle
            .write_json("run-manifest.json", &manifest)
            .map_err(|error| error.to_string())?;
        let published = bundle.commit().map_err(|error| error.to_string())?;
        return serde_json::to_string(&serde_json::json!({
            "result": "PASS", "mode": "convert-bundle", "bundle": published,
            "source_fingerprints": source_fingerprints,
            "derived_lock_fingerprint": derived.fingerprint,
            "output": report.output_path, "size_bytes": report.size_bytes,
            "sha256": report.sha256, "metadata_sha256": report.metadata_sha256,
            "tensor_catalog_sha256": report.tensor_catalog_sha256,
            "tensor_count": report.tensor_count,
        }))
        .map_err(|error| error.to_string());
    }
    Err(
        "non-dry conversions require --output-bundle; the legacy --output/--derived-lock pair cannot publish GGUF and lock as one atomic transaction"
            .to_owned(),
    )
}

fn conversion_manifest(
    arguments: &Arguments,
    converter_commit: String,
    architecture: &str,
    tensor_mode: &str,
    staged_output: &PathBuf,
    staged_lock: &PathBuf,
    selected_count: usize,
) -> Result<ToolRunManifestV1, String> {
    let mut sources = Vec::new();
    for (role, logical_name, path) in [
        ("model-lock", "model-lock.json", arguments.lock.as_ref()),
        (
            "sidecar-manifest",
            "sidecar-manifest.json",
            arguments.manifest.as_ref(),
        ),
        (
            "sidecar-artifact",
            "sidecar-artifact.bin",
            arguments.artifact.as_ref(),
        ),
    ] {
        if let Some(path) = path {
            sources.push(
                ToolFileIdentityV1::from_path(role, logical_name, path)
                    .map_err(|error| error.to_string())?,
            );
        }
    }
    sources.extend(verified_cache_identities(&arguments.cache)?);
    let recipe_bytes = serde_json::to_vec(&BTreeMap::from([
        ("architecture", architecture),
        ("tensor_mode", tensor_mode),
    ]))
    .map_err(|error| error.to_string())?;
    let executable =
        env::current_exe().map_err(|error| format!("resolve converter executable: {error}"))?;
    let executable_sha256 =
        ToolFileIdentityV1::from_path("tool-binary", "sllm-convert-gguf", executable)
            .map_err(|error| error.to_string())?
            .sha256;
    let manifest = ToolRunManifestV1 {
        schema_version: TOOL_RUN_SCHEMA_VERSION_V1.to_owned(),
        struct_size: TOOL_RUN_STRUCT_SIZE_V1,
        canonicalization: TOOL_JSON_CANONICALIZATION_V1.to_owned(),
        operation: "hf-to-gguf".to_owned(),
        state: ToolRunStateV1::Pass,
        selected_count: u64::try_from(selected_count)
            .map_err(|_| "tensor count overflow".to_owned())?,
        tool: ToolIdentityV1 {
            repository: REPOSITORY.to_owned(),
            commit: converter_commit,
            package: "sllm-cli".to_owned(),
            version: env!("CARGO_PKG_VERSION").to_owned(),
            executable_sha256,
            arguments: std::iter::once("sllm-convert-gguf".to_owned())
                .chain(arguments.raw.iter().cloned())
                .collect(),
            environment: rust_toolchain_environment(),
        },
        recipe: ToolRecipeIdentityV1 {
            id: "hf-to-gguf".to_owned(),
            version: "v1".to_owned(),
            config_sha256: sha256_bytes(&recipe_bytes),
        },
        sources,
        outputs: vec![
            ToolFileIdentityV1::from_path("gguf", "model.gguf", staged_output)
                .map_err(|error| error.to_string())?,
            ToolFileIdentityV1::from_path("derived-lock", "model.derived-lock.json", staged_lock)
                .map_err(|error| error.to_string())?,
        ],
        raw_evidence: Vec::new(),
        identities: BTreeMap::from([
            ("architecture".to_owned(), architecture.to_owned()),
            ("tensor-mode".to_owned(), tensor_mode.to_owned()),
        ]),
        metrics: BTreeMap::from([("tensor-count".to_owned(), serde_json::json!(selected_count))]),
        extensions: BTreeMap::new(),
    };
    manifest.validate().map_err(|error| error.to_string())?;
    Ok(manifest)
}

fn verified_cache_identities(root: &Path) -> Result<Vec<ToolFileIdentityV1>, String> {
    let metadata = fs::symlink_metadata(root)
        .map_err(|error| format!("stat verified cache {}: {error}", root.display()))?;
    if metadata.file_type().is_symlink() {
        return Err(format!(
            "verified cache root must not be a symlink: {}",
            root.display()
        ));
    }
    if metadata.is_file() {
        return ToolFileIdentityV1::from_path("verified-cache-file", "cache/source", root)
            .map(|identity| vec![identity])
            .map_err(|error| error.to_string());
    }
    if !metadata.is_dir() {
        return Err(format!(
            "verified cache root is not a regular file or directory: {}",
            root.display()
        ));
    }

    let mut paths = Vec::new();
    collect_regular_files(root, root, &mut paths)?;
    paths.sort_by(|left, right| left.0.cmp(&right.0));
    if paths.is_empty() {
        return Err(format!("verified cache has no files: {}", root.display()));
    }
    paths
        .into_iter()
        .map(|(logical_name, path)| {
            ToolFileIdentityV1::from_path("verified-cache-file", logical_name, path)
                .map_err(|error| error.to_string())
        })
        .collect()
}

fn collect_regular_files(
    root: &Path,
    directory: &Path,
    paths: &mut Vec<(String, PathBuf)>,
) -> Result<(), String> {
    let entries = fs::read_dir(directory)
        .map_err(|error| format!("read verified cache {}: {error}", directory.display()))?;
    for entry in entries {
        let entry = entry.map_err(|error| {
            format!("read verified cache entry {}: {error}", directory.display())
        })?;
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path)
            .map_err(|error| format!("stat verified cache entry {}: {error}", path.display()))?;
        if metadata.file_type().is_symlink() {
            let resolved = resolve_hugging_face_blob_symlink(root, &path)?;
            let relative = path
                .strip_prefix(root)
                .map_err(|_| format!("verified cache entry escaped root: {}", path.display()))?;
            let logical_name = format!("cache/{}", relative.to_string_lossy().replace('\\', "/"));
            paths.push((logical_name, resolved));
            continue;
        }
        if metadata.is_dir() {
            collect_regular_files(root, &path, paths)?;
        } else if metadata.is_file() {
            let relative = path
                .strip_prefix(root)
                .map_err(|_| format!("verified cache entry escaped root: {}", path.display()))?;
            let logical_name = format!("cache/{}", relative.to_string_lossy().replace('\\', "/"));
            paths.push((logical_name, path));
        } else {
            return Err(format!(
                "verified cache entry is not a regular file or directory: {}",
                path.display()
            ));
        }
    }
    Ok(())
}

fn resolve_hugging_face_blob_symlink(root: &Path, path: &Path) -> Result<PathBuf, String> {
    let snapshots = root.parent().ok_or_else(|| {
        format!(
            "verified cache symlink is not in a Hugging Face snapshot: {}",
            path.display()
        )
    })?;
    if snapshots.file_name().and_then(|name| name.to_str()) != Some("snapshots") {
        return Err(format!(
            "verified cache symlink is not in a Hugging Face snapshot: {}",
            path.display()
        ));
    }
    let repository = snapshots.parent().ok_or_else(|| {
        format!(
            "verified cache symlink has no Hugging Face repository root: {}",
            path.display()
        )
    })?;
    let blob_root = fs::canonicalize(repository.join("blobs")).map_err(|error| {
        format!(
            "resolve Hugging Face blob root for {}: {error}",
            path.display()
        )
    })?;
    let resolved = fs::canonicalize(path).map_err(|error| {
        format!(
            "resolve verified Hugging Face cache entry {}: {error}",
            path.display()
        )
    })?;
    let resolved_metadata = fs::symlink_metadata(&resolved).map_err(|error| {
        format!(
            "stat resolved Hugging Face cache entry {}: {error}",
            resolved.display()
        )
    })?;
    if !resolved_metadata.is_file()
        || resolved_metadata.file_type().is_symlink()
        || resolved.parent() != Some(blob_root.as_path())
    {
        return Err(format!(
            "verified cache symlink does not resolve to one repository blob: {}",
            path.display()
        ));
    }
    Ok(resolved)
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
    let mut output_bundle = None;
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
            "--output-bundle" => &mut output_bundle,
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
        output_bundle: output_bundle.map(PathBuf::from),
        converter_commit,
        dry_run,
        raw,
    })
}

fn required_path<'a>(path: &'a Option<PathBuf>, flag: &str) -> Result<&'a PathBuf, String> {
    path.as_ref().ok_or_else(|| format!("{flag} is required"))
}

fn help() -> String {
    "Usage: sllm-convert-gguf --kind qwen35-bf16 --lock PATH --cache PATH --dry-run\n       sllm-convert-gguf --kind qwen35-fp8 --lock PATH --cache PATH --manifest PATH --artifact PATH --dry-run\n       sllm-convert-gguf --kind qwen35-mxfp8-w8a8 --lock PATH --cache PATH --dry-run\n       sllm-convert-gguf --kind qwen35-mxfp6-w6a6 --lock PATH --cache PATH --dry-run\n       sllm-convert-gguf --kind gemma4-nvfp4 --lock PATH --cache PATH --dry-run\n       sllm-convert-gguf --kind gemma4moe-nvfp4 --cache PATH --dry-run\n       sllm-convert-gguf --kind gemma4-mtp-bf16 --lock TARGET_LOCK_PATH --cache ASSISTANT_CACHE_PATH --dry-run\n       sllm-convert-gguf --kind qwen35moe-mxfp4 --cache PATH --dry-run\n       Replace --dry-run with --output-bundle DIR --converter-commit SHA40 for atomic GGUF/lock/manifest publication. Legacy --output/--derived-lock publication is rejected.".to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(1);

    fn temp_dir(label: &str) -> PathBuf {
        env::temp_dir().join(format!(
            "sllm-converter-{label}-{}-{}",
            std::process::id(),
            NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ))
    }

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

    #[test]
    fn gemma4_moe_kind_is_explicit_and_needs_no_dense_lock() {
        let parsed = parse_arguments(vec![
            "--kind".to_owned(),
            "gemma4moe-nvfp4".to_owned(),
            "--cache".to_owned(),
            "immutable-snapshot".to_owned(),
            "--dry-run".to_owned(),
        ])
        .expect("Gemma 4 MoE arguments");
        assert_eq!(parsed.kind, "gemma4moe-nvfp4");
        assert!(parsed.lock.is_none());
        assert!(parsed.dry_run);
        assert!(help().contains("--kind gemma4moe-nvfp4 --cache PATH --dry-run"));
    }

    #[test]
    fn gemma4_mtp_kind_requires_the_target_lock_and_is_documented() {
        let parsed = parse_arguments(vec![
            "--kind".to_owned(),
            "gemma4-mtp-bf16".to_owned(),
            "--lock".to_owned(),
            "target-lock.json".to_owned(),
            "--cache".to_owned(),
            "assistant-cache".to_owned(),
            "--dry-run".to_owned(),
        ])
        .expect("Gemma 4 MTP arguments");
        assert_eq!(parsed.kind, "gemma4-mtp-bf16");
        assert!(parsed.lock.is_some());
        assert!(help().contains("--kind gemma4-mtp-bf16 --lock TARGET_LOCK_PATH"));
    }

    #[test]
    fn gemma4_mtp_dry_run_reports_the_plan_without_writing_payload() {
        let arguments = parse_arguments(vec![
            "--kind".to_owned(),
            "gemma4-mtp-bf16".to_owned(),
            "--cache".to_owned(),
            "assistant-cache".to_owned(),
            "--dry-run".to_owned(),
        ])
        .expect("dry-run arguments");
        let plan = GgufWritePlan {
            metadata: BTreeMap::from([(
                "general.architecture".to_owned(),
                sllm_core::GgufValue::String("gemma4mtp".to_owned()),
            )]),
            tensors: vec![sllm_core::GgufWriteTensor {
                name: "model.norm.weight".to_owned(),
                source_name: "model.norm.weight".to_owned(),
                dimensions: vec![2],
                tensor_type: sllm_core::GgufTensorType::Bf16,
            }],
        };
        let report = finish_conversion(
            &arguments,
            "gemma4mtp",
            "gemma4mtp-pair:test".to_owned(),
            vec!["sha256:test".to_owned()],
            "BF16 fixture",
            &plan,
            |_output| panic!("dry-run must not write"),
        )
        .expect("dry-run report");
        assert!(report.contains("\"mode\":\"dry-run\""));
        assert!(report.contains("\"tensor_count\":1"));
    }

    #[test]
    fn verified_cache_sources_are_real_files_in_deterministic_order() {
        let root = temp_dir("sources");
        fs::create_dir_all(root.join("nested")).unwrap();
        fs::write(root.join("z.bin"), b"z").unwrap();
        fs::write(root.join("nested/a.bin"), b"actual source").unwrap();

        let identities = verified_cache_identities(&root).unwrap();
        assert_eq!(identities.len(), 2);
        assert_eq!(identities[0].logical_name, "cache/nested/a.bin");
        assert_eq!(identities[0].size_bytes, 13);
        assert_eq!(identities[0].sha256, sha256_bytes(b"actual source"));
        assert_eq!(identities[1].logical_name, "cache/z.bin");
        assert_eq!(identities[1].sha256, sha256_bytes(b"z"));

        fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn verified_cache_sources_accept_only_hugging_face_repository_blob_symlinks() {
        use std::os::unix::fs::symlink;

        let repository = temp_dir("hf-symlinks");
        let root = repository.join("snapshots/revision");
        fs::create_dir_all(repository.join("blobs")).unwrap();
        fs::create_dir_all(&root).unwrap();
        fs::write(repository.join("blobs/abcd"), b"reviewed blob").unwrap();
        symlink("../../blobs/abcd", root.join("model.bin")).unwrap();
        let identities = verified_cache_identities(&root).unwrap();
        assert_eq!(identities.len(), 1);
        assert_eq!(identities[0].logical_name, "cache/model.bin");
        assert_eq!(identities[0].sha256, sha256_bytes(b"reviewed blob"));

        fs::write(repository.join("outside"), b"escape").unwrap();
        symlink("../../outside", root.join("escape.bin")).unwrap();
        assert!(verified_cache_identities(&root).is_err());
        fs::remove_dir_all(repository).unwrap();
    }
}
