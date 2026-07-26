// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Fail-closed resident serving backend for the inspected Gemma4 E2B text decoder.
//!
//! The model package is intentionally not interpreted as a generic BF16 checkpoint.  A
//! package must state the exact Gemma4 E2B architecture contract and bind both source files by
//! byte count and SHA-256 before `Gemma4TextExecutor` is allowed to open it.  The executor then
//! performs its own config and tensor topology validation.  This double boundary prevents an
//! arbitrary architecture that happens to be BF16 from falling through to Gemma execution.

use crate::gemma4_text_executor::Gemma4TextExecutor;
use crate::inference_api::InferenceRequest;
use crate::served_model::{ServedModel, ServedModelError};
use crate::worker_protocol::{ReleaseOutcomeEvent, WorkerAdmission, WorkerTimings};
use crate::worker_runtime::{InferenceBackend, RequestEventPublisher};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::fs::{File, Metadata, OpenOptions};
use std::io::Read;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::time::Instant;

pub const GEMMA4_E2B_FORMAT_ID: &str = "BF16_0";
pub const GEMMA4_E2B_IMPLEMENTATION_ID: &str = "gemma4_e2b_bf16_rdna4_v1";
pub const GEMMA4_E2B_EXECUTION_PROFILE: &str = "rdna4_gemma4_e2b_bf16_resident";
pub const GEMMA4_E2B_VOCAB_SIZE: usize = 262_144;
pub const GEMMA4_E2B_CONTEXT_LENGTH: usize = 4_096;
pub const GEMMA4_E2B_MAX_NEW_TOKENS: usize = 512;
pub const GEMMA4_E2B_EOS_TOKEN_IDS: &[usize] = &[1];
pub const GEMMA4_E2B_REQUIRED_HIP_KERNEL_ENV: &[&str] = &[
    "ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL",
    "ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL",
    "ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL",
];

const PACKAGE_SCHEMA_VERSION: &str = "ullm.gemma4_e2b_bf16_package.v1";
const PACKAGE_MANIFEST_MAX_BYTES: u64 = 1024 * 1024;
const PACKAGE_MODEL_ROOT: &str = "model";
const PACKAGE_CONFIG_FILE: &str = "config.json";
const PACKAGE_MODEL_FILE: &str = "model.safetensors";

/// Validates the complete served-model contract before the Gemma worker accepts it.
///
/// This is deliberately more specific than `ServedModel::worker_startup`: the latter validates
/// a generic manifest snapshot, while this admission point protects the implementation boundary.
pub fn validate_gemma4_e2b_served_model(model: &ServedModel) -> Result<(), ServedModelError> {
    if model.format.format_id != GEMMA4_E2B_FORMAT_ID
        || model.format.implementation_id != GEMMA4_E2B_IMPLEMENTATION_ID
        || model.public.upstream_id != "google/gemma-4-E2B"
        || model.public.context_length != GEMMA4_E2B_CONTEXT_LENGTH
        || model.generation.max_completion_tokens != GEMMA4_E2B_MAX_NEW_TOKENS
        || model.generation.vocab_size != GEMMA4_E2B_VOCAB_SIZE
        || model.generation.eos_token_ids != GEMMA4_E2B_EOS_TOKEN_IDS
        || model.generation.sampling.temperature
        || model.generation.sampling.top_p
        || model.generation.sampling.top_k != 1
        || model.worker.protocol != "ullm.worker.v1"
        || model.worker.identity.device != "gfx1201"
        || model.worker.identity.execution_profile != GEMMA4_E2B_EXECUTION_PROFILE
        || model.worker.execution.is_some()
        || model.product.artifact.is_some()
        || model.reasoning.is_some()
        || model.tokenizer.class_name != "GemmaTokenizer"
        || model.tokenizer.transformers_version != "5.12.1"
        || !model.tokenizer.add_generation_prompt
        || model.tokenizer.enable_thinking
    {
        return Err(ServedModelError(
            "Gemma4 E2B served-model format, tokenizer, worker identity, or greedy generation contract is unsupported".into(),
        ));
    }
    validate_exact_required_environment(&model.worker.required_environment)
        .map_err(ServedModelError)
}

fn validate_exact_required_environment(actual: &[String]) -> Result<(), String> {
    let mut actual = actual.iter().map(String::as_str).collect::<Vec<_>>();
    actual.sort_unstable();
    let mut expected = GEMMA4_E2B_REQUIRED_HIP_KERNEL_ENV.to_vec();
    expected.sort_unstable();
    if actual != expected {
        return Err(
            "Gemma4 E2B worker required_environment does not exactly match its resident HIP guard contract"
                .into(),
        );
    }
    Ok(())
}

pub struct Gemma4E2bWorkerBackend {
    executor: Gemma4TextExecutor,
    context_length: usize,
    max_new_tokens: usize,
    vocab_size: usize,
    eos_token_ids: Vec<usize>,
}

impl Gemma4E2bWorkerBackend {
    pub fn load(
        package_dir: impl AsRef<Path>,
        context_length: usize,
        max_new_tokens: usize,
        vocab_size: usize,
        eos_token_ids: Vec<usize>,
    ) -> Result<Self, String> {
        if context_length != GEMMA4_E2B_CONTEXT_LENGTH
            || max_new_tokens != GEMMA4_E2B_MAX_NEW_TOKENS
            || vocab_size != GEMMA4_E2B_VOCAB_SIZE
            || eos_token_ids != GEMMA4_E2B_EOS_TOKEN_IDS
        {
            return Err(
                "Gemma4 E2B worker profile differs from the admitted serving contract".into(),
            );
        }
        let model_dir = verify_package(package_dir.as_ref())?;
        let executor = Gemma4TextExecutor::load_resident(&model_dir)?;
        let config = executor.config();
        if config.decoder.vocab_size != vocab_size
            || config.max_position_embeddings < context_length
            || config.decoder.model_type != "gemma4_text"
        {
            return Err("Gemma4 E2B executor config differs from the worker profile".into());
        }
        Ok(Self {
            executor,
            context_length,
            max_new_tokens,
            vocab_size,
            eos_token_ids,
        })
    }

    fn reset_after_request(&mut self) {
        self.executor.reset();
    }

    fn publish_prefill_progress(
        publications: &mut RequestEventPublisher<'_>,
        prompt_tokens: usize,
    ) -> Result<(), String> {
        let mut processed = 0_usize;
        while processed < prompt_tokens {
            let width = (prompt_tokens - processed).min(128);
            processed += width;
            publications.observe_prompt_unit(processed, width)?;
        }
        publications.observe_prefill_transition()
    }
}

impl InferenceBackend for Gemma4E2bWorkerBackend {
    fn execute(
        &mut self,
        request: InferenceRequest,
        admission: WorkerAdmission,
        publications: &mut RequestEventPublisher<'_>,
    ) -> Result<(), String> {
        request
            .validate_for_worker(
                self.context_length,
                self.max_new_tokens,
                self.vocab_size,
                &self.eos_token_ids,
                1,
            )
            .map_err(|error| error.to_string())?;
        if request.reasoning.is_some() {
            return Err("Gemma4 E2B worker.v1 does not accept reasoning requests".into());
        }

        // A prior fatal request must never leave a logical cache prefix for the next admission.
        self.reset_after_request();
        publications.publish_started()?;
        if admission.cancel.is_cancelled() {
            self.reset_after_request();
            return publications.publish_released(ReleaseOutcomeEvent::Cancelled);
        }

        let prompt = request
            .prompt_token_ids
            .iter()
            .map(|&token| u32::try_from(token).map_err(|_| "Gemma4 token ID exceeds u32"))
            .collect::<Result<Vec<_>, _>>()?;
        let prefill_started = Instant::now();
        let mut next = match self.executor.prefill(&prompt) {
            Ok(trace) => trace,
            Err(error) => {
                self.reset_after_request();
                return Err(error);
            }
        };
        let prefill_ms = prefill_started.elapsed().as_secs_f64() * 1000.0;
        if admission.cancel.is_cancelled() {
            self.reset_after_request();
            return publications.publish_released(ReleaseOutcomeEvent::Cancelled);
        }
        Self::publish_prefill_progress(publications, request.prompt_token_ids.len())?;

        let decode_started = Instant::now();
        for completion_index in 0..request.max_new_tokens {
            if admission.cancel.is_cancelled() {
                self.reset_after_request();
                return publications.publish_released(ReleaseOutcomeEvent::Cancelled);
            }
            let token = usize::try_from(next.top1.token_id)
                .map_err(|_| "Gemma4 top-1 token ID exceeds usize")?;
            publications.publish_token(token)?;
            if self.eos_token_ids.contains(&token) {
                let timings = worker_timings(
                    request.prompt_token_ids.len(),
                    prefill_ms,
                    completion_index + 1,
                    decode_started.elapsed().as_secs_f64() * 1000.0,
                    self.context_length,
                    self.max_new_tokens,
                )?;
                self.reset_after_request();
                return publications
                    .publish_released_with_timings(ReleaseOutcomeEvent::Stop, timings);
            }
            if completion_index + 1 == request.max_new_tokens {
                let timings = worker_timings(
                    request.prompt_token_ids.len(),
                    prefill_ms,
                    completion_index + 1,
                    decode_started.elapsed().as_secs_f64() * 1000.0,
                    self.context_length,
                    self.max_new_tokens,
                )?;
                self.reset_after_request();
                return publications
                    .publish_released_with_timings(ReleaseOutcomeEvent::Length, timings);
            }
            next = match self.executor.decode(token as u32) {
                Ok(trace) => trace,
                Err(error) => {
                    self.reset_after_request();
                    return Err(error);
                }
            };
        }
        Err("Gemma4 completion loop ended without a terminal event".into())
    }

    fn shutdown(&mut self) -> Result<(), String> {
        self.reset_after_request();
        Ok(())
    }
}

fn worker_timings(
    prompt_tokens: usize,
    prefill_ms: f64,
    completion_tokens: usize,
    decode_ms: f64,
    context_length: usize,
    max_new_tokens: usize,
) -> Result<WorkerTimings, String> {
    WorkerTimings::from_elapsed_millis_with_limits(
        prompt_tokens,
        prefill_ms.max(0.001),
        completion_tokens,
        decode_ms.max(0.001),
        context_length,
        max_new_tokens,
    )
    .ok_or_else(|| "Gemma4 worker timings violate the request bounds".to_string())
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPackageManifest {
    schema_version: String,
    format_id: String,
    implementation_id: String,
    architecture: RawArchitecture,
    model: RawModel,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawArchitecture {
    architectures: Vec<String>,
    model_type: String,
    text_model_type: String,
    vocab_size: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawModel {
    root: String,
    files: RawModelFiles,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawModelFiles {
    #[serde(rename = "config.json")]
    config_json: RawBoundFile,
    #[serde(rename = "model.safetensors")]
    model_safetensors: RawBoundFile,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawBoundFile {
    sha256: String,
    bytes: u64,
}

fn verify_package(package_dir: &Path) -> Result<PathBuf, String> {
    require_safe_directory(package_dir, "Gemma4 package directory")?;
    let package_manifest = package_dir.join("manifest.json");
    let raw = read_stable_file(
        &package_manifest,
        PACKAGE_MANIFEST_MAX_BYTES,
        "Gemma4 package manifest",
    )?;
    let manifest: RawPackageManifest = serde_json::from_slice(&raw)
        .map_err(|_| "Gemma4 package manifest schema is invalid".to_string())?;
    validate_package_manifest(&manifest)?;

    let model_dir = package_dir.join(PACKAGE_MODEL_ROOT);
    require_safe_directory(&model_dir, "Gemma4 package model directory")?;
    verify_bound_file(
        &model_dir.join(PACKAGE_CONFIG_FILE),
        &manifest.model.files.config_json,
        "Gemma4 package config",
    )?;
    verify_bound_file(
        &model_dir.join(PACKAGE_MODEL_FILE),
        &manifest.model.files.model_safetensors,
        "Gemma4 package model",
    )?;
    Ok(model_dir)
}

fn validate_package_manifest(manifest: &RawPackageManifest) -> Result<(), String> {
    if manifest.schema_version != PACKAGE_SCHEMA_VERSION
        || manifest.format_id != GEMMA4_E2B_FORMAT_ID
        || manifest.implementation_id != GEMMA4_E2B_IMPLEMENTATION_ID
        || manifest.architecture.architectures != vec!["Gemma4ForConditionalGeneration".to_string()]
        || manifest.architecture.model_type != "gemma4"
        || manifest.architecture.text_model_type != "gemma4_text"
        || manifest.architecture.vocab_size != GEMMA4_E2B_VOCAB_SIZE
        || manifest.model.root != PACKAGE_MODEL_ROOT
    {
        return Err("Gemma4 package architecture contract is unsupported".into());
    }
    validate_bound_file(&manifest.model.files.config_json, "Gemma4 package config")?;
    validate_bound_file(
        &manifest.model.files.model_safetensors,
        "Gemma4 package model",
    )
}

fn validate_bound_file(value: &RawBoundFile, label: &str) -> Result<(), String> {
    if value.bytes == 0 || !is_lowercase_sha256(&value.sha256) {
        return Err(format!("{label} binding is invalid"));
    }
    Ok(())
}

fn verify_bound_file(path: &Path, expected: &RawBoundFile, label: &str) -> Result<(), String> {
    validate_bound_file(expected, label)?;
    let (mut file, identity) = open_stable_regular_file(path, label)?;
    if identity.size != expected.bytes {
        return Err(format!("{label} byte count differs"));
    }
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|_| format!("{label} read failed"))?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    if stable_file_identity(path, &file, identity, label)? != identity {
        return Err(format!("{label} changed while being read"));
    }
    if format!("{:x}", digest.finalize()) != expected.sha256 {
        return Err(format!("{label} SHA-256 differs"));
    }
    Ok(())
}

fn read_stable_file(path: &Path, maximum: u64, label: &str) -> Result<Vec<u8>, String> {
    let (mut file, identity) = open_stable_regular_file(path, label)?;
    if identity.size == 0 || identity.size > maximum {
        return Err(format!("{label} exceeds its size bound"));
    }
    let capacity = usize::try_from(identity.size).map_err(|_| format!("{label} is too large"))?;
    let mut raw = Vec::with_capacity(capacity);
    file.read_to_end(&mut raw)
        .map_err(|_| format!("{label} read failed"))?;
    if raw.len() != capacity || stable_file_identity(path, &file, identity, label)? != identity {
        return Err(format!("{label} changed while being read"));
    }
    Ok(raw)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FileIdentity {
    device: u64,
    inode: u64,
    mode: u32,
    links: u64,
    size: u64,
    mtime_seconds: i64,
    mtime_nanoseconds: i64,
    ctime_seconds: i64,
    ctime_nanoseconds: i64,
}

impl From<&Metadata> for FileIdentity {
    fn from(value: &Metadata) -> Self {
        Self {
            device: value.dev(),
            inode: value.ino(),
            mode: value.mode(),
            links: value.nlink(),
            size: value.size(),
            mtime_seconds: value.mtime(),
            mtime_nanoseconds: value.mtime_nsec(),
            ctime_seconds: value.ctime(),
            ctime_nanoseconds: value.ctime_nsec(),
        }
    }
}

fn require_safe_directory(path: &Path, label: &str) -> Result<(), String> {
    reject_symlink_components(path, label)?;
    let metadata =
        std::fs::symlink_metadata(path).map_err(|_| format!("{label} metadata failed"))?;
    if !metadata.file_type().is_dir() || metadata.permissions().mode() & 0o022 != 0 {
        return Err(format!("{label} is not a sealed directory"));
    }
    Ok(())
}

fn open_stable_regular_file(path: &Path, label: &str) -> Result<(File, FileIdentity), String> {
    reject_symlink_components(path, label)?;
    let before = std::fs::symlink_metadata(path).map_err(|_| format!("{label} metadata failed"))?;
    let identity = FileIdentity::from(&before);
    if !before.file_type().is_file()
        || before.nlink() != 1
        || before.permissions().mode() & 0o022 != 0
    {
        return Err(format!("{label} is not a sealed regular file"));
    }
    const O_NOFOLLOW: i32 = 0o400000;
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(O_NOFOLLOW)
        .open(path)
        .map_err(|_| format!("{label} open failed"))?;
    if stable_file_identity(path, &file, identity, label)? != identity {
        return Err(format!("{label} changed while opening"));
    }
    Ok((file, identity))
}

fn stable_file_identity(
    path: &Path,
    file: &File,
    expected: FileIdentity,
    label: &str,
) -> Result<FileIdentity, String> {
    reject_symlink_components(path, label)?;
    let fd = file
        .metadata()
        .map_err(|_| format!("{label} metadata failed"))?;
    let named = std::fs::symlink_metadata(path).map_err(|_| format!("{label} metadata failed"))?;
    let fd_identity = FileIdentity::from(&fd);
    let named_identity = FileIdentity::from(&named);
    if !fd.file_type().is_file()
        || fd.nlink() != 1
        || fd.permissions().mode() & 0o022 != 0
        || fd_identity != expected
        || named_identity != expected
    {
        return Err(format!("{label} changed while being read"));
    }
    Ok(fd_identity)
}

fn reject_symlink_components(path: &Path, label: &str) -> Result<(), String> {
    if !path.is_absolute() {
        return Err(format!("{label} path must be absolute"));
    }
    for component in path.ancestors() {
        let metadata = std::fs::symlink_metadata(component)
            .map_err(|_| format!("{label} path metadata failed"))?;
        if metadata.file_type().is_symlink() {
            return Err(format!("{label} path traverses a symlink"));
        }
    }
    Ok(())
}

fn is_lowercase_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_manifest() -> RawPackageManifest {
        serde_json::from_value(serde_json::json!({
            "schema_version": PACKAGE_SCHEMA_VERSION,
            "format_id": GEMMA4_E2B_FORMAT_ID,
            "implementation_id": GEMMA4_E2B_IMPLEMENTATION_ID,
            "architecture": {
                "architectures": ["Gemma4ForConditionalGeneration"],
                "model_type": "gemma4",
                "text_model_type": "gemma4_text",
                "vocab_size": GEMMA4_E2B_VOCAB_SIZE
            },
            "model": {
                "root": PACKAGE_MODEL_ROOT,
                "files": {
                    "config.json": {"sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "bytes": 1},
                    "model.safetensors": {"sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "bytes": 2}
                }
            }
        }))
        .unwrap()
    }

    #[test]
    fn package_manifest_requires_the_exact_gemma_architecture() {
        let manifest = valid_manifest();
        validate_package_manifest(&manifest).unwrap();

        let mut wrong_format = valid_manifest();
        wrong_format.format_id = "SQ8_0".into();
        assert!(validate_package_manifest(&wrong_format).is_err());

        let mut wrong_architecture = valid_manifest();
        wrong_architecture.architecture.architectures = vec!["Qwen3ForCausalLM".into()];
        assert!(validate_package_manifest(&wrong_architecture).is_err());

        let mut wrong_vocabulary = valid_manifest();
        wrong_vocabulary.architecture.vocab_size = GEMMA4_E2B_VOCAB_SIZE - 1;
        assert!(validate_package_manifest(&wrong_vocabulary).is_err());
    }

    #[test]
    fn required_environment_is_an_exact_set() {
        let valid = GEMMA4_E2B_REQUIRED_HIP_KERNEL_ENV
            .iter()
            .rev()
            .map(|value| (*value).to_string())
            .collect::<Vec<_>>();
        validate_exact_required_environment(&valid).unwrap();
        assert!(validate_exact_required_environment(&[]).is_err());
        let mut extra = valid;
        extra.push("ULLM_REQUIRE_HIP_UNKNOWN_KERNEL".into());
        assert!(validate_exact_required_environment(&extra).is_err());
    }

    #[test]
    fn package_manifest_rejects_duplicate_or_unknown_fields() {
        let duplicate = format!(
            r#"{{"schema_version":"{PACKAGE_SCHEMA_VERSION}","schema_version":"{PACKAGE_SCHEMA_VERSION}"}}"#
        );
        assert!(serde_json::from_str::<RawPackageManifest>(&duplicate).is_err());
        let unknown = serde_json::json!({
            "schema_version": PACKAGE_SCHEMA_VERSION,
            "format_id": GEMMA4_E2B_FORMAT_ID,
            "implementation_id": GEMMA4_E2B_IMPLEMENTATION_ID,
            "architecture": {"architectures": ["Gemma4ForConditionalGeneration"], "model_type": "gemma4", "text_model_type": "gemma4_text", "vocab_size": GEMMA4_E2B_VOCAB_SIZE},
            "model": {"root": PACKAGE_MODEL_ROOT, "files": {"config.json": {"sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "bytes": 1}, "model.safetensors": {"sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "bytes": 1}}},
            "unexpected": true
        });
        assert!(serde_json::from_value::<RawPackageManifest>(unknown).is_err());
    }
}
