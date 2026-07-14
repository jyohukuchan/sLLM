// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Offline AQ4 full-model benchmark driver.
//!
//! The driver uses the resident AQ4 model/session path exposed by the engine. It does not speak
//! the worker protocol and never mutates a running service. A result is published only after the
//! complete request has reset successfully; the artifact contains dimensions, hashes, and
//! counters, but no prompt, token id, or generated text.

use serde::{Deserialize, Serialize};
use std::env;
use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use ullm_engine::aq4_worker_backend::QWEN35_AQ4_REQUIRED_HIP_KERNEL_ENV;
use ullm_engine::inference_api::{
    CancellationToken, GenerationTimings, InferenceRequest, ReasoningUsage, ReleaseOutcome,
    SamplingParams,
};
use ullm_engine::qwen35_aq4_head_runtime::PackageLmHeadMode;
use ullm_engine::qwen35_aq4_model_runtime::{QWEN35_AQ4_KV_BLOCK_SIZE, Qwen35Aq4ModelLoadConfig};
use ullm_engine::qwen35_aq4_session::{
    QWEN35_AQ4_PREFILL_CHUNK_GRID, Qwen35Aq4InferenceSession, Qwen35Aq4SessionConfig,
};
use ullm_engine::served_model::{ServedModel, load_served_model};
use ullm_engine::worker_driver::{InferenceSession, RequestPublications, drive_worker_request};

const MAX_FIXTURE_BYTES: usize = 1_048_576;
const MAX_RESULT_BYTES: usize = 65_536;
const RESULT_SCHEMA: &str = "ullm.qwen35_aq4_p2.full_model_result.v1";
const DEFAULT_CHUNK_BYTES: usize = 1024 * 1024;
const DEFAULT_LM_HEAD_CHUNK_ROWS: usize = 8192;

#[derive(Debug, Clone, PartialEq, Eq)]
struct Args {
    served_model_manifest: PathBuf,
    fixture: PathBuf,
    output: PathBuf,
    requested_m: usize,
    case_id: Option<String>,
    device_index: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FixtureCase {
    case_id: String,
    prompt_token_ids: Vec<usize>,
    step_count: usize,
}

#[derive(Debug, Deserialize)]
struct FixtureFile {
    cases: Vec<FixtureCaseRaw>,
}

#[derive(Debug, Deserialize)]
struct FixtureCaseRaw {
    case_id: String,
    prompt_token_ids: Vec<usize>,
    step_count: usize,
}

#[derive(Debug, Clone, Serialize)]
struct Identity {
    served_model_manifest_sha256: String,
    model_id: String,
    model_revision: String,
    format_id: String,
    implementation_id: String,
    worker_binary_sha256: String,
    artifact_content_sha256: String,
    package_manifest_sha256: String,
    device: String,
    execution_profile: String,
}

#[derive(Debug, Clone, Serialize)]
struct FallbackFacts {
    count: u64,
    unexpected_count: u64,
}

#[derive(Debug, Clone, Serialize)]
struct BenchmarkResult {
    schema_version: &'static str,
    scope: &'static str,
    status: &'static str,
    identity: Identity,
    fixture_case_id: String,
    requested_m: usize,
    resolved_m: Option<usize>,
    actual_token_batch_width: Option<usize>,
    actual_request_batch_width: Option<usize>,
    timings: Option<GenerationTimings>,
    generated_tokens: usize,
    operation_audit_digest: Option<String>,
    lifecycle: Option<serde_json::Value>,
    reset: Option<serde_json::Value>,
    outcome: &'static str,
    oom: bool,
    fallback: FallbackFacts,
}

#[derive(Debug, Default)]
struct Publications {
    completion_tokens: usize,
    timings: Option<GenerationTimings>,
    outcome: Option<ReleaseOutcome>,
    reasoning_usage: Option<ReasoningUsage>,
}

impl RequestPublications for Publications {
    fn publish_started(&mut self) -> Result<(), String> {
        Ok(())
    }

    fn observe_prompt_unit(
        &mut self,
        _prompt_tokens_processed: usize,
        _execution_width: usize,
    ) -> Result<(), String> {
        Ok(())
    }

    fn observe_prefill_transition(&mut self) -> Result<(), String> {
        Ok(())
    }

    fn publish_token(&mut self, _token_id: usize) -> Result<(), String> {
        self.completion_tokens = self
            .completion_tokens
            .checked_add(1)
            .ok_or_else(|| "completion token count overflowed".to_string())?;
        Ok(())
    }

    fn publish_released(
        &mut self,
        outcome: ReleaseOutcome,
        timings: Option<GenerationTimings>,
    ) -> Result<(), String> {
        self.outcome = Some(outcome);
        self.timings = timings;
        Ok(())
    }

    fn set_reasoning_usage(&mut self, usage: Option<ReasoningUsage>) {
        self.reasoning_usage = usage;
    }

    fn run_terminal_cleanup<T, F>(&mut self, cleanup: F) -> Result<T, String>
    where
        F: FnOnce() -> Result<T, String>,
    {
        cleanup()
    }

    fn completion_tokens(&self) -> usize {
        self.completion_tokens
    }
}

fn main() -> ExitCode {
    let args = match parse_args(env::args_os().skip(1)) {
        Ok(Some(args)) => args,
        Ok(None) => return ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("ullm-aq4-p2-full-model: {error}");
            return ExitCode::FAILURE;
        }
    };
    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("ullm-aq4-p2-full-model: {error}");
            ExitCode::FAILURE
        }
    }
}

fn parse_args(args: impl IntoIterator<Item = OsString>) -> Result<Option<Args>, String> {
    let mut args = args.into_iter();
    let mut served_model_manifest = None;
    let mut fixture = None;
    let mut output = None;
    let mut requested_m = None;
    let mut case_id = None;
    let mut device_index = 0_u32;
    while let Some(arg) = args.next() {
        match arg.to_str() {
            Some("--help") | Some("-h") => {
                print_help();
                return Ok(None);
            }
            Some("--served-model-manifest") => {
                served_model_manifest = Some(PathBuf::from(next_arg(
                    &mut args,
                    "--served-model-manifest",
                )?));
            }
            Some("--fixture") => {
                fixture = Some(PathBuf::from(next_arg(&mut args, "--fixture")?));
            }
            Some("--output") => {
                output = Some(PathBuf::from(next_arg(&mut args, "--output")?));
            }
            Some("--m") | Some("--requested-m") => {
                requested_m = Some(parse_usize(next_arg(&mut args, "--m")?, "requested M")?);
            }
            Some("--case-id") => {
                case_id = Some(
                    next_arg(&mut args, "--case-id")?
                        .into_string()
                        .map_err(|_| "--case-id must be UTF-8".to_string())?,
                );
            }
            Some("--device-index") => {
                let parsed = parse_usize(next_arg(&mut args, "--device-index")?, "device index")?;
                device_index =
                    u32::try_from(parsed).map_err(|_| "device index exceeds u32".to_string())?;
            }
            Some(other) => return Err(format!("unknown argument {other}")),
            None => return Err("arguments must be UTF-8".to_string()),
        }
    }
    let requested_m = requested_m.ok_or("--m is required")?;
    if !QWEN35_AQ4_PREFILL_CHUNK_GRID.contains(&requested_m) {
        return Err(format!(
            "requested M must be one of {QWEN35_AQ4_PREFILL_CHUNK_GRID:?}"
        ));
    }
    Ok(Some(Args {
        served_model_manifest: served_model_manifest
            .ok_or("--served-model-manifest is required")?,
        fixture: fixture.ok_or("--fixture is required")?,
        output: output.ok_or("--output is required")?,
        requested_m,
        case_id,
        device_index,
    }))
}

fn next_arg<I: Iterator<Item = OsString>>(args: &mut I, name: &str) -> Result<OsString, String> {
    args.next()
        .ok_or_else(|| format!("{name} requires a value"))
}

fn print_help() {
    eprintln!(
        "Usage: ullm-aq4-p2-full-model --served-model-manifest PATH --fixture PATH --output PATH --m 1|8|16|32|64|128 [--case-id ID] [--device-index N]"
    );
}

fn parse_usize(value: OsString, label: &str) -> Result<usize, String> {
    value
        .into_string()
        .map_err(|_| format!("{label} must be UTF-8"))?
        .parse::<usize>()
        .map_err(|_| format!("{label} must be a non-negative integer"))
}

fn run(args: Args) -> Result<(), String> {
    let model = load_served_model(&args.served_model_manifest)
        .map_err(|error| format!("served-model manifest rejected: {error}"))?;
    validate_model_contract(&model)?;
    let fixture_case = load_fixture_case(&args.fixture, args.case_id.as_deref())?;
    let profile = model.profile_snapshot();
    if fixture_case.prompt_token_ids.len() + fixture_case.step_count > profile.context_length {
        return Err("fixture request exceeds served-model context length".to_string());
    }
    if fixture_case.step_count == 0 || fixture_case.step_count > profile.max_new_tokens {
        return Err("fixture step_count is outside the served-model completion bound".to_string());
    }
    if fixture_case
        .prompt_token_ids
        .iter()
        .any(|token| *token >= profile.vocab_size)
    {
        return Err("fixture prompt token is outside the served-model vocabulary".to_string());
    }
    let package_dir = model
        .product
        .root
        .join(&model.product.package.manifest_path)
        .parent()
        .ok_or("served-model package manifest has no parent")?
        .to_path_buf();
    let model_config = Qwen35Aq4ModelLoadConfig {
        package_dir,
        device_index: args.device_index,
        expected_architecture: Some(profile.device.clone()),
        chunk_bytes: DEFAULT_CHUNK_BYTES,
        context_length: profile.context_length,
        kv_block_size: QWEN35_AQ4_KV_BLOCK_SIZE,
        layer_indices: None,
        lm_head_mode: PackageLmHeadMode::GpuResidentF32,
        lm_head_chunk_rows: DEFAULT_LM_HEAD_CHUNK_ROWS,
    };
    let session_config =
        Qwen35Aq4SessionConfig::greedy(fixture_case.step_count, profile.eos_token_ids.clone())
            .with_prefill_chunk_tokens(args.requested_m)?;
    let mut session = Qwen35Aq4InferenceSession::load(model_config, session_config)
        .map_err(classify_runtime_error)?;
    let request = InferenceRequest::new_with_eos(
        "aq4-p2-full-model",
        fixture_case.prompt_token_ids.clone(),
        fixture_case.step_count,
        profile.eos_token_ids,
        SamplingParams::greedy_with_top_k(0, 1),
    );
    let mut publications = Publications::default();
    let outcome = drive_worker_request(
        &mut session,
        request,
        CancellationToken::new(),
        &mut publications,
    )?;
    let terminal = session.terminal_sanitized_execution_audit();
    let audit = session
        .terminal_operation_execution_audit()
        .map(operation_audit_digest);
    let (resolved_m, actual_token_batch_width, actual_request_batch_width, lifecycle, reset) =
        terminal
            .as_ref()
            .map_or((None, None, None, None, None), |value| {
                (
                    value
                        .get("resolved_m")
                        .and_then(|v| v.as_u64())
                        .map(|v| v as usize),
                    value
                        .get("actual_token_batch_width")
                        .and_then(|v| v.as_u64())
                        .map(|v| v as usize),
                    value
                        .get("actual_request_batch_width")
                        .and_then(|v| v.as_u64())
                        .map(|v| v as usize),
                    value.get("lifecycle").cloned(),
                    value.get("lifecycle").and_then(|v| v.get("reset")).cloned(),
                )
            });
    let result = BenchmarkResult {
        schema_version: RESULT_SCHEMA,
        scope: "full_model",
        status: "ok",
        identity: identity_from_model(&model),
        fixture_case_id: fixture_case.case_id,
        requested_m: args.requested_m,
        resolved_m,
        actual_token_batch_width,
        actual_request_batch_width,
        timings: publications.timings,
        generated_tokens: publications.completion_tokens,
        operation_audit_digest: audit,
        lifecycle,
        reset,
        outcome: outcome_name(outcome),
        oom: false,
        fallback: FallbackFacts {
            count: 0,
            unexpected_count: 0,
        },
    };
    write_atomic_json(&args.output, &result)
}

fn validate_model_contract(model: &ServedModel) -> Result<(), String> {
    if model.format.format_id != "AQ4_0"
        || model.format.implementation_id != "qwen35_aq4_rdna4_v1"
        || model.worker.identity.device != "gfx1201"
        || model.worker.identity.execution_profile != "rdna4_aq4_resident"
    {
        return Err(
            "served-model format, implementation, device, or execution profile is not the AQ4 resident identity"
                .to_string(),
        );
    }
    if model.generation.sampling.temperature
        || model.generation.sampling.top_p
        || model.generation.sampling.top_k != 1
    {
        return Err(
            "full-model driver requires the served-model greedy sampling contract".to_string(),
        );
    }
    if model.product.artifact.is_some() {
        return Err("AQ4 full-model driver requires a package-only product contract".to_string());
    }
    if model.product.package.manifest_path.is_empty()
        || model.product.package.manifest_sha256.is_empty()
    {
        return Err("served-model package identity is incomplete".to_string());
    }
    let mut actual_environment = model.worker.required_environment.iter().collect::<Vec<_>>();
    actual_environment.sort_unstable();
    let mut required_environment = QWEN35_AQ4_REQUIRED_HIP_KERNEL_ENV.to_vec();
    required_environment.sort_unstable();
    if actual_environment.len() != required_environment.len()
        || actual_environment
            .iter()
            .zip(&required_environment)
            .any(|(actual, required)| actual.as_str() != *required)
    {
        return Err("served-model AQ4 required-environment contract is incomplete".to_string());
    }
    Ok(())
}

fn identity_from_model(model: &ServedModel) -> Identity {
    Identity {
        served_model_manifest_sha256: model.manifest_sha256.clone(),
        model_id: model.public.id.clone(),
        model_revision: model.public.revision.clone(),
        format_id: model.format.format_id.clone(),
        implementation_id: model.format.implementation_id.clone(),
        worker_binary_sha256: model.worker.binary_sha256.clone(),
        artifact_content_sha256: model
            .product
            .artifact
            .as_ref()
            .map(|artifact| artifact.content_sha256.clone())
            .unwrap_or_else(|| model.product.package.manifest_sha256.clone()),
        package_manifest_sha256: model.product.package.manifest_sha256.clone(),
        device: model.worker.identity.device.clone(),
        execution_profile: model.worker.identity.execution_profile.clone(),
    }
}

fn load_fixture_case(path: &Path, case_id: Option<&str>) -> Result<FixtureCase, String> {
    let metadata =
        fs::symlink_metadata(path).map_err(|error| format!("fixture metadata failed: {error}"))?;
    if !metadata.file_type().is_file() {
        return Err("fixture must be a regular non-symlink file".to_string());
    }
    if metadata.len() > MAX_FIXTURE_BYTES as u64 {
        return Err("fixture exceeds the bounded size limit".to_string());
    }
    let mut file = File::open(path).map_err(|error| format!("fixture open failed: {error}"))?;
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.read_to_end(&mut bytes)
        .map_err(|error| format!("fixture read failed: {error}"))?;
    let fixture: FixtureFile = serde_json::from_slice(&bytes)
        .map_err(|error| format!("fixture JSON rejected: {error}"))?;
    if fixture.cases.is_empty() || fixture.cases.len() > 128 {
        return Err("fixture must contain 1..=128 cases".to_string());
    }
    let raw = match case_id {
        Some(id) => fixture
            .cases
            .into_iter()
            .find(|case| case.case_id == id)
            .ok_or_else(|| "requested fixture case is absent".to_string())?,
        None => fixture
            .cases
            .into_iter()
            .next()
            .expect("nonempty fixture checked"),
    };
    if raw.case_id.is_empty()
        || raw.case_id.len() > 128
        || !raw
            .case_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
        || raw.prompt_token_ids.is_empty()
        || raw.prompt_token_ids.len() > 4096
        || raw.step_count == 0
        || raw.step_count > 512
    {
        return Err("fixture case exceeds bounded request limits".to_string());
    }
    Ok(FixtureCase {
        case_id: raw.case_id,
        prompt_token_ids: raw.prompt_token_ids,
        step_count: raw.step_count,
    })
}

fn operation_audit_digest(
    audit: &ullm_engine::backend_operation_registry::OperationExecutionAudit,
) -> String {
    let mut encoded = String::with_capacity(64);
    for byte in audit.deterministic_digest_sha256 {
        use std::fmt::Write as _;
        let _ = write!(&mut encoded, "{byte:02x}");
    }
    format!("sha256:{encoded}")
}

fn classify_runtime_error(error: String) -> String {
    if error.to_ascii_lowercase().contains("out of memory")
        || error.to_ascii_lowercase().contains("oom")
    {
        format!("runtime OOM: {error}")
    } else {
        error
    }
}

fn outcome_name(outcome: ReleaseOutcome) -> &'static str {
    match outcome {
        ReleaseOutcome::Stop => "stop",
        ReleaseOutcome::Length => "length",
        ReleaseOutcome::Cancelled => "cancelled",
    }
}

fn write_atomic_json<T: Serialize>(path: &Path, value: &T) -> Result<(), String> {
    if path.as_os_str().is_empty() || path.exists() || fs::symlink_metadata(path).is_ok() {
        return Err("output path is empty or already exists".to_string());
    }
    let bytes = serde_json::to_vec(value)
        .map_err(|error| format!("result serialization failed: {error}"))?;
    if bytes.len() > MAX_RESULT_BYTES {
        return Err("result exceeds the bounded size limit".to_string());
    }
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    reject_symlink_components(parent)?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("result parent creation failed: {error}"))?;
    let file_name = path
        .file_name()
        .ok_or("output path has no file name")?
        .to_string_lossy();
    let temp = parent.join(format!(".{file_name}.{}.tmp", std::process::id()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp)
        .map_err(|error| format!("temporary result creation failed: {error}"))?;
    file.write_all(&bytes)
        .map_err(|error| format!("result write failed: {error}"))?;
    file.write_all(b"\n")
        .map_err(|error| format!("result newline write failed: {error}"))?;
    file.sync_all()
        .map_err(|error| format!("result fsync failed: {error}"))?;
    drop(file);
    if let Err(error) = fs::hard_link(&temp, path) {
        let _ = fs::remove_file(&temp);
        return Err(format!("result publication refused: {error}"));
    }
    fs::remove_file(&temp).map_err(|error| format!("temporary result cleanup failed: {error}"))?;
    let parent_file =
        File::open(parent).map_err(|error| format!("result parent open failed: {error}"))?;
    parent_file
        .sync_all()
        .map_err(|error| format!("result parent fsync failed: {error}"))?;
    Ok(())
}

fn reject_symlink_components(path: &Path) -> Result<(), String> {
    let mut current = PathBuf::new();
    for component in path.components() {
        current.push(component.as_os_str());
        if let Ok(metadata) = fs::symlink_metadata(&current) {
            if metadata.file_type().is_symlink() {
                return Err(format!(
                    "path component is a symlink: {}",
                    current.display()
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn parser_accepts_supported_width_and_rejects_unknown() {
        let ok = parse_args([
            OsString::from("--served-model-manifest"),
            OsString::from("m.json"),
            OsString::from("--fixture"),
            OsString::from("cases.json"),
            OsString::from("--output"),
            OsString::from("result.json"),
            OsString::from("--m"),
            OsString::from("16"),
        ])
        .unwrap()
        .unwrap();
        assert_eq!(ok.requested_m, 16);
        let error = parse_args([
            OsString::from("--served-model-manifest"),
            OsString::from("m.json"),
            OsString::from("--fixture"),
            OsString::from("cases.json"),
            OsString::from("--output"),
            OsString::from("result.json"),
            OsString::from("--m"),
            OsString::from("7"),
        ])
        .unwrap_err();
        assert!(error.contains("one of"));
    }

    #[test]
    fn fixture_loader_rejects_symlink_and_selects_public_case() {
        let root = env::temp_dir().join(format!(
            "ullm-aq4-p2-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let fixture = root.join("cases.json");
        fs::write(
            &fixture,
            br#"{"cases":[{"case_id":"public-0","prompt_token_ids":[1,2],"step_count":2}]}"#,
        )
        .unwrap();
        let case = load_fixture_case(&fixture, Some("public-0")).unwrap();
        assert_eq!(case.case_id, "public-0");
        assert_eq!(case.step_count, 2);
        let link = root.join("link.json");
        std::os::unix::fs::symlink(&fixture, &link).unwrap();
        assert!(load_fixture_case(&link, None).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn atomic_publication_refuses_overwrite() {
        let root = env::temp_dir().join(format!("ullm-aq4-p2-pub-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let path = root.join("result.json");
        write_atomic_json(&path, &serde_json::json!({"ok": true})).unwrap();
        assert!(write_atomic_json(&path, &serde_json::json!({"ok": false})).is_err());
        let dangling = root.join("dangling.json");
        std::os::unix::fs::symlink(root.join("missing.json"), &dangling).unwrap();
        assert!(write_atomic_json(&dangling, &serde_json::json!({"ok": false})).is_err());
        let target_dir = root.join("target");
        fs::create_dir_all(&target_dir).unwrap();
        let linked_dir = root.join("linked");
        std::os::unix::fs::symlink(&target_dir, &linked_dir).unwrap();
        assert!(
            write_atomic_json(
                &linked_dir.join("result.json"),
                &serde_json::json!({"ok": false})
            )
            .is_err()
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn result_serialization_has_no_prompt_or_token_fields() {
        let result = BenchmarkResult {
            schema_version: RESULT_SCHEMA,
            scope: "full_model",
            status: "ok",
            identity: Identity {
                served_model_manifest_sha256: "sha256:manifest".into(),
                model_id: "Qwen/Qwen3.5-9B".into(),
                model_revision: "r1".into(),
                format_id: "AQ4_0".into(),
                implementation_id: "impl".into(),
                worker_binary_sha256: "sha256:worker".into(),
                artifact_content_sha256: "sha256:package".into(),
                package_manifest_sha256: "sha256:package-manifest".into(),
                device: "cpu".into(),
                execution_profile: "test".into(),
            },
            fixture_case_id: "public-fixture-0".into(),
            requested_m: 8,
            resolved_m: Some(1),
            actual_token_batch_width: Some(1),
            actual_request_batch_width: Some(1),
            timings: None,
            generated_tokens: 1,
            operation_audit_digest: Some("sha256:audit".into()),
            lifecycle: None,
            reset: None,
            outcome: "length",
            oom: false,
            fallback: FallbackFacts {
                count: 0,
                unexpected_count: 0,
            },
        };
        let value = serde_json::to_value(result).unwrap();
        let text = value.to_string();
        for forbidden in [
            "prompt_token_ids",
            "token_id",
            "generated_token_ids",
            "prompt_text",
            "output_text",
        ] {
            assert!(!text.contains(forbidden), "artifact leaked {forbidden}");
        }
    }

    #[test]
    fn identity_contract_rejects_non_aq4_manifest() {
        let model = ServedModel {
            manifest_path: PathBuf::from("manifest.json"),
            manifest_sha256: "sha256:manifest".into(),
            public: ullm_engine::served_model::PublicModel {
                id: "fixture".into(),
                name: "fixture".into(),
                description: "fixture".into(),
                upstream_id: "fixture".into(),
                revision: "r1".into(),
                context_length: 128,
            },
            generation: ullm_engine::served_model::GenerationContract {
                max_completion_tokens: 8,
                vocab_size: 1024,
                eos_token_ids: vec![0],
                sampling: ullm_engine::served_model::SamplingContract {
                    top_k: 1,
                    temperature: false,
                    top_p: false,
                },
            },
            format: ullm_engine::served_model::FormatContract {
                format_id: "SQ8_0".into(),
                implementation_id: "fixture".into(),
            },
            tokenizer: ullm_engine::served_model::TokenizerContract {
                root: PathBuf::from("tokenizer"),
                transformers_version: "fixture".into(),
                class_name: "fixture".into(),
                chat_template_sha256: "sha256:template".into(),
                files: Vec::new(),
                add_generation_prompt: false,
                enable_thinking: false,
            },
            worker: ullm_engine::served_model::WorkerContract {
                protocol: "ullm.worker.v1".into(),
                binary: PathBuf::from("worker"),
                binary_sha256: "sha256:worker".into(),
                arguments: Vec::new(),
                required_environment: Vec::new(),
                identity: ullm_engine::served_model::WorkerIdentity {
                    device: "cpu".into(),
                    execution_profile: "fixture".into(),
                },
            },
            product: ullm_engine::served_model::ProductContract {
                root: PathBuf::from("."),
                artifact: None,
                package: ullm_engine::served_model::PackageIdentity {
                    manifest_path: "package/manifest.json".into(),
                    manifest_sha256: "sha256:package".into(),
                },
            },
            promotion: ullm_engine::served_model::PromotionContract {
                source_commit: "fixture".into(),
                receipt: PathBuf::from("receipt"),
                receipt_sha256: "sha256:receipt".into(),
            },
            reasoning: None,
        };
        assert!(validate_model_contract(&model).is_err());
    }
}
