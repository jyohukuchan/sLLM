// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Offline AQ4 full-model benchmark driver.
//!
//! The driver uses the resident AQ4 model/session path exposed by the engine. It does not speak
//! the worker protocol and never mutates a running service. A result is published only after the
//! complete request has reset successfully; the artifact contains dimensions, hashes, and
//! counters, but no prompt, token id, or generated text.

use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::ffi::OsString;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use ullm_engine::aq4_worker_backend::QWEN35_AQ4_REQUIRED_HIP_KERNEL_ENV;
use ullm_engine::backend_operation_registry::{OperationExecutionAudit, ResolutionKind};
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
const MAX_RESULT_BYTES: usize = 256 * 1024;
const MAX_LINK_BYTES: usize = 16 * 1024 * 1024;
const MAX_PACKAGE_FILES: usize = 65_536;
const MAX_PACKAGE_DEPTH: usize = 32;
const HASH_CHUNK_BYTES: usize = 1024 * 1024;
const P2_PREFLIGHT_FIELDS: &[&str] = &[
    "weights_bytes",
    "persistent_state_bytes",
    "kv_cache_bytes",
    "workspace_bytes",
    "temporary_bytes",
    "vram_headroom_bytes",
    "gpu_process_snapshot",
];
const RESULT_SCHEMA: &str = "ullm.qwen35_aq4_p2.full_model_driver.v2";
const RAW_TARGET_SCHEMA: &str = "ullm.aq4_production_p2_raw_result.v2";
const DEFAULT_CHUNK_BYTES: usize = 1024 * 1024;
const DEFAULT_LM_HEAD_CHUNK_ROWS: usize = 8192;

#[derive(Debug, Clone, PartialEq, Eq)]
struct Args {
    served_model_manifest: PathBuf,
    fixture: PathBuf,
    case_binding: PathBuf,
    identity_binding: PathBuf,
    preflight: PathBuf,
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
#[serde(deny_unknown_fields)]
struct FixtureFile {
    cases: Vec<FixtureCaseRaw>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FixtureCaseRaw {
    case_id: String,
    prompt_token_ids: Vec<usize>,
    step_count: usize,
}

struct StrictJsonValue(Value);

impl<'de> Deserialize<'de> for StrictJsonValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(StrictJsonVisitor)
    }
}

struct StrictJsonVisitor;

impl<'de> Visitor<'de> for StrictJsonVisitor {
    type Value = StrictJsonValue;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("strict JSON without duplicate object keys")
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        Ok(StrictJsonValue(Value::Bool(value)))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
        Ok(StrictJsonValue(Value::Number(value.into())))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
        Ok(StrictJsonValue(Value::Number(value.into())))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        serde_json::Number::from_f64(value)
            .map(Value::Number)
            .map(StrictJsonValue)
            .ok_or_else(|| E::custom("non-finite JSON number"))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.visit_string(value.to_string())
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
        Ok(StrictJsonValue(Value::String(value)))
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(StrictJsonValue(Value::Null))
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(StrictJsonValue(Value::Null))
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(StrictJsonValue(value)) = sequence.next_element()? {
            values.push(value);
        }
        Ok(StrictJsonValue(Value::Array(values)))
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut values = serde_json::Map::new();
        while let Some((key, StrictJsonValue(value))) =
            map.next_entry::<String, StrictJsonValue>()?
        {
            if values.insert(key.clone(), value).is_some() {
                return Err(de::Error::custom(format!("duplicate JSON key: {key}")));
            }
        }
        Ok(StrictJsonValue(Value::Object(values)))
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct P2CaseBinding {
    case_id: String,
    fixture_id: String,
    case_sha256: String,
    stage_id: String,
    stage_order: u64,
    scope: String,
    phase: String,
    mode: String,
    baseline_mode: String,
    prompt_tokens: usize,
    cached_prefix_tokens: usize,
    context_tokens: usize,
    decode_start_tokens: usize,
    prefill_requested_m: usize,
    resolved_m: usize,
    request_count: usize,
    decode_request_count: usize,
    generated_tokens: usize,
    device: P2CaseDevice,
    control_id: String,
    control: P2CaseControl,
    sampling: P2CaseSampling,
    format_id: String,
    implementation_id: String,
    path_oracle_case_id: Option<String>,
    path_oracle_result_sha256: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct P2CaseDevice {
    device_id: String,
    backend: String,
    name: String,
    architecture: String,
    runtime_device_index: i32,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct P2CaseControl {
    control_id: String,
    role: String,
    format_id: String,
    implementation_id: String,
    promotion_eligible: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct P2CaseSampling {
    mode: String,
    temperature: f64,
    top_p: f64,
    top_k: usize,
    seed: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
struct RuntimeDeviceIdentity {
    requested_device_index: u32,
    observed_device_id: i32,
    observed_backend: String,
    observed_name: String,
    observed_architecture: String,
}

#[derive(Debug, Clone, Serialize)]
struct Identity {
    served_model_manifest_sha256: String,
    model_id: String,
    model_revision: String,
    format_id: String,
    implementation_id: String,
    manifest_worker_binary_path: String,
    manifest_worker_binary_sha256: String,
    benchmark_binary_path: String,
    benchmark_binary_sha256: String,
    benchmark_worker_roles_distinct: bool,
    package_root: String,
    package_content_sha256: String,
    package_manifest_sha256: String,
    package_file_count: usize,
    package_bytes: u64,
    manifest_device_architecture: String,
    runtime_device: RuntimeDeviceIdentity,
    execution_profile: String,
}

#[derive(Debug, Clone, Serialize)]
struct FallbackFacts {
    count: u64,
    unexpected_count: u64,
    reasons: Vec<FallbackReason>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
struct FallbackReason {
    unavailable_primary: String,
    resolved_implementation: String,
    invocation_count: u64,
}

#[derive(Debug, Clone, Serialize)]
struct ArtifactLink {
    path: String,
    sha256: String,
}

#[derive(Debug, Clone, Serialize)]
struct EmbeddedLink {
    json_pointer: &'static str,
    sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct ResultLinks {
    case: ArtifactLink,
    identity: ArtifactLink,
    preflight: ArtifactLink,
    timing: EmbeddedLink,
    audit: EmbeddedLink,
}

#[derive(Debug, Clone, Serialize)]
struct AdapterHandshake {
    target_schema_version: &'static str,
    mapping_version: &'static str,
    exact_root_fields: bool,
    benchmark_binary_role: &'static str,
    manifest_worker_role: &'static str,
    raw_v2_requires_role_aware_adapter: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DriverPreflight {
    input: Value,
    required_environment_count: usize,
    required_environment_verified: bool,
    binary_roles_verified: bool,
    package_tree_verified: bool,
}

#[derive(Debug, Clone, Serialize)]
struct TimingFacts {
    request_elapsed_ms: f64,
    generation: Option<GenerationTimings>,
    generated_tokens: usize,
}

#[derive(Debug, Clone, Serialize)]
struct AuditFacts {
    deterministic_digest_sha256: String,
    outcome: String,
    coverage_complete: bool,
    physical_operation_invocations: u64,
    total_records: u64,
}

#[derive(Debug, Clone, Serialize)]
struct FailureFacts {
    stage: &'static str,
    reason_code: &'static str,
}

#[derive(Debug, Clone, Serialize)]
struct OomFacts {
    stage: &'static str,
    reason_code: &'static str,
}

#[derive(Debug, Clone, Serialize)]
struct BenchmarkResult {
    schema_version: &'static str,
    raw_target_schema_version: &'static str,
    scope: &'static str,
    status: &'static str,
    immutable_status: bool,
    case_id: String,
    case_sha256: String,
    identity: Option<Identity>,
    requested_m: usize,
    resolved_m: Option<usize>,
    actual_token_batch_width: Option<usize>,
    actual_request_batch_width: Option<usize>,
    timing: TimingFacts,
    audit: Option<AuditFacts>,
    lifecycle: Option<serde_json::Value>,
    reset: Option<serde_json::Value>,
    outcome: Option<&'static str>,
    oom: Option<OomFacts>,
    fallback: FallbackFacts,
    preflight: DriverPreflight,
    failure: Option<FailureFacts>,
    links: ResultLinks,
    adapter: AdapterHandshake,
}

struct PackageTreeIdentity {
    sha256: String,
    file_count: usize,
    bytes: u64,
}

struct RunBindings {
    case_id: String,
    case_sha256: String,
    case_contract: P2CaseBinding,
    identity_value: Value,
    case: ArtifactLink,
    identity: ArtifactLink,
    preflight: ArtifactLink,
    preflight_value: Value,
}

#[derive(Debug, Clone)]
struct FallbackResolution {
    unavailable_primary: String,
    resolved_implementation: String,
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
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
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
    let mut case_binding = None;
    let mut identity_binding = None;
    let mut preflight = None;
    let mut output = None;
    let mut requested_m = None;
    let mut case_id = None;
    // Match the resident production worker's isolated HIP device default. CPU callers can pass
    // --device-index 0 explicitly for contract-only experiments.
    let mut device_index = 1_u32;
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
            Some("--case") => {
                case_binding = Some(PathBuf::from(next_arg(&mut args, "--case")?));
            }
            Some("--identity") => {
                identity_binding = Some(PathBuf::from(next_arg(&mut args, "--identity")?));
            }
            Some("--preflight") => {
                preflight = Some(PathBuf::from(next_arg(&mut args, "--preflight")?));
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
        case_binding: case_binding.ok_or("--case is required")?,
        identity_binding: identity_binding.ok_or("--identity is required")?,
        preflight: preflight.ok_or("--preflight is required")?,
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
        "Usage: ullm-aq4-p2-full-model --served-model-manifest PATH --fixture PATH --case PATH --identity PATH --preflight PATH --output PATH --m 1|8|16|32|64|128 [--case-id ID] [--device-index N]"
    );
}

fn parse_usize(value: OsString, label: &str) -> Result<usize, String> {
    value
        .into_string()
        .map_err(|_| format!("{label} must be UTF-8"))?
        .parse::<usize>()
        .map_err(|_| format!("{label} must be a non-negative integer"))
}

fn run(args: Args) -> Result<bool, String> {
    let started = Instant::now();
    let bindings = load_run_bindings(&args)?;
    let mut preflight = DriverPreflight {
        input: bindings.preflight_value.clone(),
        required_environment_count: 0,
        required_environment_verified: false,
        binary_roles_verified: false,
        package_tree_verified: false,
    };
    let fixture_case = match load_fixture_case(&args.fixture, args.case_id.as_deref()) {
        Ok(case) if case.case_id == bindings.case_id => case,
        Ok(_) | Err(_) => {
            return publish_failure_result(
                &args,
                &bindings,
                None,
                preflight,
                started,
                "fixture_load",
                "fixture_case_rejected",
                false,
            );
        }
    };
    let model = match load_served_model(&args.served_model_manifest) {
        Ok(model) => model,
        Err(_) => {
            return publish_failure_result(
                &args,
                &bindings,
                None,
                preflight,
                started,
                "manifest_load",
                "served_model_manifest_rejected",
                false,
            );
        }
    };
    if validate_model_contract(&model).is_err() {
        return publish_failure_result(
            &args,
            &bindings,
            None,
            preflight,
            started,
            "identity",
            "served_model_identity_rejected",
            false,
        );
    }
    preflight.required_environment_count = model.worker.required_environment.len();
    let package_dir = model
        .product
        .root
        .join(&model.product.package.manifest_path)
        .parent()
        .ok_or("served-model package manifest has no parent")?
        .to_path_buf();
    let benchmark_binary = match current_benchmark_binary_identity(&model) {
        Ok(identity) => {
            preflight.binary_roles_verified = true;
            identity
        }
        Err(_) => {
            return publish_failure_result(
                &args,
                &bindings,
                None,
                preflight,
                started,
                "binary_identity",
                "benchmark_worker_binary_identity_rejected",
                false,
            );
        }
    };
    let package_tree = match package_tree_identity(&package_dir) {
        Ok(identity) => {
            preflight.package_tree_verified = true;
            identity
        }
        Err(_) => {
            return publish_failure_result(
                &args,
                &bindings,
                None,
                preflight,
                started,
                "package_preflight",
                "package_content_identity_rejected",
                false,
            );
        }
    };
    let runtime_device = match observe_runtime_device(args.device_index) {
        Ok(identity) => identity,
        Err(_) => {
            return publish_failure_result(
                &args,
                &bindings,
                None,
                preflight,
                started,
                "device_identity",
                "runtime_device_identity_rejected",
                false,
            );
        }
    };
    let identity = identity_from_model(
        &model,
        &package_dir,
        &benchmark_binary,
        &package_tree,
        runtime_device,
    );
    if validate_p2_bindings(&args, &bindings, &fixture_case, &model, &identity).is_err() {
        return publish_failure_result(
            &args,
            &bindings,
            Some(identity),
            preflight,
            started,
            "p2_identity",
            "p2_identity_binding_mismatch",
            false,
        );
    }
    if validate_required_environment(&model.worker.required_environment).is_err() {
        return publish_failure_result(
            &args,
            &bindings,
            Some(identity),
            preflight,
            started,
            "environment_preflight",
            "required_environment_not_one",
            false,
        );
    }
    preflight.required_environment_verified = true;
    let profile = model.profile_snapshot();
    if fixture_case.prompt_token_ids.len() + fixture_case.step_count > profile.context_length {
        return publish_failure_result(
            &args,
            &bindings,
            Some(identity),
            preflight,
            started,
            "request_validation",
            "fixture_context_exceeded",
            false,
        );
    }
    if fixture_case.step_count == 0 || fixture_case.step_count > profile.max_new_tokens {
        return publish_failure_result(
            &args,
            &bindings,
            Some(identity),
            preflight,
            started,
            "request_validation",
            "fixture_completion_exceeded",
            false,
        );
    }
    if fixture_case
        .prompt_token_ids
        .iter()
        .any(|token| *token >= profile.vocab_size)
    {
        return publish_failure_result(
            &args,
            &bindings,
            Some(identity),
            preflight,
            started,
            "request_validation",
            "fixture_vocabulary_exceeded",
            false,
        );
    }
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
    let mut session = match Qwen35Aq4InferenceSession::load(model_config, session_config) {
        Ok(session) => session,
        Err(error) => {
            return publish_failure_result(
                &args,
                &bindings,
                Some(identity),
                preflight,
                started,
                "model_load",
                if is_oom(&error) {
                    "runtime_out_of_memory"
                } else {
                    "resident_model_load_failed"
                },
                is_oom(&error),
            );
        }
    };
    let request = InferenceRequest::new_with_eos(
        "aq4-p2-full-model",
        fixture_case.prompt_token_ids.clone(),
        fixture_case.step_count,
        profile.eos_token_ids,
        SamplingParams::greedy_with_top_k(0, 1),
    );
    let mut publications = Publications::default();
    let drive_result = drive_worker_request(
        &mut session,
        request,
        CancellationToken::new(),
        &mut publications,
    );
    if let Err(error) = drive_result {
        let _ = session.abort_and_reset();
        return publish_session_result(
            &args,
            &bindings,
            identity,
            preflight,
            started,
            &session,
            &publications,
            None,
            Some((
                "request_drive",
                if is_oom(&error) {
                    "runtime_out_of_memory"
                } else {
                    "request_drive_failed"
                },
                is_oom(&error),
            )),
        );
    }
    let outcome = drive_result.expect("drive result checked");
    publish_session_result(
        &args,
        &bindings,
        identity,
        preflight,
        started,
        &session,
        &publications,
        Some(outcome),
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn publish_session_result(
    args: &Args,
    bindings: &RunBindings,
    identity: Identity,
    preflight: DriverPreflight,
    started: Instant,
    session: &Qwen35Aq4InferenceSession,
    publications: &Publications,
    outcome: Option<ReleaseOutcome>,
    failure: Option<(&'static str, &'static str, bool)>,
) -> Result<bool, String> {
    let terminal = session.terminal_sanitized_execution_audit();
    let operation_audit = session.terminal_operation_execution_audit();
    let traces = session.operation_resolution_traces();
    let fallback = fallback_facts(operation_audit, &traces, args.requested_m == 1);
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
    let audit = operation_audit.map(audit_facts);
    let implicit_failure = if failure.is_none() {
        match operation_audit {
            None => Some(("operation_audit", "operation_audit_missing", false)),
            Some(value) if !value.coverage_complete => {
                Some(("operation_audit", "operation_audit_incomplete", false))
            }
            Some(_)
                if validate_terminal_evidence(
                    bindings,
                    resolved_m,
                    actual_token_batch_width,
                    actual_request_batch_width,
                    lifecycle.as_ref(),
                    reset.as_ref(),
                    outcome.is_some(),
                )
                .is_err() =>
            {
                Some(("terminal_evidence", "terminal_evidence_incomplete", false))
            }
            Some(_) if fallback.unexpected_count > 0 => {
                Some(("operation_fallback", "unexpected_fallback_observed", false))
            }
            Some(_) => None,
        }
    } else {
        None
    };
    let effective_failure = failure.or(implicit_failure);
    let status = match effective_failure {
        Some((_, _, true)) => "oom",
        Some(_) => "failed",
        None => "ok",
    };
    let timing = TimingFacts {
        request_elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        generation: publications.timings,
        generated_tokens: publications.completion_tokens,
    };
    let result = finalize_result(
        args,
        bindings,
        Some(identity),
        preflight,
        timing,
        audit,
        fallback,
        resolved_m,
        actual_token_batch_width,
        actual_request_batch_width,
        lifecycle,
        reset,
        outcome.map(outcome_name),
        status,
        effective_failure,
    )?;
    write_atomic_json(&args.output, &result)?;
    Ok(status == "ok")
}

fn load_run_bindings(args: &Args) -> Result<RunBindings, String> {
    let (case_value, case_link) = load_link_json(&args.case_binding, "case")?;
    let (identity_value, identity_link) = load_link_json(&args.identity_binding, "identity")?;
    let (preflight_value, preflight_link) = load_link_json(&args.preflight, "preflight")?;
    let case_id = case_value
        .get("case_id")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty() && value.len() <= 128)
        .ok_or_else(|| "P2 case binding has no bounded case_id".to_string())?
        .to_string();
    let case_sha256 = case_value
        .get("case_sha256")
        .and_then(Value::as_str)
        .filter(|value| valid_sha256(value))
        .ok_or_else(|| "P2 case binding has no valid case_sha256".to_string())?
        .to_string();
    if json_self_hash(&case_value, "case_sha256")? != case_sha256 {
        return Err("P2 case self-hash differs".to_string());
    }
    let case_contract: P2CaseBinding = serde_json::from_value(case_value.clone())
        .map_err(|error| format!("P2 case exact schema rejected: {error}"))?;
    if identity_value.get("schema_version").and_then(Value::as_str)
        != Some("ullm.aq4_production_p2_identity.v2")
        || identity_value.get("status").and_then(Value::as_str) != Some("bound")
        || !identity_value
            .get("identity_sha256")
            .and_then(Value::as_str)
            .is_some_and(valid_sha256)
    {
        return Err("P2 identity binding is not a bound v2 identity".to_string());
    }
    let identity_sha256 = identity_value
        .get("identity_sha256")
        .and_then(Value::as_str)
        .expect("identity SHA shape checked above");
    if json_self_hash(&identity_value, "identity_sha256")? != identity_sha256 {
        return Err("P2 identity self-hash differs".to_string());
    }
    validate_preflight_input(&preflight_value)?;
    Ok(RunBindings {
        case_id,
        case_sha256,
        case_contract,
        identity_value,
        case: case_link,
        identity: identity_link,
        preflight: preflight_link,
        preflight_value,
    })
}

fn validate_preflight_input(value: &Value) -> Result<(), String> {
    let object = value
        .as_object()
        .ok_or_else(|| "P2 preflight root must be an object".to_string())?;
    let actual = object.keys().map(String::as_str).collect::<BTreeSet<_>>();
    let expected = P2_PREFLIGHT_FIELDS.iter().copied().collect::<BTreeSet<_>>();
    if actual != expected {
        return Err("P2 preflight fields differ".to_string());
    }
    for field in P2_PREFLIGHT_FIELDS
        .iter()
        .copied()
        .filter(|field| *field != "gpu_process_snapshot")
    {
        if object.get(field).and_then(Value::as_u64).is_none() {
            return Err(format!(
                "P2 preflight {field} must be a non-negative integer"
            ));
        }
    }
    let snapshot = object
        .get("gpu_process_snapshot")
        .and_then(Value::as_array)
        .ok_or_else(|| "P2 preflight gpu_process_snapshot must be an array".to_string())?;
    for process in snapshot {
        let process = process
            .as_object()
            .ok_or_else(|| "P2 preflight process entry must be an object".to_string())?;
        let fields = process.keys().map(String::as_str).collect::<BTreeSet<_>>();
        if fields != ["pid", "process_name", "vram_bytes"].into_iter().collect()
            || process.get("pid").and_then(Value::as_u64).is_none()
            || process
                .get("process_name")
                .and_then(Value::as_str)
                .is_none_or(str::is_empty)
            || process.get("vram_bytes").and_then(Value::as_u64).is_none()
        {
            return Err("P2 preflight process entry is invalid".to_string());
        }
    }
    Ok(())
}

fn json_self_hash(value: &Value, field: &str) -> Result<String, String> {
    let mut value = value.clone();
    let object = value
        .as_object_mut()
        .ok_or_else(|| "self-hashed JSON root must be an object".to_string())?;
    if !object.contains_key(field) {
        return Err(format!("self-hashed JSON lacks {field}"));
    }
    object.insert(field.to_string(), Value::Null);
    let bytes = serde_json::to_vec(&value)
        .map_err(|error| format!("self-hash serialization failed: {error}"))?;
    Ok(sha256_bytes(&bytes))
}

fn validate_p2_bindings(
    args: &Args,
    bindings: &RunBindings,
    fixture: &FixtureCase,
    model: &ServedModel,
    identity: &Identity,
) -> Result<(), String> {
    let case = &bindings.case_contract;
    let expected_mode = if args.requested_m == 1 {
        "all_m1"
    } else {
        "cold_batched"
    };
    let expected_resolved_m = if expected_mode == "all_m1" {
        1
    } else {
        args.requested_m
    };
    validate_case_workload(
        case,
        fixture,
        &bindings.case_id,
        &bindings.case_sha256,
        args.requested_m,
        expected_mode,
        expected_resolved_m,
        &model.format.format_id,
        &model.format.implementation_id,
    )?;
    validate_case_device(&case.device, &identity.runtime_device, model)?;
    if args.device_index != identity.runtime_device.requested_device_index {
        return Err("requested runtime device index differs from result identity".to_string());
    }
    let binding = bindings
        .identity_value
        .as_object()
        .ok_or_else(|| "P2 identity binding root must be an object".to_string())?;
    let hashes = binding
        .get("hash_binding")
        .and_then(Value::as_object)
        .ok_or_else(|| "P2 identity hash binding is missing".to_string())?;
    let artifacts = binding
        .get("artifacts")
        .and_then(Value::as_object)
        .ok_or_else(|| "P2 identity artifacts are missing".to_string())?;
    let model_identity = binding
        .get("model_identity")
        .and_then(Value::as_object)
        .ok_or_else(|| "P2 model identity is missing".to_string())?;
    if model_identity.get("id").and_then(Value::as_str) != Some(model.public.id.as_str())
        || model_identity.get("revision").and_then(Value::as_str)
            != Some(model.public.revision.as_str())
        || model_identity.get("format_id").and_then(Value::as_str)
            != Some(model.format.format_id.as_str())
        || model_identity
            .get("implementation_id")
            .and_then(Value::as_str)
            != Some(model.format.implementation_id.as_str())
    {
        return Err("P2 bound model identity differs from the served model".to_string());
    }
    let expanded_manifest_sha256 = binding
        .get("expanded_manifest_sha256")
        .and_then(Value::as_str)
        .filter(|value| valid_sha256(value));
    let bound_case_manifest_sha256 = hashes
        .get("bound_case_manifest_sha256")
        .and_then(Value::as_str)
        .filter(|value| valid_sha256(value));
    if expanded_manifest_sha256.is_none() || expanded_manifest_sha256 != bound_case_manifest_sha256
    {
        return Err("P2 identity does not bind its expanded case manifest".to_string());
    }
    let expected = [
        (
            "served_model_manifest_sha256",
            identity.served_model_manifest_sha256.as_str(),
        ),
        (
            "worker_binary_sha256",
            identity.manifest_worker_binary_sha256.as_str(),
        ),
        (
            "package_manifest_sha256",
            identity.package_manifest_sha256.as_str(),
        ),
        (
            "package_content_sha256",
            identity.package_content_sha256.as_str(),
        ),
    ];
    if expected
        .iter()
        .any(|(field, value)| hashes.get(*field).and_then(Value::as_str) != Some(*value))
    {
        return Err("P2 identity hashes differ from the served model".to_string());
    }
    let served_manifest = model
        .manifest_path
        .canonicalize()
        .map_err(|error| format!("served manifest canonicalization failed: {error}"))?;
    let worker = model
        .worker
        .binary
        .canonicalize()
        .map_err(|error| format!("manifest worker canonicalization failed: {error}"))?;
    let package = Path::new(&identity.package_root)
        .canonicalize()
        .map_err(|error| format!("package root canonicalization failed: {error}"))?;
    let path_matches = [
        (
            artifacts
                .get("served_model_manifest")
                .and_then(Value::as_str),
            served_manifest.as_path(),
        ),
        (
            artifacts.get("worker").and_then(Value::as_str),
            worker.as_path(),
        ),
        (
            artifacts.get("package_root").and_then(Value::as_str),
            package.as_path(),
        ),
    ];
    if path_matches.iter().any(|(declared, actual)| {
        declared
            .and_then(|value| Path::new(value).canonicalize().ok())
            .as_deref()
            != Some(*actual)
    }) {
        return Err("P2 identity paths differ from the served model".to_string());
    }
    if binding.get("package_file_count").and_then(Value::as_u64)
        != Some(identity.package_file_count as u64)
    {
        return Err("P2 package file count differs".to_string());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_case_workload(
    case: &P2CaseBinding,
    fixture: &FixtureCase,
    bound_case_id: &str,
    bound_case_sha256: &str,
    requested_m: usize,
    expected_mode: &str,
    expected_resolved_m: usize,
    format_id: &str,
    implementation_id: &str,
) -> Result<(), String> {
    if case.case_id != bound_case_id
        || case.case_id != fixture.case_id
        || case.fixture_id != fixture.case_id
        || case.case_sha256 != bound_case_sha256
        || case.stage_id.is_empty()
        || case.stage_order == 0
        || case.scope != "full_model"
        || case.phase != "cold_prefill"
        || case.mode != expected_mode
        || case.baseline_mode != expected_mode
        || case.prompt_tokens != fixture.prompt_token_ids.len()
        || case.cached_prefix_tokens != 0
        || case.context_tokens != fixture.prompt_token_ids.len()
        || case.decode_start_tokens
            != if fixture.step_count == 0 {
                0
            } else {
                fixture.prompt_token_ids.len()
            }
        || case.prefill_requested_m != requested_m
        || case.resolved_m != expected_resolved_m
        || case.request_count != 1
        || case.decode_request_count != 0
        || case.generated_tokens != fixture.step_count
        || case.format_id != "AQ4_0"
        || case.format_id != format_id
        || case.implementation_id != implementation_id
        || case.control_id != "aq4_0_target"
        || case.control.control_id != case.control_id
        || case.control.role != "target"
        || case.control.format_id != case.format_id
        || case.control.implementation_id != case.implementation_id
        || !case.control.promotion_eligible
        || case.sampling.mode != "greedy"
        || case.sampling.temperature != 0.0
        || case.sampling.top_p != 1.0
        || case.sampling.top_k != 1
        || case.sampling.seed != 0
        || (case.mode == "all_m1"
            && (case.path_oracle_case_id.is_some() || case.path_oracle_result_sha256.is_some()))
        || (case.mode != "all_m1"
            && (case
                .path_oracle_case_id
                .as_deref()
                .is_none_or(str::is_empty)
                || case.path_oracle_result_sha256.is_some()))
    {
        return Err("P2 case workload/control fields do not exactly match the request".to_string());
    }
    Ok(())
}

fn validate_case_device(
    case: &P2CaseDevice,
    runtime: &RuntimeDeviceIdentity,
    model: &ServedModel,
) -> Result<(), String> {
    validate_case_device_values(case, runtime, &model.worker.identity.device)
}

fn validate_case_device_values(
    case: &P2CaseDevice,
    runtime: &RuntimeDeviceIdentity,
    manifest_device_architecture: &str,
) -> Result<(), String> {
    let expected_case_architecture = match runtime.observed_architecture.as_str() {
        "gfx1201" => "RDNA4",
        "gfx1030" | "gfx1031" => "RDNA2",
        other => other,
    };
    if case.device_id != "r9700-rdna4"
        || case.backend != runtime.observed_backend
        || case.name != runtime.observed_name
        || case.architecture != expected_case_architecture
        || case.runtime_device_index != runtime.observed_device_id
        || manifest_device_architecture != runtime.observed_architecture
    {
        return Err("P2 case device differs from the exact runtime device".to_string());
    }
    Ok(())
}

fn load_link_json(path: &Path, label: &str) -> Result<(Value, ArtifactLink), String> {
    let bytes = read_bounded_regular_file(path, label, MAX_LINK_BYTES)?;
    let value = parse_strict_json(&bytes, label)?;
    let canonical = path
        .canonicalize()
        .map_err(|error| format!("{label} canonicalization failed: {error}"))?;
    Ok((
        value,
        ArtifactLink {
            path: canonical.to_string_lossy().into_owned(),
            sha256: sha256_bytes(&bytes),
        },
    ))
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_required_environment(names: &[String]) -> Result<(), String> {
    validate_required_environment_with(names, |name| env::var(name).ok())
}

fn validate_required_environment_with<F>(names: &[String], mut value: F) -> Result<(), String>
where
    F: FnMut(&str) -> Option<String>,
{
    for name in names {
        if value(name).as_deref() != Some("1") {
            return Err(format!("required environment {name} must equal 1"));
        }
    }
    Ok(())
}

fn current_benchmark_binary_identity(model: &ServedModel) -> Result<ArtifactLink, String> {
    let path = env::current_exe()
        .map_err(|error| format!("benchmark executable query failed: {error}"))?
        .canonicalize()
        .map_err(|error| format!("benchmark executable canonicalization failed: {error}"))?;
    let metadata = fs::symlink_metadata(&path)
        .map_err(|error| format!("benchmark executable metadata failed: {error}"))?;
    if !metadata.file_type().is_file() {
        return Err("benchmark executable must be a regular file".to_string());
    }
    let sha256 = sha256_file(&path)?;
    validate_binary_role_values(
        &path,
        &sha256,
        &model.worker.binary,
        &model.worker.binary_sha256,
    )?;
    Ok(ArtifactLink {
        path: path.to_string_lossy().into_owned(),
        sha256,
    })
}

fn validate_binary_role_values(
    benchmark_path: &Path,
    benchmark_sha256: &str,
    worker_path: &Path,
    worker_sha256: &str,
) -> Result<(), String> {
    if benchmark_path == worker_path || benchmark_sha256 == worker_sha256 {
        return Err("benchmark executable must not impersonate the manifest worker".to_string());
    }
    if !valid_sha256(benchmark_sha256) || !valid_sha256(worker_sha256) {
        return Err("binary identity digest is invalid".to_string());
    }
    Ok(())
}

fn package_tree_identity(root: &Path) -> Result<PackageTreeIdentity, String> {
    let root = root
        .canonicalize()
        .map_err(|error| format!("package root canonicalization failed: {error}"))?;
    let metadata = fs::symlink_metadata(&root)
        .map_err(|error| format!("package root metadata failed: {error}"))?;
    if !metadata.file_type().is_dir() {
        return Err("package root must be a non-symlink directory".to_string());
    }
    let mut pending = vec![(root.clone(), 0_usize)];
    let mut files = Vec::new();
    while let Some((directory, depth)) = pending.pop() {
        if depth > MAX_PACKAGE_DEPTH {
            return Err("package tree exceeds the depth limit".to_string());
        }
        let entries = fs::read_dir(&directory)
            .map_err(|error| format!("package directory read failed: {error}"))?;
        for entry in entries {
            let entry = entry.map_err(|error| format!("package entry read failed: {error}"))?;
            let path = entry.path();
            let metadata = fs::symlink_metadata(&path)
                .map_err(|error| format!("package entry metadata failed: {error}"))?;
            if metadata.file_type().is_symlink() {
                return Err("package tree contains a symlink".to_string());
            }
            if metadata.file_type().is_dir() {
                pending.push((path, depth + 1));
            } else if metadata.file_type().is_file() {
                files.push(path);
                if files.len() > MAX_PACKAGE_FILES {
                    return Err("package tree exceeds the file-count limit".to_string());
                }
            } else {
                return Err("package tree contains a non-regular entry".to_string());
            }
        }
    }
    if files.is_empty() {
        return Err("package tree is empty".to_string());
    }
    files.sort_by(|left, right| {
        left.strip_prefix(&root)
            .expect("walked package path has root prefix")
            .cmp(
                right
                    .strip_prefix(&root)
                    .expect("walked package path has root prefix"),
            )
    });
    let mut tree = Sha256::new();
    let mut bytes = 0_u64;
    for path in &files {
        let relative = path
            .strip_prefix(&root)
            .expect("walked package path has root prefix")
            .to_string_lossy()
            .replace(std::path::MAIN_SEPARATOR, "/");
        if relative.is_empty() || relative.len() > 4096 {
            return Err("package relative path is invalid".to_string());
        }
        let (file_digest, file_bytes) = sha256_file_bytes(path)?;
        bytes = bytes
            .checked_add(file_bytes)
            .ok_or_else(|| "package byte count overflows".to_string())?;
        tree.update(relative.as_bytes());
        tree.update(b"\0");
        tree.update(file_digest);
        tree.update(b"\n");
    }
    Ok(PackageTreeIdentity {
        sha256: hex_digest(tree.finalize()),
        file_count: files.len(),
        bytes,
    })
}

fn sha256_file(path: &Path) -> Result<String, String> {
    sha256_file_bytes(path).map(|(digest, _)| hex_digest(digest))
}

fn sha256_file_bytes(path: &Path) -> Result<([u8; 32], u64), String> {
    let metadata =
        fs::symlink_metadata(path).map_err(|error| format!("file metadata failed: {error}"))?;
    if !metadata.file_type().is_file() {
        return Err("hash input must be a regular non-symlink file".to_string());
    }
    let mut file = File::open(path).map_err(|error| format!("hash input open failed: {error}"))?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; HASH_CHUNK_BYTES];
    let mut bytes = 0_u64;
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| format!("hash input read failed: {error}"))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
        bytes = bytes
            .checked_add(read as u64)
            .ok_or_else(|| "hash byte count overflows".to_string())?;
    }
    Ok((digest.finalize().into(), bytes))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    hex_digest(Sha256::digest(bytes))
}

fn hex_digest(bytes: impl AsRef<[u8]>) -> String {
    let mut encoded = String::with_capacity(64);
    for byte in bytes.as_ref() {
        use std::fmt::Write as _;
        let _ = write!(&mut encoded, "{byte:02x}");
    }
    encoded
}

#[allow(clippy::too_many_arguments)]
fn publish_failure_result(
    args: &Args,
    bindings: &RunBindings,
    identity: Option<Identity>,
    preflight: DriverPreflight,
    started: Instant,
    stage: &'static str,
    reason_code: &'static str,
    oom: bool,
) -> Result<bool, String> {
    let status = if oom { "oom" } else { "failed" };
    let result = finalize_result(
        args,
        bindings,
        identity,
        preflight,
        TimingFacts {
            request_elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
            generation: None,
            generated_tokens: 0,
        },
        None,
        empty_fallback(),
        None,
        None,
        None,
        None,
        None,
        None,
        status,
        Some((stage, reason_code, oom)),
    )?;
    write_atomic_json(&args.output, &result)?;
    Ok(false)
}

#[allow(clippy::too_many_arguments)]
fn finalize_result(
    args: &Args,
    bindings: &RunBindings,
    identity: Option<Identity>,
    preflight: DriverPreflight,
    timing: TimingFacts,
    audit: Option<AuditFacts>,
    fallback: FallbackFacts,
    resolved_m: Option<usize>,
    actual_token_batch_width: Option<usize>,
    actual_request_batch_width: Option<usize>,
    lifecycle: Option<Value>,
    reset: Option<Value>,
    outcome: Option<&'static str>,
    status: &'static str,
    failure: Option<(&'static str, &'static str, bool)>,
) -> Result<BenchmarkResult, String> {
    if status == "ok"
        && (audit.is_none()
            || identity.is_none()
            || !preflight.required_environment_verified
            || !preflight.binary_roles_verified
            || !preflight.package_tree_verified
            || validate_terminal_evidence(
                bindings,
                resolved_m,
                actual_token_batch_width,
                actual_request_batch_width,
                lifecycle.as_ref(),
                reset.as_ref(),
                outcome.is_some(),
            )
            .is_err())
    {
        return Err("ok result lacks mandatory identity/audit/preflight evidence".to_string());
    }
    let timing_sha256 = canonical_digest(&timing)?;
    let audit_sha256 = audit
        .as_ref()
        .map(|value| value.deterministic_digest_sha256.clone());
    Ok(BenchmarkResult {
        schema_version: RESULT_SCHEMA,
        raw_target_schema_version: RAW_TARGET_SCHEMA,
        scope: "full_model",
        status,
        immutable_status: status != "ok",
        case_id: bindings.case_id.clone(),
        case_sha256: bindings.case_sha256.clone(),
        identity,
        requested_m: args.requested_m,
        resolved_m,
        actual_token_batch_width,
        actual_request_batch_width,
        timing,
        audit,
        lifecycle,
        reset,
        outcome,
        oom: failure
            .and_then(|(stage, reason_code, oom)| oom.then_some(OomFacts { stage, reason_code })),
        fallback,
        preflight,
        failure: failure.map(|(stage, reason_code, _)| FailureFacts { stage, reason_code }),
        links: ResultLinks {
            case: ArtifactLink {
                path: bindings.case.path.clone(),
                sha256: bindings.case.sha256.clone(),
            },
            identity: ArtifactLink {
                path: bindings.identity.path.clone(),
                sha256: bindings.identity.sha256.clone(),
            },
            preflight: ArtifactLink {
                path: bindings.preflight.path.clone(),
                sha256: bindings.preflight.sha256.clone(),
            },
            timing: EmbeddedLink {
                json_pointer: "/timing",
                sha256: Some(timing_sha256),
            },
            audit: EmbeddedLink {
                json_pointer: "/audit",
                sha256: audit_sha256,
            },
        },
        adapter: AdapterHandshake {
            target_schema_version: RAW_TARGET_SCHEMA,
            mapping_version: "ullm.aq4_p2_full_model_to_raw.v1",
            exact_root_fields: true,
            benchmark_binary_role: "executed_benchmark_driver",
            manifest_worker_role: "served_identity_reference",
            raw_v2_requires_role_aware_adapter: true,
        },
    })
}

#[allow(clippy::too_many_arguments)]
fn validate_terminal_evidence(
    bindings: &RunBindings,
    resolved_m: Option<usize>,
    actual_token_batch_width: Option<usize>,
    actual_request_batch_width: Option<usize>,
    lifecycle: Option<&Value>,
    reset: Option<&Value>,
    outcome_present: bool,
) -> Result<(), String> {
    let case = &bindings.case_contract;
    if resolved_m != Some(case.resolved_m)
        || actual_token_batch_width != Some(case.resolved_m)
        || actual_request_batch_width != Some(case.request_count)
        || !outcome_present
    {
        return Err("terminal widths or outcome differ from the case".to_string());
    }
    let lifecycle = lifecycle
        .and_then(Value::as_object)
        .ok_or_else(|| "terminal lifecycle is missing".to_string())?;
    let prepare = lifecycle_count(lifecycle, "prepare")?;
    let commit = lifecycle_count(lifecycle, "commit")?;
    let discard = lifecycle_count(lifecycle, "discard")?;
    if prepare == 0
        || prepare
            != commit
                .checked_add(discard)
                .ok_or("lifecycle count overflow")?
        || lifecycle_count(lifecycle, "error")? != 0
        || lifecycle_count(lifecycle, "cancel")? != 0
    {
        return Err("terminal lifecycle counts do not reconcile".to_string());
    }
    for phase in ["prefill", "publication"] {
        let phase = lifecycle
            .get(phase)
            .and_then(Value::as_object)
            .ok_or_else(|| "terminal lifecycle phase is missing".to_string())?;
        let prepared = lifecycle_count(phase, "prepare")?;
        let committed = lifecycle_count(phase, "commit")?;
        let discarded = lifecycle_count(phase, "discard")?;
        if prepared
            != committed
                .checked_add(discarded)
                .ok_or("lifecycle phase count overflow")?
        {
            return Err("terminal lifecycle phase counts do not reconcile".to_string());
        }
    }
    let lifecycle_reset = lifecycle
        .get("reset")
        .and_then(Value::as_object)
        .ok_or_else(|| "terminal lifecycle reset is missing".to_string())?;
    let reset = reset
        .and_then(Value::as_object)
        .ok_or_else(|| "terminal reset is missing".to_string())?;
    for counts in [lifecycle_reset, reset] {
        if lifecycle_count(counts, "attempted")? != 1
            || lifecycle_count(counts, "complete")? != 1
            || lifecycle_count(counts, "failed")? != 0
        {
            return Err("terminal reset did not complete exactly once".to_string());
        }
    }
    if Value::Object(lifecycle_reset.clone()) != Value::Object(reset.clone()) {
        return Err("terminal reset link differs from lifecycle reset".to_string());
    }
    Ok(())
}

fn lifecycle_count(object: &serde_json::Map<String, Value>, field: &str) -> Result<u64, String> {
    object
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("terminal lifecycle {field} is missing"))
}

fn canonical_digest<T: Serialize>(value: &T) -> Result<String, String> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| format!("canonical digest serialization failed: {error}"))?;
    Ok(sha256_bytes(&bytes))
}

fn audit_facts(audit: &OperationExecutionAudit) -> AuditFacts {
    AuditFacts {
        deterministic_digest_sha256: operation_audit_digest(audit),
        outcome: audit.outcome.to_string(),
        coverage_complete: audit.coverage_complete,
        physical_operation_invocations: audit.physical_operation_invocations,
        total_records: audit.total_records,
    }
}

fn fallback_facts(
    audit: Option<&OperationExecutionAudit>,
    traces: &[Vec<ullm_engine::backend_operation_registry::OperationResolutionTrace>],
    expected: bool,
) -> FallbackFacts {
    let Some(audit) = audit else {
        return empty_fallback();
    };
    let counts = audit
        .implementation_counts
        .iter()
        .map(|value| (value.implementation_id, value.count))
        .collect::<BTreeMap<_, _>>();
    let resolutions = traces
        .iter()
        .flatten()
        .filter_map(|trace| match trace.resolution {
            ResolutionKind::Primary => None,
            ResolutionKind::Fallback {
                unavailable_primary,
            } => Some(FallbackResolution {
                unavailable_primary: unavailable_primary.to_string(),
                resolved_implementation: trace.implementation_id.to_string(),
            }),
        })
        .collect::<Vec<_>>();
    fallback_facts_from_sources(&counts, &resolutions, expected)
}

fn fallback_facts_from_sources(
    counts: &BTreeMap<&str, u64>,
    resolutions: &[FallbackResolution],
    expected: bool,
) -> FallbackFacts {
    let mut pairs = BTreeSet::new();
    let mut implementations = BTreeSet::new();
    let mut reasons = Vec::new();
    for resolution in resolutions {
        if pairs.insert((
            resolution.unavailable_primary.as_str(),
            resolution.resolved_implementation.as_str(),
        )) {
            implementations.insert(resolution.resolved_implementation.as_str());
            reasons.push(FallbackReason {
                unavailable_primary: resolution.unavailable_primary.clone(),
                resolved_implementation: resolution.resolved_implementation.clone(),
                invocation_count: counts
                    .get(resolution.resolved_implementation.as_str())
                    .copied()
                    .unwrap_or(0),
            });
        }
    }
    let count = implementations
        .iter()
        .filter_map(|implementation| counts.get(*implementation))
        .try_fold(0_u64, |total, value| total.checked_add(*value))
        .unwrap_or(u64::MAX);
    FallbackFacts {
        count,
        unexpected_count: if expected { 0 } else { count },
        reasons,
    }
}

fn empty_fallback() -> FallbackFacts {
    FallbackFacts {
        count: 0,
        unexpected_count: 0,
        reasons: Vec::new(),
    }
}

fn is_oom(error: &str) -> bool {
    let lower = error.to_ascii_lowercase();
    lower.contains("out of memory")
        || lower.contains("out_of_memory")
        || lower.contains("hiperroroutofmemory")
        || lower.contains("oom")
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

fn identity_from_model(
    model: &ServedModel,
    package_dir: &Path,
    benchmark_binary: &ArtifactLink,
    package_tree: &PackageTreeIdentity,
    runtime_device: RuntimeDeviceIdentity,
) -> Identity {
    Identity {
        served_model_manifest_sha256: model.manifest_sha256.clone(),
        model_id: model.public.id.clone(),
        model_revision: model.public.revision.clone(),
        format_id: model.format.format_id.clone(),
        implementation_id: model.format.implementation_id.clone(),
        manifest_worker_binary_path: model.worker.binary.to_string_lossy().into_owned(),
        manifest_worker_binary_sha256: model.worker.binary_sha256.clone(),
        benchmark_binary_path: benchmark_binary.path.clone(),
        benchmark_binary_sha256: benchmark_binary.sha256.clone(),
        benchmark_worker_roles_distinct: true,
        package_root: package_dir.to_string_lossy().into_owned(),
        package_content_sha256: package_tree.sha256.clone(),
        package_manifest_sha256: model.product.package.manifest_sha256.clone(),
        package_file_count: package_tree.file_count,
        package_bytes: package_tree.bytes,
        manifest_device_architecture: model.worker.identity.device.clone(),
        runtime_device,
        execution_profile: model.worker.identity.execution_profile.clone(),
    }
}

fn observe_runtime_device(device_index: u32) -> Result<RuntimeDeviceIdentity, String> {
    let observed = ullm_runtime_sys::device_info(device_index)
        .map_err(|error| format!("runtime device query failed: {error}"))?;
    if observed.backend.is_empty()
        || observed.name.is_empty()
        || observed.gcn_arch_name.is_empty()
        || observed.device_id < 0
    {
        return Err("runtime device identity is incomplete".to_string());
    }
    Ok(RuntimeDeviceIdentity {
        requested_device_index: device_index,
        observed_device_id: observed.device_id,
        observed_backend: observed.backend,
        observed_name: observed.name,
        observed_architecture: observed.gcn_arch_name,
    })
}

fn load_fixture_case(path: &Path, case_id: Option<&str>) -> Result<FixtureCase, String> {
    let bytes = read_bounded_regular_file(path, "fixture", MAX_FIXTURE_BYTES)?;
    let fixture: FixtureFile = serde_json::from_value(parse_strict_json(&bytes, "fixture")?)
        .map_err(|error| format!("fixture JSON rejected: {error}"))?;
    if fixture.cases.is_empty() || fixture.cases.len() > 128 {
        return Err("fixture must contain 1..=128 cases".to_string());
    }
    let mut case_ids = BTreeSet::new();
    for case in &fixture.cases {
        if !case_ids.insert(case.case_id.as_str()) {
            return Err("fixture contains duplicate case_id".to_string());
        }
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

fn read_bounded_regular_file(
    path: &Path,
    label: &str,
    maximum_bytes: usize,
) -> Result<Vec<u8>, String> {
    reject_symlink_components(path)?;
    let before =
        fs::symlink_metadata(path).map_err(|error| format!("{label} metadata failed: {error}"))?;
    if !before.file_type().is_file() || before.len() > maximum_bytes as u64 {
        return Err(format!(
            "{label} must be a bounded regular non-symlink file"
        ));
    }
    let mut file = File::open(path).map_err(|error| format!("{label} open failed: {error}"))?;
    let opened = file
        .metadata()
        .map_err(|error| format!("{label} opened metadata failed: {error}"))?;
    if !opened.file_type().is_file()
        || opened.dev() != before.dev()
        || opened.ino() != before.ino()
        || opened.len() != before.len()
    {
        return Err(format!("{label} identity changed while opening"));
    }
    let mut bytes = Vec::with_capacity(opened.len() as usize);
    let mut chunk = [0_u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut chunk)
            .map_err(|error| format!("{label} read failed: {error}"))?;
        if read == 0 {
            break;
        }
        if bytes.len().saturating_add(read) > maximum_bytes {
            return Err(format!("{label} exceeds the bounded size limit"));
        }
        bytes.extend_from_slice(&chunk[..read]);
    }
    let after = file
        .metadata()
        .map_err(|error| format!("{label} final metadata failed: {error}"))?;
    if after.dev() != opened.dev()
        || after.ino() != opened.ino()
        || after.len() != opened.len()
        || after.len() != bytes.len() as u64
    {
        return Err(format!("{label} changed while reading"));
    }
    Ok(bytes)
}

fn parse_strict_json(bytes: &[u8], label: &str) -> Result<Value, String> {
    serde_json::from_slice::<StrictJsonValue>(bytes)
        .map(|value| value.0)
        .map_err(|error| format!("{label} JSON rejected: {error}"))
}

fn operation_audit_digest(
    audit: &ullm_engine::backend_operation_registry::OperationExecutionAudit,
) -> String {
    let mut encoded = String::with_capacity(64);
    for byte in audit.deterministic_digest_sha256 {
        use std::fmt::Write as _;
        let _ = write!(&mut encoded, "{byte:02x}");
    }
    encoded
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

    fn test_args() -> Args {
        Args {
            served_model_manifest: PathBuf::from("served.json"),
            fixture: PathBuf::from("fixture.json"),
            case_binding: PathBuf::from("case.json"),
            identity_binding: PathBuf::from("identity.json"),
            preflight: PathBuf::from("preflight.json"),
            output: PathBuf::from("result.json"),
            requested_m: 8,
            case_id: Some("public-fixture-0".into()),
            device_index: 1,
        }
    }

    fn test_bindings() -> RunBindings {
        RunBindings {
            case_id: "public-fixture-0".into(),
            case_sha256: "1".repeat(64),
            case_contract: test_case_contract(),
            identity_value: serde_json::json!({"status": "bound"}),
            case: ArtifactLink {
                path: "/tmp/case.json".into(),
                sha256: "2".repeat(64),
            },
            identity: ArtifactLink {
                path: "/tmp/identity.json".into(),
                sha256: "3".repeat(64),
            },
            preflight: ArtifactLink {
                path: "/tmp/preflight.json".into(),
                sha256: "4".repeat(64),
            },
            preflight_value: serde_json::json!({"vram_headroom_bytes": 1}),
        }
    }

    fn test_case_contract() -> P2CaseBinding {
        P2CaseBinding {
            case_id: "public-fixture-0".into(),
            fixture_id: "public-fixture-0".into(),
            case_sha256: "1".repeat(64),
            stage_id: "test".into(),
            stage_order: 1,
            scope: "full_model".into(),
            phase: "cold_prefill".into(),
            mode: "cold_batched".into(),
            baseline_mode: "cold_batched".into(),
            prompt_tokens: 2,
            cached_prefix_tokens: 0,
            context_tokens: 2,
            decode_start_tokens: 2,
            prefill_requested_m: 8,
            resolved_m: 8,
            request_count: 1,
            decode_request_count: 0,
            generated_tokens: 1,
            device: P2CaseDevice {
                device_id: "r9700-rdna4".into(),
                backend: "hip".into(),
                name: "Radeon AI PRO R9700".into(),
                architecture: "RDNA4".into(),
                runtime_device_index: 0,
            },
            control_id: "aq4_0_target".into(),
            control: P2CaseControl {
                control_id: "aq4_0_target".into(),
                role: "target".into(),
                format_id: "AQ4_0".into(),
                implementation_id: "qwen35_aq4_rdna4_v1".into(),
                promotion_eligible: true,
            },
            sampling: P2CaseSampling {
                mode: "greedy".into(),
                temperature: 0.0,
                top_p: 1.0,
                top_k: 1,
                seed: 0,
            },
            format_id: "AQ4_0".into(),
            implementation_id: "qwen35_aq4_rdna4_v1".into(),
            path_oracle_case_id: Some("oracle".into()),
            path_oracle_result_sha256: None,
        }
    }

    fn test_identity() -> Identity {
        Identity {
            served_model_manifest_sha256: "5".repeat(64),
            model_id: "Qwen/Qwen3.5-9B".into(),
            model_revision: "r1".into(),
            format_id: "AQ4_0".into(),
            implementation_id: "qwen35_aq4_rdna4_v1".into(),
            manifest_worker_binary_path: "/tmp/worker".into(),
            manifest_worker_binary_sha256: "6".repeat(64),
            benchmark_binary_path: "/tmp/benchmark".into(),
            benchmark_binary_sha256: "7".repeat(64),
            benchmark_worker_roles_distinct: true,
            package_root: "/tmp/package".into(),
            package_content_sha256: "8".repeat(64),
            package_manifest_sha256: "9".repeat(64),
            package_file_count: 2,
            package_bytes: 3,
            manifest_device_architecture: "gfx1201".into(),
            runtime_device: RuntimeDeviceIdentity {
                requested_device_index: 1,
                observed_device_id: 0,
                observed_backend: "hip".into(),
                observed_name: "Radeon AI PRO R9700".into(),
                observed_architecture: "gfx1201".into(),
            },
            execution_profile: "rdna4_aq4_resident".into(),
        }
    }

    fn test_preflight(verified: bool) -> DriverPreflight {
        DriverPreflight {
            input: serde_json::json!({"vram_headroom_bytes": 1}),
            required_environment_count: 2,
            required_environment_verified: verified,
            binary_roles_verified: verified,
            package_tree_verified: verified,
        }
    }

    fn test_audit() -> AuditFacts {
        AuditFacts {
            deterministic_digest_sha256: "a".repeat(64),
            outcome: "length".into(),
            coverage_complete: true,
            physical_operation_invocations: 64,
            total_records: 64,
        }
    }

    fn test_lifecycle() -> Value {
        serde_json::json!({
            "prepare": 2,
            "commit": 2,
            "discard": 0,
            "error": 0,
            "cancel": 0,
            "prefill": {"prepare": 1, "commit": 1, "discard": 0},
            "publication": {"prepare": 1, "commit": 1, "discard": 0},
            "reset": {"attempted": 1, "complete": 1, "failed": 0}
        })
    }

    fn test_reset() -> Value {
        serde_json::json!({"attempted": 1, "complete": 1, "failed": 0})
    }

    fn ok_result(audit: Option<AuditFacts>) -> Result<BenchmarkResult, String> {
        finalize_result(
            &test_args(),
            &test_bindings(),
            Some(test_identity()),
            test_preflight(true),
            TimingFacts {
                request_elapsed_ms: 1.0,
                generation: None,
                generated_tokens: 1,
            },
            audit,
            empty_fallback(),
            Some(8),
            Some(8),
            Some(1),
            Some(test_lifecycle()),
            Some(test_reset()),
            Some("length"),
            "ok",
            None,
        )
    }

    #[test]
    fn parser_accepts_supported_width_and_rejects_unknown() {
        let ok = parse_args([
            OsString::from("--served-model-manifest"),
            OsString::from("m.json"),
            OsString::from("--fixture"),
            OsString::from("cases.json"),
            OsString::from("--case"),
            OsString::from("case.json"),
            OsString::from("--identity"),
            OsString::from("identity.json"),
            OsString::from("--preflight"),
            OsString::from("preflight.json"),
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
            OsString::from("--case"),
            OsString::from("case.json"),
            OsString::from("--identity"),
            OsString::from("identity.json"),
            OsString::from("--preflight"),
            OsString::from("preflight.json"),
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
        fs::write(
            &fixture,
            br#"{"cases":[{"case_id":"public-0","prompt_token_ids":[1],"step_count":1},{"case_id":"public-0","prompt_token_ids":[2],"step_count":1}]}"#,
        )
        .unwrap();
        assert!(load_fixture_case(&fixture, None).is_err());
        let link = root.join("link.json");
        std::os::unix::fs::symlink(&fixture, &link).unwrap();
        assert!(load_fixture_case(&link, None).is_err());
        let real_parent = root.join("real-parent");
        fs::create_dir_all(&real_parent).unwrap();
        let nested = real_parent.join("nested.json");
        fs::write(
            &nested,
            br#"{"cases":[{"case_id":"parent","prompt_token_ids":[1],"step_count":1}]}"#,
        )
        .unwrap();
        let linked_parent = root.join("linked-parent");
        std::os::unix::fs::symlink(&real_parent, &linked_parent).unwrap();
        assert!(load_fixture_case(&linked_parent.join("nested.json"), None).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn fixture_json_rejects_duplicate_and_unknown_fields() {
        let root = env::temp_dir().join(format!(
            "ullm-aq4-p2-strict-fixture-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let fixture = root.join("fixture.json");
        fs::write(
            &fixture,
            br#"{"cases":[{"case_id":"duplicate","prompt_token_ids":[1],"step_count":1,"step_count":2}]}"#,
        )
        .unwrap();
        assert!(
            load_fixture_case(&fixture, None)
                .unwrap_err()
                .contains("duplicate JSON key")
        );
        fs::write(
            &fixture,
            br#"{"cases":[{"case_id":"unknown","prompt_token_ids":[1],"step_count":1,"extra":true}]}"#,
        )
        .unwrap();
        assert!(
            load_fixture_case(&fixture, None)
                .unwrap_err()
                .contains("unknown field")
        );
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
        let value = serde_json::to_value(ok_result(Some(test_audit())).unwrap()).unwrap();
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
    fn raw_adapter_contract_has_exact_root_and_link_fields() {
        let value = serde_json::to_value(ok_result(Some(test_audit())).unwrap()).unwrap();
        let actual = value
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        let expected = [
            "schema_version",
            "raw_target_schema_version",
            "scope",
            "status",
            "immutable_status",
            "case_id",
            "case_sha256",
            "identity",
            "requested_m",
            "resolved_m",
            "actual_token_batch_width",
            "actual_request_batch_width",
            "timing",
            "audit",
            "lifecycle",
            "reset",
            "outcome",
            "oom",
            "fallback",
            "preflight",
            "failure",
            "links",
            "adapter",
        ]
        .into_iter()
        .collect::<BTreeSet<_>>();
        assert_eq!(actual, expected);
        assert_eq!(value["adapter"]["target_schema_version"], RAW_TARGET_SCHEMA);
        assert_eq!(value["adapter"]["raw_v2_requires_role_aware_adapter"], true);
        assert_eq!(value["links"]["timing"]["json_pointer"], "/timing");
        assert_eq!(value["links"]["audit"]["json_pointer"], "/audit");
        assert_eq!(value["links"]["audit"]["sha256"], "a".repeat(64));
        assert_eq!(
            value["identity"]["runtime_device"]["requested_device_index"],
            1
        );
        assert_eq!(value["identity"]["runtime_device"]["observed_device_id"], 0);
        assert_eq!(
            value["identity"]["runtime_device"]["observed_architecture"],
            "gfx1201"
        );
    }

    #[test]
    fn ok_result_requires_operation_audit() {
        assert!(ok_result(None).unwrap_err().contains("mandatory"));
    }

    #[test]
    fn ok_terminal_evidence_requires_exact_widths_lifecycle_and_reset() {
        let bindings = test_bindings();
        let lifecycle = test_lifecycle();
        let reset = test_reset();
        assert!(
            validate_terminal_evidence(
                &bindings,
                Some(8),
                Some(8),
                Some(1),
                Some(&lifecycle),
                Some(&reset),
                true,
            )
            .is_ok()
        );
        let mut unbalanced = lifecycle.clone();
        unbalanced["prepare"] = Value::from(3);
        assert!(
            validate_terminal_evidence(
                &bindings,
                Some(8),
                Some(8),
                Some(1),
                Some(&unbalanced),
                Some(&reset),
                true,
            )
            .is_err()
        );
        let mut incomplete_reset = lifecycle;
        incomplete_reset["reset"]["attempted"] = Value::from(0);
        assert!(
            validate_terminal_evidence(
                &bindings,
                Some(8),
                Some(8),
                Some(1),
                Some(&incomplete_reset),
                Some(&reset),
                true,
            )
            .is_err()
        );
        assert!(
            validate_terminal_evidence(
                &bindings,
                None,
                Some(8),
                Some(1),
                Some(&test_lifecycle()),
                Some(&reset),
                true,
            )
            .is_err()
        );
    }

    #[test]
    fn case_workload_and_device_are_exact() {
        let mut case = test_case_contract();
        let fixture = FixtureCase {
            case_id: "public-fixture-0".into(),
            prompt_token_ids: vec![1, 2],
            step_count: 1,
        };
        assert!(
            validate_case_workload(
                &case,
                &fixture,
                "public-fixture-0",
                &"1".repeat(64),
                8,
                "cold_batched",
                8,
                "AQ4_0",
                "qwen35_aq4_rdna4_v1",
            )
            .is_ok()
        );
        case.request_count = 2;
        assert!(
            validate_case_workload(
                &case,
                &fixture,
                "public-fixture-0",
                &"1".repeat(64),
                8,
                "cold_batched",
                8,
                "AQ4_0",
                "qwen35_aq4_rdna4_v1",
            )
            .is_err()
        );

        let case_device = test_case_contract().device;
        let runtime = test_identity().runtime_device;
        assert!(validate_case_device_values(&case_device, &runtime, "gfx1201").is_ok());
        let mut same_arch_other_gpu = runtime.clone();
        same_arch_other_gpu.observed_name = "Different gfx1201 GPU".into();
        assert!(
            validate_case_device_values(&case_device, &same_arch_other_gpu, "gfx1201").is_err()
        );
    }

    #[test]
    fn environment_values_must_all_equal_one() {
        let names = vec!["A".to_string(), "B".to_string()];
        assert!(validate_required_environment_with(&names, |_| Some("1".into())).is_ok());
        assert!(
            validate_required_environment_with(&names, |name| {
                (name == "A").then(|| "1".to_string())
            })
            .is_err()
        );
    }

    #[test]
    fn preflight_input_requires_raw_v2_exact_fields() {
        let valid = serde_json::json!({
            "weights_bytes": 1,
            "persistent_state_bytes": 2,
            "kv_cache_bytes": 3,
            "workspace_bytes": 4,
            "temporary_bytes": 5,
            "vram_headroom_bytes": 6,
            "gpu_process_snapshot": [
                {"pid": 7, "process_name": "benchmark", "vram_bytes": 8}
            ]
        });
        assert!(validate_preflight_input(&valid).is_ok());
        let mut invalid = valid;
        invalid.as_object_mut().unwrap().remove("workspace_bytes");
        assert!(validate_preflight_input(&invalid).is_err());
    }

    #[test]
    fn benchmark_binary_cannot_impersonate_worker() {
        let digest = "a".repeat(64);
        assert!(
            validate_binary_role_values(
                Path::new("/tmp/same"),
                &digest,
                Path::new("/tmp/same"),
                &"b".repeat(64)
            )
            .is_err()
        );
        assert!(
            validate_binary_role_values(
                Path::new("/tmp/benchmark"),
                &digest,
                Path::new("/tmp/worker"),
                &digest
            )
            .is_err()
        );
    }

    #[test]
    fn oom_failure_is_immutable_and_keeps_preflight() {
        assert!(is_oom("hipErrorOutOfMemory while allocating"));
        let result = finalize_result(
            &test_args(),
            &test_bindings(),
            Some(test_identity()),
            test_preflight(true),
            TimingFacts {
                request_elapsed_ms: 2.0,
                generation: None,
                generated_tokens: 0,
            },
            None,
            empty_fallback(),
            None,
            None,
            None,
            None,
            None,
            None,
            "oom",
            Some(("model_load", "runtime_out_of_memory", true)),
        )
        .unwrap();
        let value = serde_json::to_value(result).unwrap();
        assert_eq!(value["immutable_status"], true);
        assert_eq!(value["oom"]["stage"], "model_load");
        assert_eq!(value["preflight"]["input"]["vram_headroom_bytes"], 1);
    }

    #[test]
    fn failed_status_is_atomically_published_and_returns_non_success() {
        let root = env::temp_dir().join(format!(
            "ullm-aq4-p2-failure-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let mut args = test_args();
        args.output = root.join("result.json");
        let success = publish_failure_result(
            &args,
            &test_bindings(),
            Some(test_identity()),
            test_preflight(true),
            Instant::now(),
            "model_load",
            "resident_model_load_failed",
            false,
        )
        .unwrap();
        assert!(!success);
        let value: Value = serde_json::from_slice(&fs::read(&args.output).unwrap()).unwrap();
        assert_eq!(value["status"], "failed");
        assert_eq!(value["immutable_status"], true);
        assert_eq!(value["failure"]["stage"], "model_load");
        assert_eq!(
            value["failure"]["reason_code"],
            "resident_model_load_failed"
        );
        assert!(value["oom"].is_null());
        assert_eq!(value["preflight"]["input"]["vram_headroom_bytes"], 1);
        assert!(write_atomic_json(&args.output, &value).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn fallback_uses_audited_invocation_counts_and_reasons() {
        let counts = BTreeMap::from([("fallback.impl", 7_u64), ("primary.impl", 9_u64)]);
        let resolutions = vec![
            FallbackResolution {
                unavailable_primary: "primary.impl".into(),
                resolved_implementation: "fallback.impl".into(),
            },
            FallbackResolution {
                unavailable_primary: "primary.impl".into(),
                resolved_implementation: "fallback.impl".into(),
            },
        ];
        let facts = fallback_facts_from_sources(&counts, &resolutions, false);
        assert_eq!(facts.count, 7);
        assert_eq!(facts.unexpected_count, 7);
        assert_eq!(
            facts.reasons,
            vec![FallbackReason {
                unavailable_primary: "primary.impl".into(),
                resolved_implementation: "fallback.impl".into(),
                invocation_count: 7,
            }]
        );
        assert_eq!(
            fallback_facts_from_sources(&counts, &resolutions, true).unexpected_count,
            0
        );
    }

    #[test]
    fn package_tree_hash_matches_streaming_contract() {
        let root = env::temp_dir().join(format!(
            "ullm-aq4-p2-tree-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("b"), b"B").unwrap();
        fs::write(root.join("a"), b"A").unwrap();
        let identity = package_tree_identity(&root).unwrap();
        let mut expected = Sha256::new();
        for (name, payload) in [("a", b"A".as_slice()), ("b", b"B".as_slice())] {
            expected.update(name.as_bytes());
            expected.update(b"\0");
            expected.update(Sha256::digest(payload));
            expected.update(b"\n");
        }
        assert_eq!(identity.sha256, hex_digest(expected.finalize()));
        assert_eq!(identity.file_count, 2);
        assert_eq!(identity.bytes, 2);
        fs::remove_dir_all(root).unwrap();
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
