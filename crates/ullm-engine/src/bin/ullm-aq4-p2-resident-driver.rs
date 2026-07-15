// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Resident Qwen3.5 AQ4 P2 batch driver.
//!
//! Stdout is reserved for the bounded NDJSON protocol. The R9700 lock is owned by the outer
//! runner; this process validates the selected device but never acquires a second lock.

use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::env;
use std::ffi::OsString;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Component, Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use ullm_engine::aq4_worker_backend::QWEN35_AQ4_REQUIRED_HIP_KERNEL_ENV;
use ullm_engine::backend_operation_registry::{OperationExecutionRecord, OperationExecutionStatus};
use ullm_engine::execution_batch::ExecutionPhase;
use ullm_engine::qwen35_aq4_head_runtime::PackageLmHeadMode;
use ullm_engine::qwen35_aq4_model_runtime::{
    QWEN35_AQ4_KV_BLOCK_SIZE, Qwen35Aq4ModelLoadConfig, Qwen35Aq4ModelRuntime,
};
use ullm_engine::qwen35_aq4_session::{
    QWEN35_AQ4_PREFILL_CHUNK_GRID, QWEN35_AQ4_ROPE_BASE, QWEN35_AQ4_ROTARY_DIM,
};
use ullm_engine::served_model::{ServedModel, load_served_model};

const PROTOCOL: &str = "ullm.aq4_p2_resident_driver.v2";
const MAX_LINE_BYTES: usize = 4 * 1024 * 1024;
const MAX_INPUT_BYTES: usize = 64 * 1024 * 1024;
const MAX_PACKAGE_FILES: usize = 65_536;
const MAX_PACKAGE_DEPTH: usize = 32;
const MAX_WORKER_RELEASE_ENTRIES: usize = 4096;
const MAX_WORKER_RELEASE_DEPTH: usize = 8;
const HASH_CHUNK_BYTES: usize = 1024 * 1024;
const O_NOFOLLOW: i32 = 0o00400000;
const O_CLOEXEC: i32 = 0o02000000;
const WORKER_HARDLINK_FIXTURE_RAW: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/fixtures/aq4-p2-resident-worker-hardlinks/active-production.json"
));
#[cfg(test)]
const CASE_DEVICE_FIXTURE_RAW: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/fixtures/aq4-p2-resident-case-device/active-production.json"
));
const DEFAULT_CHUNK_BYTES: usize = 1024 * 1024;
const DEFAULT_LM_HEAD_CHUNK_ROWS: usize = 8192;
const TOTAL_RUNS: usize = 12;
const PREFLIGHT_FIELDS: &[&str] = &[
    "weights_bytes",
    "persistent_state_bytes",
    "kv_cache_bytes",
    "workspace_bytes",
    "temporary_bytes",
    "vram_headroom_bytes",
    "gpu_process_snapshot",
];

#[derive(Debug, Clone)]
struct Args {
    served_model_manifest: PathBuf,
    device_index: u32,
    build_git_commit: String,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct WorkerHardlinkFixture {
    schema_version: String,
    roots: Vec<PathBuf>,
    paths: Vec<PathBuf>,
    primary_path: PathBuf,
    sha256: String,
    expected: WorkerFileIdentity,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct WorkerFileIdentity {
    device: u64,
    inode: u64,
    uid: u32,
    gid: u32,
    mode: u32,
    size: u64,
    nlink: u64,
    mtime_ns: i128,
    ctime_ns: i128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WorkerComponentIdentity {
    device: u64,
    inode: u64,
    uid: u32,
    gid: u32,
    mode: u32,
}

#[derive(Debug, Clone)]
struct WorkerHardlinkGuard {
    fixture: WorkerHardlinkFixture,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Link {
    path: String,
    sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct RuntimeDevice {
    runtime_device_index: u32,
    device_id: String,
    backend: String,
    name: String,
    architecture: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct DriverIdentity {
    binary_sha256: String,
    build_git_commit: String,
    protocol: String,
    worker_binary_sha256: String,
    package_manifest_sha256: String,
    package_content_sha256: String,
    served_model_manifest_sha256: String,
    model_id: String,
    model_revision: String,
    format_id: String,
    implementation_id: String,
    runtime_device: RuntimeDevice,
    guard_set_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct Sampling {
    mode: String,
    temperature: f64,
    top_p: f64,
    top_k: usize,
    seed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Control {
    control_id: String,
    role: String,
    format_id: String,
    implementation_id: String,
    promotion_eligible: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct Execution {
    scope: String,
    phase: String,
    mode: String,
    prompt_tokens: usize,
    cached_prefix_tokens: usize,
    context_tokens: usize,
    generated_tokens: usize,
    request_count: usize,
    requested_m: usize,
    resolved_m: usize,
    sampling: Sampling,
    control: Control,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "command", deny_unknown_fields)]
enum Command {
    #[serde(rename = "case_begin")]
    CaseBegin {
        schema_version: String,
        case_id: String,
        case_sha256: String,
        case_binding: Link,
        identity: Link,
        preflight: Link,
        policy: Link,
        fixture: Link,
        execution: Execution,
    },
    #[serde(rename = "run")]
    Run {
        schema_version: String,
        case_id: String,
        run_index: usize,
        run_kind: String,
    },
    #[serde(rename = "case_end")]
    CaseEnd {
        schema_version: String,
        case_id: String,
    },
    #[serde(rename = "cancel")]
    Cancel {
        schema_version: String,
        case_id: String,
        reason: String,
    },
    #[serde(rename = "shutdown")]
    Shutdown { schema_version: String },
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct P2Case {
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
    device: P2Device,
    control_id: String,
    control: Control,
    sampling: Sampling,
    format_id: String,
    implementation_id: String,
    path_oracle_case_id: Option<String>,
    path_oracle_result_sha256: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct P2Device {
    device_id: String,
    backend: String,
    name: String,
    architecture: String,
    runtime_device_index: i32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FixtureFile {
    #[serde(default)]
    schema_version: Option<String>,
    cases: Vec<FixtureCase>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct FixtureCase {
    case_id: String,
    prompt_token_ids: Vec<usize>,
    step_count: usize,
}

#[derive(Debug, Clone)]
struct ActiveCase {
    case_id: String,
    case_sha256: String,
    execution: Execution,
    prompt_token_ids: Vec<usize>,
    preflight: Value,
    next_run: usize,
    failed: bool,
}

#[derive(Debug)]
struct ExecutionFacts {
    elapsed_ms: f64,
    prefill_ms: f64,
    decode_ms: f64,
    generated_tokens: usize,
    actual_token_batch_width: usize,
    audit_sha256: String,
    operation_count: u64,
    state_sha256: String,
}

trait ResidentExecutor {
    fn execute(&mut self, case: &ActiveCase) -> Result<ExecutionFacts, String>;
    fn reset(&mut self) -> Result<(), String>;
    fn baseline_clean(&self) -> bool;
}

struct RealExecutor {
    model: Qwen35Aq4ModelRuntime,
    clean: bool,
}

impl RealExecutor {
    fn load(model: &ServedModel, package_dir: PathBuf, device_index: u32) -> Result<Self, String> {
        let profile = model.profile_snapshot();
        let config = Qwen35Aq4ModelLoadConfig {
            package_dir,
            device_index,
            expected_architecture: Some(profile.device),
            chunk_bytes: DEFAULT_CHUNK_BYTES,
            context_length: profile.context_length,
            kv_block_size: QWEN35_AQ4_KV_BLOCK_SIZE,
            layer_indices: None,
            lm_head_mode: PackageLmHeadMode::GpuResidentF32,
            lm_head_chunk_rows: DEFAULT_LM_HEAD_CHUNK_ROWS,
        };
        Ok(Self {
            model: Qwen35Aq4ModelRuntime::load(config)?,
            clean: true,
        })
    }
}

impl ResidentExecutor for RealExecutor {
    fn execute(&mut self, case: &ActiveCase) -> Result<ExecutionFacts, String> {
        if !self.clean {
            return Err("resident model baseline is not clean".into());
        }
        self.clean = false;
        let started = Instant::now();
        let mut audit = Sha256::new();
        audit.update(b"ullm-aq4-p2-resident-run-v2\0");
        audit.update(case.case_sha256.as_bytes());
        let mut operation_count = 0_u64;
        let mut actual_width = 0usize;
        let prefill_started = Instant::now();
        let mut offset = 0usize;
        while offset < case.prompt_token_ids.len() {
            let width = case
                .execution
                .resolved_m
                .min(case.prompt_token_ids.len() - offset);
            let label = format!("resident-{}-prefill-{offset}", case.case_id);
            if width == 1 {
                let step = self.model.dispatch_token_for_phase(
                    case.prompt_token_ids[offset],
                    QWEN35_AQ4_ROTARY_DIM,
                    QWEN35_AQ4_ROPE_BASE,
                    offset,
                    offset,
                    ExecutionPhase::ColdPrefill,
                    false,
                    &label,
                )?;
                record_pairs(
                    &mut audit,
                    &step.operation_executions,
                    width,
                    &mut operation_count,
                )?;
            } else {
                let step = self.model.dispatch_prefill_chunk_for_phase(
                    &case.prompt_token_ids[offset..offset + width],
                    QWEN35_AQ4_ROTARY_DIM,
                    QWEN35_AQ4_ROPE_BASE,
                    offset,
                    ExecutionPhase::ColdPrefill,
                    false,
                    &label,
                )?;
                if step.execution_width != width {
                    return Err("resident prefill execution width fell back".into());
                }
                for invocation in step.invocations {
                    if invocation.execution_width != width {
                        return Err("resident prefill invocation width fell back".into());
                    }
                    record_pair(&mut audit, &invocation.records, width, &mut operation_count)?;
                }
            }
            self.model.synchronize()?;
            actual_width = actual_width.max(width);
            offset += width;
        }
        let prefill_ms = elapsed_ms(prefill_started);
        let decode_started = Instant::now();
        let mut generated = Vec::with_capacity(case.execution.generated_tokens);
        if case.execution.generated_tokens > 0 {
            generated.push(top_token(&mut self.model, &case.case_id)?);
            for step in 1..case.execution.generated_tokens {
                let position = case.prompt_token_ids.len() + step - 1;
                let label = format!("resident-{}-decode-{step}", case.case_id);
                let dispatch = self.model.dispatch_token_for_phase(
                    generated[step - 1],
                    QWEN35_AQ4_ROTARY_DIM,
                    QWEN35_AQ4_ROPE_BASE,
                    position,
                    position,
                    ExecutionPhase::Decode,
                    false,
                    &label,
                )?;
                record_pairs(
                    &mut audit,
                    &dispatch.operation_executions,
                    1,
                    &mut operation_count,
                )?;
                self.model.synchronize()?;
                generated.push(top_token(&mut self.model, &case.case_id)?);
            }
        }
        let decode_ms = elapsed_ms(decode_started);
        if actual_width != case.execution.resolved_m {
            return Err(format!(
                "resident actual width {actual_width} differs from resolved M {}",
                case.execution.resolved_m
            ));
        }
        let audit_sha256 = format!("{:x}", audit.finalize());
        let mut state = Sha256::new();
        state.update(b"ullm-aq4-p2-resident-state-v1\0");
        state.update(case.case_sha256.as_bytes());
        for token in generated {
            state.update((token as u64).to_le_bytes());
        }
        Ok(ExecutionFacts {
            elapsed_ms: elapsed_ms(started),
            prefill_ms,
            decode_ms,
            generated_tokens: case.execution.generated_tokens,
            actual_token_batch_width: actual_width,
            audit_sha256,
            operation_count,
            state_sha256: format!("{:x}", state.finalize()),
        })
    }

    fn reset(&mut self) -> Result<(), String> {
        self.model.reset_all_request_state_synchronized()?;
        self.clean = true;
        Ok(())
    }

    fn baseline_clean(&self) -> bool {
        self.clean
    }
}

fn record_pairs(
    digest: &mut Sha256,
    pairs: &[[OperationExecutionRecord; 2]],
    width: usize,
    count: &mut u64,
) -> Result<(), String> {
    for pair in pairs {
        record_pair(digest, pair, width, count)?;
    }
    Ok(())
}

fn record_pair(
    digest: &mut Sha256,
    pair: &[OperationExecutionRecord; 2],
    width: usize,
    count: &mut u64,
) -> Result<(), String> {
    for record in pair {
        if record.status != OperationExecutionStatus::Succeeded {
            return Err("resident operation did not succeed".into());
        }
        digest.update((width as u64).to_le_bytes());
        digest.update(record.implementation_id.as_bytes());
        digest.update([0]);
        digest.update(format!("{:?}", record.phase).as_bytes());
        *count = count
            .checked_add(1)
            .ok_or_else(|| "resident operation count overflows".to_string())?;
    }
    Ok(())
}

fn top_token(model: &mut Qwen35Aq4ModelRuntime, case_id: &str) -> Result<usize, String> {
    let top = model.top_logits_from_last_layer(1, case_id)?;
    let top = top
        .first()
        .ok_or_else(|| "resident LM head returned no top token".to_string())?;
    if !top.logit.is_finite() {
        return Err("resident LM head returned non-finite top logit".into());
    }
    Ok(top.token_id)
}

struct ResidentDriver<E> {
    session_id: String,
    identity: DriverIdentity,
    executor: E,
    active: Option<ActiveCase>,
    seen_cases: BTreeSet<String>,
}

impl<E: ResidentExecutor> ResidentDriver<E> {
    fn new(session_id: String, identity: DriverIdentity, executor: E) -> Self {
        Self {
            session_id,
            identity,
            executor,
            active: None,
            seen_cases: BTreeSet::new(),
        }
    }

    fn ready(&self) -> Value {
        json!({
            "event": "ready", "schema_version": PROTOCOL, "model_loads": 1,
            "resident_session_id": self.session_id, "driver_identity": self.identity,
        })
    }

    fn begin(&mut self, command: BeginFields) -> Result<Value, String> {
        require_protocol(&command.schema_version)?;
        if self.active.is_some() || !self.executor.baseline_clean() {
            return Err("case_begin requires an idle clean baseline".into());
        }
        if !self.seen_cases.insert(command.case_id.clone()) {
            return Err("duplicate or reused case_id".into());
        }
        let active = load_active_case(&command, &self.identity)?;
        let event = json!({
            "event": "case_ready", "schema_version": PROTOCOL,
            "resident_session_id": self.session_id, "case_id": active.case_id,
            "requested_m": active.execution.requested_m,
            "resolved_m": active.execution.resolved_m, "baseline_clean": true,
        });
        self.active = Some(active);
        Ok(event)
    }

    fn run(
        &mut self,
        schema: &str,
        case_id: &str,
        run_index: usize,
        run_kind: &str,
    ) -> Result<(Value, bool), String> {
        require_protocol(schema)?;
        let active = self
            .active
            .as_mut()
            .ok_or_else(|| "run has no active case".to_string())?;
        if active.case_id != case_id || active.next_run != run_index || run_index >= TOTAL_RUNS {
            return Err("run case/order/reuse differs".into());
        }
        let expected_kind = if run_index < 2 { "warmup" } else { "measured" };
        if run_kind != expected_kind || active.failed {
            return Err("run kind or failed-case reuse differs".into());
        }
        let result = self.executor.execute(active).and_then(|facts| {
            if facts.actual_token_batch_width != active.execution.resolved_m {
                Err("resident executor reported an M fallback".to_string())
            } else {
                Ok(facts)
            }
        });
        let execution_error = result.as_ref().err().cloned();
        let oom = execution_error.as_deref().is_some_and(is_oom);
        let hip_fault = execution_error.as_deref().is_some_and(is_hip_fault);
        let reset = self.executor.reset();
        let reset_ok = reset.is_ok() && self.executor.baseline_clean();
        let reuse_forbidden = oom || hip_fault || !reset_ok;
        let status = if oom {
            "oom"
        } else if execution_error.is_some() || !reset_ok {
            "failed"
        } else {
            "ok"
        };
        let facts = result.ok();
        active.next_run += 1;
        active.failed = status != "ok";
        let reason = execution_error
            .as_deref()
            .or_else(|| reset.as_ref().err().map(String::as_str))
            .unwrap_or("");
        let elapsed = facts.as_ref().map_or(0.0, |facts| facts.elapsed_ms);
        let actual_width = facts.as_ref().map(|facts| facts.actual_token_batch_width);
        let audit = facts.as_ref().map(|facts| {
            json!({
                "coverage_complete": facts.operation_count > 0,
                "deterministic_digest_sha256": facts.audit_sha256,
                "physical_operation_invocations": facts.operation_count,
            })
        });
        let timing = facts.as_ref().map(|facts| {
            json!({
                "prefill_ms": facts.prefill_ms, "decode_ms": facts.decode_ms,
                "end_to_end_ms": facts.elapsed_ms, "generated_tokens": facts.generated_tokens,
            })
        });
        let state = facts.as_ref().map(|facts| {
            json!({
                "baseline_before": true, "baseline_after": reset_ok,
                "request_state_sha256": facts.state_sha256,
            })
        });
        let lifecycle = json!({
            "prepare": 1, "commit": usize::from(status == "ok"),
            "discard": usize::from(status != "ok"), "error": usize::from(status != "ok"),
            "cancel": 0, "reset": {"attempted": 1, "complete": usize::from(reset_ok), "failed": usize::from(!reset_ok)},
        });
        let event = json!({
            "event": "run_complete", "schema_version": PROTOCOL,
            "resident_session_id": self.session_id, "case_id": case_id,
            "run_index": run_index, "run_kind": run_kind, "status": status,
            "elapsed_ms": elapsed, "requested_m": active.execution.requested_m,
            "resolved_m": active.execution.resolved_m,
            "actual_token_batch_width": actual_width,
            "actual_request_batch_width": if status == "ok" { Some(active.execution.request_count) } else { None },
            "timing": timing, "audit": audit, "state": state, "lifecycle": lifecycle,
            "reset": {"attempted": 1, "complete": usize::from(reset_ok), "failed": usize::from(!reset_ok)},
            "resource": {"samples": [{"monotonic_ms": elapsed}], "peak": resource_peak(&active.preflight)},
            "terminal": {"reuse_forbidden": reuse_forbidden, "reason_code": classify_reason(reason), "oom": oom, "hip_fault": hip_fault},
        });
        Ok((event, reuse_forbidden))
    }

    fn end(&mut self, schema: &str, case_id: &str) -> Result<Value, String> {
        require_protocol(schema)?;
        let active = self
            .active
            .take()
            .ok_or_else(|| "case_end has no active case".to_string())?;
        if active.case_id != case_id {
            self.active = Some(active);
            return Err("case_end case_id differs".into());
        }
        if !active.failed && active.next_run != TOTAL_RUNS {
            self.active = Some(active);
            return Err("case_end rejected incomplete schedule".into());
        }
        if !self.executor.baseline_clean() {
            return Err("case_end baseline is not clean".into());
        }
        Ok(json!({
            "event": "case_complete", "schema_version": PROTOCOL,
            "resident_session_id": self.session_id, "case_id": case_id,
            "release": {"commit": usize::from(!active.failed), "discard": usize::from(active.failed), "reset": 1, "baseline_restored": true},
        }))
    }

    fn cancel(&mut self, schema: &str, case_id: &str, reason: &str) -> Result<Value, String> {
        require_protocol(schema)?;
        if reason.is_empty() || reason.len() > 128 {
            return Err("cancel reason is invalid".into());
        }
        let active = self
            .active
            .take()
            .ok_or_else(|| "cancel has no active case".to_string())?;
        if active.case_id != case_id {
            self.active = Some(active);
            return Err("cancel case_id differs".into());
        }
        let reset_ok = self.executor.reset().is_ok() && self.executor.baseline_clean();
        Ok(json!({
            "event": "cancel_complete", "schema_version": PROTOCOL,
            "resident_session_id": self.session_id, "case_id": case_id,
            "release": {"commit": 0, "discard": 1, "reset": usize::from(reset_ok), "baseline_restored": reset_ok},
            "terminal": {"reuse_forbidden": !reset_ok, "reason_code": "cancelled", "oom": false, "hip_fault": false},
        }))
    }
}

#[derive(Debug)]
struct BeginFields {
    schema_version: String,
    case_id: String,
    case_sha256: String,
    case_binding: Link,
    identity: Link,
    preflight: Link,
    policy: Link,
    fixture: Link,
    execution: Execution,
}

fn load_active_case(
    command: &BeginFields,
    identity: &DriverIdentity,
) -> Result<ActiveCase, String> {
    let case_root = load_link(&command.case_binding, "case binding")?;
    let case_value = case_root
        .get("cases")
        .and_then(Value::as_array)
        .and_then(|cases| {
            cases
                .iter()
                .find(|case| case.get("case_id").and_then(Value::as_str) == Some(&command.case_id))
        })
        .cloned()
        .ok_or_else(|| "case binding does not contain case_id".to_string())?;
    if case_value.get("case_sha256").and_then(Value::as_str) != Some(&command.case_sha256)
        || json_self_hash(&case_value, "case_sha256")? != command.case_sha256
    {
        return Err("case binding self-hash differs".into());
    }
    let case: P2Case = serde_json::from_value(case_value)
        .map_err(|error| format!("case exact schema rejected: {error}"))?;
    let identity_value = load_link(&command.identity, "identity")?;
    validate_identity(&identity_value, identity, &command.case_binding.sha256)?;
    let preflight = load_link(&command.preflight, "preflight")?;
    validate_preflight(&preflight)?;
    let policy = load_link(&command.policy, "policy")?;
    if policy.get("schema_version").and_then(Value::as_str)
        != Some("ullm.aq4_production_p2_threshold_policy.v1")
        || policy.get("status").and_then(Value::as_str) != Some("bound")
    {
        return Err("threshold policy is not bound v1".into());
    }
    let fixture_value = load_link(&command.fixture, "fixture")?;
    let fixture: FixtureFile = serde_json::from_value(fixture_value)
        .map_err(|error| format!("fixture exact schema rejected: {error}"))?;
    if fixture
        .schema_version
        .as_deref()
        .is_some_and(|value| value != "ullm.aq4_p2_case_fixture.v1")
    {
        return Err("fixture schema differs".into());
    }
    let fixture = fixture
        .cases
        .into_iter()
        .find(|item| item.case_id == command.case_id)
        .ok_or_else(|| "fixture case is missing".to_string())?;
    validate_case(&case, &fixture, command, identity)?;
    Ok(ActiveCase {
        case_id: command.case_id.clone(),
        case_sha256: command.case_sha256.clone(),
        execution: command.execution.clone(),
        prompt_token_ids: fixture.prompt_token_ids,
        preflight,
        next_run: 0,
        failed: false,
    })
}

fn validate_case(
    case: &P2Case,
    fixture: &FixtureCase,
    command: &BeginFields,
    identity: &DriverIdentity,
) -> Result<(), String> {
    let execution = &command.execution;
    let expected_mode = if execution.requested_m == 1 {
        "all_m1"
    } else {
        "cold_batched"
    };
    let expected_resolved = if expected_mode == "all_m1" {
        1
    } else {
        execution.requested_m
    };
    if case.case_id != command.case_id
        || case.fixture_id != fixture.case_id
        || case.case_sha256 != command.case_sha256
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
        || case.prefill_requested_m != execution.requested_m
        || case.resolved_m != expected_resolved
        || case.request_count != 1
        || case.decode_request_count != 0
        || case.generated_tokens != fixture.step_count
        || case.control_id != "aq4_0_target"
        || case.control != execution.control
        || case.sampling != execution.sampling
        || case.format_id != identity.format_id
        || case.implementation_id != identity.implementation_id
        || execution.scope != case.scope
        || execution.phase != case.phase
        || execution.mode != case.mode
        || execution.prompt_tokens != case.prompt_tokens
        || execution.cached_prefix_tokens != case.cached_prefix_tokens
        || execution.context_tokens != case.context_tokens
        || execution.generated_tokens != case.generated_tokens
        || execution.request_count != case.request_count
        || execution.resolved_m != case.resolved_m
        || case.device.device_id != identity.runtime_device.device_id
        || case.device.backend != identity.runtime_device.backend
        || case.device.name != identity.runtime_device.name
        || case.device.architecture != identity.runtime_device.architecture
        || case.device.runtime_device_index != identity.runtime_device.runtime_device_index as i32
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
        || !QWEN35_AQ4_PREFILL_CHUNK_GRID.contains(&execution.requested_m)
        || (case.mode == "all_m1"
            && (case.path_oracle_case_id.is_some() || case.path_oracle_result_sha256.is_some()))
        || (case.mode != "all_m1"
            && (case
                .path_oracle_case_id
                .as_deref()
                .is_none_or(str::is_empty)
                || case.path_oracle_result_sha256.is_some()))
    {
        return Err("case workload/control/device binding differs".into());
    }
    Ok(())
}

fn validate_identity(
    value: &Value,
    ready: &DriverIdentity,
    case_binding_sha256: &str,
) -> Result<(), String> {
    if value.get("schema_version").and_then(Value::as_str)
        != Some("ullm.aq4_production_p2_identity.v2")
        || value.get("status").and_then(Value::as_str) != Some("bound")
    {
        return Err("identity is not bound v2".into());
    }
    let declared = value
        .get("identity_sha256")
        .and_then(Value::as_str)
        .filter(|value| valid_sha256(value))
        .ok_or_else(|| "identity self-hash is missing".to_string())?;
    if json_self_hash(value, "identity_sha256")? != declared {
        return Err("identity self-hash differs".into());
    }
    let resident: DriverIdentity = serde_json::from_value(
        value
            .get("resident_driver_identity")
            .cloned()
            .ok_or_else(|| "resident identity is missing".to_string())?,
    )
    .map_err(|error| format!("resident identity schema rejected: {error}"))?;
    if &resident != ready {
        return Err("resident ready identity differs from bound identity".into());
    }
    let hashes = value
        .get("hash_binding")
        .and_then(Value::as_object)
        .ok_or_else(|| "identity hash binding is missing".to_string())?;
    if value
        .get("expanded_manifest_sha256")
        .and_then(Value::as_str)
        != Some(case_binding_sha256)
        || hashes
            .get("bound_case_manifest_sha256")
            .and_then(Value::as_str)
            != Some(case_binding_sha256)
    {
        return Err("identity does not bind the case manifest".into());
    }
    for (field, actual) in [
        ("worker_binary_sha256", ready.worker_binary_sha256.as_str()),
        (
            "package_manifest_sha256",
            ready.package_manifest_sha256.as_str(),
        ),
        (
            "package_content_sha256",
            ready.package_content_sha256.as_str(),
        ),
        (
            "served_model_manifest_sha256",
            ready.served_model_manifest_sha256.as_str(),
        ),
    ] {
        if hashes.get(field).and_then(Value::as_str) != Some(actual) {
            return Err(format!("identity {field} differs"));
        }
    }
    Ok(())
}

fn validate_preflight(value: &Value) -> Result<(), String> {
    let object = value
        .as_object()
        .ok_or_else(|| "preflight must be an object".to_string())?;
    if object.keys().map(String::as_str).collect::<BTreeSet<_>>()
        != PREFLIGHT_FIELDS.iter().copied().collect()
    {
        return Err("preflight fields differ".into());
    }
    for field in PREFLIGHT_FIELDS
        .iter()
        .copied()
        .filter(|field| *field != "gpu_process_snapshot")
    {
        if object.get(field).and_then(Value::as_u64).is_none() {
            return Err(format!("preflight {field} is invalid"));
        }
    }
    let processes = object["gpu_process_snapshot"]
        .as_array()
        .ok_or_else(|| "preflight process snapshot is invalid".to_string())?;
    for process in processes {
        let process = process
            .as_object()
            .ok_or_else(|| "preflight process is invalid".to_string())?;
        if process.keys().map(String::as_str).collect::<BTreeSet<_>>()
            != ["pid", "process_name", "vram_bytes"].into_iter().collect()
            || process.get("pid").and_then(Value::as_u64).is_none()
            || process
                .get("process_name")
                .and_then(Value::as_str)
                .is_none_or(str::is_empty)
            || process.get("vram_bytes").and_then(Value::as_u64).is_none()
        {
            return Err("preflight process fields differ".into());
        }
    }
    Ok(())
}

fn resource_peak(preflight: &Value) -> Value {
    json!({
        "vram_used_bytes": Value::Null,
        "workspace_bytes": preflight.get("workspace_bytes").cloned().unwrap_or(Value::Null),
        "temporary_bytes": preflight.get("temporary_bytes").cloned().unwrap_or(Value::Null),
    })
}

impl WorkerHardlinkGuard {
    fn capture(worker_path: &Path, expected_sha256: &str) -> Result<Self, String> {
        let fixture: WorkerHardlinkFixture = serde_json::from_str(WORKER_HARDLINK_FIXTURE_RAW)
            .map_err(|error| format!("worker hardlink fixture rejected: {error}"))?;
        validate_worker_hardlink_set(&fixture, worker_path, expected_sha256, || {})?;
        Ok(Self { fixture })
    }

    fn verify(&self, worker_path: &Path, expected_sha256: &str) -> Result<(), String> {
        validate_worker_hardlink_set(&self.fixture, worker_path, expected_sha256, || {})
    }
}

fn worker_file_identity(metadata: &fs::Metadata) -> WorkerFileIdentity {
    WorkerFileIdentity {
        device: metadata.dev(),
        inode: metadata.ino(),
        uid: metadata.uid(),
        gid: metadata.gid(),
        mode: metadata.mode(),
        size: metadata.len(),
        nlink: metadata.nlink(),
        mtime_ns: i128::from(metadata.mtime()) * 1_000_000_000 + i128::from(metadata.mtime_nsec()),
        ctime_ns: i128::from(metadata.ctime()) * 1_000_000_000 + i128::from(metadata.ctime_nsec()),
    }
}

fn worker_component_identity(metadata: &fs::Metadata) -> WorkerComponentIdentity {
    WorkerComponentIdentity {
        device: metadata.dev(),
        inode: metadata.ino(),
        uid: metadata.uid(),
        gid: metadata.gid(),
        mode: metadata.mode(),
    }
}

fn snapshot_no_symlink_components(
    path: &Path,
    label: &str,
) -> Result<Vec<(PathBuf, WorkerComponentIdentity)>, String> {
    require_absolute_normal_path(path, label)?;
    let mut current = PathBuf::new();
    let mut snapshots = Vec::new();
    for component in path.components() {
        match component {
            Component::RootDir | Component::Normal(_) => current.push(component.as_os_str()),
            _ => return Err(format!("{label} path is not canonical absolute form")),
        }
        let metadata = fs::symlink_metadata(&current)
            .map_err(|error| format!("{label} component metadata failed: {error}"))?;
        if metadata.file_type().is_symlink() {
            return Err(format!("{label} traverses a symlink"));
        }
        snapshots.push((current.clone(), worker_component_identity(&metadata)));
    }
    Ok(snapshots)
}

fn open_worker_nofollow(path: &Path, label: &str) -> Result<File, String> {
    OpenOptions::new()
        .read(true)
        .custom_flags(O_NOFOLLOW | O_CLOEXEC)
        .open(path)
        .map_err(|error| format!("{label} O_NOFOLLOW open failed: {error}"))
}

fn hash_open_worker(file: &mut File, label: &str) -> Result<(String, u64), String> {
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; HASH_CHUNK_BYTES];
    let mut total = 0u64;
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|error| format!("{label} FD hash failed: {error}"))?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
        total = total
            .checked_add(count as u64)
            .ok_or_else(|| format!("{label} byte count overflow"))?;
    }
    Ok((format!("{:x}", digest.finalize()), total))
}

fn scan_worker_inode_paths(
    root: &Path,
    expected: WorkerFileIdentity,
    label: &str,
) -> Result<BTreeSet<PathBuf>, String> {
    let root_metadata = fs::symlink_metadata(root)
        .map_err(|error| format!("{label} root metadata failed: {error}"))?;
    if !root_metadata.is_dir() || root_metadata.file_type().is_symlink() {
        return Err(format!("{label} root is not a no-symlink directory"));
    }
    let mut pending = vec![(root.to_path_buf(), 0usize)];
    let mut visited = 0usize;
    let mut matches = BTreeSet::new();
    while let Some((directory, depth)) = pending.pop() {
        if depth > MAX_WORKER_RELEASE_DEPTH {
            return Err(format!("{label} scan depth exceeds bound"));
        }
        for entry in
            fs::read_dir(&directory).map_err(|error| format!("{label} read_dir failed: {error}"))?
        {
            let path = entry
                .map_err(|error| format!("{label} entry failed: {error}"))?
                .path();
            visited = visited
                .checked_add(1)
                .ok_or_else(|| format!("{label} entry count overflow"))?;
            if visited > MAX_WORKER_RELEASE_ENTRIES {
                return Err(format!("{label} scan entry count exceeds bound"));
            }
            let metadata = fs::symlink_metadata(&path)
                .map_err(|error| format!("{label} entry metadata failed: {error}"))?;
            if metadata.file_type().is_symlink() {
                return Err(format!("{label} scan encountered a symlink"));
            }
            if metadata.is_dir() {
                pending.push((path, depth + 1));
            } else if metadata.is_file() {
                if metadata.dev() == expected.device && metadata.ino() == expected.inode {
                    matches.insert(path);
                }
            } else {
                return Err(format!("{label} scan encountered a non-regular entry"));
            }
        }
    }
    Ok(matches)
}

fn validate_worker_hardlink_set<F>(
    fixture: &WorkerHardlinkFixture,
    worker_path: &Path,
    expected_sha256: &str,
    mutation_hook: F,
) -> Result<(), String>
where
    F: FnOnce(),
{
    if fixture.schema_version != "ullm.aq4_p2_resident_worker_link_identity.v2"
        || !valid_sha256(&fixture.sha256)
        || fixture.sha256 != expected_sha256
        || fixture.expected.nlink == 0
        || fixture.paths.len() as u64 != fixture.expected.nlink
        || fixture.roots.is_empty()
        || fixture.paths.is_empty()
    {
        return Err("worker link fixture identity differs".into());
    }
    let root_set = fixture.roots.iter().cloned().collect::<BTreeSet<_>>();
    let path_set = fixture.paths.iter().cloned().collect::<BTreeSet<_>>();
    if root_set.len() != fixture.roots.len()
        || path_set.len() != fixture.paths.len()
        || fixture.paths.first() != Some(&fixture.primary_path)
        || worker_path != fixture.primary_path
    {
        return Err("worker link paths/roots differ".into());
    }
    for root in &fixture.roots {
        require_absolute_normal_path(root, "worker scan root")?;
    }
    for path in &fixture.paths {
        require_absolute_normal_path(path, "worker declared path")?;
        if !fixture
            .roots
            .iter()
            .any(|root| path != root && path.starts_with(root))
        {
            return Err("worker declared path is outside scan roots".into());
        }
    }
    let component_paths = fixture
        .roots
        .iter()
        .map(|path| (path, "worker scan root"))
        .chain(
            fixture
                .paths
                .iter()
                .map(|path| (path, "worker declared path")),
        )
        .collect::<Vec<_>>();
    let component_before = component_paths
        .iter()
        .map(|(path, label)| snapshot_no_symlink_components(path, label))
        .collect::<Result<Vec<_>, _>>()?;
    let mut files = Vec::with_capacity(fixture.paths.len());
    for path in &fixture.paths {
        let metadata = fs::symlink_metadata(path)
            .map_err(|error| format!("worker declared path metadata failed: {error}"))?;
        if !metadata.is_file()
            || metadata.file_type().is_symlink()
            || worker_file_identity(&metadata) != fixture.expected
            || metadata.mode() & 0o111 == 0
            || metadata.mode() & 0o002 != 0
        {
            return Err("worker link pre-open metadata differs".into());
        }
        let mut file = open_worker_nofollow(path, "worker declared path")?;
        if worker_file_identity(
            &file
                .metadata()
                .map_err(|error| format!("worker declared path FD metadata failed: {error}"))?,
        ) != fixture.expected
        {
            return Err("worker link open metadata differs".into());
        }
        let (digest, bytes) = hash_open_worker(&mut file, "worker declared path")?;
        if digest != fixture.sha256 || bytes != fixture.expected.size {
            return Err("worker link FD hash differs".into());
        }
        files.push(file);
    }
    for root in &fixture.roots {
        let expected_paths = path_set
            .iter()
            .filter(|path| path.starts_with(root))
            .cloned()
            .collect::<BTreeSet<_>>();
        if scan_worker_inode_paths(root, fixture.expected, "worker scan root")? != expected_paths {
            return Err("worker link path coverage differs".into());
        }
    }
    mutation_hook();
    let component_after = component_paths
        .iter()
        .map(|(path, label)| snapshot_no_symlink_components(path, label))
        .collect::<Result<Vec<_>, _>>()?;
    if component_before != component_after {
        return Err("worker link component identity changed during validation".into());
    }
    for (path, file) in fixture.paths.iter().zip(files.iter()) {
        let metadata = fs::symlink_metadata(path)
            .map_err(|error| format!("worker declared path post metadata failed: {error}"))?;
        let fd_metadata = file
            .metadata()
            .map_err(|error| format!("worker declared path FD post metadata failed: {error}"))?;
        if worker_file_identity(&metadata) != fixture.expected
            || worker_file_identity(&fd_metadata) != fixture.expected
        {
            return Err("worker link identity changed during validation".into());
        }
    }
    for root in &fixture.roots {
        let expected_paths = path_set
            .iter()
            .filter(|path| path.starts_with(root))
            .cloned()
            .collect::<BTreeSet<_>>();
        if scan_worker_inode_paths(root, fixture.expected, "worker scan root post")?
            != expected_paths
        {
            return Err("worker link identity changed during validation".into());
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    match parse_args(env::args_os().skip(1)).and_then(run) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("ullm-aq4-p2-resident-driver: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: Args) -> Result<(), String> {
    let (model, identity, package_dir, worker_guard) = startup(&args)?;
    let executor = RealExecutor::load(&model, package_dir, args.device_index)?;
    worker_guard.verify(&model.worker.binary, &model.worker.binary_sha256)?;
    let session_id = session_id(&identity);
    let mut driver = ResidentDriver::new(session_id, identity, executor);
    write_event(&driver.ready())?;
    let stdin = std::io::stdin();
    let mut reader = BufReader::new(stdin.lock());
    let mut line = Vec::new();
    loop {
        line.clear();
        let read = reader
            .by_ref()
            .take((MAX_LINE_BYTES + 1) as u64)
            .read_until(b'\n', &mut line)
            .map_err(|error| format!("protocol read failed: {error}"))?;
        if read == 0 {
            if driver.active.is_some() {
                return Err("EOF with active case".into());
            }
            return Ok(());
        }
        if read > MAX_LINE_BYTES || line.last() != Some(&b'\n') {
            return Err("protocol record is oversized or unterminated".into());
        }
        let value = parse_strict_json(&line, "protocol command")?;
        let command: Command = serde_json::from_value(value)
            .map_err(|error| format!("protocol command schema rejected: {error}"))?;
        let (event, terminal) = match command {
            Command::CaseBegin {
                schema_version,
                case_id,
                case_sha256,
                case_binding,
                identity,
                preflight,
                policy,
                fixture,
                execution,
            } => {
                let fields = BeginFields {
                    schema_version,
                    case_id,
                    case_sha256,
                    case_binding,
                    identity,
                    preflight,
                    policy,
                    fixture,
                    execution,
                };
                (Some(driver.begin(fields)?), false)
            }
            Command::Run {
                schema_version,
                case_id,
                run_index,
                run_kind,
            } => {
                let (event, terminal) =
                    driver.run(&schema_version, &case_id, run_index, &run_kind)?;
                (Some(event), terminal)
            }
            Command::CaseEnd {
                schema_version,
                case_id,
            } => (Some(driver.end(&schema_version, &case_id)?), false),
            Command::Cancel {
                schema_version,
                case_id,
                reason,
            } => {
                let event = driver.cancel(&schema_version, &case_id, &reason)?;
                let terminal = event["terminal"]["reuse_forbidden"].as_bool() == Some(true);
                (Some(event), terminal)
            }
            Command::Shutdown { schema_version } => {
                require_protocol(&schema_version)?;
                if driver.active.is_some() {
                    return Err("shutdown rejected with active case".into());
                }
                return Ok(());
            }
        };
        if let Some(event) = event {
            write_event(&event)?;
        }
        if terminal {
            return Err("resident process became non-reusable".into());
        }
    }
}

fn startup(
    args: &Args,
) -> Result<(ServedModel, DriverIdentity, PathBuf, WorkerHardlinkGuard), String> {
    require_absolute_normal_path(&args.served_model_manifest, "served model manifest")?;
    if args.build_git_commit.len() != 40
        || !args
            .build_git_commit
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err("build git commit must be 40 lowercase hex characters".into());
    }
    let model = load_served_model(&args.served_model_manifest)
        .map_err(|error| format!("served model rejected: {error}"))?;
    validate_model(&model)?;
    validate_required_environment(&model.worker.required_environment)?;
    let served_sha = sha256_file(&args.served_model_manifest)?;
    if served_sha != model.manifest_sha256 {
        return Err("served manifest hash differs".into());
    }
    let worker_guard =
        WorkerHardlinkGuard::capture(&model.worker.binary, &model.worker.binary_sha256)?;
    let package_dir = model
        .product
        .root
        .join(&model.product.package.manifest_path)
        .parent()
        .ok_or_else(|| "package manifest has no parent".to_string())?
        .to_path_buf();
    let package = package_tree_identity(&package_dir)?;
    let observed = ullm_runtime_sys::device_info(args.device_index)
        .map_err(|error| format!("runtime device query failed: {error}"))?;
    if observed.gcn_arch_name != model.worker.identity.device
        || observed.gcn_arch_name != "gfx1201"
        || observed.backend.is_empty()
        || observed.name.is_empty()
        || observed.device_id < 0
    {
        return Err("runtime device is not exact served R9700/gfx1201 identity".into());
    }
    let binary = env::current_exe()
        .map_err(|error| format!("current executable failed: {error}"))?
        .canonicalize()
        .map_err(|error| format!("current executable canonicalization failed: {error}"))?;
    let identity = DriverIdentity {
        binary_sha256: sha256_file(&binary)?,
        build_git_commit: args.build_git_commit.clone(),
        protocol: PROTOCOL.into(),
        worker_binary_sha256: model.worker.binary_sha256.clone(),
        package_manifest_sha256: model.product.package.manifest_sha256.clone(),
        package_content_sha256: package.sha256,
        served_model_manifest_sha256: model.manifest_sha256.clone(),
        model_id: model.public.id.clone(),
        model_revision: model.public.revision.clone(),
        format_id: model.format.format_id.clone(),
        implementation_id: model.format.implementation_id.clone(),
        runtime_device: RuntimeDevice {
            runtime_device_index: args.device_index,
            device_id: "r9700-rdna4".into(),
            backend: observed.backend,
            name: observed.name,
            architecture: observed.gcn_arch_name,
        },
        guard_set_sha256: guard_set_sha256(&model.worker.required_environment)?,
    };
    Ok((model, identity, package_dir, worker_guard))
}

fn validate_model(model: &ServedModel) -> Result<(), String> {
    if model.format.format_id != "AQ4_0"
        || model.format.implementation_id != "qwen35_aq4_rdna4_v1"
        || model.worker.identity.device != "gfx1201"
        || model.worker.identity.execution_profile != "rdna4_aq4_resident"
        || model.product.artifact.is_some()
        || model.product.package.manifest_path.is_empty()
    {
        return Err("served model is not exact AQ4 resident contract".into());
    }
    let mut actual = model
        .worker
        .required_environment
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    actual.sort_unstable();
    let mut expected = QWEN35_AQ4_REQUIRED_HIP_KERNEL_ENV.to_vec();
    expected.sort_unstable();
    if actual != expected {
        return Err("required guard set differs".into());
    }
    Ok(())
}

fn validate_required_environment(names: &[String]) -> Result<(), String> {
    for name in names {
        if env::var_os(name).as_deref() != Some(std::ffi::OsStr::new("1")) {
            return Err(format!("required environment {name} must equal 1"));
        }
    }
    Ok(())
}

fn guard_set_sha256(names: &[String]) -> Result<String, String> {
    let mut names = names.to_vec();
    names.sort();
    let mut digest = Sha256::new();
    digest.update(b"ullm-aq4-p2-resident-guards-v1\0");
    for name in names {
        if env::var_os(&name).as_deref() != Some(std::ffi::OsStr::new("1")) {
            return Err(format!("guard {name} is not enabled"));
        }
        digest.update(name.as_bytes());
        digest.update(b"=1\n");
    }
    Ok(format!("{:x}", digest.finalize()))
}

struct PackageIdentity {
    sha256: String,
}

fn package_tree_identity(root: &Path) -> Result<PackageIdentity, String> {
    let root = root
        .canonicalize()
        .map_err(|error| format!("package root failed: {error}"))?;
    if !fs::symlink_metadata(&root)
        .map_err(|error| format!("package root metadata failed: {error}"))?
        .is_dir()
    {
        return Err("package root is not a directory".into());
    }
    let mut pending = vec![(root.clone(), 0usize)];
    let mut files = Vec::new();
    while let Some((directory, depth)) = pending.pop() {
        if depth > MAX_PACKAGE_DEPTH {
            return Err("package depth exceeds limit".into());
        }
        for entry in
            fs::read_dir(directory).map_err(|error| format!("package read_dir failed: {error}"))?
        {
            let path = entry
                .map_err(|error| format!("package entry failed: {error}"))?
                .path();
            let metadata = fs::symlink_metadata(&path)
                .map_err(|error| format!("package metadata failed: {error}"))?;
            if metadata.file_type().is_symlink() {
                return Err("package symlink rejected".into());
            }
            if metadata.is_dir() {
                pending.push((path, depth + 1));
            } else if metadata.is_file() {
                files.push(path);
                if files.len() > MAX_PACKAGE_FILES {
                    return Err("package file count exceeds limit".into());
                }
            } else {
                return Err("package non-regular entry rejected".into());
            }
        }
    }
    files.sort();
    if files.is_empty() {
        return Err("package tree is empty".into());
    }
    let mut aggregate = Sha256::new();
    for path in files {
        let relative = path
            .strip_prefix(&root)
            .expect("walked package path")
            .to_string_lossy()
            .replace(std::path::MAIN_SEPARATOR, "/");
        let digest = sha256_file_bytes(&path)?.0;
        aggregate.update(relative.as_bytes());
        aggregate.update(b"\0");
        aggregate.update(digest);
        aggregate.update(b"\n");
    }
    Ok(PackageIdentity {
        sha256: format!("{:x}", aggregate.finalize()),
    })
}

fn load_link(link: &Link, label: &str) -> Result<Value, String> {
    if !valid_sha256(&link.sha256) {
        return Err(format!("{label} link SHA is invalid"));
    }
    let path = Path::new(&link.path);
    require_absolute_normal_path(path, label)?;
    let bytes = read_regular(path, label, MAX_INPUT_BYTES)?;
    if sha256_bytes(&bytes) != link.sha256 {
        return Err(format!("{label} link SHA differs"));
    }
    parse_strict_json(&bytes, label)
}

fn require_absolute_normal_path(path: &Path, label: &str) -> Result<(), String> {
    if !path.is_absolute()
        || path
            .components()
            .any(|component| matches!(component, Component::ParentDir))
    {
        return Err(format!(
            "{label} path must be absolute without parent traversal"
        ));
    }
    Ok(())
}

fn read_regular(path: &Path, label: &str, maximum: usize) -> Result<Vec<u8>, String> {
    let before =
        fs::symlink_metadata(path).map_err(|error| format!("{label} metadata failed: {error}"))?;
    if !before.file_type().is_file() || before.nlink() != 1 || before.len() > maximum as u64 {
        return Err(format!("{label} must be bounded single-link regular file"));
    }
    let mut file = File::open(path).map_err(|error| format!("{label} open failed: {error}"))?;
    let opened = file
        .metadata()
        .map_err(|error| format!("{label} fd metadata failed: {error}"))?;
    if file_identity(&before) != file_identity(&opened) {
        return Err(format!("{label} changed while opening"));
    }
    let mut bytes = Vec::with_capacity(before.len() as usize);
    Read::by_ref(&mut file)
        .take((maximum + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("{label} read failed: {error}"))?;
    let after = fs::symlink_metadata(path)
        .map_err(|error| format!("{label} post-read metadata failed: {error}"))?;
    if bytes.len() > maximum
        || file_identity(&before) != file_identity(&after)
        || file_identity(&before)
            != file_identity(
                &file
                    .metadata()
                    .map_err(|error| format!("{label} fd post-read failed: {error}"))?,
            )
    {
        return Err(format!("{label} changed or exceeded bound"));
    }
    Ok(bytes)
}

fn file_identity(metadata: &fs::Metadata) -> (u64, u64, u64, u32, i64, i64, i64, i64, u64) {
    (
        metadata.dev(),
        metadata.ino(),
        metadata.len(),
        metadata.mode(),
        metadata.mtime(),
        metadata.mtime_nsec(),
        metadata.ctime(),
        metadata.ctime_nsec(),
        metadata.nlink(),
    )
}
fn sha256_file(path: &Path) -> Result<String, String> {
    Ok(sha256_file_bytes(path)?
        .0
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}
fn sha256_file_bytes(path: &Path) -> Result<([u8; 32], u64), String> {
    let metadata =
        fs::symlink_metadata(path).map_err(|error| format!("hash metadata failed: {error}"))?;
    if !metadata.is_file() || metadata.nlink() != 1 {
        return Err("hash input must be single-link regular file".into());
    }
    let mut file = File::open(path).map_err(|error| format!("hash open failed: {error}"))?;
    if file_identity(&metadata)
        != file_identity(
            &file
                .metadata()
                .map_err(|error| format!("hash fd failed: {error}"))?,
        )
    {
        return Err("hash input changed while opening".into());
    }
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; HASH_CHUNK_BYTES];
    let mut count = 0u64;
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| format!("hash read failed: {error}"))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
        count = count
            .checked_add(read as u64)
            .ok_or_else(|| "hash bytes overflow".to_string())?;
    }
    let after = fs::symlink_metadata(path)
        .map_err(|error| format!("hash post metadata failed: {error}"))?;
    if file_identity(&metadata) != file_identity(&after)
        || file_identity(&metadata)
            != file_identity(
                &file
                    .metadata()
                    .map_err(|error| format!("hash fd post failed: {error}"))?,
            )
    {
        return Err("hash input changed".into());
    }
    Ok((digest.finalize().into(), count))
}
fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn json_self_hash(value: &Value, field: &str) -> Result<String, String> {
    let mut value = value.clone();
    let object = value
        .as_object_mut()
        .ok_or_else(|| "self-hash root must be object".to_string())?;
    if !object.contains_key(field) {
        return Err(format!("self-hash lacks {field}"));
    }
    object.insert(field.into(), Value::Null);
    Ok(sha256_bytes(&serde_json::to_vec(&value).map_err(
        |error| format!("self-hash serialization failed: {error}"),
    )?))
}

struct StrictValue(Value);
impl<'de> Deserialize<'de> for StrictValue {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(StrictVisitor)
    }
}
struct StrictVisitor;
impl<'de> Visitor<'de> for StrictVisitor {
    type Value = StrictValue;
    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("strict JSON")
    }
    fn visit_bool<E>(self, v: bool) -> Result<Self::Value, E> {
        Ok(StrictValue(Value::Bool(v)))
    }
    fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E> {
        Ok(StrictValue(Value::Number(v.into())))
    }
    fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E> {
        Ok(StrictValue(Value::Number(v.into())))
    }
    fn visit_f64<E: de::Error>(self, v: f64) -> Result<Self::Value, E> {
        serde_json::Number::from_f64(v)
            .map(Value::Number)
            .map(StrictValue)
            .ok_or_else(|| E::custom("non-finite"))
    }
    fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
        self.visit_string(v.to_string())
    }
    fn visit_string<E>(self, v: String) -> Result<Self::Value, E> {
        Ok(StrictValue(Value::String(v)))
    }
    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(StrictValue(Value::Null))
    }
    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(StrictValue(Value::Null))
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut a: A) -> Result<Self::Value, A::Error> {
        let mut v = Vec::new();
        while let Some(StrictValue(x)) = a.next_element()? {
            v.push(x);
        }
        Ok(StrictValue(Value::Array(v)))
    }
    fn visit_map<A: MapAccess<'de>>(self, mut a: A) -> Result<Self::Value, A::Error> {
        let mut v = serde_json::Map::new();
        while let Some((k, StrictValue(x))) = a.next_entry::<String, StrictValue>()? {
            if v.insert(k.clone(), x).is_some() {
                return Err(de::Error::custom(format!("duplicate JSON key: {k}")));
            }
        }
        Ok(StrictValue(Value::Object(v)))
    }
}
fn parse_strict_json(bytes: &[u8], label: &str) -> Result<Value, String> {
    let mut d = serde_json::Deserializer::from_slice(bytes);
    let value = StrictValue::deserialize(&mut d)
        .map_err(|error| format!("{label} JSON rejected: {error}"))?;
    d.end()
        .map_err(|error| format!("{label} trailing JSON: {error}"))?;
    Ok(value.0)
}

fn parse_args(args: impl IntoIterator<Item = OsString>) -> Result<Args, String> {
    let mut args = args.into_iter();
    let mut manifest = None;
    let mut index = 1u32;
    let mut commit = None;
    while let Some(arg) = args.next() {
        match arg.to_str() {
            Some("--served-model-manifest") => {
                manifest = Some(PathBuf::from(
                    args.next()
                        .ok_or("--served-model-manifest requires value")?,
                ));
            }
            Some("--device-index") => {
                index = args
                    .next()
                    .ok_or("--device-index requires value")?
                    .to_str()
                    .ok_or("device index UTF-8")?
                    .parse()
                    .map_err(|_| "invalid device index")?;
            }
            Some("--build-git-commit") => {
                commit = Some(
                    args.next()
                        .ok_or("--build-git-commit requires value")?
                        .into_string()
                        .map_err(|_| "build commit UTF-8")?,
                );
            }
            Some("--help") | Some("-h") => {
                return Err("Usage: ullm-aq4-p2-resident-driver --served-model-manifest PATH --device-index N --build-git-commit SHA".into());
            }
            Some(other) => return Err(format!("unknown argument {other}")),
            None => return Err("arguments must be UTF-8".into()),
        }
    }
    Ok(Args {
        served_model_manifest: manifest.ok_or("--served-model-manifest is required")?,
        device_index: index,
        build_git_commit: commit.ok_or("--build-git-commit is required")?,
    })
}
fn require_protocol(value: &str) -> Result<(), String> {
    if value == PROTOCOL {
        Ok(())
    } else {
        Err("protocol schema differs".into())
    }
}
fn elapsed_ms(start: Instant) -> f64 {
    (start.elapsed().as_secs_f64() * 1000.0).max(0.001)
}
fn is_oom(error: &str) -> bool {
    let e = error.to_ascii_lowercase();
    e.contains("out of memory")
        || e.contains("hiperroroutofmemory")
        || e.contains("memory allocation")
        || e.contains("bad_alloc")
}
fn is_hip_fault(error: &str) -> bool {
    let e = error.to_ascii_lowercase();
    e.contains("hip") || e.contains("gfx") || e.contains("device fault")
}
fn classify_reason(error: &str) -> &'static str {
    if error.is_empty() {
        "none"
    } else if is_oom(error) {
        "runtime_out_of_memory"
    } else if is_hip_fault(error) {
        "hip_runtime_fault"
    } else if error.contains("reset") {
        "reset_failed"
    } else {
        "execution_failed"
    }
}
fn session_id(identity: &DriverIdentity) -> String {
    let mut d = Sha256::new();
    d.update(b"ullm-resident-session-v2\0");
    d.update(identity.binary_sha256.as_bytes());
    d.update(std::process::id().to_le_bytes());
    format!("{:x}", d.finalize())
}
fn write_event(value: &Value) -> Result<(), String> {
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    serde_json::to_writer(&mut out, value)
        .map_err(|error| format!("protocol serialization failed: {error}"))?;
    out.write_all(b"\n")
        .map_err(|error| format!("protocol write failed: {error}"))?;
    out.flush()
        .map_err(|error| format!("protocol flush failed: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct ActiveCaseDeviceFixture {
        schema_version: String,
        case: P2Case,
        execution: Execution,
        runtime_device: RuntimeDevice,
        source_device: P2Device,
    }

    #[derive(Default)]
    struct MockExecutor {
        clean: bool,
        run_widths: Vec<usize>,
        fail: Option<String>,
        reset_fail: bool,
        actual_width: Option<usize>,
    }
    impl ResidentExecutor for MockExecutor {
        fn execute(&mut self, case: &ActiveCase) -> Result<ExecutionFacts, String> {
            if !self.clean {
                return Err("state leak".into());
            }
            self.clean = false;
            self.run_widths.push(case.execution.resolved_m);
            if let Some(error) = self.fail.clone() {
                return Err(error);
            }
            Ok(ExecutionFacts {
                elapsed_ms: 1.0,
                prefill_ms: 0.5,
                decode_ms: 0.5,
                generated_tokens: case.execution.generated_tokens,
                actual_token_batch_width: self.actual_width.unwrap_or(case.execution.resolved_m),
                audit_sha256: "a".repeat(64),
                operation_count: 2,
                state_sha256: "b".repeat(64),
            })
        }
        fn reset(&mut self) -> Result<(), String> {
            if self.reset_fail {
                return Err("reset failed".into());
            }
            self.clean = true;
            Ok(())
        }
        fn baseline_clean(&self) -> bool {
            self.clean
        }
    }
    fn identity() -> DriverIdentity {
        DriverIdentity {
            binary_sha256: "a".repeat(64),
            build_git_commit: "b".repeat(40),
            protocol: PROTOCOL.into(),
            worker_binary_sha256: "c".repeat(64),
            package_manifest_sha256: "d".repeat(64),
            package_content_sha256: "e".repeat(64),
            served_model_manifest_sha256: "f".repeat(64),
            model_id: "m".into(),
            model_revision: "r".into(),
            format_id: "AQ4_0".into(),
            implementation_id: "qwen35_aq4_rdna4_v1".into(),
            runtime_device: RuntimeDevice {
                runtime_device_index: 1,
                device_id: "r9700-rdna4".into(),
                backend: "HIP".into(),
                name: "GPU".into(),
                architecture: "gfx1201".into(),
            },
            guard_set_sha256: "1".repeat(64),
        }
    }

    fn active_case_device_fixture() -> ActiveCaseDeviceFixture {
        serde_json::from_str(CASE_DEVICE_FIXTURE_RAW).expect("active case device fixture")
    }

    fn case_begin_fields(fixture: &ActiveCaseDeviceFixture) -> BeginFields {
        let link = Link {
            path: "/fixture/not-opened-by-validate-case.json".into(),
            sha256: "0".repeat(64),
        };
        BeginFields {
            schema_version: PROTOCOL.into(),
            case_id: fixture.case.case_id.clone(),
            case_sha256: fixture.case.case_sha256.clone(),
            case_binding: link.clone(),
            identity: link.clone(),
            preflight: link.clone(),
            policy: link.clone(),
            fixture: link,
            execution: fixture.execution.clone(),
        }
    }

    fn production_identity(fixture: &ActiveCaseDeviceFixture) -> DriverIdentity {
        let mut value = identity();
        value.runtime_device = fixture.runtime_device.clone();
        value
    }

    #[test]
    fn production_case_device_uses_exact_runtime_vocabulary() {
        let fixture = active_case_device_fixture();
        assert_eq!(
            fixture.schema_version,
            "ullm.aq4_p2_resident_case_device_identity.v1"
        );
        assert_eq!(fixture.source_device.architecture, "RDNA4");
        assert_eq!(fixture.source_device.backend, "hip");
        assert_eq!(fixture.case.device.architecture, "gfx1201");
        assert_eq!(fixture.runtime_device.architecture, "gfx1201");

        let command = case_begin_fields(&fixture);
        let case_fixture = FixtureCase {
            case_id: fixture.case.fixture_id.clone(),
            prompt_token_ids: vec![1; fixture.case.prompt_tokens],
            step_count: fixture.case.generated_tokens,
        };
        validate_case(
            &fixture.case,
            &case_fixture,
            &command,
            &production_identity(&fixture),
        )
        .expect("active production case and runtime device must bind exactly");
    }

    #[test]
    fn case_device_rejects_gfx_rdna_vocabulary_swap() {
        let mut fixture = active_case_device_fixture();
        fixture.case.device.architecture = fixture.source_device.architecture.clone();
        let command = case_begin_fields(&fixture);
        let case_fixture = FixtureCase {
            case_id: fixture.case.fixture_id.clone(),
            prompt_token_ids: vec![1; fixture.case.prompt_tokens],
            step_count: fixture.case.generated_tokens,
        };
        assert_eq!(
            validate_case(
                &fixture.case,
                &case_fixture,
                &command,
                &production_identity(&fixture),
            ),
            Err("case workload/control/device binding differs".into())
        );
    }

    #[test]
    fn case_device_rejects_case_identity_mismatch() {
        let mut fixture = active_case_device_fixture();
        fixture.case.device.device_id = "different-device".into();
        let command = case_begin_fields(&fixture);
        let case_fixture = FixtureCase {
            case_id: fixture.case.fixture_id.clone(),
            prompt_token_ids: vec![1; fixture.case.prompt_tokens],
            step_count: fixture.case.generated_tokens,
        };
        assert_eq!(
            validate_case(
                &fixture.case,
                &case_fixture,
                &command,
                &production_identity(&fixture),
            ),
            Err("case workload/control/device binding differs".into())
        );
    }

    #[test]
    fn case_device_rejects_runtime_identity_mismatch() {
        let fixture = active_case_device_fixture();
        let command = case_begin_fields(&fixture);
        let case_fixture = FixtureCase {
            case_id: fixture.case.fixture_id.clone(),
            prompt_token_ids: vec![1; fixture.case.prompt_tokens],
            step_count: fixture.case.generated_tokens,
        };
        let mut runtime_identity = production_identity(&fixture);
        runtime_identity.runtime_device.runtime_device_index = 0;
        assert_eq!(
            validate_case(&fixture.case, &case_fixture, &command, &runtime_identity),
            Err("case workload/control/device binding differs".into())
        );
    }

    fn active(id: &str, m: usize) -> ActiveCase {
        ActiveCase {
            case_id: id.into(),
            case_sha256: "2".repeat(64),
            execution: Execution {
                scope: "full_model".into(),
                phase: "cold_prefill".into(),
                mode: if m == 1 {
                    "all_m1".into()
                } else {
                    "cold_batched".into()
                },
                prompt_tokens: 128,
                cached_prefix_tokens: 0,
                context_tokens: 128,
                generated_tokens: 0,
                request_count: 1,
                requested_m: m,
                resolved_m: m,
                sampling: Sampling {
                    mode: "greedy".into(),
                    temperature: 0.0,
                    top_p: 1.0,
                    top_k: 1,
                    seed: 0,
                },
                control: Control {
                    control_id: "aq4_0_target".into(),
                    role: "target".into(),
                    format_id: "AQ4_0".into(),
                    implementation_id: "qwen35_aq4_rdna4_v1".into(),
                    promotion_eligible: true,
                },
            },
            prompt_token_ids: vec![1; 128],
            preflight: json!({"workspace_bytes":1,"temporary_bytes":1}),
            next_run: 0,
            failed: false,
        }
    }

    struct WorkerTestTree {
        directory: PathBuf,
        release: PathBuf,
        deps: PathBuf,
        primary: PathBuf,
        alias: PathBuf,
        fixture: WorkerHardlinkFixture,
    }

    impl Drop for WorkerTestTree {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.directory);
        }
    }

    fn worker_test_tree(exact_two: bool) -> WorkerTestTree {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = env::temp_dir().join(format!(
            "ullm-aq4-worker-hardlinks-{}-{unique}",
            std::process::id()
        ));
        let release = directory.join("release");
        let deps = release.join("deps");
        fs::create_dir_all(&deps).unwrap();
        let primary = release.join("ullm-aq4-worker");
        let alias = deps.join("ullm_aq4_worker-03e49ec754c21dc7");
        fs::write(&primary, b"worker-hardlink-fixture").unwrap();
        fs::set_permissions(&primary, fs::Permissions::from_mode(0o755)).unwrap();
        if exact_two {
            fs::hard_link(&primary, &alias).unwrap();
        }
        let expected = worker_file_identity(&fs::symlink_metadata(&primary).unwrap());
        let paths = if exact_two {
            vec![primary.clone(), alias.clone()]
        } else {
            vec![primary.clone()]
        };
        WorkerTestTree {
            directory,
            release: release.clone(),
            deps: deps.clone(),
            primary: primary.clone(),
            alias,
            fixture: WorkerHardlinkFixture {
                schema_version: "ullm.aq4_p2_resident_worker_link_identity.v2".into(),
                roots: vec![release, deps],
                paths,
                primary_path: primary,
                sha256: sha256_bytes(b"worker-hardlink-fixture"),
                expected,
            },
        }
    }

    fn validate_test_worker(tree: &WorkerTestTree) -> Result<(), String> {
        validate_worker_hardlink_set(&tree.fixture, &tree.primary, &tree.fixture.sha256, || {})
    }

    #[test]
    fn active_production_worker_fixture_is_exact_single_link_set() {
        let fixture: WorkerHardlinkFixture =
            serde_json::from_str(WORKER_HARDLINK_FIXTURE_RAW).unwrap();
        assert_eq!(fixture.paths, vec![fixture.primary_path.clone()]);
        assert_eq!(fixture.expected.nlink, 1);
        validate_worker_hardlink_set(&fixture, &fixture.primary_path, &fixture.sha256, || {})
            .unwrap();
    }

    #[test]
    fn worker_fixture_accepts_exact_single_and_exact_two_link_sets() {
        validate_test_worker(&worker_test_tree(false)).unwrap();
        validate_test_worker(&worker_test_tree(true)).unwrap();
    }

    #[test]
    fn worker_alias_add_remove_and_different_inode_are_rejected() {
        let add = worker_test_tree(false);
        fs::hard_link(&add.primary, add.release.join("unexpected-worker-link")).unwrap();
        assert!(validate_test_worker(&add).is_err());

        let remove = worker_test_tree(true);
        fs::remove_file(&remove.alias).unwrap();
        assert!(validate_test_worker(&remove).is_err());

        let different = worker_test_tree(true);
        fs::remove_file(&different.alias).unwrap();
        fs::copy(&different.primary, &different.alias).unwrap();
        assert!(validate_test_worker(&different).is_err());

        let declaration = worker_test_tree(false);
        let mut rebound = declaration.fixture.clone();
        rebound.paths.push(declaration.alias.clone());
        assert!(
            validate_worker_hardlink_set(&rebound, &declaration.primary, &rebound.sha256, || {},)
                .is_err()
        );
    }

    #[test]
    fn worker_content_metadata_alias_and_root_escape_are_rejected() {
        let content = worker_test_tree(true);
        fs::write(&content.alias, b"changed-worker-content").unwrap();
        assert!(validate_test_worker(&content).is_err());

        let metadata = worker_test_tree(false);
        fs::set_permissions(&metadata.primary, fs::Permissions::from_mode(0o777)).unwrap();
        assert!(validate_test_worker(&metadata).is_err());

        let alias = worker_test_tree(true);
        let wrong_alias = alias.deps.join("unexpected-worker-name");
        fs::rename(&alias.alias, &wrong_alias).unwrap();
        assert!(validate_test_worker(&alias).is_err());

        let escape = worker_test_tree(true);
        let mut escaped_fixture = escape.fixture.clone();
        let escaped_path = escape.directory.join("escaped-worker");
        escaped_fixture.paths[1] = escaped_path.clone();
        fs::rename(&escape.alias, &escaped_path).unwrap();
        assert!(
            validate_worker_hardlink_set(
                &escaped_fixture,
                &escape.primary,
                &escaped_fixture.sha256,
                || {},
            )
            .is_err()
        );
    }

    #[test]
    fn worker_late_link_content_and_path_swaps_are_rejected() {
        let add = worker_test_tree(false);
        let add_primary = add.primary.clone();
        let add_path = add.release.join("late-second-link");
        assert!(
            validate_worker_hardlink_set(&add.fixture, &add.primary, &add.fixture.sha256, || {
                fs::hard_link(add_primary, add_path).unwrap()
            },)
            .is_err()
        );

        let remove = worker_test_tree(true);
        let remove_alias = remove.alias.clone();
        assert!(
            validate_worker_hardlink_set(
                &remove.fixture,
                &remove.primary,
                &remove.fixture.sha256,
                || fs::remove_file(remove_alias).unwrap(),
            )
            .is_err()
        );

        let content = worker_test_tree(true);
        let content_alias = content.alias.clone();
        assert!(
            validate_worker_hardlink_set(
                &content.fixture,
                &content.primary,
                &content.fixture.sha256,
                || fs::write(content_alias, b"late-content-mutation").unwrap(),
            )
            .is_err()
        );

        let swap = worker_test_tree(false);
        let swap_primary = swap.primary.clone();
        let moved = swap.release.join("moved-original-worker");
        assert!(
            validate_worker_hardlink_set(
                &swap.fixture,
                &swap.primary,
                &swap.fixture.sha256,
                || {
                    fs::rename(&swap_primary, &moved).unwrap();
                    fs::copy(&moved, &swap_primary).unwrap();
                },
            )
            .is_err()
        );
    }

    #[test]
    fn cargo_style_hardlink_is_rejected_and_detached_copy_is_accepted() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = env::temp_dir().join(format!(
            "ullm-aq4-resident-driver-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir(&directory).unwrap();
        let cargo_binary = directory.join("target-binary");
        let cargo_deps_binary = directory.join("deps-binary");
        let detached_binary = directory.join("detached-binary");
        fs::write(&cargo_binary, b"resident-driver").unwrap();
        fs::hard_link(&cargo_binary, &cargo_deps_binary).unwrap();
        assert!(sha256_file(&cargo_binary).is_err());
        fs::copy(&cargo_binary, &detached_binary).unwrap();
        assert_eq!(
            sha256_file(&detached_binary).unwrap(),
            sha256_bytes(b"resident-driver")
        );
        assert!(require_absolute_normal_path(&detached_binary, "driver").is_ok());
        assert!(require_absolute_normal_path(Path::new("relative-driver"), "driver").is_err());
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn case_swap_duplicate_unknown_order_and_reuse_are_rejected() {
        let mut d = ResidentDriver::new(
            "s".into(),
            identity(),
            MockExecutor {
                clean: true,
                ..Default::default()
            },
        );
        d.active = Some(active("a", 8));
        assert!(d.run(PROTOCOL, "b", 0, "warmup").is_err());
        assert!(d.run(PROTOCOL, "a", 1, "warmup").is_err());
        assert!(d.run(PROTOCOL, "a", 0, "unknown").is_err());
        let _ = d.run(PROTOCOL, "a", 0, "warmup").unwrap();
        assert!(d.run(PROTOCOL, "a", 0, "warmup").is_err());
    }
    #[test]
    fn runs_restore_baseline_and_change_width_without_state_leak() {
        let mut d = ResidentDriver::new(
            "s".into(),
            identity(),
            MockExecutor {
                clean: true,
                ..Default::default()
            },
        );
        d.active = Some(active("a", 1));
        let (first, _) = d.run(PROTOCOL, "a", 0, "warmup").unwrap();
        assert_eq!(first["actual_token_batch_width"], 1);
        d.active = None;
        d.active = Some(active("b", 128));
        let (second, _) = d.run(PROTOCOL, "b", 0, "warmup").unwrap();
        assert_eq!(second["actual_token_batch_width"], 128);
        assert_eq!(d.executor.run_widths, vec![1, 128]);
        assert!(d.executor.clean);
    }
    #[test]
    fn ready_identity_is_immutable_across_cases() {
        let id = identity();
        let d = ResidentDriver::new(
            "s".into(),
            id.clone(),
            MockExecutor {
                clean: true,
                ..Default::default()
            },
        );
        assert_eq!(
            d.ready()["driver_identity"],
            serde_json::to_value(id).unwrap()
        );
    }
    #[test]
    fn release_failure_and_oom_forbid_reuse() {
        let mut reset = ResidentDriver::new(
            "s".into(),
            identity(),
            MockExecutor {
                clean: true,
                reset_fail: true,
                ..Default::default()
            },
        );
        reset.active = Some(active("a", 8));
        let (event, terminal) = reset.run(PROTOCOL, "a", 0, "warmup").unwrap();
        assert!(terminal);
        assert_eq!(event["terminal"]["reuse_forbidden"], true);
        let mut oom = ResidentDriver::new(
            "s".into(),
            identity(),
            MockExecutor {
                clean: true,
                fail: Some("hipErrorOutOfMemory".into()),
                ..Default::default()
            },
        );
        oom.active = Some(active("o", 8));
        let (event, terminal) = oom.run(PROTOCOL, "o", 0, "warmup").unwrap();
        assert!(terminal);
        assert_eq!(event["status"], "oom");
    }

    #[test]
    fn m_fallback_and_cancel_are_fail_closed_and_restore_baseline() {
        let mut fallback = ResidentDriver::new(
            "s".into(),
            identity(),
            MockExecutor {
                clean: true,
                actual_width: Some(1),
                ..Default::default()
            },
        );
        fallback.active = Some(active("m", 128));
        let (event, terminal) = fallback.run(PROTOCOL, "m", 0, "warmup").unwrap();
        assert!(!terminal);
        assert_eq!(event["status"], "failed");
        assert!(fallback.executor.clean);

        let mut cancel = ResidentDriver::new(
            "s".into(),
            identity(),
            MockExecutor {
                clean: true,
                ..Default::default()
            },
        );
        cancel.active = Some(active("c", 8));
        let event = cancel.cancel(PROTOCOL, "c", "operator_cancel").unwrap();
        assert_eq!(event["release"]["discard"], 1);
        assert_eq!(event["terminal"]["reuse_forbidden"], false);
        assert!(cancel.executor.clean);
    }
    #[test]
    fn strict_protocol_rejects_duplicate_and_unknown_fields() {
        assert!(
            parse_strict_json(br#"{"command":"shutdown","command":"run"}\n"#, "command").is_err()
        );
        let value=parse_strict_json(br#"{"command":"shutdown","schema_version":"ullm.aq4_p2_resident_driver.v2","extra":1}"#,"command").unwrap();
        assert!(serde_json::from_value::<Command>(value).is_err());
    }
}
