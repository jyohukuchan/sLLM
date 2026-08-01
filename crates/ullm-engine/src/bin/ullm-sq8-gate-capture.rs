// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Isolated GPU producer for the frozen SQ8 numerical gate v0.2.
//!
//! This binary has no production-service entry point. It consumes a
//! create-new plan from `prepare-sq8-gate-v0.2-capture.py`, forces the CPU
//! reference token stream, captures requested F32 tensors, and writes a
//! producer manifest for the consumer-side evaluator.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;
use ullm_engine::sq_canonical::read_sq8_canonical_artifact;
use ullm_engine::sq8_embedding_runtime::QWEN3_14B_SQ8_EMBEDDING_REQUIRED_HIP_KERNEL_ENV;
use ullm_engine::sq8_layer_runtime::{
    QWEN3_14B_SQ8_PAGED_REQUIRED_HIP_KERNEL_ENV,
    QWEN3_14B_SQ8_PREFILL_CHUNK_REQUIRED_HIP_KERNEL_ENV, QWEN3_14B_SQ8_REQUIRED_HIP_KERNEL_ENV,
};
use ullm_engine::sq8_model_head_runtime::{
    QWEN3_14B_SQ8_MODEL_HEAD_REQUIRED_HIP_KERNEL_ENV, validate_qwen3_14b_sq8_r9700_device_info,
};
use ullm_engine::sq8_serving_runtime::{
    QWEN3_14B_SQ8_PAGED_DECODE_SPLIT_EXPERIMENT_TILE_ENV, QWEN3_14B_SQ8_SERVING_CONTEXT_TOKENS,
    Qwen3Sq8ServingSession, Sq8ServingPrefillMode, Sq8ServingRuntimeStatus,
    load_qwen3_14b_sq8_serving_norms,
};
use ullm_runtime_sys::{RuntimeContext, RuntimeStream, device_count, device_info};

const PLAN_SCHEMA: &str = "ullm.sq8.gate.v0.2.capture-plan.v1";
const CAPTURE_SCHEMA: &str = "ullm.sq8.gate.v0.2.capture.v1";
const GATE_SCHEMA: &str = "ullm.sq8.numerical_gate.relative_fp32.v0.2";
const EXPECTED_GATE_SHA256: &str =
    "64a43c032570bed8086e3c441b0774cc470c5ab1e8c67f99e02af2b6307f72bf";
const HIDDEN_SIZE: usize = 5_120;
const VOCAB_SIZE: usize = 151_936;
const LAYER_COUNT: usize = 40;
const UPLOAD_CHUNK_BYTES: usize = 16 * 1024 * 1024;
const EXPERIMENTAL_SELECTOR_ENV: [&str; 3] = [
    "ULLM_USE_SQ8_0_FLASH2_STAGED_WAVE32_PROTOTYPE",
    "ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_TILE",
    "ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_ALLOW_MULTITILE",
];

#[derive(Debug)]
struct Options {
    plan: PathBuf,
    output: PathBuf,
}

#[derive(Debug, Deserialize, Serialize)]
struct GateBinding {
    path: String,
    sha256: String,
}

#[derive(Debug, Deserialize)]
struct CapturePlan {
    schema_version: String,
    frozen_gate: GateBinding,
    role: String,
    candidate: CandidateIdentity,
    selector: SelectorPlan,
    artifact: String,
    package: String,
    identity: PlanIdentity,
    cases: Vec<CasePlan>,
}

#[derive(Debug, Deserialize, Serialize)]
struct CandidateIdentity {
    id: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct SelectorPlan {
    enabled: bool,
    kind: String,
    configuration: Value,
    environment: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct PlanIdentity {
    artifact_content_sha256: String,
    fixture_manifest_sha256: String,
    materialized_token_hashes: Value,
    reference_executable_sha256: String,
    reference_identity: Option<Value>,
    teacher_forced_tokens_u32le_sha256: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct CasePlan {
    case_id: String,
    mode: String,
    prompt_token_ids: Vec<usize>,
    teacher_forced_input_tokens: Vec<usize>,
    teacher_forced_tokens_u32le_sha256: String,
    positions: Vec<PositionPlan>,
}

#[derive(Debug, Deserialize)]
struct PositionPlan {
    id: String,
    case_id: String,
    mode: String,
    ordinal: usize,
    phase: String,
    phase_index: usize,
    position: usize,
    input_token_id: usize,
    layer_required: bool,
}

#[derive(Debug, Serialize)]
struct TensorDescriptor {
    path: String,
    sha256: String,
    dtype: &'static str,
    shape: [usize; 1],
    byte_count: usize,
}

#[derive(Debug, Serialize)]
struct CapturedPosition {
    id: String,
    case_id: String,
    mode: String,
    ordinal: usize,
    phase: String,
    phase_index: usize,
    position: usize,
    input_token_id: usize,
    candidate_top1_token_id: usize,
    candidate_top1_logit: f32,
    logits: TensorDescriptor,
    final_hidden: TensorDescriptor,
    layers: Vec<TensorDescriptor>,
}

#[derive(Debug, Serialize)]
struct CaptureManifest {
    schema_version: &'static str,
    role: String,
    frozen_gate: GateBinding,
    candidate: CandidateIdentity,
    selector: SelectorPlan,
    identity: Value,
    producer: Value,
    positions: Vec<CapturedPosition>,
}

fn main() -> Result<(), String> {
    let options = parse_options()?;
    run(options)
}

fn run(options: Options) -> Result<(), String> {
    let capture_started = Instant::now();
    let plan_bytes = fs::read(&options.plan).map_err(|err| {
        format!(
            "failed to read capture plan {}: {err}",
            options.plan.display()
        )
    })?;
    let plan: CapturePlan = serde_json::from_slice(&plan_bytes).map_err(|err| {
        format!(
            "failed to parse capture plan {}: {err}",
            options.plan.display()
        )
    })?;
    validate_plan(&plan)?;
    prepare_output_root(&options.output)?;
    let output = options.output.canonicalize().map_err(|err| {
        format!(
            "failed to canonicalize newly created output {}: {err}",
            options.output.display()
        )
    })?;
    let artifact = read_sq8_canonical_artifact(&plan.artifact)?;
    if artifact.manifest().integrity.content_sha256 != plan.identity.artifact_content_sha256 {
        return Err(format!(
            "artifact content SHA-256 mismatch: plan={} loaded={}",
            plan.identity.artifact_content_sha256,
            artifact.manifest().integrity.content_sha256
        ));
    }
    let runtime_index = isolated_gfx1201_device()?;
    let mut context = RuntimeContext::create(runtime_index)?;
    let mut stream = context.create_stream()?;
    let runner = runner_identity()?;
    let selector_fingerprint = sha256_bytes(
        &serde_json::to_vec(&plan.selector)
            .map_err(|err| format!("failed to serialize selector fingerprint: {err}"))?,
    );
    let mut device_identity = None;
    let mut captured_positions = Vec::new();
    let mut seen_position_ids = HashSet::new();
    let mut mode_timing = Vec::new();
    let mut mode_runtime = Vec::new();

    for mode in ["sequential_m1", "m128_chunks_with_declared_tail"] {
        let mode_cases = plan
            .cases
            .iter()
            .filter(|case| case.mode == mode)
            .collect::<Vec<_>>();
        if mode_cases.is_empty() {
            continue;
        }
        let mode_case_count = mode_cases.len();
        let mode_started = Instant::now();
        let prefill_mode = parse_prefill_mode(mode)?;
        require_hip_kernel_guards(prefill_mode)?;
        let norms = load_qwen3_14b_sq8_serving_norms(&plan.package, UPLOAD_CHUNK_BYTES)
            .map_err(|err| err.to_string())?;
        let mut session = Qwen3Sq8ServingSession::load_with_prefill_mode(
            &mut context,
            &mut stream,
            &artifact,
            &plan.package,
            norms,
            UPLOAD_CHUNK_BYTES,
            prefill_mode,
        )
        .map_err(|err| err.to_string())?;
        if plan.selector.kind == "handwritten_wmma_projection_prototype" {
            session
                .enable_handwritten_wmma_projection_prototype()
                .map_err(|err| err.to_string())?;
        }
        let report = session.load_report();
        let current_device = json!({
            "device_id": report.device.device_id,
            "backend": report.device.backend,
            "name": report.device.name,
            "gcn_arch_name": report.device.gcn_arch_name,
            "compute_major": report.device.compute_major,
            "compute_minor": report.device.compute_minor,
            "total_global_mem": report.device.total_global_mem,
        });
        if let Some(previous) = &device_identity {
            if previous != &current_device {
                return Err("capture device identity changed across prefill modes".into());
            }
        } else {
            device_identity = Some(current_device);
        }
        // A plan intentionally loads the same physical R9700 once for each
        // required prefill mode.  Keep mode-specific implementation metadata
        // as provenance, but do not mistake it for mutable device identity.
        mode_runtime.push(json!({
            "mode": mode,
            "prefill_mode": format!("{prefill_mode:?}"),
            "prefill_implementation": report.prefill_implementation,
            "paged_decode_split_source_tile": report.paged_decode_split_source_tile,
        }));
        for case in mode_cases {
            capture_case(
                &mut session,
                &mut context,
                &mut stream,
                case,
                &output,
                &mut captured_positions,
                &mut seen_position_ids,
            )?;
        }
        drop(session);
        mode_timing.push(json!({
            "mode": mode,
            "case_count": mode_case_count,
            "elapsed_seconds": mode_started.elapsed().as_secs_f64(),
        }));
    }

    let expected_positions = plan
        .cases
        .iter()
        .map(|case| case.positions.len())
        .sum::<usize>();
    if captured_positions.len() != expected_positions {
        return Err(format!(
            "capture position count mismatch: expected={expected_positions} actual={}",
            captured_positions.len()
        ));
    }
    let identity = json!({
        "artifact_content_sha256": plan.identity.artifact_content_sha256,
        "fixture_manifest_sha256": plan.identity.fixture_manifest_sha256,
        "materialized_token_hashes": plan.identity.materialized_token_hashes,
        "reference_executable_sha256": plan.identity.reference_executable_sha256,
        "reference_identity": plan.identity.reference_identity,
        "teacher_forced_tokens_u32le_sha256": plan.identity.teacher_forced_tokens_u32le_sha256,
        "executable_sha256": runner.binary_sha256,
        "selector_configuration_fingerprint": selector_fingerprint,
        "device_identity": device_identity.ok_or_else(|| "capture did not load a device".to_string())?,
        "mode_runtime": mode_runtime,
        "runtime_compiler_versions": {
            "capture_binary_git_commit": runner.git_commit,
            "capture_binary_worktree_clean": runner.worktree_clean,
            "ullm_engine_version": env!("CARGO_PKG_VERSION"),
            "cargo_features": {
                "rocm_ck_gfx1201": cfg!(feature = "rocm-ck-gfx1201"),
                "rocm_handwritten_projection_gfx1201": cfg!(feature = "rocm-handwritten-projection-gfx1201"),
            },
        },
        "hip_guard_environment": hip_guard_environment_snapshot(),
    });
    let manifest = CaptureManifest {
        schema_version: CAPTURE_SCHEMA,
        role: plan.role,
        frozen_gate: plan.frozen_gate,
        candidate: plan.candidate,
        selector: plan.selector,
        identity,
        producer: json!({
            "plan_path": options.plan.canonicalize().map_err(|err| format!("failed to canonicalize plan: {err}"))?.display().to_string(),
            "plan_sha256": sha256_bytes(&plan_bytes),
            "output_root": output.display().to_string(),
            "capture_mode": "isolated_teacher_forced_full_model",
            "elapsed_seconds": capture_started.elapsed().as_secs_f64(),
            "mode_timing": mode_timing,
        }),
        positions: captured_positions,
    };
    write_json_new(&output.join("capture-manifest.json"), &manifest)?;
    println!(
        "SQ8 v0.2 capture complete: role={} candidate={} positions={} output={}",
        manifest.role,
        manifest.candidate.id,
        manifest.positions.len(),
        output.display()
    );
    Ok(())
}

fn validate_plan(plan: &CapturePlan) -> Result<(), String> {
    if plan.schema_version != PLAN_SCHEMA {
        return Err(format!(
            "capture plan schema mismatch: expected={PLAN_SCHEMA} actual={}",
            plan.schema_version
        ));
    }
    if plan.frozen_gate.sha256 != EXPECTED_GATE_SHA256 {
        return Err(format!(
            "capture plan frozen gate SHA-256 mismatch: expected={EXPECTED_GATE_SHA256} actual={}",
            plan.frozen_gate.sha256
        ));
    }
    let gate_path = Path::new(&plan.frozen_gate.path);
    let gate_bytes = fs::read(gate_path)
        .map_err(|err| format!("failed to read frozen gate {}: {err}", gate_path.display()))?;
    let actual_gate_sha = sha256_bytes(&gate_bytes);
    if actual_gate_sha != EXPECTED_GATE_SHA256 {
        return Err(format!(
            "frozen gate changed after plan preparation: expected={EXPECTED_GATE_SHA256} actual={actual_gate_sha}"
        ));
    }
    let gate: Value = serde_json::from_slice(&gate_bytes)
        .map_err(|err| format!("failed to parse frozen gate {}: {err}", gate_path.display()))?;
    if gate.get("schema_version").and_then(Value::as_str) != Some(GATE_SCHEMA) {
        return Err("frozen gate schema changed after plan preparation".into());
    }
    match plan.role.as_str() {
        "control" if plan.selector.enabled => {
            return Err("control capture must explicitly disable its selector".into());
        }
        "candidate" if !plan.selector.enabled => {
            return Err("candidate capture must explicitly enable its selector".into());
        }
        "control" | "candidate" => {}
        other => return Err(format!("unsupported capture role {other:?}")),
    }
    if plan.cases.is_empty() {
        return Err("capture plan has no cases".into());
    }
    if let Some(unknown) = plan
        .selector
        .environment
        .keys()
        .find(|name| !EXPERIMENTAL_SELECTOR_ENV.contains(&name.as_str()))
    {
        return Err(format!(
            "capture plan declares an unknown experimental selector environment: {unknown}"
        ));
    }
    validate_selector_binding(plan)?;
    for name in EXPERIMENTAL_SELECTOR_ENV {
        let expected = plan.selector.environment.get(name).map(String::as_str);
        let actual = std::env::var(name).ok();
        if actual.as_deref() != expected {
            return Err(format!(
                "isolated selector environment mismatch: {name} expected={expected:?} actual={actual:?}"
            ));
        }
    }
    Ok(())
}

fn validate_selector_binding(plan: &CapturePlan) -> Result<(), String> {
    if plan.role == "control" {
        if plan.candidate.id != "matched-ck-or-direct-control"
            || plan.selector.kind != "matched_ck_or_direct_control"
            || !plan.selector.environment.is_empty()
        {
            return Err("control plan does not bind the selector-disabled matched control".into());
        }
        return Ok(());
    }
    let expected = match plan.candidate.id.as_str() {
        "flash2-staged-wave32" => (
            "flash2_staged_wave32_reduction",
            [("ULLM_USE_SQ8_0_FLASH2_STAGED_WAVE32_PROTOTYPE", "1")].as_slice(),
        ),
        "paged-decode-source-tile-128" => (
            "paged_decode_source_tile_split",
            [
                ("ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_TILE", "128"),
                (
                    "ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_ALLOW_MULTITILE",
                    "1",
                ),
            ]
            .as_slice(),
        ),
        "paged-decode-source-tile-256" => (
            "paged_decode_source_tile_split",
            [
                ("ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_TILE", "256"),
                (
                    "ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_ALLOW_MULTITILE",
                    "1",
                ),
            ]
            .as_slice(),
        ),
        "handwritten-wmma-projection" => ("handwritten_wmma_projection_prototype", [].as_slice()),
        other => return Err(format!("unsupported candidate selector binding {other:?}")),
    };
    if plan.selector.kind != expected.0 {
        return Err(format!(
            "candidate selector kind mismatch: candidate={} expected={} actual={}",
            plan.candidate.id, expected.0, plan.selector.kind
        ));
    }
    let actual = plan
        .selector
        .environment
        .iter()
        .map(|(name, value)| (name.as_str(), value.as_str()))
        .collect::<BTreeMap<_, _>>();
    let expected = expected.1.iter().copied().collect::<BTreeMap<_, _>>();
    if actual != expected {
        return Err(format!(
            "candidate selector environment mismatch: candidate={} expected={expected:?} actual={actual:?}",
            plan.candidate.id
        ));
    }
    Ok(())
}

fn capture_case(
    session: &mut Qwen3Sq8ServingSession,
    context: &mut RuntimeContext,
    stream: &mut RuntimeStream,
    case: &CasePlan,
    output: &Path,
    captured_positions: &mut Vec<CapturedPosition>,
    seen_position_ids: &mut HashSet<String>,
) -> Result<(), String> {
    if case.prompt_token_ids.is_empty() || case.teacher_forced_input_tokens.is_empty() {
        return Err(format!(
            "capture case {} has an empty prompt or teacher stream",
            case.case_id
        ));
    }
    if case.teacher_forced_tokens_u32le_sha256.len() != 64
        || !case
            .teacher_forced_tokens_u32le_sha256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(format!(
            "capture case {} has an invalid teacher-forced token SHA-256 binding",
            case.case_id
        ));
    }
    if case.prompt_token_ids.len() + case.teacher_forced_input_tokens.len()
        > QWEN3_14B_SQ8_SERVING_CONTEXT_TOKENS
    {
        return Err(format!(
            "capture case {} exceeds context: prompt={} decode={}",
            case.case_id,
            case.prompt_token_ids.len(),
            case.teacher_forced_input_tokens.len()
        ));
    }
    let plans_by_ordinal = case
        .positions
        .iter()
        .map(|position| (position.ordinal, position))
        .collect::<BTreeMap<_, _>>();
    if plans_by_ordinal.len() != case.positions.len() {
        return Err(format!(
            "capture case {} repeats a capture ordinal",
            case.case_id
        ));
    }
    session
        .start_teacher_forced_capture_for_testing(
            context,
            format!("sq8-gate-v0.2:{}:{}", case.mode, case.case_id),
            case.prompt_token_ids.clone(),
            case.teacher_forced_input_tokens.len(),
            stream,
        )
        .map_err(|err| err.to_string())?;
    let mut prompt_cursor = 0_usize;
    while prompt_cursor < case.prompt_token_ids.len() {
        let width = prefill_logical_width(&case.mode, prompt_cursor, case.prompt_token_ids.len())?;
        let ordinal = prompt_cursor + width - 1;
        let planned = plans_by_ordinal.get(&ordinal).copied();
        let forced = if ordinal + 1 == case.prompt_token_ids.len() {
            Some(case.teacher_forced_input_tokens[0])
        } else {
            None
        };
        let capture = session
            .advance_teacher_forced_capture_for_testing(
                forced,
                planned.is_some(),
                planned.is_some_and(|position| position.layer_required),
                stream,
            )
            .map_err(|err| err.to_string())?;
        persist_if_planned(
            planned,
            capture,
            case,
            output,
            captured_positions,
            seen_position_ids,
        )?;
        prompt_cursor += width;
    }
    if session.status() != Sq8ServingRuntimeStatus::Decoding {
        return Err(format!(
            "capture case {} did not enter decoding after prefill: {:?}",
            case.case_id,
            session.status()
        ));
    }
    for decode_index in 0..case.teacher_forced_input_tokens.len() {
        let ordinal = case.prompt_token_ids.len() + decode_index;
        let planned = plans_by_ordinal.get(&ordinal).copied();
        let forced = case
            .teacher_forced_input_tokens
            .get(decode_index + 1)
            .copied();
        let capture = session
            .advance_teacher_forced_capture_for_testing(
                forced,
                planned.is_some(),
                planned.is_some_and(|position| position.layer_required),
                stream,
            )
            .map_err(|err| err.to_string())?;
        persist_if_planned(
            planned,
            capture,
            case,
            output,
            captured_positions,
            seen_position_ids,
        )?;
    }
    if session.status() != Sq8ServingRuntimeStatus::Ready {
        return Err(format!(
            "capture case {} did not reset after final forward: {:?}",
            case.case_id,
            session.status()
        ));
    }
    for ordinal in plans_by_ordinal {
        if !captured_positions
            .iter()
            .any(|captured| captured.id == ordinal.1.id)
        {
            return Err(format!(
                "capture plan requested an ordinal that is not an execution-unit endpoint: {}",
                ordinal.1.id
            ));
        }
    }
    Ok(())
}

/// Returns the logical prompt advance for the next execution unit.
///
/// A fixed M=128 suffix after at least one full chunk is executed as an
/// overlapping M=128 chunk, but commits only its remaining real tokens.
fn prefill_logical_width(
    mode: &str,
    prompt_tokens_processed: usize,
    prompt_tokens: usize,
) -> Result<usize, String> {
    let remaining = prompt_tokens
        .checked_sub(prompt_tokens_processed)
        .ok_or_else(|| "capture prompt cursor exceeds the prompt length".to_string())?;
    if remaining == 0 {
        return Err("capture prefill width requires a nonempty remaining prompt".to_string());
    }
    match mode {
        "sequential_m1" => Ok(1),
        "m128_chunks_with_declared_tail" if remaining >= 128 => Ok(128),
        "m128_chunks_with_declared_tail" if prompt_tokens_processed >= 128 => Ok(remaining),
        "m128_chunks_with_declared_tail" => Ok(1),
        other => Err(format!("unsupported capture mode {other:?}")),
    }
}

#[cfg(test)]
mod prefill_logical_width_tests {
    use super::prefill_logical_width;

    #[test]
    fn fixed_m128_tail_commits_the_remaining_tokens_in_one_execution() {
        assert_eq!(
            prefill_logical_width("m128_chunks_with_declared_tail", 3968, 4095).unwrap(),
            127
        );
        assert_eq!(
            prefill_logical_width("m128_chunks_with_declared_tail", 896, 1000).unwrap(),
            104
        );
        assert_eq!(
            prefill_logical_width("m128_chunks_with_declared_tail", 128, 129).unwrap(),
            1
        );
    }
}

fn persist_if_planned(
    planned: Option<&PositionPlan>,
    capture: Option<ullm_engine::sq8_serving_runtime::Sq8ServingTeacherForcedCapture>,
    case: &CasePlan,
    output: &Path,
    captured_positions: &mut Vec<CapturedPosition>,
    seen_position_ids: &mut HashSet<String>,
) -> Result<(), String> {
    match (planned, capture) {
        (None, None) => Ok(()),
        (None, Some(_)) => Err(format!(
            "capture case {} produced an unplanned tensor capture",
            case.case_id
        )),
        (Some(position), None) => Err(format!(
            "capture case {} omitted required capture {}",
            case.case_id, position.id
        )),
        (Some(position), Some(capture)) => {
            if position.case_id != case.case_id
                || position.mode != case.mode
                || position.position != capture.position
                || position.ordinal != capture.position
                || position.input_token_id != capture.input_token_id
            {
                return Err(format!(
                    "capture identity mismatch at {}: plan case/mode/position/input={}/{}/{}/{} actual={}/{}/{}/{}",
                    position.id,
                    position.case_id,
                    position.mode,
                    position.position,
                    position.input_token_id,
                    case.case_id,
                    case.mode,
                    capture.position,
                    capture.input_token_id,
                ));
            }
            if !seen_position_ids.insert(position.id.clone()) {
                return Err(format!("capture repeats position id {}", position.id));
            }
            let directory = output
                .join("cases")
                .join(&case.mode)
                .join(&case.case_id)
                .join("forwards")
                .join(format!(
                    "forward-{:05}-{}-{:05}",
                    position.ordinal, position.phase, position.phase_index
                ));
            fs::create_dir_all(directory.join("layers")).map_err(|err| {
                format!(
                    "failed to create capture directory {}: {err}",
                    directory.display()
                )
            })?;
            let logits =
                write_f32_tensor(&directory.join("logits.f32le"), &capture.logits, VOCAB_SIZE)?;
            let final_hidden = write_f32_tensor(
                &directory.join("final-hidden.f32le"),
                &capture.final_hidden,
                HIDDEN_SIZE,
            )?;
            let layers = if position.layer_required {
                let values = capture.layers.ok_or_else(|| {
                    format!(
                        "capture {} omitted its required all-layer trace",
                        position.id
                    )
                })?;
                if values.len() != LAYER_COUNT {
                    return Err(format!(
                        "capture {} layer count mismatch: expected={LAYER_COUNT} actual={}",
                        position.id,
                        values.len()
                    ));
                }
                values
                    .iter()
                    .enumerate()
                    .map(|(index, values)| {
                        write_f32_tensor(
                            &directory
                                .join("layers")
                                .join(format!("layer-{index:02}-hidden.f32le")),
                            values,
                            HIDDEN_SIZE,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?
            } else {
                if capture.layers.is_some() {
                    return Err(format!(
                        "capture {} unexpectedly has a layer trace",
                        position.id
                    ));
                }
                Vec::new()
            };
            captured_positions.push(CapturedPosition {
                id: position.id.clone(),
                case_id: position.case_id.clone(),
                mode: position.mode.clone(),
                ordinal: position.ordinal,
                phase: position.phase.clone(),
                phase_index: position.phase_index,
                position: position.position,
                input_token_id: position.input_token_id,
                candidate_top1_token_id: capture.top1.token_id,
                candidate_top1_logit: capture.top1.logit,
                logits,
                final_hidden,
                layers,
            });
            Ok(())
        }
    }
}

fn write_f32_tensor(
    path: &Path,
    values: &[f32],
    expected_elements: usize,
) -> Result<TensorDescriptor, String> {
    if values.len() != expected_elements {
        return Err(format!(
            "tensor {} element count mismatch: expected={expected_elements} actual={}",
            path.display(),
            values.len()
        ));
    }
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!(
            "tensor {} is non-finite at element {index}: {value}",
            path.display()
        ));
    }
    let mut bytes = Vec::with_capacity(expected_elements * std::mem::size_of::<f32>());
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    let hash = sha256_bytes(&bytes);
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|err| format!("failed to create tensor {}: {err}", path.display()))?;
    file.write_all(&bytes)
        .map_err(|err| format!("failed to write tensor {}: {err}", path.display()))?;
    file.sync_all()
        .map_err(|err| format!("failed to sync tensor {}: {err}", path.display()))?;
    let absolute = path
        .canonicalize()
        .map_err(|err| format!("failed to canonicalize tensor {}: {err}", path.display()))?;
    Ok(TensorDescriptor {
        path: absolute.display().to_string(),
        sha256: hash,
        dtype: "f32le",
        shape: [expected_elements],
        byte_count: bytes.len(),
    })
}

fn parse_prefill_mode(mode: &str) -> Result<Sq8ServingPrefillMode, String> {
    match mode {
        "sequential_m1" => Ok(Sq8ServingPrefillMode::SequentialM1),
        "m128_chunks_with_declared_tail" => Ok(Sq8ServingPrefillMode::FixedM128Chunks),
        other => Err(format!("unsupported capture mode {other:?}")),
    }
}

fn require_hip_kernel_guards(mode: Sq8ServingPrefillMode) -> Result<(), String> {
    let mut names = QWEN3_14B_SQ8_REQUIRED_HIP_KERNEL_ENV
        .into_iter()
        .chain(QWEN3_14B_SQ8_PAGED_REQUIRED_HIP_KERNEL_ENV)
        .chain(QWEN3_14B_SQ8_MODEL_HEAD_REQUIRED_HIP_KERNEL_ENV)
        .chain(QWEN3_14B_SQ8_EMBEDDING_REQUIRED_HIP_KERNEL_ENV)
        .collect::<Vec<_>>();
    if mode != Sq8ServingPrefillMode::SequentialM1 {
        names.extend(QWEN3_14B_SQ8_PREFILL_CHUNK_REQUIRED_HIP_KERNEL_ENV);
    }
    if std::env::var_os(QWEN3_14B_SQ8_PAGED_DECODE_SPLIT_EXPERIMENT_TILE_ENV).is_some() {
        names.push("ULLM_REQUIRE_HIP_PAGED_DECODE_SPLIT_KERNEL");
    }
    names.sort_unstable();
    names.dedup();
    let invalid = names
        .into_iter()
        .filter(|name| std::env::var(name).ok().as_deref() != Some("1"))
        .collect::<Vec<_>>();
    if invalid.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "SQ8 gate capture requires HIP guards equal to 1: {}",
            invalid.join(", ")
        ))
    }
}

fn hip_guard_environment_snapshot() -> BTreeMap<String, Option<String>> {
    let mut names = QWEN3_14B_SQ8_REQUIRED_HIP_KERNEL_ENV
        .into_iter()
        .chain(QWEN3_14B_SQ8_PAGED_REQUIRED_HIP_KERNEL_ENV)
        .chain(QWEN3_14B_SQ8_MODEL_HEAD_REQUIRED_HIP_KERNEL_ENV)
        .chain(QWEN3_14B_SQ8_EMBEDDING_REQUIRED_HIP_KERNEL_ENV)
        .chain(QWEN3_14B_SQ8_PREFILL_CHUNK_REQUIRED_HIP_KERNEL_ENV)
        .collect::<Vec<_>>();
    // Record this guard even when the selector is disabled. A shared standard
    // control can therefore keep it at `1` and remain configuration-identical
    // to source-tile candidates without enabling their selector.
    names.push("ULLM_REQUIRE_HIP_PAGED_DECODE_SPLIT_KERNEL");
    names.sort_unstable();
    names.dedup();
    names
        .into_iter()
        .map(|name| (name.to_string(), std::env::var(name).ok()))
        .collect()
}

fn isolated_gfx1201_device() -> Result<u32, String> {
    let mut devices = Vec::new();
    for index in 1..device_count()? {
        let info = device_info(index)
            .map_err(|err| format!("failed to inspect runtime device {index}: {err}"))?;
        if info.backend == "hip" {
            devices.push((index, info));
        }
    }
    if devices.len() != 1 {
        return Err(format!(
            "SQ8 gate capture requires exactly one isolated HIP device, found {}",
            devices.len()
        ));
    }
    let (runtime_index, device) = devices.pop().expect("one device after count check");
    validate_qwen3_14b_sq8_r9700_device_info(&device)?;
    if device.device_id != 0 {
        return Err(format!(
            "SQ8 gate capture requires isolated HIP device 0, got {}",
            device.device_id
        ));
    }
    Ok(runtime_index)
}

#[derive(Debug)]
struct RunnerIdentity {
    git_commit: String,
    worktree_clean: bool,
    binary_sha256: String,
}

fn runner_identity() -> Result<RunnerIdentity, String> {
    let git_commit = command_text(&["git", "rev-parse", "HEAD"])?;
    let worktree_clean = Command::new("git")
        .args(["diff", "--quiet"])
        .status()
        .map_err(|err| format!("failed to inspect worktree: {err}"))?
        .success();
    let executable = std::env::current_exe()
        .map_err(|err| format!("failed to resolve capture executable: {err}"))?;
    Ok(RunnerIdentity {
        git_commit,
        worktree_clean,
        binary_sha256: sha256_file(&executable)?,
    })
}

fn command_text(command: &[&str]) -> Result<String, String> {
    let (program, args) = command
        .split_first()
        .ok_or_else(|| "empty command".to_string())?;
    let output = Command::new(program)
        .args(args)
        .output()
        .map_err(|err| format!("failed to run {program}: {err}"))?;
    if !output.status.success() {
        return Err(format!("command {program} exited with {}", output.status));
    }
    String::from_utf8(output.stdout)
        .map_err(|err| format!("command {program} did not produce UTF-8: {err}"))
        .map(|text| text.trim().to_string())
}

fn prepare_output_root(output: &Path) -> Result<(), String> {
    if output.exists() {
        return Err(format!(
            "capture output already exists; refusing overwrite: {}",
            output.display()
        ));
    }
    let parent = output
        .parent()
        .ok_or_else(|| format!("capture output has no parent: {}", output.display()))?;
    fs::create_dir_all(parent).map_err(|err| {
        format!(
            "failed to create capture output parent {}: {err}",
            parent.display()
        )
    })?;
    fs::create_dir(output).map_err(|err| {
        format!(
            "failed to create capture output {}: {err}",
            output.display()
        )
    })
}

fn write_json_new(path: &Path, value: &impl Serialize) -> Result<(), String> {
    if path.exists() {
        return Err(format!(
            "refusing to overwrite JSON receipt {}",
            path.display()
        ));
    }
    let mut bytes = serde_json::to_vec_pretty(value)
        .map_err(|err| format!("failed to serialize JSON receipt: {err}"))?;
    bytes.push(b'\n');
    let temporary = path.with_file_name(format!(
        ".{}.{}.tmp",
        path.file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("receipt"),
        std::process::id()
    ));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|err| format!("failed to create receipt {}: {err}", temporary.display()))?;
    file.write_all(&bytes)
        .map_err(|err| format!("failed to write receipt {}: {err}", temporary.display()))?;
    file.sync_all()
        .map_err(|err| format!("failed to sync receipt {}: {err}", temporary.display()))?;
    fs::rename(&temporary, path).map_err(|err| {
        format!(
            "failed to publish receipt {} as {}: {err}",
            temporary.display(),
            path.display()
        )
    })
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file =
        File::open(path).map_err(|err| format!("failed to open {}: {err}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}

fn parse_options() -> Result<Options, String> {
    let mut args = std::env::args_os().skip(1);
    let mut plan = None;
    let mut output = None;
    while let Some(argument) = args.next() {
        match argument.to_str() {
            Some("--plan") => plan = Some(PathBuf::from(next_value(&mut args, "--plan")?)),
            Some("--output") => output = Some(PathBuf::from(next_value(&mut args, "--output")?)),
            Some("--help") | Some("-h") => return Err(usage()),
            _ => return Err(format!("unknown argument {:?}\n{}", argument, usage())),
        }
    }
    Ok(Options {
        plan: plan.ok_or_else(|| "--plan is required".to_string())?,
        output: output.ok_or_else(|| "--output is required".to_string())?,
    })
}

fn next_value(
    args: &mut impl Iterator<Item = std::ffi::OsString>,
    name: &str,
) -> Result<std::ffi::OsString, String> {
    args.next()
        .ok_or_else(|| format!("{name} requires a value"))
}

fn usage() -> String {
    "usage: ullm-sq8-gate-capture --plan CAPTURE-PLAN.json --output CAPTURE-DIRECTORY".to_string()
}
