// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Resumable CPU-only artifact-FP32 reference capture for the frozen SQ8 v0.2
//! corpus.  One invocation owns exactly one independent corpus case; the
//! process launcher supplies process-level parallelism.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use ullm_engine::sq8_fp32_reference::{
    duration_seconds, process_peak_rss_kib, write_forward_capture, ArtifactFp32Forward,
    ArtifactFp32ForwardSummary, ArtifactFp32ReferenceIdentity, ArtifactFp32ReferenceModel,
    ARTIFACT_FP32_REFERENCE_SCHEMA_VERSION, QWEN3_14B_FP32_REFERENCE_MAX_CONTEXT,
};

const CORPUS_SCHEMA_VERSION: &str = "ullm.sq8.artifact_fp32_reference.corpus.v1";
const FROZEN_GATE_SCHEMA_VERSION: &str = "ullm.sq8.numerical_gate.relative_fp32.v0.2";
const FROZEN_GATE_SHA256: &str = "64a43c032570bed8086e3c441b0774cc470c5ab1e8c67f99e02af2b6307f72bf";
const DEFAULT_GATE_PATH: &str =
    "docs/plans/sq8-numerical-gate-v0.2-relative-to-fp32-reference.json";

#[derive(Debug)]
struct Options {
    artifact: PathBuf,
    package: PathBuf,
    output: PathBuf,
    gate: PathBuf,
    case_id: String,
    mode: Mode,
    threads: usize,
    resume: bool,
    expected_gate_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    SequentialM1,
    M128ChunksWithDeclaredTail,
}

impl Mode {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "sequential_m1" => Ok(Self::SequentialM1),
            "m128_chunks_with_declared_tail" => Ok(Self::M128ChunksWithDeclaredTail),
            _ => Err(format!(
                "unsupported --mode {value:?}; expected sequential_m1 or m128_chunks_with_declared_tail"
            )),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::SequentialM1 => "sequential_m1",
            Self::M128ChunksWithDeclaredTail => "m128_chunks_with_declared_tail",
        }
    }
}

#[derive(Debug, Deserialize)]
struct FrozenGate {
    schema_version: String,
    corpus: FrozenCorpus,
}

#[derive(Debug, Deserialize)]
struct FrozenCorpus {
    fixture_root: String,
    fixture_manifest_sha256: String,
    primary_decode_streams: Vec<FrozenCase>,
    required_boundary_cases: Vec<FrozenCase>,
    prefill_coverage: FrozenPrefillCoverage,
}

#[derive(Debug, Deserialize)]
struct FrozenCase {
    id: String,
    input: Option<String>,
    input_sha256: Option<String>,
    prompt_tokens: usize,
    forced_decode_tokens: usize,
}

#[derive(Debug, Deserialize)]
struct FrozenPrefillCoverage {
    m128_inputs: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct ExactChatFixture {
    expected: ExactChatExpected,
}

#[derive(Debug, Deserialize)]
struct ExactChatExpected {
    prompt_tokens: usize,
    token_ids: Vec<u32>,
}

#[derive(Debug, Clone, Serialize)]
struct MaterializedCasePlan {
    id: String,
    class: String,
    declared_input: Option<String>,
    declared_input_sha256: Option<String>,
    materialized_input_sha256: String,
    input_source: String,
    prompt_tokens: usize,
    forced_decode_tokens: usize,
    total_forwards: usize,
    m128_checkpoint_forward_indices: Vec<usize>,
}

#[derive(Debug)]
struct MaterializedCase {
    plan: MaterializedCasePlan,
    prompt: Vec<u32>,
}

#[derive(Debug, Serialize)]
struct RunPlan {
    schema_version: &'static str,
    frozen_gate_path: String,
    frozen_gate_sha256: String,
    frozen_gate_schema_version: String,
    fixture_root: String,
    fixture_manifest_sha256: String,
    mode: String,
    mode_reference_execution: &'static str,
    case: MaterializedCasePlan,
    reference_identity: Value,
    reference_schema_version: &'static str,
    thread_count: usize,
    seed: u64,
    executable: String,
    executable_sha256: String,
    cpu_model: Option<String>,
}

#[derive(Debug, Serialize)]
struct ProgressReceipt {
    schema_version: &'static str,
    status: String,
    case_id: String,
    mode: String,
    total_forwards: usize,
    completed_forwards: usize,
    resumed_verified_forwards: usize,
    last_forward: Option<ProgressForward>,
    updated_unix_seconds: u64,
}

#[derive(Debug, Clone, Serialize)]
struct ProgressForward {
    ordinal: usize,
    phase: &'static str,
    phase_index: usize,
    position: usize,
    input_token_id: u32,
    greedy_token_id: u32,
    capture_metadata_sha256: String,
}

#[derive(Debug, Serialize)]
struct M128CheckpointReceipt {
    schema_version: &'static str,
    case_id: String,
    mode: &'static str,
    checkpoint_forward_indices: Vec<usize>,
    checkpoint_capture_directories: Vec<String>,
}

#[derive(Debug, Serialize)]
struct RunReceipt {
    schema_version: &'static str,
    status: &'static str,
    execution_backend: &'static str,
    mode: String,
    seed: u64,
    initialization_elapsed_seconds: f64,
    execution_elapsed_seconds: f64,
    peak_rss_kib: Option<u64>,
    total_forwards: usize,
    resumed_verified_forwards: usize,
    output_payload_bytes: u64,
    teacher_forced_token_count: usize,
    teacher_forced_tokens_u32le_sha256: String,
    integrity_manifest: String,
    integrity_manifest_sha256: String,
    plan_sha256: String,
}

struct RunLock {
    path: PathBuf,
}

impl Drop for RunLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn main() -> Result<(), String> {
    let options = parse_options()?;
    run(options)
}

fn run(options: Options) -> Result<(), String> {
    let gate_bytes = fs::read(&options.gate).map_err(|error| {
        format!(
            "failed to read frozen gate {}: {error}",
            options.gate.display()
        )
    })?;
    let gate_sha256 = sha256_bytes(&gate_bytes);
    if gate_sha256 != options.expected_gate_sha256 {
        return Err(format!(
            "frozen gate SHA-256 mismatch: expected={} actual={gate_sha256}",
            options.expected_gate_sha256
        ));
    }
    let gate: FrozenGate = serde_json::from_slice(&gate_bytes)
        .map_err(|error| format!("failed to parse frozen gate: {error}"))?;
    if gate.schema_version != FROZEN_GATE_SCHEMA_VERSION {
        return Err(format!(
            "frozen gate schema mismatch: expected={FROZEN_GATE_SCHEMA_VERSION} actual={}",
            gate.schema_version
        ));
    }
    let fixture_root = find_fixture_root(&options.gate, &gate.corpus.fixture_root)?;
    verify_fixture_manifest(&fixture_root, &gate.corpus.fixture_manifest_sha256)?;
    let materialized =
        materialize_case(&gate.corpus, &fixture_root, &options.case_id, options.mode)?;
    if materialized.plan.total_forwards > QWEN3_14B_FP32_REFERENCE_MAX_CONTEXT {
        return Err(format!(
            "case {} requires {} context tokens, above reference limit {QWEN3_14B_FP32_REFERENCE_MAX_CONTEXT}",
            materialized.plan.id, materialized.plan.total_forwards
        ));
    }

    prepare_output_root(&options.output, options.resume)?;
    let _lock = acquire_lock(&options.output, options.resume)?;

    let initialization_start = Instant::now();
    let model =
        ArtifactFp32ReferenceModel::open(&options.artifact, &options.package, options.threads)?;
    let initialization_elapsed = initialization_start.elapsed();
    let identity = model.identity();
    let executable = std::env::current_exe()
        .map_err(|error| format!("failed to resolve current executable: {error}"))?;
    let expected_plan = RunPlan {
        schema_version: CORPUS_SCHEMA_VERSION,
        frozen_gate_path: options.gate.display().to_string(),
        frozen_gate_sha256: gate_sha256,
        frozen_gate_schema_version: gate.schema_version,
        fixture_root: fixture_root.display().to_string(),
        fixture_manifest_sha256: gate.corpus.fixture_manifest_sha256,
        mode: options.mode.as_str().to_string(),
        mode_reference_execution:
            "cpu_strict_f32_scalar_causal_reference_with_m128_checkpoint_grouping",
        case: materialized.plan.clone(),
        reference_identity: serde_json::to_value(&identity)
            .map_err(|error| format!("failed to serialize reference identity: {error}"))?,
        reference_schema_version: ARTIFACT_FP32_REFERENCE_SCHEMA_VERSION,
        thread_count: options.threads,
        seed: 0,
        executable: executable.display().to_string(),
        executable_sha256: sha256_file(&executable)?,
        cpu_model: cpu_model(),
    };
    let plan_path = options.output.join("plan.json");
    let expected_plan_bytes = json_bytes(&expected_plan)?;
    if plan_path.exists() {
        let existing = fs::read(&plan_path).map_err(|error| {
            format!(
                "failed to read existing plan {}: {error}",
                plan_path.display()
            )
        })?;
        if existing != expected_plan_bytes {
            return Err(format!(
                "existing plan {} does not exactly match this resume invocation",
                plan_path.display()
            ));
        }
    } else {
        write_create_new(&plan_path, &expected_plan_bytes)?;
    }

    let final_run_path = options.output.join("run.json");
    if final_run_path.exists() {
        println!(
            "artifact-FP32 corpus case {} ({}) is already complete at {}",
            materialized.plan.id,
            options.mode.as_str(),
            options.output.display()
        );
        return Ok(());
    }

    let execution_start = Instant::now();
    let mut session = model.session(materialized.plan.total_forwards)?;
    let mut completed = 0_usize;
    let mut resumed_verified = 0_usize;
    let mut last_progress = None;
    let mut last_greedy = None;
    let mut teacher_forced_tokens = Vec::with_capacity(materialized.plan.forced_decode_tokens + 1);

    for (prompt_index, token_id) in materialized.prompt.iter().copied().enumerate() {
        let result = run_position(
            &options.output,
            &identity,
            &mut session,
            completed,
            "prompt",
            prompt_index,
            token_id,
        )?;
        completed += 1;
        resumed_verified += usize::from(result.resumed);
        last_greedy = Some(result.greedy_token_id);
        last_progress = Some(result.progress);
        write_progress(
            &options.output,
            &materialized.plan,
            options.mode,
            "running",
            completed,
            resumed_verified,
            last_progress.as_ref(),
        )?;
    }

    let seed_token =
        last_greedy.ok_or_else(|| "materialized prompt is unexpectedly empty".to_string())?;
    teacher_forced_tokens.push(seed_token);
    let mut next_input = seed_token;
    for decode_index in 0..materialized.plan.forced_decode_tokens {
        let result = run_position(
            &options.output,
            &identity,
            &mut session,
            completed,
            "decode",
            decode_index,
            next_input,
        )?;
        completed += 1;
        resumed_verified += usize::from(result.resumed);
        next_input = result.greedy_token_id;
        teacher_forced_tokens.push(next_input);
        last_progress = Some(result.progress);
        write_progress(
            &options.output,
            &materialized.plan,
            options.mode,
            "running",
            completed,
            resumed_verified,
            last_progress.as_ref(),
        )?;
    }
    if completed != materialized.plan.total_forwards {
        return Err(format!(
            "corpus forward count mismatch: expected={} actual={completed}",
            materialized.plan.total_forwards
        ));
    }

    let teacher_path = options.output.join("teacher-forced-tokens.u32le");
    write_u32le_atomic(&teacher_path, &teacher_forced_tokens)?;
    let teacher_sha256 = sha256_file(&teacher_path)?;
    if options.mode == Mode::M128ChunksWithDeclaredTail {
        write_m128_checkpoints(&options.output, &materialized.plan)?;
    }
    let integrity_manifest_path = options.output.join("SHA256SUMS");
    write_integrity_manifest(&options.output, &materialized.plan, options.mode)?;
    let integrity_manifest_sha256 = sha256_file(&integrity_manifest_path)?;
    write_progress(
        &options.output,
        &materialized.plan,
        options.mode,
        "complete",
        completed,
        resumed_verified,
        last_progress.as_ref(),
    )?;

    let payload_bytes_per_forward = (151_936_u64 * 4) + (41_u64 * 5_120 * 4);
    let receipt = RunReceipt {
        schema_version: CORPUS_SCHEMA_VERSION,
        status: "complete",
        execution_backend: "cpu_only_no_runtime_context",
        mode: options.mode.as_str().to_string(),
        seed: 0,
        initialization_elapsed_seconds: duration_seconds(initialization_elapsed),
        execution_elapsed_seconds: duration_seconds(execution_start.elapsed()),
        peak_rss_kib: process_peak_rss_kib()?,
        total_forwards: completed,
        resumed_verified_forwards: resumed_verified,
        output_payload_bytes: payload_bytes_per_forward
            .checked_mul(
                u64::try_from(completed).map_err(|_| "forward count overflow".to_string())?,
            )
            .ok_or_else(|| "captured payload byte count overflow".to_string())?,
        teacher_forced_token_count: teacher_forced_tokens.len(),
        teacher_forced_tokens_u32le_sha256: teacher_sha256,
        integrity_manifest: "SHA256SUMS".to_string(),
        integrity_manifest_sha256,
        plan_sha256: sha256_file(&plan_path)?,
    };
    write_create_new(&final_run_path, &json_bytes(&receipt)?)?;
    println!(
        "artifact-FP32 corpus case {} ({}) complete: {} forwards at {}",
        materialized.plan.id,
        options.mode.as_str(),
        completed,
        options.output.display()
    );
    Ok(())
}

struct PositionResult {
    greedy_token_id: u32,
    resumed: bool,
    progress: ProgressForward,
}

#[allow(clippy::too_many_arguments)]
fn run_position(
    output_root: &Path,
    identity: &ArtifactFp32ReferenceIdentity,
    session: &mut ullm_engine::sq8_fp32_reference::ArtifactFp32ReferenceSession<'_>,
    ordinal: usize,
    phase: &'static str,
    phase_index: usize,
    input_token_id: u32,
) -> Result<PositionResult, String> {
    let directory_name = format!("forward-{ordinal:05}-{phase}-{phase_index:05}");
    let final_dir = output_root.join("forwards").join(&directory_name);
    let forward = session.forward_token(input_token_id)?;
    let summary = forward.summary.clone();
    let resumed = if final_dir.exists() {
        verify_existing_capture(&final_dir, identity, &summary)?;
        true
    } else {
        write_capture_atomic(output_root, &directory_name, &final_dir, identity, &forward)?;
        false
    };
    let metadata_sha256 = sha256_file(&final_dir.join("metadata.json"))?;
    Ok(PositionResult {
        greedy_token_id: summary.greedy_token_id,
        resumed,
        progress: ProgressForward {
            ordinal,
            phase,
            phase_index,
            position: summary.position,
            input_token_id: summary.input_token_id,
            greedy_token_id: summary.greedy_token_id,
            capture_metadata_sha256: metadata_sha256,
        },
    })
}

fn write_capture_atomic(
    output_root: &Path,
    directory_name: &str,
    final_dir: &Path,
    identity: &ArtifactFp32ReferenceIdentity,
    forward: &ArtifactFp32Forward,
) -> Result<(), String> {
    let staging_root = output_root.join(".staging");
    fs::create_dir_all(&staging_root).map_err(|error| {
        format!(
            "failed to create capture staging root {}: {error}",
            staging_root.display()
        )
    })?;
    let staging_dir = staging_root.join(format!("{directory_name}.pid-{}", std::process::id()));
    if staging_dir.exists() {
        return Err(format!(
            "staging capture already exists at {}; retain it for inspection and resume with a fresh process",
            staging_dir.display()
        ));
    }
    write_forward_capture(&staging_dir, identity, forward)?;
    fs::rename(&staging_dir, final_dir).map_err(|error| {
        format!(
            "failed to atomically publish capture {} -> {}: {error}",
            staging_dir.display(),
            final_dir.display()
        )
    })
}

fn verify_existing_capture(
    directory: &Path,
    identity: &ArtifactFp32ReferenceIdentity,
    summary: &ArtifactFp32ForwardSummary,
) -> Result<(), String> {
    let metadata_path = directory.join("metadata.json");
    let metadata_bytes = fs::read(&metadata_path).map_err(|error| {
        format!(
            "failed to read checkpoint {}: {error}",
            metadata_path.display()
        )
    })?;
    let metadata: Value = serde_json::from_slice(&metadata_bytes).map_err(|error| {
        format!(
            "failed to parse checkpoint {}: {error}",
            metadata_path.display()
        )
    })?;
    let expected_identity = serde_json::to_value(identity)
        .map_err(|error| format!("failed to serialize reference identity for resume: {error}"))?;
    let expected_summary = serde_json::to_value(summary)
        .map_err(|error| format!("failed to serialize forward summary for resume: {error}"))?;
    if metadata.get("identity") != Some(&expected_identity) {
        return Err(format!(
            "checkpoint identity mismatch at {}",
            metadata_path.display()
        ));
    }
    if metadata.get("forward") != Some(&expected_summary) {
        return Err(format!(
            "checkpoint forward hash/token mismatch at {}",
            metadata_path.display()
        ));
    }
    let files = metadata
        .get("files")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            format!(
                "checkpoint files map missing at {}",
                metadata_path.display()
            )
        })?;
    for (relative, expected_hash) in files {
        let expected_hash = expected_hash.as_str().ok_or_else(|| {
            format!(
                "checkpoint has non-string content hash for {relative} at {}",
                metadata_path.display()
            )
        })?;
        let actual_hash = sha256_file(&directory.join(relative))?;
        if actual_hash != expected_hash {
            return Err(format!(
                "checkpoint payload hash mismatch at {}/{}: expected={expected_hash} actual={actual_hash}",
                directory.display(),
                relative
            ));
        }
    }
    Ok(())
}

fn materialize_case(
    corpus: &FrozenCorpus,
    fixture_root: &Path,
    case_id: &str,
    mode: Mode,
) -> Result<MaterializedCase, String> {
    let (frozen, class) = corpus
        .primary_decode_streams
        .iter()
        .find(|case| case.id == case_id)
        .map(|case| (case, "primary_decode"))
        .or_else(|| {
            corpus
                .required_boundary_cases
                .iter()
                .find(|case| case.id == case_id)
                .map(|case| (case, "required_boundary"))
        })
        .ok_or_else(|| format!("case {case_id:?} is not present in the frozen v0.2 corpus"))?;
    if mode == Mode::M128ChunksWithDeclaredTail
        && !corpus
            .prefill_coverage
            .m128_inputs
            .iter()
            .any(|id| id == case_id)
    {
        return Err(format!(
            "case {case_id} is not one of the frozen m128 prefill inputs"
        ));
    }
    let (prompt, materialized_input_sha256, input_source) = match &frozen.input {
        Some(relative) if relative.ends_with(".u32le") => {
            let path = fixture_root.join(relative);
            let bytes = fs::read(&path).map_err(|error| {
                format!(
                    "failed to read frozen raw prompt {}: {error}",
                    path.display()
                )
            })?;
            let actual_sha256 = sha256_bytes(&bytes);
            verify_declared_input_hash(frozen, &actual_sha256, &path)?;
            let tokens = parse_u32le(&bytes, &path)?;
            validate_raw_range(&tokens, frozen.prompt_tokens, &path)?;
            (tokens, actual_sha256, "checked_in_raw_u32le".to_string())
        }
        Some(relative) if relative.ends_with(".json") => {
            let path = fixture_root.join(relative);
            let bytes = fs::read(&path).map_err(|error| {
                format!(
                    "failed to read frozen chat fixture {}: {error}",
                    path.display()
                )
            })?;
            let actual_sha256 = sha256_bytes(&bytes);
            verify_declared_input_hash(frozen, &actual_sha256, &path)?;
            let fixture: ExactChatFixture = serde_json::from_slice(&bytes).map_err(|error| {
                format!(
                    "failed to parse frozen chat fixture {}: {error}",
                    path.display()
                )
            })?;
            if fixture.expected.prompt_tokens != frozen.prompt_tokens
                || fixture.expected.token_ids.len() != frozen.prompt_tokens
            {
                return Err(format!(
                    "frozen chat fixture {} prompt count mismatch: declared={} expected={} actual={}",
                    path.display(),
                    frozen.prompt_tokens,
                    fixture.expected.prompt_tokens,
                    fixture.expected.token_ids.len()
                ));
            }
            (
                fixture.expected.token_ids,
                actual_sha256,
                "checked_in_exact_chat_json".to_string(),
            )
        }
        Some(relative) => {
            return Err(format!("unsupported frozen input format {relative:?}"));
        }
        None => {
            let tokens = (1..=u32::try_from(frozen.prompt_tokens)
                .map_err(|_| format!("prompt count does not fit u32 for {case_id}"))?)
                .collect::<Vec<_>>();
            let materialized = u32le_bytes(&tokens);
            (
                tokens,
                sha256_bytes(&materialized),
                "frozen_raw_range_construction".to_string(),
            )
        }
    };
    if prompt.len() != frozen.prompt_tokens {
        return Err(format!(
            "materialized case {case_id} prompt length mismatch: expected={} actual={}",
            frozen.prompt_tokens,
            prompt.len()
        ));
    }
    let total_forwards = frozen
        .prompt_tokens
        .checked_add(frozen.forced_decode_tokens)
        .ok_or_else(|| format!("frozen case {case_id} forward count overflows usize"))?;
    let m128_checkpoint_forward_indices = if mode == Mode::M128ChunksWithDeclaredTail {
        m128_checkpoint_indices(total_forwards)
    } else {
        Vec::new()
    };
    Ok(MaterializedCase {
        plan: MaterializedCasePlan {
            id: frozen.id.clone(),
            class: class.to_string(),
            declared_input: frozen.input.clone(),
            declared_input_sha256: frozen.input_sha256.clone(),
            materialized_input_sha256,
            input_source,
            prompt_tokens: frozen.prompt_tokens,
            forced_decode_tokens: frozen.forced_decode_tokens,
            total_forwards,
            m128_checkpoint_forward_indices,
        },
        prompt,
    })
}

fn verify_declared_input_hash(
    frozen: &FrozenCase,
    actual_sha256: &str,
    path: &Path,
) -> Result<(), String> {
    if let Some(expected_sha256) = &frozen.input_sha256 {
        if actual_sha256 != expected_sha256 {
            return Err(format!(
                "frozen input hash mismatch for {}: expected={expected_sha256} actual={actual_sha256}",
                path.display()
            ));
        }
    }
    Ok(())
}

fn validate_raw_range(tokens: &[u32], expected_len: usize, path: &Path) -> Result<(), String> {
    if tokens.len() != expected_len {
        return Err(format!(
            "raw prompt {} length mismatch: expected={expected_len} actual={}",
            path.display(),
            tokens.len()
        ));
    }
    for (index, token) in tokens.iter().copied().enumerate() {
        let expected = u32::try_from(index + 1).map_err(|_| {
            format!(
                "raw prompt index overflow while validating {}",
                path.display()
            )
        })?;
        if token != expected {
            return Err(format!(
                "raw prompt {} is not the frozen [1..N] range at index {index}: expected={expected} actual={token}",
                path.display()
            ));
        }
    }
    Ok(())
}

fn m128_checkpoint_indices(total_forwards: usize) -> Vec<usize> {
    let mut points = (128..=total_forwards)
        .step_by(128)
        .map(|end_exclusive| end_exclusive - 1)
        .collect::<Vec<_>>();
    if total_forwards % 128 != 0 {
        points.push(total_forwards - 1);
    }
    points
}

fn write_m128_checkpoints(output: &Path, plan: &MaterializedCasePlan) -> Result<(), String> {
    let directories = plan
        .m128_checkpoint_forward_indices
        .iter()
        .map(|ordinal| capture_directory_for_ordinal(plan, *ordinal))
        .collect::<Result<Vec<_>, _>>()?;
    let receipt = M128CheckpointReceipt {
        schema_version: CORPUS_SCHEMA_VERSION,
        case_id: plan.id.clone(),
        mode: "m128_chunks_with_declared_tail",
        checkpoint_forward_indices: plan.m128_checkpoint_forward_indices.clone(),
        checkpoint_capture_directories: directories,
    };
    write_json_atomic(&output.join("m128-checkpoints.json"), &receipt)
}

fn capture_directory_for_ordinal(
    plan: &MaterializedCasePlan,
    ordinal: usize,
) -> Result<String, String> {
    if ordinal >= plan.total_forwards {
        return Err(format!(
            "checkpoint ordinal {ordinal} exceeds case forward count {}",
            plan.total_forwards
        ));
    }
    if ordinal < plan.prompt_tokens {
        Ok(format!("forwards/forward-{ordinal:05}-prompt-{ordinal:05}"))
    } else {
        let decode_index = ordinal - plan.prompt_tokens;
        Ok(format!(
            "forwards/forward-{ordinal:05}-decode-{decode_index:05}"
        ))
    }
}

fn write_integrity_manifest(
    output: &Path,
    plan: &MaterializedCasePlan,
    mode: Mode,
) -> Result<(), String> {
    let mut relative_paths = vec![
        PathBuf::from("plan.json"),
        PathBuf::from("teacher-forced-tokens.u32le"),
    ];
    if mode == Mode::M128ChunksWithDeclaredTail {
        relative_paths.push(PathBuf::from("m128-checkpoints.json"));
    }
    for ordinal in 0..plan.total_forwards {
        relative_paths.push(
            PathBuf::from(capture_directory_for_ordinal(plan, ordinal)?).join("metadata.json"),
        );
    }
    relative_paths.sort();
    let mut content = String::new();
    for relative in relative_paths {
        let digest = sha256_file(&output.join(&relative))?;
        content.push_str(&digest);
        content.push_str("  ");
        content.push_str(&relative.to_string_lossy());
        content.push('\n');
    }
    write_atomic(&output.join("SHA256SUMS"), content.as_bytes())
}

fn write_progress(
    output: &Path,
    plan: &MaterializedCasePlan,
    mode: Mode,
    status: &str,
    completed_forwards: usize,
    resumed_verified_forwards: usize,
    last_forward: Option<&ProgressForward>,
) -> Result<(), String> {
    let receipt = ProgressReceipt {
        schema_version: CORPUS_SCHEMA_VERSION,
        status: status.to_string(),
        case_id: plan.id.clone(),
        mode: mode.as_str().to_string(),
        total_forwards: plan.total_forwards,
        completed_forwards,
        resumed_verified_forwards,
        last_forward: last_forward.cloned(),
        updated_unix_seconds: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| format!("system clock is before Unix epoch: {error}"))?
            .as_secs(),
    };
    write_json_atomic(&output.join("progress.json"), &receipt)
}

fn prepare_output_root(output: &Path, resume: bool) -> Result<(), String> {
    if output.exists() {
        if !output.is_dir() {
            return Err(format!(
                "output path exists but is not a directory: {}",
                output.display()
            ));
        }
        if !resume {
            return Err(format!(
                "output path already exists; use --resume only for an existing compatible checkpoint: {}",
                output.display()
            ));
        }
    } else {
        fs::create_dir_all(output).map_err(|error| {
            format!("failed to create output root {}: {error}", output.display())
        })?;
    }
    fs::create_dir_all(output.join("forwards"))
        .map_err(|error| format!("failed to create forward root: {error}"))?;
    Ok(())
}

fn acquire_lock(output: &Path, resume: bool) -> Result<RunLock, String> {
    let lock_path = output.join(".run.lock");
    let create = || -> Result<(), String> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&lock_path)
            .map_err(|error| {
                format!("failed to create run lock {}: {error}", lock_path.display())
            })?;
        writeln!(file, "{}", std::process::id()).map_err(|error| {
            format!("failed to write run lock {}: {error}", lock_path.display())
        })?;
        file.sync_all()
            .map_err(|error| format!("failed to sync run lock {}: {error}", lock_path.display()))
    };
    match create() {
        Ok(()) => Ok(RunLock { path: lock_path }),
        Err(error) if resume && lock_path.exists() => {
            let owner_pid = fs::read_to_string(&lock_path)
                .ok()
                .and_then(|value| value.trim().parse::<u32>().ok());
            if owner_pid.is_some_and(|pid| Path::new("/proc").join(pid.to_string()).exists()) {
                return Err(format!(
                    "checkpoint is actively locked by pid {:?}: {}",
                    owner_pid,
                    lock_path.display()
                ));
            }
            let stale_path = output.join(format!(
                ".stale-run-lock-{}-{}",
                owner_pid.unwrap_or_default(),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map_err(|clock| format!("system clock is before Unix epoch: {clock}"))?
                    .as_nanos()
            ));
            fs::rename(&lock_path, &stale_path).map_err(|rename_error| {
                format!(
                    "failed to preserve stale lock {} as {} after {error}: {rename_error}",
                    lock_path.display(),
                    stale_path.display()
                )
            })?;
            create()?;
            Ok(RunLock { path: lock_path })
        }
        Err(error) => Err(error),
    }
}

fn find_fixture_root(gate_path: &Path, fixture_relative: &str) -> Result<PathBuf, String> {
    let absolute_gate = gate_path.canonicalize().map_err(|error| {
        format!(
            "failed to canonicalize frozen gate {}: {error}",
            gate_path.display()
        )
    })?;
    for ancestor in absolute_gate.ancestors() {
        let candidate = ancestor.join(fixture_relative);
        if candidate.is_dir() {
            return Ok(candidate);
        }
    }
    Err(format!(
        "could not resolve fixture root {fixture_relative:?} from frozen gate {}",
        absolute_gate.display()
    ))
}

fn verify_fixture_manifest(fixture_root: &Path, expected_sha256: &str) -> Result<(), String> {
    let manifest = fixture_root.join("manifest.json");
    let actual_sha256 = sha256_file(&manifest)?;
    if actual_sha256 != expected_sha256 {
        return Err(format!(
            "fixture manifest SHA-256 mismatch at {}: expected={expected_sha256} actual={actual_sha256}",
            manifest.display()
        ));
    }
    Ok(())
}

fn parse_options() -> Result<Options, String> {
    let mut args = std::env::args().skip(1);
    let artifact = PathBuf::from(args.next().ok_or_else(usage)?);
    let package = PathBuf::from(args.next().ok_or_else(usage)?);
    let output = PathBuf::from(args.next().ok_or_else(usage)?);
    let mut gate = PathBuf::from(DEFAULT_GATE_PATH);
    let mut case_id = None;
    let mut mode = None;
    let mut threads = 16_usize;
    let mut resume = false;
    let mut expected_gate_sha256 = FROZEN_GATE_SHA256.to_string();
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--gate" => gate = PathBuf::from(next_value(&mut args, "--gate")?),
            "--case" => case_id = Some(next_value(&mut args, "--case")?),
            "--mode" => mode = Some(Mode::parse(&next_value(&mut args, "--mode")?)?),
            "--threads" => threads = parse_value(next_value(&mut args, "--threads")?, "--threads")?,
            "--resume" => resume = true,
            "--expected-gate-sha256" => {
                expected_gate_sha256 = next_value(&mut args, "--expected-gate-sha256")?
            }
            "--help" | "-h" => return Err(usage()),
            _ => return Err(format!("unknown argument {argument:?}\n{}", usage())),
        }
    }
    if threads == 0 || threads > 128 {
        return Err("--threads must be in 1..=128".to_string());
    }
    let case_id = case_id.ok_or_else(|| "--case is required".to_string())?;
    let mode = mode.ok_or_else(|| "--mode is required".to_string())?;
    Ok(Options {
        artifact,
        package,
        output,
        gate,
        case_id,
        mode,
        threads,
        resume,
        expected_gate_sha256,
    })
}

fn next_value(args: &mut impl Iterator<Item = String>, name: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("{name} requires a value"))
}

fn parse_value<T>(value: String, name: &str) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    value
        .parse::<T>()
        .map_err(|error| format!("invalid {name}: {error}"))
}

fn usage() -> String {
    "usage: ullm-sq8-fp32-reference-corpus ARTIFACT_DIR PACKAGE_DIR OUTPUT_DIR --case ID --mode sequential_m1|m128_chunks_with_declared_tail [--gate PATH] [--threads N] [--resume] [--expected-gate-sha256 SHA256]".to_string()
}

fn parse_u32le(bytes: &[u8], path: &Path) -> Result<Vec<u32>, String> {
    if bytes.len() % 4 != 0 {
        return Err(format!(
            "u32le input {} has non-multiple-of-four size {}",
            path.display(),
            bytes.len()
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn u32le_bytes(tokens: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(tokens.len() * 4);
    for token in tokens {
        bytes.extend_from_slice(&token.to_le_bytes());
    }
    bytes
}

fn write_u32le_atomic(path: &Path, tokens: &[u32]) -> Result<(), String> {
    write_atomic(path, &u32le_bytes(tokens))
}

fn json_bytes(value: &impl Serialize) -> Result<Vec<u8>, String> {
    let mut bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| format!("failed to serialize JSON receipt: {error}"))?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<(), String> {
    write_atomic(path, &json_bytes(value)?)
}

fn write_create_new(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("failed to create {}: {error}", path.display()))?;
    file.write_all(bytes)
        .map_err(|error| format!("failed to write {}: {error}", path.display()))?;
    file.sync_all()
        .map_err(|error| format!("failed to sync {}: {error}", path.display()))
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| format!("path has no parent: {}", path.display()))?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("failed to create parent {}: {error}", parent.display()))?;
    let temporary = parent.join(format!(
        ".{}-{}.tmp",
        path.file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| format!("non-UTF-8 file name: {}", path.display()))?,
        std::process::id()
    ));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|error| {
            format!(
                "failed to create temporary {}: {error}",
                temporary.display()
            )
        })?;
    file.write_all(bytes)
        .map_err(|error| format!("failed to write temporary {}: {error}", temporary.display()))?;
    file.sync_all()
        .map_err(|error| format!("failed to sync temporary {}: {error}", temporary.display()))?;
    fs::rename(&temporary, path).map_err(|error| {
        format!(
            "failed to atomically replace {} with {}: {error}",
            path.display(),
            temporary.display()
        )
    })
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = File::open(path)
        .map_err(|error| format!("failed to open {} for SHA-256: {error}", path.display()))?;
    let mut buffer = vec![0_u8; 1024 * 1024];
    let mut digest = Sha256::new();
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| format!("failed to hash {}: {error}", path.display()))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}

fn cpu_model() -> Option<String> {
    let content = fs::read_to_string("/proc/cpuinfo").ok()?;
    content.lines().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        (key.trim() == "model name").then(|| value.trim().to_string())
    })
}
