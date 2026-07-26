// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Actual-artifact differential for the private gfx1201 handwritten SQ8_0
//! projection experiment.
//!
//! This tool intentionally has no dispatcher, manifest, campaign, or release
//! side effects. It runs the same fixed-M8 prefill and first feedback decode
//! as the frozen full-model gate, captures every actual M=1 layer workspace,
//! then replays the first divergent projection by K128 prefix at CK's real
//! BF16-workspace-to-F32 boundary.

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use ullm_engine::host_bytes::encode_f32_to_bytes;
use ullm_engine::loader::read_named_passthrough_f32;
use ullm_engine::sq_canonical::{Sq8CanonicalArtifact, read_sq8_canonical_artifact};
use ullm_engine::sq_runtime::Sq8CanonicalResidentRuntimeTensor;
use ullm_engine::sq8_layer_oracle::QWEN3_14B_HIDDEN_SIZE;
use ullm_engine::sq8_layer_runtime::{
    Qwen3Sq8LayerNormValues, Sq8LayerQuantizedActivationTrace, Sq8LayerRuntimeTrace,
    load_qwen3_14b_sq8_layer_weights,
};
use ullm_engine::sq8_serving_runtime::{
    Qwen3Sq8ServingSession, Sq8CancellationToken, Sq8ServingAdvance, Sq8ServingPrefillMode,
    Sq8ServingRequest, Sq8ServingRuntimeStatus, load_qwen3_14b_sq8_serving_norms,
};
use ullm_runtime_sys::{
    DeviceInfo, RuntimeBuffer, RuntimeContext, RuntimeStream, Sq8CkQuantizedActivation,
    device_count, device_info, sq8_ck_projection_buffer_bytes, sq8_ck_projection_f32,
    sq8_handwritten_gfx1201_m1_projection_f32,
};

const SCHEMA_VERSION: &str = "ullm.sq8_0.handwritten_projection_contract.v2";
const UPLOAD_CHUNK_BYTES: usize = 16 * 1024 * 1024;
const SCALE_BLOCK: usize = 128;
const GATE_MAX_NEW_TOKENS: usize = 4;

#[derive(Debug)]
struct Options {
    artifact: PathBuf,
    package: PathBuf,
    prompt_token_ids_u32le: PathBuf,
    output: PathBuf,
}

#[derive(Debug, Serialize)]
struct ContractReport {
    schema_version: &'static str,
    scope: &'static str,
    source: SourceRecord,
    device: DeviceRecord,
    actual_decode_route: ActualDecodeRoute,
    layer_stage_differentials: Vec<LayerStageDifferential>,
    first_stage_divergence: Option<FirstStageDivergence>,
    selected_projection: Option<ProjectionContractReport>,
    fragment_lane_probe: FragmentLaneProbe,
    interpretation: Interpretation,
}

#[derive(Debug, Serialize)]
struct SourceRecord {
    artifact_content_sha256: String,
    prompt_token_count: usize,
    prompt_u32le_sha256: String,
    prefill_mode: &'static str,
    max_new_tokens: usize,
}

#[derive(Debug, Serialize)]
struct DeviceRecord {
    runtime_index: u32,
    backend_device_id: i32,
    backend: String,
    name: String,
    gcn_arch_name: String,
    compute_major: i32,
    compute_minor: i32,
}

#[derive(Debug, Serialize)]
struct ActualDecodeRoute {
    ck_prefill_generated_token: usize,
    handwritten_prefill_generated_token: usize,
    ck_input_token_id: usize,
    handwritten_input_token_id: usize,
    ck_position: usize,
    handwritten_position: usize,
    ck_profile: String,
    handwritten_profile: String,
    method: &'static str,
}

#[derive(Debug, Serialize)]
struct LayerStageDifferential {
    layer_index: usize,
    stages: Vec<StageDifferential>,
}

#[derive(Debug, Serialize)]
struct FirstStageDivergence {
    layer_index: usize,
    stage: String,
    bitwise_mismatches: usize,
    first_mismatch: Option<usize>,
    max_abs: f32,
}

#[derive(Debug, Serialize)]
struct StageDifferential {
    stage: &'static str,
    elements: usize,
    bitwise_mismatches: usize,
    first_mismatch: Option<usize>,
    max_abs: f32,
    max_rel: f64,
    ck_sha256: String,
    handwritten_sha256: String,
}

#[derive(Debug, Serialize)]
struct ProjectionContractReport {
    projection: &'static str,
    m: usize,
    n: usize,
    k: usize,
    direct_replay: ProjectionDifferential,
    direct_replay_matches_layer_trace: bool,
    activation_replay_matches_layer_trace: bool,
    k128_prefix_method: &'static str,
    k128_prefixes: Vec<K128PrefixDifferential>,
    first_mismatching_prefix_blocks: Option<usize>,
    k128_single_blocks: Vec<K128BlockDifferential>,
    first_mismatching_single_block: Option<usize>,
    k16_prefixes_for_first_mismatching_single_block: Vec<K16PrefixDifferential>,
    contract_diagnosis: &'static str,
    actual_artifact: ProjectionArtifactRecord,
}

#[derive(Debug, Serialize)]
struct ProjectionArtifactRecord {
    input_f32le: String,
    activation_values: String,
    activation_scales_f32le: String,
    ck_output_f32le: String,
    handwritten_output_f32le: String,
}

#[derive(Debug, Serialize)]
struct ProjectionDifferential {
    elements: usize,
    bitwise_mismatches: usize,
    first_mismatch: Option<usize>,
    max_abs: f32,
    max_rel: f64,
    ck_sha256: String,
    handwritten_sha256: String,
}

#[derive(Debug, Serialize)]
struct K128PrefixDifferential {
    prefix_blocks: usize,
    prefix_k_elements: usize,
    differential: ProjectionDifferential,
    ck_prefix_f32_le_sha256: String,
    handwritten_prefix_f32_le_sha256: String,
    ck_prefix_f32le: String,
    handwritten_prefix_f32le: String,
}

#[derive(Debug, Serialize)]
struct K128BlockDifferential {
    block_index: usize,
    k_start: usize,
    differential: ProjectionDifferential,
    ck_block_f32le: String,
    handwritten_block_f32le: String,
}

#[derive(Debug, Serialize)]
struct K16PrefixDifferential {
    k128_block_index: usize,
    prefix_subtiles: usize,
    prefix_k_elements_within_block: usize,
    differential: ProjectionDifferential,
    ck_prefix_f32le: String,
    handwritten_prefix_f32le: String,
}

#[derive(Debug, Serialize)]
struct FragmentLaneProbe {
    method: &'static str,
    cases: Vec<FragmentLaneCase>,
    passed: bool,
}

#[derive(Debug, Serialize)]
struct FragmentLaneCase {
    k_lane: usize,
    differential: ProjectionDifferential,
}

#[derive(Debug, Serialize)]
struct Interpretation {
    ck_scale_contract_from_source: &'static str,
    observation_boundary: &'static str,
    fragment_lane_conclusion: &'static str,
    k128_conclusion: &'static str,
    full_model_status: &'static str,
}

#[derive(Clone, Copy)]
enum ProjectionKind {
    Q,
    K,
    V,
    O,
    Gate,
    Up,
    Down,
}

impl ProjectionKind {
    const fn name(self) -> &'static str {
        match self {
            Self::Q => "q",
            Self::K => "k",
            Self::V => "v",
            Self::O => "o",
            Self::Gate => "gate",
            Self::Up => "up",
            Self::Down => "down",
        }
    }

    fn from_stage(stage: &str) -> Option<Self> {
        match stage {
            "q_projected" => Some(Self::Q),
            "k_projected" => Some(Self::K),
            "v_projected" => Some(Self::V),
            "o_projected" => Some(Self::O),
            "gate_projected" => Some(Self::Gate),
            "up_projected" => Some(Self::Up),
            "down_projected" => Some(Self::Down),
            _ => None,
        }
    }
}

#[derive(Debug)]
struct ServingTraceCapture {
    prefill_generated_token: usize,
    input_token_id: usize,
    position: usize,
    profile: String,
    layers: Vec<Sq8LayerRuntimeTrace>,
}

fn main() -> Result<(), String> {
    let options = parse_options()?;
    std::fs::create_dir(&options.output).map_err(|err| {
        format!(
            "failed to create fresh output directory {}: {err}",
            options.output.display()
        )
    })?;

    let artifact = read_sq8_canonical_artifact(&options.artifact)?;
    let prompt_token_bytes = std::fs::read(&options.prompt_token_ids_u32le).map_err(|err| {
        format!(
            "failed to read prompt token ids {}: {err}",
            options.prompt_token_ids_u32le.display()
        )
    })?;
    let prompt_token_ids = parse_prompt_token_ids(&prompt_token_bytes)?;
    let (runtime_index, device) = isolated_gfx1201_device()?;
    let ck_capture = capture_actual_decode(
        &artifact,
        &options.package,
        &prompt_token_ids,
        runtime_index,
        false,
    )?;
    let handwritten_capture = capture_actual_decode(
        &artifact,
        &options.package,
        &prompt_token_ids,
        runtime_index,
        true,
    )?;
    if ck_capture.input_token_id != handwritten_capture.input_token_id
        || ck_capture.position != handwritten_capture.position
    {
        return Err(format!(
            "actual serving decode routes differ before the projection experiment: CK token/position={}/{} handwritten={}/{}",
            ck_capture.input_token_id,
            ck_capture.position,
            handwritten_capture.input_token_id,
            handwritten_capture.position
        ));
    }
    let layer_stage_differentials = layer_stage_differentials(&ck_capture, &handwritten_capture)?;
    let first_stage_divergence = first_stage_divergence(&layer_stage_differentials);
    let selected = first_stage_divergence.as_ref().and_then(|first| {
        ProjectionKind::from_stage(&first.stage).map(|kind| (first.layer_index, kind))
    });

    let mut context = RuntimeContext::create(runtime_index)?;
    let mut stream = context.create_stream()?;
    let selected_projection = match selected {
        Some((layer_index, kind)) => {
            let norms = read_norms(&options.package, layer_index)?;
            let weights = load_qwen3_14b_sq8_layer_weights(
                &mut context,
                &mut stream,
                &artifact,
                layer_index,
                &norms,
                UPLOAD_CHUNK_BYTES,
            )?;
            Some(run_projection_contract(
                kind,
                &weights,
                &ck_capture.layers[layer_index],
                &handwritten_capture.layers[layer_index],
                &mut context,
                &mut stream,
                &options.output,
                layer_index,
            )?)
        }
        None => None,
    };
    let fragment_lane_probe = run_fragment_lane_probe(&mut context, &mut stream)?;

    let report = ContractReport {
        schema_version: SCHEMA_VERSION,
        scope: "private R9700/gfx1201 diagnostic; no default dispatch, manifest, campaign, authorization, or release mutation",
        source: SourceRecord {
            artifact_content_sha256: artifact.manifest().integrity.content_sha256.clone(),
            prompt_token_count: prompt_token_ids.len(),
            prompt_u32le_sha256: bytes_sha256(&prompt_token_bytes),
            prefill_mode: "m8-chunk8",
            max_new_tokens: GATE_MAX_NEW_TOKENS,
        },
        device: DeviceRecord {
            runtime_index,
            backend_device_id: device.device_id,
            backend: device.backend,
            name: device.name,
            gcn_arch_name: device.gcn_arch_name,
            compute_major: device.compute_major,
            compute_minor: device.compute_minor,
        },
        actual_decode_route: ActualDecodeRoute {
            ck_prefill_generated_token: ck_capture.prefill_generated_token,
            handwritten_prefill_generated_token: handwritten_capture.prefill_generated_token,
            ck_input_token_id: ck_capture.input_token_id,
            handwritten_input_token_id: handwritten_capture.input_token_id,
            ck_position: ck_capture.position,
            handwritten_position: handwritten_capture.position,
            ck_profile: ck_capture.profile.clone(),
            handwritten_profile: handwritten_capture.profile.clone(),
            method: "Both sessions use the frozen full-model gate's explicit 512-token prompt, max_new_tokens=4, and FixedM8Chunks prefill. The terminal test-only API runs the normal first feedback M=1 decode through all 40 layers, but reads each layer workspace after execution before head/token commit.",
        },
        layer_stage_differentials,
        first_stage_divergence,
        selected_projection,
        fragment_lane_probe,
        interpretation: Interpretation {
            ck_scale_contract_from_source: "CK blockwise_gemm_pipeline_xdlops_v1_ab_scale accumulates each ScaleBlockK=128 raw fragment, multiplies that partial by activation_scale * weight_scale in FP32, then adds it to the FP32 C accumulator.",
            observation_boundary: "Every replay comparison is taken after CK's real BF16 workspace boundary and BF16-to-F32 conversion. K128 prefixes are independently replayed with all later input K128 blocks quantized as zero, which preserves the active prefix and observes its cumulative result at that boundary.",
            fragment_lane_conclusion: "The lane probe is evidence only for the tested 16 source-K lanes and first output tile; a passing result rules out a gross fragment/lane transpose for that tile but does not prove the opaque hardware reduction association.",
            k128_conclusion: "A mismatch in an isolated K128 block proves that the discrepancy is already present before association among different K128 blocks; K128-to-K128 scale accumulation is therefore not its sole cause. A cumulative-prefix mismatch with every isolated block exact would instead evidence inter-K128 association. Prefix outcomes can be non-monotonic after the BF16 boundary, so they do not by themselves identify a unique K16/lane mapping.",
            full_model_status: "Candidate timing and default promotion remain prohibited unless the unchanged full-model multi-step gate passes.",
        },
    };
    write_json_create_new(&options.output.join("report.json"), &report)?;
    println!(
        "wrote={} first_stage_divergence={}",
        options.output.join("report.json").display(),
        report
            .first_stage_divergence
            .as_ref()
            .map_or("none", |first| first.stage.as_str())
    );
    Ok(())
}

fn parse_options() -> Result<Options, String> {
    let mut artifact = None;
    let mut package = None;
    let mut prompt_token_ids_u32le = None;
    let mut output = None;
    let mut args = std::env::args_os().skip(1);
    while let Some(argument) = args.next() {
        match argument.to_string_lossy().as_ref() {
            "--artifact" => artifact = Some(PathBuf::from(next_arg(&mut args, "--artifact")?)),
            "--package" => package = Some(PathBuf::from(next_arg(&mut args, "--package")?)),
            "--prompt-token-ids-u32le" => {
                prompt_token_ids_u32le = Some(PathBuf::from(next_arg(
                    &mut args,
                    "--prompt-token-ids-u32le",
                )?))
            }
            "--output" => output = Some(PathBuf::from(next_arg(&mut args, "--output")?)),
            _ => return Err(usage()),
        }
    }
    let artifact = artifact.ok_or_else(usage)?;
    let package = package.ok_or_else(usage)?;
    let prompt_token_ids_u32le = prompt_token_ids_u32le.ok_or_else(usage)?;
    let output = output.ok_or_else(usage)?;
    if output.exists() {
        return Err(format!(
            "output directory already exists: {}",
            output.display()
        ));
    }
    Ok(Options {
        artifact,
        package,
        prompt_token_ids_u32le,
        output,
    })
}

fn next_arg(
    args: &mut impl Iterator<Item = std::ffi::OsString>,
    option: &str,
) -> Result<std::ffi::OsString, String> {
    args.next()
        .ok_or_else(|| format!("{option} requires a value"))
}

fn usage() -> String {
    "usage: sq8_handwritten_projection_contract --artifact ARTIFACT_DIR --package THIN_PACKAGE --prompt-token-ids-u32le PROMPT_U32LE --output NEW_DIRECTORY".to_string()
}

fn read_norms(package: &std::path::Path, layer: usize) -> Result<Qwen3Sq8LayerNormValues, String> {
    let prefix = format!("model.layers.{layer}");
    Ok(Qwen3Sq8LayerNormValues {
        input: read_named_passthrough_f32(
            package,
            &format!("{prefix}.input_layernorm.weight"),
            UPLOAD_CHUNK_BYTES,
        )?
        .values,
        post_attention: read_named_passthrough_f32(
            package,
            &format!("{prefix}.post_attention_layernorm.weight"),
            UPLOAD_CHUNK_BYTES,
        )?
        .values,
        q: read_named_passthrough_f32(
            package,
            &format!("{prefix}.self_attn.q_norm.weight"),
            UPLOAD_CHUNK_BYTES,
        )?
        .values,
        k: read_named_passthrough_f32(
            package,
            &format!("{prefix}.self_attn.k_norm.weight"),
            UPLOAD_CHUNK_BYTES,
        )?
        .values,
    })
}

fn parse_prompt_token_ids(bytes: &[u8]) -> Result<Vec<usize>, String> {
    if bytes.is_empty() || !bytes.len().is_multiple_of(std::mem::size_of::<u32>()) {
        return Err(format!(
            "prompt token payload must be a nonempty u32le stream, got {} bytes",
            bytes.len()
        ));
    }
    Ok(bytes
        .chunks_exact(std::mem::size_of::<u32>())
        .map(|chunk| {
            usize::try_from(u32::from_le_bytes(
                chunk.try_into().expect("four u32 bytes"),
            ))
            .expect("u32 always fits usize on supported host")
        })
        .collect())
}

fn capture_actual_decode(
    artifact: &Sq8CanonicalArtifact,
    package: &std::path::Path,
    prompt_token_ids: &[usize],
    runtime_index: u32,
    handwritten: bool,
) -> Result<ServingTraceCapture, String> {
    let norms = load_qwen3_14b_sq8_serving_norms(package, UPLOAD_CHUNK_BYTES)
        .map_err(|err| err.to_string())?;
    let mut context = RuntimeContext::create(runtime_index)?;
    let mut stream = context.create_stream()?;
    let mut session = Qwen3Sq8ServingSession::load_with_prefill_mode(
        &mut context,
        &mut stream,
        artifact,
        package,
        norms,
        UPLOAD_CHUNK_BYTES,
        Sq8ServingPrefillMode::FixedM8Chunks,
    )
    .map_err(|err| err.to_string())?;
    if handwritten {
        session
            .enable_handwritten_wmma_projection_prototype()
            .map_err(|err| err.to_string())?;
    }
    session
        .start(
            &mut context,
            Sq8ServingRequest::greedy(
                if handwritten {
                    "sq8-projection-contract-handwritten"
                } else {
                    "sq8-projection-contract-ck"
                },
                prompt_token_ids.to_vec(),
                GATE_MAX_NEW_TOKENS,
            ),
            Sq8CancellationToken::new(),
            &mut stream,
        )
        .map_err(|err| err.to_string())?;
    let mut prefill_generated_token = None;
    while session.status() == Sq8ServingRuntimeStatus::Prefilling {
        match session
            .advance_synchronized(&mut stream)
            .map_err(|err| err.to_string())?
        {
            Sq8ServingAdvance::PromptProgress { .. } => {}
            Sq8ServingAdvance::Token {
                token_id,
                generated_index,
                terminal_reason,
                ..
            } => {
                if generated_index != 0
                    || terminal_reason.is_some()
                    || prefill_generated_token.is_some()
                {
                    return Err(format!(
                        "unexpected prefill token state: token={token_id} generated_index={generated_index} terminal={terminal_reason:?}"
                    ));
                }
                prefill_generated_token = Some(token_id);
            }
            Sq8ServingAdvance::CancellationObserved => {
                return Err("actual serving capture was cancelled during prefill".into());
            }
        }
    }
    if session.status() != Sq8ServingRuntimeStatus::Decoding {
        return Err(format!(
            "actual serving capture did not enter Decoding after prefill: {:?}",
            session.status()
        ));
    }
    let trace = session
        .trace_next_decode_layers_for_testing_synchronized(&mut stream)
        .map_err(|err| err.to_string())?;
    let result = ServingTraceCapture {
        prefill_generated_token: prefill_generated_token.ok_or_else(|| {
            "actual serving capture produced no prefill feedback token".to_string()
        })?,
        input_token_id: trace.input_token_id,
        position: trace.position,
        profile: format!("{:?}", trace.profile),
        layers: trace.layers,
    };
    drop(session);
    drop(stream);
    drop(context);
    Ok(result)
}

fn isolated_gfx1201_device() -> Result<(u32, DeviceInfo), String> {
    let count = device_count()?;
    let mut selected = None;
    for index in 0..count {
        let info = device_info(index)?;
        if info.backend == "hip"
            && info.gcn_arch_name.starts_with("gfx1201")
            && info.compute_major == 12
            && info.compute_minor == 0
        {
            if selected.replace((index, info)).is_some() {
                return Err("diagnostic requires exactly one visible gfx1201 HIP device".into());
            }
        }
    }
    selected.ok_or_else(|| "diagnostic requires one visible gfx1201 HIP device".into())
}

fn stage_differentials(
    ck: &Sq8LayerRuntimeTrace,
    handwritten: &Sq8LayerRuntimeTrace,
) -> Vec<StageDifferential> {
    [
        ("input_normed", &ck.input_normed, &handwritten.input_normed),
        ("q_projected", &ck.q_projected, &handwritten.q_projected),
        ("k_projected", &ck.k_projected, &handwritten.k_projected),
        ("v_projected", &ck.v_projected, &handwritten.v_projected),
        ("q_normed", &ck.q_normed, &handwritten.q_normed),
        ("k_normed", &ck.k_normed, &handwritten.k_normed),
        ("q_rope", &ck.q_rope, &handwritten.q_rope),
        ("k_rope", &ck.k_rope, &handwritten.k_rope),
        ("attention", &ck.attention, &handwritten.attention),
        ("o_projected", &ck.o_projected, &handwritten.o_projected),
        (
            "attention_residual",
            &ck.attention_residual,
            &handwritten.attention_residual,
        ),
        ("post_normed", &ck.post_normed, &handwritten.post_normed),
        (
            "gate_projected",
            &ck.gate_projected,
            &handwritten.gate_projected,
        ),
        ("up_projected", &ck.up_projected, &handwritten.up_projected),
        (
            "mlp_activation",
            &ck.mlp_activation,
            &handwritten.mlp_activation,
        ),
        (
            "down_projected",
            &ck.down_projected,
            &handwritten.down_projected,
        ),
        ("output", &ck.output, &handwritten.output),
    ]
    .into_iter()
    .map(|(stage, left, right)| stage_differential(stage, left, right))
    .collect()
}

fn layer_stage_differentials(
    ck: &ServingTraceCapture,
    handwritten: &ServingTraceCapture,
) -> Result<Vec<LayerStageDifferential>, String> {
    if ck.layers.len() != handwritten.layers.len() {
        return Err(format!(
            "actual serving layer trace count differs: CK={} handwritten={}",
            ck.layers.len(),
            handwritten.layers.len()
        ));
    }
    Ok(ck
        .layers
        .iter()
        .zip(&handwritten.layers)
        .enumerate()
        .map(|(layer_index, (ck, handwritten))| LayerStageDifferential {
            layer_index,
            stages: stage_differentials(ck, handwritten),
        })
        .collect())
}

fn stage_differential(stage: &'static str, ck: &[f32], handwritten: &[f32]) -> StageDifferential {
    let differential = projection_differential(ck, handwritten);
    StageDifferential {
        stage,
        elements: differential.elements,
        bitwise_mismatches: differential.bitwise_mismatches,
        first_mismatch: differential.first_mismatch,
        max_abs: differential.max_abs,
        max_rel: differential.max_rel,
        ck_sha256: differential.ck_sha256,
        handwritten_sha256: differential.handwritten_sha256,
    }
}

fn first_stage_divergence(layers: &[LayerStageDifferential]) -> Option<FirstStageDivergence> {
    layers.iter().find_map(|layer| {
        layer
            .stages
            .iter()
            .find(|stage| stage.bitwise_mismatches != 0)
            .map(|stage| FirstStageDivergence {
                layer_index: layer.layer_index,
                stage: stage.stage.to_string(),
                bitwise_mismatches: stage.bitwise_mismatches,
                first_mismatch: stage.first_mismatch,
                max_abs: stage.max_abs,
            })
    })
}

fn run_projection_contract(
    kind: ProjectionKind,
    weights: &ullm_engine::sq8_layer_runtime::Qwen3Sq8LayerWeights,
    ck_trace: &Sq8LayerRuntimeTrace,
    handwritten_trace: &Sq8LayerRuntimeTrace,
    context: &mut RuntimeContext,
    stream: &mut RuntimeStream,
    output_root: &PathBuf,
    layer_index: usize,
) -> Result<ProjectionContractReport, String> {
    let (weight, activation_trace, activation_input, expected_output) =
        projection_inputs(kind, weights, ck_trace)?;
    let handwritten_expected = projection_output(kind, handwritten_trace);
    let n = weight.rows;
    let k = weight.cols;
    if k % SCALE_BLOCK != 0 || activation_input.len() != k || expected_output.len() != n {
        return Err(format!(
            "{} diagnostic shape mismatch: n={} k={} input={} expected={}",
            kind.name(),
            n,
            k,
            activation_input.len(),
            expected_output.len()
        ));
    }
    let mut input = context.alloc_buffer(k * std::mem::size_of::<f32>())?;
    let mut activation = Sq8CkQuantizedActivation::allocate(context, 1, k)?;
    let (workspace_bytes, output_bytes) = sq8_ck_projection_buffer_bytes(1, n)?;
    let mut ck_workspace = context.alloc_buffer(workspace_bytes)?;
    let mut ck_output = context.alloc_buffer(output_bytes)?;
    let mut handwritten_output = context.alloc_buffer(output_bytes)?;

    upload_f32(&mut input, activation_input, stream)?;
    activation.quantize_f32(&input, Some(stream))?;
    let activation_replay_matches_layer_trace =
        activation_matches_trace(&activation, activation_trace, stream)?;
    let (ck_full, handwritten_full) = run_pair(
        &activation,
        weight,
        n,
        &mut ck_workspace,
        &mut ck_output,
        &mut handwritten_output,
        stream,
    )?;
    let direct_replay = projection_differential(&ck_full, &handwritten_full);
    let direct_replay_matches_layer_trace = f32_bitwise_equal(&ck_full, expected_output)
        && f32_bitwise_equal(&handwritten_full, handwritten_expected);

    let actual_directory = output_root.join("actual-artifact");
    std::fs::create_dir(&actual_directory).map_err(|err| {
        format!(
            "failed to create actual artifact directory {}: {err}",
            actual_directory.display()
        )
    })?;
    let artifact_stem = format!("layer{layer_index:02}-{}", kind.name());
    let input_relative = format!("actual-artifact/{artifact_stem}.input.f32le");
    let activation_values_relative =
        format!("actual-artifact/{artifact_stem}.activation.values.u8");
    let activation_scales_relative =
        format!("actual-artifact/{artifact_stem}.activation.scales.f32le");
    let ck_output_relative = format!("actual-artifact/{artifact_stem}.ck-output.f32le");
    let handwritten_output_relative =
        format!("actual-artifact/{artifact_stem}.handwritten-output.f32le");
    write_f32_create_new(&output_root.join(&input_relative), activation_input)?;
    write_bytes_create_new(
        &output_root.join(&activation_values_relative),
        &activation_trace.values,
    )?;
    write_f32_create_new(
        &output_root.join(&activation_scales_relative),
        &activation_trace.scales,
    )?;
    write_f32_create_new(&output_root.join(&ck_output_relative), &ck_full)?;
    write_f32_create_new(
        &output_root.join(&handwritten_output_relative),
        &handwritten_full,
    )?;

    let prefix_root = output_root.join("k128-prefixes");
    std::fs::create_dir(&prefix_root).map_err(|err| {
        format!(
            "failed to create K128 prefix directory {}: {err}",
            prefix_root.display()
        )
    })?;
    let prefix_directory_name = format!("layer{layer_index:02}-{}", kind.name());
    let prefix_directory = prefix_root.join(&prefix_directory_name);
    std::fs::create_dir(&prefix_directory).map_err(|err| {
        format!(
            "failed to create K128 projection directory {}: {err}",
            prefix_directory.display()
        )
    })?;

    let mut prefixes = Vec::with_capacity(k / SCALE_BLOCK);
    let mut first_mismatch_prefix_blocks = None;
    for prefix_blocks in 1..=k / SCALE_BLOCK {
        let mut prefix_input = activation_input.to_vec();
        prefix_input[prefix_blocks * SCALE_BLOCK..].fill(0.0);
        upload_f32(&mut input, &prefix_input, stream)?;
        activation.quantize_f32(&input, Some(stream))?;
        let (ck_prefix, handwritten_prefix) = run_pair(
            &activation,
            weight,
            n,
            &mut ck_workspace,
            &mut ck_output,
            &mut handwritten_output,
            stream,
        )?;
        let differential = projection_differential(&ck_prefix, &handwritten_prefix);
        if differential.bitwise_mismatches != 0 && first_mismatch_prefix_blocks.is_none() {
            first_mismatch_prefix_blocks = Some(prefix_blocks);
        }
        let ck_prefix_f32le =
            format!("k128-prefixes/{prefix_directory_name}/prefix-{prefix_blocks:03}.ck.f32le");
        let handwritten_prefix_f32le = format!(
            "k128-prefixes/{prefix_directory_name}/prefix-{prefix_blocks:03}.handwritten.f32le"
        );
        write_f32_create_new(&output_root.join(&ck_prefix_f32le), &ck_prefix)?;
        write_f32_create_new(
            &output_root.join(&handwritten_prefix_f32le),
            &handwritten_prefix,
        )?;
        prefixes.push(K128PrefixDifferential {
            prefix_blocks,
            prefix_k_elements: prefix_blocks * SCALE_BLOCK,
            ck_prefix_f32_le_sha256: differential.ck_sha256.clone(),
            handwritten_prefix_f32_le_sha256: differential.handwritten_sha256.clone(),
            ck_prefix_f32le,
            handwritten_prefix_f32le,
            differential,
        });
    }

    let single_blocks_root = output_root.join("k128-single-blocks");
    std::fs::create_dir(&single_blocks_root).map_err(|err| {
        format!(
            "failed to create K128 single-block directory {}: {err}",
            single_blocks_root.display()
        )
    })?;
    let single_blocks_directory_name = format!("layer{layer_index:02}-{}", kind.name());
    let single_blocks_directory = single_blocks_root.join(&single_blocks_directory_name);
    std::fs::create_dir(&single_blocks_directory).map_err(|err| {
        format!(
            "failed to create K128 single-block projection directory {}: {err}",
            single_blocks_directory.display()
        )
    })?;
    let mut single_blocks = Vec::with_capacity(k / SCALE_BLOCK);
    let mut first_mismatching_single_block = None;
    for block_index in 0..k / SCALE_BLOCK {
        let k_start = block_index * SCALE_BLOCK;
        let mut block_input = vec![0.0_f32; k];
        block_input[k_start..k_start + SCALE_BLOCK]
            .copy_from_slice(&activation_input[k_start..k_start + SCALE_BLOCK]);
        upload_f32(&mut input, &block_input, stream)?;
        activation.quantize_f32(&input, Some(stream))?;
        let (ck_block, handwritten_block) = run_pair(
            &activation,
            weight,
            n,
            &mut ck_workspace,
            &mut ck_output,
            &mut handwritten_output,
            stream,
        )?;
        let differential = projection_differential(&ck_block, &handwritten_block);
        if differential.bitwise_mismatches != 0 && first_mismatching_single_block.is_none() {
            first_mismatching_single_block = Some(block_index);
        }
        let ck_block_f32le = format!(
            "k128-single-blocks/{single_blocks_directory_name}/block-{block_index:03}.ck.f32le"
        );
        let handwritten_block_f32le = format!(
            "k128-single-blocks/{single_blocks_directory_name}/block-{block_index:03}.handwritten.f32le"
        );
        write_f32_create_new(&output_root.join(&ck_block_f32le), &ck_block)?;
        write_f32_create_new(
            &output_root.join(&handwritten_block_f32le),
            &handwritten_block,
        )?;
        single_blocks.push(K128BlockDifferential {
            block_index,
            k_start,
            differential,
            ck_block_f32le,
            handwritten_block_f32le,
        });
    }

    let mut k16_prefixes_for_first_mismatching_single_block = Vec::new();
    if let Some(block_index) = first_mismatching_single_block {
        let k16_root = output_root.join("k16-prefixes");
        std::fs::create_dir(&k16_root).map_err(|err| {
            format!(
                "failed to create K16 prefix directory {}: {err}",
                k16_root.display()
            )
        })?;
        let k16_directory_name = format!(
            "layer{layer_index:02}-{}-block{block_index:03}",
            kind.name()
        );
        let k16_directory = k16_root.join(&k16_directory_name);
        std::fs::create_dir(&k16_directory).map_err(|err| {
            format!(
                "failed to create K16 prefix projection directory {}: {err}",
                k16_directory.display()
            )
        })?;
        let k_start = block_index * SCALE_BLOCK;
        for prefix_subtiles in 1..=SCALE_BLOCK / 16 {
            let prefix_k_elements_within_block = prefix_subtiles * 16;
            let mut k16_input = vec![0.0_f32; k];
            k16_input[k_start..k_start + prefix_k_elements_within_block].copy_from_slice(
                &activation_input[k_start..k_start + prefix_k_elements_within_block],
            );
            upload_f32(&mut input, &k16_input, stream)?;
            activation.quantize_f32(&input, Some(stream))?;
            let (ck_prefix, handwritten_prefix) = run_pair(
                &activation,
                weight,
                n,
                &mut ck_workspace,
                &mut ck_output,
                &mut handwritten_output,
                stream,
            )?;
            let differential = projection_differential(&ck_prefix, &handwritten_prefix);
            let ck_prefix_f32le =
                format!("k16-prefixes/{k16_directory_name}/prefix-{prefix_subtiles}.ck.f32le");
            let handwritten_prefix_f32le = format!(
                "k16-prefixes/{k16_directory_name}/prefix-{prefix_subtiles}.handwritten.f32le"
            );
            write_f32_create_new(&output_root.join(&ck_prefix_f32le), &ck_prefix)?;
            write_f32_create_new(
                &output_root.join(&handwritten_prefix_f32le),
                &handwritten_prefix,
            )?;
            k16_prefixes_for_first_mismatching_single_block.push(K16PrefixDifferential {
                k128_block_index: block_index,
                prefix_subtiles,
                prefix_k_elements_within_block,
                differential,
                ck_prefix_f32le,
                handwritten_prefix_f32le,
            });
        }
    }
    let contract_diagnosis = match (first_mismatching_single_block, first_mismatch_prefix_blocks) {
        (Some(_), _) => {
            "At least one isolated K128 block differs, proving that the discrepancy is present before association among different K128 blocks; K128-to-K128 scale accumulation is therefore not its sole cause."
        }
        (None, Some(_)) => {
            "Every isolated K128 block is bitwise exact while a cumulative prefix differs, proving that the discrepancy first arises from K128-to-K128 scale/accumulator association."
        }
        (None, None) => {
            "No K128 prefix or isolated K128 block differs for this selected projection replay."
        }
    };

    Ok(ProjectionContractReport {
        projection: kind.name(),
        m: 1,
        n,
        k,
        direct_replay,
        direct_replay_matches_layer_trace,
        activation_replay_matches_layer_trace,
        k128_prefix_method: "For prefix p, original f32 activation values in K128 blocks p..end are set to 0 before the existing CK quantizer. Quantization is block-local, so blocks < p reproduce the actual activation bytes/scales and later blocks contribute exact zero. CK and handwritten replay the unchanged full-N/full-K kernels, and their BF16-boundary output is captured after each cumulative prefix.",
        k128_prefixes: prefixes,
        first_mismatching_prefix_blocks: first_mismatch_prefix_blocks,
        k128_single_blocks: single_blocks,
        first_mismatching_single_block,
        k16_prefixes_for_first_mismatching_single_block,
        contract_diagnosis,
        actual_artifact: ProjectionArtifactRecord {
            input_f32le: input_relative,
            activation_values: activation_values_relative,
            activation_scales_f32le: activation_scales_relative,
            ck_output_f32le: ck_output_relative,
            handwritten_output_f32le: handwritten_output_relative,
        },
    })
}

fn projection_inputs<'a>(
    kind: ProjectionKind,
    weights: &'a ullm_engine::sq8_layer_runtime::Qwen3Sq8LayerWeights,
    trace: &'a Sq8LayerRuntimeTrace,
) -> Result<
    (
        &'a Sq8CanonicalResidentRuntimeTensor,
        &'a Sq8LayerQuantizedActivationTrace,
        &'a [f32],
        &'a [f32],
    ),
    String,
> {
    let selected = match kind {
        ProjectionKind::Q => (
            &weights.q,
            trace.qkv_activation.as_ref(),
            trace.input_normed.as_slice(),
            trace.q_projected.as_slice(),
        ),
        ProjectionKind::K => (
            &weights.k,
            trace.qkv_activation.as_ref(),
            trace.input_normed.as_slice(),
            trace.k_projected.as_slice(),
        ),
        ProjectionKind::V => (
            &weights.v,
            trace.qkv_activation.as_ref(),
            trace.input_normed.as_slice(),
            trace.v_projected.as_slice(),
        ),
        ProjectionKind::O => (
            &weights.o,
            trace.o_activation.as_ref(),
            trace.attention.as_slice(),
            trace.o_projected.as_slice(),
        ),
        ProjectionKind::Gate => (
            &weights.gate,
            trace.gate_up_activation.as_ref(),
            trace.post_normed.as_slice(),
            trace.gate_projected.as_slice(),
        ),
        ProjectionKind::Up => (
            &weights.up,
            trace.gate_up_activation.as_ref(),
            trace.post_normed.as_slice(),
            trace.up_projected.as_slice(),
        ),
        ProjectionKind::Down => (
            &weights.down,
            trace.down_activation.as_ref(),
            trace.mlp_activation.as_slice(),
            trace.down_projected.as_slice(),
        ),
    };
    Ok((
        selected.0,
        selected
            .1
            .ok_or_else(|| format!("{} is missing activation trace", kind.name()))?,
        selected.2,
        selected.3,
    ))
}

fn projection_output(kind: ProjectionKind, trace: &Sq8LayerRuntimeTrace) -> &[f32] {
    match kind {
        ProjectionKind::Q => &trace.q_projected,
        ProjectionKind::K => &trace.k_projected,
        ProjectionKind::V => &trace.v_projected,
        ProjectionKind::O => &trace.o_projected,
        ProjectionKind::Gate => &trace.gate_projected,
        ProjectionKind::Up => &trace.up_projected,
        ProjectionKind::Down => &trace.down_projected,
    }
}

fn run_pair(
    activation: &Sq8CkQuantizedActivation,
    weight: &Sq8CanonicalResidentRuntimeTensor,
    n: usize,
    ck_workspace: &mut RuntimeBuffer,
    ck_output: &mut RuntimeBuffer,
    handwritten_output: &mut RuntimeBuffer,
    stream: &mut RuntimeStream,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    sq8_ck_projection_f32(
        activation,
        &weight.payload_buffer,
        &weight.scale_buffer,
        n,
        ck_workspace,
        ck_output,
        Some(&mut *stream),
    )?;
    sq8_handwritten_gfx1201_m1_projection_f32(
        activation,
        &weight.payload_buffer,
        &weight.scale_buffer,
        n,
        handwritten_output,
        Some(&mut *stream),
    )?;
    Ok((
        read_f32(ck_output, n, stream)?,
        read_f32(handwritten_output, n, stream)?,
    ))
}

fn activation_matches_trace(
    activation: &Sq8CkQuantizedActivation,
    expected: &Sq8LayerQuantizedActivationTrace,
    stream: &mut RuntimeStream,
) -> Result<bool, String> {
    let mut bytes = vec![0_u8; activation.quantized_bytes()];
    activation
        .quantized_buffer()
        .copy_to_host(0, &mut bytes, Some(&mut *stream))?;
    let scales = read_f32_values(
        activation.scale_buffer(),
        activation.scale_bytes() / std::mem::size_of::<f32>(),
        stream,
    )?;
    Ok(bytes == expected.values && f32_bitwise_equal(&scales, &expected.scales))
}

fn run_fragment_lane_probe(
    context: &mut RuntimeContext,
    stream: &mut RuntimeStream,
) -> Result<FragmentLaneProbe, String> {
    let n = QWEN3_14B_HIDDEN_SIZE;
    let k = QWEN3_14B_HIDDEN_SIZE;
    let weight_bytes = n
        .checked_mul(k)
        .ok_or_else(|| "fragment lane weight byte size overflows".to_string())?;
    let mut payload = vec![0_u8; weight_bytes];
    let lane_codes = [
        0x30_u8, 0x38, 0x3c, 0x40, 0x44, 0x48, 0x4a, 0x4c, 0x4e, 0x50, 0x52, 0x54, 0x56, 0x58,
        0x5a, 0x5c,
    ];
    for k_lane in 0..16 {
        for (output, code) in lane_codes.iter().copied().enumerate() {
            payload[output * k + k_lane] = code;
        }
    }
    let scales = vec![1.0_f32; (n / SCALE_BLOCK) * (k / SCALE_BLOCK)];
    let mut weight = context.alloc_buffer(payload.len())?;
    weight.copy_from_host(0, &payload, Some(&mut *stream))?;
    let mut weight_scales = context.alloc_buffer(scales.len() * std::mem::size_of::<f32>())?;
    upload_f32(&mut weight_scales, &scales, stream)?;
    let mut input = context.alloc_buffer(k * std::mem::size_of::<f32>())?;
    let mut activation = Sq8CkQuantizedActivation::allocate(context, 1, k)?;
    let (workspace_bytes, output_bytes) = sq8_ck_projection_buffer_bytes(1, n)?;
    let mut ck_workspace = context.alloc_buffer(workspace_bytes)?;
    let mut ck_output = context.alloc_buffer(output_bytes)?;
    let mut handwritten_output = context.alloc_buffer(output_bytes)?;
    let mut cases = Vec::with_capacity(16);
    for k_lane in 0..16 {
        let mut one_hot = vec![0.0_f32; k];
        one_hot[k_lane] = 1.0;
        upload_f32(&mut input, &one_hot, stream)?;
        activation.quantize_f32(&input, Some(&mut *stream))?;
        let (ck, handwritten) = run_pair_with_buffers(
            &activation,
            &weight,
            &weight_scales,
            n,
            &mut ck_workspace,
            &mut ck_output,
            &mut handwritten_output,
            stream,
        )?;
        cases.push(FragmentLaneCase {
            k_lane,
            differential: projection_differential(&ck, &handwritten),
        });
    }
    let passed = cases
        .iter()
        .all(|case| case.differential.bitwise_mismatches == 0);
    Ok(FragmentLaneProbe {
        method: "For each K lane 0..15, dynamically quantize an otherwise zero M=1 input with exactly that lane nonzero. A synthetic N=5120/K=5120 weight has distinct finite FP8 values in output rows 0..15 at that same K lane and zero elsewhere; F32-expanded [128,128] scales are one. Compare CK and handwritten BF16-boundary output bitwise.",
        cases,
        passed,
    })
}

fn run_pair_with_buffers(
    activation: &Sq8CkQuantizedActivation,
    weight: &RuntimeBuffer,
    weight_scales: &RuntimeBuffer,
    n: usize,
    ck_workspace: &mut RuntimeBuffer,
    ck_output: &mut RuntimeBuffer,
    handwritten_output: &mut RuntimeBuffer,
    stream: &mut RuntimeStream,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    sq8_ck_projection_f32(
        activation,
        weight,
        weight_scales,
        n,
        ck_workspace,
        ck_output,
        Some(&mut *stream),
    )?;
    sq8_handwritten_gfx1201_m1_projection_f32(
        activation,
        weight,
        weight_scales,
        n,
        handwritten_output,
        Some(&mut *stream),
    )?;
    Ok((
        read_f32(ck_output, n, stream)?,
        read_f32(handwritten_output, n, stream)?,
    ))
}

fn upload_f32(
    buffer: &mut RuntimeBuffer,
    values: &[f32],
    stream: &mut RuntimeStream,
) -> Result<(), String> {
    buffer.copy_from_host(0, &encode_f32_to_bytes(values), Some(stream))
}

fn read_f32(
    buffer: &RuntimeBuffer,
    elements: usize,
    stream: &mut RuntimeStream,
) -> Result<Vec<f32>, String> {
    read_f32_values(buffer, elements, stream)
}

fn read_f32_values(
    buffer: &RuntimeBuffer,
    elements: usize,
    stream: &mut RuntimeStream,
) -> Result<Vec<f32>, String> {
    let mut bytes = vec![0_u8; elements * std::mem::size_of::<f32>()];
    buffer.copy_to_host(0, &mut bytes, Some(&mut *stream))?;
    stream.synchronize()?;
    Ok(bytes
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four f32 bytes")))
        .collect())
}

fn projection_differential(ck: &[f32], handwritten: &[f32]) -> ProjectionDifferential {
    assert_eq!(ck.len(), handwritten.len(), "differential input length");
    let mut bitwise_mismatches = 0_usize;
    let mut first_mismatch = None;
    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f64;
    for (index, (&left, &right)) in ck.iter().zip(handwritten).enumerate() {
        if left.to_bits() != right.to_bits() {
            bitwise_mismatches += 1;
            first_mismatch.get_or_insert(index);
        }
        let absolute = (left - right).abs();
        max_abs = max_abs.max(absolute);
        max_rel = max_rel.max(f64::from(absolute) / f64::from(left.abs().max(1.0e-30)));
    }
    ProjectionDifferential {
        elements: ck.len(),
        bitwise_mismatches,
        first_mismatch,
        max_abs,
        max_rel,
        ck_sha256: f32_sha256(ck),
        handwritten_sha256: f32_sha256(handwritten),
    }
}

fn f32_bitwise_equal(left: &[f32], right: &[f32]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

fn f32_sha256(values: &[f32]) -> String {
    let mut digest = Sha256::new();
    for value in values {
        digest.update(value.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn bytes_sha256(values: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(values);
    format!("{:x}", digest.finalize())
}

fn write_bytes_create_new(path: &PathBuf, value: &[u8]) -> Result<(), String> {
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|err| format!("failed to create {}: {err}", path.display()))?;
    let mut writer = BufWriter::new(file);
    writer
        .write_all(value)
        .and_then(|_| writer.flush())
        .map_err(|err| format!("failed to write {}: {err}", path.display()))?;
    let file: File = writer
        .into_inner()
        .map_err(|err| format!("failed to finish {}: {err}", path.display()))?;
    file.sync_all()
        .map_err(|err| format!("failed to sync {}: {err}", path.display()))
}

fn write_f32_create_new(path: &PathBuf, value: &[f32]) -> Result<(), String> {
    let mut bytes = Vec::with_capacity(value.len() * std::mem::size_of::<f32>());
    for value in value {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    write_bytes_create_new(path, &bytes)
}

fn write_json_create_new(path: &PathBuf, value: &impl Serialize) -> Result<(), String> {
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|err| format!("failed to create {}: {err}", path.display()))?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, value)
        .map_err(|err| format!("failed to serialize {}: {err}", path.display()))?;
    writer
        .write_all(b"\n")
        .and_then(|_| writer.flush())
        .map_err(|err| format!("failed to flush {}: {err}", path.display()))?;
    let file: File = writer
        .into_inner()
        .map_err(|err| format!("failed to finish {}: {err}", path.display()))?;
    file.sync_all()
        .map_err(|err| format!("failed to sync {}: {err}", path.display()))
}
