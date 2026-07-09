// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

use crate::sq_canonical::Sq8CanonicalArtifact;
use crate::sq_optimized_reference::{
    Sq8DynamicActivation, Sq8DynamicActivationHashes, quantize_sq8_dynamic_activation,
    run_sq8_optimized_reference_projection,
};
use crate::sq_reference::sq8_f32_le_sha256;
use serde::Serialize;
use std::collections::BTreeSet;

pub const SQ8_LAYER_ORACLE_TRACE_SCHEMA_VERSION: &str = "ullm.sq8.layer_oracle.v1";
pub const QWEN3_14B_HIDDEN_SIZE: usize = 5120;
pub const QWEN3_14B_Q_HEADS: usize = 40;
pub const QWEN3_14B_KV_HEADS: usize = 8;
pub const QWEN3_14B_HEAD_DIM: usize = 128;
pub const QWEN3_14B_VALUE_DIM: usize = 128;
pub const QWEN3_14B_INTERMEDIATE_SIZE: usize = 17_408;
pub const QWEN3_14B_RMS_NORM_EPSILON: f32 = 1.0e-6;
pub const QWEN3_14B_ROPE_THETA: f32 = 1_000_000.0;
pub const QWEN3_14B_SQ8_LAYER_ORACLE_MAX_SEQUENCE_LEN: usize = 128;
pub const MAX_EXACT_F32_INTEGER: usize = 1 << 24;

const QWEN3_14B_Q_WIDTH: usize = QWEN3_14B_Q_HEADS * QWEN3_14B_HEAD_DIM;
const QWEN3_14B_KV_WIDTH: usize = QWEN3_14B_KV_HEADS * QWEN3_14B_HEAD_DIM;

#[derive(Debug, Clone, Copy)]
pub struct Sq8LayerProjectionNames<'a> {
    pub q_proj: &'a str,
    pub k_proj: &'a str,
    pub v_proj: &'a str,
    pub o_proj: &'a str,
    pub gate_proj: &'a str,
    pub up_proj: &'a str,
    pub down_proj: &'a str,
}

impl<'a> Sq8LayerProjectionNames<'a> {
    fn entries(self) -> [(&'static str, &'a str); 7] {
        [
            ("q_proj", self.q_proj),
            ("k_proj", self.k_proj),
            ("v_proj", self.v_proj),
            ("o_proj", self.o_proj),
            ("gate_proj", self.gate_proj),
            ("up_proj", self.up_proj),
            ("down_proj", self.down_proj),
        ]
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Sq8LayerNormWeights<'a> {
    pub input: &'a [f32],
    pub post_attention: &'a [f32],
    pub q: &'a [f32],
    pub k: &'a [f32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Sq8LayerProjectionOutputContract {
    F32,
    Bf16RneThenF32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Sq8LayerOracleOptions {
    pub position_offset: usize,
    pub projection_output_contract: Sq8LayerProjectionOutputContract,
}

impl Default for Sq8LayerOracleOptions {
    fn default() -> Self {
        Self {
            position_offset: 0,
            projection_output_contract: Sq8LayerProjectionOutputContract::Bf16RneThenF32,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Sq8LayerOwnedProjectionNames {
    pub q_proj: String,
    pub k_proj: String,
    pub v_proj: String,
    pub o_proj: String,
    pub gate_proj: String,
    pub up_proj: String,
    pub down_proj: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Sq8LayerTensorTrace {
    pub shape: [usize; 2],
    pub f32_le_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Sq8LayerNormWeightTraces {
    pub input: Sq8LayerTensorTrace,
    pub post_attention: Sq8LayerTensorTrace,
    pub q: Sq8LayerTensorTrace,
    pub k: Sq8LayerTensorTrace,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Sq8LayerActivationTrace {
    pub shape: [usize; 2],
    pub block_cols: usize,
    pub scale_shape: [usize; 2],
    pub hashes: Sq8DynamicActivationHashes,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Sq8LayerActivationTraces {
    pub input_norm_qkv: Sq8LayerActivationTrace,
    pub attention_o: Sq8LayerActivationTrace,
    pub post_norm_gate_up: Sq8LayerActivationTrace,
    pub mlp_down: Sq8LayerActivationTrace,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Sq8LayerProjectionTrace {
    pub tensor: String,
    pub output_shape: [usize; 2],
    pub cpu_worker_threads: usize,
    pub raw_output_f32_le_sha256: String,
    pub contracted_output_f32_le_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Sq8LayerProjectionTraces {
    pub q_proj: Sq8LayerProjectionTrace,
    pub k_proj: Sq8LayerProjectionTrace,
    pub v_proj: Sq8LayerProjectionTrace,
    pub o_proj: Sq8LayerProjectionTrace,
    pub gate_proj: Sq8LayerProjectionTrace,
    pub up_proj: Sq8LayerProjectionTrace,
    pub down_proj: Sq8LayerProjectionTrace,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Sq8LayerIntermediateTensorTraces {
    pub input_hidden: Sq8LayerTensorTrace,
    pub input_norm: Sq8LayerTensorTrace,
    pub q_projected: Sq8LayerTensorTrace,
    pub k_projected: Sq8LayerTensorTrace,
    pub v_projected: Sq8LayerTensorTrace,
    pub q_norm: Sq8LayerTensorTrace,
    pub k_norm: Sq8LayerTensorTrace,
    pub q_rope: Sq8LayerTensorTrace,
    pub k_rope: Sq8LayerTensorTrace,
    pub attention: Sq8LayerTensorTrace,
    pub o_projected: Sq8LayerTensorTrace,
    pub attention_residual: Sq8LayerTensorTrace,
    pub post_attention_norm: Sq8LayerTensorTrace,
    pub gate_projected: Sq8LayerTensorTrace,
    pub up_projected: Sq8LayerTensorTrace,
    pub silu_gate_mul_up: Sq8LayerTensorTrace,
    pub down_projected: Sq8LayerTensorTrace,
    pub output_hidden: Sq8LayerTensorTrace,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Sq8LayerOracleTrace {
    pub schema_version: &'static str,
    pub artifact_content_sha256: String,
    pub sequence_len: usize,
    pub options: Sq8LayerOracleOptions,
    pub projection_tensors: Sq8LayerOwnedProjectionNames,
    pub norm_weights: Sq8LayerNormWeightTraces,
    pub activations: Sq8LayerActivationTraces,
    pub projections: Sq8LayerProjectionTraces,
    pub tensors: Sq8LayerIntermediateTensorTraces,
    pub output_f32_le_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Sq8LayerOracleIntermediates {
    pub input_norm: Vec<f32>,
    pub q_projected: Vec<f32>,
    pub k_projected: Vec<f32>,
    pub v_projected: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub q_rope: Vec<f32>,
    pub k_rope: Vec<f32>,
    pub attention: Vec<f32>,
    pub o_projected: Vec<f32>,
    pub attention_residual: Vec<f32>,
    pub post_attention_norm: Vec<f32>,
    pub gate_projected: Vec<f32>,
    pub up_projected: Vec<f32>,
    pub silu_gate_mul_up: Vec<f32>,
    pub down_projected: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Sq8LayerOracleOutput {
    pub shape: [usize; 2],
    pub output: Vec<f32>,
    pub intermediates: Sq8LayerOracleIntermediates,
    pub trace: Sq8LayerOracleTrace,
}

pub fn run_qwen3_14b_sq8_layer_oracle(
    artifact: &Sq8CanonicalArtifact,
    projection_names: Sq8LayerProjectionNames<'_>,
    norm_weights: Sq8LayerNormWeights<'_>,
    hidden: &[f32],
    sequence_len: usize,
) -> Result<Sq8LayerOracleOutput, String> {
    run_qwen3_14b_sq8_layer_oracle_with_options(
        artifact,
        projection_names,
        norm_weights,
        hidden,
        sequence_len,
        Sq8LayerOracleOptions::default(),
    )
}

pub fn run_qwen3_14b_sq8_layer_oracle_with_options(
    artifact: &Sq8CanonicalArtifact,
    projection_names: Sq8LayerProjectionNames<'_>,
    norm_weights: Sq8LayerNormWeights<'_>,
    hidden: &[f32],
    sequence_len: usize,
    options: Sq8LayerOracleOptions,
) -> Result<Sq8LayerOracleOutput, String> {
    validate_sequence_contract(sequence_len, options.position_offset)?;
    let hidden_elements = checked_elements(sequence_len, QWEN3_14B_HIDDEN_SIZE, "input hidden")?;
    if hidden.len() != hidden_elements {
        return Err(format!(
            "SQ8 layer oracle input hidden length mismatch: expected={hidden_elements} actual={}",
            hidden.len()
        ));
    }
    validate_finite(hidden, "input hidden")?;

    let projection_tensors = resolve_projection_tensors(artifact, projection_names)?;
    let norm_weight_traces = validate_and_trace_norm_weights(norm_weights)?;
    let input_hidden_trace =
        tensor_trace(hidden, sequence_len, QWEN3_14B_HIDDEN_SIZE, "input hidden")?;

    let input_norm = rmsnorm_rows(
        hidden,
        sequence_len,
        QWEN3_14B_HIDDEN_SIZE,
        norm_weights.input,
        QWEN3_14B_RMS_NORM_EPSILON,
        "input RMSNorm",
    )?;
    let input_norm_trace = tensor_trace(
        &input_norm,
        sequence_len,
        QWEN3_14B_HIDDEN_SIZE,
        "input RMSNorm",
    )?;
    let input_norm_values = input_norm.clone();
    let (qkv_activation, input_norm_qkv_activation_trace) = quantize_activation_traced(
        &input_norm,
        sequence_len,
        QWEN3_14B_HIDDEN_SIZE,
        "input-norm QKV activation",
    )?;
    drop(input_norm);

    let (q_projected, q_projection_trace) = run_projection_traced(
        artifact,
        &projection_tensors.q_proj,
        &qkv_activation,
        sequence_len,
        QWEN3_14B_Q_WIDTH,
        options.projection_output_contract,
        "q projection",
    )?;
    let q_projected_trace =
        tensor_trace(&q_projected, sequence_len, QWEN3_14B_Q_WIDTH, "q projected")?;
    let q_projected_values = q_projected.clone();
    let (k_projected, k_projection_trace) = run_projection_traced(
        artifact,
        &projection_tensors.k_proj,
        &qkv_activation,
        sequence_len,
        QWEN3_14B_KV_WIDTH,
        options.projection_output_contract,
        "k projection",
    )?;
    let k_projected_trace = tensor_trace(
        &k_projected,
        sequence_len,
        QWEN3_14B_KV_WIDTH,
        "k projected",
    )?;
    let k_projected_values = k_projected.clone();
    let (v_projected, v_projection_trace) = run_projection_traced(
        artifact,
        &projection_tensors.v_proj,
        &qkv_activation,
        sequence_len,
        QWEN3_14B_KV_WIDTH,
        options.projection_output_contract,
        "v projection",
    )?;
    let v_projected_trace = tensor_trace(
        &v_projected,
        sequence_len,
        QWEN3_14B_KV_WIDTH,
        "v projected",
    )?;
    let v_projected_values = v_projected.clone();
    drop(qkv_activation);

    let q_head_rows = sequence_len
        .checked_mul(QWEN3_14B_Q_HEADS)
        .ok_or_else(|| "SQ8 layer oracle q head row count overflows usize".to_string())?;
    let q_norm = rmsnorm_rows(
        &q_projected,
        q_head_rows,
        QWEN3_14B_HEAD_DIM,
        norm_weights.q,
        QWEN3_14B_RMS_NORM_EPSILON,
        "q headwise RMSNorm",
    )?;
    drop(q_projected);
    let q_norm_trace = tensor_trace(
        &q_norm,
        sequence_len,
        QWEN3_14B_Q_WIDTH,
        "q headwise RMSNorm",
    )?;
    let q_norm_values = q_norm.clone();

    let k_head_rows = sequence_len
        .checked_mul(QWEN3_14B_KV_HEADS)
        .ok_or_else(|| "SQ8 layer oracle k head row count overflows usize".to_string())?;
    let k_norm = rmsnorm_rows(
        &k_projected,
        k_head_rows,
        QWEN3_14B_HEAD_DIM,
        norm_weights.k,
        QWEN3_14B_RMS_NORM_EPSILON,
        "k headwise RMSNorm",
    )?;
    drop(k_projected);
    let k_norm_trace = tensor_trace(
        &k_norm,
        sequence_len,
        QWEN3_14B_KV_WIDTH,
        "k headwise RMSNorm",
    )?;
    let k_norm_values = k_norm.clone();

    let q_rope = rope_split_half(
        &q_norm,
        sequence_len,
        QWEN3_14B_Q_HEADS,
        QWEN3_14B_HEAD_DIM,
        QWEN3_14B_HEAD_DIM,
        options.position_offset,
        QWEN3_14B_ROPE_THETA,
        "q full RoPE",
    )?;
    drop(q_norm);
    let q_rope_trace = tensor_trace(&q_rope, sequence_len, QWEN3_14B_Q_WIDTH, "q full RoPE")?;
    let q_rope_values = q_rope.clone();
    let k_rope = rope_split_half(
        &k_norm,
        sequence_len,
        QWEN3_14B_KV_HEADS,
        QWEN3_14B_HEAD_DIM,
        QWEN3_14B_HEAD_DIM,
        options.position_offset,
        QWEN3_14B_ROPE_THETA,
        "k full RoPE",
    )?;
    drop(k_norm);
    let k_rope_trace = tensor_trace(&k_rope, sequence_len, QWEN3_14B_KV_WIDTH, "k full RoPE")?;
    let k_rope_values = k_rope.clone();

    let attention = causal_gqa_attention(
        &q_rope,
        &k_rope,
        &v_projected,
        sequence_len,
        QWEN3_14B_Q_HEADS,
        QWEN3_14B_KV_HEADS,
        QWEN3_14B_HEAD_DIM,
        QWEN3_14B_VALUE_DIM,
    )?;
    drop(q_rope);
    drop(k_rope);
    drop(v_projected);
    let attention_trace = tensor_trace(
        &attention,
        sequence_len,
        QWEN3_14B_HIDDEN_SIZE,
        "causal GQA attention",
    )?;
    let attention_values = attention.clone();
    let (attention_activation, attention_o_activation_trace) = quantize_activation_traced(
        &attention,
        sequence_len,
        QWEN3_14B_HIDDEN_SIZE,
        "attention o activation",
    )?;
    drop(attention);

    let (mut attention_residual, o_projection_trace) = run_projection_traced(
        artifact,
        &projection_tensors.o_proj,
        &attention_activation,
        sequence_len,
        QWEN3_14B_HIDDEN_SIZE,
        options.projection_output_contract,
        "o projection",
    )?;
    drop(attention_activation);
    let o_projected_trace = tensor_trace(
        &attention_residual,
        sequence_len,
        QWEN3_14B_HIDDEN_SIZE,
        "o projected",
    )?;
    let o_projected_values = attention_residual.clone();
    add_in_place(&mut attention_residual, hidden, "attention residual")?;
    let attention_residual_trace = tensor_trace(
        &attention_residual,
        sequence_len,
        QWEN3_14B_HIDDEN_SIZE,
        "attention residual",
    )?;
    let attention_residual_values = attention_residual.clone();

    let post_attention_norm = rmsnorm_rows(
        &attention_residual,
        sequence_len,
        QWEN3_14B_HIDDEN_SIZE,
        norm_weights.post_attention,
        QWEN3_14B_RMS_NORM_EPSILON,
        "post-attention RMSNorm",
    )?;
    let post_attention_norm_trace = tensor_trace(
        &post_attention_norm,
        sequence_len,
        QWEN3_14B_HIDDEN_SIZE,
        "post-attention RMSNorm",
    )?;
    let post_attention_norm_values = post_attention_norm.clone();
    let (gate_up_activation, post_norm_gate_up_activation_trace) = quantize_activation_traced(
        &post_attention_norm,
        sequence_len,
        QWEN3_14B_HIDDEN_SIZE,
        "post-norm gate-up activation",
    )?;
    drop(post_attention_norm);

    let (mut gate_projected, gate_projection_trace) = run_projection_traced(
        artifact,
        &projection_tensors.gate_proj,
        &gate_up_activation,
        sequence_len,
        QWEN3_14B_INTERMEDIATE_SIZE,
        options.projection_output_contract,
        "gate projection",
    )?;
    let gate_projected_trace = tensor_trace(
        &gate_projected,
        sequence_len,
        QWEN3_14B_INTERMEDIATE_SIZE,
        "gate projected",
    )?;
    let gate_projected_values = gate_projected.clone();
    let (up_projected, up_projection_trace) = run_projection_traced(
        artifact,
        &projection_tensors.up_proj,
        &gate_up_activation,
        sequence_len,
        QWEN3_14B_INTERMEDIATE_SIZE,
        options.projection_output_contract,
        "up projection",
    )?;
    drop(gate_up_activation);
    let up_projected_trace = tensor_trace(
        &up_projected,
        sequence_len,
        QWEN3_14B_INTERMEDIATE_SIZE,
        "up projected",
    )?;
    let up_projected_values = up_projected.clone();
    silu_mul_in_place(&mut gate_projected, &up_projected)?;
    drop(up_projected);
    let silu_gate_mul_up_trace = tensor_trace(
        &gate_projected,
        sequence_len,
        QWEN3_14B_INTERMEDIATE_SIZE,
        "SiLU gate multiplied by up",
    )?;
    let silu_gate_mul_up_values = gate_projected.clone();
    let (down_activation, mlp_down_activation_trace) = quantize_activation_traced(
        &gate_projected,
        sequence_len,
        QWEN3_14B_INTERMEDIATE_SIZE,
        "MLP down activation",
    )?;
    drop(gate_projected);

    let (mut output, down_projection_trace) = run_projection_traced(
        artifact,
        &projection_tensors.down_proj,
        &down_activation,
        sequence_len,
        QWEN3_14B_HIDDEN_SIZE,
        options.projection_output_contract,
        "down projection",
    )?;
    drop(down_activation);
    let down_projected_trace = tensor_trace(
        &output,
        sequence_len,
        QWEN3_14B_HIDDEN_SIZE,
        "down projected",
    )?;
    let down_projected_values = output.clone();
    add_in_place(&mut output, &attention_residual, "MLP residual")?;
    drop(attention_residual);
    let output_hidden_trace = tensor_trace(
        &output,
        sequence_len,
        QWEN3_14B_HIDDEN_SIZE,
        "output hidden",
    )?;
    let output_f32_le_sha256 = output_hidden_trace.f32_le_sha256.clone();

    Ok(Sq8LayerOracleOutput {
        shape: [sequence_len, QWEN3_14B_HIDDEN_SIZE],
        output,
        intermediates: Sq8LayerOracleIntermediates {
            input_norm: input_norm_values,
            q_projected: q_projected_values,
            k_projected: k_projected_values,
            v_projected: v_projected_values,
            q_norm: q_norm_values,
            k_norm: k_norm_values,
            q_rope: q_rope_values,
            k_rope: k_rope_values,
            attention: attention_values,
            o_projected: o_projected_values,
            attention_residual: attention_residual_values,
            post_attention_norm: post_attention_norm_values,
            gate_projected: gate_projected_values,
            up_projected: up_projected_values,
            silu_gate_mul_up: silu_gate_mul_up_values,
            down_projected: down_projected_values,
        },
        trace: Sq8LayerOracleTrace {
            schema_version: SQ8_LAYER_ORACLE_TRACE_SCHEMA_VERSION,
            artifact_content_sha256: artifact.manifest().integrity.content_sha256.clone(),
            sequence_len,
            options,
            projection_tensors,
            norm_weights: norm_weight_traces,
            activations: Sq8LayerActivationTraces {
                input_norm_qkv: input_norm_qkv_activation_trace,
                attention_o: attention_o_activation_trace,
                post_norm_gate_up: post_norm_gate_up_activation_trace,
                mlp_down: mlp_down_activation_trace,
            },
            projections: Sq8LayerProjectionTraces {
                q_proj: q_projection_trace,
                k_proj: k_projection_trace,
                v_proj: v_projection_trace,
                o_proj: o_projection_trace,
                gate_proj: gate_projection_trace,
                up_proj: up_projection_trace,
                down_proj: down_projection_trace,
            },
            tensors: Sq8LayerIntermediateTensorTraces {
                input_hidden: input_hidden_trace,
                input_norm: input_norm_trace,
                q_projected: q_projected_trace,
                k_projected: k_projected_trace,
                v_projected: v_projected_trace,
                q_norm: q_norm_trace,
                k_norm: k_norm_trace,
                q_rope: q_rope_trace,
                k_rope: k_rope_trace,
                attention: attention_trace,
                o_projected: o_projected_trace,
                attention_residual: attention_residual_trace,
                post_attention_norm: post_attention_norm_trace,
                gate_projected: gate_projected_trace,
                up_projected: up_projected_trace,
                silu_gate_mul_up: silu_gate_mul_up_trace,
                down_projected: down_projected_trace,
                output_hidden: output_hidden_trace,
            },
            output_f32_le_sha256,
        },
    })
}

fn validate_sequence_contract(sequence_len: usize, position_offset: usize) -> Result<(), String> {
    if sequence_len == 0 || sequence_len > QWEN3_14B_SQ8_LAYER_ORACLE_MAX_SEQUENCE_LEN {
        return Err(format!(
            "SQ8 layer oracle sequence_len must be in 1..={QWEN3_14B_SQ8_LAYER_ORACLE_MAX_SEQUENCE_LEN}, got {sequence_len}"
        ));
    }
    let final_position = position_offset
        .checked_add(sequence_len - 1)
        .ok_or_else(|| "SQ8 layer oracle final RoPE position overflows usize".to_string())?;
    if final_position > MAX_EXACT_F32_INTEGER {
        return Err(format!(
            "SQ8 layer oracle final RoPE position {final_position} exceeds the exact F32 integer limit {MAX_EXACT_F32_INTEGER}"
        ));
    }
    Ok(())
}

fn resolve_projection_tensors(
    artifact: &Sq8CanonicalArtifact,
    names: Sq8LayerProjectionNames<'_>,
) -> Result<Sq8LayerOwnedProjectionNames, String> {
    for (label, name) in names.entries() {
        if name.is_empty() {
            return Err(format!(
                "SQ8 layer oracle {label} tensor name must not be empty"
            ));
        }
    }
    let mut canonical_names = BTreeSet::new();
    let q_proj = resolve_projection_tensor(
        artifact,
        names.q_proj,
        QWEN3_14B_Q_WIDTH,
        QWEN3_14B_HIDDEN_SIZE,
        "q_proj",
        &mut canonical_names,
    )?;
    let k_proj = resolve_projection_tensor(
        artifact,
        names.k_proj,
        QWEN3_14B_KV_WIDTH,
        QWEN3_14B_HIDDEN_SIZE,
        "k_proj",
        &mut canonical_names,
    )?;
    let v_proj = resolve_projection_tensor(
        artifact,
        names.v_proj,
        QWEN3_14B_KV_WIDTH,
        QWEN3_14B_HIDDEN_SIZE,
        "v_proj",
        &mut canonical_names,
    )?;
    let o_proj = resolve_projection_tensor(
        artifact,
        names.o_proj,
        QWEN3_14B_HIDDEN_SIZE,
        QWEN3_14B_HIDDEN_SIZE,
        "o_proj",
        &mut canonical_names,
    )?;
    let gate_proj = resolve_projection_tensor(
        artifact,
        names.gate_proj,
        QWEN3_14B_INTERMEDIATE_SIZE,
        QWEN3_14B_HIDDEN_SIZE,
        "gate_proj",
        &mut canonical_names,
    )?;
    let up_proj = resolve_projection_tensor(
        artifact,
        names.up_proj,
        QWEN3_14B_INTERMEDIATE_SIZE,
        QWEN3_14B_HIDDEN_SIZE,
        "up_proj",
        &mut canonical_names,
    )?;
    let down_proj = resolve_projection_tensor(
        artifact,
        names.down_proj,
        QWEN3_14B_HIDDEN_SIZE,
        QWEN3_14B_INTERMEDIATE_SIZE,
        "down_proj",
        &mut canonical_names,
    )?;
    Ok(Sq8LayerOwnedProjectionNames {
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        gate_proj,
        up_proj,
        down_proj,
    })
}

fn resolve_projection_tensor(
    artifact: &Sq8CanonicalArtifact,
    requested_name: &str,
    expected_rows: usize,
    expected_cols: usize,
    label: &str,
    canonical_names: &mut BTreeSet<String>,
) -> Result<String, String> {
    let pair = artifact
        .tensor_pair(requested_name)
        .map_err(|err| format!("SQ8 layer oracle failed to resolve {label}: {err}"))?;
    let expected_shape = [expected_rows as u64, expected_cols as u64];
    if pair.shape != expected_shape {
        return Err(format!(
            "SQ8 layer oracle {label} tensor {} shape mismatch: expected={expected_shape:?} actual={:?}",
            pair.name, pair.shape
        ));
    }
    if !canonical_names.insert(pair.name.clone()) {
        return Err(format!(
            "SQ8 layer oracle projection tensor {} is selected more than once",
            pair.name
        ));
    }
    Ok(pair.name.clone())
}

fn validate_and_trace_norm_weights(
    weights: Sq8LayerNormWeights<'_>,
) -> Result<Sq8LayerNormWeightTraces, String> {
    Ok(Sq8LayerNormWeightTraces {
        input: tensor_trace(
            weights.input,
            1,
            QWEN3_14B_HIDDEN_SIZE,
            "input RMSNorm weight",
        )?,
        post_attention: tensor_trace(
            weights.post_attention,
            1,
            QWEN3_14B_HIDDEN_SIZE,
            "post-attention RMSNorm weight",
        )?,
        q: tensor_trace(weights.q, 1, QWEN3_14B_HEAD_DIM, "q RMSNorm weight")?,
        k: tensor_trace(weights.k, 1, QWEN3_14B_HEAD_DIM, "k RMSNorm weight")?,
    })
}

fn tensor_trace(
    values: &[f32],
    rows: usize,
    cols: usize,
    label: &str,
) -> Result<Sq8LayerTensorTrace, String> {
    let expected_elements = checked_elements(rows, cols, label)?;
    if values.len() != expected_elements {
        return Err(format!(
            "SQ8 layer oracle {label} tensor length mismatch: expected={expected_elements} actual={}",
            values.len()
        ));
    }
    let f32_le_sha256 = sq8_f32_le_sha256(values)
        .map_err(|err| format!("SQ8 layer oracle failed to hash {label}: {err}"))?;
    Ok(Sq8LayerTensorTrace {
        shape: [rows, cols],
        f32_le_sha256,
    })
}

fn quantize_activation_traced(
    input: &[f32],
    rows: usize,
    cols: usize,
    label: &str,
) -> Result<(Sq8DynamicActivation, Sq8LayerActivationTrace), String> {
    let input_trace = tensor_trace(input, rows, cols, label)?;
    let activation = quantize_sq8_dynamic_activation(input, rows, cols)
        .map_err(|err| format!("SQ8 layer oracle failed to quantize {label}: {err}"))?;
    let hashes = activation
        .hashes()
        .map_err(|err| format!("SQ8 layer oracle failed to hash {label}: {err}"))?;
    if hashes.input_f32_le_sha256 != input_trace.f32_le_sha256 {
        return Err(format!(
            "SQ8 layer oracle {label} quantizer input hash mismatch: tensor={} activation={}",
            input_trace.f32_le_sha256, hashes.input_f32_le_sha256
        ));
    }
    let trace = Sq8LayerActivationTrace {
        shape: [activation.rows(), activation.cols()],
        block_cols: activation.block_cols(),
        scale_shape: activation.scale_shape(),
        hashes,
    };
    Ok((activation, trace))
}

#[allow(clippy::too_many_arguments)]
fn run_projection_traced(
    artifact: &Sq8CanonicalArtifact,
    tensor_name: &str,
    activation: &Sq8DynamicActivation,
    expected_output_rows: usize,
    expected_output_cols: usize,
    output_contract: Sq8LayerProjectionOutputContract,
    label: &str,
) -> Result<(Vec<f32>, Sq8LayerProjectionTrace), String> {
    let projection = run_sq8_optimized_reference_projection(artifact, tensor_name, activation)
        .map_err(|err| format!("SQ8 layer oracle failed to run {label}: {err}"))?;
    if projection.tensor != tensor_name {
        return Err(format!(
            "SQ8 layer oracle {label} returned unexpected tensor: expected={tensor_name} actual={}",
            projection.tensor
        ));
    }
    if projection.output_rows != expected_output_rows
        || projection.output_cols != expected_output_cols
    {
        return Err(format!(
            "SQ8 layer oracle {label} output shape mismatch: expected=[{expected_output_rows},{expected_output_cols}] actual=[{},{}]",
            projection.output_rows, projection.output_cols
        ));
    }
    if projection.cpu_worker_threads == 0 {
        return Err(format!(
            "SQ8 layer oracle {label} reported zero CPU worker threads"
        ));
    }
    let raw_output_f32_le_sha256 = sq8_f32_le_sha256(&projection.output)
        .map_err(|err| format!("SQ8 layer oracle failed to hash raw {label} output: {err}"))?;
    let mut output = projection.output;
    apply_projection_output_contract(&mut output, output_contract, label)?;
    let contracted_output_f32_le_sha256 = sq8_f32_le_sha256(&output).map_err(|err| {
        format!("SQ8 layer oracle failed to hash contracted {label} output: {err}")
    })?;
    Ok((
        output,
        Sq8LayerProjectionTrace {
            tensor: tensor_name.to_string(),
            output_shape: [expected_output_rows, expected_output_cols],
            cpu_worker_threads: projection.cpu_worker_threads,
            raw_output_f32_le_sha256,
            contracted_output_f32_le_sha256,
        },
    ))
}

fn apply_projection_output_contract(
    output: &mut [f32],
    contract: Sq8LayerProjectionOutputContract,
    label: &str,
) -> Result<(), String> {
    match contract {
        Sq8LayerProjectionOutputContract::F32 => validate_finite(output, label),
        Sq8LayerProjectionOutputContract::Bf16RneThenF32 => round_slice_to_bf16_rne(output, label),
    }
}

fn checked_elements(rows: usize, cols: usize, label: &str) -> Result<usize, String> {
    if rows == 0 || cols == 0 {
        return Err(format!(
            "SQ8 layer oracle {label} shape must be non-zero, got [{rows},{cols}]"
        ));
    }
    rows.checked_mul(cols)
        .ok_or_else(|| format!("SQ8 layer oracle {label} shape [{rows},{cols}] overflows usize"))
}

fn validate_finite(values: &[f32], label: &str) -> Result<(), String> {
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!(
            "SQ8 layer oracle {label} contains non-finite value {value} at index {index}"
        ));
    }
    Ok(())
}

fn zeroed_f32(elements: usize, label: &str) -> Result<Vec<f32>, String> {
    let mut values = Vec::new();
    values.try_reserve_exact(elements).map_err(|err| {
        format!("SQ8 layer oracle failed to reserve {elements} F32 values for {label}: {err}")
    })?;
    values.resize(elements, 0.0);
    Ok(values)
}

fn rmsnorm_rows(
    input: &[f32],
    rows: usize,
    cols: usize,
    weight: &[f32],
    epsilon: f32,
    label: &str,
) -> Result<Vec<f32>, String> {
    let elements = checked_elements(rows, cols, label)?;
    if input.len() != elements {
        return Err(format!(
            "SQ8 layer oracle {label} input length mismatch: expected={elements} actual={}",
            input.len()
        ));
    }
    if weight.len() != cols {
        return Err(format!(
            "SQ8 layer oracle {label} weight length mismatch: expected={cols} actual={}",
            weight.len()
        ));
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(format!(
            "SQ8 layer oracle {label} epsilon must be finite and greater than zero, got {epsilon}"
        ));
    }
    validate_finite(input, &format!("{label} input"))?;
    validate_finite(weight, &format!("{label} weight"))?;

    let mut output = zeroed_f32(elements, label)?;
    for row in 0..rows {
        let start = row * cols;
        let end = start + cols;
        let input_row = &input[start..end];
        let mean_square = input_row
            .iter()
            .copied()
            .map(f64::from)
            .map(|value| value * value)
            .sum::<f64>()
            / cols as f64;
        let denominator = (mean_square + f64::from(epsilon)).sqrt();
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(format!(
                "SQ8 layer oracle {label} produced invalid RMS denominator {denominator} at row {row}"
            ));
        }
        let inverse_rms = 1.0_f64 / denominator;
        for col in 0..cols {
            let value = (f64::from(input_row[col]) * inverse_rms * f64::from(weight[col])) as f32;
            if !value.is_finite() {
                return Err(format!(
                    "SQ8 layer oracle {label} produced non-finite output {value} at [{row},{col}]"
                ));
            }
            output[start + col] = value;
        }
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn rope_split_half(
    input: &[f32],
    sequence_len: usize,
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    position_offset: usize,
    theta: f32,
    label: &str,
) -> Result<Vec<f32>, String> {
    if rotary_dim == 0 || rotary_dim > head_dim || !rotary_dim.is_multiple_of(2) {
        return Err(format!(
            "SQ8 layer oracle {label} rotary_dim must be positive, even, and no greater than head_dim: rotary_dim={rotary_dim} head_dim={head_dim}"
        ));
    }
    if !theta.is_finite() || theta <= 1.0 {
        return Err(format!(
            "SQ8 layer oracle {label} theta must be finite and greater than one, got {theta}"
        ));
    }
    let rows = sequence_len.checked_mul(heads).ok_or_else(|| {
        format!("SQ8 layer oracle {label} sequence-head row count overflows usize")
    })?;
    let elements = checked_elements(rows, head_dim, label)?;
    if input.len() != elements {
        return Err(format!(
            "SQ8 layer oracle {label} input length mismatch: expected={elements} actual={}",
            input.len()
        ));
    }
    let final_position = position_offset
        .checked_add(sequence_len - 1)
        .ok_or_else(|| format!("SQ8 layer oracle {label} final position overflows usize"))?;
    if final_position > MAX_EXACT_F32_INTEGER {
        return Err(format!(
            "SQ8 layer oracle {label} final position {final_position} exceeds the exact F32 integer limit {MAX_EXACT_F32_INTEGER}"
        ));
    }
    validate_finite(input, &format!("{label} input"))?;

    let mut output = zeroed_f32(elements, label)?;
    let half = rotary_dim / 2;
    for timestep in 0..sequence_len {
        let position = (position_offset + timestep) as f64;
        for head in 0..heads {
            let base = (timestep * heads + head) * head_dim;
            for pair_dim in 0..half {
                let exponent = (2 * pair_dim) as f64 / rotary_dim as f64;
                let angle = position / f64::from(theta).powf(exponent);
                let (sin, cos) = angle.sin_cos();
                let first = f64::from(input[base + pair_dim]);
                let second = f64::from(input[base + half + pair_dim]);
                let rotated_first = (first * cos - second * sin) as f32;
                let rotated_second = (second * cos + first * sin) as f32;
                if !rotated_first.is_finite() || !rotated_second.is_finite() {
                    return Err(format!(
                        "SQ8 layer oracle {label} produced non-finite RoPE output at timestep={timestep} head={head} pair_dim={pair_dim}"
                    ));
                }
                output[base + pair_dim] = rotated_first;
                output[base + half + pair_dim] = rotated_second;
            }
            output[base + rotary_dim..base + head_dim]
                .copy_from_slice(&input[base + rotary_dim..base + head_dim]);
        }
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn causal_gqa_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    sequence_len: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
) -> Result<Vec<f32>, String> {
    if q_heads == 0 || kv_heads == 0 || head_dim == 0 || value_dim == 0 {
        return Err(
            "SQ8 layer oracle attention dimensions must all be greater than zero".to_string(),
        );
    }
    if !q_heads.is_multiple_of(kv_heads) {
        return Err(format!(
            "SQ8 layer oracle q_heads must be a multiple of kv_heads: q_heads={q_heads} kv_heads={kv_heads}"
        ));
    }
    let q_rows = sequence_len
        .checked_mul(q_heads)
        .ok_or_else(|| "SQ8 layer oracle attention q row count overflows usize".to_string())?;
    let kv_rows = sequence_len
        .checked_mul(kv_heads)
        .ok_or_else(|| "SQ8 layer oracle attention kv row count overflows usize".to_string())?;
    let q_elements = checked_elements(q_rows, head_dim, "attention q")?;
    let k_elements = checked_elements(kv_rows, head_dim, "attention k")?;
    let v_elements = checked_elements(kv_rows, value_dim, "attention v")?;
    if q.len() != q_elements || k.len() != k_elements || v.len() != v_elements {
        return Err(format!(
            "SQ8 layer oracle attention input length mismatch: q expected={q_elements} actual={} k expected={k_elements} actual={} v expected={v_elements} actual={}",
            q.len(),
            k.len(),
            v.len()
        ));
    }
    validate_finite(q, "attention q")?;
    validate_finite(k, "attention k")?;
    validate_finite(v, "attention v")?;

    let output_elements = checked_elements(q_rows, value_dim, "attention output")?;
    let mut output = zeroed_f32(output_elements, "attention output")?;
    let q_per_kv = q_heads / kv_heads;
    let softmax_scale = 1.0_f64 / (head_dim as f64).sqrt();
    let mut scores = Vec::<f64>::new();
    scores.try_reserve_exact(sequence_len).map_err(|err| {
        format!("SQ8 layer oracle failed to reserve {sequence_len} attention scores: {err}")
    })?;

    for timestep in 0..sequence_len {
        for q_head in 0..q_heads {
            let kv_head = q_head / q_per_kv;
            let q_base = (timestep * q_heads + q_head) * head_dim;
            scores.clear();
            let mut max_score = f64::NEG_INFINITY;
            for source_timestep in 0..=timestep {
                let k_base = (source_timestep * kv_heads + kv_head) * head_dim;
                let mut dot = 0.0_f64;
                for dim in 0..head_dim {
                    dot += f64::from(q[q_base + dim]) * f64::from(k[k_base + dim]);
                }
                let score = dot * softmax_scale;
                if !score.is_finite() {
                    return Err(format!(
                        "SQ8 layer oracle attention produced non-finite score at timestep={timestep} q_head={q_head} source_timestep={source_timestep}"
                    ));
                }
                max_score = max_score.max(score);
                scores.push(score);
            }
            let mut denominator = 0.0_f64;
            for score in &mut scores {
                *score = (*score - max_score).exp();
                denominator += *score;
            }
            if !denominator.is_finite() || denominator <= 0.0 {
                return Err(format!(
                    "SQ8 layer oracle attention produced invalid softmax denominator {denominator} at timestep={timestep} q_head={q_head}"
                ));
            }
            let output_base = (timestep * q_heads + q_head) * value_dim;
            for value_dim_index in 0..value_dim {
                let mut weighted = 0.0_f64;
                for (source_timestep, weight) in scores.iter().copied().enumerate() {
                    let v_index =
                        (source_timestep * kv_heads + kv_head) * value_dim + value_dim_index;
                    weighted += weight * f64::from(v[v_index]);
                }
                let value = (weighted / denominator) as f32;
                if !value.is_finite() {
                    return Err(format!(
                        "SQ8 layer oracle attention produced non-finite output at timestep={timestep} q_head={q_head} value_dim={value_dim_index}"
                    ));
                }
                output[output_base + value_dim_index] = value;
            }
        }
    }
    Ok(output)
}

fn round_f32_to_bf16_rne(value: f32, label: &str, index: usize) -> Result<f32, String> {
    if !value.is_finite() {
        return Err(format!(
            "SQ8 layer oracle {label} cannot round non-finite value {value} at index {index} to BF16"
        ));
    }
    let bits = value.to_bits();
    let retained_lsb = (bits >> 16) & 1;
    let rounded_bits = bits.wrapping_add(0x7fff + retained_lsb) & 0xffff_0000;
    let rounded = f32::from_bits(rounded_bits);
    if !rounded.is_finite() {
        return Err(format!(
            "SQ8 layer oracle {label} BF16 rounding overflowed at index {index}: input={value}"
        ));
    }
    Ok(rounded)
}

fn round_slice_to_bf16_rne(values: &mut [f32], label: &str) -> Result<(), String> {
    for (index, value) in values.iter_mut().enumerate() {
        *value = round_f32_to_bf16_rne(*value, label, index)?;
    }
    Ok(())
}

fn add_in_place(lhs: &mut [f32], rhs: &[f32], label: &str) -> Result<(), String> {
    if lhs.len() != rhs.len() {
        return Err(format!(
            "SQ8 layer oracle {label} add length mismatch: lhs={} rhs={}",
            lhs.len(),
            rhs.len()
        ));
    }
    validate_finite(lhs, &format!("{label} lhs"))?;
    validate_finite(rhs, &format!("{label} rhs"))?;
    for index in 0..lhs.len() {
        let value = lhs[index] + rhs[index];
        if !value.is_finite() {
            return Err(format!(
                "SQ8 layer oracle {label} add produced non-finite output at index {index}"
            ));
        }
        lhs[index] = value;
    }
    Ok(())
}

fn silu_mul_in_place(gate: &mut [f32], up: &[f32]) -> Result<(), String> {
    if gate.len() != up.len() {
        return Err(format!(
            "SQ8 layer oracle SiLU multiply length mismatch: gate={} up={}",
            gate.len(),
            up.len()
        ));
    }
    validate_finite(gate, "SiLU gate")?;
    validate_finite(up, "SiLU up")?;
    for index in 0..gate.len() {
        let gate_f64 = f64::from(gate[index]);
        let sigmoid = if gate_f64 >= 0.0 {
            1.0 / (1.0 + (-gate_f64).exp())
        } else {
            let exp = gate_f64.exp();
            exp / (1.0 + exp)
        };
        let value = (gate_f64 * sigmoid * f64::from(up[index])) as f32;
        if !value.is_finite() {
            return Err(format!(
                "SQ8 layer oracle SiLU multiply produced non-finite output at index {index}"
            ));
        }
        gate[index] = value;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (*actual - *expected).abs() <= tolerance,
                "index={index} actual={actual} expected={expected} tolerance={tolerance}"
            );
        }
    }

    #[test]
    fn pure_rmsnorm_uses_independent_rows() {
        let input = [3.0, 4.0, 0.0, 0.0];
        let output = rmsnorm_rows(&input, 2, 2, &[1.0, 2.0], 1.0e-6, "test").unwrap();
        let first_inverse = 1.0 / (12.5_f32 + 1.0e-6).sqrt();
        assert_close(
            &output,
            &[3.0 * first_inverse, 8.0 * first_inverse, 0.0, 0.0],
            1.0e-6,
        );
    }

    #[test]
    fn pure_rope_uses_split_half_pairs() {
        let input = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0];
        let output = rope_split_half(&input, 2, 1, 4, 4, 0, 10_000.0, "test").unwrap();
        assert_eq!(&output[..4], &input[..4]);
        let (sin0, cos0) = 1.0_f32.sin_cos();
        let (sin1, cos1) = 0.01_f32.sin_cos();
        assert_close(
            &output[4..],
            &[
                cos0 - 3.0 * sin0,
                2.0 * cos1 - 4.0 * sin1,
                3.0 * cos0 + sin0,
                4.0 * cos1 + 2.0 * sin1,
            ],
            2.0e-6,
        );
    }

    #[test]
    fn pure_causal_gqa_maps_query_groups_and_masks_future_tokens() {
        let q = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let k = [1.0, 0.0, 0.0, 1.0];
        let v = [2.0, 6.0];
        let output = causal_gqa_attention(&q, &k, &v, 2, 2, 1, 2, 1).unwrap();
        assert_eq!(&output[..2], &[2.0, 2.0]);
        let scale = 1.0_f64 / 2.0_f64.sqrt();
        let first_weight = scale.exp() / (scale.exp() + 1.0);
        let expected_first_head = (first_weight * 2.0 + (1.0 - first_weight) * 6.0) as f32;
        let expected_second_head = ((1.0 - first_weight) * 2.0 + first_weight * 6.0) as f32;
        assert_close(
            &output[2..],
            &[expected_first_head, expected_second_head],
            1.0e-6,
        );
    }

    #[test]
    fn pure_bf16_rounding_is_ties_to_even_and_rejects_nonfinite() {
        let even_tie = f32::from_bits(0x3f80_8000);
        let odd_tie = f32::from_bits(0x3f81_8000);
        assert_eq!(
            round_f32_to_bf16_rne(even_tie, "test", 0)
                .unwrap()
                .to_bits(),
            0x3f80_0000
        );
        assert_eq!(
            round_f32_to_bf16_rne(odd_tie, "test", 1).unwrap().to_bits(),
            0x3f82_0000
        );
        assert!(round_f32_to_bf16_rne(f32::INFINITY, "test", 2).is_err());
    }

    #[test]
    fn projection_output_contract_defaults_to_bf16_and_can_preserve_f32() {
        assert_eq!(
            Sq8LayerOracleOptions::default().projection_output_contract,
            Sq8LayerProjectionOutputContract::Bf16RneThenF32
        );
        let source = f32::from_bits(0x3f80_4000);
        let mut bf16 = [source];
        apply_projection_output_contract(
            &mut bf16,
            Sq8LayerProjectionOutputContract::Bf16RneThenF32,
            "test",
        )
        .unwrap();
        assert_eq!(bf16[0].to_bits(), 0x3f80_0000);
        let mut f32_output = [source];
        apply_projection_output_contract(
            &mut f32_output,
            Sq8LayerProjectionOutputContract::F32,
            "test",
        )
        .unwrap();
        assert_eq!(f32_output[0].to_bits(), source.to_bits());
    }

    #[test]
    fn sequence_contract_rejects_unmeasured_length_and_inexact_f32_position() {
        validate_sequence_contract(QWEN3_14B_SQ8_LAYER_ORACLE_MAX_SEQUENCE_LEN, 0).unwrap();
        validate_sequence_contract(1, MAX_EXACT_F32_INTEGER).unwrap();
        assert!(validate_sequence_contract(0, 0).is_err());
        assert!(
            validate_sequence_contract(QWEN3_14B_SQ8_LAYER_ORACLE_MAX_SEQUENCE_LEN + 1, 0).is_err()
        );
        assert!(validate_sequence_contract(2, MAX_EXACT_F32_INTEGER).is_err());
    }

    #[test]
    fn pure_silu_mul_and_add_are_finite_and_shape_checked() {
        let mut gate = [-2.0, 0.0, 2.0];
        silu_mul_in_place(&mut gate, &[3.0, 4.0, 5.0]).unwrap();
        let expected = [
            -6.0 / (1.0 + 2.0_f32.exp()),
            0.0,
            10.0 / (1.0 + (-2.0_f32).exp()),
        ];
        assert_close(&gate, &expected, 1.0e-6);
        add_in_place(&mut gate, &[1.0, 2.0, 3.0], "test").unwrap();
        assert_close(&gate, &[expected[0] + 1.0, 2.0, expected[2] + 3.0], 1.0e-6);
        assert!(add_in_place(&mut gate, &[1.0], "bad shape").is_err());
        assert!(silu_mul_in_place(&mut gate, &[1.0, f32::NAN, 1.0]).is_err());
    }
}
