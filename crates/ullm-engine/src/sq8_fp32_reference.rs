// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! CPU-only strict artifact-FP32 reference for the canonical Qwen3-14B SQ8_0
//! product.
//!
//! This module intentionally does not use a runtime context, BLAS, HIP, or a
//! quantized activation path.  It validates the canonical artifact and bound
//! raw-package payloads, reconstructs each SQ8 weight in binary32 at use, and
//! performs a fixed increasing-K binary32 fused multiply-add reduction.  The
//! matrix payloads are streamed by output-row partition; only scale grids and
//! small non-SQ parameters stay resident.

use crate::loader::{
    PassthroughPayloadVerification, read_named_passthrough_f32, read_named_passthrough_f32_rows,
    verify_named_passthrough_payload,
};
use crate::package::{PassthroughPayloadBundle, list_passthrough_payload_bundles};
use crate::sq::fp8_e4m3fn_to_f32;
use crate::sq_canonical::{Sq8CanonicalArtifact, read_sq8_canonical_artifact};
use crate::sq_reference::{
    Sq8CorrectnessMetrics, compare_sq8_correctness, run_sq8_reference_projection, sq8_f32_le_sha256,
};
use crate::sq8_fnuz_prepack::{bf16_bits_to_f32, scan_sq8_canonical_artifact_for_fnuz_prepack};
use crate::sq8_layer_oracle::{
    QWEN3_14B_HEAD_DIM, QWEN3_14B_HIDDEN_SIZE, QWEN3_14B_INTERMEDIATE_SIZE, QWEN3_14B_KV_HEADS,
    QWEN3_14B_Q_HEADS, QWEN3_14B_RMS_NORM_EPSILON, QWEN3_14B_ROPE_THETA, QWEN3_14B_VALUE_DIM,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

pub const ARTIFACT_FP32_REFERENCE_ID: &str = "artifact_fp32_strict_v1";
pub const ARTIFACT_FP32_REFERENCE_SCHEMA_VERSION: &str = "ullm.sq8.artifact_fp32_reference.v1";
pub const QWEN3_14B_FP32_REFERENCE_LAYERS: usize = 40;
pub const QWEN3_14B_FP32_REFERENCE_VOCAB_SIZE: usize = 151_936;
pub const QWEN3_14B_FP32_REFERENCE_MAX_CONTEXT: usize = 4_096;
pub const QWEN3_14B_FP32_REFERENCE_DEFAULT_THREADS: usize = 64;

const Q_WIDTH: usize = QWEN3_14B_Q_HEADS * QWEN3_14B_HEAD_DIM;
const KV_WIDTH: usize = QWEN3_14B_KV_HEADS * QWEN3_14B_HEAD_DIM;
const FP8_BLOCK: usize = 128;
const WEIGHT_READ_BUFFER_BYTES: usize = 8 * 1024 * 1024;
const PACKAGE_VERIFY_CHUNK_BYTES: usize = 8 * 1024 * 1024;
const PARAMETER_READ_CHUNK_BYTES: usize = 1024 * 1024;
const EMBEDDING_TENSOR: &str = "model.embed_tokens.weight";
const FINAL_NORM_TENSOR: &str = "model.norm.weight";
const LM_HEAD_TENSOR: &str = "lm_head.weight";

#[derive(Debug, Clone, Serialize)]
pub struct ArtifactFp32PackageVerification {
    pub manifest_sha256: String,
    pub manifest_bytes: u64,
    pub passthrough_tensor_count: usize,
    pub payload_bytes: u64,
    pub verified_chunks: u64,
    pub finite_parameter_values: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ArtifactFp32ReferenceIdentity {
    pub reference_id: &'static str,
    pub artifact_content_sha256: String,
    pub artifact_weight_payload_bytes: u64,
    pub artifact_scale_payload_bytes: u64,
    pub artifact_quantized_tensor_count: usize,
    pub package: ArtifactFp32PackageVerification,
    pub layer_count: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub thread_count: usize,
    pub matrix_reduction: &'static str,
    pub residency: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct ArtifactFp32ForwardSummary {
    pub position: usize,
    pub input_token_id: u32,
    pub greedy_token_id: u32,
    pub logits_f32le_sha256: String,
    pub final_hidden_f32le_sha256: String,
    pub layer_hidden_f32le_sha256: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ArtifactFp32Forward {
    pub summary: ArtifactFp32ForwardSummary,
    pub logits: Vec<f32>,
    pub final_hidden: Vec<f32>,
    pub layer_hidden: Vec<Vec<f32>>,
}

/// A non-primary comparison of the strict binary32 projection with the
/// existing CPU streaming projection reference.  The latter intentionally
/// uses an F64 accumulator, so this records numerical proximity and shared
/// artifact-decoding semantics rather than requiring byte equality.
#[derive(Debug, Clone, Serialize)]
pub struct ArtifactFp32ProjectionCrossCheck {
    pub tensor: String,
    pub input_token_id: u32,
    pub input_f32le_sha256: String,
    pub strict_f32_output_f32le_sha256: String,
    pub existing_cpu_f64_output_f32le_sha256: String,
    pub existing_cpu_f64_vs_strict_f32: Sq8CorrectnessMetrics,
}

#[derive(Debug, Clone, Serialize)]
pub struct ArtifactFp32CaptureReceipt {
    pub schema_version: &'static str,
    pub identity: ArtifactFp32ReferenceIdentity,
    pub forward: ArtifactFp32ForwardSummary,
    pub files: BTreeMap<String, String>,
}

#[derive(Debug, Clone)]
struct LayerNorms {
    input: Vec<f32>,
    post_attention: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
}

#[derive(Debug)]
struct LayerKv {
    keys: Vec<f32>,
    values: Vec<f32>,
}

impl LayerKv {
    fn with_capacity(max_context: usize) -> Result<Self, String> {
        let elements = max_context
            .checked_mul(KV_WIDTH)
            .ok_or_else(|| "artifact-FP32 KV capacity overflows usize".to_string())?;
        let mut keys = Vec::new();
        let mut values = Vec::new();
        keys.try_reserve_exact(elements).map_err(|err| {
            format!("artifact-FP32 failed to reserve F32 K cache capacity: {err}")
        })?;
        values.try_reserve_exact(elements).map_err(|err| {
            format!("artifact-FP32 failed to reserve F32 V cache capacity: {err}")
        })?;
        Ok(Self { keys, values })
    }

    fn push(&mut self, key: &[f32], value: &[f32]) -> Result<(), String> {
        if key.len() != KV_WIDTH || value.len() != KV_WIDTH {
            return Err(format!(
                "artifact-FP32 KV write width mismatch: key={} value={} expected={KV_WIDTH}",
                key.len(),
                value.len()
            ));
        }
        self.keys
            .try_reserve(key.len())
            .map_err(|err| format!("artifact-FP32 failed to grow F32 K cache: {err}"))?;
        self.values
            .try_reserve(value.len())
            .map_err(|err| format!("artifact-FP32 failed to grow F32 V cache: {err}"))?;
        self.keys.extend_from_slice(key);
        self.values.extend_from_slice(value);
        Ok(())
    }
}

/// Integrity-verified raw inputs shared by the CPU and GPU F32 controls.
///
/// This intentionally shares only canonical artifact/package admission and
/// immutable raw payload locations.  It does not expose CPU-decoded FP8,
/// BF16, scale, norm, or activation values, so a GPU control must perform its
/// own numerical reconstruction from raw bytes.
#[derive(Debug)]
pub(crate) struct ArtifactFp32VerifiedInputs {
    pub(crate) artifact: Sq8CanonicalArtifact,
    pub(crate) package_dir: PathBuf,
    pub(crate) package_verification: ArtifactFp32PackageVerification,
    pub(crate) bundles: BTreeMap<String, PassthroughPayloadBundle>,
}

impl ArtifactFp32VerifiedInputs {
    pub(crate) fn open(
        artifact_dir: impl AsRef<Path>,
        package_dir: impl AsRef<Path>,
    ) -> Result<Self, String> {
        let artifact = read_sq8_canonical_artifact(artifact_dir)
            .map_err(|err| format!("artifact-FP32 canonical artifact validation failed: {err}"))?;
        validate_canonical_model_binding(&artifact)?;

        // This reuses the independent whole-artifact CPU integrity decoder.
        // It is admission-only: consumers below still decode numerical values
        // independently from raw canonical bytes.
        let integrity_scan =
            scan_sq8_canonical_artifact_for_fnuz_prepack(&artifact, PACKAGE_VERIFY_CHUNK_BYTES)
                .map_err(|err| format!("artifact-FP32 canonical integrity scan failed: {err}"))?;
        if integrity_scan.ocp_nan_0x7f_count != 0
            || integrity_scan.ocp_nan_0xff_count != 0
            || integrity_scan.invalid_bf16_scale_count != 0
        {
            return Err(format!(
                "artifact-FP32 canonical integrity scan rejected non-finite data: fp8_0x7f={} fp8_0xff={} invalid_scales={}",
                integrity_scan.ocp_nan_0x7f_count,
                integrity_scan.ocp_nan_0xff_count,
                integrity_scan.invalid_bf16_scale_count
            ));
        }

        let package_dir = package_dir.as_ref().to_path_buf();
        let (bundles, package_verification) = verify_bound_package(&package_dir, &artifact)?;
        Ok(Self {
            artifact,
            package_dir,
            package_verification,
            bundles,
        })
    }

    pub(crate) fn bundle(&self, tensor_name: &str) -> Result<&PassthroughPayloadBundle, String> {
        self.bundles
            .get(tensor_name)
            .ok_or_else(|| format!("artifact-FP32 bound package misses raw payload {tensor_name}"))
    }
}

/// A validated model whose large matrix weights are streamed from the immutable
/// canonical artifact/package for every forward pass.
#[derive(Debug)]
pub struct ArtifactFp32ReferenceModel {
    artifact: Sq8CanonicalArtifact,
    package_dir: PathBuf,
    package_verification: ArtifactFp32PackageVerification,
    scales: BTreeMap<String, Arc<[f32]>>,
    norms: Vec<LayerNorms>,
    final_norm: Vec<f32>,
    embedding: PassthroughPayloadBundle,
    lm_head: PassthroughPayloadBundle,
    decode_table: [f32; 256],
    thread_count: usize,
}

impl ArtifactFp32ReferenceModel {
    /// Opens only the bound canonical artifact and product package.  No source
    /// checkpoint path is read by this method.
    pub fn open(
        artifact_dir: impl AsRef<Path>,
        package_dir: impl AsRef<Path>,
        thread_count: usize,
    ) -> Result<Self, String> {
        if thread_count == 0 {
            return Err("artifact-FP32 thread_count must be greater than zero".into());
        }
        if thread_count > 128 {
            return Err("artifact-FP32 thread_count must not exceed 128".into());
        }

        let ArtifactFp32VerifiedInputs {
            artifact,
            package_dir,
            package_verification,
            bundles,
        } = ArtifactFp32VerifiedInputs::open(artifact_dir, package_dir)?;

        let mut decode_table = [0.0_f32; 256];
        for (index, value) in decode_table.iter_mut().enumerate() {
            *value = fp8_e4m3fn_to_f32(index as u8);
        }
        if decode_table[0x7f].is_finite() || decode_table[0xff].is_finite() {
            return Err(
                "artifact-FP32 E4M3FN decoder did not reject canonical NaN encodings".into(),
            );
        }
        if decode_table
            .iter()
            .enumerate()
            .any(|(index, value)| !value.is_finite() && !matches!(index, 0x7f | 0xff))
        {
            return Err("artifact-FP32 E4M3FN decoder rejected an unexpected payload byte".into());
        }

        let mut scales = BTreeMap::new();
        for pair in &artifact.manifest().quantized_tensors {
            let values = artifact
                .read_tensor_scales_f32(&pair.name, PARAMETER_READ_CHUNK_BYTES)
                .map_err(|err| {
                    format!("artifact-FP32 failed to decode scale {}: {err}", pair.name)
                })?;
            if values
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
            {
                return Err(format!(
                    "artifact-FP32 decoded non-finite/non-positive scale for {}",
                    pair.name
                ));
            }
            if scales
                .insert(pair.name.clone(), Arc::from(values))
                .is_some()
            {
                return Err(format!(
                    "artifact-FP32 duplicate canonical scale entry {}",
                    pair.name
                ));
            }
        }

        let mut norms = Vec::with_capacity(QWEN3_14B_FP32_REFERENCE_LAYERS);
        for layer_index in 0..QWEN3_14B_FP32_REFERENCE_LAYERS {
            let prefix = format!("model.layers.{layer_index}");
            norms.push(LayerNorms {
                input: load_small_bf16_parameter(
                    &package_dir,
                    &format!("{prefix}.input_layernorm.weight"),
                    QWEN3_14B_HIDDEN_SIZE,
                )?,
                post_attention: load_small_bf16_parameter(
                    &package_dir,
                    &format!("{prefix}.post_attention_layernorm.weight"),
                    QWEN3_14B_HIDDEN_SIZE,
                )?,
                q: load_small_bf16_parameter(
                    &package_dir,
                    &format!("{prefix}.self_attn.q_norm.weight"),
                    QWEN3_14B_HEAD_DIM,
                )?,
                k: load_small_bf16_parameter(
                    &package_dir,
                    &format!("{prefix}.self_attn.k_norm.weight"),
                    QWEN3_14B_HEAD_DIM,
                )?,
            });
        }

        let final_norm =
            load_small_bf16_parameter(&package_dir, FINAL_NORM_TENSOR, QWEN3_14B_HIDDEN_SIZE)?;
        let embedding = bundles
            .get(EMBEDDING_TENSOR)
            .cloned()
            .ok_or_else(|| "artifact-FP32 bound package misses embedding payload".to_string())?;
        let lm_head = bundles
            .get(LM_HEAD_TENSOR)
            .cloned()
            .ok_or_else(|| "artifact-FP32 bound package misses LM-head payload".to_string())?;

        Ok(Self {
            artifact,
            package_dir,
            package_verification,
            scales,
            norms,
            final_norm,
            embedding,
            lm_head,
            decode_table,
            thread_count,
        })
    }

    pub fn identity(&self) -> ArtifactFp32ReferenceIdentity {
        let checksums = self.artifact.checksum_report();
        ArtifactFp32ReferenceIdentity {
            reference_id: ARTIFACT_FP32_REFERENCE_ID,
            artifact_content_sha256: self.artifact.manifest().integrity.content_sha256.clone(),
            artifact_weight_payload_bytes: checksums.weight_payload_bytes,
            artifact_scale_payload_bytes: checksums.scale_payload_bytes,
            artifact_quantized_tensor_count: self.artifact.manifest().quantized_tensors.len(),
            package: self.package_verification.clone(),
            layer_count: QWEN3_14B_FP32_REFERENCE_LAYERS,
            hidden_size: QWEN3_14B_HIDDEN_SIZE,
            intermediate_size: QWEN3_14B_INTERMEDIATE_SIZE,
            q_heads: QWEN3_14B_Q_HEADS,
            kv_heads: QWEN3_14B_KV_HEADS,
            head_dim: QWEN3_14B_HEAD_DIM,
            vocab_size: QWEN3_14B_FP32_REFERENCE_VOCAB_SIZE,
            thread_count: self.thread_count,
            matrix_reduction: "increasing_k_f32_mul_add",
            residency: "layer_streaming_f32_reconstruction",
        }
    }

    pub fn session(&self, max_context: usize) -> Result<ArtifactFp32ReferenceSession<'_>, String> {
        if max_context == 0 || max_context > QWEN3_14B_FP32_REFERENCE_MAX_CONTEXT {
            return Err(format!(
                "artifact-FP32 max_context must be in 1..={}, got {max_context}",
                QWEN3_14B_FP32_REFERENCE_MAX_CONTEXT
            ));
        }
        let mut cache = Vec::with_capacity(QWEN3_14B_FP32_REFERENCE_LAYERS);
        for _ in 0..QWEN3_14B_FP32_REFERENCE_LAYERS {
            cache.push(LayerKv::with_capacity(max_context)?);
        }
        Ok(ArtifactFp32ReferenceSession {
            model: self,
            cache,
            position: 0,
            max_context,
            poisoned: None,
        })
    }

    /// Cross-checks the first Q projection from a real canonical-artifact
    /// embedding row against the pre-existing CPU streaming reference.  It is
    /// deliberately not used as the primary reference because that path has
    /// an F64 accumulator and covers only one projection.
    pub fn cross_check_layer0_q_projection(
        &self,
        token_id: u32,
    ) -> Result<ArtifactFp32ProjectionCrossCheck, String> {
        const TENSOR: &str = "model.layers.0.self_attn.q_proj.weight";

        let input = self.embedding_row(token_id)?;
        let input_f32le_sha256 = f32_le_sha256(&input, "cross-check input")?;
        let strict = self.artifact_matvec(TENSOR, &input)?;
        let existing = run_sq8_reference_projection(&self.artifact, TENSOR, &input)
            .map_err(|err| {
                format!(
                    "artifact-FP32 existing CPU F64 projection cross-check failed for {TENSOR}: {err}"
                )
            })?;
        if existing.input_f32_le_sha256 != input_f32le_sha256 {
            return Err(format!(
                "artifact-FP32 existing CPU F64 projection input hash mismatch: expected={input_f32le_sha256} actual={}",
                existing.input_f32_le_sha256
            ));
        }
        let metrics = compare_sq8_correctness(&existing.output, &strict).map_err(|err| {
            format!("artifact-FP32 existing CPU F64 projection metric computation failed: {err}")
        })?;
        Ok(ArtifactFp32ProjectionCrossCheck {
            tensor: TENSOR.to_string(),
            input_token_id: token_id,
            input_f32le_sha256,
            strict_f32_output_f32le_sha256: f32_le_sha256(&strict, "strict F32 cross-check")?,
            existing_cpu_f64_output_f32le_sha256: sq8_f32_le_sha256(&existing.output).map_err(
                |err| {
                    format!(
                        "artifact-FP32 failed to hash existing CPU F64 cross-check output: {err}"
                    )
                },
            )?,
            existing_cpu_f64_vs_strict_f32: metrics,
        })
    }

    fn embedding_row(&self, token_id: u32) -> Result<Vec<f32>, String> {
        let token_index = usize::try_from(token_id)
            .map_err(|_| format!("artifact-FP32 token ID {token_id} does not fit usize"))?;
        if token_index >= QWEN3_14B_FP32_REFERENCE_VOCAB_SIZE {
            return Err(format!(
                "artifact-FP32 token ID {token_id} is outside vocabulary 0..{}",
                QWEN3_14B_FP32_REFERENCE_VOCAB_SIZE
            ));
        }
        let rows = read_named_passthrough_f32_rows(
            &self.package_dir,
            &self.embedding.tensor_name,
            &[token_index],
        )
        .map_err(|err| format!("artifact-FP32 failed to read embedding row {token_id}: {err}"))?;
        if rows.dtype != "BF16"
            || rows.shape
                != [
                    QWEN3_14B_FP32_REFERENCE_VOCAB_SIZE as u64,
                    QWEN3_14B_HIDDEN_SIZE as u64,
                ]
            || rows.columns != QWEN3_14B_HIDDEN_SIZE
            || rows.values.len() != QWEN3_14B_HIDDEN_SIZE
        {
            return Err("artifact-FP32 embedding row contract changed after verification".into());
        }
        validate_finite(&rows.values, "embedding row")?;
        Ok(rows.values)
    }

    fn artifact_matvec(&self, tensor_name: &str, input: &[f32]) -> Result<Vec<f32>, String> {
        validate_finite(input, &format!("{tensor_name} input"))?;
        let pair = self
            .artifact
            .tensor_pair(tensor_name)
            .map_err(|err| format!("artifact-FP32 failed to select {tensor_name}: {err}"))?;
        let rows = usize::try_from(pair.shape[0])
            .map_err(|_| format!("artifact-FP32 {tensor_name} rows do not fit usize"))?;
        let cols = usize::try_from(pair.shape[1])
            .map_err(|_| format!("artifact-FP32 {tensor_name} columns do not fit usize"))?;
        if input.len() != cols {
            return Err(format!(
                "artifact-FP32 {tensor_name} input length mismatch: expected={cols} actual={}",
                input.len()
            ));
        }
        let scale_cols = usize::try_from(pair.scale.shape[1])
            .map_err(|_| format!("artifact-FP32 {tensor_name} scale columns do not fit usize"))?;
        let expected_scale_cols = cols.div_ceil(FP8_BLOCK);
        if scale_cols != expected_scale_cols || pair.scale.block_shape != [128, 128] {
            return Err(format!(
                "artifact-FP32 {tensor_name} scale layout no longer has [128,128] blocks"
            ));
        }
        let scales = self.scales.get(tensor_name).ok_or_else(|| {
            format!("artifact-FP32 missing decoded scale cache for {tensor_name}")
        })?;
        let expected_scales = rows
            .div_ceil(FP8_BLOCK)
            .checked_mul(scale_cols)
            .ok_or_else(|| format!("artifact-FP32 {tensor_name} scale length overflows usize"))?;
        if scales.len() != expected_scales {
            return Err(format!(
                "artifact-FP32 {tensor_name} cached scale length mismatch: expected={expected_scales} actual={}",
                scales.len()
            ));
        }
        let paths = self
            .artifact
            .tensor_payload_paths(tensor_name)
            .map_err(|err| {
                format!("artifact-FP32 failed to resolve {tensor_name} payload: {err}")
            })?;
        let expected_bytes = rows.checked_mul(cols).ok_or_else(|| {
            format!("artifact-FP32 {tensor_name} weight byte count overflows usize")
        })?;
        if usize::try_from(pair.weight.bytes).ok() != Some(expected_bytes) {
            return Err(format!(
                "artifact-FP32 {tensor_name} weight size changed after validation"
            ));
        }

        let worker_count = self.thread_count.min(rows).max(1);
        let rows_per_worker = rows.div_ceil(worker_count);
        let mut output = zeroed_f32(rows, &format!("{tensor_name} output"))?;
        let result = thread::scope(|scope| {
            let mut workers = Vec::with_capacity(worker_count);
            for (worker_index, output_partition) in output.chunks_mut(rows_per_worker).enumerate() {
                let start_row = worker_index * rows_per_worker;
                let path = paths.weight.clone();
                let scales = Arc::clone(scales);
                let decode_table = &self.decode_table;
                workers.push(scope.spawn(move || -> Result<(), String> {
                    stream_artifact_row_partition(
                        &path,
                        tensor_name,
                        start_row,
                        output_partition,
                        cols,
                        scale_cols,
                        &scales,
                        decode_table,
                        input,
                    )
                }));
            }
            join_workers(workers, tensor_name)
        });
        result?;
        validate_finite(&output, &format!("{tensor_name} output"))?;
        Ok(output)
    }

    fn lm_head_matvec(&self, hidden: &[f32]) -> Result<Vec<f32>, String> {
        validate_finite(hidden, "LM-head input")?;
        if hidden.len() != QWEN3_14B_HIDDEN_SIZE {
            return Err(format!(
                "artifact-FP32 LM-head input length mismatch: expected={QWEN3_14B_HIDDEN_SIZE} actual={}",
                hidden.len()
            ));
        }
        if self.lm_head.dtype.as_deref() != Some("BF16")
            || self.lm_head.shape
                != [
                    QWEN3_14B_FP32_REFERENCE_VOCAB_SIZE as u64,
                    QWEN3_14B_HIDDEN_SIZE as u64,
                ]
        {
            return Err("artifact-FP32 LM-head contract changed after verification".into());
        }
        let rows = QWEN3_14B_FP32_REFERENCE_VOCAB_SIZE;
        let worker_count = self.thread_count.min(rows).max(1);
        let rows_per_worker = rows.div_ceil(worker_count);
        let mut output = zeroed_f32(rows, "LM-head output")?;
        let path = self.lm_head.payload_file.absolute_path.clone();
        let result = thread::scope(|scope| {
            let mut workers = Vec::with_capacity(worker_count);
            for (worker_index, output_partition) in output.chunks_mut(rows_per_worker).enumerate() {
                let start_row = worker_index * rows_per_worker;
                let path = path.clone();
                workers.push(scope.spawn(move || -> Result<(), String> {
                    stream_bf16_row_partition(
                        &path,
                        LM_HEAD_TENSOR,
                        start_row,
                        output_partition,
                        QWEN3_14B_HIDDEN_SIZE,
                        hidden,
                    )
                }));
            }
            join_workers(workers, LM_HEAD_TENSOR)
        });
        result?;
        validate_finite(&output, "LM-head output")?;
        Ok(output)
    }
}

/// Stateful F32 causal-KV decode session.  A failed forward poisons the
/// session because a partially written cache cannot be safely reused.
#[derive(Debug)]
pub struct ArtifactFp32ReferenceSession<'a> {
    model: &'a ArtifactFp32ReferenceModel,
    cache: Vec<LayerKv>,
    position: usize,
    max_context: usize,
    poisoned: Option<String>,
}

impl<'a> ArtifactFp32ReferenceSession<'a> {
    pub fn position(&self) -> usize {
        self.position
    }

    pub fn forward_token(&mut self, token_id: u32) -> Result<ArtifactFp32Forward, String> {
        if let Some(reason) = &self.poisoned {
            return Err(format!("artifact-FP32 session is poisoned: {reason}"));
        }
        if self.position >= self.max_context {
            return Err(format!(
                "artifact-FP32 context limit {} reached before token {token_id}",
                self.max_context
            ));
        }
        match self.forward_token_inner(token_id) {
            Ok(output) => {
                self.position += 1;
                Ok(output)
            }
            Err(error) => {
                self.poisoned = Some(error.clone());
                Err(error)
            }
        }
    }

    fn forward_token_inner(&mut self, token_id: u32) -> Result<ArtifactFp32Forward, String> {
        let position = self.position;
        let mut hidden = self.model.embedding_row(token_id)?;
        let mut layer_hidden = Vec::with_capacity(QWEN3_14B_FP32_REFERENCE_LAYERS);

        for layer_index in 0..QWEN3_14B_FP32_REFERENCE_LAYERS {
            let norms = self.model.norms.get(layer_index).ok_or_else(|| {
                format!("artifact-FP32 missing resident norm layer {layer_index}")
            })?;
            let prefix = format!("model.layers.{layer_index}");
            let input_norm = rmsnorm_f32(
                &hidden,
                1,
                QWEN3_14B_HIDDEN_SIZE,
                &norms.input,
                "input RMSNorm",
            )?;
            let q = self
                .model
                .artifact_matvec(&format!("{prefix}.self_attn.q_proj.weight"), &input_norm)?;
            let k = self
                .model
                .artifact_matvec(&format!("{prefix}.self_attn.k_proj.weight"), &input_norm)?;
            let v = self
                .model
                .artifact_matvec(&format!("{prefix}.self_attn.v_proj.weight"), &input_norm)?;

            let q_norm = rmsnorm_f32(
                &q,
                QWEN3_14B_Q_HEADS,
                QWEN3_14B_HEAD_DIM,
                &norms.q,
                "Q head RMSNorm",
            )?;
            let k_norm = rmsnorm_f32(
                &k,
                QWEN3_14B_KV_HEADS,
                QWEN3_14B_HEAD_DIM,
                &norms.k,
                "K head RMSNorm",
            )?;
            let q_rope = rope_split_half_f32(&q_norm, QWEN3_14B_Q_HEADS, position, "Q RoPE")?;
            let k_rope = rope_split_half_f32(&k_norm, QWEN3_14B_KV_HEADS, position, "K RoPE")?;

            let cache = self
                .cache
                .get_mut(layer_index)
                .ok_or_else(|| format!("artifact-FP32 missing F32 KV cache layer {layer_index}"))?;
            cache.push(&k_rope, &v)?;
            let expected_cache_elements = (position + 1)
                .checked_mul(KV_WIDTH)
                .ok_or_else(|| "artifact-FP32 KV length overflows usize".to_string())?;
            if cache.keys.len() != expected_cache_elements
                || cache.values.len() != expected_cache_elements
            {
                return Err(format!(
                    "artifact-FP32 KV cache length mismatch at layer {layer_index}: keys={} values={} expected={expected_cache_elements}",
                    cache.keys.len(),
                    cache.values.len()
                ));
            }
            let attention = causal_gqa_decode_f32(&q_rope, &cache.keys, &cache.values, position)?;
            let o = self
                .model
                .artifact_matvec(&format!("{prefix}.self_attn.o_proj.weight"), &attention)?;
            add_f32(&mut hidden, &o, "attention residual")?;

            let post_attention_norm = rmsnorm_f32(
                &hidden,
                1,
                QWEN3_14B_HIDDEN_SIZE,
                &norms.post_attention,
                "post-attention RMSNorm",
            )?;
            let mut gate = self.model.artifact_matvec(
                &format!("{prefix}.mlp.gate_proj.weight"),
                &post_attention_norm,
            )?;
            let up = self.model.artifact_matvec(
                &format!("{prefix}.mlp.up_proj.weight"),
                &post_attention_norm,
            )?;
            silu_mul_f32(&mut gate, &up)?;
            let down = self
                .model
                .artifact_matvec(&format!("{prefix}.mlp.down_proj.weight"), &gate)?;
            add_f32(&mut hidden, &down, "MLP residual")?;
            layer_hidden.push(hidden.clone());
        }

        if layer_hidden.len() != QWEN3_14B_FP32_REFERENCE_LAYERS {
            return Err("artifact-FP32 did not retain all layer hidden states".into());
        }
        let final_hidden = rmsnorm_f32(
            &hidden,
            1,
            QWEN3_14B_HIDDEN_SIZE,
            &self.model.final_norm,
            "final RMSNorm",
        )?;
        let logits = self.model.lm_head_matvec(&final_hidden)?;
        let greedy_token_id = greedy_token(&logits)?;
        let layer_hidden_f32le_sha256 = layer_hidden
            .iter()
            .map(|values| f32_le_sha256(values, "layer hidden"))
            .collect::<Result<Vec<_>, _>>()?;
        let summary = ArtifactFp32ForwardSummary {
            position,
            input_token_id: token_id,
            greedy_token_id,
            logits_f32le_sha256: f32_le_sha256(&logits, "logits")?,
            final_hidden_f32le_sha256: f32_le_sha256(&final_hidden, "final hidden")?,
            layer_hidden_f32le_sha256,
        };
        Ok(ArtifactFp32Forward {
            summary,
            logits,
            final_hidden,
            layer_hidden,
        })
    }
}

/// Writes a no-clobber reusable capture for one full-model forward.
pub fn write_forward_capture(
    root: impl AsRef<Path>,
    identity: &ArtifactFp32ReferenceIdentity,
    forward: &ArtifactFp32Forward,
) -> Result<ArtifactFp32CaptureReceipt, String> {
    let root = root.as_ref();
    fs::create_dir_all(root).map_err(|err| {
        format!(
            "artifact-FP32 failed to create capture root {}: {err}",
            root.display()
        )
    })?;
    let layer_root = root.join("layers");
    fs::create_dir_all(&layer_root).map_err(|err| {
        format!(
            "artifact-FP32 failed to create layer capture root {}: {err}",
            layer_root.display()
        )
    })?;

    let mut files = BTreeMap::new();
    let logits_path = root.join("logits.f32le");
    files.insert(
        "logits.f32le".to_string(),
        write_f32le_create_new(&logits_path, &forward.logits, "logits")?,
    );
    let final_hidden_path = root.join("final-hidden.f32le");
    files.insert(
        "final-hidden.f32le".to_string(),
        write_f32le_create_new(&final_hidden_path, &forward.final_hidden, "final hidden")?,
    );
    for (layer_index, values) in forward.layer_hidden.iter().enumerate() {
        let relative = format!("layers/layer-{layer_index:02}-hidden.f32le");
        files.insert(
            relative.clone(),
            write_f32le_create_new(&root.join(&relative), values, "layer hidden")?,
        );
    }
    let receipt = ArtifactFp32CaptureReceipt {
        schema_version: ARTIFACT_FP32_REFERENCE_SCHEMA_VERSION,
        identity: identity.clone(),
        forward: forward.summary.clone(),
        files,
    };
    let metadata_path = root.join("metadata.json");
    write_json_create_new(&metadata_path, &receipt)?;
    Ok(receipt)
}

pub fn process_peak_rss_kib() -> Result<Option<u64>, String> {
    let status = match fs::read_to_string("/proc/self/status") {
        Ok(value) => value,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(format!(
                "artifact-FP32 failed to read /proc/self/status: {error}"
            ));
        }
    };
    for line in status.lines() {
        let Some(value) = line.strip_prefix("VmHWM:") else {
            continue;
        };
        let kib = value
            .split_whitespace()
            .next()
            .ok_or_else(|| "artifact-FP32 VmHWM has no numeric value".to_string())?
            .parse::<u64>()
            .map_err(|error| format!("artifact-FP32 VmHWM is invalid: {error}"))?;
        return Ok(Some(kib));
    }
    Ok(None)
}

pub fn duration_seconds(duration: Duration) -> f64 {
    duration.as_secs_f64()
}

fn validate_canonical_model_binding(artifact: &Sq8CanonicalArtifact) -> Result<(), String> {
    let manifest = artifact.manifest();
    if manifest.source.model_name != "Qwen3-14B-FP8"
        || manifest.coverage.scope != "full_model"
        || manifest.quantized_tensors.len() != QWEN3_14B_FP32_REFERENCE_LAYERS * 7
        || manifest.passthrough_tensors.len() != 163
    {
        return Err(format!(
            "artifact-FP32 requires the 40-layer Qwen3-14B full canonical product, got model={} scope={} pairs={} passthrough={}",
            manifest.source.model_name,
            manifest.coverage.scope,
            manifest.quantized_tensors.len(),
            manifest.passthrough_tensors.len()
        ));
    }
    for layer_index in 0..QWEN3_14B_FP32_REFERENCE_LAYERS {
        let prefix = format!("model.layers.{layer_index}");
        for (suffix, rows, cols) in [
            ("self_attn.q_proj.weight", Q_WIDTH, QWEN3_14B_HIDDEN_SIZE),
            ("self_attn.k_proj.weight", KV_WIDTH, QWEN3_14B_HIDDEN_SIZE),
            ("self_attn.v_proj.weight", KV_WIDTH, QWEN3_14B_HIDDEN_SIZE),
            (
                "self_attn.o_proj.weight",
                QWEN3_14B_HIDDEN_SIZE,
                QWEN3_14B_HIDDEN_SIZE,
            ),
            (
                "mlp.gate_proj.weight",
                QWEN3_14B_INTERMEDIATE_SIZE,
                QWEN3_14B_HIDDEN_SIZE,
            ),
            (
                "mlp.up_proj.weight",
                QWEN3_14B_INTERMEDIATE_SIZE,
                QWEN3_14B_HIDDEN_SIZE,
            ),
            (
                "mlp.down_proj.weight",
                QWEN3_14B_HIDDEN_SIZE,
                QWEN3_14B_INTERMEDIATE_SIZE,
            ),
        ] {
            let name = format!("{prefix}.{suffix}");
            let pair = artifact
                .tensor_pair(&name)
                .map_err(|err| format!("artifact-FP32 missing {name}: {err}"))?;
            if pair.name != name || pair.shape != [rows as u64, cols as u64] {
                return Err(format!(
                    "artifact-FP32 tensor binding mismatch for {name}: actual_name={} actual_shape={:?}",
                    pair.name, pair.shape
                ));
            }
        }
    }
    Ok(())
}

fn verify_bound_package(
    package_dir: &Path,
    artifact: &Sq8CanonicalArtifact,
) -> Result<
    (
        BTreeMap<String, PassthroughPayloadBundle>,
        ArtifactFp32PackageVerification,
    ),
    String,
> {
    let manifest_path = package_dir.join("manifest.json");
    let manifest_before = sha256_regular_file(&manifest_path)?;
    let manifest_bytes = fs::metadata(&manifest_path)
        .map_err(|err| format!("artifact-FP32 failed to stat package manifest: {err}"))?
        .len();
    let bundles = list_passthrough_payload_bundles(package_dir).map_err(|err| {
        format!("artifact-FP32 failed to enumerate package passthrough data: {err}")
    })?;
    if bundles.len() != artifact.manifest().passthrough_tensors.len() {
        return Err(format!(
            "artifact-FP32 package/artifact passthrough count mismatch: package={} artifact={}",
            bundles.len(),
            artifact.manifest().passthrough_tensors.len()
        ));
    }

    let expected = artifact
        .manifest()
        .passthrough_tensors
        .iter()
        .map(|tensor| (tensor.name.as_str(), tensor))
        .collect::<BTreeMap<_, _>>();
    let mut named = BTreeMap::new();
    let mut verified_chunks = 0_u64;
    let mut finite_parameter_values = 0_u64;
    let mut payload_bytes = 0_u64;
    for bundle in bundles {
        let expected_tensor = expected.get(bundle.tensor_name.as_str()).ok_or_else(|| {
            format!(
                "artifact-FP32 package contains undeclared passthrough tensor {}",
                bundle.tensor_name
            )
        })?;
        if bundle.dtype.as_deref() != Some(expected_tensor.dtype.as_str())
            || bundle.shape != expected_tensor.shape
            || bundle.elements != expected_tensor.elements
            || bundle.payload_encoding.as_deref() != Some("raw_safetensors_payload")
            || expected_tensor.dtype != "BF16"
        {
            return Err(format!(
                "artifact-FP32 package tensor contract mismatch for {}",
                bundle.tensor_name
            ));
        }
        let verification = verify_named_passthrough_payload(
            package_dir,
            &bundle.tensor_name,
            &expected_tensor.dtype,
            &expected_tensor.shape,
            PACKAGE_VERIFY_CHUNK_BYTES,
        )
        .map_err(|err| {
            format!(
                "artifact-FP32 package payload verification failed for {}: {err}",
                bundle.tensor_name
            )
        })?;
        validate_package_verification(&bundle, &verification)?;
        finite_parameter_values = finite_parameter_values
            .checked_add(scan_bf16_finite_payload(&bundle)?)
            .ok_or_else(|| "artifact-FP32 finite package element count overflows".to_string())?;
        verified_chunks = verified_chunks
            .checked_add(verification.verified_chunks)
            .ok_or_else(|| "artifact-FP32 package chunk count overflows".to_string())?;
        payload_bytes = payload_bytes
            .checked_add(verification.payload_bytes)
            .ok_or_else(|| "artifact-FP32 package byte count overflows".to_string())?;
        if named.insert(bundle.tensor_name.clone(), bundle).is_some() {
            return Err("artifact-FP32 package contains a duplicate passthrough name".into());
        }
    }
    if named.len() != expected.len()
        || named
            .keys()
            .any(|name| !expected.contains_key(name.as_str()))
    {
        return Err(
            "artifact-FP32 package passthrough name set differs from canonical artifact".into(),
        );
    }
    let manifest_after = sha256_regular_file(&manifest_path)?;
    if manifest_before != manifest_after {
        return Err("artifact-FP32 package manifest changed during verification".into());
    }
    Ok((
        named,
        ArtifactFp32PackageVerification {
            manifest_sha256: manifest_before,
            manifest_bytes,
            passthrough_tensor_count: expected.len(),
            payload_bytes,
            verified_chunks,
            finite_parameter_values,
        },
    ))
}

fn validate_package_verification(
    bundle: &PassthroughPayloadBundle,
    verification: &PassthroughPayloadVerification,
) -> Result<(), String> {
    if verification.tensor_name != bundle.tensor_name
        || verification.dtype != "BF16"
        || verification.shape != bundle.shape
        || verification.elements != bundle.elements
        || verification.payload_bytes != bundle.payload_bytes
        || bundle.payload_sha256.as_deref() != Some(verification.payload_sha256.as_str())
    {
        return Err(format!(
            "artifact-FP32 verification identity mismatch for package tensor {}",
            bundle.tensor_name
        ));
    }
    Ok(())
}

fn scan_bf16_finite_payload(bundle: &PassthroughPayloadBundle) -> Result<u64, String> {
    let expected_bytes = bundle.elements.checked_mul(2).ok_or_else(|| {
        format!(
            "artifact-FP32 {} BF16 byte count overflows",
            bundle.tensor_name
        )
    })?;
    if bundle.payload_bytes != expected_bytes || bundle.payload_file.bytes != expected_bytes {
        return Err(format!(
            "artifact-FP32 {} BF16 payload byte count changed after verification",
            bundle.tensor_name
        ));
    }
    let mut file = File::open(&bundle.payload_file.absolute_path).map_err(|err| {
        format!(
            "artifact-FP32 failed to open {} for finite scan: {err}",
            bundle.payload_file.absolute_path.display()
        )
    })?;
    let mut buffer = vec![0_u8; PACKAGE_VERIFY_CHUNK_BYTES];
    let mut remaining = expected_bytes;
    let mut elements = 0_u64;
    while remaining > 0 {
        let read_len = usize::try_from(remaining.min(buffer.len() as u64))
            .map_err(|_| "artifact-FP32 package finite scan length does not fit usize")?;
        file.read_exact(&mut buffer[..read_len]).map_err(|err| {
            format!(
                "artifact-FP32 failed to scan {} for finite values: {err}",
                bundle.tensor_name
            )
        })?;
        if !read_len.is_multiple_of(2) {
            return Err(format!(
                "artifact-FP32 BF16 finite scan has odd chunk for {}",
                bundle.tensor_name
            ));
        }
        for (index, raw) in buffer[..read_len].chunks_exact(2).enumerate() {
            let value = bf16_bits_to_f32(u16::from_le_bytes([raw[0], raw[1]]));
            if !value.is_finite() {
                return Err(format!(
                    "artifact-FP32 package tensor {} has non-finite BF16 value at byte {}",
                    bundle.tensor_name,
                    expected_bytes - remaining + (index * 2) as u64
                ));
            }
        }
        elements = elements
            .checked_add((read_len / 2) as u64)
            .ok_or_else(|| "artifact-FP32 finite scan element count overflows".to_string())?;
        remaining -= read_len as u64;
    }
    let mut trailing = [0_u8; 1];
    if file.read(&mut trailing).map_err(|err| {
        format!(
            "artifact-FP32 failed to verify EOF after BF16 scan for {}: {err}",
            bundle.tensor_name
        )
    })? != 0
    {
        return Err(format!(
            "artifact-FP32 package tensor {} gained trailing bytes",
            bundle.tensor_name
        ));
    }
    Ok(elements)
}

fn load_small_bf16_parameter(
    package_dir: &Path,
    tensor_name: &str,
    expected_elements: usize,
) -> Result<Vec<f32>, String> {
    let parameter =
        read_named_passthrough_f32(package_dir, tensor_name, PARAMETER_READ_CHUNK_BYTES)
            .map_err(|err| format!("artifact-FP32 failed to read {tensor_name}: {err}"))?;
    if parameter.dtype != "BF16"
        || parameter.shape != [expected_elements as u64]
        || parameter.values.len() != expected_elements
    {
        return Err(format!(
            "artifact-FP32 small parameter contract mismatch for {tensor_name}"
        ));
    }
    validate_finite(&parameter.values, tensor_name)?;
    Ok(parameter.values)
}

fn stream_artifact_row_partition(
    path: &Path,
    tensor_name: &str,
    start_row: usize,
    output: &mut [f32],
    cols: usize,
    scale_cols: usize,
    scales: &[f32],
    decode_table: &[f32; 256],
    input: &[f32],
) -> Result<(), String> {
    let start_offset = start_row
        .checked_mul(cols)
        .and_then(|value| u64::try_from(value).ok())
        .ok_or_else(|| format!("artifact-FP32 {tensor_name} row offset overflows"))?;
    let file = File::open(path)
        .map_err(|err| format!("artifact-FP32 failed to open {tensor_name} weight: {err}"))?;
    let expected_end =
        start_offset
            .checked_add(
                u64::try_from(output.len().checked_mul(cols).ok_or_else(|| {
                    format!("artifact-FP32 {tensor_name} partition size overflows")
                })?)
                .map_err(|_| format!("artifact-FP32 {tensor_name} partition does not fit u64"))?,
            )
            .ok_or_else(|| format!("artifact-FP32 {tensor_name} partition end overflows"))?;
    let file_bytes = file
        .metadata()
        .map_err(|err| format!("artifact-FP32 failed to stat {tensor_name} weight: {err}"))?
        .len();
    if expected_end > file_bytes {
        return Err(format!(
            "artifact-FP32 {tensor_name} partition exceeds verified payload: end={expected_end} bytes={file_bytes}"
        ));
    }
    let mut reader = BufReader::with_capacity(WEIGHT_READ_BUFFER_BYTES, file);
    reader
        .seek(SeekFrom::Start(start_offset))
        .map_err(|err| format!("artifact-FP32 failed to seek {tensor_name}: {err}"))?;
    let mut row_bytes = vec![0_u8; cols];
    for (local_row, output_value) in output.iter_mut().enumerate() {
        let row = start_row + local_row;
        reader.read_exact(&mut row_bytes).map_err(|err| {
            format!("artifact-FP32 failed to read {tensor_name} row {row}: {err}")
        })?;
        let scale_row = row / FP8_BLOCK;
        let scale_base = scale_row
            .checked_mul(scale_cols)
            .ok_or_else(|| format!("artifact-FP32 {tensor_name} scale row offset overflows"))?;
        let mut accumulator = 0.0_f32;
        for block_col in 0..scale_cols {
            let scale = *scales
                .get(scale_base + block_col)
                .ok_or_else(|| format!("artifact-FP32 {tensor_name} scale index escaped cache"))?;
            let start_col = block_col * FP8_BLOCK;
            let end_col = (start_col + FP8_BLOCK).min(cols);
            for col in start_col..end_col {
                let decoded = decode_table[usize::from(row_bytes[col])];
                if !decoded.is_finite() {
                    return Err(format!(
                        "artifact-FP32 {tensor_name} encountered non-finite E4M3 payload at [{row},{col}]"
                    ));
                }
                let weight = decoded * scale;
                if !weight.is_finite() {
                    return Err(format!(
                        "artifact-FP32 {tensor_name} reconstructed non-finite F32 weight at [{row},{col}]"
                    ));
                }
                accumulator = f32_mac(accumulator, weight, input[col]);
            }
        }
        if !accumulator.is_finite() {
            return Err(format!(
                "artifact-FP32 {tensor_name} produced non-finite output at row {row}"
            ));
        }
        *output_value = accumulator;
    }
    Ok(())
}

fn stream_bf16_row_partition(
    path: &Path,
    tensor_name: &str,
    start_row: usize,
    output: &mut [f32],
    cols: usize,
    input: &[f32],
) -> Result<(), String> {
    let row_bytes_len = cols
        .checked_mul(2)
        .ok_or_else(|| format!("artifact-FP32 {tensor_name} row bytes overflow"))?;
    let start_offset = start_row
        .checked_mul(row_bytes_len)
        .and_then(|value| u64::try_from(value).ok())
        .ok_or_else(|| format!("artifact-FP32 {tensor_name} row offset overflows"))?;
    let file = File::open(path)
        .map_err(|err| format!("artifact-FP32 failed to open {tensor_name}: {err}"))?;
    let expected_end =
        start_offset
            .checked_add(
                u64::try_from(output.len().checked_mul(row_bytes_len).ok_or_else(|| {
                    format!("artifact-FP32 {tensor_name} partition size overflows")
                })?)
                .map_err(|_| format!("artifact-FP32 {tensor_name} partition does not fit u64"))?,
            )
            .ok_or_else(|| format!("artifact-FP32 {tensor_name} partition end overflows"))?;
    let file_bytes = file
        .metadata()
        .map_err(|err| format!("artifact-FP32 failed to stat {tensor_name}: {err}"))?
        .len();
    if expected_end > file_bytes {
        return Err(format!(
            "artifact-FP32 {tensor_name} partition exceeds verified payload: end={expected_end} bytes={file_bytes}"
        ));
    }
    let mut reader = BufReader::with_capacity(WEIGHT_READ_BUFFER_BYTES, file);
    reader
        .seek(SeekFrom::Start(start_offset))
        .map_err(|err| format!("artifact-FP32 failed to seek {tensor_name}: {err}"))?;
    let mut row_bytes = vec![0_u8; row_bytes_len];
    for (local_row, output_value) in output.iter_mut().enumerate() {
        let row = start_row + local_row;
        reader.read_exact(&mut row_bytes).map_err(|err| {
            format!("artifact-FP32 failed to read {tensor_name} row {row}: {err}")
        })?;
        let mut accumulator = 0.0_f32;
        for col in 0..cols {
            let byte_index = col * 2;
            let weight = bf16_bits_to_f32(u16::from_le_bytes([
                row_bytes[byte_index],
                row_bytes[byte_index + 1],
            ]));
            if !weight.is_finite() {
                return Err(format!(
                    "artifact-FP32 {tensor_name} encountered non-finite BF16 value at [{row},{col}]"
                ));
            }
            accumulator = f32_mac(accumulator, weight, input[col]);
        }
        if !accumulator.is_finite() {
            return Err(format!(
                "artifact-FP32 {tensor_name} produced non-finite output at row {row}"
            ));
        }
        *output_value = accumulator;
    }
    Ok(())
}

fn join_workers<T>(
    workers: Vec<thread::ScopedJoinHandle<'_, Result<T, String>>>,
    label: &str,
) -> Result<(), String> {
    for worker in workers {
        match worker.join() {
            Ok(Ok(_)) => {}
            Ok(Err(error)) => return Err(error),
            Err(_) => return Err(format!("artifact-FP32 {label} worker thread panicked")),
        }
    }
    Ok(())
}

#[inline(always)]
fn f32_mac(accumulator: f32, lhs: f32, rhs: f32) -> f32 {
    lhs.mul_add(rhs, accumulator)
}

fn rmsnorm_f32(
    input: &[f32],
    rows: usize,
    cols: usize,
    weight: &[f32],
    label: &str,
) -> Result<Vec<f32>, String> {
    let elements = rows
        .checked_mul(cols)
        .ok_or_else(|| format!("artifact-FP32 {label} shape overflows"))?;
    if input.len() != elements || weight.len() != cols {
        return Err(format!(
            "artifact-FP32 {label} shape mismatch: input={} expected={elements} weight={} expected={cols}",
            input.len(),
            weight.len()
        ));
    }
    validate_finite(input, &format!("{label} input"))?;
    validate_finite(weight, &format!("{label} weight"))?;
    let mut output = zeroed_f32(elements, label)?;
    for row in 0..rows {
        let start = row * cols;
        let input_row = &input[start..start + cols];
        let mut sum_squares = 0.0_f32;
        for value in input_row {
            sum_squares = f32_mac(sum_squares, *value, *value);
        }
        let mean_square = sum_squares / cols as f32;
        let inverse_rms = (mean_square + QWEN3_14B_RMS_NORM_EPSILON).sqrt().recip();
        if !inverse_rms.is_finite() || inverse_rms <= 0.0 {
            return Err(format!(
                "artifact-FP32 {label} produced invalid inverse RMS at row {row}"
            ));
        }
        for col in 0..cols {
            let value = (input_row[col] * inverse_rms) * weight[col];
            if !value.is_finite() {
                return Err(format!(
                    "artifact-FP32 {label} produced non-finite value at [{row},{col}]"
                ));
            }
            output[start + col] = value;
        }
    }
    Ok(output)
}

fn rope_split_half_f32(
    input: &[f32],
    heads: usize,
    position: usize,
    label: &str,
) -> Result<Vec<f32>, String> {
    if position >= (1 << 24) {
        return Err(format!(
            "artifact-FP32 {label} position {position} exceeds exact F32 integer range"
        ));
    }
    let expected = heads
        .checked_mul(QWEN3_14B_HEAD_DIM)
        .ok_or_else(|| format!("artifact-FP32 {label} shape overflows"))?;
    if input.len() != expected {
        return Err(format!(
            "artifact-FP32 {label} input mismatch: expected={expected} actual={}",
            input.len()
        ));
    }
    validate_finite(input, &format!("{label} input"))?;
    let mut output = zeroed_f32(expected, label)?;
    let half = QWEN3_14B_HEAD_DIM / 2;
    let position = position as f32;
    for head in 0..heads {
        let base = head * QWEN3_14B_HEAD_DIM;
        for pair_dim in 0..half {
            let exponent = (2 * pair_dim) as f32 / QWEN3_14B_HEAD_DIM as f32;
            let angle = position / QWEN3_14B_ROPE_THETA.powf(exponent);
            let (sin, cos) = angle.sin_cos();
            let first = input[base + pair_dim];
            let second = input[base + half + pair_dim];
            let rotated_first = first * cos - second * sin;
            let rotated_second = second * cos + first * sin;
            if !rotated_first.is_finite() || !rotated_second.is_finite() {
                return Err(format!(
                    "artifact-FP32 {label} produced non-finite value at head={head} pair={pair_dim}"
                ));
            }
            output[base + pair_dim] = rotated_first;
            output[base + half + pair_dim] = rotated_second;
        }
    }
    Ok(output)
}

fn causal_gqa_decode_f32(
    q: &[f32],
    keys: &[f32],
    values: &[f32],
    position: usize,
) -> Result<Vec<f32>, String> {
    if q.len() != Q_WIDTH {
        return Err(format!(
            "artifact-FP32 attention Q width mismatch: expected={Q_WIDTH} actual={}",
            q.len()
        ));
    }
    let tokens = position
        .checked_add(1)
        .ok_or_else(|| "artifact-FP32 attention position overflows".to_string())?;
    let kv_elements = tokens
        .checked_mul(KV_WIDTH)
        .ok_or_else(|| "artifact-FP32 attention KV shape overflows".to_string())?;
    if keys.len() != kv_elements || values.len() != kv_elements {
        return Err(format!(
            "artifact-FP32 attention KV length mismatch: keys={} values={} expected={kv_elements}",
            keys.len(),
            values.len()
        ));
    }
    validate_finite(q, "attention Q")?;
    validate_finite(keys, "attention K")?;
    validate_finite(values, "attention V")?;
    let q_per_kv = QWEN3_14B_Q_HEADS / QWEN3_14B_KV_HEADS;
    let softmax_scale = (QWEN3_14B_HEAD_DIM as f32).sqrt().recip();
    let mut output = zeroed_f32(Q_WIDTH, "attention output")?;
    let mut scores = vec![0.0_f32; tokens];
    for q_head in 0..QWEN3_14B_Q_HEADS {
        let kv_head = q_head / q_per_kv;
        let q_base = q_head * QWEN3_14B_HEAD_DIM;
        let mut max_score = f32::NEG_INFINITY;
        for source_position in 0..tokens {
            let key_base = (source_position * QWEN3_14B_KV_HEADS + kv_head) * QWEN3_14B_HEAD_DIM;
            let mut dot = 0.0_f32;
            for dim in 0..QWEN3_14B_HEAD_DIM {
                dot = f32_mac(dot, q[q_base + dim], keys[key_base + dim]);
            }
            let score = dot * softmax_scale;
            if !score.is_finite() {
                return Err(format!(
                    "artifact-FP32 attention produced non-finite score at q_head={q_head} source={source_position}"
                ));
            }
            scores[source_position] = score;
            max_score = max_score.max(score);
        }
        let mut denominator = 0.0_f32;
        for score in &mut scores {
            *score = (*score - max_score).exp();
            denominator += *score;
        }
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(format!(
                "artifact-FP32 attention produced invalid softmax denominator at q_head={q_head}"
            ));
        }
        let output_base = q_head * QWEN3_14B_VALUE_DIM;
        for dim in 0..QWEN3_14B_VALUE_DIM {
            let mut weighted = 0.0_f32;
            for source_position in 0..tokens {
                let value_index =
                    (source_position * QWEN3_14B_KV_HEADS + kv_head) * QWEN3_14B_VALUE_DIM + dim;
                weighted = f32_mac(weighted, scores[source_position], values[value_index]);
            }
            let value = weighted / denominator;
            if !value.is_finite() {
                return Err(format!(
                    "artifact-FP32 attention produced non-finite output at q_head={q_head} dim={dim}"
                ));
            }
            output[output_base + dim] = value;
        }
    }
    Ok(output)
}

fn add_f32(lhs: &mut [f32], rhs: &[f32], label: &str) -> Result<(), String> {
    if lhs.len() != rhs.len() {
        return Err(format!(
            "artifact-FP32 {label} length mismatch: lhs={} rhs={}",
            lhs.len(),
            rhs.len()
        ));
    }
    for (index, (left, right)) in lhs.iter_mut().zip(rhs).enumerate() {
        let value = *left + *right;
        if !value.is_finite() {
            return Err(format!(
                "artifact-FP32 {label} produced non-finite output at index {index}"
            ));
        }
        *left = value;
    }
    Ok(())
}

fn silu_mul_f32(gate: &mut [f32], up: &[f32]) -> Result<(), String> {
    if gate.len() != up.len() {
        return Err(format!(
            "artifact-FP32 SiLU multiply length mismatch: gate={} up={}",
            gate.len(),
            up.len()
        ));
    }
    for (index, (gate_value, up_value)) in gate.iter_mut().zip(up).enumerate() {
        let x = *gate_value;
        let sigmoid = if x >= 0.0 {
            1.0_f32 / (1.0_f32 + (-x).exp())
        } else {
            let exp = x.exp();
            exp / (1.0_f32 + exp)
        };
        let value = (x * sigmoid) * *up_value;
        if !value.is_finite() {
            return Err(format!(
                "artifact-FP32 SiLU multiply produced non-finite output at index {index}"
            ));
        }
        *gate_value = value;
    }
    Ok(())
}

pub(crate) fn greedy_token(logits: &[f32]) -> Result<u32, String> {
    if logits.len() != QWEN3_14B_FP32_REFERENCE_VOCAB_SIZE {
        return Err(format!(
            "artifact-FP32 logits length mismatch: expected={} actual={}",
            QWEN3_14B_FP32_REFERENCE_VOCAB_SIZE,
            logits.len()
        ));
    }
    let mut best_index = 0_usize;
    let mut best_value = f32::NEG_INFINITY;
    for (index, value) in logits.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(format!(
                "artifact-FP32 logits contain non-finite value at {index}"
            ));
        }
        if value > best_value || (value == best_value && index < best_index) {
            best_value = value;
            best_index = index;
        }
    }
    u32::try_from(best_index)
        .map_err(|_| "artifact-FP32 greedy token index does not fit u32".to_string())
}

fn zeroed_f32(elements: usize, label: &str) -> Result<Vec<f32>, String> {
    let mut values = Vec::new();
    values.try_reserve_exact(elements).map_err(|err| {
        format!("artifact-FP32 failed to reserve {elements} F32 values for {label}: {err}")
    })?;
    values.resize(elements, 0.0);
    Ok(values)
}

pub(crate) fn validate_finite(values: &[f32], label: &str) -> Result<(), String> {
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!(
            "artifact-FP32 {label} contains non-finite F32 value {value} at index {index}"
        ));
    }
    Ok(())
}

pub(crate) fn f32_le_sha256(values: &[f32], label: &str) -> Result<String, String> {
    validate_finite(values, label)?;
    let mut digest = Sha256::new();
    for value in values {
        digest.update(value.to_le_bytes());
    }
    Ok(format!("{:x}", digest.finalize()))
}

pub(crate) fn write_f32le_create_new(
    path: &Path,
    values: &[f32],
    label: &str,
) -> Result<String, String> {
    validate_finite(values, label)?;
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|err| format!("artifact-FP32 failed to create {}: {err}", path.display()))?;
    let mut writer = BufWriter::new(file);
    let mut digest = Sha256::new();
    for value in values {
        let bytes = value.to_le_bytes();
        writer
            .write_all(&bytes)
            .map_err(|err| format!("artifact-FP32 failed to write {}: {err}", path.display()))?;
        digest.update(bytes);
    }
    writer
        .flush()
        .map_err(|err| format!("artifact-FP32 failed to flush {}: {err}", path.display()))?;
    Ok(format!("{:x}", digest.finalize()))
}

pub(crate) fn write_json_create_new(path: &Path, value: &impl Serialize) -> Result<(), String> {
    let serialized = serde_json::to_vec_pretty(value)
        .map_err(|err| format!("artifact-FP32 failed to serialize capture metadata: {err}"))?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|err| format!("artifact-FP32 failed to create {}: {err}", path.display()))?;
    file.write_all(&serialized)
        .map_err(|err| format!("artifact-FP32 failed to write {}: {err}", path.display()))?;
    file.write_all(b"\n")
        .map_err(|err| format!("artifact-FP32 failed to finish {}: {err}", path.display()))?;
    Ok(())
}

fn sha256_regular_file(path: &Path) -> Result<String, String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|err| format!("artifact-FP32 failed to stat {}: {err}", path.display()))?;
    if !metadata.file_type().is_file() {
        return Err(format!(
            "artifact-FP32 path is not a regular file: {}",
            path.display()
        ));
    }
    let mut file = File::open(path)
        .map_err(|err| format!("artifact-FP32 failed to open {}: {err}", path.display()))?;
    let mut buffer = vec![0_u8; PACKAGE_VERIFY_CHUNK_BYTES];
    let mut digest = Sha256::new();
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|err| format!("artifact-FP32 failed to read {}: {err}", path.display()))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temporary_file(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "ullm-sq8-fp32-reference-{label}-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    #[test]
    fn f32_mac_uses_binary32_fused_operation() {
        let accumulated = f32_mac(1.0, 2.0, 3.0);
        assert_eq!(accumulated.to_bits(), 7.0_f32.to_bits());
    }

    #[test]
    fn rope_position_zero_is_identity() {
        let values = (0..QWEN3_14B_HEAD_DIM)
            .map(|index| (index as f32 - 64.0) / 32.0)
            .collect::<Vec<_>>();
        let rotated = rope_split_half_f32(&values, 1, 0, "test").unwrap();
        assert_eq!(rotated, values);
    }

    #[test]
    fn one_token_causal_attention_returns_value_for_each_gqa_head() {
        let q = vec![1.0_f32; Q_WIDTH];
        let keys = vec![0.5_f32; KV_WIDTH];
        let values = (0..KV_WIDTH)
            .map(|index| index as f32 / KV_WIDTH as f32)
            .collect::<Vec<_>>();
        let output = causal_gqa_decode_f32(&q, &keys, &values, 0).unwrap();
        for q_head in 0..QWEN3_14B_Q_HEADS {
            let kv_head = q_head / (QWEN3_14B_Q_HEADS / QWEN3_14B_KV_HEADS);
            let output_base = q_head * QWEN3_14B_VALUE_DIM;
            let value_base = kv_head * QWEN3_14B_VALUE_DIM;
            assert_eq!(
                &output[output_base..output_base + QWEN3_14B_VALUE_DIM],
                &values[value_base..value_base + QWEN3_14B_VALUE_DIM]
            );
        }
    }

    #[test]
    fn greedy_token_uses_lower_id_for_equal_logits() {
        let mut logits = vec![-1.0_f32; QWEN3_14B_FP32_REFERENCE_VOCAB_SIZE];
        logits[17] = 3.0;
        logits[29] = 3.0;
        assert_eq!(greedy_token(&logits).unwrap(), 17);
    }

    #[test]
    fn artifact_partition_applies_128_by_128_scales_in_f32() {
        let path = temporary_file("block-scales");
        let rows = 129;
        let cols = 129;
        fs::write(&path, vec![0x38_u8; rows * cols]).unwrap(); // E4M3FN 1.0
        let mut output = vec![0.0_f32; rows];
        let mut decode_table = [0.0_f32; 256];
        for (index, value) in decode_table.iter_mut().enumerate() {
            *value = fp8_e4m3fn_to_f32(index as u8);
        }
        let input = vec![1.0_f32; cols];
        stream_artifact_row_partition(
            &path,
            "test",
            0,
            &mut output,
            cols,
            2,
            &[1.0, 2.0, 3.0, 4.0],
            &decode_table,
            &input,
        )
        .unwrap();
        assert_eq!(output[0].to_bits(), 130.0_f32.to_bits());
        assert_eq!(output[128].to_bits(), 388.0_f32.to_bits());
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn bf16_partition_decodes_before_f32_multiply_accumulate() {
        let path = temporary_file("bf16");
        let mut payload = Vec::new();
        for value in [1.0_f32, 2.0, 3.0, 4.0] {
            payload.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
        }
        fs::write(&path, payload).unwrap();
        let mut output = vec![0.0_f32; 2];
        stream_bf16_row_partition(&path, "test", 0, &mut output, 2, &[5.0, 6.0]).unwrap();
        assert_eq!(output, vec![17.0, 39.0]);
        fs::remove_file(path).unwrap();
    }
}
