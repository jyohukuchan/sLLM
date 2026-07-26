// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Independent gfx1201 F32 reference for the canonical Qwen3-14B `SQ8_0`
//! artifact.
//!
//! This module shares only immutable artifact/package admission with the CPU
//! strict reference.  The HIP session receives raw OCP E4M3FN, raw BF16 scale,
//! and raw BF16 parameter bytes; its C++ control performs its own decode and
//! F32 computation without entering the optimized SQ8 runtime dispatch.

use crate::package::PassthroughPayloadBundle;
use crate::sq8_fp32_reference::{
    ArtifactFp32Forward, ArtifactFp32ForwardSummary, ArtifactFp32PackageVerification,
    ArtifactFp32VerifiedInputs, QWEN3_14B_FP32_REFERENCE_LAYERS,
    QWEN3_14B_FP32_REFERENCE_MAX_CONTEXT, QWEN3_14B_FP32_REFERENCE_VOCAB_SIZE, f32_le_sha256,
    greedy_token, write_f32le_create_new, write_json_create_new,
};
use crate::sq8_layer_oracle::{
    QWEN3_14B_HEAD_DIM, QWEN3_14B_HIDDEN_SIZE, QWEN3_14B_INTERMEDIATE_SIZE, QWEN3_14B_KV_HEADS,
    QWEN3_14B_Q_HEADS,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::Read;
use std::path::Path;
use ullm_runtime_sys::{Sq8Fp32GpuReferenceGfx1201DeviceInfo, Sq8Fp32GpuReferenceGfx1201Session};

pub const ARTIFACT_GPU_FP32_REFERENCE_ID: &str = "artifact_gpu_fp32_hipblas_v1";
pub const ARTIFACT_GPU_FP32_REFERENCE_SCHEMA_VERSION: &str =
    "ullm.sq8.artifact_gpu_fp32_reference.v1";

const FP8_BLOCK: usize = 128;
const IO_CHUNK_BYTES: usize = 8 * 1024 * 1024;
const EMBEDDING_TENSOR: &str = "model.embed_tokens.weight";
const FINAL_NORM_TENSOR: &str = "model.norm.weight";
const LM_HEAD_TENSOR: &str = "lm_head.weight";

#[derive(Debug, Clone, Serialize)]
pub struct ArtifactGpuFp32ReferenceDeviceInfo {
    pub total_global_mem_bytes: u64,
    pub free_global_mem_bytes: u64,
    pub name: String,
    pub gcn_arch_name: String,
    pub pci_bdf: String,
}

impl From<Sq8Fp32GpuReferenceGfx1201DeviceInfo> for ArtifactGpuFp32ReferenceDeviceInfo {
    fn from(value: Sq8Fp32GpuReferenceGfx1201DeviceInfo) -> Self {
        Self {
            total_global_mem_bytes: value.total_global_mem_bytes,
            free_global_mem_bytes: value.free_global_mem_bytes,
            name: value.name,
            gcn_arch_name: value.gcn_arch_name,
            pci_bdf: value.pci_bdf,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ArtifactGpuFp32ReferenceIdentity {
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
    pub max_context: usize,
    pub device_before_upload: ArtifactGpuFp32ReferenceDeviceInfo,
    pub device_after_finalize: ArtifactGpuFp32ReferenceDeviceInfo,
    pub fp8_decode: &'static str,
    pub bf16_decode: &'static str,
    pub projection: &'static str,
    pub attention: &'static str,
    pub kv_cache: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct ArtifactGpuFp32CaptureReceipt {
    pub schema_version: &'static str,
    pub identity: ArtifactGpuFp32ReferenceIdentity,
    pub forward: ArtifactFp32ForwardSummary,
    pub files: BTreeMap<String, String>,
}

/// Stateful full-model control.  The opaque HIP session maintains an F32 KV
/// cache and poisons itself after a failed forward; `reset` creates a new
/// sequence boundary only after synchronization.
pub struct ArtifactGpuFp32ReferenceModel {
    session: Sq8Fp32GpuReferenceGfx1201Session,
    identity: ArtifactGpuFp32ReferenceIdentity,
    position: usize,
    max_context: usize,
}

impl ArtifactGpuFp32ReferenceModel {
    /// Validates the canonical inputs, uploads their raw bytes to the
    /// standalone GPU control, then finalizes device-resident F32 workspace.
    pub fn open(
        artifact_dir: impl AsRef<Path>,
        package_dir: impl AsRef<Path>,
        max_context: usize,
    ) -> Result<Self, String> {
        if max_context == 0 || max_context > QWEN3_14B_FP32_REFERENCE_MAX_CONTEXT {
            return Err(format!(
                "GPU artifact-FP32 max_context must be in 1..={}, got {max_context}",
                QWEN3_14B_FP32_REFERENCE_MAX_CONTEXT
            ));
        }
        let inputs = ArtifactFp32VerifiedInputs::open(artifact_dir, package_dir)?;
        let mut session = Sq8Fp32GpuReferenceGfx1201Session::create(max_context)?;
        let device_before_upload = ArtifactGpuFp32ReferenceDeviceInfo::from(session.device_info()?);
        if !exact_gfx1201(&device_before_upload.gcn_arch_name) {
            return Err(format!(
                "GPU artifact-FP32 selected unexpected architecture {}; expected exact gfx1201",
                device_before_upload.gcn_arch_name
            ));
        }

        upload_model_raw_bytes(&mut session, &inputs)?;
        session.finalize_model()?;
        let device_after_finalize =
            ArtifactGpuFp32ReferenceDeviceInfo::from(session.device_info()?);
        let checksums = inputs.artifact.checksum_report();
        let identity = ArtifactGpuFp32ReferenceIdentity {
            reference_id: ARTIFACT_GPU_FP32_REFERENCE_ID,
            artifact_content_sha256: inputs.artifact.manifest().integrity.content_sha256.clone(),
            artifact_weight_payload_bytes: checksums.weight_payload_bytes,
            artifact_scale_payload_bytes: checksums.scale_payload_bytes,
            artifact_quantized_tensor_count: inputs.artifact.manifest().quantized_tensors.len(),
            package: inputs.package_verification.clone(),
            layer_count: QWEN3_14B_FP32_REFERENCE_LAYERS,
            hidden_size: QWEN3_14B_HIDDEN_SIZE,
            intermediate_size: QWEN3_14B_INTERMEDIATE_SIZE,
            q_heads: QWEN3_14B_Q_HEADS,
            kv_heads: QWEN3_14B_KV_HEADS,
            head_dim: QWEN3_14B_HEAD_DIM,
            vocab_size: QWEN3_14B_FP32_REFERENCE_VOCAB_SIZE,
            max_context,
            device_before_upload,
            device_after_finalize,
            fp8_decode: "direct_scalar_ocp_e4m3fn_to_f32_on_gpu",
            bf16_decode: "direct_scalar_bf16_to_f32_on_gpu",
            projection: "standard_hipblas_sgemm_f32_no_quantized_kernel",
            attention: "literal_three_pass_serial_head_causal_softmax_f32",
            kv_cache: "device_resident_f32",
        };
        Ok(Self {
            session,
            identity,
            position: 0,
            max_context,
        })
    }

    pub fn identity(&self) -> &ArtifactGpuFp32ReferenceIdentity {
        &self.identity
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn forward_token(&mut self, token_id: u32) -> Result<ArtifactFp32Forward, String> {
        if self.position >= self.max_context {
            return Err(format!(
                "GPU artifact-FP32 context limit {} reached before token {token_id}",
                self.max_context
            ));
        }
        let mut logits = zeroed_f32(QWEN3_14B_FP32_REFERENCE_VOCAB_SIZE, "GPU logits")?;
        let mut final_hidden = zeroed_f32(QWEN3_14B_HIDDEN_SIZE, "GPU final hidden")?;
        let layer_elements = QWEN3_14B_FP32_REFERENCE_LAYERS
            .checked_mul(QWEN3_14B_HIDDEN_SIZE)
            .ok_or_else(|| "GPU artifact-FP32 layer capture length overflows".to_string())?;
        let mut layer_flat = zeroed_f32(layer_elements, "GPU layer hidden")?;
        self.session
            .forward(token_id, &mut logits, &mut final_hidden, &mut layer_flat)?;

        let greedy_token_id = greedy_token(&logits)?;
        let layer_hidden = layer_flat
            .chunks_exact(QWEN3_14B_HIDDEN_SIZE)
            .map(<[f32]>::to_vec)
            .collect::<Vec<_>>();
        if layer_hidden.len() != QWEN3_14B_FP32_REFERENCE_LAYERS {
            return Err("GPU artifact-FP32 did not retain all layer hidden states".to_string());
        }
        let summary = ArtifactFp32ForwardSummary {
            position: self.position,
            input_token_id: token_id,
            greedy_token_id,
            logits_f32le_sha256: f32_le_sha256(&logits, "GPU logits")?,
            final_hidden_f32le_sha256: f32_le_sha256(&final_hidden, "GPU final hidden")?,
            layer_hidden_f32le_sha256: layer_hidden
                .iter()
                .map(|values| f32_le_sha256(values, "GPU layer hidden"))
                .collect::<Result<Vec<_>, _>>()?,
        };
        self.position += 1;
        Ok(ArtifactFp32Forward {
            summary,
            logits,
            final_hidden,
            layer_hidden,
        })
    }

    pub fn reset(&mut self) -> Result<(), String> {
        self.session.reset()?;
        self.position = 0;
        Ok(())
    }
}

/// Writes a no-clobber capture with the same tensor layout and per-tensor
/// content hashes as the CPU strict F32 reference.
pub fn write_gpu_forward_capture(
    root: impl AsRef<Path>,
    identity: &ArtifactGpuFp32ReferenceIdentity,
    forward: &ArtifactFp32Forward,
) -> Result<ArtifactGpuFp32CaptureReceipt, String> {
    let root = root.as_ref();
    fs::create_dir_all(root).map_err(|error| {
        format!(
            "GPU artifact-FP32 failed to create capture root {}: {error}",
            root.display()
        )
    })?;
    let layer_root = root.join("layers");
    fs::create_dir_all(&layer_root).map_err(|error| {
        format!(
            "GPU artifact-FP32 failed to create layer capture root {}: {error}",
            layer_root.display()
        )
    })?;

    let mut files = BTreeMap::new();
    let logits_path = root.join("logits.f32le");
    files.insert(
        "logits.f32le".to_string(),
        write_f32le_create_new(&logits_path, &forward.logits, "GPU logits")?,
    );
    let final_hidden_path = root.join("final-hidden.f32le");
    files.insert(
        "final-hidden.f32le".to_string(),
        write_f32le_create_new(
            &final_hidden_path,
            &forward.final_hidden,
            "GPU final hidden",
        )?,
    );
    for (layer_index, values) in forward.layer_hidden.iter().enumerate() {
        let relative = format!("layers/layer-{layer_index:02}-hidden.f32le");
        files.insert(
            relative.clone(),
            write_f32le_create_new(&root.join(&relative), values, "GPU layer hidden")?,
        );
    }
    let receipt = ArtifactGpuFp32CaptureReceipt {
        schema_version: ARTIFACT_GPU_FP32_REFERENCE_SCHEMA_VERSION,
        identity: identity.clone(),
        forward: forward.summary.clone(),
        files,
    };
    write_json_create_new(&root.join("metadata.json"), &receipt)?;
    Ok(receipt)
}

fn exact_gfx1201(value: &str) -> bool {
    value == "gfx1201" || value.starts_with("gfx1201:")
}

fn zeroed_f32(elements: usize, label: &str) -> Result<Vec<f32>, String> {
    let mut values = Vec::new();
    values.try_reserve_exact(elements).map_err(|error| {
        format!("GPU artifact-FP32 failed to reserve {elements} F32 values for {label}: {error}")
    })?;
    values.resize(elements, 0.0);
    Ok(values)
}

fn upload_model_raw_bytes(
    session: &mut Sq8Fp32GpuReferenceGfx1201Session,
    inputs: &ArtifactFp32VerifiedInputs,
) -> Result<(), String> {
    for layer_index in 0..QWEN3_14B_FP32_REFERENCE_LAYERS {
        let prefix = format!("model.layers.{layer_index}");
        for (suffix, rows, cols) in [
            (
                "self_attn.q_proj.weight",
                QWEN3_14B_Q_HEADS * QWEN3_14B_HEAD_DIM,
                QWEN3_14B_HIDDEN_SIZE,
            ),
            (
                "self_attn.k_proj.weight",
                QWEN3_14B_KV_HEADS * QWEN3_14B_HEAD_DIM,
                QWEN3_14B_HIDDEN_SIZE,
            ),
            (
                "self_attn.v_proj.weight",
                QWEN3_14B_KV_HEADS * QWEN3_14B_HEAD_DIM,
                QWEN3_14B_HIDDEN_SIZE,
            ),
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
            upload_quantized_weight(session, inputs, &format!("{prefix}.{suffix}"), rows, cols)?;
        }

        let input = read_bundle_small(
            inputs.bundle(&format!("{prefix}.input_layernorm.weight"))?,
            QWEN3_14B_HIDDEN_SIZE * std::mem::size_of::<u16>(),
        )?;
        let post_attention = read_bundle_small(
            inputs.bundle(&format!("{prefix}.post_attention_layernorm.weight"))?,
            QWEN3_14B_HIDDEN_SIZE * std::mem::size_of::<u16>(),
        )?;
        let q = read_bundle_small(
            inputs.bundle(&format!("{prefix}.self_attn.q_norm.weight"))?,
            QWEN3_14B_HEAD_DIM * std::mem::size_of::<u16>(),
        )?;
        let k = read_bundle_small(
            inputs.bundle(&format!("{prefix}.self_attn.k_norm.weight"))?,
            QWEN3_14B_HEAD_DIM * std::mem::size_of::<u16>(),
        )?;
        session.upload_layer_norms(layer_index, &input, &post_attention, &q, &k)?;
    }

    let final_norm = read_bundle_small(
        inputs.bundle(FINAL_NORM_TENSOR)?,
        QWEN3_14B_HIDDEN_SIZE * std::mem::size_of::<u16>(),
    )?;
    session.upload_final_norm(&final_norm)?;
    upload_bf16_tensor(
        session,
        "embedding",
        inputs.bundle(EMBEDDING_TENSOR)?,
        QWEN3_14B_FP32_REFERENCE_VOCAB_SIZE * QWEN3_14B_HIDDEN_SIZE,
    )?;
    upload_bf16_tensor(
        session,
        "lm_head",
        inputs.bundle(LM_HEAD_TENSOR)?,
        QWEN3_14B_FP32_REFERENCE_VOCAB_SIZE * QWEN3_14B_HIDDEN_SIZE,
    )?;
    Ok(())
}

fn upload_quantized_weight(
    session: &mut Sq8Fp32GpuReferenceGfx1201Session,
    inputs: &ArtifactFp32VerifiedInputs,
    tensor_name: &str,
    expected_rows: usize,
    expected_cols: usize,
) -> Result<(), String> {
    if expected_rows % FP8_BLOCK != 0 || expected_cols % FP8_BLOCK != 0 {
        return Err(format!(
            "GPU artifact-FP32 expected non-block-aligned canonical weight {tensor_name}"
        ));
    }
    let pair = inputs.artifact.tensor_pair(tensor_name).map_err(|error| {
        format!("GPU artifact-FP32 missing canonical weight {tensor_name}: {error}")
    })?;
    if pair.shape != [expected_rows as u64, expected_cols as u64]
        || pair.weight.bytes != (expected_rows * expected_cols) as u64
        || pair.scale.block_shape != [FP8_BLOCK as u64, FP8_BLOCK as u64]
        || pair.scale.shape
            != [
                (expected_rows / FP8_BLOCK) as u64,
                (expected_cols / FP8_BLOCK) as u64,
            ]
    {
        return Err(format!(
            "GPU artifact-FP32 canonical SQ8 contract mismatch for {tensor_name}"
        ));
    }
    let paths = inputs
        .artifact
        .tensor_payload_paths(tensor_name)
        .map_err(|error| format!("GPU artifact-FP32 could not resolve {tensor_name}: {error}"))?;
    let scales = read_verified_small_file(
        &paths.scale,
        pair.scale.bytes,
        &pair.scale.sha256,
        &format!("{tensor_name} BF16 scales"),
    )?;
    session.reserve_sq8_weight(tensor_name, expected_rows, expected_cols, &scales)?;
    stream_verified_file(
        &paths.weight,
        pair.weight.bytes,
        &pair.weight.sha256,
        &format!("{tensor_name} OCP payload"),
        |offset, bytes| session.upload_sq8_weight_chunk(tensor_name, offset, bytes),
    )
}

fn upload_bf16_tensor(
    session: &mut Sq8Fp32GpuReferenceGfx1201Session,
    slot: &str,
    bundle: &PassthroughPayloadBundle,
    expected_elements: usize,
) -> Result<(), String> {
    if bundle.dtype.as_deref() != Some("BF16")
        || bundle.elements != expected_elements as u64
        || bundle.payload_bytes
            != u64::try_from(expected_elements * std::mem::size_of::<u16>())
                .map_err(|_| format!("GPU artifact-FP32 {slot} byte count overflows u64"))?
    {
        return Err(format!("GPU artifact-FP32 {slot} bundle contract mismatch"));
    }
    let expected_sha256 = bundle.payload_sha256.as_deref().ok_or_else(|| {
        format!("GPU artifact-FP32 {slot} bundle is missing its verified payload SHA-256")
    })?;
    session.reserve_bf16_tensor(slot, expected_elements)?;
    stream_verified_file(
        &bundle.payload_file.absolute_path,
        bundle.payload_bytes,
        expected_sha256,
        &format!("{slot} BF16 payload"),
        |offset, bytes| session.upload_bf16_tensor_chunk(slot, offset, bytes),
    )
}

fn read_bundle_small(
    bundle: &PassthroughPayloadBundle,
    expected_bytes: usize,
) -> Result<Vec<u8>, String> {
    let actual_bytes = usize::try_from(bundle.payload_bytes).map_err(|_| {
        format!(
            "GPU artifact-FP32 {} bytes do not fit usize",
            bundle.tensor_name
        )
    })?;
    if bundle.dtype.as_deref() != Some("BF16") || actual_bytes != expected_bytes {
        return Err(format!(
            "GPU artifact-FP32 raw BF16 small parameter contract mismatch for {}",
            bundle.tensor_name
        ));
    }
    let expected_sha256 = bundle.payload_sha256.as_deref().ok_or_else(|| {
        format!(
            "GPU artifact-FP32 {} is missing its verified payload SHA-256",
            bundle.tensor_name
        )
    })?;
    read_verified_small_file(
        &bundle.payload_file.absolute_path,
        bundle.payload_bytes,
        expected_sha256,
        &bundle.tensor_name,
    )
}

fn read_verified_small_file(
    path: &Path,
    expected_bytes: u64,
    expected_sha256: &str,
    label: &str,
) -> Result<Vec<u8>, String> {
    let bytes = usize::try_from(expected_bytes)
        .map_err(|_| format!("GPU artifact-FP32 {label} byte count does not fit usize"))?;
    if bytes > IO_CHUNK_BYTES {
        return Err(format!(
            "GPU artifact-FP32 {label} unexpectedly exceeds compact raw-input limit"
        ));
    }
    let mut result = vec![0_u8; bytes];
    stream_verified_file(
        path,
        expected_bytes,
        expected_sha256,
        label,
        |offset, chunk| {
            let end = offset
                .checked_add(chunk.len())
                .ok_or_else(|| format!("GPU artifact-FP32 {label} copy offset overflows"))?;
            result
                .get_mut(offset..end)
                .ok_or_else(|| format!("GPU artifact-FP32 {label} copy escaped destination"))?
                .copy_from_slice(chunk);
            Ok(())
        },
    )?;
    Ok(result)
}

fn stream_verified_file(
    path: &Path,
    expected_bytes: u64,
    expected_sha256: &str,
    label: &str,
    mut consume: impl FnMut(usize, &[u8]) -> Result<(), String>,
) -> Result<(), String> {
    let expected = usize::try_from(expected_bytes)
        .map_err(|_| format!("GPU artifact-FP32 {label} byte count does not fit usize"))?;
    let metadata = fs::metadata(path)
        .map_err(|error| format!("GPU artifact-FP32 failed to stat {label}: {error}"))?;
    if metadata.len() != expected_bytes {
        return Err(format!(
            "GPU artifact-FP32 {label} byte count changed: expected={expected_bytes} actual={}",
            metadata.len()
        ));
    }
    let mut file = File::open(path)
        .map_err(|error| format!("GPU artifact-FP32 failed to open {label}: {error}"))?;
    let mut buffer = vec![0_u8; IO_CHUNK_BYTES.min(expected.max(1))];
    let mut digest = Sha256::new();
    let mut offset = 0_usize;
    while offset < expected {
        let read_len = (expected - offset).min(buffer.len());
        file.read_exact(&mut buffer[..read_len])
            .map_err(|error| format!("GPU artifact-FP32 failed to read {label}: {error}"))?;
        digest.update(&buffer[..read_len]);
        consume(offset, &buffer[..read_len])?;
        offset = offset
            .checked_add(read_len)
            .ok_or_else(|| format!("GPU artifact-FP32 {label} offset overflows"))?;
    }
    let mut trailing = [0_u8; 1];
    if file
        .read(&mut trailing)
        .map_err(|error| format!("GPU artifact-FP32 failed to check EOF for {label}: {error}"))?
        != 0
    {
        return Err(format!("GPU artifact-FP32 {label} gained trailing bytes"));
    }
    let actual_sha256 = format!("{:x}", digest.finalize());
    if actual_sha256 != expected_sha256 {
        return Err(format!(
            "GPU artifact-FP32 {label} SHA-256 changed: expected={expected_sha256} actual={actual_sha256}"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gfx1201_admission_is_exact_and_fail_closed() {
        assert!(exact_gfx1201("gfx1201"));
        assert!(exact_gfx1201("gfx1201:sramecc+:xnack-"));
        assert!(!exact_gfx1201("gfx1030"));
        assert!(!exact_gfx1201("gfx12010"));
        assert!(!exact_gfx1201("gfx1200"));
        assert!(!exact_gfx1201(""));
    }

    #[test]
    fn control_source_has_no_optimized_sq8_header_dependency() {
        let source = include_str!("../../../runtime/src/sq8_fp32_gpu_reference_gfx1201.hip.cpp");
        for forbidden in [
            "#include \"sq8_ck",
            "#include \"sq8_handwritten",
            "#include <ck/",
            "#include <rocwmma",
        ] {
            assert!(
                !source.contains(forbidden),
                "GPU F32 control must not include optimized SQ8 dependency {forbidden}"
            );
        }
        assert!(source.contains("hipblasSgemm"));
        assert!(source.contains("dequant_sq8_ocp_block128_to_f32"));
    }
}
