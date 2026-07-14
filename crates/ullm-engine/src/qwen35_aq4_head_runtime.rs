// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Resident embedding, final RMSNorm, and LM-head runtime for Qwen3.5 AQ4 packages.

use std::env;
use std::fs::File;
use std::io::Read;

use crate::aq4_package_runtime::PackageAq4ResidentMatvec;
use crate::host_bytes::{decode_f32_le_values, encode_f32_to_bytes};
use crate::loader::{
    PassthroughF32Data, WeightRegistry, matrix_shape_rows_cols, read_named_passthrough_f32,
    read_named_passthrough_f32_row_range, resolve_passthrough_dtype,
    validate_passthrough_shape_elements,
};
use crate::package::{
    TensorSelector, select_passthrough_payload_bundle, select_tensor_payload_bundle,
};

pub const QWEN3_EMBED_TOKENS_TENSOR: &str = "model.language_model.embed_tokens.weight";
pub const QWEN3_FINAL_NORM_TENSOR: &str = "model.language_model.norm.weight";
pub const QWEN3_LM_HEAD_TENSOR: &str = "lm_head.weight";

/// Number of f32 values exposed by one calibration visitor call.
///
/// Calibration is deliberately chunked: callers may reduce a complete hidden/logit vector, but
/// cannot retain a borrow into the runtime-owned staging buffer after the callback returns.
pub const QWEN35_AQ4_CALIBRATION_VISIT_CHUNK_ELEMENTS: usize = 1024;

#[derive(Debug, Clone, PartialEq)]
pub struct PackageTokenLogit {
    pub token_id: usize,
    pub logit: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackageLmHeadMode {
    CpuChunked,
    GpuResidentF32,
}

impl PackageLmHeadMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CpuChunked => "cpu_chunked",
            Self::GpuResidentF32 => "gpu_resident_f32",
        }
    }
}

#[derive(Clone, Copy)]
pub enum PackageLmHeadMatrixStorage {
    F32,
    Bf16,
}

impl PackageLmHeadMatrixStorage {
    fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::Bf16 => "BF16",
        }
    }

    fn element_size(self) -> usize {
        match self {
            Self::F32 => std::mem::size_of::<f32>(),
            Self::Bf16 => std::mem::size_of::<u16>(),
        }
    }
}

fn checked_f32_byte_len(elements: usize, label: &str) -> Result<usize, String> {
    elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| format!("{label} byte size overflows"))
}

fn read_runtime_buffer_f32(
    buffer: &ullm_runtime_sys::RuntimeBuffer,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    elements: usize,
    label: &str,
) -> Result<Vec<f32>, String> {
    let mut bytes = vec![0_u8; checked_f32_byte_len(elements, label)?];
    buffer
        .copy_to_host(0, &mut bytes, Some(stream))
        .map_err(|err| format!("failed to copy {label}: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize {label}: {err}"))?;
    Ok(decode_f32_le_values(&bytes))
}

fn env_flag_enabled(name: &str) -> bool {
    env::var(name)
        .map(|value| flag_value_enabled(&value))
        .unwrap_or(false)
}

fn flag_value_enabled(value: &str) -> bool {
    matches!(value, "1" | "true" | "TRUE" | "yes" | "YES")
}

pub enum PackageLmHeadRuntime {
    CpuChunked {
        chunk_rows: usize,
    },
    GpuResidentAq4 {
        shape: Vec<u64>,
        vocab: usize,
        hidden: usize,
        matrix: PackageAq4ResidentMatvec,
        input_buffer: ullm_runtime_sys::RuntimeBuffer,
        logits_buffer: ullm_runtime_sys::RuntimeBuffer,
        top1_partial_count: usize,
        top1_partial_values_buffer: ullm_runtime_sys::RuntimeBuffer,
        top1_partial_indices_buffer: ullm_runtime_sys::RuntimeBuffer,
        top1_partial_values_host: Vec<u8>,
        top1_partial_indices_host: Vec<u8>,
        logits_host: Vec<u8>,
        /// Immutable execution policy captured while the resident head is loaded.
        direct_top1_enabled: bool,
    },
    GpuResidentF32 {
        dtype: String,
        shape: Vec<u64>,
        vocab: usize,
        hidden: usize,
        matrix_storage: PackageLmHeadMatrixStorage,
        top1_partial_count: usize,
        matrix_buffer: ullm_runtime_sys::RuntimeBuffer,
        input_buffer: ullm_runtime_sys::RuntimeBuffer,
        logits_buffer: ullm_runtime_sys::RuntimeBuffer,
        top1_partial_values_buffer: ullm_runtime_sys::RuntimeBuffer,
        top1_partial_indices_buffer: ullm_runtime_sys::RuntimeBuffer,
        top1_partial_values_host: Vec<u8>,
        top1_partial_indices_host: Vec<u8>,
        logits_host: Vec<u8>,
        matrix_bytes: usize,
    },
}

impl PackageLmHeadRuntime {
    pub fn load(
        mode: PackageLmHeadMode,
        context: &mut ullm_runtime_sys::RuntimeContext,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        path: &str,
        chunk_bytes: usize,
        hidden: usize,
        chunk_rows: usize,
    ) -> Result<Self, String> {
        match mode {
            PackageLmHeadMode::CpuChunked => Ok(Self::CpuChunked { chunk_rows }),
            PackageLmHeadMode::GpuResidentF32 => {
                let direct_top1_enabled = env_flag_enabled("ULLM_ENABLE_AQ4_LM_HEAD_DIRECT_TOP1");
                let selector = TensorSelector::Name(QWEN3_LM_HEAD_TENSOR.to_string());
                if select_tensor_payload_bundle(path, &selector).is_ok() {
                    let mut registry = WeightRegistry::new();
                    let matrix = PackageAq4ResidentMatvec::load(
                        context,
                        stream,
                        &mut registry,
                        path,
                        QWEN3_LM_HEAD_TENSOR,
                        chunk_bytes,
                    )
                    .map_err(|err| format!("failed to load resident AQ4 lm_head tensor: {err}"))?;
                    let vocab = matrix.rows;
                    let cols = matrix.cols;
                    if cols != hidden {
                        return Err(format!(
                            "resident AQ4 lm_head hidden mismatch: lm_head={cols} hidden={hidden}"
                        ));
                    }
                    let shape = vec![
                        u64::try_from(vocab)
                            .map_err(|_| "resident AQ4 lm_head vocab exceeds u64".to_string())?,
                        u64::try_from(hidden)
                            .map_err(|_| "resident AQ4 lm_head hidden exceeds u64".to_string())?,
                    ];
                    let hidden_bytes = checked_f32_byte_len(hidden, "resident AQ4 lm_head input")?;
                    let logits_bytes = checked_f32_byte_len(vocab, "resident AQ4 lm_head logits")?;
                    let logits_top1_partial_count = ullm_runtime_sys::top1_partial_count(vocab)
                        .map_err(|err| {
                            format!("failed to size resident AQ4 lm_head top1: {err}")
                        })?;
                    let aq4_direct_top1_partial_count =
                        ullm_runtime_sys::aq4_matvec_top1_partial_count(vocab).map_err(|err| {
                            format!("failed to size resident AQ4 lm_head direct top1: {err}")
                        })?;
                    let top1_partial_count =
                        logits_top1_partial_count.max(aq4_direct_top1_partial_count);
                    let top1_partial_values_bytes = checked_f32_byte_len(
                        top1_partial_count,
                        "resident AQ4 lm_head top1 values",
                    )?;
                    let top1_partial_indices_bytes = top1_partial_count
                        .checked_mul(std::mem::size_of::<u32>())
                        .ok_or_else(|| {
                            "resident AQ4 lm_head top1 index byte size overflows".to_string()
                        })?;
                    let mut input_buffer = context.alloc_buffer(hidden_bytes).map_err(|err| {
                        format!("failed to allocate resident AQ4 lm_head input: {err}")
                    })?;
                    let mut logits_buffer = context.alloc_buffer(logits_bytes).map_err(|err| {
                        format!("failed to allocate resident AQ4 lm_head logits: {err}")
                    })?;
                    let mut top1_partial_values_buffer = context
                        .alloc_buffer(top1_partial_values_bytes)
                        .map_err(|err| {
                            format!("failed to allocate resident AQ4 lm_head top1 values: {err}")
                        })?;
                    let mut top1_partial_indices_buffer = context
                        .alloc_buffer(top1_partial_indices_bytes)
                        .map_err(|err| {
                            format!("failed to allocate resident AQ4 lm_head top1 indices: {err}")
                        })?;
                    stream.synchronize().map_err(|err| {
                        format!("failed to synchronize resident AQ4 lm_head load: {err}")
                    })?;
                    let mut top1_partial_values_host = vec![0_u8; top1_partial_values_bytes];
                    let mut top1_partial_indices_host = vec![0_u8; top1_partial_indices_bytes];
                    let mut logits_host = vec![0_u8; logits_bytes];
                    let zero_hidden_values = vec![0.0_f32; hidden];
                    input_buffer
                        .copy_from_host(0, &encode_f32_to_bytes(&zero_hidden_values), Some(stream))
                        .map_err(|err| {
                            format!("failed to copy resident AQ4 lm_head prewarm input: {err}")
                        })?;
                    package_gpu_resident_aq4_lm_head_top_logits(
                        stream,
                        &matrix,
                        &input_buffer,
                        vocab,
                        hidden,
                        &mut logits_buffer,
                        &mut top1_partial_values_buffer,
                        &mut top1_partial_indices_buffer,
                        &mut top1_partial_values_host,
                        &mut top1_partial_indices_host,
                        &mut logits_host,
                        1,
                        direct_top1_enabled,
                    )
                    .map_err(|err| format!("failed to prewarm resident AQ4 lm_head: {err}"))?;
                    return Ok(Self::GpuResidentAq4 {
                        shape,
                        vocab,
                        hidden,
                        matrix,
                        input_buffer,
                        logits_buffer,
                        top1_partial_count,
                        top1_partial_values_buffer,
                        top1_partial_indices_buffer,
                        top1_partial_values_host,
                        top1_partial_indices_host,
                        logits_host,
                        direct_top1_enabled,
                    });
                }
                let bundle = select_passthrough_payload_bundle(path, &selector)
                    .map_err(|err| format!("failed to select resident lm_head tensor: {err}"))?;
                validate_passthrough_shape_elements(&bundle)
                    .map_err(|err| format!("invalid resident lm_head shape: {err}"))?;
                let dtype = resolve_passthrough_dtype(&bundle, QWEN3_LM_HEAD_TENSOR)?.to_string();
                if bundle.shape.len() != 2 {
                    return Err(format!(
                        "resident lm_head must be 2D, got shape {:?}",
                        bundle.shape
                    ));
                }
                let vocab = usize::try_from(bundle.shape[0])
                    .map_err(|_| "resident lm_head vocab size is too large".to_string())?;
                let cols = usize::try_from(bundle.shape[1])
                    .map_err(|_| "resident lm_head hidden size is too large".to_string())?;
                if cols != hidden {
                    return Err(format!(
                        "resident lm_head hidden mismatch: lm_head={cols} hidden={hidden}"
                    ));
                }
                let expected_values = vocab
                    .checked_mul(hidden)
                    .ok_or_else(|| "resident lm_head element count overflows".to_string())?;
                if u64::try_from(expected_values).ok() != Some(bundle.elements) {
                    return Err(format!(
                        "resident lm_head element count mismatch: got {} expected {expected_values}",
                        bundle.elements
                    ));
                }
                let matrix_storage = match dtype.as_str() {
                    "BF16" => PackageLmHeadMatrixStorage::Bf16,
                    _ => PackageLmHeadMatrixStorage::F32,
                };
                let matrix_bytes = expected_values
                    .checked_mul(matrix_storage.element_size())
                    .ok_or_else(|| "resident lm_head matrix byte size overflows".to_string())?;
                let hidden_bytes = checked_f32_byte_len(hidden, "resident lm_head input")?;
                let logits_bytes = checked_f32_byte_len(vocab, "resident lm_head logits")?;
                let top1_partial_count = ullm_runtime_sys::top1_partial_count(vocab)
                    .map_err(|err| format!("failed to size resident lm_head top1: {err}"))?;
                let top1_partial_values_bytes =
                    checked_f32_byte_len(top1_partial_count, "resident lm_head top1 values")?;
                let top1_partial_indices_bytes = top1_partial_count
                    .checked_mul(std::mem::size_of::<u32>())
                    .ok_or_else(|| "resident lm_head top1 index byte size overflows".to_string())?;
                let mut matrix_buffer = context
                    .alloc_buffer(matrix_bytes)
                    .map_err(|err| format!("failed to allocate resident lm_head matrix: {err}"))?;
                match matrix_storage {
                    PackageLmHeadMatrixStorage::Bf16 => {
                        let payload_bytes = if bundle.payload_bytes == 0 {
                            bundle.payload_file.bytes
                        } else {
                            bundle.payload_bytes
                        };
                        if payload_bytes != bundle.payload_file.bytes {
                            return Err(format!(
                                "resident lm_head payload bytes mismatch: declared {} actual {}",
                                payload_bytes, bundle.payload_file.bytes
                            ));
                        }
                        if usize::try_from(payload_bytes).ok() != Some(matrix_bytes) {
                            return Err(format!(
                                "resident lm_head BF16 payload bytes mismatch: got {payload_bytes} expected {matrix_bytes}"
                            ));
                        }
                        match bundle.payload_encoding.as_deref() {
                            None | Some("raw_safetensors_payload") => {}
                            Some(encoding) => {
                                return Err(format!(
                                    "resident lm_head has unsupported payload encoding {encoding}"
                                ));
                            }
                        }
                        copy_file_to_runtime_buffer_chunked(
                            &mut matrix_buffer,
                            &bundle.payload_file.absolute_path,
                            matrix_bytes,
                            chunk_bytes,
                            stream,
                            "resident lm_head BF16 matrix",
                        )?;
                    }
                    PackageLmHeadMatrixStorage::F32 => {
                        let data =
                            read_named_passthrough_f32(path, QWEN3_LM_HEAD_TENSOR, chunk_bytes)
                                .map_err(|err| {
                                    format!("failed to read resident lm_head tensor: {err}")
                                })?;
                        if data.values.len() != expected_values {
                            return Err(format!(
                                "resident lm_head value count mismatch: got {} expected {expected_values}",
                                data.values.len()
                            ));
                        }
                        copy_f32_values_to_runtime_buffer_chunked(
                            &mut matrix_buffer,
                            &data.values,
                            stream,
                            "resident lm_head matrix",
                        )?;
                    }
                }
                let mut input_buffer = context
                    .alloc_buffer(hidden_bytes)
                    .map_err(|err| format!("failed to allocate resident lm_head input: {err}"))?;
                let mut logits_buffer = context
                    .alloc_buffer(logits_bytes)
                    .map_err(|err| format!("failed to allocate resident lm_head logits: {err}"))?;
                let mut top1_partial_values_buffer = context
                    .alloc_buffer(top1_partial_values_bytes)
                    .map_err(|err| {
                        format!("failed to allocate resident lm_head top1 values: {err}")
                    })?;
                let mut top1_partial_indices_buffer = context
                    .alloc_buffer(top1_partial_indices_bytes)
                    .map_err(|err| {
                        format!("failed to allocate resident lm_head top1 indices: {err}")
                    })?;
                stream
                    .synchronize()
                    .map_err(|err| format!("failed to synchronize resident lm_head load: {err}"))?;
                let mut top1_partial_values_host = vec![0_u8; top1_partial_values_bytes];
                let mut top1_partial_indices_host = vec![0_u8; top1_partial_indices_bytes];
                let mut logits_host = vec![0_u8; logits_bytes];
                let zero_hidden_values = vec![0.0_f32; hidden];
                input_buffer
                    .copy_from_host(0, &encode_f32_to_bytes(&zero_hidden_values), Some(stream))
                    .map_err(|err| {
                        format!("failed to copy resident lm_head prewarm input: {err}")
                    })?;
                package_gpu_resident_lm_head_top_logits(
                    stream,
                    &matrix_buffer,
                    &input_buffer,
                    matrix_storage,
                    vocab,
                    hidden,
                    &mut logits_buffer,
                    &mut top1_partial_values_buffer,
                    &mut top1_partial_indices_buffer,
                    &mut top1_partial_values_host,
                    &mut top1_partial_indices_host,
                    &mut logits_host,
                    1,
                )
                .map_err(|err| format!("failed to prewarm resident lm_head: {err}"))?;
                Ok(Self::GpuResidentF32 {
                    dtype,
                    shape: bundle.shape,
                    vocab,
                    hidden,
                    matrix_storage,
                    top1_partial_count,
                    matrix_buffer,
                    input_buffer,
                    logits_buffer,
                    top1_partial_values_buffer,
                    top1_partial_indices_buffer,
                    top1_partial_values_host,
                    top1_partial_indices_host,
                    logits_host,
                    matrix_bytes,
                })
            }
        }
    }

    pub fn top_logits(
        &mut self,
        path: &str,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        hidden_values: &[f32],
        top_k: usize,
    ) -> Result<Vec<PackageTokenLogit>, String> {
        match self {
            Self::CpuChunked { chunk_rows } => {
                let (_vocab, _dtype, _shape, top_logits) =
                    package_lm_head_top_k_from_rows(path, hidden_values, top_k, *chunk_rows)?;
                Ok(top_logits)
            }
            Self::GpuResidentAq4 {
                vocab,
                hidden,
                matrix,
                input_buffer,
                logits_buffer,
                top1_partial_values_buffer,
                top1_partial_indices_buffer,
                top1_partial_values_host,
                top1_partial_indices_host,
                logits_host,
                direct_top1_enabled,
                ..
            } => {
                if hidden_values.len() != *hidden {
                    return Err(format!(
                        "resident AQ4 lm_head input length mismatch: got {} expected {}",
                        hidden_values.len(),
                        hidden
                    ));
                }
                input_buffer
                    .copy_from_host(0, &encode_f32_to_bytes(hidden_values), Some(stream))
                    .map_err(|err| format!("failed to copy resident AQ4 lm_head input: {err}"))?;
                package_gpu_resident_aq4_lm_head_top_logits(
                    stream,
                    matrix,
                    input_buffer,
                    *vocab,
                    *hidden,
                    logits_buffer,
                    top1_partial_values_buffer,
                    top1_partial_indices_buffer,
                    top1_partial_values_host,
                    top1_partial_indices_host,
                    logits_host,
                    top_k,
                    *direct_top1_enabled,
                )
            }
            Self::GpuResidentF32 {
                vocab,
                hidden,
                matrix_storage,
                matrix_buffer,
                input_buffer,
                logits_buffer,
                top1_partial_values_buffer,
                top1_partial_indices_buffer,
                top1_partial_values_host,
                top1_partial_indices_host,
                logits_host,
                ..
            } => {
                if hidden_values.len() != *hidden {
                    return Err(format!(
                        "resident lm_head input length mismatch: got {} expected {}",
                        hidden_values.len(),
                        hidden
                    ));
                }
                input_buffer
                    .copy_from_host(0, &encode_f32_to_bytes(hidden_values), Some(stream))
                    .map_err(|err| format!("failed to copy resident lm_head input: {err}"))?;
                package_gpu_resident_lm_head_top_logits(
                    stream,
                    matrix_buffer,
                    input_buffer,
                    *matrix_storage,
                    *vocab,
                    *hidden,
                    logits_buffer,
                    top1_partial_values_buffer,
                    top1_partial_indices_buffer,
                    top1_partial_values_host,
                    top1_partial_indices_host,
                    logits_host,
                    top_k,
                )
            }
        }
    }

    pub fn supports_device_input(&self) -> bool {
        matches!(
            self,
            Self::GpuResidentAq4 { .. } | Self::GpuResidentF32 { .. }
        )
    }

    pub fn top_logits_from_device_buffer(
        &mut self,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        hidden_buffer: &ullm_runtime_sys::RuntimeBuffer,
        top_k: usize,
    ) -> Result<Vec<PackageTokenLogit>, String> {
        match self {
            Self::CpuChunked { .. } => {
                Err("device lm_head input requires gpu_resident_f32 lm_head mode".to_string())
            }
            Self::GpuResidentAq4 {
                vocab,
                hidden,
                matrix,
                logits_buffer,
                top1_partial_values_buffer,
                top1_partial_indices_buffer,
                top1_partial_values_host,
                top1_partial_indices_host,
                logits_host,
                direct_top1_enabled,
                ..
            } => package_gpu_resident_aq4_lm_head_top_logits(
                stream,
                matrix,
                hidden_buffer,
                *vocab,
                *hidden,
                logits_buffer,
                top1_partial_values_buffer,
                top1_partial_indices_buffer,
                top1_partial_values_host,
                top1_partial_indices_host,
                logits_host,
                top_k,
                *direct_top1_enabled,
            ),
            Self::GpuResidentF32 {
                vocab,
                hidden,
                matrix_storage,
                matrix_buffer,
                logits_buffer,
                top1_partial_values_buffer,
                top1_partial_indices_buffer,
                top1_partial_values_host,
                top1_partial_indices_host,
                logits_host,
                ..
            } => package_gpu_resident_lm_head_top_logits(
                stream,
                matrix_buffer,
                hidden_buffer,
                *matrix_storage,
                *vocab,
                *hidden,
                logits_buffer,
                top1_partial_values_buffer,
                top1_partial_indices_buffer,
                top1_partial_values_host,
                top1_partial_indices_host,
                logits_host,
                top_k,
            ),
        }
    }

    /// Copies the already-produced resident logit row to the runtime-owned host staging buffer
    /// and visits it in monotonically increasing token-id chunks.
    ///
    /// This never launches the LM head again and allocates no second vocabulary-sized vector.
    /// It is intended only for an opt-in calibration path after `top_logits_from_device_buffer`
    /// has produced the current prepared token's logits.
    pub fn visit_last_device_logits(
        &mut self,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        visitor: &mut dyn FnMut(usize, &[f32]) -> Result<(), String>,
    ) -> Result<usize, String> {
        let (vocab, logits_buffer, logits_host) = match self {
            Self::CpuChunked { .. } => {
                return Err(
                    "calibration logit observation requires a resident device LM head".into(),
                );
            }
            Self::GpuResidentAq4 {
                vocab,
                logits_buffer,
                logits_host,
                ..
            }
            | Self::GpuResidentF32 {
                vocab,
                logits_buffer,
                logits_host,
                ..
            } => (*vocab, logits_buffer, logits_host),
        };
        let expected_bytes = checked_f32_byte_len(vocab, "calibration logits")?;
        if logits_host.len() != expected_bytes {
            return Err(format!(
                "calibration logit host staging length differs: got {} expected {expected_bytes}",
                logits_host.len()
            ));
        }
        logits_buffer
            .copy_to_host(0, logits_host, Some(stream))
            .map_err(|err| format!("failed to copy calibration logits: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize calibration logits: {err}"))?;
        visit_f32_bytes_in_chunks(logits_host, vocab, "calibration logits", visitor)?;
        Ok(vocab)
    }

    pub fn report_json(&self, load_ms: f64) -> serde_json::Value {
        match self {
            Self::CpuChunked { chunk_rows } => serde_json::json!({
                "mode": PackageLmHeadMode::CpuChunked.as_str(),
                "chunk_rows": chunk_rows,
                "load_ms": load_ms,
            }),
            Self::GpuResidentAq4 {
                shape,
                vocab,
                hidden,
                matrix,
                top1_partial_count,
                ..
            } => serde_json::json!({
                "mode": PackageLmHeadMode::GpuResidentF32.as_str(),
                "tensor": QWEN3_LM_HEAD_TENSOR,
                "dtype": "AQ4",
                "shape": shape,
                "vocab": vocab,
                "hidden": hidden,
                "matrix_storage_dtype": "AQ4",
                "group_size": matrix.group_size,
                "scale_count": matrix.scale_count,
                "tensor_scale": matrix.tensor_scale,
                "top1_partial_count": top1_partial_count,
                "prewarmed_top1": true,
                "load_ms": load_ms,
            }),
            Self::GpuResidentF32 {
                dtype,
                shape,
                vocab,
                hidden,
                matrix_storage,
                top1_partial_count,
                matrix_bytes,
                ..
            } => serde_json::json!({
                "mode": PackageLmHeadMode::GpuResidentF32.as_str(),
                "tensor": QWEN3_LM_HEAD_TENSOR,
                "dtype": dtype,
                "shape": shape,
                "vocab": vocab,
                "hidden": hidden,
                "matrix_storage_dtype": matrix_storage.as_str(),
                "top1_partial_count": top1_partial_count,
                "matrix_bytes": matrix_bytes,
                "prewarmed_top1": true,
                "load_ms": load_ms,
            }),
        }
    }
}

pub struct PackageFinalNormRuntime {
    hidden: usize,
    weight_buffer: ullm_runtime_sys::RuntimeBuffer,
    output_buffer: ullm_runtime_sys::RuntimeBuffer,
}

enum PackageEmbeddingStorage {
    Bf16 {
        matrix_buffer: ullm_runtime_sys::RuntimeBuffer,
        matrix_bytes: usize,
    },
    Aq4 {
        matrix: PackageAq4ResidentMatvec,
    },
}

pub struct PackageEmbeddingRuntime {
    dtype: String,
    shape: Vec<u64>,
    vocab: usize,
    hidden: usize,
    storage: PackageEmbeddingStorage,
    output_buffer: ullm_runtime_sys::RuntimeBuffer,
}

impl PackageEmbeddingRuntime {
    pub fn load_if_available(
        context: &mut ullm_runtime_sys::RuntimeContext,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        path: &str,
        chunk_bytes: usize,
        hidden: usize,
    ) -> Result<Option<Self>, String> {
        let selector = TensorSelector::Name(QWEN3_EMBED_TOKENS_TENSOR.to_string());
        if select_tensor_payload_bundle(path, &selector).is_ok() {
            let mut registry = WeightRegistry::new();
            let matrix = PackageAq4ResidentMatvec::load(
                context,
                stream,
                &mut registry,
                path,
                QWEN3_EMBED_TOKENS_TENSOR,
                chunk_bytes,
            )
            .map_err(|err| format!("failed to load resident AQ4 embedding tensor: {err}"))?;
            let vocab = matrix.rows;
            let cols = matrix.cols;
            if cols != hidden {
                return Err(format!(
                    "resident AQ4 embedding hidden mismatch: embedding={cols} hidden={hidden}"
                ));
            }
            let shape = vec![
                u64::try_from(vocab)
                    .map_err(|_| "resident AQ4 embedding vocab exceeds u64".to_string())?,
                u64::try_from(hidden)
                    .map_err(|_| "resident AQ4 embedding hidden exceeds u64".to_string())?,
            ];
            let mut output_buffer = context
                .alloc_buffer(checked_f32_byte_len(
                    hidden,
                    "resident AQ4 embedding output",
                )?)
                .map_err(|err| {
                    format!("failed to allocate resident AQ4 embedding output: {err}")
                })?;
            matrix.row_f32(
                0,
                &mut output_buffer,
                stream,
                "resident AQ4 embedding prewarm",
            )?;
            stream.synchronize().map_err(|err| {
                format!("failed to synchronize resident AQ4 embedding load: {err}")
            })?;
            return Ok(Some(Self {
                dtype: "AQ4".to_string(),
                shape,
                vocab,
                hidden,
                storage: PackageEmbeddingStorage::Aq4 { matrix },
                output_buffer,
            }));
        }

        let bundle = select_passthrough_payload_bundle(path, &selector)
            .map_err(|err| format!("failed to select resident embedding tensor: {err}"))?;
        validate_passthrough_shape_elements(&bundle)
            .map_err(|err| format!("invalid resident embedding shape: {err}"))?;
        let dtype = resolve_passthrough_dtype(&bundle, QWEN3_EMBED_TOKENS_TENSOR)?.to_string();
        if dtype != "BF16" {
            return Ok(None);
        }
        if bundle.shape.len() != 2 {
            return Err(format!(
                "resident embedding must be 2D, got shape {:?}",
                bundle.shape
            ));
        }
        let vocab = usize::try_from(bundle.shape[0])
            .map_err(|_| "resident embedding vocab size is too large".to_string())?;
        let cols = usize::try_from(bundle.shape[1])
            .map_err(|_| "resident embedding hidden size is too large".to_string())?;
        if cols != hidden {
            return Err(format!(
                "resident embedding hidden mismatch: embedding={cols} hidden={hidden}"
            ));
        }
        let expected_values = vocab
            .checked_mul(hidden)
            .ok_or_else(|| "resident embedding element count overflows".to_string())?;
        if u64::try_from(expected_values).ok() != Some(bundle.elements) {
            return Err(format!(
                "resident embedding element count mismatch: got {} expected {expected_values}",
                bundle.elements
            ));
        }
        let matrix_bytes = expected_values
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or_else(|| "resident embedding matrix byte size overflows".to_string())?;
        let payload_bytes = if bundle.payload_bytes == 0 {
            bundle.payload_file.bytes
        } else {
            bundle.payload_bytes
        };
        if payload_bytes != bundle.payload_file.bytes {
            return Err(format!(
                "resident embedding payload bytes mismatch: declared {} actual {}",
                payload_bytes, bundle.payload_file.bytes
            ));
        }
        if usize::try_from(payload_bytes).ok() != Some(matrix_bytes) {
            return Err(format!(
                "resident embedding BF16 payload bytes mismatch: got {payload_bytes} expected {matrix_bytes}"
            ));
        }
        match bundle.payload_encoding.as_deref() {
            None | Some("raw_safetensors_payload") => {}
            Some(encoding) => {
                return Err(format!(
                    "resident embedding has unsupported payload encoding {encoding}"
                ));
            }
        }

        let mut matrix_buffer = context
            .alloc_buffer(matrix_bytes)
            .map_err(|err| format!("failed to allocate resident embedding matrix: {err}"))?;
        copy_file_to_runtime_buffer_chunked(
            &mut matrix_buffer,
            &bundle.payload_file.absolute_path,
            matrix_bytes,
            chunk_bytes,
            stream,
            "resident embedding BF16 matrix",
        )?;
        let mut output_buffer = context
            .alloc_buffer(checked_f32_byte_len(hidden, "resident embedding output")?)
            .map_err(|err| format!("failed to allocate resident embedding output: {err}"))?;
        ullm_runtime_sys::bf16_row_f32(
            &matrix_buffer,
            vocab,
            hidden,
            0,
            &mut output_buffer,
            Some(stream),
        )
        .map_err(|err| format!("failed to prewarm resident embedding row gather: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize resident embedding load: {err}"))?;
        Ok(Some(Self {
            dtype,
            shape: bundle.shape,
            vocab,
            hidden,
            storage: PackageEmbeddingStorage::Bf16 {
                matrix_buffer,
                matrix_bytes,
            },
            output_buffer,
        }))
    }

    pub fn gather_token(
        &mut self,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        token_id: usize,
        label: &str,
    ) -> Result<(), String> {
        if token_id >= self.vocab {
            return Err(format!(
                "{label} token id {token_id} is out of resident embedding range 0..{}",
                self.vocab
            ));
        }
        match &self.storage {
            PackageEmbeddingStorage::Bf16 { matrix_buffer, .. } => ullm_runtime_sys::bf16_row_f32(
                matrix_buffer,
                self.vocab,
                self.hidden,
                token_id,
                &mut self.output_buffer,
                Some(stream),
            )
            .map_err(|err| format!("failed to gather {label} resident BF16 embedding row: {err}")),
            PackageEmbeddingStorage::Aq4 { matrix } => matrix.row_f32(
                token_id,
                &mut self.output_buffer,
                stream,
                &format!("{label} resident AQ4 embedding"),
            ),
        }
    }

    pub fn gather_token_to_buffer(
        &mut self,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        token_id: usize,
        output_buffer: &mut ullm_runtime_sys::RuntimeBuffer,
        label: &str,
    ) -> Result<(), String> {
        let required_bytes =
            checked_f32_byte_len(self.hidden, "mixed request-state embedding output")?;
        let actual_bytes = output_buffer.size().map_err(|err| {
            format!(
                "failed to query {label} mixed request-state embedding output buffer size: {err}"
            )
        })?;
        if actual_bytes < required_bytes {
            return Err(format!(
                "{label} mixed request-state embedding output buffer is too small: got {actual_bytes} bytes expected at least {required_bytes}"
            ));
        }
        if token_id >= self.vocab {
            return Err(format!(
                "{label} token id {token_id} is out of resident embedding range 0..{}",
                self.vocab
            ));
        }
        match &self.storage {
            PackageEmbeddingStorage::Bf16 { matrix_buffer, .. } => ullm_runtime_sys::bf16_row_f32(
                matrix_buffer,
                self.vocab,
                self.hidden,
                token_id,
                output_buffer,
                Some(stream),
            )
            .map_err(|err| format!("failed to gather {label} resident BF16 embedding row: {err}")),
            PackageEmbeddingStorage::Aq4 { matrix } => matrix.row_f32(
                token_id,
                output_buffer,
                stream,
                &format!("{label} resident AQ4 embedding"),
            ),
        }
    }

    pub fn gather_token_values(
        &mut self,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        token_id: usize,
        label: &str,
    ) -> Result<Vec<f32>, String> {
        self.gather_token(stream, token_id, label)?;
        read_runtime_buffer_f32(&self.output_buffer, stream, self.hidden, label)
    }

    pub fn output_buffer(&self) -> &ullm_runtime_sys::RuntimeBuffer {
        &self.output_buffer
    }

    pub fn report_json(&self) -> serde_json::Value {
        match &self.storage {
            PackageEmbeddingStorage::Bf16 { matrix_bytes, .. } => serde_json::json!({
                "mode": "gpu_resident_bf16",
                "tensor": QWEN3_EMBED_TOKENS_TENSOR,
                "dtype": self.dtype,
                "shape": self.shape,
                "vocab": self.vocab,
                "hidden": self.hidden,
                "matrix_bytes": matrix_bytes,
            }),
            PackageEmbeddingStorage::Aq4 { matrix } => serde_json::json!({
                "mode": "gpu_resident_aq4",
                "tensor": QWEN3_EMBED_TOKENS_TENSOR,
                "dtype": self.dtype,
                "shape": self.shape,
                "vocab": self.vocab,
                "hidden": self.hidden,
                "group_size": matrix.group_size,
                "scale_count": matrix.scale_count,
                "tensor_scale": matrix.tensor_scale,
            }),
        }
    }
}

pub fn package_embedding_shape(path: &str) -> Result<(usize, usize), String> {
    let selector = TensorSelector::Name(QWEN3_EMBED_TOKENS_TENSOR.to_string());
    if let Ok(bundle) = select_tensor_payload_bundle(path, &selector) {
        let elements = usize::try_from(bundle.elements)
            .map_err(|_| "resident AQ4 embedding element count exceeds usize".to_string())?;
        return matrix_shape_rows_cols(&bundle.shape, elements)
            .map_err(|err| format!("invalid resident AQ4 embedding shape: {err}"));
    }

    let bundle = select_passthrough_payload_bundle(path, &selector)
        .map_err(|err| format!("failed to select resident embedding tensor: {err}"))?;
    validate_passthrough_shape_elements(&bundle)
        .map_err(|err| format!("invalid resident embedding shape: {err}"))?;
    if bundle.shape.len() != 2 {
        return Err(format!(
            "resident embedding must be 2D, got shape {:?}",
            bundle.shape
        ));
    }
    let rows = usize::try_from(bundle.shape[0])
        .map_err(|_| "resident embedding vocab size is too large".to_string())?;
    let cols = usize::try_from(bundle.shape[1])
        .map_err(|_| "resident embedding hidden size is too large".to_string())?;
    Ok((rows, cols))
}

impl PackageFinalNormRuntime {
    pub fn load(
        context: &mut ullm_runtime_sys::RuntimeContext,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        final_norm: &PassthroughF32Data,
        hidden: usize,
    ) -> Result<Self, String> {
        if final_norm.values.len() != hidden {
            return Err(format!(
                "incremental final RMSNorm length mismatch: len={} hidden={hidden}",
                final_norm.values.len()
            ));
        }
        let norm_bytes = checked_f32_byte_len(hidden, "incremental final RMSNorm")?;
        let mut weight_buffer = context
            .alloc_buffer(norm_bytes)
            .map_err(|err| format!("failed to allocate incremental final RMSNorm weight: {err}"))?;
        let output_buffer = context
            .alloc_buffer(norm_bytes)
            .map_err(|err| format!("failed to allocate incremental final RMSNorm output: {err}"))?;
        weight_buffer
            .copy_from_host(0, &encode_f32_to_bytes(&final_norm.values), Some(stream))
            .map_err(|err| format!("failed to copy incremental final RMSNorm weight: {err}"))?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize incremental final RMSNorm runtime setup: {err}")
        })?;
        Ok(Self {
            hidden,
            weight_buffer,
            output_buffer,
        })
    }

    pub fn normalize_device(
        &mut self,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        input_buffer: &ullm_runtime_sys::RuntimeBuffer,
        label: &str,
    ) -> Result<(), String> {
        let input_bytes = input_buffer
            .size()
            .map_err(|err| format!("failed to query {label} final hidden buffer size: {err}"))?;
        let required_bytes = checked_f32_byte_len(self.hidden, "incremental final RMSNorm input")?;
        if input_bytes < required_bytes {
            return Err(format!(
                "{label} final hidden buffer is too small: got {input_bytes} bytes expected at least {required_bytes}"
            ));
        }
        ullm_runtime_sys::rmsnorm_f32(
            input_buffer,
            &self.weight_buffer,
            self.hidden,
            1e-6_f32,
            &mut self.output_buffer,
            Some(stream),
        )
        .map_err(|err| format!("failed to run {label} final RMSNorm: {err}"))
    }

    pub fn output_buffer(&self) -> &ullm_runtime_sys::RuntimeBuffer {
        &self.output_buffer
    }

    /// Visits the current post-final-RMSNorm hidden row without allocating a hidden-sized host
    /// vector. Each device-to-host chunk is synchronized before its borrowed f32 view is exposed.
    pub fn visit_last_output(
        &self,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        visitor: &mut dyn FnMut(usize, &[f32]) -> Result<(), String>,
    ) -> Result<usize, String> {
        let mut bytes =
            [0_u8; QWEN35_AQ4_CALIBRATION_VISIT_CHUNK_ELEMENTS * std::mem::size_of::<f32>()];
        let mut values = [0.0_f32; QWEN35_AQ4_CALIBRATION_VISIT_CHUNK_ELEMENTS];
        let mut start = 0_usize;
        while start < self.hidden {
            let elements = (self.hidden - start).min(QWEN35_AQ4_CALIBRATION_VISIT_CHUNK_ELEMENTS);
            let byte_len = checked_f32_byte_len(elements, "calibration hidden chunk")?;
            let byte_offset = checked_f32_byte_len(start, "calibration hidden offset")?;
            self.output_buffer
                .copy_to_host(byte_offset, &mut bytes[..byte_len], Some(stream))
                .map_err(|err| {
                    format!("failed to copy calibration hidden chunk at {start}: {err}")
                })?;
            stream.synchronize().map_err(|err| {
                format!("failed to synchronize calibration hidden chunk at {start}: {err}")
            })?;
            decode_f32_chunk(&bytes[..byte_len], &mut values[..elements]);
            visitor(start, &values[..elements])?;
            start = start
                .checked_add(elements)
                .ok_or_else(|| "calibration hidden visitor offset overflows".to_string())?;
        }
        Ok(self.hidden)
    }

    pub fn report_json(&self) -> serde_json::Value {
        serde_json::json!({
            "mode": "gpu_resident_f32",
            "hidden": self.hidden,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn package_gpu_resident_lm_head_top_logits(
    stream: &mut ullm_runtime_sys::RuntimeStream,
    matrix_buffer: &ullm_runtime_sys::RuntimeBuffer,
    input_buffer: &ullm_runtime_sys::RuntimeBuffer,
    matrix_storage: PackageLmHeadMatrixStorage,
    vocab: usize,
    hidden: usize,
    logits_buffer: &mut ullm_runtime_sys::RuntimeBuffer,
    top1_partial_values_buffer: &mut ullm_runtime_sys::RuntimeBuffer,
    top1_partial_indices_buffer: &mut ullm_runtime_sys::RuntimeBuffer,
    top1_partial_values_host: &mut [u8],
    top1_partial_indices_host: &mut [u8],
    logits_host: &mut [u8],
    top_k: usize,
) -> Result<Vec<PackageTokenLogit>, String> {
    let required_input_bytes = checked_f32_byte_len(hidden, "resident lm_head input")?;
    let input_bytes = input_buffer
        .size()
        .map_err(|err| format!("failed to query resident lm_head input buffer size: {err}"))?;
    if input_bytes < required_input_bytes {
        return Err(format!(
            "resident lm_head input buffer is too small: got {input_bytes} bytes expected at least {required_input_bytes}"
        ));
    }
    match matrix_storage {
        PackageLmHeadMatrixStorage::F32 => ullm_runtime_sys::matvec_f32(
            matrix_buffer,
            input_buffer,
            vocab,
            hidden,
            logits_buffer,
            Some(stream),
        ),
        PackageLmHeadMatrixStorage::Bf16 => ullm_runtime_sys::matvec_bf16_f32(
            matrix_buffer,
            input_buffer,
            vocab,
            hidden,
            logits_buffer,
            Some(stream),
        ),
    }
    .map_err(|err| format!("resident lm_head matvec failed: {err}"))?;
    package_resident_lm_head_top_logits_from_logits(
        stream,
        logits_buffer,
        vocab,
        top1_partial_values_buffer,
        top1_partial_indices_buffer,
        top1_partial_values_host,
        top1_partial_indices_host,
        logits_host,
        top_k,
    )
}

#[allow(clippy::too_many_arguments)]
fn package_gpu_resident_aq4_lm_head_top_logits(
    stream: &mut ullm_runtime_sys::RuntimeStream,
    matrix: &PackageAq4ResidentMatvec,
    input_buffer: &ullm_runtime_sys::RuntimeBuffer,
    vocab: usize,
    hidden: usize,
    logits_buffer: &mut ullm_runtime_sys::RuntimeBuffer,
    top1_partial_values_buffer: &mut ullm_runtime_sys::RuntimeBuffer,
    top1_partial_indices_buffer: &mut ullm_runtime_sys::RuntimeBuffer,
    top1_partial_values_host: &mut [u8],
    top1_partial_indices_host: &mut [u8],
    logits_host: &mut [u8],
    top_k: usize,
    direct_top1_enabled: bool,
) -> Result<Vec<PackageTokenLogit>, String> {
    if matrix.rows != vocab || matrix.cols != hidden {
        return Err(format!(
            "resident AQ4 lm_head shape mismatch: matrix=[{},{}] expected=[{vocab},{hidden}]",
            matrix.rows, matrix.cols
        ));
    }
    let required_input_bytes = checked_f32_byte_len(hidden, "resident AQ4 lm_head input")?;
    let input_bytes = input_buffer
        .size()
        .map_err(|err| format!("failed to query resident AQ4 lm_head input buffer size: {err}"))?;
    if input_bytes < required_input_bytes {
        return Err(format!(
            "resident AQ4 lm_head input buffer is too small: got {input_bytes} bytes expected at least {required_input_bytes}"
        ));
    }
    if top_k == 1 && direct_top1_enabled {
        let first_stage_partial_count = matrix.matvec_top1(
            input_buffer,
            top1_partial_values_buffer,
            top1_partial_indices_buffer,
            stream,
            "resident AQ4 lm_head",
        )?;
        let partial_count = if first_stage_partial_count > 1 {
            ullm_runtime_sys::top1_pairs_f32_in_place(
                top1_partial_values_buffer,
                top1_partial_indices_buffer,
                first_stage_partial_count,
                Some(stream),
            )
            .map_err(|err| format!("resident AQ4 lm_head direct top1 pair reduce failed: {err}"))?
        } else {
            first_stage_partial_count
        };
        let partial_values_bytes = checked_f32_byte_len(
            partial_count,
            "resident AQ4 lm_head direct top1 partial values",
        )?;
        let partial_indices_bytes = partial_count
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| {
                "resident AQ4 lm_head direct top1 partial index byte size overflows".to_string()
            })?;
        if top1_partial_values_host.len() < partial_values_bytes {
            return Err(format!(
                "resident AQ4 lm_head direct top1 value host buffer is too small: got {} bytes expected at least {partial_values_bytes}",
                top1_partial_values_host.len()
            ));
        }
        if top1_partial_indices_host.len() < partial_indices_bytes {
            return Err(format!(
                "resident AQ4 lm_head direct top1 index host buffer is too small: got {} bytes expected at least {partial_indices_bytes}",
                top1_partial_indices_host.len()
            ));
        }
        top1_partial_values_buffer
            .copy_to_host(
                0,
                &mut top1_partial_values_host[..partial_values_bytes],
                Some(stream),
            )
            .map_err(|err| {
                format!("failed to copy resident AQ4 lm_head direct top1 partial values: {err}")
            })?;
        top1_partial_indices_buffer
            .copy_to_host(
                0,
                &mut top1_partial_indices_host[..partial_indices_bytes],
                Some(stream),
            )
            .map_err(|err| {
                format!("failed to copy resident AQ4 lm_head direct top1 partial indices: {err}")
            })?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize resident AQ4 lm_head direct top1 partials: {err}")
        })?;
        return package_top1_from_partial_bytes(
            &top1_partial_values_host[..partial_values_bytes],
            &top1_partial_indices_host[..partial_indices_bytes],
        );
    }
    matrix.matvec(input_buffer, logits_buffer, stream, "resident AQ4 lm_head")?;
    package_resident_lm_head_top_logits_from_logits(
        stream,
        logits_buffer,
        vocab,
        top1_partial_values_buffer,
        top1_partial_indices_buffer,
        top1_partial_values_host,
        top1_partial_indices_host,
        logits_host,
        top_k,
    )
}

#[allow(clippy::too_many_arguments)]
fn package_resident_lm_head_top_logits_from_logits(
    stream: &mut ullm_runtime_sys::RuntimeStream,
    logits_buffer: &mut ullm_runtime_sys::RuntimeBuffer,
    vocab: usize,
    top1_partial_values_buffer: &mut ullm_runtime_sys::RuntimeBuffer,
    top1_partial_indices_buffer: &mut ullm_runtime_sys::RuntimeBuffer,
    top1_partial_values_host: &mut [u8],
    top1_partial_indices_host: &mut [u8],
    logits_host: &mut [u8],
    top_k: usize,
) -> Result<Vec<PackageTokenLogit>, String> {
    if top_k == 1 {
        let partial_count = ullm_runtime_sys::top1_f32(
            logits_buffer,
            vocab,
            top1_partial_values_buffer,
            top1_partial_indices_buffer,
            Some(stream),
        )
        .map_err(|err| format!("resident lm_head top1 failed: {err}"))?;
        let partial_values_bytes =
            checked_f32_byte_len(partial_count, "resident lm_head top1 partial values")?;
        let partial_indices_bytes = partial_count
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| "resident lm_head top1 partial index byte size overflows".to_string())?;
        if top1_partial_values_host.len() < partial_values_bytes {
            return Err(format!(
                "resident lm_head top1 value host buffer is too small: got {} bytes expected at least {partial_values_bytes}",
                top1_partial_values_host.len()
            ));
        }
        if top1_partial_indices_host.len() < partial_indices_bytes {
            return Err(format!(
                "resident lm_head top1 index host buffer is too small: got {} bytes expected at least {partial_indices_bytes}",
                top1_partial_indices_host.len()
            ));
        }
        top1_partial_values_buffer
            .copy_to_host(
                0,
                &mut top1_partial_values_host[..partial_values_bytes],
                Some(stream),
            )
            .map_err(|err| format!("failed to copy resident lm_head top1 partial values: {err}"))?;
        top1_partial_indices_buffer
            .copy_to_host(
                0,
                &mut top1_partial_indices_host[..partial_indices_bytes],
                Some(stream),
            )
            .map_err(|err| {
                format!("failed to copy resident lm_head top1 partial indices: {err}")
            })?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize resident lm_head top1 partials: {err}")
        })?;
        return package_top1_from_partial_bytes(
            &top1_partial_values_host[..partial_values_bytes],
            &top1_partial_indices_host[..partial_indices_bytes],
        );
    }
    logits_buffer
        .copy_to_host(0, logits_host, Some(stream))
        .map_err(|err| format!("failed to copy resident lm_head logits: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize resident lm_head logits: {err}"))?;
    package_top_logits_from_f32_bytes(logits_host, top_k)
}

pub fn copy_f32_values_to_runtime_buffer_chunked(
    buffer: &mut ullm_runtime_sys::RuntimeBuffer,
    values: &[f32],
    stream: &mut ullm_runtime_sys::RuntimeStream,
    label: &str,
) -> Result<(), String> {
    const COPY_CHUNK_F32: usize = 1 << 20;
    for (chunk_index, chunk) in values.chunks(COPY_CHUNK_F32).enumerate() {
        let offset_elements = chunk_index
            .checked_mul(COPY_CHUNK_F32)
            .ok_or_else(|| format!("{label} copy offset overflows"))?;
        let offset_bytes = offset_elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| format!("{label} copy byte offset overflows"))?;
        let bytes = encode_f32_to_bytes(chunk);
        buffer
            .copy_from_host(offset_bytes, &bytes, Some(stream))
            .map_err(|err| format!("failed to copy {label} chunk {chunk_index}: {err}"))?;
    }
    Ok(())
}

fn copy_file_to_runtime_buffer_chunked(
    buffer: &mut ullm_runtime_sys::RuntimeBuffer,
    path: &std::path::Path,
    bytes: usize,
    chunk_bytes: usize,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    label: &str,
) -> Result<(), String> {
    if chunk_bytes == 0 {
        return Err(format!("{label} chunk bytes must be greater than zero"));
    }
    let mut file = File::open(path)
        .map_err(|err| format!("failed to open {label} {}: {err}", path.display()))?;
    let mut chunk = vec![0_u8; chunk_bytes.min(bytes.max(1))];
    let mut offset = 0_usize;
    while offset < bytes {
        let remaining = bytes - offset;
        let read_len = remaining.min(chunk.len());
        file.read_exact(&mut chunk[..read_len])
            .map_err(|err| format!("failed to read {label} at byte offset {offset}: {err}"))?;
        buffer
            .copy_from_host(offset, &chunk[..read_len], Some(stream))
            .map_err(|err| {
                format!("failed to copy {label} chunk at byte offset {offset}: {err}")
            })?;
        offset = offset
            .checked_add(read_len)
            .ok_or_else(|| format!("{label} copy offset overflows"))?;
    }
    Ok(())
}

fn package_logit_precedes(left: &PackageTokenLogit, right: &PackageTokenLogit) -> bool {
    left.logit
        .total_cmp(&right.logit)
        .reverse()
        .then_with(|| left.token_id.cmp(&right.token_id))
        .is_lt()
}

fn push_package_top_logit(
    top_logits: &mut Vec<PackageTokenLogit>,
    top_k: usize,
    candidate: PackageTokenLogit,
) {
    if top_logits.len() < top_k {
        top_logits.push(candidate);
        top_logits.sort_by(|left, right| {
            right
                .logit
                .total_cmp(&left.logit)
                .then_with(|| left.token_id.cmp(&right.token_id))
        });
        return;
    }
    if let Some(last) = top_logits.last() {
        if !package_logit_precedes(&candidate, last) {
            return;
        }
    }
    if let Some(last) = top_logits.last_mut() {
        *last = candidate;
    }
    top_logits.sort_by(|left, right| {
        right
            .logit
            .total_cmp(&left.logit)
            .then_with(|| left.token_id.cmp(&right.token_id))
    });
}

fn package_top_logits_from_f32_bytes(
    logits_bytes: &[u8],
    top_k: usize,
) -> Result<Vec<PackageTokenLogit>, String> {
    if top_k == 0 {
        return Err("top k must be greater than zero".to_string());
    }
    if !logits_bytes
        .len()
        .is_multiple_of(std::mem::size_of::<f32>())
    {
        return Err("resident lm_head logits byte length is not f32-aligned".to_string());
    }
    let mut top_logits = Vec::with_capacity(top_k);
    for (token_id, chunk) in logits_bytes
        .chunks_exact(std::mem::size_of::<f32>())
        .enumerate()
    {
        let logit = f32::from_le_bytes(chunk.try_into().expect("f32 chunk"));
        if !logit.is_finite() {
            return Err(format!("lm_head logit for token {token_id} is not finite"));
        }
        push_package_top_logit(
            &mut top_logits,
            top_k,
            PackageTokenLogit { token_id, logit },
        );
    }
    Ok(top_logits)
}

fn decode_f32_chunk(bytes: &[u8], values: &mut [f32]) {
    debug_assert_eq!(bytes.len(), values.len() * std::mem::size_of::<f32>());
    for (value, chunk) in values
        .iter_mut()
        .zip(bytes.chunks_exact(std::mem::size_of::<f32>()))
    {
        *value = f32::from_le_bytes(chunk.try_into().expect("f32 chunk"));
    }
}

fn visit_f32_bytes_in_chunks(
    bytes: &[u8],
    expected_elements: usize,
    label: &str,
    visitor: &mut dyn FnMut(usize, &[f32]) -> Result<(), String>,
) -> Result<(), String> {
    let expected_bytes = checked_f32_byte_len(expected_elements, label)?;
    if bytes.len() != expected_bytes {
        return Err(format!(
            "{label} byte length differs: got {} expected {expected_bytes}",
            bytes.len()
        ));
    }
    let mut values = [0.0_f32; QWEN35_AQ4_CALIBRATION_VISIT_CHUNK_ELEMENTS];
    for (chunk_index, chunk) in bytes
        .chunks(QWEN35_AQ4_CALIBRATION_VISIT_CHUNK_ELEMENTS * std::mem::size_of::<f32>())
        .enumerate()
    {
        let elements = chunk.len() / std::mem::size_of::<f32>();
        decode_f32_chunk(chunk, &mut values[..elements]);
        let start = chunk_index
            .checked_mul(QWEN35_AQ4_CALIBRATION_VISIT_CHUNK_ELEMENTS)
            .ok_or_else(|| format!("{label} visitor offset overflows"))?;
        visitor(start, &values[..elements])?;
    }
    Ok(())
}

fn package_top1_from_partial_bytes(
    partial_value_bytes: &[u8],
    partial_index_bytes: &[u8],
) -> Result<Vec<PackageTokenLogit>, String> {
    if !partial_value_bytes
        .len()
        .is_multiple_of(std::mem::size_of::<f32>())
    {
        return Err("resident lm_head top1 partial value byte length is not f32-aligned".into());
    }
    if !partial_index_bytes
        .len()
        .is_multiple_of(std::mem::size_of::<u32>())
    {
        return Err("resident lm_head top1 partial index byte length is not u32-aligned".into());
    }
    let partial_count = partial_value_bytes.len() / std::mem::size_of::<f32>();
    if partial_count == 0 || partial_index_bytes.len() / std::mem::size_of::<u32>() != partial_count
    {
        return Err("resident lm_head top1 partial value/index length mismatch".into());
    }
    let mut best: Option<PackageTokenLogit> = None;
    for partial in 0..partial_count {
        let value_offset = partial * std::mem::size_of::<f32>();
        let index_offset = partial * std::mem::size_of::<u32>();
        let logit = f32::from_le_bytes(
            partial_value_bytes[value_offset..value_offset + std::mem::size_of::<f32>()]
                .try_into()
                .expect("f32 partial"),
        );
        let token_id_u32 = u32::from_le_bytes(
            partial_index_bytes[index_offset..index_offset + std::mem::size_of::<u32>()]
                .try_into()
                .expect("u32 partial"),
        );
        if !logit.is_finite() {
            return Err(format!(
                "resident lm_head top1 partial {partial} is not finite"
            ));
        }
        let token_id = usize::try_from(token_id_u32)
            .map_err(|_| format!("resident lm_head top1 token id {token_id_u32} is too large"))?;
        let candidate = PackageTokenLogit { token_id, logit };
        if best
            .as_ref()
            .map(|current| package_logit_precedes(&candidate, current))
            .unwrap_or(true)
        {
            best = Some(candidate);
        }
    }
    best.map(|entry| vec![entry])
        .ok_or_else(|| "resident lm_head top1 produced no partials".into())
}

pub fn package_lm_head_top_k_from_rows(
    path: &str,
    hidden: &[f32],
    top_k: usize,
    chunk_rows: usize,
) -> Result<(usize, String, Vec<u64>, Vec<PackageTokenLogit>), String> {
    if hidden.is_empty() {
        return Err("lm_head top-k hidden vector must not be empty".to_string());
    }
    if top_k == 0 || chunk_rows == 0 {
        return Err("lm_head top-k and chunk_rows must be greater than zero".to_string());
    }
    let first_row = read_named_passthrough_f32_row_range(path, QWEN3_LM_HEAD_TENSOR, 0, 1)
        .map_err(|err| format!("failed to read lm_head first row: {err}"))?;
    if first_row.shape.len() != 2 {
        return Err(format!(
            "lm_head must be 2D, got shape {:?}",
            first_row.shape
        ));
    }
    let vocab = usize::try_from(first_row.shape[0])
        .map_err(|_| "lm_head vocab size is too large for this host".to_string())?;
    if first_row.columns != hidden.len() {
        return Err(format!(
            "lm_head hidden size mismatch: columns={} hidden={}",
            first_row.columns,
            hidden.len()
        ));
    }

    let mut top_logits = Vec::new();
    let mut start = 0_usize;
    while start < vocab {
        let end = start
            .checked_add(chunk_rows)
            .map(|candidate| candidate.min(vocab))
            .ok_or_else(|| "lm_head chunk end overflows".to_string())?;
        let rows =
            read_named_passthrough_f32_row_range(path, QWEN3_LM_HEAD_TENSOR, start, end - start)
                .map_err(|err| format!("failed to read lm_head rows {start}..{end}: {err}"))?;
        if rows.columns != hidden.len() || rows.shape != first_row.shape {
            return Err(format!(
                "lm_head chunk shape changed for rows {start}..{end}: columns={} shape={:?}",
                rows.columns, rows.shape
            ));
        }
        for (offset, row) in rows.values.chunks_exact(hidden.len()).enumerate() {
            let mut logit = 0.0_f32;
            for (weight, value) in row.iter().zip(hidden.iter()) {
                logit += weight * value;
            }
            if !logit.is_finite() {
                return Err(format!(
                    "lm_head logit for token {} is not finite",
                    start + offset
                ));
            }
            top_logits.push(PackageTokenLogit {
                token_id: start + offset,
                logit,
            });
        }
        top_logits.sort_by(|left, right| {
            right
                .logit
                .total_cmp(&left.logit)
                .then_with(|| left.token_id.cmp(&right.token_id))
        });
        top_logits.truncate(top_k);
        start = end;
    }

    Ok((vocab, first_row.dtype, first_row.shape, top_logits))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lm_head_mode_labels_are_stable() {
        assert_eq!(PackageLmHeadMode::CpuChunked.as_str(), "cpu_chunked");
        assert_eq!(
            PackageLmHeadMode::GpuResidentF32.as_str(),
            "gpu_resident_f32"
        );
    }

    #[test]
    fn direct_top1_load_flag_has_a_closed_default_and_explicit_true_values() {
        for value in ["1", "true", "TRUE", "yes", "YES"] {
            assert!(flag_value_enabled(value));
        }
        for value in ["", "0", "false", "True", "on"] {
            assert!(!flag_value_enabled(value));
        }
    }

    #[test]
    fn calibration_byte_visitor_is_chunked_and_token_ordered() {
        let element_count = QWEN35_AQ4_CALIBRATION_VISIT_CHUNK_ELEMENTS * 2 + 7;
        let bytes = (0..element_count)
            .flat_map(|value| (value as f32).to_le_bytes())
            .collect::<Vec<_>>();
        let mut starts = Vec::new();
        let mut observed = Vec::new();
        visit_f32_bytes_in_chunks(
            &bytes,
            element_count,
            "test calibration logits",
            &mut |start, values| {
                starts.push((start, values.len()));
                observed.extend_from_slice(values);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(
            starts,
            vec![
                (0, QWEN35_AQ4_CALIBRATION_VISIT_CHUNK_ELEMENTS),
                (
                    QWEN35_AQ4_CALIBRATION_VISIT_CHUNK_ELEMENTS,
                    QWEN35_AQ4_CALIBRATION_VISIT_CHUNK_ELEMENTS
                ),
                (QWEN35_AQ4_CALIBRATION_VISIT_CHUNK_ELEMENTS * 2, 7),
            ]
        );
        assert_eq!(observed.len(), element_count);
        assert!(
            observed
                .iter()
                .enumerate()
                .all(|(index, value)| *value == index as f32)
        );
    }

    #[test]
    fn calibration_byte_visitor_rejects_length_and_stops_on_callback_failure() {
        let bytes = [1.0_f32, 2.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        assert!(visit_f32_bytes_in_chunks(&bytes, 3, "test", &mut |_, _| Ok(())).is_err());
        let error =
            visit_f32_bytes_in_chunks(&bytes, 2, "test", &mut |_, _| Err("observer closed".into()))
                .unwrap_err();
        assert_eq!(error, "observer closed");
    }

    #[test]
    fn top_logits_are_descending_and_tie_break_on_token_id() {
        let bytes = [1.0_f32, 3.0, 3.0, -2.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let logits = package_top_logits_from_f32_bytes(&bytes, 3).unwrap();
        assert_eq!(
            logits
                .iter()
                .map(|entry| entry.token_id)
                .collect::<Vec<_>>(),
            vec![1, 2, 0]
        );
    }

    #[test]
    fn top_logits_reject_zero_top_k() {
        assert_eq!(
            package_top_logits_from_f32_bytes(&0.0_f32.to_le_bytes(), 0).unwrap_err(),
            "top k must be greater than zero"
        );
    }
}
