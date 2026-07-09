// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

use crate::host_bytes::{decode_f32_le_values, encode_f32_to_bytes};
use crate::sq::{SqFp8Artifact, read_named_sq_fp8_tensor_compact_bytes};
use crate::sq_canonical::{SQ8_CANONICAL_MAX_VERIFY_CHUNK_BYTES, Sq8CanonicalArtifact};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use ullm_runtime_sys::{
    RuntimeBuffer, RuntimeContext, RuntimeStream, SqFp8ExecutionPath, sq_fp8_matvec_block2d_f32,
};

pub const SQ8_SCALE_TENSOR_KIND: u32 = ullm_runtime_sys::SQ_FP8_SCALE_TENSOR;
pub const SQ8_SCALE_ROW_KIND: u32 = ullm_runtime_sys::SQ_FP8_SCALE_ROW;
pub const SQ8_SCALE_ROW_BLOCK_KIND: u32 = ullm_runtime_sys::SQ_FP8_SCALE_ROW_BLOCK;
pub const SQ8_CANONICAL_UPLOAD_CHUNK_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug)]
pub struct Sq8ResidentRuntimeTensor {
    pub tensor_index: usize,
    pub tensor_name: String,
    pub rows: usize,
    pub cols: usize,
    pub scale_count: usize,
    pub payload_buffer: RuntimeBuffer,
    pub scale_buffer: RuntimeBuffer,
    pub scale_kind: u32,
    pub scale_block_cols: usize,
}

pub struct Sq8ResidentRuntimeTensorRef<'a> {
    pub payload_buffer: &'a RuntimeBuffer,
    pub scale_buffer: &'a RuntimeBuffer,
    pub scale_kind: u32,
    pub scale_block_cols: usize,
}

#[derive(Debug)]
pub struct Sq8CanonicalResidentRuntimeTensor {
    pub tensor_name: String,
    pub rows: usize,
    pub cols: usize,
    pub scale_rows: usize,
    pub scale_cols: usize,
    pub block_rows: usize,
    pub block_cols: usize,
    pub payload_buffer: RuntimeBuffer,
    pub scale_buffer: RuntimeBuffer,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sq8CanonicalRuntimeProjection {
    pub execution_path: SqFp8ExecutionPath,
    pub output: Vec<f32>,
}

impl Sq8ResidentRuntimeTensor {
    pub fn storage_ref(&self) -> Sq8ResidentRuntimeTensorRef<'_> {
        Sq8ResidentRuntimeTensorRef {
            payload_buffer: &self.payload_buffer,
            scale_buffer: &self.scale_buffer,
            scale_kind: self.scale_kind,
            scale_block_cols: self.scale_block_cols,
        }
    }
}

pub fn sq8_scale_kind(
    scale_granularity: &str,
    scale_block_cols: usize,
    label: &str,
) -> Result<u32, String> {
    match scale_granularity {
        "tensor" => Ok(SQ8_SCALE_TENSOR_KIND),
        "row" => Ok(SQ8_SCALE_ROW_KIND),
        "row_block" => {
            if scale_block_cols == 0 {
                return Err(format!(
                    "{label} SQ FP8 row_block scale_block_cols must be greater than zero"
                ));
            }
            Ok(SQ8_SCALE_ROW_BLOCK_KIND)
        }
        other => Err(format!(
            "{label} SQ FP8 scale_granularity must be tensor|row|row_block, got {other}"
        )),
    }
}

pub fn load_sq8_resident_tensor(
    context: &mut RuntimeContext,
    stream: &mut RuntimeStream,
    artifact: &SqFp8Artifact,
    tensor_name: &str,
) -> Result<Option<Sq8ResidentRuntimeTensor>, String> {
    let compact = match read_named_sq_fp8_tensor_compact_bytes(artifact, tensor_name)? {
        Some(compact) => compact,
        None => return Ok(None),
    };

    let scale_kind = sq8_scale_kind(
        &compact.scale_granularity,
        compact.scale_block_cols,
        tensor_name,
    )?;

    let mut payload_buffer = context
        .alloc_buffer(compact.payload.len())
        .map_err(|err| format!("failed to allocate SQ FP8 payload for {tensor_name}: {err}"))?;
    payload_buffer
        .copy_from_host(0, &compact.payload, Some(stream))
        .map_err(|err| format!("failed to copy SQ FP8 payload for {tensor_name}: {err}"))?;

    let scale_bytes = encode_f32_to_bytes(&compact.scale_values);
    let mut scale_buffer = context
        .alloc_buffer(scale_bytes.len())
        .map_err(|err| format!("failed to allocate SQ FP8 scales for {tensor_name}: {err}"))?;
    scale_buffer
        .copy_from_host(0, &scale_bytes, Some(stream))
        .map_err(|err| format!("failed to copy SQ FP8 scales for {tensor_name}: {err}"))?;

    Ok(Some(Sq8ResidentRuntimeTensor {
        tensor_index: compact.tensor_index,
        tensor_name: compact.tensor_name,
        rows: compact.rows,
        cols: compact.cols,
        scale_count: compact.scale_values.len(),
        payload_buffer,
        scale_buffer,
        scale_kind,
        scale_block_cols: compact.scale_block_cols,
    }))
}

pub fn load_sq8_canonical_resident_tensor(
    context: &mut RuntimeContext,
    stream: &mut RuntimeStream,
    artifact: &Sq8CanonicalArtifact,
    tensor_name: &str,
    upload_chunk_bytes: usize,
) -> Result<Sq8CanonicalResidentRuntimeTensor, String> {
    if upload_chunk_bytes == 0 {
        return Err("SQ8 canonical upload chunk size must be greater than zero".to_string());
    }
    let upload_chunk_bytes = upload_chunk_bytes.min(SQ8_CANONICAL_MAX_VERIFY_CHUNK_BYTES);
    let pair = artifact.tensor_pair(tensor_name)?;
    artifact.verify_tensor_payloads(tensor_name, upload_chunk_bytes)?;
    let paths = artifact.tensor_payload_paths(tensor_name)?;

    let rows = usize::try_from(pair.shape[0])
        .map_err(|_| format!("SQ8 canonical tensor {} rows do not fit usize", pair.name))?;
    let cols = usize::try_from(pair.shape[1])
        .map_err(|_| format!("SQ8 canonical tensor {} cols do not fit usize", pair.name))?;
    let scale_rows = usize::try_from(pair.scale.shape[0]).map_err(|_| {
        format!(
            "SQ8 canonical tensor {} scale rows do not fit usize",
            pair.name
        )
    })?;
    let scale_cols = usize::try_from(pair.scale.shape[1]).map_err(|_| {
        format!(
            "SQ8 canonical tensor {} scale cols do not fit usize",
            pair.name
        )
    })?;
    let block_rows = usize::try_from(pair.scale.block_shape[0]).map_err(|_| {
        format!(
            "SQ8 canonical tensor {} block rows do not fit usize",
            pair.name
        )
    })?;
    let block_cols = usize::try_from(pair.scale.block_shape[1]).map_err(|_| {
        format!(
            "SQ8 canonical tensor {} block cols do not fit usize",
            pair.name
        )
    })?;
    let payload_bytes = usize::try_from(pair.weight.bytes).map_err(|_| {
        format!(
            "SQ8 canonical tensor {} weight bytes do not fit usize",
            pair.name
        )
    })?;

    let mut payload_buffer = context.alloc_buffer(payload_bytes).map_err(|err| {
        format!(
            "failed to allocate SQ8 canonical payload for {}: {err}",
            pair.name
        )
    })?;
    upload_verified_payload(
        &paths.weight,
        pair.weight.bytes,
        &pair.weight.sha256,
        upload_chunk_bytes,
        &mut payload_buffer,
        stream,
        &format!("{} weight", pair.name),
    )?;

    let scale_values = artifact.read_tensor_scales_f32(tensor_name, upload_chunk_bytes)?;
    let expected_scale_values = scale_rows
        .checked_mul(scale_cols)
        .ok_or_else(|| format!("SQ8 canonical tensor {} scale shape overflows", pair.name))?;
    if scale_values.len() != expected_scale_values {
        return Err(format!(
            "SQ8 canonical tensor {} scale count mismatch: expected={expected_scale_values} actual={}",
            pair.name,
            scale_values.len()
        ));
    }
    let scale_bytes = encode_f32_to_bytes(&scale_values);
    let mut scale_buffer = context.alloc_buffer(scale_bytes.len()).map_err(|err| {
        format!(
            "failed to allocate SQ8 canonical scales for {}: {err}",
            pair.name
        )
    })?;
    scale_buffer
        .copy_from_host(0, &scale_bytes, Some(stream))
        .map_err(|err| {
            format!(
                "failed to copy SQ8 canonical scales for {}: {err}",
                pair.name
            )
        })?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize SQ8 canonical scale upload: {err}"))?;

    Ok(Sq8CanonicalResidentRuntimeTensor {
        tensor_name: pair.name.clone(),
        rows,
        cols,
        scale_rows,
        scale_cols,
        block_rows,
        block_cols,
        payload_buffer,
        scale_buffer,
    })
}

pub fn run_sq8_canonical_resident_projection_f32(
    context: &mut RuntimeContext,
    stream: &mut RuntimeStream,
    resident: &Sq8CanonicalResidentRuntimeTensor,
    input: &[f32],
) -> Result<Sq8CanonicalRuntimeProjection, String> {
    if input.len() != resident.cols {
        return Err(format!(
            "SQ8 canonical runtime tensor {} input length mismatch: expected={} actual={}",
            resident.tensor_name,
            resident.cols,
            input.len()
        ));
    }
    if let Some((index, value)) = input
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!(
            "SQ8 canonical runtime tensor {} input contains non-finite value {value} at index {index}",
            resident.tensor_name
        ));
    }

    let input_bytes = encode_f32_to_bytes(input);
    let output_bytes_len = resident
        .rows
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            format!(
                "SQ8 canonical runtime tensor {} output byte size overflows",
                resident.tensor_name
            )
        })?;
    let mut input_buffer = context.alloc_buffer(input_bytes.len()).map_err(|err| {
        format!(
            "failed to allocate SQ8 canonical input for {}: {err}",
            resident.tensor_name
        )
    })?;
    let mut output_buffer = context.alloc_buffer(output_bytes_len).map_err(|err| {
        format!(
            "failed to allocate SQ8 canonical output for {}: {err}",
            resident.tensor_name
        )
    })?;
    input_buffer
        .copy_from_host(0, &input_bytes, Some(stream))
        .map_err(|err| {
            format!(
                "failed to copy SQ8 canonical input for {}: {err}",
                resident.tensor_name
            )
        })?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize SQ8 canonical input upload: {err}"))?;

    let execution_path = sq_fp8_matvec_block2d_f32(
        &resident.payload_buffer,
        &resident.scale_buffer,
        &input_buffer,
        resident.rows,
        resident.cols,
        resident.block_rows,
        resident.block_cols,
        &mut output_buffer,
        Some(stream),
    )?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize SQ8 canonical projection: {err}"))?;

    let mut output_bytes = vec![0_u8; output_bytes_len];
    output_buffer
        .copy_to_host(0, &mut output_bytes, Some(stream))
        .map_err(|err| {
            format!(
                "failed to read SQ8 canonical output for {}: {err}",
                resident.tensor_name
            )
        })?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize SQ8 canonical output readback: {err}"))?;
    let output = decode_f32_le_values(&output_bytes);
    if let Some((index, value)) = output
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!(
            "SQ8 canonical runtime tensor {} output contains non-finite value {value} at index {index}",
            resident.tensor_name
        ));
    }
    Ok(Sq8CanonicalRuntimeProjection {
        execution_path,
        output,
    })
}

#[allow(clippy::too_many_arguments)]
fn upload_verified_payload(
    path: &Path,
    expected_bytes: u64,
    expected_sha256: &str,
    chunk_bytes: usize,
    destination: &mut RuntimeBuffer,
    stream: &mut RuntimeStream,
    label: &str,
) -> Result<(), String> {
    let mut file = File::open(path)
        .map_err(|err| format!("failed to open SQ8 canonical {label} for upload: {err}"))?;
    let opened_bytes = file
        .metadata()
        .map_err(|err| format!("failed to stat opened SQ8 canonical {label}: {err}"))?
        .len();
    if opened_bytes != expected_bytes {
        return Err(format!(
            "SQ8 canonical {label} byte length mismatch before upload: manifest={expected_bytes} file={opened_bytes}"
        ));
    }
    let buffer_len = usize::try_from(expected_bytes.min(chunk_bytes as u64))
        .map_err(|_| format!("SQ8 canonical {label} upload chunk does not fit usize"))?
        .max(1);
    let mut buffer = vec![0_u8; buffer_len];
    let mut digest = Sha256::new();
    let mut remaining = expected_bytes;
    let mut offset = 0_u64;
    while remaining > 0 {
        let read_len = usize::try_from(remaining.min(buffer.len() as u64))
            .map_err(|_| format!("SQ8 canonical {label} upload read length does not fit usize"))?;
        file.read_exact(&mut buffer[..read_len])
            .map_err(|err| format!("failed to read SQ8 canonical {label} at {offset}: {err}"))?;
        digest.update(&buffer[..read_len]);
        let destination_offset = usize::try_from(offset)
            .map_err(|_| format!("SQ8 canonical {label} upload offset does not fit usize"))?;
        destination
            .copy_from_host(destination_offset, &buffer[..read_len], Some(stream))
            .map_err(|err| format!("failed to upload SQ8 canonical {label} at {offset}: {err}"))?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize SQ8 canonical {label} upload at {offset}: {err}")
        })?;
        remaining -= read_len as u64;
        offset += read_len as u64;
    }
    let mut trailing = [0_u8; 1];
    if file
        .read(&mut trailing)
        .map_err(|err| format!("failed to verify SQ8 canonical {label} upload EOF: {err}"))?
        != 0
    {
        return Err(format!(
            "SQ8 canonical {label} has trailing data after declared {expected_bytes} bytes"
        ));
    }
    let final_bytes = file
        .metadata()
        .map_err(|err| format!("failed to re-stat opened SQ8 canonical {label}: {err}"))?
        .len();
    if final_bytes != opened_bytes || final_bytes != expected_bytes {
        return Err(format!(
            "SQ8 canonical {label} byte length changed during upload: manifest={expected_bytes} before={opened_bytes} after={final_bytes}"
        ));
    }
    let actual_sha256 = format!("{:x}", digest.finalize());
    if actual_sha256 != expected_sha256 {
        return Err(format!(
            "SQ8 canonical {label} checksum mismatch during upload: manifest={expected_sha256} file={actual_sha256}"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::sq8_scale_kind;
    use super::{SQ8_SCALE_ROW_BLOCK_KIND, SQ8_SCALE_ROW_KIND, SQ8_SCALE_TENSOR_KIND};

    #[test]
    fn maps_sq8_scale_granularity_to_runtime_constants() {
        assert_eq!(
            sq8_scale_kind("tensor", 0, "tensor"),
            Ok(SQ8_SCALE_TENSOR_KIND)
        );
        assert_eq!(sq8_scale_kind("row", 0, "row"), Ok(SQ8_SCALE_ROW_KIND));
        assert_eq!(
            sq8_scale_kind("row_block", 16, "row_block"),
            Ok(SQ8_SCALE_ROW_BLOCK_KIND)
        );
    }

    #[test]
    fn rejects_unknown_sq8_scale_granularity_with_stable_error() {
        let err = sq8_scale_kind("channel", 0, "tensor.weight").unwrap_err();
        assert_eq!(
            err,
            "tensor.weight SQ FP8 scale_granularity must be tensor|row|row_block, got channel"
                .to_string()
        );
    }

    #[test]
    fn rejects_zero_row_block_cols_for_row_block_scale() {
        let err = sq8_scale_kind("row_block", 0, "tensor.weight").unwrap_err();
        assert_eq!(
            err,
            "tensor.weight SQ FP8 row_block scale_block_cols must be greater than zero".to_string()
        );
    }
}
