// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

use crate::host_bytes::encode_f32_to_bytes;
use crate::sq::{SqFp8Artifact, read_named_sq_fp8_tensor_compact_bytes};
use ullm_runtime_sys::{RuntimeBuffer, RuntimeContext, RuntimeStream};

pub const SQ8_SCALE_TENSOR_KIND: u32 = ullm_runtime_sys::SQ_FP8_SCALE_TENSOR;
pub const SQ8_SCALE_ROW_KIND: u32 = ullm_runtime_sys::SQ_FP8_SCALE_ROW;
pub const SQ8_SCALE_ROW_BLOCK_KIND: u32 = ullm_runtime_sys::SQ_FP8_SCALE_ROW_BLOCK;

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
