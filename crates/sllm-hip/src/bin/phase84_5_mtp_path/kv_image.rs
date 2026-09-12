//! State-image based KV evidence for the Phase84.5 diagnostic.
//!
//! The public request image already exposes all four standard MXFP8 E4
//! planes.  This helper keeps the diagnostic independent of the private
//! FP16-only readback ABI: each returned K/V payload is the published prefix
//! of its value plane followed by the corresponding E8M0 scale plane.

use sllm_core::{
    ExecutionStateImageV1, KvCacheEncoding, QwenExecutionRequest, QwenKvPayloadEvidence,
    StatePlaneKindV1,
};

/// Exports the published MXFP8 E4 KV prefix as `(layer, key, value)` payloads.
///
/// Native state-image plane sizes may describe the physical capacity, while
/// some adapters can already return only the published prefix.  The helper
/// accepts either representation and trims to the exact published token
/// count after validating the encoding, topology, and per-token geometry.
pub(super) fn kv_payload(
    request: &QwenExecutionRequest,
) -> Result<Vec<QwenKvPayloadEvidence>, String> {
    let state = request
        .export_state_image()
        .map_err(|error| format!("KV state-image export failed: {error}"))?;
    let published_length = state.committed_length();
    if published_length == 0 {
        return Err("KV state-image export returned an empty request".to_owned());
    }

    if state.kv_layers().is_empty() {
        return Err("KV state-image has no KV layers".to_owned());
    }
    let mut payloads = Vec::with_capacity(state.kv_layers().len());
    for (&layer, layer_image) in state.kv_layers() {
        let descriptor = layer_image.descriptor();
        if descriptor.sliding_window().is_some() {
            return Err("diagnostic KV prefix requires non-sliding storage".to_owned());
        }
        if descriptor.cache_encoding() != KvCacheEncoding::Mxfp8E4 {
            return Err(format!(
                "layer {layer} has unsupported KV encoding {}",
                descriptor.cache_encoding().canonical_name()
            ));
        }
        let metadata = layer_image.image().metadata();
        if metadata.published_length != published_length || metadata.active_slot.is_some() {
            return Err(format!(
                "layer {layer} has inconsistent KV image publication metadata"
            ));
        }

        let mxfp8 = descriptor
            .kv_mxfp8_descriptor()
            .ok_or_else(|| format!("layer {layer} is missing its MXFP8 descriptor"))?;
        let heads = descriptor.layout().heads();
        let head_dim = descriptor.layout().head_dim();
        let blocks_per_head = mxfp8.blocks_per_head(head_dim);
        let value_bytes_per_token = heads
            .checked_mul(mxfp8.padded_head_dim(head_dim))
            .ok_or_else(|| format!("layer {layer} value stride overflowed"))?;
        let scale_bytes_per_token = heads
            .checked_mul(blocks_per_head)
            .ok_or_else(|| format!("layer {layer} scale stride overflowed"))?;
        let value_prefix = prefix_len(published_length, value_bytes_per_token, layer, "value")?;
        let scale_prefix = prefix_len(published_length, scale_bytes_per_token, layer, "scale")?;

        let image = layer_image.image();
        let key_values = plane(image, StatePlaneKindV1::KvKey, layer)?;
        let value_values = plane(image, StatePlaneKindV1::KvValue, layer)?;
        let key_scales = plane(image, StatePlaneKindV1::KvKeyScale, layer)?;
        let value_scales = plane(image, StatePlaneKindV1::KvValueScale, layer)?;
        let key = concat_prefix(
            key_values,
            value_prefix,
            key_scales,
            scale_prefix,
            layer,
            "key",
        )?;
        let value = concat_prefix(
            value_values,
            value_prefix,
            value_scales,
            scale_prefix,
            layer,
            "value",
        )?;
        payloads.push((layer, key, value));
    }
    Ok(payloads)
}

fn prefix_len(
    published_length: u64,
    bytes_per_token: usize,
    layer: u32,
    plane_name: &str,
) -> Result<usize, String> {
    let published_length = usize::try_from(published_length)
        .map_err(|_| format!("layer {layer} published length exceeds host usize"))?;
    published_length
        .checked_mul(bytes_per_token)
        .ok_or_else(|| format!("layer {layer} {plane_name} prefix length overflowed"))
}

fn plane(
    image: &ExecutionStateImageV1,
    kind: StatePlaneKindV1,
    layer: u32,
) -> Result<&[u8], String> {
    image
        .planes()
        .iter()
        .find(|plane| plane.plane == kind)
        .map(|plane| plane.bytes.as_slice())
        .ok_or_else(|| format!("layer {layer} state image omitted {kind:?}"))
}

fn concat_prefix(
    values: &[u8],
    value_prefix: usize,
    scales: &[u8],
    scale_prefix: usize,
    layer: u32,
    semantic: &str,
) -> Result<Vec<u8>, String> {
    if values.len() < value_prefix {
        return Err(format!(
            "layer {layer} {semantic} value plane is shorter than published prefix: {} < {}",
            values.len(),
            value_prefix
        ));
    }
    if scales.len() < scale_prefix {
        return Err(format!(
            "layer {layer} {semantic} scale plane is shorter than published prefix: {} < {}",
            scales.len(),
            scale_prefix
        ));
    }
    let mut payload = Vec::with_capacity(value_prefix.saturating_add(scale_prefix));
    payload.extend_from_slice(&values[..value_prefix]);
    payload.extend_from_slice(&scales[..scale_prefix]);
    Ok(payload)
}
