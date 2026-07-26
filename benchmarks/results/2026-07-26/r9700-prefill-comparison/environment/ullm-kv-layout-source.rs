// Extracted from clean uLLM runtime HEAD
// 0216b131cf5377d90125abd9c1c49c5a8a210511,
// crates/ullm-engine/src/sq8_serving_runtime.rs.

pub fn qwen3_14b_sq8_serving_kv_cache_bytes_per_layer() -> Result<usize, Sq8ServingError> {
    let shape = qwen3_14b_sq8_serving_cache_shape();
    shape.validate().map_err(Sq8ServingError::invalid_configuration)?;
    shape
        .k_cache_elements()
        .and_then(|k| shape.v_cache_elements().and_then(|v| k.checked_add(v)))
        .and_then(|elements| {
            elements
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| "KV cache bytes overflow".into())
        })
        .map_err(Sq8ServingError::invalid_configuration)
}

// The same source has a test named
// serving_cache_byte_count_matches_frozen_f32_layout(), which expects
// 33_554_432 bytes/layer and 1_342_177_280 bytes for all 40 layers.
