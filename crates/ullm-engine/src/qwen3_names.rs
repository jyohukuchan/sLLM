// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

const LANGUAGE_MODEL_PREFIX: &str = "model.language_model.";
const HF_MODEL_PREFIX: &str = "model.";

fn is_supported_qwen3_relative_name(relative: &str) -> bool {
    relative.starts_with("layers.")
        || relative.starts_with("embed_tokens.")
        || relative.starts_with("norm.")
}

pub fn qwen3_tensor_name_alias(name: &str) -> Option<String> {
    if let Some(relative) = name.strip_prefix(LANGUAGE_MODEL_PREFIX) {
        if is_supported_qwen3_relative_name(relative) {
            return Some(format!("{HF_MODEL_PREFIX}{relative}"));
        }
        return None;
    }

    let relative = name.strip_prefix(HF_MODEL_PREFIX)?;
    if is_supported_qwen3_relative_name(relative) {
        return Some(format!("{LANGUAGE_MODEL_PREFIX}{relative}"));
    }
    None
}

pub fn qwen3_layer_index_from_tensor_suffix(tensor_name: &str, suffix: &str) -> Option<usize> {
    let relative = tensor_name
        .strip_prefix(LANGUAGE_MODEL_PREFIX)
        .or_else(|| tensor_name.strip_prefix(HF_MODEL_PREFIX))?;
    let layer_and_suffix = relative.strip_prefix("layers.")?;
    let layer = layer_and_suffix.strip_suffix(suffix)?;
    layer.parse::<usize>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen3_tensor_name_alias_swaps_supported_model_prefixes() {
        assert_eq!(
            qwen3_tensor_name_alias("model.language_model.layers.0.self_attn.q_proj.weight"),
            Some("model.layers.0.self_attn.q_proj.weight".to_string())
        );
        assert_eq!(
            qwen3_tensor_name_alias("model.layers.0.self_attn.q_proj.weight"),
            Some("model.language_model.layers.0.self_attn.q_proj.weight".to_string())
        );
        assert_eq!(
            qwen3_tensor_name_alias("model.language_model.embed_tokens.weight"),
            Some("model.embed_tokens.weight".to_string())
        );
        assert_eq!(
            qwen3_tensor_name_alias("model.norm.weight"),
            Some("model.language_model.norm.weight".to_string())
        );
        assert_eq!(qwen3_tensor_name_alias("lm_head.weight"), None);
        assert_eq!(
            qwen3_tensor_name_alias("model.visual.patch_embed.weight"),
            None
        );
    }

    #[test]
    fn qwen3_layer_index_from_tensor_suffix_accepts_both_namespaces() {
        assert_eq!(
            qwen3_layer_index_from_tensor_suffix(
                "model.language_model.layers.12.self_attn.q_proj.weight",
                ".self_attn.q_proj.weight",
            ),
            Some(12)
        );
        assert_eq!(
            qwen3_layer_index_from_tensor_suffix(
                "model.layers.7.linear_attn.in_proj_qkv.weight",
                ".linear_attn.in_proj_qkv.weight",
            ),
            Some(7)
        );
        assert_eq!(
            qwen3_layer_index_from_tensor_suffix(
                "model.layers.7.linear_attn.in_proj_qkv.weight_scale_inv",
                ".linear_attn.in_proj_qkv.weight",
            ),
            None
        );
    }
}
