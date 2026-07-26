// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Prints the fail-closed text-decoder contract resolved from a local model or
//! uLLM package.  This is diagnostic-only; it does not load weights or create
//! a runtime context.

use std::path::PathBuf;
use std::process::ExitCode;
use ullm_engine::model_config::{
    LoadedModelConfig, ModelConfig, ModelExecutionStatus, load_model_config_from_dir,
    load_model_config_from_package,
};

fn usage() -> &'static str {
    "usage: ullm-model-config-inspect (--model-dir PATH | --package PATH) [--require-executor]"
}

fn main() -> ExitCode {
    match run(std::env::args().skip(1).collect()) {
        Ok(output) => {
            println!("{output}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("ullm-model-config-inspect: {error}");
            ExitCode::from(2)
        }
    }
}

fn run(args: Vec<String>) -> Result<serde_json::Value, String> {
    let mut model_dir = None::<PathBuf>;
    let mut package_dir = None::<PathBuf>;
    let mut require_executor = false;
    let mut arguments = args.into_iter();
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--model-dir" => {
                let value = arguments
                    .next()
                    .ok_or_else(|| format!("--model-dir requires a path; {}", usage()))?;
                if model_dir.replace(PathBuf::from(value)).is_some() {
                    return Err(format!(
                        "--model-dir was supplied more than once; {}",
                        usage()
                    ));
                }
            }
            "--package" => {
                let value = arguments
                    .next()
                    .ok_or_else(|| format!("--package requires a path; {}", usage()))?;
                if package_dir.replace(PathBuf::from(value)).is_some() {
                    return Err(format!(
                        "--package was supplied more than once; {}",
                        usage()
                    ));
                }
            }
            "--require-executor" => require_executor = true,
            "--help" | "-h" => return Err(usage().to_string()),
            _ => return Err(format!("unknown argument {argument:?}; {}", usage())),
        }
    }
    let loaded = match (model_dir, package_dir) {
        (Some(_), Some(_)) => return Err(format!("choose one input source; {}", usage())),
        (Some(model_dir), None) => load_model_config_from_dir(model_dir)?,
        (None, Some(package_dir)) => load_model_config_from_package(package_dir)?,
        (None, None) => return Err(usage().to_string()),
    };
    if require_executor {
        loaded.require_implemented_executor()?;
    }
    Ok(summary_json(&loaded))
}

fn summary_json(loaded: &LoadedModelConfig) -> serde_json::Value {
    let execution = match loaded.execution_status() {
        ModelExecutionStatus::Qwen3FullAttention => serde_json::json!({
            "status": "implemented",
            "executor": "Qwen3FullAttention"
        }),
        ModelExecutionStatus::Qwen35Aq4Text => serde_json::json!({
            "status": "implemented",
            "executor": "Qwen35Aq4Text"
        }),
        ModelExecutionStatus::Gemma4TextNonquantized => serde_json::json!({
            "status": "implemented_diagnostic_only",
            "executor": "Gemma4TextExecutor",
            "weight_path": "BF16 source safetensors",
            "activation_path": "F32",
            "quantized_serving": false
        }),
        ModelExecutionStatus::Unimplemented {
            required_executor,
            reason,
        } => serde_json::json!({
            "status": "unimplemented",
            "required_executor": required_executor,
            "reason": reason
        }),
    };
    serde_json::json!({
        "source_model_dir": &loaded.source_model_dir,
        "config_path": &loaded.config_path,
        "config_sha256": &loaded.config_sha256,
        "architecture": loaded.architecture_kind().architecture_name(),
        "execution": execution,
        "text_decoder": decoder_summary(&loaded.model),
    })
}

fn decoder_summary(config: &ModelConfig) -> serde_json::Value {
    match config {
        ModelConfig::Qwen3(config) => serde_json::json!({
            "model_type": &config.decoder.model_type,
            "hidden_size": config.decoder.hidden_size,
            "num_hidden_layers": config.decoder.num_hidden_layers,
            "num_attention_heads": config.decoder.num_attention_heads,
            "num_key_value_heads": config.decoder.num_key_value_heads,
            "head_dim": config.decoder.head_dim,
            "intermediate_size": config.dense_mlp.intermediate_size,
            "hidden_act": &config.dense_mlp.activation,
            "rms_norm_eps": config.decoder.rms_norm_eps,
            "rope_theta": config.rope_theta,
            "max_position_embeddings": config.max_position_embeddings,
            "max_window_layers": config.max_window_layers,
            "use_sliding_window": config.use_sliding_window,
            "sliding_window": config.sliding_window,
            "vocab_size": config.decoder.vocab_size,
            "tie_word_embeddings": config.decoder.tie_word_embeddings,
            "legacy_runtime_rotary_dim": config.legacy_runtime_rotary_dim().ok(),
        }),
        ModelConfig::Gemma4Text(config) => serde_json::json!({
            "model_type": &config.decoder.model_type,
            "hidden_size": config.decoder.hidden_size,
            "num_hidden_layers": config.decoder.num_hidden_layers,
            "num_attention_heads": config.decoder.num_attention_heads,
            "num_key_value_heads": config.decoder.num_key_value_heads,
            "local_head_dim": config.local_head_dim,
            "global_head_dim": config.global_head_dim,
            "num_global_key_value_heads": config.num_global_key_value_heads,
            "intermediate_size": config.dense_mlp.intermediate_size,
            "hidden_activation": &config.dense_mlp.activation,
            "layer_types": config.layer_types.iter().map(|kind| kind.as_str()).collect::<Vec<_>>(),
            "sliding_window": config.sliding_window,
            "sliding_rope": {"type": &config.sliding_rope.rope_type, "theta": config.sliding_rope.rope_theta},
            "full_rope": {"type": &config.full_rope.rope_type, "theta": config.full_rope.rope_theta, "partial_rotary_factor": config.full_rope.partial_rotary_factor},
            "num_kv_shared_layers": config.num_kv_shared_layers,
            "use_double_wide_mlp": config.use_double_wide_mlp,
            "hidden_size_per_layer_input": config.hidden_size_per_layer_input,
            "vocab_size_per_layer_input": config.vocab_size_per_layer_input,
            "final_logit_softcapping": config.final_logit_softcapping,
            "max_position_embeddings": config.max_position_embeddings,
            "use_bidirectional_attention": &config.use_bidirectional_attention,
            "tie_word_embeddings": config.decoder.tie_word_embeddings,
        }),
        ModelConfig::Qwen35DenseText(config) => qwen35_summary(
            &config.hybrid,
            serde_json::json!({"intermediate_size": config.dense_mlp.intermediate_size}),
        ),
        ModelConfig::Qwen35MoeText(config) => qwen35_summary(
            &config.hybrid,
            serde_json::json!({
                "num_experts": config.moe.num_experts,
                "num_experts_per_tok": config.moe.num_experts_per_tok,
                "moe_intermediate_size": config.moe.expert_intermediate_size,
                "shared_expert_intermediate_size": config.moe.shared_expert_intermediate_size,
                "router_aux_loss_coef": config.moe.router_aux_loss_coef,
            }),
        ),
    }
}

fn qwen35_summary(
    config: &ullm_engine::model_config::Qwen35HybridTextConfig,
    mlp: serde_json::Value,
) -> serde_json::Value {
    serde_json::json!({
        "model_type": &config.decoder.model_type,
        "hidden_size": config.decoder.hidden_size,
        "num_hidden_layers": config.decoder.num_hidden_layers,
        "num_attention_heads": config.decoder.num_attention_heads,
        "num_key_value_heads": config.decoder.num_key_value_heads,
        "head_dim": config.decoder.head_dim,
        "hidden_act": &config.activation,
        "layer_types": config.layer_types.iter().map(|kind| kind.as_str()).collect::<Vec<_>>(),
        "full_attention_interval": config.full_attention_interval,
        "attn_output_gate": config.attn_output_gate,
        "linear_attention": {
            "conv_kernel_dim": config.linear_attention.conv_kernel_dim,
            "key_head_dim": config.linear_attention.key_head_dim,
            "num_key_heads": config.linear_attention.num_key_heads,
            "num_value_heads": config.linear_attention.num_value_heads,
            "value_head_dim": config.linear_attention.value_head_dim,
            "state_dtype": &config.linear_attention.state_dtype,
        },
        "rope": {
            "type": &config.rope.rope_type,
            "theta": config.rope.rope_theta,
            "partial_rotary_factor": config.rope.partial_rotary_factor,
            "mrope_interleaved": config.rope.mrope_interleaved,
            "mrope_sections": &config.rope.mrope_sections,
        },
        "mlp_only_layers": &config.mlp_only_layers,
        "mtp": {
            "num_hidden_layers": config.mtp.num_hidden_layers,
            "use_dedicated_embeddings": config.mtp.use_dedicated_embeddings,
        },
        "vocab_size": config.decoder.vocab_size,
        "tie_word_embeddings": config.decoder.tie_word_embeddings,
        "mlp": mlp,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requires_exactly_one_input_source() {
        let error = run(Vec::new()).unwrap_err();
        assert!(error.contains("usage:"), "{error}");
        let error = run(vec![
            "--model-dir".to_string(),
            "one".to_string(),
            "--package".to_string(),
            "two".to_string(),
        ])
        .unwrap_err();
        assert!(error.contains("choose one input source"), "{error}");
    }
}
