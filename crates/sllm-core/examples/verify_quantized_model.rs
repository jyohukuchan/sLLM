use sllm_core::{
    QuantizedTensorRole, build_gemma4_graph, build_gemma4_quantized_execution_layout,
    build_unsloth_gemma4_nvfp4_weight_load_plan, parse_gemma4_model_lock,
    verify_unsloth_gemma4_nvfp4,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::env::args()
        .nth(1)
        .ok_or("usage: verify_quantized_model MODEL_DIR [GEMMA_IT_LOCK]")?;
    let model = verify_unsloth_gemma4_nvfp4(root)?;
    let plan = std::env::args()
        .nth(2)
        .map(|path| -> Result<_, Box<dyn std::error::Error>> {
            let lock = parse_gemma4_model_lock(&std::fs::read(path)?)?;
            let plan = build_unsloth_gemma4_nvfp4_weight_load_plan(&lock, &model)?;
            let graph = build_gemma4_graph(&lock, &plan, 1, 0, 16)?;
            let layout = build_gemma4_quantized_execution_layout(&graph, &plan, &model)?;
            Ok((
                plan.entries.len(),
                plan.digest_hex(),
                layout.model_weight_bytes(),
                layout.workspace_bytes(),
                layout.request_state_bytes(),
            ))
        })
        .transpose()?;
    let mlp = model
        .tensors()
        .filter(|tensor| tensor.role == QuantizedTensorRole::MlpProjection)
        .count();
    let attention = model
        .tensors()
        .filter(|tensor| tensor.role == QuantizedTensorRole::AttentionProjection)
        .count();
    println!(
        "quantized model: PASS repository={} revision={} tensors={} mlp={} attention={} kv_layers={} recipe_digest={} topology_plan={:?}",
        model.repository(),
        model.resolved_revision(),
        model.tensors().len(),
        mlp,
        attention,
        (0..48)
            .filter(|layer| model.kv_scale(*layer).is_some())
            .count(),
        model.recipe_digest(),
        plan
    );
    Ok(())
}
