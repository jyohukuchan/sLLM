use std::path::PathBuf;

use sllm_core::{
    LayerType, ModelLock, TensorDType, TensorDescriptor, WEIGHT_LOAD_CHUNK_BYTES,
    WeightClassification, build_weight_load_plan, read_model_lock,
};

fn repository_path(relative: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(relative)
}

fn qwen_lock() -> ModelLock {
    read_model_lock(repository_path("docs/models/locks/qwen3.5-4b-bf16.json"))
        .expect("fixed Qwen lock parses")
}

fn descriptor(name: String, source_file: &str, start: u64, byte_size: u64) -> TensorDescriptor {
    TensorDescriptor {
        tensor_name: name,
        source_file: source_file.to_owned(),
        dtype: TensorDType::Bf16,
        shape: vec![byte_size],
        header_length_field_bytes: 8,
        header_length_bytes: 1,
        data_buffer_start: 9,
        data_offset_basis: "safetensors-data-buffer".to_owned(),
        data_offsets: [start - 9, start - 9 + byte_size],
        absolute_byte_range: [start, start + byte_size],
        byte_size,
    }
}

fn complete_descriptors(lock: &ModelLock) -> Vec<TensorDescriptor> {
    let source = lock
        .model
        .files
        .iter()
        .find(|file| file.path.ends_with(".safetensors"))
        .expect("Qwen lock has a weight shard");
    let mut names = vec![
        "model.language_model.embed_tokens.weight".to_owned(),
        "model.language_model.norm.weight".to_owned(),
    ];
    for (layer, layer_type) in lock
        .model
        .architecture
        .text_config
        .layer_types
        .iter()
        .enumerate()
    {
        let prefix = format!("model.language_model.layers.{layer}.");
        for suffix in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ] {
            names.push(format!("{prefix}{suffix}"));
        }
        let class_suffixes: &[&str] = match layer_type {
            LayerType::LinearAttention => &[
                "linear_attn.in_proj_qkv.weight",
                "linear_attn.in_proj_z.weight",
                "linear_attn.in_proj_b.weight",
                "linear_attn.in_proj_a.weight",
                "linear_attn.conv1d.weight",
                "linear_attn.A_log",
                "linear_attn.dt_bias",
                "linear_attn.norm.weight",
                "linear_attn.out_proj.weight",
            ],
            LayerType::FullAttention => &[
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "self_attn.q_norm.weight",
                "self_attn.k_norm.weight",
            ],
        };
        names.extend(
            class_suffixes
                .iter()
                .map(|suffix| format!("{prefix}{suffix}")),
        );
    }
    names.extend(
        (0..lock.model.architecture.vision.tensor_count).map(|index| {
            format!(
                "{}{index}.weight",
                lock.model.architecture.vision.tensor_prefix
            )
        }),
    );
    names.extend((0..lock.model.architecture.mtp.tensor_count).map(|index| {
        format!(
            "{}{index}.weight",
            lock.model.architecture.mtp.tensor_prefix
        )
    }));

    let mut cursor = 17_u64;
    names
        .into_iter()
        .map(|name| {
            let byte_size = if name == "model.language_model.embed_tokens.weight" {
                WEIGHT_LOAD_CHUNK_BYTES + 1
            } else {
                2
            };
            let output = descriptor(name, &source.path, cursor, byte_size);
            cursor += byte_size;
            assert!(cursor <= source.size_bytes, "synthetic metadata fits shard");
            output
        })
        .collect()
}

#[test]
fn complete_plan_is_order_independent_and_preserves_unconsumed_metadata() {
    let lock = qwen_lock();
    let descriptors = complete_descriptors(&lock);
    let plan = build_weight_load_plan(&lock, descriptors.iter()).expect("complete plan builds");

    assert_eq!(plan.entries.len(), 738);
    assert_eq!(
        plan.entries
            .iter()
            .filter(|entry| entry.classification == WeightClassification::Required)
            .count(),
        426
    );
    assert_eq!(
        plan.entries
            .iter()
            .filter(|entry| entry.classification == WeightClassification::KnownUnconsumed)
            .count(),
        312
    );
    assert!(
        plan.entries
            .windows(2)
            .all(|pair| pair[0].tensor_name < pair[1].tensor_name)
    );
    assert!(
        plan.entries
            .iter()
            .filter(|entry| entry.classification == WeightClassification::KnownUnconsumed)
            .all(|entry| entry.consumer.is_none()
                && entry.destination_start.is_none()
                && entry.chunks.is_empty())
    );

    let embedding = plan
        .entries
        .iter()
        .find(|entry| entry.tensor_name == "model.language_model.embed_tokens.weight")
        .expect("embedding is present");
    assert_eq!(
        embedding
            .chunks
            .iter()
            .map(|chunk| chunk.byte_length)
            .collect::<Vec<_>>(),
        [WEIGHT_LOAD_CHUNK_BYTES, 1]
    );
    assert_eq!(
        plan.total_destination_bytes,
        WEIGHT_LOAD_CHUNK_BYTES + 1 + 425 * 2
    );

    let mut reversed = descriptors.clone();
    reversed.reverse();
    let reversed_plan =
        build_weight_load_plan(&lock, reversed.iter()).expect("reversed input builds");
    assert_eq!(plan.digest(), reversed_plan.digest());
    assert_eq!(plan.entries, reversed_plan.entries);
}

#[test]
fn missing_duplicate_unknown_and_wrong_layer_class_fail_closed() {
    let lock = qwen_lock();
    let baseline = complete_descriptors(&lock);

    let mut missing = baseline.clone();
    missing.retain(|entry| entry.tensor_name != "model.language_model.norm.weight");
    assert!(build_weight_load_plan(&lock, missing.iter()).is_err());

    let mut duplicate = baseline.clone();
    duplicate.push(duplicate[0].clone());
    assert!(build_weight_load_plan(&lock, duplicate.iter()).is_err());

    let mut unknown = baseline.clone();
    unknown[0].tensor_name = "model.language_model.unexpected.weight".to_owned();
    assert!(build_weight_load_plan(&lock, unknown.iter()).is_err());

    let linear_layer = lock
        .model
        .architecture
        .text_config
        .layer_types
        .iter()
        .position(|layer| *layer == LayerType::LinearAttention)
        .expect("Qwen schedule has a linear layer");
    let expected = format!("model.language_model.layers.{linear_layer}.linear_attn.A_log");
    let replacement = format!("model.language_model.layers.{linear_layer}.self_attn.q_proj.weight");
    let mut wrong_class = baseline.clone();
    wrong_class
        .iter_mut()
        .find(|entry| entry.tensor_name == expected)
        .expect("linear tensor exists")
        .tensor_name = replacement;
    assert!(build_weight_load_plan(&lock, wrong_class.iter()).is_err());
}

#[test]
fn source_range_and_identity_mutations_fail_closed() {
    let lock = qwen_lock();
    let baseline = complete_descriptors(&lock);

    let mut reversed = baseline.clone();
    reversed[0].absolute_byte_range = [20, 19];
    assert!(build_weight_load_plan(&lock, reversed.iter()).is_err());

    let mut mismatched = baseline.clone();
    mismatched[0].byte_size += 1;
    assert!(build_weight_load_plan(&lock, mismatched.iter()).is_err());

    let mut unknown_source = baseline.clone();
    unknown_source[0].source_file = "unknown.safetensors".to_owned();
    assert!(build_weight_load_plan(&lock, unknown_source.iter()).is_err());

    let mut wrong_identity = lock.clone();
    wrong_identity.fingerprint = format!("sha256:{}", "0".repeat(64));
    assert!(build_weight_load_plan(&wrong_identity, baseline.iter()).is_err());
}
