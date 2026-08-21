use std::path::PathBuf;
use std::sync::Arc;

use serde_json::json;
use sha2::{Digest, Sha256};
use sllm_core::{
    AdapterModelDimsV1, AdapterRequestSetV1, ControlVectorSelectionV1, LayerType,
    LoraAdapterSelectionV1, ModelLock, TensorDType, TensorDescriptor,
    VerifiedControlVectorPayloadV1, VerifiedLoraPayloadV1, WEIGHT_LOAD_CHUNK_BYTES,
    apply_control_vector_bf16, apply_lora_bf16, build_weight_load_plan, read_model_lock,
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

fn descriptor(
    name: String,
    source_file: &str,
    start: u64,
    byte_size: u64,
    shape: Vec<u64>,
) -> TensorDescriptor {
    TensorDescriptor {
        tensor_name: name,
        source_file: source_file.to_owned(),
        dtype: TensorDType::Bf16,
        shape,
        header_length_field_bytes: 8,
        header_length_bytes: 1,
        data_buffer_start: 9,
        data_offset_basis: "safetensors-data-buffer".to_owned(),
        data_offsets: [start - 9, start - 9 + byte_size],
        absolute_byte_range: [start, start + byte_size],
        byte_size,
    }
}

// The production planner intentionally validates metadata, not real tensor
// contents. This compact fixture mirrors the complete Qwen descriptor set and
// gives one required BF16 matrix a small two-dimensional shape for the oracle.
fn complete_descriptors(lock: &ModelLock) -> Vec<TensorDescriptor> {
    let source = lock
        .model
        .files
        .iter()
        .find(|file| file.path.ends_with(".safetensors"))
        .expect("Qwen lock has a weight shard");
    let target = "model.language_model.layers.0.mlp.gate_proj.weight";
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
            let shape = if name == target {
                vec![3, 2]
            } else {
                vec![if name == "model.language_model.embed_tokens.weight" {
                    WEIGHT_LOAD_CHUNK_BYTES + 1
                } else {
                    2
                }]
            };
            let byte_size = if name == target {
                12
            } else if name == "model.language_model.embed_tokens.weight" {
                WEIGHT_LOAD_CHUNK_BYTES + 1
            } else {
                2
            };
            let output = descriptor(name, &source.path, cursor, byte_size, shape);
            cursor += byte_size;
            assert!(cursor <= source.size_bytes, "synthetic metadata fits shard");
            output
        })
        .collect()
}

fn plan_fixture() -> (ModelLock, sllm_core::WeightLoadPlan) {
    let lock = qwen_lock();
    let plan = build_weight_load_plan(&lock, complete_descriptors(&lock).iter())
        .expect("complete fixture plan builds");
    (lock, plan)
}

fn sha256_identity(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}

fn lora_lock_json(lock: &ModelLock, plan: &sllm_core::WeightLoadPlan, payload: &[u8]) -> Vec<u8> {
    serde_json::to_vec(&json!({
        "schema_version": "sllm-adapter-lock-v1",
        "kind": "lora",
        "artifact_id": "lora-fixture-v1",
        "alpha": 2.0,
        "base_model_fingerprint": lock.fingerprint(),
        "base_weight_plan_digest": plan.digest_hex(),
        "payload_sha256": sha256_identity(payload),
        "payload_size": payload.len(),
        "targets": [{
            "tensor_name": "model.language_model.layers.0.mlp.gate_proj.weight",
            "dtype": "BF16",
            "target_shape": [3, 2],
            "rank": 2,
            "a_offset": 0,
            "a_size": 8,
            "b_offset": 8,
            "b_size": 12,
        }],
    }))
    .expect("fixture lock serializes")
}

fn control_lock_json(
    lock: &ModelLock,
    plan: &sllm_core::WeightLoadPlan,
    dims: AdapterModelDimsV1,
    payload: &[u8],
) -> Vec<u8> {
    serde_json::to_vec(&json!({
        "schema_version": "sllm-adapter-lock-v1",
        "kind": "control-vector",
        "artifact_id": "control-fixture-v1",
        "dtype": "bf16",
        "base_model_fingerprint": lock.fingerprint(),
        "base_weight_plan_digest": plan.digest_hex(),
        "payload_sha256": sha256_identity(payload),
        "payload_size": payload.len(),
        "hidden_size": dims.hidden_size(),
        "layer_start": 0,
        "layer_end": 1,
        "vector_offset": 0,
        "vector_size": payload.len(),
    }))
    .expect("fixture lock serializes")
}

fn bf16(value: f32) -> u16 {
    (value.to_bits() >> 16) as u16
}

fn bf16_payload(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&value| bf16(value).to_le_bytes())
        .collect()
}

#[test]
fn verified_payloads_bind_identity_and_cpu_oracles() {
    let (lock, plan) = plan_fixture();
    // A is [rank, input] = [[1, 2], [3, 4]], B is [output, rank]
    // = [[5, 6], [7, 8], [9, 10]]. The asymmetric fixture catches a
    // transposed A/B oracle implementation.
    let lora_payload = bf16_payload(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    let lora = VerifiedLoraPayloadV1::from_bytes(
        &lora_lock_json(&lock, &plan, &lora_payload),
        Arc::<[u8]>::from(lora_payload.clone()),
        lock.fingerprint(),
        &plan,
    )
    .expect("LoRA lock verifies");
    assert_eq!(lora.targets().len(), 1);
    assert_eq!(lora.targets()[0].target_shape(), [3, 2]);
    assert_eq!(
        lora.identity().canonical_string(),
        format!(
            "lora:lora-fixture-v1:{}:{}:20:v1",
            lora.identity().lock_sha256(),
            sha256_identity(&lora_payload)
        )
    );

    let mut output = [bf16(1.0), bf16(2.0), bf16(3.0)];
    apply_lora_bf16(
        &lora,
        &lora.targets()[0],
        &[bf16(1.0), bf16(2.0)],
        &mut output,
        0.5,
    )
    .expect("LoRA oracle applies");
    assert_eq!(output, [bf16(46.5), bf16(63.5), bf16(80.5)]);

    let dims = AdapterModelDimsV1::new(
        lock.model.architecture.text_config.hidden_size,
        lock.model.architecture.text_config.num_hidden_layers,
    )
    .expect("Qwen dimensions are nonzero");
    let mut control_payload = vec![0_u8; dims.hidden_size() as usize * 2];
    for pair in control_payload.chunks_exact_mut(2) {
        pair.copy_from_slice(&[0x80, 0x3f]);
    }
    let control = VerifiedControlVectorPayloadV1::from_bytes(
        &control_lock_json(&lock, &plan, dims, &control_payload),
        Arc::<[u8]>::from(control_payload.clone()),
        lock.fingerprint(),
        &plan,
        dims,
    )
    .expect("control-vector lock verifies");
    let mut hidden = vec![bf16(0.0); dims.hidden_size() as usize];
    apply_control_vector_bf16(&control, &mut hidden, dims, 0, 0.5)
        .expect("control-vector oracle applies");
    assert!(hidden.iter().all(|&value| value == bf16(0.5)));
}

#[test]
fn verification_rejects_identity_shape_hash_and_unknown_field_mutations() {
    let (lock, plan) = plan_fixture();
    let payload = vec![0_u8; 20];
    let base = serde_json::from_slice::<serde_json::Value>(&lora_lock_json(&lock, &plan, &payload))
        .expect("fixture lock is JSON");

    let mut wrong_model = base.clone();
    wrong_model["base_model_fingerprint"] = json!(format!("sha256:{}", "0".repeat(64)));
    assert!(
        VerifiedLoraPayloadV1::from_bytes(
            &serde_json::to_vec(&wrong_model).unwrap(),
            Arc::<[u8]>::from(payload.clone()),
            lock.fingerprint(),
            &plan,
        )
        .is_err()
    );

    let mut wrong_plan = base.clone();
    wrong_plan["base_weight_plan_digest"] = json!(format!("sha256:{}", "1".repeat(64)));
    assert!(
        VerifiedLoraPayloadV1::from_bytes(
            &serde_json::to_vec(&wrong_plan).unwrap(),
            Arc::<[u8]>::from(payload.clone()),
            lock.fingerprint(),
            &plan,
        )
        .is_err()
    );

    let mut wrong_shape = base.clone();
    wrong_shape["targets"][0]["target_shape"] = json!([1, 2]);
    assert!(
        VerifiedLoraPayloadV1::from_bytes(
            &serde_json::to_vec(&wrong_shape).unwrap(),
            Arc::<[u8]>::from(payload.clone()),
            lock.fingerprint(),
            &plan,
        )
        .is_err()
    );

    let mut wrong_hash = base.clone();
    wrong_hash["payload_sha256"] = json!(format!("sha256:{}", "2".repeat(64)));
    assert!(
        VerifiedLoraPayloadV1::from_bytes(
            &serde_json::to_vec(&wrong_hash).unwrap(),
            Arc::<[u8]>::from(payload.clone()),
            lock.fingerprint(),
            &plan,
        )
        .is_err()
    );

    let original = VerifiedLoraPayloadV1::from_bytes(
        &serde_json::to_vec(&base).unwrap(),
        Arc::<[u8]>::from(payload.clone()),
        lock.fingerprint(),
        &plan,
    )
    .expect("baseline lock verifies");
    let mut changed_alpha = base.clone();
    changed_alpha["alpha"] = json!(1.0);
    let changed = VerifiedLoraPayloadV1::from_bytes(
        &serde_json::to_vec(&changed_alpha).unwrap(),
        Arc::<[u8]>::from(payload.clone()),
        lock.fingerprint(),
        &plan,
    )
    .expect("positive alpha mutation verifies");
    assert_ne!(
        original.identity().lock_sha256(),
        changed.identity().lock_sha256(),
        "lock semantics must be part of artifact identity"
    );

    let mut zero_alpha = base.clone();
    zero_alpha["alpha"] = json!(0.0);
    assert!(
        VerifiedLoraPayloadV1::from_bytes(
            &serde_json::to_vec(&zero_alpha).unwrap(),
            Arc::<[u8]>::from(payload.clone()),
            lock.fingerprint(),
            &plan,
        )
        .is_err()
    );

    let mut unknown = base;
    unknown["unexpected"] = json!(true);
    assert!(
        VerifiedLoraPayloadV1::from_bytes(
            &serde_json::to_vec(&unknown).unwrap(),
            Arc::<[u8]>::from(payload.clone()),
            lock.fingerprint(),
            &plan,
        )
        .is_err()
    );
}

#[test]
fn request_identity_is_stable_and_aliases_are_bounded() {
    let (lock, plan) = plan_fixture();
    let payload = vec![0_u8; 20];
    let lora = Arc::new(
        VerifiedLoraPayloadV1::from_bytes(
            &lora_lock_json(&lock, &plan, &payload),
            Arc::<[u8]>::from(payload.clone()),
            lock.fingerprint(),
            &plan,
        )
        .expect("LoRA lock verifies"),
    );
    let dims = AdapterModelDimsV1::new(
        lock.model.architecture.text_config.hidden_size,
        lock.model.architecture.text_config.num_hidden_layers,
    )
    .unwrap();
    let control_payload = vec![0_u8; dims.hidden_size() as usize * 2];
    let control = Arc::new(
        VerifiedControlVectorPayloadV1::from_bytes(
            &control_lock_json(&lock, &plan, dims, &control_payload),
            Arc::<[u8]>::from(control_payload),
            lock.fingerprint(),
            &plan,
            dims,
        )
        .expect("control lock verifies"),
    );
    let request = AdapterRequestSetV1::new(
        vec![LoraAdapterSelectionV1 {
            alias: "tone".to_owned(),
            artifact: Arc::clone(&lora),
            scale: 0.5,
        }],
        vec![ControlVectorSelectionV1 {
            alias: "style".to_owned(),
            artifact: Arc::clone(&control),
            scale: -1.0,
        }],
    )
    .expect("request validates");
    assert_eq!(
        request.identity(),
        format!(
            "adapter:set-v1|lora:tone:lora-fixture-v1:{}:3f000000|control:style:control-fixture-v1:{}:bf800000",
            lora.identity().lock_sha256(),
            control.identity().lock_sha256(),
        )
    );
    assert_eq!(
        AdapterRequestSetV1::disabled().identity(),
        "adapter:none-v1"
    );

    assert!(
        AdapterRequestSetV1::new(
            vec![LoraAdapterSelectionV1 {
                alias: "same".to_owned(),
                artifact: Arc::clone(&lora),
                scale: 1.0,
            }],
            vec![ControlVectorSelectionV1 {
                alias: "same".to_owned(),
                artifact: Arc::clone(&control),
                scale: 1.0,
            }],
        )
        .is_err()
    );
    assert!(
        AdapterRequestSetV1::new(
            Vec::new(),
            vec![
                ControlVectorSelectionV1 {
                    alias: "a".to_owned(),
                    artifact: Arc::clone(&control),
                    scale: 1.0,
                },
                ControlVectorSelectionV1 {
                    alias: "b".to_owned(),
                    artifact: Arc::clone(&control),
                    scale: 1.0,
                },
            ],
        )
        .is_err()
    );
    assert!(
        AdapterRequestSetV1::new(
            vec![LoraAdapterSelectionV1 {
                alias: "bad alias".to_owned(),
                artifact: Arc::clone(&lora),
                scale: 1.0,
            }],
            Vec::new(),
        )
        .is_err()
    );
    assert!(
        AdapterRequestSetV1::new(
            vec![LoraAdapterSelectionV1 {
                alias: "tone".to_owned(),
                artifact: Arc::clone(&lora),
                scale: f32::NAN,
            }],
            Vec::new(),
        )
        .is_err()
    );
}

#[test]
fn control_vector_range_and_request_count_fail_closed() {
    let (lock, plan) = plan_fixture();
    let dims = AdapterModelDimsV1::new(
        lock.model.architecture.text_config.hidden_size,
        lock.model.architecture.text_config.num_hidden_layers,
    )
    .unwrap();
    let payload = vec![0_u8; dims.hidden_size() as usize * 2];
    let mut value = serde_json::from_slice::<serde_json::Value>(&control_lock_json(
        &lock, &plan, dims, &payload,
    ))
    .unwrap();
    value["vector_size"] = json!(payload.len() - 2);
    assert!(
        VerifiedControlVectorPayloadV1::from_bytes(
            &serde_json::to_vec(&value).unwrap(),
            Arc::<[u8]>::from(payload.clone()),
            lock.fingerprint(),
            &plan,
            dims,
        )
        .is_err()
    );

    value["vector_size"] = json!(payload.len());
    value["dtype"] = json!("BF16");
    assert!(
        VerifiedControlVectorPayloadV1::from_bytes(
            &serde_json::to_vec(&value).unwrap(),
            Arc::<[u8]>::from(payload),
            lock.fingerprint(),
            &plan,
            dims,
        )
        .is_err()
    );
}
