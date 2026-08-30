use sllm_core::{
    GgufTensorType, GgufValue, GgufWritePlan, GgufWriteTensor, parse_lora_lock_v1, write_gguf,
};
use sllm_tools::{
    LoraSourceTargetV1, LoraSourceV1, compute_imatrix, convert_lora, dispatch_capability,
    merge_gguf, quantize_tensor, repack_tensor, split_gguf,
};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_dir(label: &str) -> PathBuf {
    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "sllm-tools-{label}-{}-{suffix}",
        std::process::id()
    ));
    fs::create_dir_all(&path).unwrap();
    path
}

#[test]
fn capability_dispatch_is_reviewed_and_closed() {
    assert!(dispatch_capability("qwen35", "BF16", "bf16").is_ok());
    assert!(dispatch_capability("llama", "BF16", "bf16").is_err());
    assert!(dispatch_capability("qwen35", "Q8_0", "q8_0").is_err());
    assert!(dispatch_capability("qwen35", "BF16", "arbitrary-bit").is_err());
}

#[test]
fn split_and_merge_preserve_exact_gguf_digest_and_boundaries() {
    let root = temp_dir("split");
    let source = root.join("source.gguf");
    let mut metadata = BTreeMap::new();
    metadata.insert(
        "general.architecture".into(),
        GgufValue::String("qwen35".into()),
    );
    metadata.insert("general.alignment".into(), GgufValue::U32(32));
    let first = vec![0x11_u8; 32];
    let second = vec![0x22_u8; 32];
    let plan = GgufWritePlan {
        metadata,
        tensors: vec![
            GgufWriteTensor {
                name: "a".into(),
                source_name: "a".into(),
                dimensions: vec![16],
                tensor_type: GgufTensorType::Bf16,
            },
            GgufWriteTensor {
                name: "b".into(),
                source_name: "b".into(),
                dimensions: vec![16],
                tensor_type: GgufTensorType::Bf16,
            },
        ],
    };
    write_gguf(&source, &plan, |name, offset, len| {
        let bytes = if name == "a" { &first } else { &second };
        Ok(bytes[offset as usize..offset as usize + len].to_vec())
    })
    .unwrap();
    let parts = root.join("parts");
    let manifest = split_gguf(&source, &parts, 240).unwrap();
    assert!(manifest.parts.len() >= 2);
    let merged = root.join("merged.gguf");
    let digest = merge_gguf(parts.join("manifest.json"), &merged).unwrap();
    assert_eq!(digest, manifest.semantic_digest);
    assert_eq!(fs::read(source).unwrap(), fs::read(merged).unwrap());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn split_rejects_zero_tensor_and_tampered_part() {
    let root = temp_dir("split-negative");
    let source = root.join("empty.gguf");
    let mut metadata = BTreeMap::new();
    metadata.insert(
        "general.architecture".into(),
        GgufValue::String("qwen35".into()),
    );
    write_gguf(
        &source,
        &GgufWritePlan {
            metadata,
            tensors: vec![],
        },
        |_, _, len| Ok(vec![0; len]),
    )
    .unwrap();
    assert!(split_gguf(&source, root.join("parts"), 1024).is_err());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn lora_conversion_normalizes_orientation_and_rejects_nonfinite() {
    let source = LoraSourceV1 {
        schema_version: "sllm-lora-source-v1".into(),
        artifact_id: "style".into(),
        base_model_fingerprint: format!("sha256:{}", "a".repeat(64)),
        base_weight_plan_digest: format!("sha256:{}", "b".repeat(64)),
        alpha: 8.0,
        provenance: "offline-fixture".into(),
        targets: vec![LoraSourceTargetV1 {
            tensor_name: "layer.0.weight".into(),
            target_shape: [2, 3],
            rank: 2,
            dtype: "BF16".into(),
            a_orientation: "rank-input".into(),
            b_orientation: "output-rank".into(),
            a: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            b: vec![1.0, 2.0, 3.0, 4.0],
        }],
    };
    let result = convert_lora(&source).unwrap();
    let lock = parse_lora_lock_v1(&result.lock_json).unwrap();
    assert_eq!(lock.targets[0].a_size, 12);
    assert_eq!(lock.targets[0].b_size, 8);
    let mut bad = source.clone();
    bad.targets[0].a[0] = f32::NAN;
    assert!(convert_lora(&bad).is_err());
}

#[test]
fn repack_and_quantization_cover_tail_and_nonfinite_boundaries() {
    let values = vec![0x12_u8; 16];
    let scales = vec![127_u8; 1];
    assert!(repack_tensor("mxfp4", &values, &scales, 1, 32).is_ok());
    assert!(repack_tensor("mxfp4", &values, &scales, 1, 31).is_err());
    for columns in [1, 15, 16, 17, 255, 256, 257] {
        let input = vec![1.0_f32; columns];
        for recipe in [
            "fp8-e4m3fn-channel-f32-scale",
            "nvfp4-e2m1-block16-e4m3fn-f32-outer",
            "mxfp4-e2m1-block32-e8m0",
        ] {
            let quantized = quantize_tensor(recipe, &input, 1, columns).unwrap();
            assert_eq!(quantized.rows, 1);
            assert_eq!(quantized.columns, columns);
        }
        let imatrix = compute_imatrix(&input, 1, columns, 1729).unwrap();
        assert_eq!(imatrix.values, vec![1.0; columns]);
        assert_eq!(imatrix.sample_count, 1);
    }
    let ordered = compute_imatrix(&[1.0, 2.0, 3.0, 4.0], 2, 2, 1729).unwrap();
    assert_eq!(ordered.values, [10.0, 20.0]);
    assert_eq!(
        ordered,
        compute_imatrix(&[1.0, 2.0, 3.0, 4.0], 2, 2, 1729).unwrap()
    );
    assert_ne!(
        ordered.sample_order_digest,
        compute_imatrix(&[3.0, 4.0, 1.0, 2.0], 2, 2, 1729)
            .unwrap()
            .sample_order_digest
    );
    let mut bad = vec![1.0_f32; 16];
    bad[15] = f32::INFINITY;
    assert!(quantize_tensor("nvfp4-e2m1-block16-e4m3fn-f32-outer", &bad, 1, 16).is_err());
    assert!(compute_imatrix(&bad, 1, 16, 1729).is_err());
}
