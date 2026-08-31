use sha2::{Digest, Sha256};
use sllm_core::{
    DerivedGgufConverter, DerivedGgufLock, GgufArray, GgufError, GgufRecipeEncoding,
    GgufScaleBinding, GgufScaleRole, GgufTensorBinding, GgufTensorRecipeV1, GgufTensorScope,
    GgufTensorType, GgufValue, GgufWritePlan, GgufWriteTensor, SLLM_EXTENSION_VERSION_KEY,
    SLLM_FRONTEND_CONFIG_KEY, SLLM_FRONTEND_TOKENIZER_CONFIG_KEY, SLLM_FRONTEND_TOKENIZER_KEY,
    SLLM_TENSOR_RECIPE_KEY, SLLM_TENSOR_RECIPE_SHA256_KEY, VerifiedGguf, verify_derived_gguf,
    write_gguf,
};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_FILE: AtomicU64 = AtomicU64::new(1);

#[derive(Clone)]
enum TestValue {
    U32(u32),
    String(String),
    StringArray(Vec<String>),
}

#[derive(Clone)]
struct TestTensor {
    name: String,
    dimensions: Vec<u64>,
    raw_type: u32,
    relative_offset: u64,
    payload: Vec<u8>,
}

fn string(value: &str, output: &mut Vec<u8>) {
    output.extend_from_slice(&(value.len() as u64).to_le_bytes());
    output.extend_from_slice(value.as_bytes());
}

fn build(metadata: Vec<(String, TestValue)>, tensors: Vec<TestTensor>) -> Vec<u8> {
    let mut output = Vec::new();
    output.extend_from_slice(b"GGUF");
    output.extend_from_slice(&3_u32.to_le_bytes());
    output.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    output.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
    for (key, value) in metadata {
        string(&key, &mut output);
        match value {
            TestValue::U32(value) => {
                output.extend_from_slice(&4_u32.to_le_bytes());
                output.extend_from_slice(&value.to_le_bytes());
            }
            TestValue::String(value) => {
                output.extend_from_slice(&8_u32.to_le_bytes());
                string(&value, &mut output);
            }
            TestValue::StringArray(values) => {
                output.extend_from_slice(&9_u32.to_le_bytes());
                output.extend_from_slice(&8_u32.to_le_bytes());
                output.extend_from_slice(&(values.len() as u64).to_le_bytes());
                for value in values {
                    string(&value, &mut output);
                }
            }
        }
    }
    for tensor in &tensors {
        string(&tensor.name, &mut output);
        output.extend_from_slice(&(tensor.dimensions.len() as u32).to_le_bytes());
        for dimension in &tensor.dimensions {
            output.extend_from_slice(&dimension.to_le_bytes());
        }
        output.extend_from_slice(&tensor.raw_type.to_le_bytes());
        output.extend_from_slice(&tensor.relative_offset.to_le_bytes());
    }
    if !tensors.is_empty() {
        while output.len() % 32 != 0 {
            output.push(0);
        }
        let data_start = output.len();
        for tensor in tensors {
            let start = data_start + tensor.relative_offset as usize;
            if output.len() < start {
                output.resize(start, 0);
            }
            let end = start + tensor.payload.len();
            if output.len() < end {
                output.resize(end, 0);
            }
            output[start..end].copy_from_slice(&tensor.payload);
        }
    }
    output
}

fn base_metadata() -> Vec<(String, TestValue)> {
    vec![
        (
            "general.architecture".to_owned(),
            TestValue::String("qwen35".to_owned()),
        ),
        ("general.alignment".to_owned(), TestValue::U32(32)),
        (
            "tokenizer.ggml.tokens".to_owned(),
            TestValue::StringArray(vec!["a".to_owned(), "b".to_owned()]),
        ),
    ]
}

fn tensor(name: &str, tensor_type: GgufTensorType, dimensions: &[u64], offset: u64) -> TestTensor {
    let elements = dimensions.iter().product::<u64>();
    let size = elements / tensor_type.block_size() * tensor_type.type_size();
    TestTensor {
        name: name.to_owned(),
        dimensions: dimensions.to_vec(),
        raw_type: tensor_type.raw(),
        relative_offset: offset,
        payload: (0..size).map(|value| value as u8).collect(),
    }
}

fn temp_file(bytes: &[u8]) -> PathBuf {
    let number = NEXT_FILE.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "sllm-gguf-contract-{}-{number}.gguf",
        std::process::id()
    ));
    fs::write(&path, bytes).expect("write synthetic GGUF");
    path
}

fn temp_output(label: &str) -> PathBuf {
    let number = NEXT_FILE.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "sllm-gguf-{label}-{}-{number}.gguf",
        std::process::id()
    ))
}

fn sha256(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("sha256:{:x}", hasher.finalize())
}

fn open_bytes(bytes: &[u8]) -> Result<VerifiedGguf, GgufError> {
    let path = temp_file(bytes);
    let result = VerifiedGguf::open(&path);
    fs::remove_file(path).expect("remove synthetic GGUF");
    result
}

fn assert_invalid(bytes: Vec<u8>, text: &str) {
    let error = open_bytes(&bytes).expect_err("GGUF must fail closed");
    assert!(
        error.to_string().contains(text),
        "expected {text:?}, got {error}"
    );
}

#[test]
fn accepts_zero_and_one_tensor_files() {
    let empty = open_bytes(&build(base_metadata(), vec![])).expect("zero tensor GGUF");
    assert!(empty.tensors().is_empty());
    assert_eq!(empty.architecture(), "qwen35");
    assert_eq!(empty.alignment(), 32);

    let one = open_bytes(&build(
        base_metadata(),
        vec![tensor("weight", GgufTensorType::Bf16, &[3, 5], 0)],
    ))
    .expect("one tensor GGUF");
    assert_eq!(one.tensors().len(), 1);
    assert_eq!(one.tensor("weight").expect("weight").byte_length(), 30);
    assert_eq!(
        one.read_tensor_range("weight", 1, 3).expect("read"),
        [1, 2, 3]
    );
    assert!(one.metadata_sha256().starts_with("sha256:"));
    assert!(one.tensor_catalog_sha256().starts_with("sha256:"));
}

#[test]
fn accepts_supported_architectures_without_extra_required_metadata_and_tokenizer_arrays() {
    for architecture in [
        "qwen35",
        "qwen35moe",
        "gemma4",
        "gemma4moe",
        "deepseek4",
        "minimax-m3",
        "diffusion-gemma",
        "mistral3",
    ] {
        let mut metadata = base_metadata();
        metadata[0].1 = TestValue::String(architecture.to_owned());
        let verified = open_bytes(&build(metadata, vec![])).expect("supported architecture");
        assert_eq!(verified.architecture(), architecture);
        assert!(matches!(
            verified.metadata_value("tokenizer.ggml.tokens"),
            Some(GgufValue::Array(_))
        ));
    }
}

#[test]
fn diffusion_gemma_is_parser_only_and_writer_rejects_it() {
    let output = temp_output("diffusion-gemma-write-disabled");
    let error = write_gguf(
        &output,
        &GgufWritePlan {
            metadata: BTreeMap::from([
                (
                    "general.architecture".to_owned(),
                    GgufValue::String("diffusion-gemma".to_owned()),
                ),
                ("general.alignment".to_owned(), GgufValue::U32(32)),
            ]),
            tensors: vec![],
        },
        |_, _, _| unreachable!("write-disabled plan must fail before payload reads"),
    )
    .expect_err("DiffusionGemma writer must remain disabled");
    assert!(error.to_string().contains("architecture is unsupported"));
    assert!(!output.exists());
}

#[test]
fn rejects_magic_version_architecture_and_alignment_drift() {
    let mut wrong_magic = build(base_metadata(), vec![]);
    wrong_magic[0] = b'X';
    assert_invalid(wrong_magic, "magic differs");

    let mut wrong_version = build(base_metadata(), vec![]);
    wrong_version[4..8].copy_from_slice(&4_u32.to_le_bytes());
    assert_invalid(wrong_version, "unsupported version 4");

    let mut unknown_architecture = base_metadata();
    unknown_architecture[0].1 = TestValue::String("nearby".to_owned());
    assert_invalid(
        build(unknown_architecture, vec![]),
        "unsupported architecture",
    );

    let mut wrong_alignment = base_metadata();
    wrong_alignment[1].1 = TestValue::U32(64);
    assert_invalid(build(wrong_alignment, vec![]), "alignment 64 is not 32");
}

#[test]
fn rejects_duplicate_metadata_and_tensor_names() {
    let mut duplicate_metadata = base_metadata();
    duplicate_metadata.push((
        "general.architecture".to_owned(),
        TestValue::String("qwen35".to_owned()),
    ));
    assert_invalid(build(duplicate_metadata, vec![]), "duplicate metadata key");

    let tensors = vec![
        tensor("weight", GgufTensorType::Bf16, &[16], 0),
        tensor("weight", GgufTensorType::Bf16, &[16], 32),
    ];
    assert_invalid(build(base_metadata(), tensors), "duplicate tensor name");
}

#[test]
fn rejects_unknown_type_misalignment_overlap_and_truncation() {
    let mut unknown = tensor("weight", GgufTensorType::Bf16, &[16], 0);
    unknown.raw_type = 99;
    assert_invalid(
        build(base_metadata(), vec![unknown]),
        "unsupported tensor type 99",
    );

    assert_invalid(
        build(
            base_metadata(),
            vec![tensor("weight", GgufTensorType::Bf16, &[16], 1)],
        ),
        "offset is misaligned",
    );

    assert_invalid(
        build(
            base_metadata(),
            vec![
                tensor("a", GgufTensorType::Bf16, &[32], 0),
                tensor("b", GgufTensorType::Bf16, &[32], 32),
            ],
        ),
        "tensor ranges overlap",
    );

    let mut truncated = build(
        base_metadata(),
        vec![tensor("weight", GgufTensorType::Bf16, &[16], 0)],
    );
    truncated.truncate(truncated.len() - 1);
    assert_invalid(truncated, "exceeds file");
}

#[test]
fn rejects_dimension_overflow_and_unknown_extension_version() {
    assert_invalid(
        build(
            base_metadata(),
            vec![tensor(
                "rank-five",
                GgufTensorType::Bf16,
                &[1, 1, 1, 1, 2],
                0,
            )],
        ),
        "dimension count",
    );
    assert!(
        GgufWriteTensor {
            name: "rank-five".to_owned(),
            source_name: "rank-five".to_owned(),
            dimensions: vec![1, 1, 1, 1, 2],
            tensor_type: GgufTensorType::Bf16,
        }
        .byte_length()
        .expect_err("writer must reject rank five")
        .to_string()
        .contains("dimension count")
    );

    let overflowing = TestTensor {
        name: "overflow".to_owned(),
        dimensions: vec![u64::MAX, 2],
        raw_type: GgufTensorType::Bf16.raw(),
        relative_offset: 0,
        payload: vec![],
    };
    assert_invalid(
        build(base_metadata(), vec![overflowing]),
        "element count overflows",
    );

    let recipe = fp8_recipe();
    let mut metadata = base_metadata();
    metadata.extend([
        (SLLM_EXTENSION_VERSION_KEY.to_owned(), TestValue::U32(2)),
        (
            SLLM_TENSOR_RECIPE_KEY.to_owned(),
            TestValue::String(recipe.canonical_json().expect("canonical recipe")),
        ),
        (
            SLLM_TENSOR_RECIPE_SHA256_KEY.to_owned(),
            TestValue::String(recipe.digest().expect("digest")),
        ),
    ]);
    assert_invalid(build(metadata, vec![]), "unknown sLLM extension version 2");
}

#[test]
fn enforces_mxfp4_and_nvfp4_block_boundaries() {
    for dimension in [31, 33] {
        assert_invalid(
            build(
                base_metadata(),
                vec![tensor("mxfp4", GgufTensorType::Mxfp4, &[dimension], 0)],
            ),
            "not divisible by block size 32",
        );
    }
    let mxfp4 = open_bytes(&build(
        base_metadata(),
        vec![tensor("mxfp4", GgufTensorType::Mxfp4, &[32], 0)],
    ))
    .expect("MXFP4 boundary");
    assert_eq!(mxfp4.tensor("mxfp4").expect("tensor").byte_length(), 17);

    for dimension in [63, 65] {
        assert_invalid(
            build(
                base_metadata(),
                vec![tensor("nvfp4", GgufTensorType::Nvfp4, &[dimension], 0)],
            ),
            "not divisible by block size 64",
        );
    }
    let nvfp4 = open_bytes(&build(
        base_metadata(),
        vec![tensor("nvfp4", GgufTensorType::Nvfp4, &[64], 0)],
    ))
    .expect("NVFP4 boundary");
    assert_eq!(nvfp4.tensor("nvfp4").expect("tensor").byte_length(), 36);
}

fn fp8_recipe() -> GgufTensorRecipeV1 {
    GgufTensorRecipeV1 {
        schema_version: "sllm-gguf-tensor-recipe-v1".to_owned(),
        semantic_model_id: "qwen35-test".to_owned(),
        source_lock_fingerprints: vec![format!("sha256:{}", "1".repeat(64))],
        bindings: vec![GgufTensorBinding {
            logical_tensor: "weight".to_owned(),
            value_tensor: "weight.fp8".to_owned(),
            encoding: GgufRecipeEncoding::Fp8E4m3fnChannelBf16Scale,
            role: "attention-weight".to_owned(),
            logical_shape: vec![4],
            scope: GgufTensorScope::Consumed,
            scales: vec![GgufScaleBinding {
                tensor: "weight.scale".to_owned(),
                role: GgufScaleRole::Channel,
            }],
        }],
        logical_shapes: vec![],
        static_fp8_kv: vec![],
        known_unconsumed_tensors: vec![],
    }
}

fn fp8_file(recipe: &GgufTensorRecipeV1, digest: String) -> Vec<u8> {
    let mut metadata = base_metadata();
    metadata.extend([
        (SLLM_EXTENSION_VERSION_KEY.to_owned(), TestValue::U32(1)),
        (
            SLLM_TENSOR_RECIPE_KEY.to_owned(),
            TestValue::String(recipe.canonical_json().expect("canonical recipe")),
        ),
        (
            SLLM_TENSOR_RECIPE_SHA256_KEY.to_owned(),
            TestValue::String(digest),
        ),
    ]);
    build(
        metadata,
        vec![
            tensor("weight.fp8", GgufTensorType::I8Carrier, &[4], 0),
            tensor("weight.scale", GgufTensorType::Bf16, &[4], 32),
        ],
    )
}

#[test]
fn accepts_versioned_fp8_i8_carrier_and_scale_binding() {
    let recipe = fp8_recipe();
    let verified =
        open_bytes(&fp8_file(&recipe, recipe.digest().expect("digest"))).expect("valid extension");
    let extension = verified.extension().expect("extension");
    assert_eq!(extension.recipe, recipe);
    assert_eq!(
        extension.recipe_sha256,
        extension.recipe.digest().expect("digest")
    );
}

#[test]
fn rejects_i8_without_extension_and_extension_drift() {
    assert_invalid(
        build(
            base_metadata(),
            vec![tensor("weight", GgufTensorType::I8Carrier, &[4], 0)],
        ),
        "I8 carrier requires",
    );

    let recipe = fp8_recipe();
    assert_invalid(
        fp8_file(&recipe, format!("sha256:{}", "0".repeat(64))),
        "tensor recipe digest differs",
    );

    let mut missing_scale = recipe.clone();
    missing_scale.bindings[0].scales.clear();
    assert_invalid(
        fp8_file(
            &missing_scale,
            missing_scale.digest().expect("missing-scale digest"),
        ),
        "requires exactly one channel scale",
    );
}

#[cfg(unix)]
#[test]
fn payload_reads_keep_the_verified_open_descriptor() {
    let bytes = build(
        base_metadata(),
        vec![tensor("weight", GgufTensorType::Bf16, &[16], 0)],
    );
    let path = temp_file(&bytes);
    let verified = VerifiedGguf::open(&path).expect("verify original file");
    fs::remove_file(&path).expect("unlink original path");
    fs::write(&path, vec![0_u8; bytes.len()]).expect("replace path contents");

    assert_eq!(
        verified
            .read_tensor_range("weight", 0, 4)
            .expect("descriptor read"),
        [0, 1, 2, 3]
    );
    fs::remove_file(path).expect("remove replacement");
}

#[test]
fn deterministic_writer_round_trips_frontend_assets_and_derived_lock() {
    let recipe = GgufTensorRecipeV1 {
        schema_version: "sllm-gguf-tensor-recipe-v1".to_owned(),
        semantic_model_id: "qwen35-writer-test".to_owned(),
        source_lock_fingerprints: vec![format!("sha256:{}", "2".repeat(64))],
        bindings: vec![],
        logical_shapes: vec![],
        static_fp8_kv: vec![],
        known_unconsumed_tensors: vec![],
    };
    let config = br#"{"model_type":"qwen3_5"}"#;
    let tokenizer = br#"{"version":"1.0"}"#;
    let tokenizer_config = br#"{"eos_token":"<eos>"}"#;
    let generation_config = br#"{"do_sample":false}"#;
    let hf_quant_config = br#"{"quant_method":"modelopt"}"#;
    let mut metadata = BTreeMap::from([
        (
            "general.architecture".to_owned(),
            GgufValue::String("qwen35".to_owned()),
        ),
        ("general.alignment".to_owned(), GgufValue::U32(32)),
        (
            "tokenizer.ggml.tokens".to_owned(),
            GgufValue::Array(GgufArray::String(vec!["a".to_owned(), "b".to_owned()])),
        ),
        (SLLM_EXTENSION_VERSION_KEY.to_owned(), GgufValue::U32(1)),
        (
            SLLM_TENSOR_RECIPE_KEY.to_owned(),
            GgufValue::String(recipe.canonical_json().expect("recipe")),
        ),
        (
            SLLM_TENSOR_RECIPE_SHA256_KEY.to_owned(),
            GgufValue::String(recipe.digest().expect("recipe digest")),
        ),
        (
            "sllm.source.artifact.fingerprint".to_owned(),
            GgufValue::String(format!("sha256:{}", "3".repeat(64))),
        ),
        (
            "sllm.source.semantic.repository".to_owned(),
            GgufValue::String("example/semantic-model".to_owned()),
        ),
        (
            "sllm.source.semantic.revision".to_owned(),
            GgufValue::String("4".repeat(40)),
        ),
        (
            "sllm.source.recipe.producer".to_owned(),
            GgufValue::String("modelopt@test".to_owned()),
        ),
        (
            "sllm.kv.fp8.scheme".to_owned(),
            GgufValue::String("implicit-unit-test".to_owned()),
        ),
        (
            "sllm.kv.fp8.implicit_decode_scale_bf16".to_owned(),
            GgufValue::U16(0x3f80),
        ),
    ]);
    for (key, value) in [
        (SLLM_FRONTEND_CONFIG_KEY, config.as_slice()),
        (SLLM_FRONTEND_TOKENIZER_KEY, tokenizer.as_slice()),
        (
            SLLM_FRONTEND_TOKENIZER_CONFIG_KEY,
            tokenizer_config.as_slice(),
        ),
        (
            "sllm.frontend.generation_config_json",
            generation_config.as_slice(),
        ),
        (
            "sllm.source.hf_quant_config_json",
            hf_quant_config.as_slice(),
        ),
    ] {
        metadata.insert(
            key.to_owned(),
            GgufValue::String(String::from_utf8(value.to_vec()).expect("UTF-8 fixture")),
        );
        metadata.insert(format!("{key}.sha256"), GgufValue::String(sha256(value)));
    }
    let plan = GgufWritePlan {
        metadata,
        tensors: vec![
            GgufWriteTensor {
                name: "z.weight".to_owned(),
                source_name: "z".to_owned(),
                dimensions: vec![3, 5],
                tensor_type: GgufTensorType::Bf16,
            },
            GgufWriteTensor {
                name: "a.weight".to_owned(),
                source_name: "a".to_owned(),
                dimensions: vec![7],
                tensor_type: GgufTensorType::F32,
            },
        ],
    };
    let sources = BTreeMap::from([
        ("z".to_owned(), (0..30_u8).collect::<Vec<_>>()),
        ("a".to_owned(), (100..128_u8).collect::<Vec<_>>()),
    ]);
    let first_path = temp_output("writer-first");
    let second_path = temp_output("writer-second");
    let write = |path: &PathBuf| {
        write_gguf(path, &plan, |name, offset, length| {
            let source = sources
                .get(name)
                .ok_or_else(|| GgufError::Invalid("unknown test source".to_owned()))?;
            let start = offset as usize;
            Ok(source[start..start + length].to_vec())
        })
        .expect("write deterministic GGUF")
    };
    let first = write(&first_path);
    let second = write(&second_path);
    assert_eq!(first.sha256, second.sha256);
    assert_eq!(first.size_bytes, second.size_bytes);
    assert_eq!(
        fs::read(&first_path).expect("first bytes"),
        fs::read(&second_path).expect("second bytes")
    );

    let verified = VerifiedGguf::open(&first_path).expect("read written GGUF");
    assert_eq!(verified.tensors()[0].name, "a.weight");
    assert_eq!(
        verified
            .read_tensor_range("z.weight", 0, 4)
            .expect("payload"),
        [0, 1, 2, 3]
    );
    assert_eq!(
        verified.frontend_asset("tokenizer.json"),
        Some(tokenizer.as_slice())
    );
    assert_eq!(
        verified.frontend_asset("generation_config.json"),
        Some(generation_config.as_slice())
    );
    assert_eq!(
        verified.frontend_asset("hf_quant_config.json"),
        Some(hf_quant_config.as_slice())
    );
    assert_eq!(
        verified.metadata_value("sllm.kv.fp8.implicit_decode_scale_bf16"),
        Some(&GgufValue::U16(0x3f80))
    );

    let lock = DerivedGgufLock::new(
        "qwen35-writer-test".to_owned(),
        recipe.source_lock_fingerprints.clone(),
        DerivedGgufConverter {
            repository: "https://github.com/jyohukuchan/sLLM".to_owned(),
            commit: "3".repeat(40),
            arguments: vec!["convert".to_owned(), "fixture".to_owned()],
            effective_config: BTreeMap::from([("alignment".to_owned(), "32".to_owned())]),
            environment: BTreeMap::from([("rust".to_owned(), "test".to_owned())]),
        },
        &first,
    )
    .expect("derived lock");
    let lock_bytes = lock.canonical_json().expect("lock JSON");
    let reparsed = DerivedGgufLock::parse(&lock_bytes).expect("parse canonical lock");
    verify_derived_gguf(reparsed.clone(), &first_path).expect("verify derived GGUF");

    let tensor = verified.tensor("z.weight").expect("tensor");
    let mut bytes = fs::read(&first_path).expect("read for tamper");
    bytes[tensor.absolute_range[0] as usize] ^= 0xff;
    fs::write(&first_path, bytes).expect("tamper payload");
    assert!(verify_derived_gguf(reparsed, &first_path).is_err());

    fs::remove_file(first_path).expect("remove first");
    fs::remove_file(second_path).expect("remove second");
}
