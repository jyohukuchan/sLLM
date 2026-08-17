//! Source-importer to GGUF conversion plans.

use crate::gguf::{
    GGUF_ALIGNMENT, GgufArray, GgufError, GgufRecipeEncoding, GgufScaleBinding, GgufScaleRole,
    GgufStaticFp8KvBinding, GgufTensorBinding, GgufTensorRecipeV1, GgufTensorScope, GgufTensorType,
    GgufValue, SLLM_EXTENSION_VERSION_KEY, SLLM_FRONTEND_CONFIG_KEY,
    SLLM_FRONTEND_PREPROCESSOR_CONFIG_KEY, SLLM_FRONTEND_TOKENIZER_CONFIG_KEY,
    SLLM_FRONTEND_TOKENIZER_KEY, SLLM_TENSOR_RECIPE_KEY, SLLM_TENSOR_RECIPE_SHA256_KEY,
};
use crate::gguf_writer::{GgufWritePlan, GgufWriteReport, GgufWriteTensor, write_gguf};
use crate::{
    FrontendAssetKind, Gemma4ModelLock, ModelLock, QWEN35_MOE_LICENSE,
    QWEN35_MOE_MODEL_FINGERPRINT, QWEN35_MOE_REPOSITORY, QWEN35_MOE_REVISION, QuantizedScalePlane,
    QuantizedTensorDescriptor, QuantizedTensorEncoding, QuantizedTensorRole, Qwen35MoeExpertTensor,
    ScalePlaneRole, TensorDType, UNSLOTH_GEMMA4_NVFP4_MODEL_SHA256,
    UNSLOTH_GEMMA4_NVFP4_REPOSITORY, UNSLOTH_GEMMA4_NVFP4_REVISION, VerifiedCache,
    VerifiedFp8Sidecar, VerifiedQwen35Moe, VerifiedUnslothGemma4Nvfp4, qwen35_reviewed_spec,
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::Path;

const MXFP4_BLOCK_ELEMENTS: usize = 32;
const MXFP4_BLOCK_BYTES: usize = 17;
const NVFP4_SUBBLOCK_ELEMENTS: usize = 16;
const NVFP4_BLOCK_ELEMENTS: usize = 64;
const NVFP4_BLOCK_BYTES: usize = 36;

/// Repack row-major adjacent-pair MXFP4 values and one E8M0 scale per block
/// into the standard GGML `block_mxfp4` byte layout without changing a code.
pub fn repack_mxfp4_standard(
    packed_values: &[u8],
    scales: &[u8],
    rows: usize,
    columns: usize,
) -> Result<Vec<u8>, GgufError> {
    validate_lowbit_shape(rows, columns, MXFP4_BLOCK_ELEMENTS, packed_values, scales)?;
    let blocks_per_row = columns / MXFP4_BLOCK_ELEMENTS;
    let block_count = rows
        .checked_mul(blocks_per_row)
        .ok_or_else(|| invalid("MXFP4 block count overflows"))?;
    let mut output = Vec::with_capacity(
        block_count
            .checked_mul(MXFP4_BLOCK_BYTES)
            .ok_or_else(|| invalid("MXFP4 output length overflows"))?,
    );
    for block in 0..block_count {
        output.push(scales[block]);
        let source = &packed_values[block * 16..(block + 1) * 16];
        repack_adjacent_nibbles(source, &mut output, MXFP4_BLOCK_ELEMENTS);
    }
    Ok(output)
}

/// Repack four adjacent-pair NVFP4 block-16 planes into one standard GGML
/// `block_nvfp4` (four UE4M3 scales followed by 32 E2M1 bytes). The mandatory
/// FP32 outer scale remains a separately bound recipe tensor.
pub fn repack_nvfp4_standard(
    packed_values: &[u8],
    block_scales: &[u8],
    rows: usize,
    columns: usize,
) -> Result<Vec<u8>, GgufError> {
    validate_lowbit_shape(
        rows,
        columns,
        NVFP4_BLOCK_ELEMENTS,
        packed_values,
        block_scales,
    )?;
    let blocks_per_row = columns / NVFP4_BLOCK_ELEMENTS;
    let block_count = rows
        .checked_mul(blocks_per_row)
        .ok_or_else(|| invalid("NVFP4 block count overflows"))?;
    let mut output = Vec::with_capacity(
        block_count
            .checked_mul(NVFP4_BLOCK_BYTES)
            .ok_or_else(|| invalid("NVFP4 output length overflows"))?,
    );
    for block in 0..block_count {
        let scale_start = block * 4;
        output.extend_from_slice(&block_scales[scale_start..scale_start + 4]);
        let value_start = block * 32;
        for subblock in 0..4 {
            let start = value_start + subblock * 8;
            repack_adjacent_nibbles(
                &packed_values[start..start + 8],
                &mut output,
                NVFP4_SUBBLOCK_ELEMENTS,
            );
        }
    }
    Ok(output)
}

fn validate_lowbit_shape(
    rows: usize,
    columns: usize,
    standard_block: usize,
    packed_values: &[u8],
    scales: &[u8],
) -> Result<(), GgufError> {
    if rows == 0 || columns == 0 || columns % standard_block != 0 {
        return Err(invalid(format!(
            "low-bit shape must be nonzero and K divisible by {standard_block}"
        )));
    }
    let elements = rows
        .checked_mul(columns)
        .ok_or_else(|| invalid("low-bit element count overflows"))?;
    if packed_values.len() != elements / 2 {
        return Err(invalid("low-bit packed value length differs"));
    }
    let scale_count = if standard_block == MXFP4_BLOCK_ELEMENTS {
        elements / MXFP4_BLOCK_ELEMENTS
    } else {
        elements / NVFP4_SUBBLOCK_ELEMENTS
    };
    if scales.len() != scale_count {
        return Err(invalid("low-bit scale length differs"));
    }
    Ok(())
}

fn repack_adjacent_nibbles(source: &[u8], output: &mut Vec<u8>, elements: usize) {
    let half = elements / 2;
    for index in 0..half {
        let low = source_code(source, index);
        let high = source_code(source, index + half);
        output.push(low | high << 4);
    }
}

fn source_code(source: &[u8], index: usize) -> u8 {
    let byte = source[index / 2];
    if index & 1 == 0 {
        byte & 0x0f
    } else {
        byte >> 4
    }
}

pub fn build_qwen35_bf16_gguf_plan(
    lock: &ModelLock,
    cache: &VerifiedCache,
) -> Result<GgufWritePlan, GgufError> {
    if cache.lock_fingerprint != lock.fingerprint() {
        return Err(invalid(
            "verified cache fingerprint differs from model lock",
        ));
    }
    let spec = qwen35_reviewed_spec(&lock.model.repo_id)
        .filter(|spec| {
            spec.revision == lock.model.resolved_revision && spec.fingerprint == lock.fingerprint()
        })
        .ok_or_else(|| invalid("model is not a reviewed Qwen3.5 identity"))?;
    let config = read_asset(cache, FrontendAssetKind::ConfigJson)?;
    let tokenizer = read_asset(cache, FrontendAssetKind::TokenizerJson)?;
    let tokenizer_config = read_asset(cache, FrontendAssetKind::TokenizerConfigJson)?;
    let preprocessor = read_asset(cache, FrontendAssetKind::PreprocessorConfigJson)?;
    let chat_template = read_asset(cache, FrontendAssetKind::ChatTemplateJinja)?;
    let mut metadata = qwen_metadata(lock, spec, &tokenizer, &chat_template)?;
    let known_unconsumed_tensors = cache
        .tensors()
        .filter(|descriptor| {
            descriptor.tensor_name.starts_with("model.visual.")
                || descriptor.tensor_name.starts_with("mtp.")
        })
        .map(|descriptor| descriptor.tensor_name.clone())
        .collect();
    let recipe = GgufTensorRecipeV1 {
        schema_version: "sllm-gguf-tensor-recipe-v1".to_owned(),
        semantic_model_id: format!("qwen35:{}", lock.fingerprint()),
        source_lock_fingerprints: vec![lock.fingerprint().to_owned()],
        bindings: vec![],
        static_fp8_kv: vec![],
        known_unconsumed_tensors,
    };
    metadata.insert(SLLM_EXTENSION_VERSION_KEY.to_owned(), GgufValue::U32(1));
    metadata.insert(
        SLLM_TENSOR_RECIPE_KEY.to_owned(),
        GgufValue::String(recipe.canonical_json()?),
    );
    metadata.insert(
        SLLM_TENSOR_RECIPE_SHA256_KEY.to_owned(),
        GgufValue::String(recipe.digest()?),
    );
    for (key, bytes) in [
        (SLLM_FRONTEND_CONFIG_KEY, config),
        (SLLM_FRONTEND_TOKENIZER_KEY, tokenizer),
        (SLLM_FRONTEND_TOKENIZER_CONFIG_KEY, tokenizer_config),
        (SLLM_FRONTEND_PREPROCESSOR_CONFIG_KEY, preprocessor),
    ] {
        let text = String::from_utf8(bytes)
            .map_err(|_| invalid(format!("frontend asset {key} is not UTF-8")))?;
        metadata.insert(key.to_owned(), GgufValue::String(text.clone()));
        metadata.insert(
            format!("{key}.sha256"),
            GgufValue::String(sha256(text.as_bytes())),
        );
    }

    let mut tensors = Vec::new();
    for descriptor in cache.tensors() {
        let tensor_type = match descriptor.dtype {
            TensorDType::Bf16 => GgufTensorType::Bf16,
            TensorDType::F16 => GgufTensorType::F16,
            TensorDType::F32 => GgufTensorType::F32,
            dtype => {
                return Err(invalid(format!(
                    "tensor {} has unsupported BF16-converter dtype {dtype:?}",
                    descriptor.tensor_name
                )));
            }
        };
        let mut dimensions = descriptor.shape.clone();
        dimensions.reverse();
        let tensor = GgufWriteTensor {
            name: descriptor.tensor_name.clone(),
            source_name: descriptor.tensor_name.clone(),
            dimensions,
            tensor_type,
        };
        if tensor.byte_length()? != descriptor.byte_size {
            return Err(invalid(format!(
                "tensor {} GGUF byte length differs from source",
                descriptor.tensor_name
            )));
        }
        tensors.push(tensor);
    }
    if tensors.len() as u64 != lock.model.tensor_contract.indexed_tensor_count {
        return Err(invalid(
            "Qwen GGUF tensor inventory differs from model lock",
        ));
    }
    Ok(GgufWritePlan { metadata, tensors })
}

pub fn write_qwen35_bf16_gguf(
    lock: &ModelLock,
    cache: &VerifiedCache,
    output_path: impl AsRef<Path>,
) -> Result<GgufWriteReport, GgufError> {
    let plan = build_qwen35_bf16_gguf_plan(lock, cache)?;
    write_gguf(output_path, &plan, |tensor, offset, length| {
        cache
            .read_tensor_range(tensor, offset, length)
            .map_err(|error| invalid(error.to_string()))
    })
}

pub fn build_qwen35_fp8_gguf_plan(
    lock: &ModelLock,
    cache: &VerifiedCache,
    sidecar: &VerifiedFp8Sidecar,
) -> Result<GgufWritePlan, GgufError> {
    if sidecar.source_lock_fingerprint() != lock.fingerprint() {
        return Err(invalid("FP8 sidecar source identity differs"));
    }
    let mut plan = build_qwen35_bf16_gguf_plan(lock, cache)?;
    let mut bindings = Vec::with_capacity(sidecar.tensors().len());
    for tensor in sidecar.tensors() {
        let value = plan
            .tensors
            .iter_mut()
            .find(|candidate| candidate.name == tensor.name)
            .ok_or_else(|| invalid(format!("FP8 source tensor is absent: {}", tensor.name)))?;
        if value.tensor_type != GgufTensorType::Bf16
            || value.dimensions != [tensor.shape[1], tensor.shape[0]]
        {
            return Err(invalid(format!(
                "FP8 tensor shape/type differs from BF16 source: {}",
                tensor.name
            )));
        }
        value.tensor_type = GgufTensorType::I8Carrier;
        value.source_name = format!("qwen-fp8-value::{}", tensor.name);
        let scale_name = format!("{}.sllm.scale.channel", tensor.name);
        plan.tensors.push(GgufWriteTensor {
            name: scale_name.clone(),
            source_name: format!("qwen-fp8-scale::{}", tensor.name),
            dimensions: vec![tensor.shape[0]],
            tensor_type: GgufTensorType::F32,
        });
        bindings.push(GgufTensorBinding {
            logical_tensor: tensor.name.clone(),
            value_tensor: tensor.name.clone(),
            encoding: GgufRecipeEncoding::Fp8E4m3fnChannelF32Scale,
            role: "text-linear-weight".to_owned(),
            logical_shape: tensor.shape.to_vec(),
            scope: GgufTensorScope::Consumed,
            scales: vec![GgufScaleBinding {
                tensor: scale_name,
                role: GgufScaleRole::Channel,
            }],
        });
    }
    let known_unconsumed_tensors = plan
        .tensors
        .iter()
        .filter(|tensor| {
            tensor.name.starts_with("model.visual.") || tensor.name.starts_with("mtp.")
        })
        .map(|tensor| tensor.name.clone())
        .collect();
    let recipe = GgufTensorRecipeV1 {
        schema_version: "sllm-gguf-tensor-recipe-v1".to_owned(),
        semantic_model_id: format!("qwen35:{}", lock.fingerprint()),
        source_lock_fingerprints: vec![lock.fingerprint().to_owned()],
        bindings,
        static_fp8_kv: vec![],
        known_unconsumed_tensors,
    };
    insert_recipe_metadata(&mut plan.metadata, &recipe)?;
    plan.metadata.insert(
        "sllm.source.fp8_manifest_fingerprint".to_owned(),
        GgufValue::String(sidecar.manifest_fingerprint().to_owned()),
    );
    plan.metadata.insert(
        "sllm.source.fp8_artifact.sha256".to_owned(),
        GgufValue::String(sidecar.artifact_sha256().to_owned()),
    );
    Ok(plan)
}

pub fn write_qwen35_fp8_gguf(
    lock: &ModelLock,
    cache: &VerifiedCache,
    sidecar: &VerifiedFp8Sidecar,
    output_path: impl AsRef<Path>,
) -> Result<GgufWriteReport, GgufError> {
    let plan = build_qwen35_fp8_gguf_plan(lock, cache, sidecar)?;
    write_gguf(output_path, &plan, |source, offset, length| {
        if let Some(name) = source.strip_prefix("qwen-fp8-value::") {
            return read_fp8_sidecar_range(sidecar, name, false, offset, length);
        }
        if let Some(name) = source.strip_prefix("qwen-fp8-scale::") {
            return read_fp8_sidecar_range(sidecar, name, true, offset, length);
        }
        cache
            .read_tensor_range(source, offset, length)
            .map_err(|error| invalid(error.to_string()))
    })
}

pub fn build_gemma4_nvfp4_gguf_plan(
    lock: &Gemma4ModelLock,
    artifact: &VerifiedUnslothGemma4Nvfp4,
) -> Result<GgufWritePlan, GgufError> {
    if !crate::gemma4::is_reviewed_gemma4_identity(lock) || !lock.supports_chat_messages() {
        return Err(invalid("Gemma GGUF requires the reviewed instruction lock"));
    }
    let mut tensors = Vec::new();
    let mut bindings = Vec::new();
    let mut known_unconsumed_tensors = Vec::new();
    for descriptor in artifact.tensors() {
        let mut dimensions = descriptor.logical_shape.clone();
        dimensions.reverse();
        match descriptor.encoding {
            QuantizedTensorEncoding::UnquantizedBf16 => {
                tensors.push(GgufWriteTensor {
                    name: descriptor.logical_name.clone(),
                    source_name: format!("gemma-direct::{}", descriptor.logical_name),
                    dimensions,
                    tensor_type: GgufTensorType::Bf16,
                });
                if descriptor.role == QuantizedTensorRole::KnownUnconsumed {
                    known_unconsumed_tensors.push(descriptor.logical_name.clone());
                }
            }
            QuantizedTensorEncoding::OcpFp8E4M3FnChannelBf16Scale => {
                let scale = require_scale(descriptor, ScalePlaneRole::WeightChannel)?;
                let scale_name = format!("{}.sllm.scale.channel", descriptor.logical_name);
                tensors.push(GgufWriteTensor {
                    name: descriptor.logical_name.clone(),
                    source_name: format!("gemma-fp8::{}", descriptor.logical_name),
                    dimensions,
                    tensor_type: GgufTensorType::I8Carrier,
                });
                let mut scale_dimensions = scale.shape.clone();
                scale_dimensions.reverse();
                tensors.push(GgufWriteTensor {
                    name: scale_name.clone(),
                    source_name: format!("gemma-scale-channel::{}", descriptor.logical_name),
                    dimensions: scale_dimensions,
                    tensor_type: GgufTensorType::Bf16,
                });
                bindings.push(GgufTensorBinding {
                    logical_tensor: descriptor.logical_name.clone(),
                    value_tensor: descriptor.logical_name.clone(),
                    encoding: GgufRecipeEncoding::Fp8E4m3fnChannelBf16Scale,
                    role: quantized_role(descriptor.role).to_owned(),
                    logical_shape: descriptor.logical_shape.clone(),
                    scope: scope(descriptor.role),
                    scales: vec![GgufScaleBinding {
                        tensor: scale_name,
                        role: GgufScaleRole::Channel,
                    }],
                });
            }
            QuantizedTensorEncoding::Nvfp4E2M1Block16E4M3FnF32Outer => {
                if dimensions.first().copied().unwrap_or(0) % NVFP4_BLOCK_ELEMENTS as u64 != 0 {
                    return Err(invalid(format!(
                        "NVFP4 tensor {} K dimension is not divisible by 64",
                        descriptor.logical_name
                    )));
                }
                let outer = require_scale(descriptor, ScalePlaneRole::WeightOuter)?;
                let input = require_scale(descriptor, ScalePlaneRole::InputOuter)?;
                let outer_name = format!("{}.sllm.scale.outer", descriptor.logical_name);
                let input_name = format!("{}.sllm.scale.input", descriptor.logical_name);
                tensors.push(GgufWriteTensor {
                    name: descriptor.logical_name.clone(),
                    source_name: format!("gemma-nvfp4::{}", descriptor.logical_name),
                    dimensions,
                    tensor_type: GgufTensorType::Nvfp4,
                });
                for (name, source, plane) in [
                    (&outer_name, "gemma-scale-outer", outer),
                    (&input_name, "gemma-scale-input", input),
                ] {
                    tensors.push(GgufWriteTensor {
                        name: name.clone(),
                        source_name: format!("{source}::{}", descriptor.logical_name),
                        dimensions: plane.shape.clone(),
                        tensor_type: GgufTensorType::F32,
                    });
                }
                bindings.push(GgufTensorBinding {
                    logical_tensor: descriptor.logical_name.clone(),
                    value_tensor: descriptor.logical_name.clone(),
                    encoding: GgufRecipeEncoding::Nvfp4E2m1Block16E4m3fnF32Outer,
                    role: quantized_role(descriptor.role).to_owned(),
                    logical_shape: descriptor.logical_shape.clone(),
                    scope: scope(descriptor.role),
                    scales: vec![
                        GgufScaleBinding {
                            tensor: outer_name,
                            role: GgufScaleRole::Outer,
                        },
                        GgufScaleBinding {
                            tensor: input_name,
                            role: GgufScaleRole::Input,
                        },
                    ],
                });
            }
            encoding => {
                return Err(invalid(format!(
                    "Gemma converter does not support {encoding:?}"
                )));
            }
        }
    }
    let static_fp8_kv = (0..48)
        .map(|layer| {
            let scale = artifact
                .kv_scale(layer)
                .ok_or_else(|| invalid(format!("missing static FP8 KV layer {layer}")))?;
            Ok(GgufStaticFp8KvBinding {
                layer,
                key_decode_scale_bf16: scale.key_decode_scale_bf16,
                value_decode_scale_bf16: scale.value_decode_scale_bf16,
            })
        })
        .collect::<Result<Vec<_>, GgufError>>()?;
    let recipe = GgufTensorRecipeV1 {
        schema_version: "sllm-gguf-tensor-recipe-v1".to_owned(),
        semantic_model_id: format!("gemma4:{}", lock.fingerprint()),
        source_lock_fingerprints: vec![
            lock.fingerprint().to_owned(),
            format!("sha256:{UNSLOTH_GEMMA4_NVFP4_MODEL_SHA256}"),
        ],
        bindings,
        static_fp8_kv,
        known_unconsumed_tensors,
    };
    let mut metadata = BTreeMap::from([
        (
            "general.architecture".to_owned(),
            GgufValue::String("gemma4".to_owned()),
        ),
        (
            "general.alignment".to_owned(),
            GgufValue::U32(GGUF_ALIGNMENT as u32),
        ),
        (
            "general.name".to_owned(),
            GgufValue::String(UNSLOTH_GEMMA4_NVFP4_REPOSITORY.to_owned()),
        ),
        (
            "general.source.url".to_owned(),
            GgufValue::String(UNSLOTH_GEMMA4_NVFP4_REVISION.to_owned()),
        ),
        (
            "general.license".to_owned(),
            GgufValue::String("gemma".to_owned()),
        ),
        (
            "sllm.source.recipe.sha256".to_owned(),
            GgufValue::String(artifact.recipe_digest().to_owned()),
        ),
    ]);
    insert_recipe_metadata(&mut metadata, &recipe)?;
    for (key, kind) in [
        (SLLM_FRONTEND_CONFIG_KEY, FrontendAssetKind::ConfigJson),
        (
            SLLM_FRONTEND_TOKENIZER_KEY,
            FrontendAssetKind::TokenizerJson,
        ),
        (
            SLLM_FRONTEND_TOKENIZER_CONFIG_KEY,
            FrontendAssetKind::TokenizerConfigJson,
        ),
        (
            SLLM_FRONTEND_PREPROCESSOR_CONFIG_KEY,
            FrontendAssetKind::PreprocessorConfigJson,
        ),
    ] {
        insert_frontend_asset(
            &mut metadata,
            key,
            artifact
                .read_frontend_asset(kind)
                .map_err(|error| invalid(error.to_string()))?,
        )?;
    }
    Ok(GgufWritePlan { metadata, tensors })
}

pub fn write_gemma4_nvfp4_gguf(
    lock: &Gemma4ModelLock,
    artifact: &VerifiedUnslothGemma4Nvfp4,
    output_path: impl AsRef<Path>,
) -> Result<GgufWriteReport, GgufError> {
    let plan = build_gemma4_nvfp4_gguf_plan(lock, artifact)?;
    write_gguf(output_path, &plan, |source, offset, length| {
        read_gemma_source(artifact, source, offset, length)
    })
}

pub fn build_qwen35_moe_mxfp4_gguf_plan(
    artifact: &VerifiedQwen35Moe,
) -> Result<GgufWritePlan, GgufError> {
    if artifact.recipe().encoding != QuantizedTensorEncoding::Mxfp4E2M1Block32E8M0
        || artifact.recipe().group_size != MXFP4_BLOCK_ELEMENTS as u32
    {
        return Err(invalid("Qwen MoE MXFP4 recipe differs"));
    }
    let expert_values: BTreeMap<&str, &Qwen35MoeExpertTensor> = artifact
        .experts()
        .map(|expert| (expert.value.source_name.as_str(), expert))
        .collect();
    let expert_scales: std::collections::BTreeSet<&str> = artifact
        .experts()
        .map(|expert| expert.scale.source_name.as_str())
        .collect();
    let mut tensors = Vec::new();
    let mut bindings = Vec::with_capacity(expert_values.len());
    let mut known_unconsumed_tensors = Vec::new();
    for plane in artifact.all_planes() {
        if expert_scales.contains(plane.source_name.as_str()) {
            continue;
        }
        if let Some(expert) = expert_values.get(plane.source_name.as_str()) {
            let mut dimensions = expert.logical_shape.to_vec();
            dimensions.reverse();
            tensors.push(GgufWriteTensor {
                name: expert.value.source_name.clone(),
                source_name: format!("moe-mxfp4::{}", expert.value.source_name),
                dimensions,
                tensor_type: GgufTensorType::Mxfp4,
            });
            bindings.push(GgufTensorBinding {
                logical_tensor: expert.value.source_name.clone(),
                value_tensor: expert.value.source_name.clone(),
                encoding: GgufRecipeEncoding::Mxfp4E2m1Block32E8m0,
                role: "routed-expert-projection".to_owned(),
                logical_shape: expert.logical_shape.to_vec(),
                scope: GgufTensorScope::Consumed,
                scales: vec![],
            });
            continue;
        }
        let tensor_type = match plane.dtype.as_str() {
            "BF16" => GgufTensorType::Bf16,
            "F16" => GgufTensorType::F16,
            "F32" => GgufTensorType::F32,
            dtype => {
                return Err(invalid(format!(
                    "non-expert MoE tensor {} has unsupported dtype {dtype}",
                    plane.source_name
                )));
            }
        };
        let mut dimensions = plane.shape.clone();
        dimensions.reverse();
        tensors.push(GgufWriteTensor {
            name: plane.source_name.clone(),
            source_name: format!("moe-direct::{}", plane.source_name),
            dimensions,
            tensor_type,
        });
        if plane.source_name.starts_with("model.visual.") || plane.source_name.starts_with("mtp.") {
            known_unconsumed_tensors.push(plane.source_name.clone());
        }
    }
    if bindings.len() != artifact.experts().len() {
        return Err(invalid("Qwen MoE expert binding count differs"));
    }
    let recipe = GgufTensorRecipeV1 {
        schema_version: "sllm-gguf-tensor-recipe-v1".to_owned(),
        semantic_model_id: format!("qwen35moe:{QWEN35_MOE_MODEL_FINGERPRINT}"),
        source_lock_fingerprints: vec![QWEN35_MOE_MODEL_FINGERPRINT.to_owned()],
        bindings,
        static_fp8_kv: vec![],
        known_unconsumed_tensors,
    };
    let mut metadata = BTreeMap::from([
        (
            "general.architecture".to_owned(),
            GgufValue::String("qwen35moe".to_owned()),
        ),
        (
            "general.alignment".to_owned(),
            GgufValue::U32(GGUF_ALIGNMENT as u32),
        ),
        (
            "general.name".to_owned(),
            GgufValue::String(QWEN35_MOE_REPOSITORY.to_owned()),
        ),
        (
            "general.source.url".to_owned(),
            GgufValue::String(QWEN35_MOE_REVISION.to_owned()),
        ),
        (
            "general.license".to_owned(),
            GgufValue::String(QWEN35_MOE_LICENSE.to_owned()),
        ),
        (
            "qwen35moe.block_count".to_owned(),
            GgufValue::U32(artifact.config().layer_count),
        ),
        (
            "qwen35moe.embedding_length".to_owned(),
            GgufValue::U32(artifact.config().hidden_size),
        ),
        (
            "qwen35moe.expert_count".to_owned(),
            GgufValue::U32(artifact.config().expert_count),
        ),
        (
            "qwen35moe.expert_used_count".to_owned(),
            GgufValue::U32(artifact.config().selected_expert_count),
        ),
    ]);
    insert_recipe_metadata(&mut metadata, &recipe)?;
    for (key, name) in [
        (SLLM_FRONTEND_CONFIG_KEY, "config.json"),
        (SLLM_FRONTEND_TOKENIZER_KEY, "tokenizer.json"),
        (SLLM_FRONTEND_TOKENIZER_CONFIG_KEY, "tokenizer_config.json"),
    ] {
        insert_frontend_asset(
            &mut metadata,
            key,
            artifact
                .read_support_file(name)
                .map_err(|error| invalid(error.to_string()))?,
        )?;
    }
    let chat = artifact
        .read_support_file("chat_template.jinja")
        .map_err(|error| invalid(error.to_string()))?;
    let chat = String::from_utf8(chat).map_err(|_| invalid("MoE chat template is not UTF-8"))?;
    metadata.insert(
        "tokenizer.chat_template".to_owned(),
        GgufValue::String(chat),
    );
    Ok(GgufWritePlan { metadata, tensors })
}

pub fn write_qwen35_moe_mxfp4_gguf(
    artifact: &VerifiedQwen35Moe,
    output_path: impl AsRef<Path>,
) -> Result<GgufWriteReport, GgufError> {
    let plan = build_qwen35_moe_mxfp4_gguf_plan(artifact)?;
    let experts: BTreeMap<&str, &Qwen35MoeExpertTensor> = artifact
        .experts()
        .map(|expert| (expert.value.source_name.as_str(), expert))
        .collect();
    write_gguf(output_path, &plan, |source, offset, length| {
        let (kind, name) = source
            .split_once("::")
            .ok_or_else(|| invalid("malformed MoE GGUF source"))?;
        match kind {
            "moe-direct" => {
                let plane = artifact
                    .any_plane(name)
                    .ok_or_else(|| invalid(format!("missing MoE tensor {name}")))?;
                artifact
                    .read_plane_range(plane, offset, length)
                    .map_err(|error| invalid(error.to_string()))
            }
            "moe-mxfp4" => read_moe_mxfp4_range(
                artifact,
                experts
                    .get(name)
                    .copied()
                    .ok_or_else(|| invalid(format!("missing MoE expert {name}")))?,
                offset,
                length,
            ),
            _ => Err(invalid(format!("unknown MoE GGUF source {kind}"))),
        }
    })
}

fn read_gemma_source(
    artifact: &VerifiedUnslothGemma4Nvfp4,
    source: &str,
    offset: u64,
    length: usize,
) -> Result<Vec<u8>, GgufError> {
    let (kind, logical) = source
        .split_once("::")
        .ok_or_else(|| invalid("malformed Gemma GGUF source"))?;
    let descriptor = artifact
        .tensor(logical)
        .ok_or_else(|| invalid(format!("missing Gemma tensor {logical}")))?;
    match kind {
        "gemma-direct" | "gemma-fp8" => {
            read_quantized_range(artifact, descriptor.value_range, offset, length)
        }
        "gemma-scale-channel" => read_scale_range(
            artifact,
            require_scale(descriptor, ScalePlaneRole::WeightChannel)?,
            offset,
            length,
            false,
        ),
        "gemma-scale-outer" => read_scale_range(
            artifact,
            require_scale(descriptor, ScalePlaneRole::WeightOuter)?,
            offset,
            length,
            true,
        ),
        "gemma-scale-input" => read_scale_range(
            artifact,
            require_scale(descriptor, ScalePlaneRole::InputOuter)?,
            offset,
            length,
            true,
        ),
        "gemma-nvfp4" => read_gemma_nvfp4_range(artifact, descriptor, offset, length),
        _ => Err(invalid(format!("unknown Gemma GGUF source {kind}"))),
    }
}

fn qwen_metadata(
    lock: &ModelLock,
    spec: crate::Qwen35ReviewedSpec,
    tokenizer_bytes: &[u8],
    chat_template_bytes: &[u8],
) -> Result<BTreeMap<String, GgufValue>, GgufError> {
    let text = &lock.model.architecture.text_config;
    let rope_sections = text
        .rope_parameters
        .mrope_section
        .iter()
        .copied()
        .chain(std::iter::once(0))
        .map(|value| i32::try_from(value).map_err(|_| invalid("mRoPE section exceeds i32")))
        .collect::<Result<Vec<_>, _>>()?;
    let rms_epsilon = text
        .rms_norm_eps
        .parse::<f32>()
        .map_err(|_| invalid("RMS epsilon is not f32"))?;
    let partial_rotary = text
        .rope_parameters
        .partial_rotary_factor
        .parse::<f32>()
        .map_err(|_| invalid("partial rotary factor is not f32"))?;
    let rope_dimension = (text.head_dim as f32 * partial_rotary) as u64;
    if rope_dimension == 0 || rope_dimension * 4 != text.head_dim {
        return Err(invalid("Qwen partial rotary dimension differs"));
    }
    let mut metadata = BTreeMap::from([
        (
            "general.architecture".to_owned(),
            GgufValue::String("qwen35".to_owned()),
        ),
        (
            "general.alignment".to_owned(),
            GgufValue::U32(GGUF_ALIGNMENT as u32),
        ),
        (
            "general.type".to_owned(),
            GgufValue::String("model".to_owned()),
        ),
        (
            "general.name".to_owned(),
            GgufValue::String(lock.model.resolved_revision.clone()),
        ),
        (
            "general.license".to_owned(),
            GgufValue::String("apache-2.0".to_owned()),
        ),
        ("general.file_type".to_owned(), GgufValue::U32(32)),
        ("general.quantization_version".to_owned(), GgufValue::U32(2)),
        (
            "qwen35.block_count".to_owned(),
            GgufValue::U32(to_u32(text.num_hidden_layers, "block count")?),
        ),
        (
            "qwen35.context_length".to_owned(),
            GgufValue::U32(to_u32(text.max_position_embeddings, "context length")?),
        ),
        (
            "qwen35.embedding_length".to_owned(),
            GgufValue::U32(to_u32(text.hidden_size, "embedding length")?),
        ),
        (
            "qwen35.feed_forward_length".to_owned(),
            GgufValue::U32(to_u32(text.intermediate_size, "feed-forward length")?),
        ),
        (
            "qwen35.attention.head_count".to_owned(),
            GgufValue::U32(to_u32(text.num_attention_heads, "attention head count")?),
        ),
        (
            "qwen35.attention.head_count_kv".to_owned(),
            GgufValue::U32(to_u32(text.num_key_value_heads, "KV head count")?),
        ),
        (
            "qwen35.attention.key_length".to_owned(),
            GgufValue::U32(to_u32(text.head_dim, "key length")?),
        ),
        (
            "qwen35.attention.value_length".to_owned(),
            GgufValue::U32(to_u32(text.head_dim, "value length")?),
        ),
        (
            "qwen35.attention.layer_norm_rms_epsilon".to_owned(),
            GgufValue::F32(rms_epsilon),
        ),
        (
            "qwen35.rope.dimension_sections".to_owned(),
            GgufValue::Array(GgufArray::I32(rope_sections)),
        ),
        (
            "qwen35.rope.freq_base".to_owned(),
            GgufValue::F32(text.rope_parameters.rope_theta as f32),
        ),
        (
            "qwen35.rope.dimension_count".to_owned(),
            GgufValue::U32(to_u32(rope_dimension, "rope dimension")?),
        ),
        (
            "qwen35.full_attention_interval".to_owned(),
            GgufValue::U32(to_u32(text.full_attention_interval, "attention interval")?),
        ),
        ("qwen35.ssm.conv_kernel".to_owned(), GgufValue::U32(4)),
        ("qwen35.ssm.state_size".to_owned(), GgufValue::U32(128)),
        (
            "qwen35.ssm.group_count".to_owned(),
            GgufValue::U32(to_u32(spec.linear_qk_heads, "SSM group count")?),
        ),
        (
            "qwen35.ssm.time_step_rank".to_owned(),
            GgufValue::U32(to_u32(spec.linear_value_heads, "SSM time-step rank")?),
        ),
        (
            "qwen35.ssm.inner_size".to_owned(),
            GgufValue::U32(to_u32(
                spec.linear_value_heads * spec.linear_head_dim,
                "SSM inner size",
            )?),
        ),
        (
            "tokenizer.ggml.model".to_owned(),
            GgufValue::String("gpt2".to_owned()),
        ),
        (
            "tokenizer.ggml.pre".to_owned(),
            GgufValue::String("qwen35".to_owned()),
        ),
        (
            "tokenizer.ggml.eos_token_id".to_owned(),
            GgufValue::U32(to_u32(
                lock.model
                    .tokenizer_contract
                    .stop_identity
                    .tokenizer_eos
                    .token_id,
                "tokenizer EOS",
            )?),
        ),
        (
            "tokenizer.ggml.padding_token_id".to_owned(),
            GgufValue::U32(to_u32(
                lock.model
                    .tokenizer_contract
                    .stop_identity
                    .config_eos
                    .token_id,
                "padding token",
            )?),
        ),
        (
            "tokenizer.ggml.add_bos_token".to_owned(),
            GgufValue::Bool(false),
        ),
        (
            "tokenizer.chat_template".to_owned(),
            GgufValue::String(
                String::from_utf8(chat_template_bytes.to_vec())
                    .map_err(|_| invalid("chat template is not UTF-8"))?,
            ),
        ),
    ]);
    let tokenizer = tokenizer_metadata(tokenizer_bytes, text.vocab_size)?;
    metadata.extend(tokenizer);
    Ok(metadata)
}

fn tokenizer_metadata(
    bytes: &[u8],
    model_vocab_size: u64,
) -> Result<BTreeMap<String, GgufValue>, GgufError> {
    let root: Value = serde_json::from_slice(bytes)
        .map_err(|error| invalid(format!("tokenizer JSON: {error}")))?;
    let model = root
        .get("model")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid("tokenizer model is missing"))?;
    if model.get("type").and_then(Value::as_str) != Some("BPE") {
        return Err(invalid("Qwen tokenizer is not BPE"));
    }
    let capacity = usize::try_from(model_vocab_size)
        .map_err(|_| invalid("model vocabulary does not fit usize"))?;
    let mut tokens: Vec<Option<String>> = vec![None; capacity];
    let mut token_types = vec![5_i32; capacity];
    let vocab = model
        .get("vocab")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid("tokenizer vocabulary is missing"))?;
    for (token, id) in vocab {
        insert_token(&mut tokens, &mut token_types, token, id, 1)?;
    }
    let added = root
        .get("added_tokens")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("tokenizer added_tokens is missing"))?;
    for entry in added {
        let object = entry
            .as_object()
            .ok_or_else(|| invalid("added token is not an object"))?;
        let content = object
            .get("content")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("added token content is missing"))?;
        let id = object
            .get("id")
            .ok_or_else(|| invalid("added token ID is missing"))?;
        let token_type = if object.get("special").and_then(Value::as_bool) == Some(true) {
            3
        } else {
            4
        };
        insert_token(&mut tokens, &mut token_types, content, id, token_type)?;
    }
    let tokens = tokens
        .into_iter()
        .enumerate()
        .map(|(id, token)| token.unwrap_or_else(|| format!("[PAD{id}]")))
        .collect();
    let merges = model
        .get("merges")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("tokenizer merges are missing"))?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| invalid("tokenizer merge is not a string"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(BTreeMap::from([
        (
            "tokenizer.ggml.tokens".to_owned(),
            GgufValue::Array(GgufArray::String(tokens)),
        ),
        (
            "tokenizer.ggml.token_type".to_owned(),
            GgufValue::Array(GgufArray::I32(token_types)),
        ),
        (
            "tokenizer.ggml.merges".to_owned(),
            GgufValue::Array(GgufArray::String(merges)),
        ),
    ]))
}

fn insert_token(
    tokens: &mut [Option<String>],
    token_types: &mut [i32],
    token: &str,
    id: &Value,
    token_type: i32,
) -> Result<(), GgufError> {
    let id = id.as_u64().ok_or_else(|| invalid("token ID is not u64"))?;
    let index = usize::try_from(id).map_err(|_| invalid("token ID does not fit usize"))?;
    let slot = tokens
        .get_mut(index)
        .ok_or_else(|| invalid(format!("token ID {id} exceeds model vocabulary")))?;
    if slot.replace(token.to_owned()).is_some() {
        return Err(invalid(format!("duplicate tokenizer ID {id}")));
    }
    token_types[index] = token_type;
    Ok(())
}

fn read_asset(cache: &VerifiedCache, kind: FrontendAssetKind) -> Result<Vec<u8>, GgufError> {
    cache
        .read_frontend_asset(kind)
        .map_err(|error| invalid(error.to_string()))
}

fn require_scale(
    descriptor: &QuantizedTensorDescriptor,
    role: ScalePlaneRole,
) -> Result<&QuantizedScalePlane, GgufError> {
    let mut matching = descriptor
        .scale_planes
        .iter()
        .filter(|plane| plane.role == role);
    let plane = matching.next().ok_or_else(|| {
        invalid(format!(
            "tensor {} is missing scale {role:?}",
            descriptor.logical_name
        ))
    })?;
    if matching.next().is_some() {
        return Err(invalid(format!(
            "tensor {} has duplicate scale {role:?}",
            descriptor.logical_name
        )));
    }
    Ok(plane)
}

fn quantized_role(role: QuantizedTensorRole) -> &'static str {
    match role {
        QuantizedTensorRole::Embedding => "embedding",
        QuantizedTensorRole::AttentionProjection => "attention-projection",
        QuantizedTensorRole::MlpProjection => "mlp-projection",
        QuantizedTensorRole::Normalization => "normalization",
        QuantizedTensorRole::Scalar => "scalar",
        QuantizedTensorRole::KnownUnconsumed => "known-unconsumed",
    }
}

fn scope(role: QuantizedTensorRole) -> GgufTensorScope {
    if role == QuantizedTensorRole::KnownUnconsumed {
        GgufTensorScope::KnownUnconsumed
    } else {
        GgufTensorScope::Consumed
    }
}

fn insert_recipe_metadata(
    metadata: &mut BTreeMap<String, GgufValue>,
    recipe: &GgufTensorRecipeV1,
) -> Result<(), GgufError> {
    metadata.insert(SLLM_EXTENSION_VERSION_KEY.to_owned(), GgufValue::U32(1));
    metadata.insert(
        SLLM_TENSOR_RECIPE_KEY.to_owned(),
        GgufValue::String(recipe.canonical_json()?),
    );
    metadata.insert(
        SLLM_TENSOR_RECIPE_SHA256_KEY.to_owned(),
        GgufValue::String(recipe.digest()?),
    );
    Ok(())
}

fn insert_frontend_asset(
    metadata: &mut BTreeMap<String, GgufValue>,
    key: &str,
    bytes: Vec<u8>,
) -> Result<(), GgufError> {
    let text = String::from_utf8(bytes)
        .map_err(|_| invalid(format!("frontend asset {key} is not UTF-8")))?;
    metadata.insert(key.to_owned(), GgufValue::String(text.clone()));
    metadata.insert(
        format!("{key}.sha256"),
        GgufValue::String(sha256(text.as_bytes())),
    );
    Ok(())
}

fn checked_subrange(range: [u64; 2], offset: u64, length: usize) -> Result<[u64; 2], GgufError> {
    let start = range[0]
        .checked_add(offset)
        .ok_or_else(|| invalid("source subrange start overflows"))?;
    let end = start
        .checked_add(u64::try_from(length).map_err(|_| invalid("source length exceeds u64"))?)
        .ok_or_else(|| invalid("source subrange end overflows"))?;
    if start < range[0] || end > range[1] {
        return Err(invalid("source subrange exceeds verified tensor plane"));
    }
    Ok([start, end])
}

fn read_quantized_range(
    artifact: &VerifiedUnslothGemma4Nvfp4,
    range: [u64; 2],
    offset: u64,
    length: usize,
) -> Result<Vec<u8>, GgufError> {
    artifact
        .read_source_range(checked_subrange(range, offset, length)?)
        .map_err(|error| invalid(error.to_string()))
}

fn read_scale_range(
    artifact: &VerifiedUnslothGemma4Nvfp4,
    plane: &QuantizedScalePlane,
    offset: u64,
    length: usize,
    reciprocal: bool,
) -> Result<Vec<u8>, GgufError> {
    if reciprocal {
        if offset != 0 || length != 4 {
            return Err(invalid("reciprocal F32 scale read is not exact"));
        }
        return artifact
            .read_f32_reciprocal(plane)
            .map(|value| value.to_le_bytes().to_vec())
            .map_err(|error| invalid(error.to_string()));
    }
    read_quantized_range(artifact, plane.source_range, offset, length)
}

fn read_gemma_nvfp4_range(
    artifact: &VerifiedUnslothGemma4Nvfp4,
    descriptor: &QuantizedTensorDescriptor,
    offset: u64,
    length: usize,
) -> Result<Vec<u8>, GgufError> {
    let end = offset
        .checked_add(u64::try_from(length).map_err(|_| invalid("NVFP4 length exceeds u64"))?)
        .ok_or_else(|| invalid("NVFP4 output range overflows"))?;
    let first_block = offset / NVFP4_BLOCK_BYTES as u64;
    let end_block = end.div_ceil(NVFP4_BLOCK_BYTES as u64);
    let block_count = end_block
        .checked_sub(first_block)
        .ok_or_else(|| invalid("NVFP4 block range underflows"))?;
    let value_offset = first_block
        .checked_mul(32)
        .ok_or_else(|| invalid("NVFP4 value offset overflows"))?;
    let value_length = usize::try_from(
        block_count
            .checked_mul(32)
            .ok_or_else(|| invalid("NVFP4 value length overflows"))?,
    )
    .map_err(|_| invalid("NVFP4 value length exceeds usize"))?;
    let scale = require_scale(descriptor, ScalePlaneRole::WeightBlock)?;
    let scale_offset = first_block
        .checked_mul(4)
        .ok_or_else(|| invalid("NVFP4 scale offset overflows"))?;
    let scale_length = usize::try_from(
        block_count
            .checked_mul(4)
            .ok_or_else(|| invalid("NVFP4 scale length overflows"))?,
    )
    .map_err(|_| invalid("NVFP4 scale length exceeds usize"))?;
    let values =
        read_quantized_range(artifact, descriptor.value_range, value_offset, value_length)?;
    let scales = read_quantized_range(artifact, scale.source_range, scale_offset, scale_length)?;
    let blocks =
        usize::try_from(block_count).map_err(|_| invalid("NVFP4 block count exceeds usize"))?;
    let repacked = repack_nvfp4_standard(&values, &scales, blocks, NVFP4_BLOCK_ELEMENTS)?;
    let within = usize::try_from(offset % NVFP4_BLOCK_BYTES as u64)
        .map_err(|_| invalid("NVFP4 slice offset exceeds usize"))?;
    let slice_end = within
        .checked_add(length)
        .ok_or_else(|| invalid("NVFP4 slice end overflows"))?;
    repacked
        .get(within..slice_end)
        .map(ToOwned::to_owned)
        .ok_or_else(|| invalid("NVFP4 repack returned a short range"))
}

fn read_fp8_sidecar_range(
    sidecar: &VerifiedFp8Sidecar,
    name: &str,
    scale: bool,
    offset: u64,
    length: usize,
) -> Result<Vec<u8>, GgufError> {
    sidecar
        .read_tensor_range(
            name,
            scale,
            offset,
            u64::try_from(length).map_err(|_| invalid("FP8 read length exceeds u64"))?,
        )
        .map_err(|error| invalid(error.to_string()))
}

fn read_moe_mxfp4_range(
    artifact: &VerifiedQwen35Moe,
    expert: &Qwen35MoeExpertTensor,
    offset: u64,
    length: usize,
) -> Result<Vec<u8>, GgufError> {
    let end = offset
        .checked_add(u64::try_from(length).map_err(|_| invalid("MXFP4 length exceeds u64"))?)
        .ok_or_else(|| invalid("MXFP4 output range overflows"))?;
    let first_block = offset / MXFP4_BLOCK_BYTES as u64;
    let end_block = end.div_ceil(MXFP4_BLOCK_BYTES as u64);
    let block_count = end_block
        .checked_sub(first_block)
        .ok_or_else(|| invalid("MXFP4 block range underflows"))?;
    let value_offset = first_block
        .checked_mul(16)
        .ok_or_else(|| invalid("MXFP4 value offset overflows"))?;
    let value_length = usize::try_from(
        block_count
            .checked_mul(16)
            .ok_or_else(|| invalid("MXFP4 value length overflows"))?,
    )
    .map_err(|_| invalid("MXFP4 value length exceeds usize"))?;
    let scale_length =
        usize::try_from(block_count).map_err(|_| invalid("MXFP4 scale length exceeds usize"))?;
    let values = artifact
        .read_plane_range(&expert.value, value_offset, value_length)
        .map_err(|error| invalid(error.to_string()))?;
    let scales = artifact
        .read_plane_range(&expert.scale, first_block, scale_length)
        .map_err(|error| invalid(error.to_string()))?;
    let blocks = usize::try_from(block_count).map_err(|_| invalid("MXFP4 blocks exceed usize"))?;
    let repacked = repack_mxfp4_standard(&values, &scales, blocks, MXFP4_BLOCK_ELEMENTS)?;
    let within = usize::try_from(offset % MXFP4_BLOCK_BYTES as u64)
        .map_err(|_| invalid("MXFP4 slice offset exceeds usize"))?;
    let slice_end = within
        .checked_add(length)
        .ok_or_else(|| invalid("MXFP4 slice end overflows"))?;
    repacked
        .get(within..slice_end)
        .map(ToOwned::to_owned)
        .ok_or_else(|| invalid("MXFP4 repack returned a short range"))
}

fn to_u32(value: u64, label: &str) -> Result<u32, GgufError> {
    u32::try_from(value).map_err(|_| invalid(format!("{label} exceeds u32")))
}

fn sha256(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("sha256:{:x}", hasher.finalize())
}

fn invalid(message: impl Into<String>) -> GgufError {
    GgufError::Invalid(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{decode_e2m1, decode_e4m3fn, decode_e8m0, decode_mxfp4};

    #[test]
    fn tokenizer_padding_and_boundary_ids_are_deterministic() {
        let bytes = br#"{
          "added_tokens":[
            {"id":3,"content":"<special>","special":true},
            {"id":4,"content":"<ordinary>","special":false}
          ],
          "model":{"type":"BPE","vocab":{"a":0,"b":2},"merges":["a b"]}
        }"#;
        let metadata = tokenizer_metadata(bytes, 7).expect("tokenizer metadata");
        assert_eq!(
            metadata.get("tokenizer.ggml.tokens"),
            Some(&GgufValue::Array(GgufArray::String(vec![
                "a".to_owned(),
                "[PAD1]".to_owned(),
                "b".to_owned(),
                "<special>".to_owned(),
                "<ordinary>".to_owned(),
                "[PAD5]".to_owned(),
                "[PAD6]".to_owned(),
            ])))
        );
        assert_eq!(
            metadata.get("tokenizer.ggml.token_type"),
            Some(&GgufValue::Array(GgufArray::I32(vec![1, 5, 1, 3, 4, 5, 5])))
        );
    }

    #[test]
    fn tokenizer_rejects_both_sides_of_capacity_and_duplicate_ids() {
        let too_large = br#"{
          "added_tokens":[],
          "model":{"type":"BPE","vocab":{"a":7},"merges":[]}
        }"#;
        assert!(tokenizer_metadata(too_large, 7).is_err());
        let duplicate = br#"{
          "added_tokens":[{"id":0,"content":"b","special":true}],
          "model":{"type":"BPE","vocab":{"a":0},"merges":[]}
        }"#;
        assert!(tokenizer_metadata(duplicate, 1).is_err());
    }

    #[test]
    fn mxfp4_repack_matches_independent_standard_decoder() {
        let rows = 2;
        let columns = 32;
        let codes: Vec<u8> = (0..rows * columns)
            .map(|index| (index % 16) as u8)
            .collect();
        let packed = pack_adjacent(&codes);
        let scales = vec![127, 128];
        let source = decode_mxfp4(&packed, &scales, rows, columns).expect("source decode");
        let standard =
            repack_mxfp4_standard(&packed, &scales, rows, columns).expect("standard repack");
        assert_eq!(standard.len(), rows * MXFP4_BLOCK_BYTES);
        assert_eq!(decode_standard_mxfp4(&standard), source);

        for columns in [31, 33] {
            let values = vec![0; rows * columns / 2];
            assert!(repack_mxfp4_standard(&values, &[127, 127], rows, columns).is_err());
        }
    }

    #[test]
    fn nvfp4_repack_preserves_codes_scales_and_outer_scale_semantics() {
        let rows = 2;
        let columns = 64;
        let codes: Vec<u8> = (0..rows * columns)
            .map(|index| ((index * 5 + 3) % 16) as u8)
            .collect();
        let packed = pack_adjacent(&codes);
        let scales = vec![0x38, 0x40, 0x48, 0x50, 0x30, 0x38, 0x40, 0x48];
        let outer = 0.375_f32;
        let standard =
            repack_nvfp4_standard(&packed, &scales, rows, columns).expect("standard repack");
        assert_eq!(standard.len(), rows * NVFP4_BLOCK_BYTES);
        let decoded = decode_standard_nvfp4(&standard, outer);
        let expected: Vec<f32> = codes
            .iter()
            .enumerate()
            .map(|(index, code)| {
                decode_e2m1(*code)
                    * decode_e4m3fn(scales[(index / columns) * 4 + (index % columns) / 16])
                    * outer
            })
            .collect();
        assert_eq!(decoded, expected);

        for columns in [63, 65] {
            let values = vec![0; rows * columns / 2];
            let scales = vec![0; rows * columns.div_ceil(16)];
            assert!(repack_nvfp4_standard(&values, &scales, rows, columns).is_err());
        }
    }

    fn pack_adjacent(codes: &[u8]) -> Vec<u8> {
        let mut output = vec![0; codes.len().div_ceil(2)];
        for (index, code) in codes.iter().copied().enumerate() {
            if index & 1 == 0 {
                output[index / 2] = code;
            } else {
                output[index / 2] |= code << 4;
            }
        }
        output
    }

    fn decode_standard_mxfp4(bytes: &[u8]) -> Vec<f32> {
        let mut output = Vec::new();
        for block in bytes.chunks_exact(MXFP4_BLOCK_BYTES) {
            let scale = decode_e8m0(block[0]);
            for index in 0..MXFP4_BLOCK_ELEMENTS {
                let packed = block[1 + index % 16];
                let code = if index < 16 {
                    packed & 0x0f
                } else {
                    packed >> 4
                };
                output.push(decode_e2m1(code) * scale);
            }
        }
        output
    }

    fn decode_standard_nvfp4(bytes: &[u8], outer: f32) -> Vec<f32> {
        let mut output = Vec::new();
        for block in bytes.chunks_exact(NVFP4_BLOCK_BYTES) {
            for subblock in 0..4 {
                let scale = decode_e4m3fn(block[subblock]);
                let values = &block[4 + subblock * 8..4 + (subblock + 1) * 8];
                for index in 0..NVFP4_SUBBLOCK_ELEMENTS {
                    let packed = values[index % 8];
                    let code = if index < 8 {
                        packed & 0x0f
                    } else {
                        packed >> 4
                    };
                    output.push(decode_e2m1(code) * scale * outer);
                }
            }
        }
        output
    }
}
