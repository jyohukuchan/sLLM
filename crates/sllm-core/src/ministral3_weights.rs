//! Strict resident weight source for the official Ministral 3 3B text GGUF.
//!
//! [`VerifiedOfficialMinistral3Gguf`] has already checked the immutable GGUF
//! header and its 236-entry text catalog.  This module binds that catalog to
//! the container-neutral Ministral 3 graph and constructs the packed resident
//! layout consumed by a runtime.  It intentionally has no vision mapping:
//! the official file is multimodal, but the current production contract is
//! text-only.

use crate::ministral3_gguf::VerifiedOfficialMinistral3Gguf;
use crate::ministral3_graph::{Ministral3TensorClass, build_ministral3_text_graph};
use crate::model::{TensorDType, TensorDescriptor};
use crate::weights::{
    VerifiedWeightPlanMetadata, WEIGHT_LOAD_CHUNK_BYTES, WeightClassification, WeightConsumer,
    WeightConsumerKey, WeightLoadChunk, WeightLoadEntry, WeightLoadPlan, WeightPlanError,
    WeightRangeSource, WeightUploadError,
};
use crate::{GgufTensorType, WeightLoadPlan as PublicWeightLoadPlan};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// The schema identifies this packed, text-only resident layout.
pub const MINISTRAL3_WEIGHT_PLAN_SCHEMA: &str =
    "ministral3-official-gguf-text-f32-norm-to-bf16-plan-v1";
pub const MINISTRAL3_WEIGHT_LOCK_FINGERPRINT: &str =
    "sha256:17ef932bea952e007f9dad63151da5699132ec513d1033d618df7382e24aa3ee";
pub const MINISTRAL3_WEIGHT_TENSOR_COUNT: usize = 236;
pub const MINISTRAL3_WEIGHT_BF16_TENSOR_COUNT: usize = 236;
pub const MINISTRAL3_WEIGHT_F32_NORM_TENSOR_COUNT: usize = 53;
/// Packed destination bytes after the exact F32-norm-to-BF16 conversion.
pub const MINISTRAL3_WEIGHT_RESIDENT_BYTES: u64 = 6_858_012_672;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Ministral3WeightError {
    Invalid(String),
    Plan(WeightPlanError),
}

impl fmt::Display for Ministral3WeightError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => {
                write!(formatter, "invalid Ministral 3 resident weights: {message}")
            }
            Self::Plan(error) => write!(formatter, "invalid Ministral 3 weight plan: {error}"),
        }
    }
}

impl std::error::Error for Ministral3WeightError {}

impl From<WeightPlanError> for Ministral3WeightError {
    fn from(error: WeightPlanError) -> Self {
        Self::Plan(error)
    }
}

fn invalid(message: impl Into<String>) -> Ministral3WeightError {
    Ministral3WeightError::Invalid(message.into())
}

fn checked_product(shape: &[u64], label: &str) -> Result<u64, Ministral3WeightError> {
    if shape.is_empty() || shape.contains(&0) {
        return Err(invalid(format!("{label} has an empty or zero shape")));
    }
    shape.iter().try_fold(1_u64, |product, dimension| {
        product
            .checked_mul(*dimension)
            .ok_or_else(|| invalid(format!("{label} shape product overflows")))
    })
}

fn checked_byte_length(
    shape: &[u64],
    dtype: TensorDType,
    label: &str,
) -> Result<u64, Ministral3WeightError> {
    checked_product(shape, label)?
        .checked_mul(match dtype {
            TensorDType::Bf16 | TensorDType::F16 => 2,
            TensorDType::F32 | TensorDType::I32 => 4,
            TensorDType::I64 => 8,
            TensorDType::U8 => 1,
        })
        .ok_or_else(|| invalid(format!("{label} byte length overflows")))
}

/// Convert an official F32 RMS norm tensor to the graph's BF16 storage. The
/// official artifact was produced from BF16 source values, therefore every
/// finite F32 value must have an all-zero lower BF16 half. Any drift is
/// rejected instead of silently rounding a changed artifact.
fn convert_f32_norm_payload(
    tensor_name: &str,
    raw: &[u8],
) -> Result<Vec<u8>, Ministral3WeightError> {
    if raw.len() % 4 != 0 {
        return Err(invalid(format!(
            "F32 norm payload is not element aligned: {tensor_name}"
        )));
    }
    let mut converted = Vec::with_capacity(raw.len() / 2);
    for (index, bytes) in raw.chunks_exact(4).enumerate() {
        let bits = u32::from_le_bytes(bytes.try_into().expect("chunks_exact gives four bytes"));
        let value = f32::from_bits(bits);
        if !value.is_finite() {
            return Err(invalid(format!(
                "F32 norm contains a non-finite value at {tensor_name}[{index}]"
            )));
        }
        if bits & 0xffff != 0 {
            return Err(invalid(format!(
                "F32 norm is not exactly representable as BF16 at {tensor_name}[{index}]"
            )));
        }
        converted.extend_from_slice(&((bits >> 16) as u16).to_le_bytes());
    }
    Ok(converted)
}

/// A verified official GGUF plus descriptors in the resident graph's logical
/// shape convention.  GGUF dimensions are reversed when creating descriptors;
/// this is the same convention used by the existing model weight plans.
#[derive(Clone)]
pub struct VerifiedMinistral3WeightSource {
    verified: VerifiedOfficialMinistral3Gguf,
    descriptors: BTreeMap<String, TensorDescriptor>,
    /// Official GGUF stores RMS norms as F32. The reviewed graph and HIP
    /// RMSNorm contract consume BF16, so these values are converted once at
    /// verification time and retained as stable resident source bytes.
    converted_norms: BTreeMap<String, Vec<u8>>,
}

impl fmt::Debug for VerifiedMinistral3WeightSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VerifiedMinistral3WeightSource")
            .field("path", &self.verified.gguf().path())
            .field("tensor_count", &self.descriptors.len())
            .field("resident_bytes", &MINISTRAL3_WEIGHT_RESIDENT_BYTES)
            .finish_non_exhaustive()
    }
}

impl VerifiedMinistral3WeightSource {
    /// Bind an already verified official GGUF to the text graph's exact
    /// consumer, shape, type, range, and byte-accounting contract.
    pub fn from_verified_gguf(
        verified: VerifiedOfficialMinistral3Gguf,
    ) -> Result<Self, Ministral3WeightError> {
        let expected = expected_tensor_specs()?;
        let gguf = verified.gguf();
        if gguf.tensors().len() != MINISTRAL3_WEIGHT_TENSOR_COUNT {
            return Err(invalid(format!(
                "official GGUF tensor count is {}, expected {MINISTRAL3_WEIGHT_TENSOR_COUNT}",
                gguf.tensors().len()
            )));
        }

        let source_file = gguf.path().display().to_string();
        let mut descriptors = BTreeMap::new();
        let mut bf16_count = 0_usize;
        let mut f32_norm_count = 0_usize;
        let mut resident_bytes = 0_u64;
        let mut converted_norms = BTreeMap::new();
        for tensor in gguf.tensors() {
            let spec = expected.get(tensor.name.as_str()).ok_or_else(|| {
                invalid(format!(
                    "unknown or vision tensor in text resident source: {}",
                    tensor.name
                ))
            })?;
            if tensor.tensor_type != spec.gguf_type {
                return Err(invalid(format!("tensor type differs: {}", tensor.name)));
            }
            let mut logical_shape = tensor.dimensions.clone();
            logical_shape.reverse();
            if logical_shape != spec.shape {
                return Err(invalid(format!("tensor shape differs: {}", tensor.name)));
            }
            let resident_byte_length =
                checked_byte_length(&logical_shape, spec.dtype, &tensor.name)?;
            let physical_dtype = match spec.gguf_type {
                GgufTensorType::Bf16 => TensorDType::Bf16,
                GgufTensorType::F32 => TensorDType::F32,
                _ => {
                    return Err(invalid(format!(
                        "unsupported GGUF source type: {}",
                        tensor.name
                    )));
                }
            };
            let physical_byte_length =
                checked_byte_length(&logical_shape, physical_dtype, &tensor.name)?;
            if tensor.byte_length() != physical_byte_length {
                return Err(invalid(format!(
                    "tensor byte length differs: {}",
                    tensor.name
                )));
            }
            if tensor.relative_offset % gguf.alignment() != 0 {
                return Err(invalid(format!(
                    "tensor offset is misaligned: {}",
                    tensor.name
                )));
            }
            let physical_relative_end = tensor
                .relative_offset
                .checked_add(tensor.byte_length())
                .ok_or_else(|| {
                    invalid(format!("tensor relative range overflows: {}", tensor.name))
                })?;
            let absolute_start = gguf
                .data_offset()
                .checked_add(tensor.relative_offset)
                .ok_or_else(|| {
                    invalid(format!("tensor absolute start overflows: {}", tensor.name))
                })?;
            let physical_absolute_end = absolute_start
                .checked_add(tensor.byte_length())
                .ok_or_else(|| {
                    invalid(format!("tensor absolute end overflows: {}", tensor.name))
                })?;
            if physical_relative_end > gguf.file_size().saturating_sub(gguf.data_offset())
                || tensor.absolute_range != [absolute_start, physical_absolute_end]
                || physical_absolute_end > gguf.file_size()
                || tensor.absolute_range[0] >= tensor.absolute_range[1]
            {
                return Err(invalid(format!(
                    "tensor range differs or exceeds file: {}",
                    tensor.name
                )));
            }
            let resident_relative_end = tensor
                .relative_offset
                .checked_add(resident_byte_length)
                .ok_or_else(|| {
                    invalid(format!(
                        "resident relative range overflows: {}",
                        tensor.name
                    ))
                })?;
            let resident_absolute_end = absolute_start
                .checked_add(resident_byte_length)
                .ok_or_else(|| {
                    invalid(format!(
                        "resident absolute range overflows: {}",
                        tensor.name
                    ))
                })?;
            let (data_offsets, absolute_byte_range) = if spec.gguf_type == GgufTensorType::F32 {
                let physical_length = usize::try_from(tensor.byte_length())
                    .map_err(|_| invalid("norm tensor is too large"))?;
                let raw = gguf
                    .read_tensor_range(&tensor.name, 0, physical_length)
                    .map_err(|error| invalid(error.to_string()))?;
                let converted = convert_f32_norm_payload(&tensor.name, &raw)?;
                if u64::try_from(converted.len()).ok() != Some(resident_byte_length) {
                    return Err(invalid(format!(
                        "converted norm byte length differs: {}",
                        tensor.name
                    )));
                }
                if converted_norms
                    .insert(tensor.name.clone(), converted)
                    .is_some()
                {
                    return Err(invalid(format!(
                        "duplicate converted norm: {}",
                        tensor.name
                    )));
                }
                (
                    [tensor.relative_offset, resident_relative_end],
                    [absolute_start, resident_absolute_end],
                )
            } else {
                (
                    [tensor.relative_offset, physical_relative_end],
                    tensor.absolute_range,
                )
            };
            let descriptor = TensorDescriptor {
                tensor_name: tensor.name.clone(),
                source_file: source_file.clone(),
                dtype: spec.dtype,
                shape: spec.shape.clone(),
                header_length_field_bytes: 0,
                header_length_bytes: gguf.data_offset(),
                data_buffer_start: gguf.data_offset(),
                data_offset_basis: if spec.gguf_type == GgufTensorType::F32 {
                    "gguf-v3-tensor-data-f32-norm-to-bf16-virtual"
                } else {
                    "gguf-v3-tensor-data"
                }
                .to_owned(),
                data_offsets,
                absolute_byte_range,
                byte_size: resident_byte_length,
            };
            if descriptors
                .insert(tensor.name.clone(), descriptor)
                .is_some()
            {
                return Err(invalid(format!(
                    "duplicate tensor descriptor: {}",
                    tensor.name
                )));
            }
            if spec.dtype != TensorDType::Bf16 {
                return Err(invalid(
                    "official text graph weight must use BF16 resident dtype",
                ));
            }
            bf16_count = bf16_count
                .checked_add(1)
                .ok_or_else(|| invalid("BF16 count overflows"))?;
            if spec.gguf_type == GgufTensorType::F32 {
                f32_norm_count = f32_norm_count
                    .checked_add(1)
                    .ok_or_else(|| invalid("F32 norm count overflows"))?;
            }
            resident_bytes = resident_bytes
                .checked_add(resident_byte_length)
                .ok_or_else(|| invalid("resident byte accounting overflows"))?;
        }
        if descriptors.len() != expected.len()
            || bf16_count != MINISTRAL3_WEIGHT_BF16_TENSOR_COUNT
            || f32_norm_count != MINISTRAL3_WEIGHT_F32_NORM_TENSOR_COUNT
            || converted_norms.len() != MINISTRAL3_WEIGHT_F32_NORM_TENSOR_COUNT
            || resident_bytes != MINISTRAL3_WEIGHT_RESIDENT_BYTES
        {
            return Err(invalid(format!(
                "resident catalog/count/bytes differ: descriptors={}, expected={}, bf16={bf16_count}, f32_norm={f32_norm_count}, converted_norms={}, bytes={resident_bytes}",
                descriptors.len(),
                expected.len(),
                converted_norms.len()
            )));
        }
        Ok(Self {
            verified,
            descriptors,
            converted_norms,
        })
    }

    pub fn gguf(&self) -> &crate::VerifiedGguf {
        self.verified.gguf()
    }

    pub fn verified(&self) -> &VerifiedOfficialMinistral3Gguf {
        &self.verified
    }

    pub fn repository(&self) -> &'static str {
        self.verified.repository()
    }

    pub fn revision(&self) -> &'static str {
        self.verified.revision()
    }

    pub fn file_sha256(&self) -> &'static str {
        self.verified.expected_lfs_sha256()
    }

    pub fn lock_fingerprint(&self) -> &'static str {
        MINISTRAL3_WEIGHT_LOCK_FINGERPRINT
    }

    pub fn resident_bytes(&self) -> u64 {
        MINISTRAL3_WEIGHT_RESIDENT_BYTES
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorDescriptor> {
        self.descriptors.get(name)
    }

    pub fn tensors(&self) -> impl ExactSizeIterator<Item = &TensorDescriptor> {
        self.descriptors.values()
    }

    /// Read a checked range relative to the resident tensor payload. Matrix
    /// ranges are direct GGUF bytes; F32 norm ranges are the stable virtual
    /// BF16 bytes retained during verification. Callers cannot accidentally
    /// read a neighbouring tensor or reinterpret Q/K bytes with a second
    /// permutation.
    pub fn read_tensor_range(
        &self,
        tensor_name: &str,
        offset: u64,
        length: usize,
    ) -> Result<Vec<u8>, Ministral3WeightError> {
        let descriptor = self
            .descriptors
            .get(tensor_name)
            .ok_or_else(|| invalid(format!("unknown text tensor: {tensor_name}")))?;
        let length_u64 = u64::try_from(length).map_err(|_| invalid("read length overflows u64"))?;
        let end = offset
            .checked_add(length_u64)
            .ok_or_else(|| invalid("read range overflows"))?;
        if end > descriptor.byte_size {
            return Err(invalid(format!("read exceeds tensor range: {tensor_name}")));
        }
        let bytes = if let Some(converted) = self.converted_norms.get(tensor_name) {
            let start =
                usize::try_from(offset).map_err(|_| invalid("read offset does not fit usize"))?;
            let end = start
                .checked_add(length)
                .ok_or_else(|| invalid("read range overflows usize"))?;
            converted
                .get(start..end)
                .ok_or_else(|| {
                    invalid(format!("read exceeds converted norm range: {tensor_name}"))
                })?
                .to_vec()
        } else {
            self.gguf()
                .read_tensor_range(tensor_name, offset, length)
                .map_err(|error| invalid(error.to_string()))?
        };
        if bytes.len() != length {
            return Err(invalid(format!("short read: {tensor_name}")));
        }
        Ok(bytes)
    }

    pub fn build_weight_load_plan(&self) -> Result<WeightLoadPlan, Ministral3WeightError> {
        build_ministral3_weight_load_plan(self)
    }
}

impl WeightRangeSource for VerifiedMinistral3WeightSource {
    fn lock_fingerprint(&self) -> &str {
        self.lock_fingerprint()
    }

    fn tensor(&self, tensor_name: &str) -> Option<&TensorDescriptor> {
        self.tensor(tensor_name)
    }

    fn read_tensor_range(
        &self,
        tensor_name: &str,
        offset: u64,
        length: usize,
    ) -> Result<Vec<u8>, WeightUploadError> {
        self.read_tensor_range(tensor_name, offset, length)
            .map_err(|error| WeightUploadError::invalid(error.to_string()))
    }
}

#[derive(Clone)]
struct TensorSpec {
    consumer: WeightConsumerKey,
    dtype: TensorDType,
    gguf_type: GgufTensorType,
    shape: Vec<u64>,
}

fn expected_tensor_specs() -> Result<BTreeMap<String, TensorSpec>, Ministral3WeightError> {
    let mut expected = BTreeMap::new();
    let insert = |expected: &mut BTreeMap<String, TensorSpec>, name: String, spec: TensorSpec| {
        if expected.insert(name.clone(), spec).is_some() {
            Err(invalid(format!(
                "duplicate expected resident tensor: {name}"
            )))
        } else {
            Ok(())
        }
    };
    insert(
        &mut expected,
        "token_embd.weight".to_owned(),
        TensorSpec {
            consumer: WeightConsumerKey {
                layer: None,
                role: WeightConsumer::EmbeddingAndTiedOutput,
            },
            dtype: TensorDType::Bf16,
            gguf_type: GgufTensorType::Bf16,
            shape: vec![131_072, 3_072],
        },
    )?;
    insert(
        &mut expected,
        "output_norm.weight".to_owned(),
        TensorSpec {
            consumer: WeightConsumerKey {
                layer: None,
                role: WeightConsumer::FinalNorm,
            },
            dtype: TensorDType::Bf16,
            gguf_type: GgufTensorType::F32,
            shape: vec![3_072],
        },
    )?;
    for layer in 0..26_u64 {
        let layer_specs = [
            (
                "attn_norm",
                WeightConsumer::InputNorm,
                TensorDType::Bf16,
                GgufTensorType::F32,
                vec![3_072],
            ),
            (
                "attn_q",
                WeightConsumer::AttentionQ,
                TensorDType::Bf16,
                GgufTensorType::Bf16,
                vec![4_096, 3_072],
            ),
            (
                "attn_k",
                WeightConsumer::AttentionK,
                TensorDType::Bf16,
                GgufTensorType::Bf16,
                vec![1_024, 3_072],
            ),
            (
                "attn_v",
                WeightConsumer::AttentionV,
                TensorDType::Bf16,
                GgufTensorType::Bf16,
                vec![1_024, 3_072],
            ),
            (
                "attn_output",
                WeightConsumer::AttentionO,
                TensorDType::Bf16,
                GgufTensorType::Bf16,
                vec![3_072, 4_096],
            ),
            (
                "ffn_norm",
                WeightConsumer::PostAttentionNorm,
                TensorDType::Bf16,
                GgufTensorType::F32,
                vec![3_072],
            ),
            (
                "ffn_gate",
                WeightConsumer::MlpGate,
                TensorDType::Bf16,
                GgufTensorType::Bf16,
                vec![9_216, 3_072],
            ),
            (
                "ffn_down",
                WeightConsumer::MlpDown,
                TensorDType::Bf16,
                GgufTensorType::Bf16,
                vec![3_072, 9_216],
            ),
            (
                "ffn_up",
                WeightConsumer::MlpUp,
                TensorDType::Bf16,
                GgufTensorType::Bf16,
                vec![9_216, 3_072],
            ),
        ];
        for (suffix, role, dtype, gguf_type, shape) in layer_specs {
            insert(
                &mut expected,
                format!("blk.{layer}.{suffix}.weight"),
                TensorSpec {
                    consumer: WeightConsumerKey {
                        layer: Some(layer),
                        role,
                    },
                    dtype,
                    gguf_type,
                    shape,
                },
            )?;
        }
    }
    if expected.len() != MINISTRAL3_WEIGHT_TENSOR_COUNT {
        return Err(invalid(format!(
            "expected resident catalog has {} tensors",
            expected.len()
        )));
    }
    let graph = build_ministral3_text_graph(1, 0, 1)
        .map_err(|error| invalid(format!("text graph contract failed: {error}")))?;
    let graph_specs: BTreeMap<_, _> = graph
        .tensors()
        .iter()
        .filter(|tensor| tensor.class() == Ministral3TensorClass::Weight)
        .filter_map(|tensor| {
            tensor.weight().map(|key| {
                (
                    key,
                    tensor
                        .view()
                        .shape()
                        .iter()
                        .map(|dimension| u64::try_from(*dimension).expect("usize fits u64"))
                        .collect::<Vec<_>>(),
                )
            })
        })
        .collect();
    let expected_graph_specs: BTreeMap<_, _> = expected
        .values()
        .map(|spec| (spec.consumer, spec.shape.clone()))
        .collect();
    if graph_specs != expected_graph_specs {
        return Err(invalid(
            "resident catalog does not match text graph weights",
        ));
    }
    Ok(expected)
}

/// Build the exact packed text-only resident plan from a verified official
/// GGUF source.
pub fn build_ministral3_weight_load_plan(
    source: &VerifiedMinistral3WeightSource,
) -> Result<WeightLoadPlan, Ministral3WeightError> {
    let expected = expected_tensor_specs()?;
    let mut observed_consumers = BTreeSet::new();
    let mut destination_cursor = 0_u64;
    let mut entries = Vec::with_capacity(expected.len());
    for (name, spec) in &expected {
        let descriptor = source
            .tensor(name)
            .ok_or_else(|| invalid(format!("missing required text tensor: {name}")))?;
        if !observed_consumers.insert(spec.consumer) {
            return Err(invalid(format!(
                "duplicate resident consumer: {:?}",
                spec.consumer
            )));
        }
        let destination_start = destination_cursor;
        let mut chunks = Vec::new();
        let mut consumed = 0_u64;
        while consumed < descriptor.byte_size {
            let remaining = descriptor.byte_size - consumed;
            let byte_length = remaining.min(WEIGHT_LOAD_CHUNK_BYTES);
            let source_offset = descriptor.absolute_byte_range[0]
                .checked_add(consumed)
                .ok_or_else(|| invalid(format!("source chunk offset overflows: {name}")))?;
            let destination_offset = destination_start
                .checked_add(consumed)
                .ok_or_else(|| invalid(format!("destination chunk offset overflows: {name}")))?;
            chunks.push(WeightLoadChunk {
                source_offset,
                destination_offset,
                byte_length,
            });
            consumed = consumed
                .checked_add(byte_length)
                .ok_or_else(|| invalid(format!("chunk accounting overflows: {name}")))?;
        }
        if consumed != descriptor.byte_size || chunks.is_empty() {
            return Err(invalid(format!("chunks do not cover tensor: {name}")));
        }
        destination_cursor = destination_cursor
            .checked_add(descriptor.byte_size)
            .ok_or_else(|| invalid("destination byte accounting overflows"))?;
        entries.push(WeightLoadEntry {
            tensor_name: name.clone(),
            classification: WeightClassification::Required,
            consumer: Some(spec.consumer),
            dtype: spec.dtype,
            shape: spec.shape.clone(),
            source_file: source.gguf().path().display().to_string(),
            locked_file_size: source.gguf().file_size(),
            locked_file_sha256: source.file_sha256().to_owned(),
            source_range: descriptor.absolute_byte_range,
            destination_start: Some(destination_start),
            chunks,
        });
    }
    if destination_cursor != MINISTRAL3_WEIGHT_RESIDENT_BYTES
        || observed_consumers.len() != MINISTRAL3_WEIGHT_TENSOR_COUNT
    {
        return Err(invalid(format!(
            "resident plan accounting differs: bytes={destination_cursor}, consumers={}",
            observed_consumers.len()
        )));
    }
    let plan = WeightLoadPlan::from_verified_entries(
        VerifiedWeightPlanMetadata {
            schema_version: MINISTRAL3_WEIGHT_PLAN_SCHEMA.to_owned(),
            repo_id: source.repository().to_owned(),
            resolved_revision: source.revision().to_owned(),
            lock_fingerprint: source.lock_fingerprint().to_owned(),
            tied_embeddings: true,
            chunk_size: WEIGHT_LOAD_CHUNK_BYTES,
            total_destination_bytes: destination_cursor,
        },
        entries,
    )?;
    Ok(plan)
}

/// Alias for callers that use the source-verifier naming style.
pub fn build_verified_ministral3_weight_load_plan(
    verified: VerifiedOfficialMinistral3Gguf,
) -> Result<(VerifiedMinistral3WeightSource, PublicWeightLoadPlan), Ministral3WeightError> {
    let source = VerifiedMinistral3WeightSource::from_verified_gguf(verified)?;
    let plan = build_ministral3_weight_load_plan(&source)?;
    Ok((source, plan))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expected_catalog_is_exact_and_graph_aligned() {
        let expected = expected_tensor_specs().expect("catalog");
        assert_eq!(expected.len(), MINISTRAL3_WEIGHT_TENSOR_COUNT);
        assert_eq!(
            expected
                .values()
                .filter(|spec| spec.dtype == TensorDType::Bf16)
                .count(),
            MINISTRAL3_WEIGHT_BF16_TENSOR_COUNT
        );
        assert_eq!(
            expected
                .values()
                .filter(|spec| spec.gguf_type == GgufTensorType::F32)
                .count(),
            MINISTRAL3_WEIGHT_F32_NORM_TENSOR_COUNT
        );
        assert!(!expected.contains_key("output.weight"));
    }

    #[test]
    fn resident_byte_accounting_uses_bf16_norms() {
        let expected = expected_tensor_specs().expect("catalog");
        let bytes: u64 = expected
            .values()
            .map(|spec| checked_byte_length(&spec.shape, spec.dtype, "fixture").unwrap())
            .sum();
        assert_eq!(bytes, MINISTRAL3_WEIGHT_RESIDENT_BYTES);
    }

    #[test]
    fn f32_norm_conversion_accepts_exact_bf16_and_rejects_drift() {
        let exact = [0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x80, 0xbf];
        assert_eq!(
            convert_f32_norm_payload("norm", &exact).unwrap(),
            [0x80, 0x3f, 0x80, 0xbf]
        );
        let mut mutated = exact;
        mutated[0] = 1;
        assert!(convert_f32_norm_payload("norm", &mutated).is_err());
        let infinity = [0x00, 0x00, 0x80, 0x7f];
        assert!(convert_f32_norm_payload("norm", &infinity).is_err());
        let nan = [0x00, 0x00, 0xc0, 0x7f];
        assert!(convert_f32_norm_payload("norm", &nan).is_err());
    }

    #[test]
    fn converted_norm_range_boundaries_are_exact() {
        let raw = [0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40];
        let converted = convert_f32_norm_payload("norm", &raw).unwrap();
        assert_eq!(&converted[..2], &[0x80, 0x3f]);
        assert_eq!(&converted[2..], &[0x00, 0x40]);
        assert_eq!(converted.len(), raw.len() / 2);
    }

    #[test]
    fn chunk_boundaries_are_checked_for_non_aligned_tensor_sizes() {
        let size = WEIGHT_LOAD_CHUNK_BYTES + 7;
        let mut chunks = Vec::new();
        let mut consumed = 0_u64;
        while consumed < size {
            let byte_length = (size - consumed).min(WEIGHT_LOAD_CHUNK_BYTES);
            chunks.push((consumed, byte_length));
            consumed += byte_length;
        }
        assert_eq!(
            chunks,
            vec![(0, WEIGHT_LOAD_CHUNK_BYTES), (WEIGHT_LOAD_CHUNK_BYTES, 7)]
        );
        assert_eq!(consumed, size);
    }

    #[test]
    fn source_catalog_rejects_unknown_vision_name_in_expected_lookup() {
        let expected = expected_tensor_specs().expect("catalog");
        assert!(!expected.contains_key("vision_tower.patch_conv.weight"));
        assert!(!expected.contains_key("multi_modal_projector.linear_1.weight"));
    }

    #[test]
    #[ignore = "requires the exact official Ministral 3 GGUF under /tmp/sllm-phase60.2pDfxs"]
    fn exact_official_source_converts_norms_and_builds_bf16_plan() {
        let path = std::path::Path::new("/tmp/sllm-phase60.2pDfxs")
            .join(crate::ministral3_gguf::MINISTRAL3_OFFICIAL_GGUF_FILE_NAME);
        let gguf = crate::VerifiedGguf::open(path).expect("open official GGUF");
        let verified = crate::verify_official_ministral3_gguf(gguf).expect("verify GGUF");
        let source = VerifiedMinistral3WeightSource::from_verified_gguf(verified)
            .expect("build resident source");
        assert_eq!(source.resident_bytes(), MINISTRAL3_WEIGHT_RESIDENT_BYTES);
        let norm = source.tensor("blk.0.attn_norm.weight").expect("norm");
        assert_eq!(norm.dtype, TensorDType::Bf16);
        assert_eq!(norm.byte_size, 6_144);
        assert_eq!(
            norm.absolute_byte_range[1] - norm.absolute_byte_range[0],
            6_144
        );
        assert_eq!(
            source
                .read_tensor_range("blk.0.attn_norm.weight", 0, 2)
                .unwrap()
                .len(),
            2
        );
        assert!(
            source
                .read_tensor_range("blk.0.attn_norm.weight", 6_144, 1)
                .is_err()
        );
        let matrix = source.tensor("blk.0.attn_q.weight").expect("query");
        assert_eq!(matrix.dtype, TensorDType::Bf16);
        assert_eq!(matrix.byte_size, 25_165_824);
        let plan = source.build_weight_load_plan().expect("build plan");
        assert_eq!(
            plan.total_destination_bytes,
            MINISTRAL3_WEIGHT_RESIDENT_BYTES
        );
        assert!(
            plan.entries
                .iter()
                .all(|entry| entry.dtype == TensorDType::Bf16)
        );
        assert!(
            plan.entries
                .iter()
                .all(|entry| entry.classification == WeightClassification::Required)
        );
    }
}
