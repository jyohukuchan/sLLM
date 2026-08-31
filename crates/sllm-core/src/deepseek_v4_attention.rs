//! Container-neutral FP32 semantic oracles for DeepSeek V4 compressed attention.
//!
//! These routines freeze completed-block publication, CSA/HCA feature-wise
//! compression, and Lightning Indexer visibility. They are intentionally not a
//! production attention implementation or a CPU fallback for a device backend.

use std::cmp::Ordering;
use std::fmt;

pub const DEEPSEEK_V4_CSA_RATIO: u32 = 4;
pub const DEEPSEEK_V4_HCA_RATIO: u32 = 128;
pub const DEEPSEEK_V4_RAW_SLIDING_WINDOW: u32 = 128;
pub const DEEPSEEK_V4_LIGHTNING_INDEXER_TOP_K: u32 = 512;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DeepSeekV4AttentionCompression {
    Uncompressed,
    Csa4To1,
    Hca128To1,
}

impl DeepSeekV4AttentionCompression {
    pub const fn ratio(self) -> Option<u32> {
        match self {
            Self::Uncompressed => None,
            Self::Csa4To1 => Some(DEEPSEEK_V4_CSA_RATIO),
            Self::Hca128To1 => Some(DEEPSEEK_V4_HCA_RATIO),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DeepSeekV4AttentionPlane {
    FirstCandidateKey,
    FirstCandidateValue,
    FirstCandidateKeyScore,
    FirstCandidateValueScore,
    SecondCandidateKey,
    SecondCandidateValue,
    SecondCandidateKeyScore,
    SecondCandidateValueScore,
    HcaKey,
    HcaValue,
    HcaKeyScore,
    HcaValueScore,
    LightningIndexerScore,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DeepSeekV4AttentionStage {
    CsaKeySoftmax,
    CsaValueSoftmax,
    HcaKeySoftmax,
    HcaValueSoftmax,
    CsaKeyWeightedSum,
    CsaValueWeightedSum,
    HcaKeyWeightedSum,
    HcaValueWeightedSum,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DeepSeekV4AttentionSemanticError {
    InvalidFeatureCount {
        key_features: u32,
        value_features: u32,
    },
    WrongCompressionMode {
        expected: DeepSeekV4AttentionCompression,
        actual: DeepSeekV4AttentionCompression,
    },
    InvalidTopK(u32),
    LightningIndexerRequiresCsa {
        actual: DeepSeekV4AttentionCompression,
    },
    ElementCountOverflow {
        plane: DeepSeekV4AttentionPlane,
    },
    ElementCountMismatch {
        plane: DeepSeekV4AttentionPlane,
        expected: usize,
        actual: usize,
    },
    NonFiniteInput {
        plane: DeepSeekV4AttentionPlane,
        index: usize,
    },
    NonFiniteIntermediate {
        stage: DeepSeekV4AttentionStage,
        block: u64,
        feature: u32,
    },
    AllocationFailed,
    PositionOverflow,
    BlockCountOverflow,
}

impl fmt::Display for DeepSeekV4AttentionSemanticError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "DeepSeek V4 attention semantic error: {self:?}")
    }
}

impl std::error::Error for DeepSeekV4AttentionSemanticError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeepSeekV4CompressionDescriptor {
    mode: DeepSeekV4AttentionCompression,
    token_count: u64,
    key_feature_count: u32,
    value_feature_count: u32,
}

impl DeepSeekV4CompressionDescriptor {
    pub fn new(
        mode: DeepSeekV4AttentionCompression,
        token_count: u64,
        key_feature_count: u32,
        value_feature_count: u32,
    ) -> Result<Self, DeepSeekV4AttentionSemanticError> {
        if key_feature_count == 0 || value_feature_count == 0 {
            return Err(DeepSeekV4AttentionSemanticError::InvalidFeatureCount {
                key_features: key_feature_count,
                value_features: value_feature_count,
            });
        }
        Ok(Self {
            mode,
            token_count,
            key_feature_count,
            value_feature_count,
        })
    }

    pub const fn mode(self) -> DeepSeekV4AttentionCompression {
        self.mode
    }

    pub const fn token_count(self) -> u64 {
        self.token_count
    }

    pub const fn key_feature_count(self) -> u32 {
        self.key_feature_count
    }

    pub const fn value_feature_count(self) -> u32 {
        self.value_feature_count
    }
}

/// One candidate plane with independent feature-wise scores for K and V.
/// Shapes are `[token_count, key_features]` and
/// `[token_count, value_features]` in row-major order.
#[derive(Clone, Copy, Debug)]
pub struct DeepSeekV4CompressionCandidate<'a> {
    pub keys: &'a [f32],
    pub values: &'a [f32],
    pub key_scores: &'a [f32],
    pub value_scores: &'a [f32],
}

#[derive(Clone, Copy, Debug)]
pub struct DeepSeekV4CsaCompressionInput<'a> {
    pub descriptor: DeepSeekV4CompressionDescriptor,
    /// Candidate 1 from the previous four-token window. For the first output
    /// block the oracle substitutes four zero-KV, negative-infinity-score rows.
    pub first: DeepSeekV4CompressionCandidate<'a>,
    /// Candidate 2 from the current four-token window.
    pub second: DeepSeekV4CompressionCandidate<'a>,
}

#[derive(Clone, Copy, Debug)]
pub struct DeepSeekV4HcaCompressionInput<'a> {
    pub descriptor: DeepSeekV4CompressionDescriptor,
    /// The single candidate plane compressed within each non-overlapping
    /// 128-token block.
    pub candidate: DeepSeekV4CompressionCandidate<'a>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DeepSeekV4CompressedKv {
    mode: DeepSeekV4AttentionCompression,
    source_token_count: u64,
    completed_block_count: u64,
    candidates_per_block: u32,
    key_feature_count: u32,
    value_feature_count: u32,
    keys: Vec<f32>,
    values: Vec<f32>,
}

impl DeepSeekV4CompressedKv {
    pub const fn mode(&self) -> DeepSeekV4AttentionCompression {
        self.mode
    }

    pub const fn source_token_count(&self) -> u64 {
        self.source_token_count
    }

    pub const fn completed_block_count(&self) -> u64 {
        self.completed_block_count
    }

    pub const fn candidates_per_block(&self) -> u32 {
        self.candidates_per_block
    }

    pub const fn key_feature_count(&self) -> u32 {
        self.key_feature_count
    }

    pub const fn value_feature_count(&self) -> u32 {
        self.value_feature_count
    }

    /// Row-major `[completed_blocks, key_features]` FP32 reference output.
    pub fn keys(&self) -> &[f32] {
        &self.keys
    }

    /// Row-major `[completed_blocks, value_features]` FP32 reference output.
    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

/// Completed compressed blocks for a token count. Incomplete blocks are never
/// published by this contract.
pub fn deepseek_v4_attention_completed_blocks(
    token_count: u64,
    mode: DeepSeekV4AttentionCompression,
) -> Result<u64, DeepSeekV4AttentionSemanticError> {
    mode.ratio()
        .map(|ratio| token_count / u64::from(ratio))
        .ok_or(DeepSeekV4AttentionSemanticError::WrongCompressionMode {
            expected: DeepSeekV4AttentionCompression::Csa4To1,
            actual: mode,
        })
}

/// Completed compressed blocks visible at zero-based query `position`, equal
/// to `floor((position + 1) / ratio)`.
pub fn deepseek_v4_attention_visible_blocks_at(
    position: u64,
    mode: DeepSeekV4AttentionCompression,
) -> Result<u64, DeepSeekV4AttentionSemanticError> {
    let token_count = position
        .checked_add(1)
        .ok_or(DeepSeekV4AttentionSemanticError::PositionOverflow)?;
    deepseek_v4_attention_completed_blocks(token_count, mode)
}

fn checked_element_count(
    token_count: u64,
    feature_count: u32,
    plane: DeepSeekV4AttentionPlane,
) -> Result<usize, DeepSeekV4AttentionSemanticError> {
    let tokens = usize::try_from(token_count)
        .map_err(|_| DeepSeekV4AttentionSemanticError::ElementCountOverflow { plane })?;
    let features = usize::try_from(feature_count)
        .map_err(|_| DeepSeekV4AttentionSemanticError::ElementCountOverflow { plane })?;
    tokens
        .checked_mul(features)
        .ok_or(DeepSeekV4AttentionSemanticError::ElementCountOverflow { plane })
}

fn checked_output_count(
    block_count: u64,
    feature_count: u32,
    plane: DeepSeekV4AttentionPlane,
) -> Result<usize, DeepSeekV4AttentionSemanticError> {
    checked_element_count(block_count, feature_count, plane)
}

fn validate_plane(
    values: &[f32],
    expected: usize,
    plane: DeepSeekV4AttentionPlane,
) -> Result<(), DeepSeekV4AttentionSemanticError> {
    if values.len() != expected {
        return Err(DeepSeekV4AttentionSemanticError::ElementCountMismatch {
            plane,
            expected,
            actual: values.len(),
        });
    }
    for (index, value) in values.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(DeepSeekV4AttentionSemanticError::NonFiniteInput { plane, index });
        }
    }
    Ok(())
}

fn validate_candidate(
    descriptor: DeepSeekV4CompressionDescriptor,
    candidate: DeepSeekV4CompressionCandidate<'_>,
    planes: [DeepSeekV4AttentionPlane; 4],
) -> Result<(usize, usize), DeepSeekV4AttentionSemanticError> {
    let key_count = checked_element_count(
        descriptor.token_count,
        descriptor.key_feature_count,
        planes[0],
    )?;
    let value_count = checked_element_count(
        descriptor.token_count,
        descriptor.value_feature_count,
        planes[1],
    )?;
    validate_plane(candidate.keys, key_count, planes[0])?;
    validate_plane(candidate.values, value_count, planes[1])?;
    validate_plane(candidate.key_scores, key_count, planes[2])?;
    validate_plane(candidate.value_scores, value_count, planes[3])?;
    Ok((key_count, value_count))
}

fn reserve_f32(count: usize) -> Result<Vec<f32>, DeepSeekV4AttentionSemanticError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| DeepSeekV4AttentionSemanticError::AllocationFailed)?;
    Ok(values)
}

fn stable_weighted_feature<I>(
    candidates: I,
    stage_softmax: DeepSeekV4AttentionStage,
    stage_sum: DeepSeekV4AttentionStage,
    block: u64,
    feature: u32,
) -> Result<f32, DeepSeekV4AttentionSemanticError>
where
    I: Clone + Iterator<Item = (f32, f32)>,
{
    let mut maximum = f32::NEG_INFINITY;
    for (_, score) in candidates.clone() {
        maximum = maximum.max(score);
    }
    if !maximum.is_finite() {
        return Err(DeepSeekV4AttentionSemanticError::NonFiniteIntermediate {
            stage: stage_softmax,
            block,
            feature,
        });
    }

    let mut denominator = 0.0_f32;
    let mut numerator = 0.0_f32;
    for (value, score) in candidates {
        let weight = (score - maximum).exp();
        denominator += weight;
        numerator += weight * value;
        if !denominator.is_finite() || !numerator.is_finite() {
            return Err(DeepSeekV4AttentionSemanticError::NonFiniteIntermediate {
                stage: stage_sum,
                block,
                feature,
            });
        }
    }
    if denominator <= 0.0 {
        return Err(DeepSeekV4AttentionSemanticError::NonFiniteIntermediate {
            stage: stage_softmax,
            block,
            feature,
        });
    }
    let output = numerator / denominator;
    if !output.is_finite() {
        return Err(DeepSeekV4AttentionSemanticError::NonFiniteIntermediate {
            stage: stage_sum,
            block,
            feature,
        });
    }
    Ok(output)
}

#[derive(Clone, Copy)]
struct CsaFeaturePlanes<'a> {
    first_values: &'a [f32],
    first_scores: &'a [f32],
    second_values: &'a [f32],
    second_scores: &'a [f32],
    softmax_stage: DeepSeekV4AttentionStage,
    sum_stage: DeepSeekV4AttentionStage,
}

fn csa_feature(
    block: usize,
    feature: usize,
    feature_count: usize,
    planes: CsaFeaturePlanes<'_>,
) -> Result<f32, DeepSeekV4AttentionSemanticError> {
    // The first block has four synthetic (zero, -inf) candidates. They have
    // zero softmax weight and are omitted from the arithmetic iterator while
    // remaining part of the eight-candidate semantic contract.
    let previous = if block == 0 {
        None
    } else {
        Some((block - 1) * DEEPSEEK_V4_CSA_RATIO as usize)
    };
    let current = block * DEEPSEEK_V4_CSA_RATIO as usize;
    let first = previous.into_iter().flat_map(|start| {
        (start..start + DEEPSEEK_V4_CSA_RATIO as usize).map(move |token| {
            let index = token * feature_count + feature;
            (planes.first_values[index], planes.first_scores[index])
        })
    });
    let second = (current..current + DEEPSEEK_V4_CSA_RATIO as usize).map(|token| {
        let index = token * feature_count + feature;
        (planes.second_values[index], planes.second_scores[index])
    });
    stable_weighted_feature(
        first.chain(second),
        planes.softmax_stage,
        planes.sum_stage,
        block as u64,
        feature as u32,
    )
}

/// FP32 oracle for 4:1 CSA compression.
///
/// Each completed block combines candidate 1 from the previous four-token
/// window and candidate 2 from the current window. The first previous window
/// consists of four synthetic zero-KV, negative-infinity-score rows.
pub fn reference_deepseek_v4_csa_compression(
    input: DeepSeekV4CsaCompressionInput<'_>,
) -> Result<DeepSeekV4CompressedKv, DeepSeekV4AttentionSemanticError> {
    let descriptor = input.descriptor;
    if descriptor.mode != DeepSeekV4AttentionCompression::Csa4To1 {
        return Err(DeepSeekV4AttentionSemanticError::WrongCompressionMode {
            expected: DeepSeekV4AttentionCompression::Csa4To1,
            actual: descriptor.mode,
        });
    }
    validate_candidate(
        descriptor,
        input.first,
        [
            DeepSeekV4AttentionPlane::FirstCandidateKey,
            DeepSeekV4AttentionPlane::FirstCandidateValue,
            DeepSeekV4AttentionPlane::FirstCandidateKeyScore,
            DeepSeekV4AttentionPlane::FirstCandidateValueScore,
        ],
    )?;
    validate_candidate(
        descriptor,
        input.second,
        [
            DeepSeekV4AttentionPlane::SecondCandidateKey,
            DeepSeekV4AttentionPlane::SecondCandidateValue,
            DeepSeekV4AttentionPlane::SecondCandidateKeyScore,
            DeepSeekV4AttentionPlane::SecondCandidateValueScore,
        ],
    )?;

    let blocks = deepseek_v4_attention_completed_blocks(descriptor.token_count, descriptor.mode)?;
    let key_output_count = checked_output_count(
        blocks,
        descriptor.key_feature_count,
        DeepSeekV4AttentionPlane::FirstCandidateKey,
    )?;
    let value_output_count = checked_output_count(
        blocks,
        descriptor.value_feature_count,
        DeepSeekV4AttentionPlane::FirstCandidateValue,
    )?;
    let mut keys = reserve_f32(key_output_count)?;
    let mut values = reserve_f32(value_output_count)?;
    let key_features = descriptor.key_feature_count as usize;
    let value_features = descriptor.value_feature_count as usize;
    let block_count = usize::try_from(blocks)
        .map_err(|_| DeepSeekV4AttentionSemanticError::BlockCountOverflow)?;
    for block in 0..block_count {
        for feature in 0..key_features {
            keys.push(csa_feature(
                block,
                feature,
                key_features,
                CsaFeaturePlanes {
                    first_values: input.first.keys,
                    first_scores: input.first.key_scores,
                    second_values: input.second.keys,
                    second_scores: input.second.key_scores,
                    softmax_stage: DeepSeekV4AttentionStage::CsaKeySoftmax,
                    sum_stage: DeepSeekV4AttentionStage::CsaKeyWeightedSum,
                },
            )?);
        }
        for feature in 0..value_features {
            values.push(csa_feature(
                block,
                feature,
                value_features,
                CsaFeaturePlanes {
                    first_values: input.first.values,
                    first_scores: input.first.value_scores,
                    second_values: input.second.values,
                    second_scores: input.second.value_scores,
                    softmax_stage: DeepSeekV4AttentionStage::CsaValueSoftmax,
                    sum_stage: DeepSeekV4AttentionStage::CsaValueWeightedSum,
                },
            )?);
        }
    }
    Ok(DeepSeekV4CompressedKv {
        mode: descriptor.mode,
        source_token_count: descriptor.token_count,
        completed_block_count: blocks,
        candidates_per_block: 8,
        key_feature_count: descriptor.key_feature_count,
        value_feature_count: descriptor.value_feature_count,
        keys,
        values,
    })
}

fn hca_feature(
    block: usize,
    feature: usize,
    feature_count: usize,
    values: &[f32],
    scores: &[f32],
    softmax_stage: DeepSeekV4AttentionStage,
    sum_stage: DeepSeekV4AttentionStage,
) -> Result<f32, DeepSeekV4AttentionSemanticError> {
    let start = block * DEEPSEEK_V4_HCA_RATIO as usize;
    let candidates = (start..start + DEEPSEEK_V4_HCA_RATIO as usize).map(|token| {
        let index = token * feature_count + feature;
        (values[index], scores[index])
    });
    stable_weighted_feature(
        candidates,
        softmax_stage,
        sum_stage,
        block as u64,
        feature as u32,
    )
}

/// FP32 oracle for feature-wise compression of non-overlapping 128-token HCA
/// blocks. Any incomplete final block remains unpublished.
pub fn reference_deepseek_v4_hca_compression(
    input: DeepSeekV4HcaCompressionInput<'_>,
) -> Result<DeepSeekV4CompressedKv, DeepSeekV4AttentionSemanticError> {
    let descriptor = input.descriptor;
    if descriptor.mode != DeepSeekV4AttentionCompression::Hca128To1 {
        return Err(DeepSeekV4AttentionSemanticError::WrongCompressionMode {
            expected: DeepSeekV4AttentionCompression::Hca128To1,
            actual: descriptor.mode,
        });
    }
    validate_candidate(
        descriptor,
        input.candidate,
        [
            DeepSeekV4AttentionPlane::HcaKey,
            DeepSeekV4AttentionPlane::HcaValue,
            DeepSeekV4AttentionPlane::HcaKeyScore,
            DeepSeekV4AttentionPlane::HcaValueScore,
        ],
    )?;
    let blocks = deepseek_v4_attention_completed_blocks(descriptor.token_count, descriptor.mode)?;
    let key_output_count = checked_output_count(
        blocks,
        descriptor.key_feature_count,
        DeepSeekV4AttentionPlane::HcaKey,
    )?;
    let value_output_count = checked_output_count(
        blocks,
        descriptor.value_feature_count,
        DeepSeekV4AttentionPlane::HcaValue,
    )?;
    let mut keys = reserve_f32(key_output_count)?;
    let mut values = reserve_f32(value_output_count)?;
    let key_features = descriptor.key_feature_count as usize;
    let value_features = descriptor.value_feature_count as usize;
    let block_count = usize::try_from(blocks)
        .map_err(|_| DeepSeekV4AttentionSemanticError::BlockCountOverflow)?;
    for block in 0..block_count {
        for feature in 0..key_features {
            keys.push(hca_feature(
                block,
                feature,
                key_features,
                input.candidate.keys,
                input.candidate.key_scores,
                DeepSeekV4AttentionStage::HcaKeySoftmax,
                DeepSeekV4AttentionStage::HcaKeyWeightedSum,
            )?);
        }
        for feature in 0..value_features {
            values.push(hca_feature(
                block,
                feature,
                value_features,
                input.candidate.values,
                input.candidate.value_scores,
                DeepSeekV4AttentionStage::HcaValueSoftmax,
                DeepSeekV4AttentionStage::HcaValueWeightedSum,
            )?);
        }
    }
    Ok(DeepSeekV4CompressedKv {
        mode: descriptor.mode,
        source_token_count: descriptor.token_count,
        completed_block_count: blocks,
        candidates_per_block: DEEPSEEK_V4_HCA_RATIO,
        key_feature_count: descriptor.key_feature_count,
        value_feature_count: descriptor.value_feature_count,
        keys,
        values,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeepSeekV4LightningIndexerDescriptor {
    mode: DeepSeekV4AttentionCompression,
    query_position: u64,
    top_k: u32,
}

impl DeepSeekV4LightningIndexerDescriptor {
    pub fn new(
        mode: DeepSeekV4AttentionCompression,
        query_position: u64,
        top_k: u32,
    ) -> Result<Self, DeepSeekV4AttentionSemanticError> {
        if top_k != DEEPSEEK_V4_LIGHTNING_INDEXER_TOP_K {
            return Err(DeepSeekV4AttentionSemanticError::InvalidTopK(top_k));
        }
        Ok(Self {
            mode,
            query_position,
            top_k,
        })
    }

    pub const fn mode(self) -> DeepSeekV4AttentionCompression {
        self.mode
    }

    pub const fn query_position(self) -> u64 {
        self.query_position
    }

    pub const fn top_k(self) -> u32 {
        self.top_k
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepSeekV4AttentionVisibility {
    query_position: u64,
    raw_token_range: [u64; 2],
    completed_compressed_block_count: u64,
    selected_compressed_block_ids: Vec<u64>,
}

impl DeepSeekV4AttentionVisibility {
    pub const fn query_position(&self) -> u64 {
        self.query_position
    }

    /// Half-open raw token range `[start, end)`. It always contains the latest
    /// `min(position + 1, 128)` tokens independently of compressed selection.
    pub const fn raw_token_range(&self) -> [u64; 2] {
        self.raw_token_range
    }

    pub const fn completed_compressed_block_count(&self) -> u64 {
        self.completed_compressed_block_count
    }

    /// Rank order: larger score first, then smaller block ID for exact ties.
    pub fn selected_compressed_block_ids(&self) -> &[u64] {
        &self.selected_compressed_block_ids
    }
}

/// Stable Lightning Indexer selection over completed CSA blocks only.
///
/// `compressed_block_scores` must contain exactly one finite score for every
/// completed 4-token block visible at `query_position`. HCA and uncompressed
/// modes are rejected. Raw sliding-window visibility is returned separately
/// and is never removed by top-k selection.
pub fn reference_deepseek_v4_lightning_indexer(
    descriptor: DeepSeekV4LightningIndexerDescriptor,
    compressed_block_scores: &[f32],
) -> Result<DeepSeekV4AttentionVisibility, DeepSeekV4AttentionSemanticError> {
    if descriptor.mode != DeepSeekV4AttentionCompression::Csa4To1 {
        return Err(
            DeepSeekV4AttentionSemanticError::LightningIndexerRequiresCsa {
                actual: descriptor.mode,
            },
        );
    }
    let end = descriptor
        .query_position
        .checked_add(1)
        .ok_or(DeepSeekV4AttentionSemanticError::PositionOverflow)?;
    let completed =
        deepseek_v4_attention_visible_blocks_at(descriptor.query_position, descriptor.mode)?;
    let expected = usize::try_from(completed)
        .map_err(|_| DeepSeekV4AttentionSemanticError::BlockCountOverflow)?;
    validate_plane(
        compressed_block_scores,
        expected,
        DeepSeekV4AttentionPlane::LightningIndexerScore,
    )?;
    let mut ranked = Vec::new();
    ranked
        .try_reserve_exact(expected)
        .map_err(|_| DeepSeekV4AttentionSemanticError::AllocationFailed)?;
    for (block, score) in compressed_block_scores.iter().copied().enumerate() {
        ranked.push((block as u64, score));
    }
    ranked.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });
    ranked.truncate(expected.min(descriptor.top_k as usize));
    let selected_compressed_block_ids = ranked.into_iter().map(|(block, _)| block).collect();
    Ok(DeepSeekV4AttentionVisibility {
        query_position: descriptor.query_position,
        raw_token_range: [
            end.saturating_sub(u64::from(DEEPSEEK_V4_RAW_SLIDING_WINDOW)),
            end,
        ],
        completed_compressed_block_count: completed,
        selected_compressed_block_ids,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate<'a>(
        keys: &'a [f32],
        values: &'a [f32],
        key_scores: &'a [f32],
        value_scores: &'a [f32],
    ) -> DeepSeekV4CompressionCandidate<'a> {
        DeepSeekV4CompressionCandidate {
            keys,
            values,
            key_scores,
            value_scores,
        }
    }

    fn csa_fixture(
        token_count: u64,
        key_features: u32,
        value_features: u32,
    ) -> DeepSeekV4CompressedKv {
        let key_count = token_count as usize * key_features as usize;
        let value_count = token_count as usize * value_features as usize;
        let first_keys = vec![100.0; key_count];
        let first_values = vec![200.0; value_count];
        let first_key_scores = vec![0.0; key_count];
        let first_value_scores = vec![0.0; value_count];
        let second_keys = vec![2.0; key_count];
        let second_values = vec![3.0; value_count];
        let second_key_scores = vec![0.0; key_count];
        let second_value_scores = vec![0.0; value_count];
        reference_deepseek_v4_csa_compression(DeepSeekV4CsaCompressionInput {
            descriptor: DeepSeekV4CompressionDescriptor::new(
                DeepSeekV4AttentionCompression::Csa4To1,
                token_count,
                key_features,
                value_features,
            )
            .unwrap(),
            first: candidate(
                &first_keys,
                &first_values,
                &first_key_scores,
                &first_value_scores,
            ),
            second: candidate(
                &second_keys,
                &second_values,
                &second_key_scores,
                &second_value_scores,
            ),
        })
        .unwrap()
    }

    #[test]
    fn completed_block_boundaries_publish_only_complete_windows() {
        for (tokens, csa, hca) in [
            (3, 0, 0),
            (4, 1, 0),
            (5, 1, 0),
            (127, 31, 0),
            (128, 32, 1),
            (129, 32, 1),
        ] {
            assert_eq!(
                deepseek_v4_attention_completed_blocks(
                    tokens,
                    DeepSeekV4AttentionCompression::Csa4To1
                )
                .unwrap(),
                csa
            );
            assert_eq!(
                deepseek_v4_attention_completed_blocks(
                    tokens,
                    DeepSeekV4AttentionCompression::Hca128To1
                )
                .unwrap(),
                hca
            );
        }
        assert_eq!(
            deepseek_v4_attention_visible_blocks_at(3, DeepSeekV4AttentionCompression::Csa4To1)
                .unwrap(),
            1
        );
        assert_eq!(
            deepseek_v4_attention_visible_blocks_at(127, DeepSeekV4AttentionCompression::Hca128To1)
                .unwrap(),
            1
        );
    }

    #[test]
    fn csa_first_synthetic_window_and_non_aligned_features_are_exact() {
        for (tokens, expected_blocks) in [(3_u64, 0_u64), (4, 1), (5, 1)] {
            let boundary = csa_fixture(tokens, 3, 5);
            assert_eq!(boundary.completed_block_count(), expected_blocks);
            assert_eq!(boundary.keys().len(), expected_blocks as usize * 3);
            assert_eq!(boundary.values().len(), expected_blocks as usize * 5);
        }
        let output = csa_fixture(5, 3, 5);
        assert_eq!(output.completed_block_count(), 1);
        assert_eq!(output.candidates_per_block(), 8);
        assert_eq!(output.keys(), &[2.0, 2.0, 2.0]);
        assert_eq!(output.values(), &[3.0, 3.0, 3.0, 3.0, 3.0]);
        // Candidate 1 is 100/200, but the synthetic previous window makes it
        // invisible for the first completed block.
        assert!(!output.keys().contains(&100.0));
        assert!(!output.values().contains(&200.0));
    }

    #[test]
    fn csa_second_block_combines_previous_first_and_current_second_candidates() {
        let output = csa_fixture(8, 1, 1);
        assert_eq!(output.completed_block_count(), 2);
        assert_eq!(output.keys(), &[2.0, 51.0]);
        assert_eq!(output.values(), &[3.0, 101.5]);
    }

    #[test]
    fn csa_feature_wise_softmax_is_stable_under_skew() {
        let descriptor =
            DeepSeekV4CompressionDescriptor::new(DeepSeekV4AttentionCompression::Csa4To1, 4, 1, 1)
                .unwrap();
        let zeros = [0.0; 4];
        let values = [1.0, 2.0, 3.0, 9.0];
        let scores = [-1_000.0, -1_000.0, -1_000.0, 1_000.0];
        let output = reference_deepseek_v4_csa_compression(DeepSeekV4CsaCompressionInput {
            descriptor,
            first: candidate(&zeros, &zeros, &zeros, &zeros),
            second: candidate(&values, &values, &scores, &scores),
        })
        .unwrap();
        assert_eq!(output.keys(), &[9.0]);
        assert_eq!(output.values(), &[9.0]);
    }

    #[test]
    fn hca_non_overlapping_blocks_and_127_128_129_boundaries_are_exact() {
        for (tokens, expected_blocks) in [(127_u64, 0_u64), (128, 1), (129, 1)] {
            let key_features = 3;
            let value_features = 5;
            let keys = vec![4.0; tokens as usize * key_features];
            let values = vec![7.0; tokens as usize * value_features];
            let key_scores = vec![0.0; keys.len()];
            let value_scores = vec![0.0; values.len()];
            let output = reference_deepseek_v4_hca_compression(DeepSeekV4HcaCompressionInput {
                descriptor: DeepSeekV4CompressionDescriptor::new(
                    DeepSeekV4AttentionCompression::Hca128To1,
                    tokens,
                    key_features as u32,
                    value_features as u32,
                )
                .unwrap(),
                candidate: candidate(&keys, &values, &key_scores, &value_scores),
            })
            .unwrap();
            assert_eq!(output.completed_block_count(), expected_blocks);
            assert_eq!(output.keys().len(), expected_blocks as usize * key_features);
            assert!(output.keys().iter().all(|value| *value == 4.0));
            assert!(output.values().iter().all(|value| *value == 7.0));
        }
    }

    #[test]
    fn lightning_indexer_top_k_boundary_ties_skew_and_raw_window_are_exact() {
        assert!(
            DeepSeekV4LightningIndexerDescriptor::new(
                DeepSeekV4AttentionCompression::Csa4To1,
                0,
                511
            )
            .is_err()
        );
        assert!(
            DeepSeekV4LightningIndexerDescriptor::new(
                DeepSeekV4AttentionCompression::Csa4To1,
                0,
                513
            )
            .is_err()
        );
        let descriptor = DeepSeekV4LightningIndexerDescriptor::new(
            DeepSeekV4AttentionCompression::Csa4To1,
            513 * 4 - 1,
            512,
        )
        .unwrap();
        let ties = vec![1.0; 513];
        let tied = reference_deepseek_v4_lightning_indexer(descriptor, &ties).unwrap();
        assert_eq!(tied.completed_compressed_block_count(), 513);
        assert_eq!(tied.selected_compressed_block_ids().len(), 512);
        assert_eq!(tied.selected_compressed_block_ids()[0], 0);
        assert_eq!(tied.selected_compressed_block_ids()[511], 511);
        assert_eq!(tied.raw_token_range(), [1_924, 2_052]);

        let mut skew = ties;
        skew[512] = 1.0e30;
        let skewed = reference_deepseek_v4_lightning_indexer(descriptor, &skew).unwrap();
        assert_eq!(skewed.selected_compressed_block_ids()[0], 512);
        assert_eq!(skewed.selected_compressed_block_ids()[1], 0);
        assert_eq!(skewed.selected_compressed_block_ids()[511], 510);
    }

    #[test]
    fn lightning_indexer_rejects_hca_incomplete_shapes_and_nonfinite_scores() {
        let hca = DeepSeekV4LightningIndexerDescriptor::new(
            DeepSeekV4AttentionCompression::Hca128To1,
            127,
            512,
        )
        .unwrap();
        assert!(matches!(
            reference_deepseek_v4_lightning_indexer(hca, &[0.0]),
            Err(DeepSeekV4AttentionSemanticError::LightningIndexerRequiresCsa { .. })
        ));
        let csa = DeepSeekV4LightningIndexerDescriptor::new(
            DeepSeekV4AttentionCompression::Csa4To1,
            4,
            512,
        )
        .unwrap();
        assert!(matches!(
            reference_deepseek_v4_lightning_indexer(csa, &[0.0, 0.0]),
            Err(DeepSeekV4AttentionSemanticError::ElementCountMismatch { .. })
        ));
        assert!(matches!(
            reference_deepseek_v4_lightning_indexer(csa, &[f32::NAN]),
            Err(DeepSeekV4AttentionSemanticError::NonFiniteInput { .. })
        ));
    }

    #[test]
    fn compression_rejects_wrong_mode_shape_nonfinite_and_overflow() {
        let descriptor =
            DeepSeekV4CompressionDescriptor::new(DeepSeekV4AttentionCompression::Csa4To1, 4, 1, 1)
                .unwrap();
        let finite = [0.0; 4];
        let short = [0.0; 3];
        let bad_shape = reference_deepseek_v4_csa_compression(DeepSeekV4CsaCompressionInput {
            descriptor,
            first: candidate(&short, &finite, &finite, &finite),
            second: candidate(&finite, &finite, &finite, &finite),
        });
        assert!(matches!(
            bad_shape,
            Err(DeepSeekV4AttentionSemanticError::ElementCountMismatch { .. })
        ));
        let nan = [0.0, f32::NAN, 0.0, 0.0];
        let bad_finite = reference_deepseek_v4_csa_compression(DeepSeekV4CsaCompressionInput {
            descriptor,
            first: candidate(&finite, &finite, &finite, &finite),
            second: candidate(&finite, &finite, &nan, &finite),
        });
        assert!(matches!(
            bad_finite,
            Err(DeepSeekV4AttentionSemanticError::NonFiniteInput { .. })
        ));
        let wrong = DeepSeekV4CompressionDescriptor::new(
            DeepSeekV4AttentionCompression::Hca128To1,
            4,
            1,
            1,
        )
        .unwrap();
        assert!(matches!(
            reference_deepseek_v4_csa_compression(DeepSeekV4CsaCompressionInput {
                descriptor: wrong,
                first: candidate(&finite, &finite, &finite, &finite),
                second: candidate(&finite, &finite, &finite, &finite),
            }),
            Err(DeepSeekV4AttentionSemanticError::WrongCompressionMode { .. })
        ));

        let overflow = DeepSeekV4CompressionDescriptor::new(
            DeepSeekV4AttentionCompression::Csa4To1,
            u64::MAX,
            2,
            1,
        )
        .unwrap();
        let empty = [];
        assert!(matches!(
            reference_deepseek_v4_csa_compression(DeepSeekV4CsaCompressionInput {
                descriptor: overflow,
                first: candidate(&empty, &empty, &empty, &empty),
                second: candidate(&empty, &empty, &empty, &empty),
            }),
            Err(DeepSeekV4AttentionSemanticError::ElementCountOverflow { .. })
        ));
        let max_position = DeepSeekV4LightningIndexerDescriptor::new(
            DeepSeekV4AttentionCompression::Csa4To1,
            u64::MAX,
            512,
        )
        .unwrap();
        assert_eq!(
            reference_deepseek_v4_lightning_indexer(max_position, &[]),
            Err(DeepSeekV4AttentionSemanticError::PositionOverflow)
        );
    }
}
