//! Container-neutral FP32 semantic oracle for MiniMax M3 sparse attention.
//!
//! The oracle freezes the paper-level MSA contract: causal block-max index
//! scores, per-GQA-group selection with a reserved local-block slot, and exact
//! scaled-dot-product softmax over the causally visible tokens in the selected
//! blocks. It is not a production attention implementation or a CPU fallback
//! for a device backend.

use std::cmp::Ordering;
use std::fmt;

pub const MINIMAX_M3_ATTENTION_LAYER_COUNT: u32 = 60;
pub const MINIMAX_M3_DENSE_ATTENTION_LAYER_COUNT: u32 = 3;
pub const MINIMAX_M3_MSA_BLOCK_SIZE: u32 = 128;
pub const MINIMAX_M3_MSA_TOP_K_BLOCKS: u32 = 16;
pub const MINIMAX_M3_MSA_GQA_GROUP_COUNT: u32 = 4;
pub const MINIMAX_M3_MSA_INDEX_HEAD_COUNT: u32 = 4;
pub const MINIMAX_M3_MSA_INDEX_DIMENSION: u32 = 128;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MiniMaxM3AttentionKind {
    Dense,
    SparseMsa,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MiniMaxM3MsaPlane {
    IndexQuery,
    IndexKey,
    MainQuery,
    MainKey,
    MainValue,
    MainOutput,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MiniMaxM3MsaStage {
    IndexDotProduct,
    IndexBlockMaximum,
    MainDotProduct,
    MainSoftmax,
    MainWeightedSum,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MiniMaxM3MsaSemanticError {
    InvalidLayerIndex(u32),
    DenseLayerDoesNotUseMsa(u32),
    InvalidQueryHeadCount(u32),
    InvalidFeatureCount {
        key_features: u32,
        value_features: u32,
    },
    PositionOverflow,
    BlockCountOverflow,
    ElementCountOverflow {
        plane: MiniMaxM3MsaPlane,
    },
    ElementCountMismatch {
        plane: MiniMaxM3MsaPlane,
        expected: usize,
        actual: usize,
    },
    NonFiniteInput {
        plane: MiniMaxM3MsaPlane,
        index: usize,
    },
    NonFiniteIntermediate {
        stage: MiniMaxM3MsaStage,
        group: u32,
        head: u32,
        token: u64,
    },
    SelectionDescriptorMismatch,
    InvalidSelectionGroupCount {
        expected: usize,
        actual: usize,
    },
    InvalidSelectionGroup {
        expected: u32,
        actual: u32,
    },
    InvalidSelectedBlockCount {
        group: u32,
        expected: usize,
        actual: usize,
    },
    InvalidSelectedBlock {
        group: u32,
        block: u64,
    },
    DuplicateSelectedBlock {
        group: u32,
        block: u64,
    },
    LocalBlockIsNotReserved {
        group: u32,
        local_block: u64,
    },
    AllocationFailed,
}

impl fmt::Display for MiniMaxM3MsaSemanticError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "MiniMax M3 MSA semantic error: {self:?}")
    }
}

impl std::error::Error for MiniMaxM3MsaSemanticError {}

pub fn minimax_m3_attention_kind(
    layer_index: u32,
) -> Result<MiniMaxM3AttentionKind, MiniMaxM3MsaSemanticError> {
    if layer_index >= MINIMAX_M3_ATTENTION_LAYER_COUNT {
        return Err(MiniMaxM3MsaSemanticError::InvalidLayerIndex(layer_index));
    }
    Ok(if layer_index < MINIMAX_M3_DENSE_ATTENTION_LAYER_COUNT {
        MiniMaxM3AttentionKind::Dense
    } else {
        MiniMaxM3AttentionKind::SparseMsa
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MiniMaxM3MsaDescriptor {
    layer_index: u32,
    query_position: u64,
    token_count: u64,
    query_head_count: u32,
    key_feature_count: u32,
    value_feature_count: u32,
}

impl MiniMaxM3MsaDescriptor {
    pub fn new(
        layer_index: u32,
        query_position: u64,
        query_head_count: u32,
        key_feature_count: u32,
        value_feature_count: u32,
    ) -> Result<Self, MiniMaxM3MsaSemanticError> {
        match minimax_m3_attention_kind(layer_index)? {
            MiniMaxM3AttentionKind::Dense => {
                return Err(MiniMaxM3MsaSemanticError::DenseLayerDoesNotUseMsa(
                    layer_index,
                ));
            }
            MiniMaxM3AttentionKind::SparseMsa => {}
        }
        if query_head_count == 0 || query_head_count % MINIMAX_M3_MSA_GQA_GROUP_COUNT != 0 {
            return Err(MiniMaxM3MsaSemanticError::InvalidQueryHeadCount(
                query_head_count,
            ));
        }
        if key_feature_count == 0 || value_feature_count == 0 {
            return Err(MiniMaxM3MsaSemanticError::InvalidFeatureCount {
                key_features: key_feature_count,
                value_features: value_feature_count,
            });
        }
        let token_count = query_position
            .checked_add(1)
            .ok_or(MiniMaxM3MsaSemanticError::PositionOverflow)?;
        Ok(Self {
            layer_index,
            query_position,
            token_count,
            query_head_count,
            key_feature_count,
            value_feature_count,
        })
    }

    pub const fn layer_index(self) -> u32 {
        self.layer_index
    }

    pub const fn query_position(self) -> u64 {
        self.query_position
    }

    pub const fn token_count(self) -> u64 {
        self.token_count
    }

    pub const fn query_head_count(self) -> u32 {
        self.query_head_count
    }

    pub const fn key_feature_count(self) -> u32 {
        self.key_feature_count
    }

    pub const fn value_feature_count(self) -> u32 {
        self.value_feature_count
    }

    pub const fn heads_per_group(self) -> u32 {
        self.query_head_count / MINIMAX_M3_MSA_GQA_GROUP_COUNT
    }

    pub fn visible_block_count(self) -> Result<u64, MiniMaxM3MsaSemanticError> {
        let last_block = self.query_position / u64::from(MINIMAX_M3_MSA_BLOCK_SIZE);
        last_block
            .checked_add(1)
            .ok_or(MiniMaxM3MsaSemanticError::BlockCountOverflow)
    }

    pub const fn local_block_id(self) -> u64 {
        self.query_position / MINIMAX_M3_MSA_BLOCK_SIZE as u64
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MiniMaxM3MsaIndexInput<'a> {
    pub descriptor: MiniMaxM3MsaDescriptor,
    /// Row-major `[4, 128]`, one index query head per GQA group.
    pub index_queries: &'a [f32],
    /// Row-major `[query_position + 1, 128]`, shared across groups.
    pub index_keys: &'a [f32],
}

#[derive(Clone, Debug, PartialEq)]
pub struct MiniMaxM3MsaGroupSelection {
    group: u32,
    ranked_block_ids: Vec<u64>,
}

impl MiniMaxM3MsaGroupSelection {
    pub const fn group(&self) -> u32 {
        self.group
    }

    /// Selection order. The forced local block occupies the first slot. The
    /// remaining blocks are ordered by decreasing block-max score, then by
    /// increasing block ID for exact ties.
    pub fn ranked_block_ids(&self) -> &[u64] {
        &self.ranked_block_ids
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MiniMaxM3MsaSelection {
    descriptor: MiniMaxM3MsaDescriptor,
    visible_block_count: u64,
    local_block_id: u64,
    groups: Vec<MiniMaxM3MsaGroupSelection>,
}

impl MiniMaxM3MsaSelection {
    pub const fn descriptor(&self) -> MiniMaxM3MsaDescriptor {
        self.descriptor
    }

    pub const fn visible_block_count(&self) -> u64 {
        self.visible_block_count
    }

    pub const fn local_block_id(&self) -> u64 {
        self.local_block_id
    }

    pub fn groups(&self) -> &[MiniMaxM3MsaGroupSelection] {
        &self.groups
    }

    pub fn group(&self, group: u32) -> Option<&MiniMaxM3MsaGroupSelection> {
        self.groups.get(group as usize)
    }
}

fn checked_element_count(
    factors: &[u64],
    plane: MiniMaxM3MsaPlane,
) -> Result<usize, MiniMaxM3MsaSemanticError> {
    let count = factors.iter().copied().try_fold(1_u64, |count, factor| {
        count
            .checked_mul(factor)
            .ok_or(MiniMaxM3MsaSemanticError::ElementCountOverflow { plane })
    })?;
    usize::try_from(count).map_err(|_| MiniMaxM3MsaSemanticError::ElementCountOverflow { plane })
}

fn validate_plane(
    values: &[f32],
    expected: usize,
    plane: MiniMaxM3MsaPlane,
) -> Result<(), MiniMaxM3MsaSemanticError> {
    if values.len() != expected {
        return Err(MiniMaxM3MsaSemanticError::ElementCountMismatch {
            plane,
            expected,
            actual: values.len(),
        });
    }
    for (index, value) in values.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(MiniMaxM3MsaSemanticError::NonFiniteInput { plane, index });
        }
    }
    Ok(())
}

fn reserve_exact<T>(count: usize) -> Result<Vec<T>, MiniMaxM3MsaSemanticError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| MiniMaxM3MsaSemanticError::AllocationFailed)?;
    Ok(values)
}

fn index_block_score(
    group: usize,
    block: usize,
    token_count: usize,
    index_queries: &[f32],
    index_keys: &[f32],
) -> Result<f32, MiniMaxM3MsaSemanticError> {
    let dimension = MINIMAX_M3_MSA_INDEX_DIMENSION as usize;
    let block_size = MINIMAX_M3_MSA_BLOCK_SIZE as usize;
    let start = block
        .checked_mul(block_size)
        .ok_or(MiniMaxM3MsaSemanticError::BlockCountOverflow)?;
    let end = start
        .checked_add(block_size)
        .ok_or(MiniMaxM3MsaSemanticError::BlockCountOverflow)?
        .min(token_count);
    let query_start = group * dimension;
    let scale = 1.0_f32 / (MINIMAX_M3_MSA_INDEX_DIMENSION as f32).sqrt();
    let mut maximum = f32::NEG_INFINITY;
    for token in start..end {
        let key_start = token * dimension;
        let mut dot = 0.0_f32;
        for feature in 0..dimension {
            dot =
                index_queries[query_start + feature].mul_add(index_keys[key_start + feature], dot);
        }
        let score = dot * scale;
        if !score.is_finite() {
            return Err(MiniMaxM3MsaSemanticError::NonFiniteIntermediate {
                stage: MiniMaxM3MsaStage::IndexDotProduct,
                group: group as u32,
                head: group as u32,
                token: token as u64,
            });
        }
        maximum = maximum.max(score);
    }
    if !maximum.is_finite() {
        return Err(MiniMaxM3MsaSemanticError::NonFiniteIntermediate {
            stage: MiniMaxM3MsaStage::IndexBlockMaximum,
            group: group as u32,
            head: group as u32,
            token: start as u64,
        });
    }
    Ok(maximum)
}

/// FP32 reference selection for one query position.
///
/// The local block always occupies one of the at-most-16 selected slots. The
/// other slots use causal index-query/index-key dot products, max-pooled over
/// each visible block independently for every GQA group.
pub fn reference_minimax_m3_msa_selection(
    input: MiniMaxM3MsaIndexInput<'_>,
) -> Result<MiniMaxM3MsaSelection, MiniMaxM3MsaSemanticError> {
    let descriptor = input.descriptor;
    let query_count = checked_element_count(
        &[
            u64::from(MINIMAX_M3_MSA_INDEX_HEAD_COUNT),
            u64::from(MINIMAX_M3_MSA_INDEX_DIMENSION),
        ],
        MiniMaxM3MsaPlane::IndexQuery,
    )?;
    let key_count = checked_element_count(
        &[
            descriptor.token_count,
            u64::from(MINIMAX_M3_MSA_INDEX_DIMENSION),
        ],
        MiniMaxM3MsaPlane::IndexKey,
    )?;
    validate_plane(
        input.index_queries,
        query_count,
        MiniMaxM3MsaPlane::IndexQuery,
    )?;
    validate_plane(input.index_keys, key_count, MiniMaxM3MsaPlane::IndexKey)?;

    let visible_block_count = descriptor.visible_block_count()?;
    let block_count = usize::try_from(visible_block_count)
        .map_err(|_| MiniMaxM3MsaSemanticError::BlockCountOverflow)?;
    let token_count = usize::try_from(descriptor.token_count).map_err(|_| {
        MiniMaxM3MsaSemanticError::ElementCountOverflow {
            plane: MiniMaxM3MsaPlane::IndexKey,
        }
    })?;
    let local_block_id = descriptor.local_block_id();
    let selected_count = block_count.min(MINIMAX_M3_MSA_TOP_K_BLOCKS as usize);
    let mut groups = reserve_exact(MINIMAX_M3_MSA_GQA_GROUP_COUNT as usize)?;
    for group in 0..MINIMAX_M3_MSA_GQA_GROUP_COUNT as usize {
        let mut ranked = reserve_exact(block_count.saturating_sub(1))?;
        for block in 0..block_count {
            if block as u64 == local_block_id {
                continue;
            }
            ranked.push((
                block as u64,
                index_block_score(
                    group,
                    block,
                    token_count,
                    input.index_queries,
                    input.index_keys,
                )?,
            ));
        }
        // The score plus unique block ID is a total order, so the in-place
        // unstable sort remains deterministic without an untracked scratch
        // allocation.
        ranked.sort_unstable_by(|left, right| {
            right
                .1
                .partial_cmp(&left.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.0.cmp(&right.0))
        });
        let mut ranked_block_ids = reserve_exact(selected_count)?;
        ranked_block_ids.push(local_block_id);
        ranked_block_ids.extend(
            ranked
                .into_iter()
                .take(selected_count.saturating_sub(1))
                .map(|(block, _)| block),
        );
        groups.push(MiniMaxM3MsaGroupSelection {
            group: group as u32,
            ranked_block_ids,
        });
    }
    Ok(MiniMaxM3MsaSelection {
        descriptor,
        visible_block_count,
        local_block_id,
        groups,
    })
}

#[derive(Clone, Copy, Debug)]
pub struct MiniMaxM3MsaAttentionInput<'a> {
    pub descriptor: MiniMaxM3MsaDescriptor,
    pub selection: &'a MiniMaxM3MsaSelection,
    /// Row-major `[query_heads, key_features]`.
    pub queries: &'a [f32],
    /// Row-major `[query_position + 1, 4, key_features]`.
    pub keys: &'a [f32],
    /// Row-major `[query_position + 1, 4, value_features]`.
    pub values: &'a [f32],
}

#[derive(Clone, Debug, PartialEq)]
pub struct MiniMaxM3MsaOutput {
    descriptor: MiniMaxM3MsaDescriptor,
    values: Vec<f32>,
}

impl MiniMaxM3MsaOutput {
    pub const fn descriptor(&self) -> MiniMaxM3MsaDescriptor {
        self.descriptor
    }

    /// Row-major `[query_heads, value_features]` FP32 reference output.
    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

fn validate_selection(
    descriptor: MiniMaxM3MsaDescriptor,
    selection: &MiniMaxM3MsaSelection,
) -> Result<(), MiniMaxM3MsaSemanticError> {
    if selection.descriptor != descriptor
        || selection.visible_block_count != descriptor.visible_block_count()?
        || selection.local_block_id != descriptor.local_block_id()
    {
        return Err(MiniMaxM3MsaSemanticError::SelectionDescriptorMismatch);
    }
    let expected_groups = MINIMAX_M3_MSA_GQA_GROUP_COUNT as usize;
    if selection.groups.len() != expected_groups {
        return Err(MiniMaxM3MsaSemanticError::InvalidSelectionGroupCount {
            expected: expected_groups,
            actual: selection.groups.len(),
        });
    }
    let expected_blocks = usize::try_from(selection.visible_block_count)
        .map_err(|_| MiniMaxM3MsaSemanticError::BlockCountOverflow)?
        .min(MINIMAX_M3_MSA_TOP_K_BLOCKS as usize);
    for (expected_group, group) in selection.groups.iter().enumerate() {
        if group.group != expected_group as u32 {
            return Err(MiniMaxM3MsaSemanticError::InvalidSelectionGroup {
                expected: expected_group as u32,
                actual: group.group,
            });
        }
        if group.ranked_block_ids.len() != expected_blocks {
            return Err(MiniMaxM3MsaSemanticError::InvalidSelectedBlockCount {
                group: group.group,
                expected: expected_blocks,
                actual: group.ranked_block_ids.len(),
            });
        }
        if group.ranked_block_ids.first().copied() != Some(selection.local_block_id) {
            return Err(MiniMaxM3MsaSemanticError::LocalBlockIsNotReserved {
                group: group.group,
                local_block: selection.local_block_id,
            });
        }
        for (index, block) in group.ranked_block_ids.iter().copied().enumerate() {
            if block >= selection.visible_block_count {
                return Err(MiniMaxM3MsaSemanticError::InvalidSelectedBlock {
                    group: group.group,
                    block,
                });
            }
            if group.ranked_block_ids[..index].contains(&block) {
                return Err(MiniMaxM3MsaSemanticError::DuplicateSelectedBlock {
                    group: group.group,
                    block,
                });
            }
        }
    }
    Ok(())
}

fn main_score(
    descriptor: MiniMaxM3MsaDescriptor,
    queries: &[f32],
    keys: &[f32],
    head: usize,
    group: usize,
    token: usize,
) -> Result<f32, MiniMaxM3MsaSemanticError> {
    let features = descriptor.key_feature_count as usize;
    let query_start = head * features;
    let key_start = (token * MINIMAX_M3_MSA_GQA_GROUP_COUNT as usize + group) * features;
    let mut dot = 0.0_f32;
    for feature in 0..features {
        dot = queries[query_start + feature].mul_add(keys[key_start + feature], dot);
    }
    let score = dot / (descriptor.key_feature_count as f32).sqrt();
    if !score.is_finite() {
        return Err(MiniMaxM3MsaSemanticError::NonFiniteIntermediate {
            stage: MiniMaxM3MsaStage::MainDotProduct,
            group: group as u32,
            head: head as u32,
            token: token as u64,
        });
    }
    Ok(score)
}

fn token_is_selected(token: usize, selected_blocks: &[u64]) -> bool {
    let block = token as u64 / u64::from(MINIMAX_M3_MSA_BLOCK_SIZE);
    selected_blocks.contains(&block)
}

/// FP32 exact Main Branch attention for one query position.
///
/// Every query head uses the block selection of its adjacent GQA group and
/// that group's K/V head. Scores are standard scaled dot products, and the
/// softmax is evaluated only over causally visible tokens in selected blocks.
pub fn reference_minimax_m3_msa_attention(
    input: MiniMaxM3MsaAttentionInput<'_>,
) -> Result<MiniMaxM3MsaOutput, MiniMaxM3MsaSemanticError> {
    let descriptor = input.descriptor;
    validate_selection(descriptor, input.selection)?;
    let query_count = checked_element_count(
        &[
            u64::from(descriptor.query_head_count),
            u64::from(descriptor.key_feature_count),
        ],
        MiniMaxM3MsaPlane::MainQuery,
    )?;
    let key_count = checked_element_count(
        &[
            descriptor.token_count,
            u64::from(MINIMAX_M3_MSA_GQA_GROUP_COUNT),
            u64::from(descriptor.key_feature_count),
        ],
        MiniMaxM3MsaPlane::MainKey,
    )?;
    let value_count = checked_element_count(
        &[
            descriptor.token_count,
            u64::from(MINIMAX_M3_MSA_GQA_GROUP_COUNT),
            u64::from(descriptor.value_feature_count),
        ],
        MiniMaxM3MsaPlane::MainValue,
    )?;
    let output_count = checked_element_count(
        &[
            u64::from(descriptor.query_head_count),
            u64::from(descriptor.value_feature_count),
        ],
        MiniMaxM3MsaPlane::MainOutput,
    )?;
    validate_plane(input.queries, query_count, MiniMaxM3MsaPlane::MainQuery)?;
    validate_plane(input.keys, key_count, MiniMaxM3MsaPlane::MainKey)?;
    validate_plane(input.values, value_count, MiniMaxM3MsaPlane::MainValue)?;

    let token_count = usize::try_from(descriptor.token_count).map_err(|_| {
        MiniMaxM3MsaSemanticError::ElementCountOverflow {
            plane: MiniMaxM3MsaPlane::MainKey,
        }
    })?;
    let query_heads = descriptor.query_head_count as usize;
    let value_features = descriptor.value_feature_count as usize;
    let heads_per_group = descriptor.heads_per_group() as usize;
    let mut output = reserve_exact(output_count)?;
    output.resize(output_count, 0.0_f32);
    for head in 0..query_heads {
        let group = head / heads_per_group;
        let selected = input.selection.groups[group].ranked_block_ids.as_slice();
        let mut maximum = f32::NEG_INFINITY;
        for token in 0..token_count {
            if token_is_selected(token, selected) {
                maximum = maximum.max(main_score(
                    descriptor,
                    input.queries,
                    input.keys,
                    head,
                    group,
                    token,
                )?);
            }
        }
        if !maximum.is_finite() {
            return Err(MiniMaxM3MsaSemanticError::NonFiniteIntermediate {
                stage: MiniMaxM3MsaStage::MainSoftmax,
                group: group as u32,
                head: head as u32,
                token: descriptor.query_position,
            });
        }
        let output_start = head * value_features;
        let mut denominator = 0.0_f32;
        for token in 0..token_count {
            if !token_is_selected(token, selected) {
                continue;
            }
            let score = main_score(descriptor, input.queries, input.keys, head, group, token)?;
            let weight = (score - maximum).exp();
            denominator += weight;
            if !weight.is_finite() || !denominator.is_finite() {
                return Err(MiniMaxM3MsaSemanticError::NonFiniteIntermediate {
                    stage: MiniMaxM3MsaStage::MainSoftmax,
                    group: group as u32,
                    head: head as u32,
                    token: token as u64,
                });
            }
            let value_start =
                (token * MINIMAX_M3_MSA_GQA_GROUP_COUNT as usize + group) * value_features;
            for feature in 0..value_features {
                let index = output_start + feature;
                output[index] = input.values[value_start + feature].mul_add(weight, output[index]);
                if !output[index].is_finite() {
                    return Err(MiniMaxM3MsaSemanticError::NonFiniteIntermediate {
                        stage: MiniMaxM3MsaStage::MainWeightedSum,
                        group: group as u32,
                        head: head as u32,
                        token: token as u64,
                    });
                }
            }
        }
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(MiniMaxM3MsaSemanticError::NonFiniteIntermediate {
                stage: MiniMaxM3MsaStage::MainSoftmax,
                group: group as u32,
                head: head as u32,
                token: descriptor.query_position,
            });
        }
        for feature in 0..value_features {
            let index = output_start + feature;
            output[index] /= denominator;
            if !output[index].is_finite() {
                return Err(MiniMaxM3MsaSemanticError::NonFiniteIntermediate {
                    stage: MiniMaxM3MsaStage::MainWeightedSum,
                    group: group as u32,
                    head: head as u32,
                    token: descriptor.query_position,
                });
            }
        }
    }
    Ok(MiniMaxM3MsaOutput {
        descriptor,
        values: output,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor(
        token_count: u64,
        query_heads: u32,
        key_features: u32,
        value_features: u32,
    ) -> MiniMaxM3MsaDescriptor {
        MiniMaxM3MsaDescriptor::new(
            3,
            token_count.checked_sub(1).unwrap(),
            query_heads,
            key_features,
            value_features,
        )
        .unwrap()
    }

    fn zero_selection(descriptor: MiniMaxM3MsaDescriptor) -> MiniMaxM3MsaSelection {
        let queries = vec![
            0.0;
            MINIMAX_M3_MSA_INDEX_HEAD_COUNT as usize
                * MINIMAX_M3_MSA_INDEX_DIMENSION as usize
        ];
        let keys =
            vec![0.0; descriptor.token_count as usize * MINIMAX_M3_MSA_INDEX_DIMENSION as usize];
        reference_minimax_m3_msa_selection(MiniMaxM3MsaIndexInput {
            descriptor,
            index_queries: &queries,
            index_keys: &keys,
        })
        .unwrap()
    }

    #[test]
    fn official_layer_schedule_and_fixed_index_contract_are_exact() {
        assert_eq!(MINIMAX_M3_MSA_BLOCK_SIZE, 128);
        assert_eq!(MINIMAX_M3_MSA_TOP_K_BLOCKS, 16);
        assert_eq!(MINIMAX_M3_MSA_GQA_GROUP_COUNT, 4);
        assert_eq!(MINIMAX_M3_MSA_INDEX_HEAD_COUNT, 4);
        assert_eq!(MINIMAX_M3_MSA_INDEX_DIMENSION, 128);
        for layer in 0..3 {
            assert_eq!(
                minimax_m3_attention_kind(layer).unwrap(),
                MiniMaxM3AttentionKind::Dense
            );
            assert!(matches!(
                MiniMaxM3MsaDescriptor::new(layer, 0, 4, 3, 5),
                Err(MiniMaxM3MsaSemanticError::DenseLayerDoesNotUseMsa(_))
            ));
        }
        for layer in 3..60 {
            assert_eq!(
                minimax_m3_attention_kind(layer).unwrap(),
                MiniMaxM3AttentionKind::SparseMsa
            );
        }
        assert!(matches!(
            minimax_m3_attention_kind(60),
            Err(MiniMaxM3MsaSemanticError::InvalidLayerIndex(60))
        ));
    }

    #[test]
    fn token_count_127_128_129_boundaries_include_partial_current_block() {
        for (tokens, visible_blocks, local_block, expected_ids) in [
            (127_u64, 1_u64, 0_u64, vec![0_u64]),
            (128, 1, 0, vec![0]),
            (129, 2, 1, vec![1, 0]),
        ] {
            let selection = zero_selection(descriptor(tokens, 4, 3, 5));
            assert_eq!(selection.visible_block_count(), visible_blocks);
            assert_eq!(selection.local_block_id(), local_block);
            for group in selection.groups() {
                assert_eq!(group.ranked_block_ids(), expected_ids);
            }
        }
    }

    #[test]
    fn local_block_reserves_one_of_sixteen_slots_and_ties_use_low_ids() {
        let tokens = 17_u64 * u64::from(MINIMAX_M3_MSA_BLOCK_SIZE);
        let descriptor = descriptor(tokens, 4, 1, 1);
        let mut queries = vec![0.0; 4 * 128];
        for group in 0..4 {
            queries[group * 128] = 1.0;
        }
        let mut keys = vec![0.0; tokens as usize * 128];
        for block in 0..17_usize {
            let score = if block == 16 { -1.0e30 } else { 1.0 };
            for token in block * 128..(block + 1) * 128 {
                keys[token * 128] = score;
            }
        }
        let selection = reference_minimax_m3_msa_selection(MiniMaxM3MsaIndexInput {
            descriptor,
            index_queries: &queries,
            index_keys: &keys,
        })
        .unwrap();
        let expected: Vec<u64> = std::iter::once(16).chain(0..15).collect();
        for group in selection.groups() {
            assert_eq!(group.ranked_block_ids(), expected);
            assert!(!group.ranked_block_ids().contains(&15));
        }
    }

    #[test]
    fn four_groups_select_independently_and_stable_ties_are_deterministic() {
        let tokens = 3_u64 * u64::from(MINIMAX_M3_MSA_BLOCK_SIZE);
        let descriptor = descriptor(tokens, 8, 3, 5);
        let mut queries = vec![0.0; 4 * 128];
        for group in 0..4 {
            queries[group * 128 + group] = 1.0;
        }
        let mut keys = vec![0.0; tokens as usize * 128];
        let block_scores = [[10.0, 0.0, 5.0, 1.0], [0.0, 10.0, 5.0, 1.0]];
        for (block, scores) in block_scores.into_iter().enumerate() {
            for token in block * 128..(block + 1) * 128 {
                keys[token * 128..token * 128 + 4].copy_from_slice(&scores);
            }
        }
        let selection = reference_minimax_m3_msa_selection(MiniMaxM3MsaIndexInput {
            descriptor,
            index_queries: &queries,
            index_keys: &keys,
        })
        .unwrap();
        assert_eq!(selection.group(0).unwrap().ranked_block_ids(), &[2, 0, 1]);
        assert_eq!(selection.group(1).unwrap().ranked_block_ids(), &[2, 1, 0]);
        assert_eq!(selection.group(2).unwrap().ranked_block_ids(), &[2, 0, 1]);
        assert_eq!(selection.group(3).unwrap().ranked_block_ids(), &[2, 0, 1]);
    }

    #[test]
    fn exact_softmax_uses_causal_partial_block_and_non_aligned_features() {
        let descriptor = descriptor(129, 8, 3, 5);
        let selection = zero_selection(descriptor);
        let mut queries = vec![0.0; 8 * 3];
        for head in 0..8 {
            queries[head * 3] = 1.0;
        }
        let mut keys = vec![0.0; 129 * 4 * 3];
        let mut values = vec![0.0; 129 * 4 * 5];
        for token in 0..129 {
            for group in 0..4 {
                keys[(token * 4 + group) * 3] = if token == 128 { 1.0e30 } else { -1.0e30 };
                for feature in 0..5 {
                    values[(token * 4 + group) * 5 + feature] = if token == 128 {
                        (group * 10 + feature) as f32
                    } else {
                        0.0
                    };
                }
            }
        }
        let output = reference_minimax_m3_msa_attention(MiniMaxM3MsaAttentionInput {
            descriptor,
            selection: &selection,
            queries: &queries,
            keys: &keys,
            values: &values,
        })
        .unwrap();
        assert_eq!(output.values().len(), 8 * 5);
        for head in 0..8 {
            let group = head / 2;
            for feature in 0..5 {
                assert_eq!(
                    output.values()[head * 5 + feature],
                    (group * 10 + feature) as f32
                );
            }
        }
    }

    #[test]
    fn unselected_block_does_not_affect_main_attention() {
        let tokens = 17_u64 * u64::from(MINIMAX_M3_MSA_BLOCK_SIZE);
        let descriptor = descriptor(tokens, 4, 1, 1);
        let mut index_queries = vec![0.0; 4 * 128];
        for group in 0..4 {
            index_queries[group * 128] = 1.0;
        }
        let mut index_keys = vec![0.0; tokens as usize * 128];
        for block in 0..17_usize {
            let score = if block == 16 { -1.0e30 } else { 1.0 };
            for token in block * 128..(block + 1) * 128 {
                index_keys[token * 128] = score;
            }
        }
        let selection = reference_minimax_m3_msa_selection(MiniMaxM3MsaIndexInput {
            descriptor,
            index_queries: &index_queries,
            index_keys: &index_keys,
        })
        .unwrap();
        let queries = vec![0.0; 4];
        let keys = vec![0.0; tokens as usize * 4];
        let mut values = vec![1.0; tokens as usize * 4];
        for token in 15 * 128..16 * 128 {
            for group in 0..4 {
                values[token * 4 + group] = 1.0e30;
            }
        }
        let output = reference_minimax_m3_msa_attention(MiniMaxM3MsaAttentionInput {
            descriptor,
            selection: &selection,
            queries: &queries,
            keys: &keys,
            values: &values,
        })
        .unwrap();
        assert_eq!(output.values(), &[1.0; 4]);
    }

    #[test]
    fn shape_nonfinite_and_finite_overflow_fail_closed() {
        let descriptor = descriptor(129, 4, 3, 5);
        let queries = vec![0.0; 4 * 128];
        let keys = vec![0.0; 129 * 128];
        assert!(matches!(
            reference_minimax_m3_msa_selection(MiniMaxM3MsaIndexInput {
                descriptor,
                index_queries: &queries[..queries.len() - 1],
                index_keys: &keys,
            }),
            Err(MiniMaxM3MsaSemanticError::ElementCountMismatch {
                plane: MiniMaxM3MsaPlane::IndexQuery,
                ..
            })
        ));
        let mut nonfinite_keys = keys.clone();
        nonfinite_keys[17] = f32::NAN;
        assert!(matches!(
            reference_minimax_m3_msa_selection(MiniMaxM3MsaIndexInput {
                descriptor,
                index_queries: &queries,
                index_keys: &nonfinite_keys,
            }),
            Err(MiniMaxM3MsaSemanticError::NonFiniteInput {
                plane: MiniMaxM3MsaPlane::IndexKey,
                index: 17
            })
        ));
        let overflow_queries = vec![f32::MAX; 4 * 128];
        let overflow_keys = vec![f32::MAX; 129 * 128];
        assert!(matches!(
            reference_minimax_m3_msa_selection(MiniMaxM3MsaIndexInput {
                descriptor,
                index_queries: &overflow_queries,
                index_keys: &overflow_keys,
            }),
            Err(MiniMaxM3MsaSemanticError::NonFiniteIntermediate {
                stage: MiniMaxM3MsaStage::IndexDotProduct,
                ..
            })
        ));

        let selection = zero_selection(descriptor);
        let main_queries = vec![0.0; 4 * 3];
        let main_keys = vec![0.0; 129 * 4 * 3];
        let main_values = vec![0.0; 129 * 4 * 5];
        assert!(matches!(
            reference_minimax_m3_msa_attention(MiniMaxM3MsaAttentionInput {
                descriptor,
                selection: &selection,
                queries: &main_queries,
                keys: &main_keys[..main_keys.len() - 1],
                values: &main_values,
            }),
            Err(MiniMaxM3MsaSemanticError::ElementCountMismatch {
                plane: MiniMaxM3MsaPlane::MainKey,
                ..
            })
        ));
        let mut main_values = vec![0.0; 129 * 4 * 5];
        main_values[31] = f32::INFINITY;
        assert!(matches!(
            reference_minimax_m3_msa_attention(MiniMaxM3MsaAttentionInput {
                descriptor,
                selection: &selection,
                queries: &main_queries,
                keys: &main_keys,
                values: &main_values,
            }),
            Err(MiniMaxM3MsaSemanticError::NonFiniteInput {
                plane: MiniMaxM3MsaPlane::MainValue,
                index: 31
            })
        ));
    }

    #[test]
    fn invalid_descriptor_and_checked_arithmetic_precede_allocation() {
        assert!(matches!(
            MiniMaxM3MsaDescriptor::new(3, 0, 5, 3, 5),
            Err(MiniMaxM3MsaSemanticError::InvalidQueryHeadCount(5))
        ));
        assert!(matches!(
            MiniMaxM3MsaDescriptor::new(3, 0, 4, 0, 5),
            Err(MiniMaxM3MsaSemanticError::InvalidFeatureCount { .. })
        ));
        assert_eq!(
            MiniMaxM3MsaDescriptor::new(3, u64::MAX, 4, 3, 5),
            Err(MiniMaxM3MsaSemanticError::PositionOverflow)
        );
        let huge = MiniMaxM3MsaDescriptor::new(3, u64::MAX - 1, 4, 3, 5).unwrap();
        assert!(matches!(
            reference_minimax_m3_msa_selection(MiniMaxM3MsaIndexInput {
                descriptor: huge,
                index_queries: &[],
                index_keys: &[],
            }),
            Err(MiniMaxM3MsaSemanticError::ElementCountOverflow {
                plane: MiniMaxM3MsaPlane::IndexKey
            })
        ));
    }
}
