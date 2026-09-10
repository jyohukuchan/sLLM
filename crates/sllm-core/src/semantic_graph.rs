//! Model-neutral rewrites over graphs that retain semantic operation
//! descriptors and tensor identities.
//!
//! Model adapters own their node metadata (weights, state boundaries, and
//! model-specific kinds).  This module only recognizes the numerical
//! `Add -> RmsNorm` contract and returns the information needed by an adapter
//! to replace the pair with `ResidualRmsNorm`.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsStr;

use crate::op::{ResidualRmsNormContract, SemanticOpDescriptor, SemanticOpKind};
use crate::tensor::TensorView;

/// Request-local rollback for the common residual/RMSNorm rewrite.
///
/// The default is enabled.  `0` is the explicit rollback and malformed
/// values fail closed.  Model adapters still decide whether their backend and
/// target can execute the fused descriptor.
pub(crate) const RESIDUAL_RMSNORM_FUSION_ENV: &str = "SLLM_PREPARED_RESIDUAL_RMSNORM_FUSION";

pub(crate) fn residual_rmsnorm_fusion_enabled(env_value: Option<&OsStr>) -> bool {
    match env_value {
        None => true,
        Some(value) if value == OsStr::new("1") => true,
        Some(value) if value == OsStr::new("0") => false,
        Some(_) => false,
    }
}

/// Return the process environment value used by normal request entry points.
pub(crate) fn residual_rmsnorm_fusion_enabled_from_env() -> bool {
    residual_rmsnorm_fusion_enabled(std::env::var_os(RESIDUAL_RMSNORM_FUSION_ENV).as_deref())
}

/// Minimal view of an adapter-owned graph node needed by the common matcher.
pub(crate) trait ResidualRmsNormGraphNode {
    fn semantic_operation(&self) -> Option<&SemanticOpDescriptor>;
    fn semantic_inputs(&self) -> &[usize];
    fn semantic_outputs(&self) -> &[usize];
    fn semantic_dependencies(&self) -> &[usize];
    fn semantic_boundary_after(&self) -> Option<crate::ExecutionBoundaryKind>;
}

/// One validated Add/RMSNorm dataflow pair.
#[derive(Clone, Debug)]
pub(crate) struct ResidualRmsNormPair {
    add_index: usize,
    norm_index: usize,
    operation: SemanticOpDescriptor,
    fused_inputs: [usize; 3],
    fused_outputs: [usize; 2],
    dependencies: Vec<usize>,
    boundary_after: Option<crate::ExecutionBoundaryKind>,
}

impl ResidualRmsNormPair {
    pub(crate) const fn add_index(&self) -> usize {
        self.add_index
    }

    pub(crate) const fn norm_index(&self) -> usize {
        self.norm_index
    }

    pub(crate) fn operation(&self) -> &SemanticOpDescriptor {
        &self.operation
    }

    pub(crate) const fn fused_inputs(&self) -> &[usize; 3] {
        &self.fused_inputs
    }

    pub(crate) const fn fused_outputs(&self) -> &[usize; 2] {
        &self.fused_outputs
    }

    pub(crate) fn dependencies(&self) -> &[usize] {
        &self.dependencies
    }

    pub(crate) const fn boundary_after(&self) -> Option<crate::ExecutionBoundaryKind> {
        self.boundary_after
    }
}

/// Safe common rewrite plan.  The plan deliberately has no model labels or
/// expected pair count.  An adapter can apply it while preserving its own
/// node kind and ownership metadata.
#[derive(Clone, Debug, Default)]
pub(crate) struct ResidualRmsNormRewritePlan {
    pairs_by_add: BTreeMap<usize, ResidualRmsNormPair>,
    pairs_by_norm: BTreeSet<usize>,
}

impl ResidualRmsNormRewritePlan {
    pub(crate) fn is_empty(&self) -> bool {
        self.pairs_by_add.is_empty()
    }

    pub(crate) fn pair_for_add(&self, index: usize) -> Option<&ResidualRmsNormPair> {
        self.pairs_by_add.get(&index)
    }

    pub(crate) fn is_norm_removed(&self, index: usize) -> bool {
        self.pairs_by_norm.contains(&index)
    }

    pub(crate) fn pairs(&self) -> impl Iterator<Item = &ResidualRmsNormPair> {
        self.pairs_by_add.values()
    }
}

/// Find every safe semantic Add -> RMSNorm pair.
///
/// A pair can be non-adjacent in the node vector when no intermediate node
/// consumes the Add result and no completion boundary is crossed.  The fused
/// operation is emitted at the Add position, so all original consumers retain
/// their tensor IDs and execution order.  Ambiguous or malformed candidates
/// are left decomposed; they do not make an otherwise valid graph unusable.
pub(crate) fn plan_residual_rmsnorm_rewrite<N: ResidualRmsNormGraphNode>(
    nodes: &[N],
    tensors: &[TensorView],
) -> ResidualRmsNormRewritePlan {
    let mut norms_by_input: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (index, node) in nodes.iter().enumerate() {
        let Some(operation) = node.semantic_operation() else {
            continue;
        };
        if operation.kind() != SemanticOpKind::RmsNorm
            || node.semantic_inputs().len() != 2
            || node.semantic_outputs().len() != 1
            || operation.inputs().len() != 2
            || operation.outputs().len() != 1
            || operation.rms_norm_contract().is_none()
        {
            continue;
        }
        norms_by_input
            .entry(node.semantic_inputs()[0])
            .or_default()
            .push(index);
    }

    let mut pairs_by_add = BTreeMap::new();
    let mut pairs_by_norm = BTreeSet::new();
    for (add_index, add_node) in nodes.iter().enumerate() {
        let Some(add_operation) = add_node.semantic_operation() else {
            continue;
        };
        if add_operation.kind() != SemanticOpKind::Add
            || add_node.semantic_inputs().len() != 2
            || add_node.semantic_outputs().len() != 1
            || add_operation.inputs().len() != 2
            || add_operation.outputs().len() != 1
        {
            continue;
        }
        let add_output = add_node.semantic_outputs()[0];
        let Some(norm_indices) = norms_by_input.get(&add_output) else {
            continue;
        };
        if norm_indices.len() != 1 {
            continue;
        }
        let norm_index = norm_indices[0];
        let norm_node = &nodes[norm_index];
        if norm_index <= add_index
            || add_node.semantic_boundary_after().is_some()
            || (norm_index != add_index + 1 && norm_node.semantic_boundary_after().is_some())
            || pairs_by_norm.contains(&norm_index)
            || !input_views_match(add_operation, add_node.semantic_inputs(), tensors)
            || !output_views_match(add_operation, add_node.semantic_outputs(), tensors)
            || !input_views_match(
                norm_node
                    .semantic_operation()
                    .expect("RMSNorm candidate has an operation"),
                norm_node.semantic_inputs(),
                tensors,
            )
            || !output_views_match(
                norm_node
                    .semantic_operation()
                    .expect("RMSNorm candidate has an operation"),
                norm_node.semantic_outputs(),
                tensors,
            )
            || norm_node
                .semantic_dependencies()
                .iter()
                .any(|dependency| *dependency > add_index)
            || !intermediate_movement_is_safe(nodes, add_index, norm_index, add_output)
        {
            continue;
        }
        let norm_operation = norm_node
            .semantic_operation()
            .expect("RMSNorm candidate has an operation");
        let Some(rms_contract) = norm_operation.rms_norm_contract() else {
            continue;
        };
        let Some(&scale) = norm_node.semantic_inputs().get(1) else {
            continue;
        };
        let Some(&normalized_output) = norm_node.semantic_outputs().first() else {
            continue;
        };
        let Some(operation) = SemanticOpDescriptor::new_residual_rms_norm_with_contract(
            vec![
                tensors[add_node.semantic_inputs()[0]].clone(),
                tensors[add_node.semantic_inputs()[1]].clone(),
                tensors[scale].clone(),
            ],
            vec![
                tensors[add_output].clone(),
                tensors[normalized_output].clone(),
            ],
            ResidualRmsNormContract::from_rms_norm(rms_contract),
        )
        .ok() else {
            continue;
        };

        let mut dependencies = add_node.semantic_dependencies().to_vec();
        dependencies.extend(norm_node.semantic_dependencies());
        dependencies.sort_unstable();
        dependencies.dedup();
        dependencies.retain(|dependency| *dependency != add_index && *dependency != norm_index);
        // The admission checks prohibit crossing the Add boundary or moving
        // a non-adjacent Norm boundary ahead of intermediate work.
        let boundary_after = norm_node.semantic_boundary_after();

        pairs_by_norm.insert(norm_index);
        pairs_by_add.insert(
            add_index,
            ResidualRmsNormPair {
                add_index,
                norm_index,
                operation,
                fused_inputs: [
                    add_node.semantic_inputs()[0],
                    add_node.semantic_inputs()[1],
                    scale,
                ],
                fused_outputs: [add_output, normalized_output],
                dependencies,
                boundary_after,
            },
        );
    }

    ResidualRmsNormRewritePlan {
        pairs_by_add,
        pairs_by_norm,
    }
}

fn input_views_match(
    operation: &SemanticOpDescriptor,
    tensor_ids: &[usize],
    tensors: &[TensorView],
) -> bool {
    operation.inputs().len() == tensor_ids.len()
        && tensor_ids
            .iter()
            .zip(operation.inputs())
            .all(|(id, operation)| tensors.get(*id) == Some(operation))
}

fn output_views_match(
    operation: &SemanticOpDescriptor,
    tensor_ids: &[usize],
    tensors: &[TensorView],
) -> bool {
    operation.outputs().len() == tensor_ids.len()
        && tensor_ids
            .iter()
            .zip(operation.outputs())
            .all(|(id, operation)| tensors.get(*id) == Some(operation))
}

fn intermediate_movement_is_safe<N: ResidualRmsNormGraphNode>(
    nodes: &[N],
    add_index: usize,
    norm_index: usize,
    add_output: usize,
) -> bool {
    nodes[add_index + 1..norm_index].iter().all(|node| {
        !node.semantic_dependencies().contains(&add_index)
            && !node.semantic_inputs().contains(&add_output)
            && node.semantic_boundary_after().is_none()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct TestNode {
        operation: Option<SemanticOpDescriptor>,
        inputs: Vec<usize>,
        outputs: Vec<usize>,
        dependencies: Vec<usize>,
        boundary: Option<crate::ExecutionBoundaryKind>,
    }

    impl ResidualRmsNormGraphNode for TestNode {
        fn semantic_operation(&self) -> Option<&SemanticOpDescriptor> {
            self.operation.as_ref()
        }

        fn semantic_inputs(&self) -> &[usize] {
            &self.inputs
        }

        fn semantic_outputs(&self) -> &[usize] {
            &self.outputs
        }

        fn semantic_dependencies(&self) -> &[usize] {
            &self.dependencies
        }

        fn semantic_boundary_after(&self) -> Option<crate::ExecutionBoundaryKind> {
            self.boundary
        }
    }

    fn late_scale_fixture() -> (Vec<TestNode>, Vec<TensorView>) {
        let activation = TensorView::contiguous(crate::DType::Bf16, &[2, 4]).unwrap();
        let scale = TensorView::contiguous(crate::DType::Bf16, &[4]).unwrap();
        let tensors = vec![
            activation.clone(),
            activation.clone(),
            activation.clone(),
            scale.clone(),
            activation.clone(),
        ];
        let add = SemanticOpDescriptor::new(
            SemanticOpKind::Add,
            vec![activation.clone(), activation.clone()],
            vec![activation.clone()],
        )
        .unwrap();
        let norm = SemanticOpDescriptor::new_rms_norm(
            vec![activation.clone(), scale],
            vec![activation],
            1.0e-5,
            crate::RmsNormScaleMode::Direct,
        )
        .unwrap();
        let nodes = vec![
            TestNode {
                operation: Some(add),
                inputs: vec![0, 1],
                outputs: vec![2],
                dependencies: vec![],
                boundary: None,
            },
            TestNode {
                operation: None,
                inputs: vec![],
                outputs: vec![3],
                dependencies: vec![],
                boundary: None,
            },
            TestNode {
                operation: Some(norm),
                inputs: vec![2, 3],
                outputs: vec![4],
                dependencies: vec![1],
                boundary: None,
            },
        ];
        (nodes, tensors)
    }

    #[test]
    fn late_scale_producer_is_not_moved_before_fused_add() {
        let (nodes, tensors) = late_scale_fixture();
        assert!(plan_residual_rmsnorm_rewrite(&nodes, &tensors).is_empty());
    }

    #[test]
    fn fusion_does_not_cross_or_move_a_completion_boundary() {
        let (mut nodes, tensors) = late_scale_fixture();
        // Make scale a resident input and the intermediate node independent.
        nodes[1].outputs.clear();
        nodes[2].dependencies = vec![0];
        assert_eq!(
            plan_residual_rmsnorm_rewrite(&nodes, &tensors)
                .pairs()
                .count(),
            1
        );
        for boundary_index in [0, 1, 2] {
            nodes[boundary_index].boundary = Some(crate::ExecutionBoundaryKind::StatePublication);
            assert!(plan_residual_rmsnorm_rewrite(&nodes, &tensors).is_empty());
            nodes[boundary_index].boundary = None;
        }
        nodes.remove(1);
        nodes[1].boundary = Some(crate::ExecutionBoundaryKind::StatePublication);
        let plan = plan_residual_rmsnorm_rewrite(&nodes, &tensors);
        assert_eq!(plan.pairs().count(), 1);
        assert_eq!(
            plan.pair_for_add(0).unwrap().boundary_after(),
            nodes[1].boundary
        );
    }
}
