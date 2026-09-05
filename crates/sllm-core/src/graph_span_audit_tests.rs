use super::*;

struct GraphOwner {
    logical_members: Vec<(String, DispatchEvidence)>,
    dispatch: DispatchEvidence,
}

impl SegmentCompletionOwner for GraphOwner {
    fn query(&mut self) -> Result<ExecutionState, ExecutionError> {
        Ok(ExecutionState::Success)
    }

    fn dispatch(&self) -> &DispatchEvidence {
        &self.dispatch
    }

    fn graph_logical_dispatches(&self) -> Option<&[(String, DispatchEvidence)]> {
        Some(&self.logical_members)
    }
}

fn evidence(dispatch_count: u32, kernel_id: u32, symbol: &str) -> DispatchEvidence {
    DispatchEvidence {
        abi_version: 1,
        info_version: 1,
        dispatch_id: u64::from(kernel_id),
        dispatch_count,
        kernel_id,
        workgroup_size_x: 256,
        grid_size_x: 1,
        row_count: 1,
        normalized_size: 1,
        backend: 1,
        fallback_allowed: false,
        fallback_used: false,
        kernel_symbol: symbol.to_owned(),
        device_symbol: symbol.to_owned(),
        target: "graph-audit-test".to_owned(),
    }
}

#[test]
fn graph_audit_preserves_member_labels_dispatch_counts_and_one_physical_replay() {
    let logical_members = vec![
        (
            "layer.17.mlp_gate_matmul.qwen38_projection_pack2".to_owned(),
            evidence(3, 17, "projection_gate"),
        ),
        (
            "layer.17.mlp_up_matmul.qwen38_projection_pack2".to_owned(),
            evidence(3, 18, "projection_up"),
        ),
    ];
    let mut segment = ExecutionSegment::default();
    segment.retain_test_owner(
        "qwen38.graph_span",
        GraphOwner {
            logical_members,
            dispatch: evidence(99, 99, "physical_graph"),
        },
    );

    let mut audit = ExecutionAuditAccumulator::new(1);
    segment
        .flush(ExecutionBoundaryKind::TerminalReadback, &mut audit)
        .unwrap();
    let snapshot = audit.snapshot().unwrap();

    assert_eq!(snapshot.submission_count(), 2);
    assert_eq!(snapshot.kernel_dispatch_count(), 6);
    assert_eq!(snapshot.graph_replay_count(), 1);
    assert_eq!(snapshot.projection_pack_submission_count(), 2);
    assert_eq!(snapshot.projection_pack_member_count(), 4);
    assert_eq!(snapshot.projection_pack_activation_quantize_count(), 2);
    assert_eq!(snapshot.boundary_count(), 1);
    assert_eq!(
        snapshot
            .kernel_dispatches_by_identity()
            .get(&(17, "projection_gate".to_owned())),
        Some(&3)
    );
    assert_eq!(
        snapshot
            .kernel_dispatches_by_identity()
            .get(&(18, "projection_up".to_owned())),
        Some(&3)
    );
    assert_eq!(
        snapshot
            .kernel_dispatches_by_identity()
            .get(&(99, "physical_graph".to_owned())),
        None
    );
    assert!(segment.is_empty());
}
