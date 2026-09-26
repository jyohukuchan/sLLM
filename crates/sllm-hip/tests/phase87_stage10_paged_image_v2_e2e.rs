//! Phase 87 Stage 10: Rust/core to HIP Paged image V2 roundtrip.
//!
//! This is an exact GPU test. It intentionally has no fake backend or CPU
//! fallback path; the ignored test is enabled by the exact gfx1030/gfx1201
//! integration command.

use std::env;
use std::sync::Arc;
use std::time::Duration;

use sllm_core::{
    AccessMode, Backend, CausalAttentionDescriptor, DType, ExecutionSessionRequest, ExecutionState,
    KvCacheEncoding, KvFp8PhysicalVariant, KvPagedImageMetadataV1, KvPagedImageTopologyV1,
    KvStateDescriptor, TensorView,
};
use sllm_hip::HipBackend;

const TIMEOUT: Duration = Duration::from_secs(60);
const PREFIX: usize = 129;
const KV_HEADS: usize = 2;
const QUERY_HEADS: usize = 8;
const HEAD_DIM: usize = 128;

fn target() -> String {
    env::var("SLLM_TEST_EXPECTED_TARGET")
        .or_else(|_| env::var("SLLM_HIP_TARGET"))
        .expect("exact target environment is required")
}

fn bf16_bytes(count: usize, seed: u32) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(count * 2);
    for index in 0..count {
        let value = (seed.wrapping_add(index as u32) & 0x7fff) as u16;
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn upload(
    session: &sllm_core::ExecutionSession,
    queue: &sllm_core::ExecutionQueue,
    buffer: &sllm_core::ExecutionBuffer,
    bytes: Vec<u8>,
) {
    let range = buffer
        .range(0, bytes.len() as u64)
        .expect("upload range must be valid");
    let mut transfer = session
        .upload(queue, range, Arc::from(bytes.into_boxed_slice()))
        .expect("GPU upload must be admitted");
    assert_eq!(
        transfer.wait(TIMEOUT).expect("GPU upload must complete"),
        ExecutionState::Success
    );
}

fn bind(
    session: &sllm_core::ExecutionSession,
    buffer: &sllm_core::ExecutionBuffer,
    shape: &[usize],
    access: AccessMode,
) -> sllm_core::OwnedTensorBinding {
    session
        .bind(
            buffer,
            TensorView::contiguous(DType::Bf16, shape).expect("tensor view"),
            access,
        )
        .expect("owned tensor binding")
}

fn append(
    session: &sllm_core::ExecutionSession,
    state: &sllm_core::KvState,
    queue: &sllm_core::ExecutionQueue,
    key: sllm_core::OwnedTensorBinding,
    value: sllm_core::OwnedTensorBinding,
    start: u64,
    count: usize,
) {
    let mut submission = session
        .append_kv_state(state, queue, key, value, start, start)
        .expect("Paged append must be admitted");
    assert_eq!(
        submission
            .wait(TIMEOUT)
            .expect("Paged append must complete"),
        ExecutionState::Success
    );
    assert_eq!(submission.request().token_count(), count as u64);
}

#[test]
#[ignore = "exact gfx1030/gfx1201 GPU integration test"]
fn rust_paged_image_v2_roundtrip_mxfp8_e4() {
    let expected_target = target();
    assert!(expected_target == "gfx1030" || expected_target == "gfx1201");

    let backend = HipBackend::connect().expect("HIP must be available");
    let session = backend
        .open_execution_session(
            ExecutionSessionRequest::new(0, expected_target.clone()).expect("session request"),
        )
        .expect("exact Paged HIP session");
    let queue = session.create_queue().expect("GPU queue");

    let descriptor = KvStateDescriptor::new_with_kv_mxfp8(
        0,
        256,
        KV_HEADS,
        HEAD_DIM,
        KvCacheEncoding::Mxfp8E4,
        KvFp8PhysicalVariant::OcpE4M3Fn,
    )
    .expect("MXFP8 E4 descriptor");
    let source = session
        .create_kv_state(descriptor)
        .expect("source Paged state");
    let destination = session
        .create_kv_state(descriptor)
        .expect("destination Paged state");

    let token_bytes = KV_HEADS * HEAD_DIM * 2;
    let prefix_key_buffer = session
        .allocate((PREFIX * token_bytes) as u64)
        .expect("prefix key buffer");
    let prefix_value_buffer = session
        .allocate((PREFIX * token_bytes) as u64)
        .expect("prefix value buffer");
    upload(
        &session,
        &queue,
        &prefix_key_buffer,
        bf16_bytes(PREFIX * KV_HEADS * HEAD_DIM, 17),
    );
    upload(
        &session,
        &queue,
        &prefix_value_buffer,
        bf16_bytes(PREFIX * KV_HEADS * HEAD_DIM, 71),
    );
    append(
        &session,
        &source,
        &queue,
        bind(
            &session,
            &prefix_key_buffer,
            &[PREFIX, KV_HEADS, HEAD_DIM],
            AccessMode::Read,
        ),
        bind(
            &session,
            &prefix_value_buffer,
            &[PREFIX, KV_HEADS, HEAD_DIM],
            AccessMode::Read,
        ),
        0,
        PREFIX,
    );

    let source_snapshot = session.kv_state_snapshot(&source).expect("source snapshot");
    assert_eq!(source_snapshot.length(), PREFIX as u64);
    let image = session
        .export_kv_state_image_v2(&source)
        .expect("Rust V2 image export");
    assert_eq!(image.paged_metadata().descriptor(), descriptor);
    assert_eq!(image.metadata().published_length, PREFIX as u64);

    // Core rejects malformed topology/section payloads before the adapter is
    // called. This is the malformed/missing-table gate for the Rust path.
    let (metadata, paged_metadata, planes) = image.clone().into_parts();
    if let KvPagedImageTopologyV1::LogicalTable(table) = paged_metadata.topology().clone() {
        let mut duplicate = table;
        if duplicate.len() > 1 {
            duplicate[1] = duplicate[0];
        }
        assert!(
            KvPagedImageMetadataV1::new(
                descriptor,
                PREFIX as u64,
                image.paged_metadata().generation(),
                0,
                0,
                paged_metadata.physical_block_capacity(),
                paged_metadata.plane_strides(),
                KvPagedImageTopologyV1::LogicalTable(duplicate),
            )
            .is_err()
        );
    }
    let mut missing_plane = planes;
    missing_plane[0].clear();
    assert!(
        sllm_core::ExecutionStateImageV2::new(
            metadata,
            image.paged_metadata().clone(),
            missing_plane
        )
        .is_err()
    );

    session
        .import_kv_state_image_v2(&destination, &image)
        .expect("Rust V2 image import");
    let destination_snapshot = session
        .kv_state_snapshot(&destination)
        .expect("destination snapshot after import");
    assert_eq!(destination_snapshot.length(), source_snapshot.length());
    let destination_image = session
        .export_kv_state_image_v2(&destination)
        .expect("destination V2 image export");
    assert_eq!(
        destination_image.paged_metadata().generation(),
        image.paged_metadata().generation()
    );
    assert_eq!(destination_image, image);

    let tail_key_buffer = session
        .allocate(token_bytes as u64)
        .expect("tail key buffer");
    let tail_value_buffer = session
        .allocate(token_bytes as u64)
        .expect("tail value buffer");
    upload(
        &session,
        &queue,
        &tail_key_buffer,
        bf16_bytes(KV_HEADS * HEAD_DIM, 101),
    );
    upload(
        &session,
        &queue,
        &tail_value_buffer,
        bf16_bytes(KV_HEADS * HEAD_DIM, 151),
    );
    for state in [&source, &destination] {
        append(
            &session,
            state,
            &queue,
            bind(
                &session,
                &tail_key_buffer,
                &[1, KV_HEADS, HEAD_DIM],
                AccessMode::Read,
            ),
            bind(
                &session,
                &tail_value_buffer,
                &[1, KV_HEADS, HEAD_DIM],
                AccessMode::Read,
            ),
            PREFIX as u64,
            1,
        );
    }

    let query_buffer = session
        .allocate((QUERY_HEADS * HEAD_DIM * 2) as u64)
        .expect("query buffer");
    let source_output = session
        .allocate((QUERY_HEADS * HEAD_DIM * 2) as u64)
        .expect("source output buffer");
    let destination_output = session
        .allocate((QUERY_HEADS * HEAD_DIM * 2) as u64)
        .expect("destination output buffer");
    upload(
        &session,
        &queue,
        &query_buffer,
        bf16_bytes(QUERY_HEADS * HEAD_DIM, 211),
    );
    let attention_descriptor =
        CausalAttentionDescriptor::new(PREFIX as u64, 1, 130).expect("attention descriptor");
    let mut source_attention = session
        .causal_attention(
            &source,
            &queue,
            bind(
                &session,
                &query_buffer,
                &[1, QUERY_HEADS, HEAD_DIM],
                AccessMode::Read,
            ),
            bind(
                &session,
                &source_output,
                &[1, QUERY_HEADS, HEAD_DIM],
                AccessMode::Write,
            ),
            attention_descriptor,
        )
        .expect("source attention");
    let mut destination_attention = session
        .causal_attention(
            &destination,
            &queue,
            bind(
                &session,
                &query_buffer,
                &[1, QUERY_HEADS, HEAD_DIM],
                AccessMode::Read,
            ),
            bind(
                &session,
                &destination_output,
                &[1, QUERY_HEADS, HEAD_DIM],
                AccessMode::Write,
            ),
            attention_descriptor,
        )
        .expect("destination attention");
    assert_eq!(
        source_attention
            .wait(TIMEOUT)
            .expect("source attention wait"),
        ExecutionState::Success
    );
    assert_eq!(
        destination_attention
            .wait(TIMEOUT)
            .expect("destination attention wait"),
        ExecutionState::Success
    );
    let mut source_readback = session
        .readback(
            &queue,
            source_output
                .range(0, (QUERY_HEADS * HEAD_DIM * 2) as u64)
                .unwrap(),
        )
        .expect("source output readback");
    let mut destination_readback = session
        .readback(
            &queue,
            destination_output
                .range(0, (QUERY_HEADS * HEAD_DIM * 2) as u64)
                .unwrap(),
        )
        .expect("destination output readback");
    assert_eq!(
        source_readback.wait(TIMEOUT).unwrap(),
        ExecutionState::Success
    );
    assert_eq!(
        destination_readback.wait(TIMEOUT).unwrap(),
        ExecutionState::Success
    );
    let mut source_bytes = vec![0_u8; QUERY_HEADS * HEAD_DIM * 2];
    let mut destination_bytes = vec![0_u8; QUERY_HEADS * HEAD_DIM * 2];
    source_readback.read_into(&mut source_bytes).unwrap();
    destination_readback
        .read_into(&mut destination_bytes)
        .unwrap();
    assert_eq!(source_bytes, destination_bytes);
}
