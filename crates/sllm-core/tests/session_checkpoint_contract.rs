use sha2::{Digest, Sha256};
use sllm_core::{
    CheckpointError, CheckpointIdentity, CheckpointPayload, CheckpointStore, KvCacheEncoding,
    OpaqueStatePlane, SessionCheckpoint, StateLayerMetadataV1, StateOwnerKindV1, StatePlaneKindV1,
};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static TEST_COUNTER: AtomicU64 = AtomicU64::new(1);

fn temporary_directory(label: &str) -> PathBuf {
    let id = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "sllm-session-checkpoint-{label}-{}-{id}",
        std::process::id()
    ));
    fs::create_dir(&path).expect("create test directory");
    #[cfg(unix)]
    fs::set_permissions(&path, std::os::unix::fs::PermissionsExt::from_mode(0o700))
        .expect("secure test directory");
    path
}

fn cleanup(path: &Path) {
    let _ = fs::remove_dir_all(path);
}

fn identity(tokens: &[u32]) -> CheckpointIdentity {
    CheckpointIdentity::for_tokens(
        format!("sha256:{}", "a".repeat(64)),
        "artifact:sha256:recipe",
        "adapter-v1",
        "renderer-v1:sha256:renderer",
        "tokenizer-v1:sha256:tokenizer",
        "gfx1030|wave32|fp16",
        "sha256:plan",
        tokens,
        KvCacheEncoding::Fp8E4M3Fn,
        [2u8; 32],
        [3u8; 32],
    )
    .expect("valid identity")
}

fn checkpoint() -> SessionCheckpoint {
    let tokens = vec![1, 7, 42, 99];
    SessionCheckpoint::new(
        identity(&tokens),
        20,
        16,
        3,
        CheckpointPayload {
            token_history: tokens,
            conversation: b"system: hi\nuser: hello".to_vec(),
            state_layers: vec![
                StateLayerMetadataV1 {
                    owner: StateOwnerKindV1::Kv,
                    layer_id: 7,
                    published_length: 4,
                    generation: 3,
                    active_slot: None,
                },
                StateLayerMetadataV1 {
                    owner: StateOwnerKindV1::LinearAttention,
                    layer_id: 11,
                    published_length: 4,
                    generation: 9,
                    active_slot: Some(1),
                },
            ],
            state_planes: vec![
                OpaqueStatePlane {
                    owner: StateOwnerKindV1::Kv,
                    layer_id: 7,
                    plane: StatePlaneKindV1::KvKey,
                    bytes: vec![0x11, 0x22, 0x33],
                },
                OpaqueStatePlane {
                    owner: StateOwnerKindV1::Kv,
                    layer_id: 7,
                    plane: StatePlaneKindV1::KvValue,
                    bytes: vec![0x44, 0x55],
                },
                OpaqueStatePlane {
                    owner: StateOwnerKindV1::Kv,
                    layer_id: 7,
                    plane: StatePlaneKindV1::KvKeyScale,
                    bytes: vec![0x56],
                },
                OpaqueStatePlane {
                    owner: StateOwnerKindV1::Kv,
                    layer_id: 7,
                    plane: StatePlaneKindV1::KvValueScale,
                    bytes: vec![0x57],
                },
                OpaqueStatePlane {
                    owner: StateOwnerKindV1::LinearAttention,
                    layer_id: 11,
                    plane: StatePlaneKindV1::LinearConvSlot0,
                    bytes: vec![0x90, 0x91],
                },
                OpaqueStatePlane {
                    owner: StateOwnerKindV1::LinearAttention,
                    layer_id: 11,
                    plane: StatePlaneKindV1::LinearConvSlot1,
                    bytes: vec![0x92],
                },
                OpaqueStatePlane {
                    owner: StateOwnerKindV1::LinearAttention,
                    layer_id: 11,
                    plane: StatePlaneKindV1::LinearRecurrentSlot0,
                    bytes: vec![0x93],
                },
                OpaqueStatePlane {
                    owner: StateOwnerKindV1::LinearAttention,
                    layer_id: 11,
                    plane: StatePlaneKindV1::LinearRecurrentSlot1,
                    bytes: vec![0x94],
                },
                OpaqueStatePlane {
                    owner: StateOwnerKindV1::LinearAttention,
                    layer_id: 11,
                    plane: StatePlaneKindV1::LinearScratch,
                    bytes: vec![0x95],
                },
            ],
            sampler_state: vec![0xA0],
            grammar_state: vec![0xB0, 0xB1],
            stop_state: vec![0xC0],
        },
    )
    .expect("valid checkpoint")
}

fn recompute_envelope_checksum(bytes: &mut [u8]) {
    bytes[28..60].fill(0);
    let digest: [u8; 32] = Sha256::digest(&*bytes).into();
    bytes[28..60].copy_from_slice(&digest);
}

#[test]
fn roundtrip_preserves_identity_and_all_opaque_planes() {
    let original = checkpoint();
    let bytes = original.encode().expect("encode");
    let restored = SessionCheckpoint::decode(&bytes).expect("decode");
    assert_eq!(restored, original);
    let expected = identity(&original.payload.token_history);
    assert!(SessionCheckpoint::decode_with_identity(&bytes, Some(&expected)).is_ok());
}

#[test]
fn every_kv_encoding_preserves_value_scale_and_outer_planes_exactly() {
    for encoding in [
        KvCacheEncoding::Fp16,
        KvCacheEncoding::Fp8E4M3Fn,
        KvCacheEncoding::Fp8E4M3FnStatic,
        KvCacheEncoding::Nvfp4,
        KvCacheEncoding::Fp8E4M3Block16,
        KvCacheEncoding::Fp8E5M2Block16,
        KvCacheEncoding::Mxfp8E4,
        KvCacheEncoding::Mxfp8E5,
    ] {
        let mut original = checkpoint();
        original.header.identity.kv_encoding = encoding;
        let kv_planes: &[StatePlaneKindV1] = match encoding {
            KvCacheEncoding::Fp16 | KvCacheEncoding::Fp8E4M3FnStatic => {
                &[StatePlaneKindV1::KvKey, StatePlaneKindV1::KvValue]
            }
            KvCacheEncoding::Fp8E4M3Fn
            | KvCacheEncoding::Fp8E4M3Block16
            | KvCacheEncoding::Fp8E5M2Block16
            | KvCacheEncoding::Mxfp8E4
            | KvCacheEncoding::Mxfp8E5 => &[
                StatePlaneKindV1::KvKey,
                StatePlaneKindV1::KvValue,
                StatePlaneKindV1::KvKeyScale,
                StatePlaneKindV1::KvValueScale,
            ],
            KvCacheEncoding::Nvfp4 => &[
                StatePlaneKindV1::KvKey,
                StatePlaneKindV1::KvValue,
                StatePlaneKindV1::KvKeyScale,
                StatePlaneKindV1::KvValueScale,
                StatePlaneKindV1::KvKeyOuterScale,
                StatePlaneKindV1::KvValueOuterScale,
            ],
        };
        original.payload.state_planes = kv_planes
            .iter()
            .copied()
            .enumerate()
            .map(|(index, plane)| OpaqueStatePlane {
                owner: StateOwnerKindV1::Kv,
                layer_id: 7,
                plane,
                bytes: vec![index as u8 + 1; index + 1],
            })
            .chain(
                [
                    StatePlaneKindV1::LinearConvSlot0,
                    StatePlaneKindV1::LinearConvSlot1,
                    StatePlaneKindV1::LinearRecurrentSlot0,
                    StatePlaneKindV1::LinearRecurrentSlot1,
                    StatePlaneKindV1::LinearScratch,
                ]
                .into_iter()
                .enumerate()
                .map(|(index, plane)| OpaqueStatePlane {
                    owner: StateOwnerKindV1::LinearAttention,
                    layer_id: 11,
                    plane,
                    bytes: vec![0x80 + index as u8; index + 2],
                }),
            )
            .collect();
        let bytes = original.encode().expect("encode all native planes");
        let restored =
            SessionCheckpoint::decode_with_identity(&bytes, Some(&original.header.identity))
                .expect("restore exact encoding");
        assert_eq!(restored, original);
    }
}

#[test]
fn lowbit_checkpoint_tags_and_descriptor_digests_do_not_collide() {
    let mut original = checkpoint();
    original.header.identity.kv_encoding = KvCacheEncoding::Fp8E4M3Block16;
    original.header.identity.kv_descriptor_digest = [0xe4; 32];
    let bytes = original.encode().unwrap();

    let mut wrong_logical = original.header.identity.clone();
    wrong_logical.kv_encoding = KvCacheEncoding::Fp8E5M2Block16;
    assert!(matches!(
        SessionCheckpoint::decode_with_identity(&bytes, Some(&wrong_logical)),
        Err(CheckpointError::IdentityMismatch {
            field: "kv_encoding",
            ..
        })
    ));

    let mut wrong_physical_descriptor = original.header.identity.clone();
    wrong_physical_descriptor.kv_descriptor_digest = [0xf0; 32];
    assert!(matches!(
        SessionCheckpoint::decode_with_identity(&bytes, Some(&wrong_physical_descriptor)),
        Err(CheckpointError::IdentityMismatch {
            field: "kv_descriptor_digest",
            ..
        })
    ));

    let mut mx = checkpoint();
    mx.header.identity.kv_encoding = KvCacheEncoding::Mxfp8E4;
    mx.header.identity.kv_descriptor_digest = [0x4d; 32];
    let mx_bytes = mx.encode().unwrap();
    for wrong_encoding in [KvCacheEncoding::Mxfp8E5, KvCacheEncoding::Fp8E4M3Block16] {
        let mut wrong = mx.header.identity.clone();
        wrong.kv_encoding = wrong_encoding;
        assert!(matches!(
            SessionCheckpoint::decode_with_identity(&mx_bytes, Some(&wrong)),
            Err(CheckpointError::IdentityMismatch {
                field: "kv_encoding",
                ..
            })
        ));
    }
}

#[test]
fn state_layer_identity_and_active_slot_are_strict() {
    let mut missing_scale = checkpoint();
    missing_scale
        .payload
        .state_planes
        .retain(|plane| plane.plane != StatePlaneKindV1::KvValueScale);
    assert!(matches!(
        missing_scale.encode(),
        Err(CheckpointError::Invalid(_))
    ));

    let mut wrong_owner = checkpoint();
    wrong_owner.payload.state_planes[0].owner = StateOwnerKindV1::LinearAttention;
    assert!(matches!(
        wrong_owner.encode(),
        Err(CheckpointError::Invalid(_))
    ));

    let mut missing_layer = checkpoint();
    missing_layer.payload.state_planes[0].layer_id = 999;
    assert!(matches!(
        missing_layer.encode(),
        Err(CheckpointError::Invalid(_))
    ));

    let mut wrong_slot = checkpoint();
    wrong_slot.payload.state_layers[1].active_slot = Some(2);
    assert!(matches!(
        wrong_slot.encode(),
        Err(CheckpointError::Invalid(_))
    ));

    let mut kv_slot = checkpoint();
    kv_slot.payload.state_layers[0].active_slot = Some(0);
    assert!(matches!(kv_slot.encode(), Err(CheckpointError::Invalid(_))));
}

#[test]
fn oversized_header_and_unknown_kv_encoding_fail_closed() {
    let mut oversized = checkpoint();
    let field = "x".repeat(1024);
    oversized.header.identity.derived_artifact_identity = field.clone();
    oversized.header.identity.adapter_identity = field.clone();
    oversized.header.identity.renderer_identity = field.clone();
    oversized.header.identity.tokenizer_identity = field.clone();
    oversized.header.identity.target_semantics = field.clone();
    oversized.header.identity.plan_digest = field;
    assert!(matches!(
        oversized.encode(),
        Err(CheckpointError::Bounds(_))
    ));

    let mut bytes = checkpoint().encode().expect("encode");
    let mut cursor = 96usize;
    for _ in 0..7 {
        let length = u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap()) as usize;
        cursor += 4 + length;
    }
    cursor += 32;
    bytes[cursor] = 255;
    recompute_envelope_checksum(&mut bytes);
    assert!(matches!(
        SessionCheckpoint::decode(&bytes),
        Err(CheckpointError::Corrupt(_))
    ));
}

#[test]
fn debug_output_redacts_tokens_conversation_and_state_bytes() {
    let checkpoint = checkpoint();
    let debug = format!("{checkpoint:?}");
    assert!(!debug.contains("system: hi"));
    assert!(!debug.contains("[1, 7, 42, 99]"));
    assert!(!debug.contains("[144, 145]"));
    assert!(debug.contains("token_count: 4"));
    assert!(debug.contains("conversation_bytes: 22"));
}

#[test]
fn malformed_envelopes_fail_closed() {
    let mut reversed_position = checkpoint();
    reversed_position.header.absolute_position = 15;
    assert!(matches!(
        reversed_position.validate(),
        Err(CheckpointError::Invalid(_))
    ));
    let mut oversized_delta = checkpoint();
    oversized_delta.header.absolute_position = u64::MAX;
    oversized_delta.header.logical_position = 0;
    assert!(matches!(
        oversized_delta.validate(),
        Err(CheckpointError::Bounds(_))
    ));

    let bytes = checkpoint().encode().expect("encode");

    let mut corrupt = bytes.clone();
    let last = corrupt.len() - 1;
    corrupt[last] ^= 0x80;
    assert!(matches!(
        SessionCheckpoint::decode(&corrupt),
        Err(CheckpointError::Corrupt(_))
    ));

    let truncated = &bytes[..bytes.len() - 1];
    assert!(matches!(
        SessionCheckpoint::decode(truncated),
        Err(CheckpointError::Truncated)
    ));

    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(matches!(
        SessionCheckpoint::decode(&trailing),
        Err(CheckpointError::TrailingBytes)
    ));

    let mut version = bytes.clone();
    version[8..10].copy_from_slice(&99u16.to_le_bytes());
    assert!(matches!(
        SessionCheckpoint::decode(&version),
        Err(CheckpointError::UnsupportedVersion(99))
    ));
}

#[test]
fn duplicate_and_overlapping_sections_are_rejected() {
    let bytes = checkpoint().encode().expect("encode");
    let identity_len =
        u32::from_le_bytes(bytes[60..64].try_into().expect("identity length")) as usize;
    let table = 96 + identity_len;

    let mut duplicate = bytes.clone();
    let first_kind = duplicate[table..table + 2].to_vec();
    duplicate[table + 56..table + 58].copy_from_slice(&first_kind);
    recompute_envelope_checksum(&mut duplicate);
    assert!(matches!(
        SessionCheckpoint::decode(&duplicate),
        Err(CheckpointError::Corrupt(_))
    ));

    let mut overlap = bytes.clone();
    let first_offset = overlap[table + 8..table + 16].to_vec();
    overlap[table + 56 + 8..table + 56 + 16].copy_from_slice(&first_offset);
    recompute_envelope_checksum(&mut overlap);
    assert!(matches!(
        SessionCheckpoint::decode(&overlap),
        Err(CheckpointError::Corrupt(_))
    ));
}

#[test]
fn identity_mismatch_is_rejected_before_restore() {
    let bytes = checkpoint().encode().expect("encode");
    let mut wrong = identity(&[1, 7, 42, 99]);
    wrong.adapter_identity = "other-adapter".into();
    assert!(matches!(
        SessionCheckpoint::decode_with_identity(&bytes, Some(&wrong)),
        Err(CheckpointError::IdentityMismatch {
            field: "adapter_identity",
            ..
        })
    ));
}

#[cfg(unix)]
#[test]
fn store_is_atomic_owner_checked_and_quota_bounded() {
    use std::os::unix::fs::{PermissionsExt, symlink};

    let root = temporary_directory("store");
    fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).expect("secure root");
    let cp = checkpoint();
    let bytes = cp.encode().expect("encode");
    let store = CheckpointStore::with_limits(&root, bytes.len() as u64 * 2, bytes.len() as u64 + 1)
        .expect("store");
    let path = store.save("session-1", &cp).expect("atomic save");
    assert_eq!(
        fs::metadata(&path).expect("metadata").permissions().mode() & 0o777,
        0o600
    );
    assert_eq!(store.usage_bytes().expect("usage"), bytes.len() as u64);
    assert_eq!(
        store.load("session-1", &cp.header.identity).expect("load"),
        cp
    );
    assert_eq!(fs::read_dir(&root).expect("entries").count(), 1);

    fs::hard_link(&path, root.join("session-hardlink.ckpt")).expect("hard link");
    assert!(matches!(
        store.load("session-1", &cp.header.identity),
        Err(CheckpointError::Security(_))
    ));
    fs::remove_file(root.join("session-hardlink.ckpt")).expect("remove hard link");

    assert!(matches!(
        store.save("../escape", &cp),
        Err(CheckpointError::PathViolation(_))
    ));
    symlink(&path, root.join("session-link.ckpt")).expect("symlink");
    assert!(matches!(
        store.load("session-link", &cp.header.identity),
        Err(CheckpointError::Security(_))
    ));
    let mut open = fs::metadata(&path).expect("metadata").permissions();
    open.set_mode(0o644);
    fs::set_permissions(&path, open).expect("open permissions");
    assert!(matches!(
        store.load("session-1", &cp.header.identity),
        Err(CheckpointError::Security(_))
    ));

    cleanup(&root);
}

#[test]
fn store_rejects_quota_before_creating_temp_file() {
    let root = temporary_directory("quota");
    let cp = checkpoint();
    let bytes = cp.encode().expect("encode");
    let store = CheckpointStore::with_limits(&root, bytes.len() as u64 - 1, bytes.len() as u64)
        .expect("store");
    assert!(matches!(
        store.save("too-large", &cp),
        Err(CheckpointError::QuotaExceeded { .. })
    ));
    assert_eq!(fs::read_dir(&root).expect("entries").count(), 0);
    cleanup(&root);
}
