use std::sync::{Arc, Barrier};
use std::thread;

use sllm_core::{
    KvCacheEncoding, PrefixCacheConfigV1, PrefixCacheError, PrefixCacheKeyV1, PrefixCacheV1,
    PrefixCacheValueV1, PrefixKvLayoutV1, PrefixLookupKind, PrefixStateIdentityV1,
};

fn identity(model: &[u8]) -> PrefixStateIdentityV1 {
    PrefixStateIdentityV1::new(
        model,
        b"recipe:artifact-v1",
        b"adapter:base",
        b"renderer:qwen-template-v1",
        b"tokenizer:qwen-v1",
        KvCacheEncoding::Fp16,
        PrefixKvLayoutV1::new(4, 256).unwrap(),
        b"gfx1030:wave32:rope-v1",
        1,
    )
    .unwrap()
}

#[test]
fn configuration_and_identity_hard_caps_fail_closed() {
    assert_eq!(
        PrefixCacheConfigV1::new(257, 1, 1),
        Err(PrefixCacheError::InvalidQuota)
    );
    assert_eq!(
        PrefixCacheConfigV1::new(1, 1_048_577, 1),
        Err(PrefixCacheError::InvalidQuota)
    );
    let oversized = vec![b'x'; sllm_core::MAX_PREFIX_IDENTITY_BYTES + 1];
    assert_eq!(
        PrefixStateIdentityV1::new(
            oversized,
            b"recipe",
            b"adapter",
            b"renderer",
            b"tokenizer",
            KvCacheEncoding::Fp16,
            PrefixKvLayoutV1::new(1, 1).unwrap(),
            b"target",
            1,
        ),
        Err(PrefixCacheError::IdentityTooLarge)
    );
}

fn value(len: usize, resident_bytes: u64, marker: u8) -> PrefixCacheValueV1 {
    PrefixCacheValueV1::new(len as u64, resident_bytes, [marker; 32]).unwrap()
}

fn publish(cache: &PrefixCacheV1, id: &PrefixStateIdentityV1, tokens: &[u32], marker: u8) {
    let key = PrefixCacheKeyV1::new(id.clone(), tokens).unwrap();
    cache.publish(key, value(tokens.len(), 10, marker)).unwrap();
}

#[test]
fn longest_prefix_is_exact_and_covers_non_aligned_boundaries() {
    let cache = PrefixCacheV1::new(PrefixCacheConfigV1::new(16, 2_000, 10_000).unwrap());
    let id = identity(b"model-a");
    let tokens: Vec<u32> = (0..300).collect();
    for (marker, length) in [1_usize, 63, 64, 65, 255, 256, 257].into_iter().enumerate() {
        publish(&cache, &id, &tokens[..length], marker as u8);
    }

    for length in [1_usize, 63, 64, 65, 255, 256, 257] {
        let result = cache.lookup(&id, &tokens[..length]).unwrap().unwrap();
        assert_eq!(result.kind(), PrefixLookupKind::ExactHit);
        assert_eq!(result.matched_len(), length);
        assert_eq!(result.remaining_len(), 0);
        assert!(result.suffix(&tokens[..length]).unwrap().is_empty());
    }

    let partial = cache.lookup(&id, &tokens).unwrap().unwrap();
    assert_eq!(partial.kind(), PrefixLookupKind::PartialHit);
    assert_eq!(partial.matched_len(), 257);
    assert_eq!(partial.remaining_len(), 43);
    assert_eq!(partial.suffix(&tokens).unwrap(), &tokens[257..]);

    // A mismatch after token 64 cannot be hidden by a digest collision or a
    // shorter exact key: the longest matching entry is exactly length 64.
    let mut mismatch = tokens.clone();
    mismatch[64] = u32::MAX;
    let result = cache.lookup(&id, &mismatch).unwrap().unwrap();
    assert_eq!(result.matched_len(), 64);
    assert_eq!(result.kind(), PrefixLookupKind::PartialHit);

    // Empty input is a miss; publishing an empty state is rejected.
    assert!(cache.lookup(&id, &[]).unwrap().is_none());
    assert_eq!(
        PrefixCacheKeyV1::new(id.clone(), []),
        Err(PrefixCacheError::EmptyTokenSequence)
    );

    let audit = cache.audit_snapshot().unwrap();
    assert_eq!(audit.exact_hits(), 7);
    assert_eq!(audit.partial_hits(), 2);
    assert_eq!(audit.misses(), 1);
    assert!(audit.last_key_digest().is_some());
}

#[test]
fn identity_mismatch_is_a_miss_and_redacted_digest_changes() {
    let cache = PrefixCacheV1::new(PrefixCacheConfigV1::new(4, 100, 100).unwrap());
    let id_a = identity(b"model-a");
    let id_b = identity(b"model-b");
    let tokens = [7_u32, 8, 9];
    let key = PrefixCacheKeyV1::new(id_a.clone(), tokens).unwrap();
    let digest_a = key.redacted_digest();
    cache.publish(key, value(tokens.len(), 12, 1)).unwrap();

    assert!(cache.lookup(&id_b, &tokens).unwrap().is_none());
    let hit = cache.lookup(&id_a, &tokens).unwrap().unwrap();
    assert_ne!(digest_a, [0_u8; 32]);
    assert_eq!(hit.lease().redacted_digest(), digest_a);
    assert_eq!(cache.audit_snapshot().unwrap().misses(), 1);
}

#[test]
fn debug_output_contains_only_redacted_identity_and_token_metadata() {
    let key = PrefixCacheKeyV1::new(identity(b"adapter-secret"), [91, 92, 93]).unwrap();
    let debug = format!("{key:?}");
    assert!(!debug.contains("adapter-secret"));
    assert!(!debug.contains("[91, 92, 93]"));
    assert!(debug.contains("<redacted>"));
    assert!(debug.contains("token_count: 3"));
}

#[test]
fn active_lease_blocks_eviction_until_released() {
    let cache = PrefixCacheV1::new(PrefixCacheConfigV1::new(1, 100, 100).unwrap());
    let id = identity(b"model-a");
    let first = [1_u32, 2, 3];
    let second = [4_u32, 5, 6];
    publish(&cache, &id, &first, 1);
    let lease = cache.lookup(&id, &first).unwrap().unwrap().into_lease();
    assert_eq!(lease.reader_count(), 1);
    let second_key = PrefixCacheKeyV1::new(id.clone(), second).unwrap();
    assert_eq!(
        cache.publish(second_key.clone(), value(second.len(), 10, 2)),
        Err(PrefixCacheError::QuotaExceeded)
    );
    assert_eq!(cache.audit_snapshot().unwrap().lease_eviction_blocks(), 1);
    drop(lease);

    cache
        .publish(second_key, value(second.len(), 10, 2))
        .unwrap();
    assert!(cache.lookup(&id, &first).unwrap().is_none());
    assert!(cache.lookup(&id, &second).unwrap().is_some());
    assert_eq!(cache.audit_snapshot().unwrap().evictions(), 1);
}

#[test]
fn lru_evicts_oldest_unleased_entry_and_keeps_accounting_closed() {
    let cache = PrefixCacheV1::new(PrefixCacheConfigV1::new(2, 10, 25).unwrap());
    let id = identity(b"model-a");
    let a = [1_u32];
    let b = [2_u32];
    let c = [3_u32];
    publish(&cache, &id, &a, 1);
    publish(&cache, &id, &b, 2);
    // Touch A so B is the LRU victim.
    let a_lease = cache.lookup(&id, &a).unwrap().unwrap().into_lease();
    let c_key = PrefixCacheKeyV1::new(id.clone(), c).unwrap();
    cache.publish(c_key, value(c.len(), 10, 3)).unwrap();
    assert!(cache.lookup(&id, &b).unwrap().is_none());
    assert!(cache.lookup(&id, &a).unwrap().is_some());
    assert!(cache.lookup(&id, &c).unwrap().is_some());
    assert_eq!(cache.audit_snapshot().unwrap().total_logical_tokens(), 2);
    assert_eq!(cache.audit_snapshot().unwrap().total_resident_bytes(), 20);
    drop(a_lease);
}

#[test]
fn concurrent_readers_are_all_leases_and_prevent_replacement() {
    let cache = Arc::new(PrefixCacheV1::new(
        PrefixCacheConfigV1::new(1, 100, 100).unwrap(),
    ));
    let id = identity(b"model-a");
    let tokens = Arc::new([11_u32, 12, 13, 14]);
    publish(&cache, &id, tokens.as_ref(), 1);
    let barrier = Arc::new(Barrier::new(5));
    let mut workers = Vec::new();
    for _ in 0..4 {
        let cache = Arc::clone(&cache);
        let id = id.clone();
        let tokens = Arc::clone(&tokens);
        let barrier = Arc::clone(&barrier);
        workers.push(thread::spawn(move || {
            let result = cache.lookup(&id, tokens.as_ref()).unwrap().unwrap();
            barrier.wait();
            assert_eq!(result.lease().reader_count(), 4);
            barrier.wait();
            result
        }));
    }
    barrier.wait();
    let replacement = PrefixCacheKeyV1::new(id.clone(), tokens.as_ref()).unwrap();
    assert_eq!(
        cache.publish(replacement.clone(), value(tokens.len(), 10, 9)),
        Err(PrefixCacheError::EntryBusy)
    );
    barrier.wait();
    for worker in workers {
        worker.join().unwrap();
    }
    cache
        .publish(replacement, value(tokens.len(), 10, 9))
        .unwrap();
    assert_eq!(cache.audit_snapshot().unwrap().total_resident_bytes(), 10);
}

#[test]
fn checked_quota_and_length_mismatch_reject_without_partial_mutation() {
    let cache = PrefixCacheV1::new(PrefixCacheConfigV1::new(4, 3, 20).unwrap());
    let id = identity(b"model-a");
    let tokens = [1_u32, 2];
    let key = PrefixCacheKeyV1::new(id, tokens).unwrap();
    assert_eq!(
        cache.publish(key.clone(), value(1, 10, 1)),
        Err(PrefixCacheError::LogicalLengthMismatch {
            key_tokens: 2,
            value_tokens: 1,
        })
    );
    assert_eq!(cache.len().unwrap(), 0);
    cache.publish(key, value(2, 10, 1)).unwrap();
    let lease = cache
        .lookup(&identity(b"model-a"), &tokens)
        .unwrap()
        .unwrap();
    let too_large = PrefixCacheKeyV1::new(identity(b"model-b"), [3_u32, 4]).unwrap();
    assert_eq!(
        cache.publish(too_large, value(2, 11, 2)),
        Err(PrefixCacheError::QuotaExceeded)
    );
    assert_eq!(cache.len().unwrap(), 1);
    assert_eq!(cache.audit_snapshot().unwrap().total_logical_tokens(), 2);
    assert_eq!(cache.audit_snapshot().unwrap().total_resident_bytes(), 10);
    drop(lease);
}
