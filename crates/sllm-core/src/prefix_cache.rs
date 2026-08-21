//! Identity-safe, bounded prefix-cache indexing contracts.
//!
//! This module owns the host-side index and lease semantics for Phase 41.  It
//! deliberately does not know how a backend stores KV pages: a published entry
//! carries only accounting and an opaque state digest, while backend adapters
//! attach their own state fork to the entry id.  In particular, token ids are
//! retained for exact prefix confirmation instead of relying on a digest as a
//! lookup key.  The digest is only a redacted audit identifier.

use std::collections::HashMap;
use std::fmt;
use std::num::{NonZeroU64, NonZeroUsize};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use sha2::{Digest, Sha256};

use crate::kv_state::KvCacheEncoding;

/// Maximum number of published entries in the default bounded index.
pub const DEFAULT_PREFIX_CACHE_MAX_ENTRIES: usize = 256;
/// Maximum sum of logical tokens in the default bounded index.
pub const DEFAULT_PREFIX_CACHE_MAX_LOGICAL_TOKENS: u64 = 1_048_576;
/// A conservative default resident-byte quota.  Callers should normally set
/// this from their model/device memory budget.
pub const DEFAULT_PREFIX_CACHE_MAX_RESIDENT_BYTES: u64 = 1 << 30;
/// Hard cap for any identity component accepted from configuration or model metadata.
pub const MAX_PREFIX_IDENTITY_BYTES: usize = 4096;

/// A KV layout component of the prefix identity.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct PrefixKvLayoutV1 {
    heads: u32,
    head_dim: u32,
}

impl PrefixKvLayoutV1 {
    pub fn new(heads: u32, head_dim: u32) -> Result<Self, PrefixCacheError> {
        if heads == 0 || head_dim == 0 {
            return Err(PrefixCacheError::InvalidKvLayout);
        }
        Ok(Self { heads, head_dim })
    }

    pub const fn heads(self) -> u32 {
        self.heads
    }

    pub const fn head_dim(self) -> u32 {
        self.head_dim
    }
}

/// Identity dimensions that must match before a prefix can be reused.
///
/// The byte strings are retained (rather than only their digest) so that an
/// accidental digest collision still results in an exact inequality check.
/// Callers should pass model-lock/recipe/template digests or other stable
/// canonical identity bytes, not filesystem paths or user-visible aliases.
#[derive(Clone, Eq, Hash, PartialEq)]
pub struct PrefixStateIdentityV1 {
    model_lock_fingerprint: Arc<[u8]>,
    derived_artifact_identity: Arc<[u8]>,
    adapter_identity: Arc<[u8]>,
    renderer_template_digest: Arc<[u8]>,
    tokenizer_identity: Arc<[u8]>,
    kv_cache_encoding: KvCacheEncoding,
    kv_layout: PrefixKvLayoutV1,
    target_semantics: Arc<[u8]>,
    context_policy_version: u32,
}

impl fmt::Debug for PrefixStateIdentityV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PrefixStateIdentityV1")
            .field("identity", &"<redacted>")
            .field("kv_cache_encoding", &self.kv_cache_encoding)
            .field("kv_layout", &self.kv_layout)
            .field("context_policy_version", &self.context_policy_version)
            .finish()
    }
}

impl PrefixStateIdentityV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model_lock_fingerprint: impl AsRef<[u8]>,
        derived_artifact_identity: impl AsRef<[u8]>,
        adapter_identity: impl AsRef<[u8]>,
        renderer_template_digest: impl AsRef<[u8]>,
        tokenizer_identity: impl AsRef<[u8]>,
        kv_cache_encoding: KvCacheEncoding,
        kv_layout: PrefixKvLayoutV1,
        target_semantics: impl AsRef<[u8]>,
        context_policy_version: u32,
    ) -> Result<Self, PrefixCacheError> {
        let model_lock_fingerprint = copy_identity(model_lock_fingerprint.as_ref())?;
        let derived_artifact_identity = copy_identity(derived_artifact_identity.as_ref())?;
        let adapter_identity = copy_identity(adapter_identity.as_ref())?;
        let renderer_template_digest = copy_identity(renderer_template_digest.as_ref())?;
        let tokenizer_identity = copy_identity(tokenizer_identity.as_ref())?;
        let target_semantics = copy_identity(target_semantics.as_ref())?;
        Ok(Self {
            model_lock_fingerprint,
            derived_artifact_identity,
            adapter_identity,
            renderer_template_digest,
            tokenizer_identity,
            kv_cache_encoding,
            kv_layout,
            target_semantics,
            context_policy_version,
        })
    }

    pub fn model_lock_fingerprint(&self) -> &[u8] {
        &self.model_lock_fingerprint
    }

    pub fn adapter_identity(&self) -> &[u8] {
        &self.adapter_identity
    }

    pub fn derived_artifact_identity(&self) -> &[u8] {
        &self.derived_artifact_identity
    }

    pub fn renderer_template_digest(&self) -> &[u8] {
        &self.renderer_template_digest
    }

    pub fn tokenizer_identity(&self) -> &[u8] {
        &self.tokenizer_identity
    }

    pub const fn kv_cache_encoding(&self) -> KvCacheEncoding {
        self.kv_cache_encoding
    }

    pub const fn kv_layout(&self) -> PrefixKvLayoutV1 {
        self.kv_layout
    }

    pub fn target_semantics(&self) -> &[u8] {
        &self.target_semantics
    }

    pub const fn context_policy_version(&self) -> u32 {
        self.context_policy_version
    }
}

fn copy_identity(bytes: &[u8]) -> Result<Arc<[u8]>, PrefixCacheError> {
    if bytes.is_empty() {
        return Err(PrefixCacheError::EmptyIdentity);
    }
    if bytes.len() > MAX_PREFIX_IDENTITY_BYTES {
        return Err(PrefixCacheError::IdentityTooLarge);
    }
    Ok(Arc::from(bytes.to_vec().into_boxed_slice()))
}

/// An exact key.  The token sequence is compared directly on lookup; its
/// SHA-256 is never used as the sole equality predicate.
#[derive(Clone, Eq, Hash, PartialEq)]
pub struct PrefixCacheKeyV1 {
    identity: PrefixStateIdentityV1,
    tokens: Arc<[u32]>,
}

impl fmt::Debug for PrefixCacheKeyV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PrefixCacheKeyV1")
            .field("identity", &self.identity)
            .field("token_count", &self.tokens.len())
            .field("key_digest", &HexDigest(self.redacted_digest()))
            .finish()
    }
}

struct HexDigest([u8; 32]);

impl fmt::Debug for HexDigest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl PrefixCacheKeyV1 {
    pub fn new(
        identity: PrefixStateIdentityV1,
        tokens: impl AsRef<[u32]>,
    ) -> Result<Self, PrefixCacheError> {
        let tokens = tokens.as_ref();
        if tokens.is_empty() {
            return Err(PrefixCacheError::EmptyTokenSequence);
        }
        Ok(Self {
            identity,
            tokens: Arc::from(tokens.to_vec().into_boxed_slice()),
        })
    }

    pub fn identity(&self) -> &PrefixStateIdentityV1 {
        &self.identity
    }

    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Redacted key id for audit/metrics.  It intentionally contains no raw
    /// token ids or identity strings.
    pub fn redacted_digest(&self) -> [u8; 32] {
        digest_key(&self.identity, &self.tokens)
    }
}

/// Accounting metadata for one immutable published prefix.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrefixCacheValueV1 {
    logical_tokens: NonZeroU64,
    resident_bytes: u64,
    state_digest: [u8; 32],
}

impl PrefixCacheValueV1 {
    pub fn new(
        logical_tokens: u64,
        resident_bytes: u64,
        state_digest: [u8; 32],
    ) -> Result<Self, PrefixCacheError> {
        Ok(Self {
            logical_tokens: NonZeroU64::new(logical_tokens)
                .ok_or(PrefixCacheError::ZeroLogicalTokens)?,
            resident_bytes,
            state_digest,
        })
    }

    pub const fn logical_tokens(self) -> u64 {
        self.logical_tokens.get()
    }

    pub const fn resident_bytes(self) -> u64 {
        self.resident_bytes
    }

    pub const fn state_digest(self) -> [u8; 32] {
        self.state_digest
    }
}

/// Bounded cache limits.  All additions/subtractions use checked arithmetic;
/// an invalid limit is rejected at construction instead of wrapping.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrefixCacheConfigV1 {
    max_entries: NonZeroUsize,
    max_logical_tokens: NonZeroU64,
    max_resident_bytes: NonZeroU64,
}

impl PrefixCacheConfigV1 {
    pub fn new(
        max_entries: usize,
        max_logical_tokens: u64,
        max_resident_bytes: u64,
    ) -> Result<Self, PrefixCacheError> {
        if max_entries > DEFAULT_PREFIX_CACHE_MAX_ENTRIES
            || max_logical_tokens > DEFAULT_PREFIX_CACHE_MAX_LOGICAL_TOKENS
        {
            return Err(PrefixCacheError::InvalidQuota);
        }
        Ok(Self {
            max_entries: NonZeroUsize::new(max_entries).ok_or(PrefixCacheError::InvalidQuota)?,
            max_logical_tokens: NonZeroU64::new(max_logical_tokens)
                .ok_or(PrefixCacheError::InvalidQuota)?,
            max_resident_bytes: NonZeroU64::new(max_resident_bytes)
                .ok_or(PrefixCacheError::InvalidQuota)?,
        })
    }

    pub const fn max_entries(self) -> usize {
        self.max_entries.get()
    }

    pub const fn max_logical_tokens(self) -> u64 {
        self.max_logical_tokens.get()
    }

    pub const fn max_resident_bytes(self) -> u64 {
        self.max_resident_bytes.get()
    }
}

impl Default for PrefixCacheConfigV1 {
    fn default() -> Self {
        Self::new(
            DEFAULT_PREFIX_CACHE_MAX_ENTRIES,
            DEFAULT_PREFIX_CACHE_MAX_LOGICAL_TOKENS,
            DEFAULT_PREFIX_CACHE_MAX_RESIDENT_BYTES,
        )
        .expect("default prefix cache quotas are non-zero")
    }
}

/// Hit type recorded in the redacted audit stream.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrefixLookupKind {
    ExactHit,
    PartialHit,
    Miss,
}

/// Redacted bounded audit counters.  No identity bytes or token ids are
/// exposed; the most recent key is represented only by SHA-256.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrefixCacheAuditSnapshot {
    lookups: u64,
    exact_hits: u64,
    partial_hits: u64,
    misses: u64,
    evictions: u64,
    lease_eviction_blocks: u64,
    total_logical_tokens: u64,
    total_resident_bytes: u64,
    last_kind: PrefixLookupKind,
    last_key_digest: Option<[u8; 32]>,
}

impl Default for PrefixCacheAuditSnapshot {
    fn default() -> Self {
        Self {
            lookups: 0,
            exact_hits: 0,
            partial_hits: 0,
            misses: 0,
            evictions: 0,
            lease_eviction_blocks: 0,
            total_logical_tokens: 0,
            total_resident_bytes: 0,
            last_kind: PrefixLookupKind::Miss,
            last_key_digest: None,
        }
    }
}

impl PrefixCacheAuditSnapshot {
    pub const fn lookups(self) -> u64 {
        self.lookups
    }

    pub const fn exact_hits(self) -> u64 {
        self.exact_hits
    }

    pub const fn partial_hits(self) -> u64 {
        self.partial_hits
    }

    pub const fn misses(self) -> u64 {
        self.misses
    }

    pub const fn evictions(self) -> u64 {
        self.evictions
    }

    pub const fn lease_eviction_blocks(self) -> u64 {
        self.lease_eviction_blocks
    }

    pub const fn total_logical_tokens(self) -> u64 {
        self.total_logical_tokens
    }

    pub const fn total_resident_bytes(self) -> u64 {
        self.total_resident_bytes
    }

    pub const fn last_kind(self) -> PrefixLookupKind {
        self.last_kind
    }

    pub const fn last_key_digest(self) -> Option<[u8; 32]> {
        self.last_key_digest
    }
}

#[derive(Debug)]
struct PrefixCacheEntry {
    id: u64,
    key: PrefixCacheKeyV1,
    value: PrefixCacheValueV1,
    readers: AtomicUsize,
    last_used: AtomicU64,
}

#[derive(Debug)]
struct PrefixCacheState {
    next_id: u64,
    clock: u64,
    entries: HashMap<u64, Arc<PrefixCacheEntry>>,
    total_logical_tokens: u64,
    total_resident_bytes: u64,
    audit: PrefixCacheAuditSnapshot,
}

#[derive(Debug)]
struct PrefixCacheShared {
    state: Mutex<PrefixCacheState>,
    config: PrefixCacheConfigV1,
}

/// Thread-safe bounded longest-prefix index.
#[derive(Clone, Debug)]
pub struct PrefixCacheV1 {
    shared: Arc<PrefixCacheShared>,
}

/// Backend-owned opaque state associated with the host index.
///
/// The index deliberately exposes only an entry id and a lease. Implementors
/// keep device handles and page tables behind this boundary, create a fresh
/// mutable request owner from a leased immutable state, and drop state after
/// the corresponding index entry is removed. Neither frontend nor server
/// code needs to inspect a native pointer to reuse a prefix.
pub trait PrefixCacheBackendV1 {
    type PublishedState;
    type RequestState;
    type Error;

    /// Attach an immutable backend owner after the index has admitted `id`.
    /// Replacing an existing owner returns it to the caller for explicit drop.
    fn publish_state(
        &mut self,
        id: PrefixEntryIdV1,
        state: Self::PublishedState,
    ) -> Result<Option<Self::PublishedState>, Self::Error>;

    /// Fork a fresh mutable request owner while `lease` prevents eviction.
    fn fork_state(&self, lease: &PrefixLeaseV1) -> Result<Self::RequestState, Self::Error>;

    /// Detach an owner whose host-index entry is no longer published.
    fn remove_state(&mut self, id: PrefixEntryIdV1) -> Option<Self::PublishedState>;
}

impl PrefixCacheV1 {
    pub fn new(config: PrefixCacheConfigV1) -> Self {
        Self {
            shared: Arc::new(PrefixCacheShared {
                state: Mutex::new(PrefixCacheState {
                    next_id: 1,
                    clock: 0,
                    entries: HashMap::new(),
                    total_logical_tokens: 0,
                    total_resident_bytes: 0,
                    audit: PrefixCacheAuditSnapshot::default(),
                }),
                config,
            }),
        }
    }

    pub fn config(&self) -> PrefixCacheConfigV1 {
        self.shared.config
    }

    /// Publish an immutable prefix.  If admission needs space, only entries
    /// with no active lease are evicted, oldest first.  The mutation is
    /// atomic: if enough unleased entries do not exist, nothing is removed.
    pub fn publish(
        &self,
        key: PrefixCacheKeyV1,
        value: PrefixCacheValueV1,
    ) -> Result<PrefixEntryIdV1, PrefixCacheError> {
        let key_tokens = u64::try_from(key.len()).map_err(|_| PrefixCacheError::QuotaOverflow)?;
        if key_tokens != value.logical_tokens() {
            return Err(PrefixCacheError::LogicalLengthMismatch {
                key_tokens,
                value_tokens: value.logical_tokens(),
            });
        }
        let mut state = self
            .shared
            .state
            .lock()
            .map_err(|_| PrefixCacheError::Poisoned)?;

        let replacement = state
            .entries
            .values()
            .find(|entry| entry.key == key)
            .cloned();
        if let Some(entry) = replacement.as_ref() {
            if entry.readers.load(Ordering::Acquire) != 0 {
                return Err(PrefixCacheError::EntryBusy);
            }
        }

        let old_logical = replacement
            .as_ref()
            .map_or(0, |entry| entry.value.logical_tokens());
        let old_resident = replacement
            .as_ref()
            .map_or(0, |entry| entry.value.resident_bytes());
        let old_count = if replacement.is_some() { 1 } else { 0 };
        let target_logical = state
            .total_logical_tokens
            .checked_sub(old_logical)
            .and_then(|total| total.checked_add(value.logical_tokens()))
            .ok_or(PrefixCacheError::QuotaOverflow)?;
        let target_resident = state
            .total_resident_bytes
            .checked_sub(old_resident)
            .and_then(|total| total.checked_add(value.resident_bytes()))
            .ok_or(PrefixCacheError::QuotaOverflow)?;
        let target_count = state
            .entries
            .len()
            .checked_sub(old_count)
            .and_then(|value| value.checked_add(1))
            .ok_or(PrefixCacheError::QuotaOverflow)?;

        let mut victims = Vec::new();
        let mut projected_count = target_count;
        let mut projected_logical = target_logical;
        let mut projected_resident = target_resident;
        if projected_count > self.shared.config.max_entries()
            || projected_logical > self.shared.config.max_logical_tokens()
            || projected_resident > self.shared.config.max_resident_bytes()
        {
            let mut candidates: Vec<_> = state
                .entries
                .values()
                .filter(|entry| replacement.as_ref().is_none_or(|old| old.id != entry.id))
                .filter(|entry| entry.readers.load(Ordering::Acquire) == 0)
                .cloned()
                .collect();
            candidates.sort_by_key(|entry| entry.last_used.load(Ordering::Relaxed));
            for entry in candidates {
                if projected_count <= self.shared.config.max_entries()
                    && projected_logical <= self.shared.config.max_logical_tokens()
                    && projected_resident <= self.shared.config.max_resident_bytes()
                {
                    break;
                }
                projected_count = projected_count
                    .checked_sub(1)
                    .ok_or(PrefixCacheError::QuotaOverflow)?;
                projected_logical = projected_logical
                    .checked_sub(entry.value.logical_tokens())
                    .ok_or(PrefixCacheError::QuotaOverflow)?;
                projected_resident = projected_resident
                    .checked_sub(entry.value.resident_bytes())
                    .ok_or(PrefixCacheError::QuotaOverflow)?;
                victims.push(entry.id);
            }
            if projected_count > self.shared.config.max_entries()
                || projected_logical > self.shared.config.max_logical_tokens()
                || projected_resident > self.shared.config.max_resident_bytes()
            {
                if state
                    .entries
                    .values()
                    .any(|entry| entry.readers.load(Ordering::Acquire) != 0)
                {
                    state.audit.lease_eviction_blocks = state
                        .audit
                        .lease_eviction_blocks
                        .checked_add(1)
                        .ok_or(PrefixCacheError::AuditOverflow)?;
                }
                return Err(PrefixCacheError::QuotaExceeded);
            }
        }

        // Preflight every fallible counter before mutating the index. This
        // preserves the advertised all-or-nothing admission semantics even
        // when an audit/id/LRU counter is exhausted.
        let id = PrefixEntryIdV1(state.next_id);
        let next_id = state
            .next_id
            .checked_add(1)
            .ok_or(PrefixCacheError::IdOverflow)?;
        let clock = state
            .clock
            .checked_add(1)
            .ok_or(PrefixCacheError::ClockOverflow)?;
        let eviction_count =
            u64::try_from(victims.len()).map_err(|_| PrefixCacheError::AuditOverflow)?;
        let evictions = state
            .audit
            .evictions
            .checked_add(eviction_count)
            .ok_or(PrefixCacheError::AuditOverflow)?;
        let entry = Arc::new(PrefixCacheEntry {
            id: id.0,
            key,
            value,
            readers: AtomicUsize::new(0),
            last_used: AtomicU64::new(clock),
        });

        for victim in &victims {
            state.entries.remove(victim);
        }
        if let Some(old) = replacement {
            state.entries.remove(&old.id);
        }
        state.next_id = next_id;
        state.clock = clock;
        state.total_logical_tokens = projected_logical;
        state.total_resident_bytes = projected_resident;
        state.audit.evictions = evictions;
        state.entries.insert(id.0, entry);
        state.audit.total_logical_tokens = state.total_logical_tokens;
        state.audit.total_resident_bytes = state.total_resident_bytes;
        Ok(id)
    }

    /// Find the longest exact token prefix for this identity.  The bounded
    /// entry count makes a linear scan predictable and avoids exposing a trie
    /// implementation to backend owners; every candidate still performs an
    /// exact token-by-token comparison.
    pub fn lookup(
        &self,
        identity: &PrefixStateIdentityV1,
        requested_tokens: &[u32],
    ) -> Result<Option<PrefixLookupResultV1>, PrefixCacheError> {
        let mut state = self
            .shared
            .state
            .lock()
            .map_err(|_| PrefixCacheError::Poisoned)?;
        let requested_len =
            u64::try_from(requested_tokens.len()).map_err(|_| PrefixCacheError::QuotaOverflow)?;
        if requested_len > self.shared.config.max_logical_tokens() {
            return Err(PrefixCacheError::TokenLimitExceeded {
                tokens: requested_len,
            });
        }
        let lookups = state
            .audit
            .lookups
            .checked_add(1)
            .ok_or(PrefixCacheError::AuditOverflow)?;
        let query_digest = digest_key(identity, requested_tokens);
        let best = state
            .entries
            .values()
            .filter(|entry| entry.key.identity() == identity)
            .filter(|entry| entry.key.len() <= requested_tokens.len())
            .filter(|entry| requested_tokens.starts_with(entry.key.tokens()))
            .max_by_key(|entry| entry.key.len())
            .cloned();
        let Some(entry) = best else {
            let misses = state
                .audit
                .misses
                .checked_add(1)
                .ok_or(PrefixCacheError::AuditOverflow)?;
            state.audit.lookups = lookups;
            state.audit.misses = misses;
            state.audit.last_kind = PrefixLookupKind::Miss;
            state.audit.last_key_digest = Some(query_digest);
            return Ok(None);
        };
        let clock = state
            .clock
            .checked_add(1)
            .ok_or(PrefixCacheError::ClockOverflow)?;
        let matched_len = entry.key.len();
        let (kind, exact_hits, partial_hits) = if matched_len == requested_tokens.len() {
            let exact_hits = state
                .audit
                .exact_hits
                .checked_add(1)
                .ok_or(PrefixCacheError::AuditOverflow)?;
            (
                PrefixLookupKind::ExactHit,
                exact_hits,
                state.audit.partial_hits,
            )
        } else {
            let partial_hits = state
                .audit
                .partial_hits
                .checked_add(1)
                .ok_or(PrefixCacheError::AuditOverflow)?;
            (
                PrefixLookupKind::PartialHit,
                state.audit.exact_hits,
                partial_hits,
            )
        };
        entry
            .readers
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                current.checked_add(1)
            })
            .map_err(|_| PrefixCacheError::ReaderOverflow)?;
        state.audit.lookups = lookups;
        state.audit.exact_hits = exact_hits;
        state.audit.partial_hits = partial_hits;
        state.clock = clock;
        entry.last_used.store(clock, Ordering::Release);
        state.audit.last_kind = kind;
        state.audit.last_key_digest = Some(query_digest);
        Ok(Some(PrefixLookupResultV1 {
            lease: PrefixLeaseV1 {
                shared: Arc::clone(&self.shared),
                entry,
            },
            requested_len: requested_tokens.len(),
            matched_len,
            kind,
        }))
    }

    pub fn audit_snapshot(&self) -> Result<PrefixCacheAuditSnapshot, PrefixCacheError> {
        let state = self
            .shared
            .state
            .lock()
            .map_err(|_| PrefixCacheError::Poisoned)?;
        let mut audit = state.audit;
        audit.total_logical_tokens = state.total_logical_tokens;
        audit.total_resident_bytes = state.total_resident_bytes;
        Ok(audit)
    }

    pub fn len(&self) -> Result<usize, PrefixCacheError> {
        let state = self
            .shared
            .state
            .lock()
            .map_err(|_| PrefixCacheError::Poisoned)?;
        Ok(state.entries.len())
    }

    pub fn is_empty(&self) -> Result<bool, PrefixCacheError> {
        Ok(self.len()? == 0)
    }

    /// Returns whether an entry id is still published. Backend-owned opaque
    /// state maps use this bounded reconciliation hook after publication so
    /// evicted/replaced owners are dropped without exposing cache internals.
    pub fn contains_entry(&self, id: PrefixEntryIdV1) -> Result<bool, PrefixCacheError> {
        let state = self
            .shared
            .state
            .lock()
            .map_err(|_| PrefixCacheError::Poisoned)?;
        Ok(state.entries.contains_key(&id.0))
    }

    /// Bounded, identity-free snapshot for reconciling separately-owned
    /// backend state after LRU eviction or replacement.
    pub fn published_entry_ids(&self) -> Result<Vec<PrefixEntryIdV1>, PrefixCacheError> {
        let state = self
            .shared
            .state
            .lock()
            .map_err(|_| PrefixCacheError::Poisoned)?;
        Ok(state.entries.keys().copied().map(PrefixEntryIdV1).collect())
    }
}

/// Stable id for a published prefix.  Backend state maps may use this id to
/// attach an opaque page set without exposing a device pointer to the cache.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct PrefixEntryIdV1(u64);

impl PrefixEntryIdV1 {
    pub const fn raw(self) -> u64 {
        self.0
    }
}

/// A leased immutable prefix.  A lease is the eviction barrier: while one or
/// more leases exist, the entry cannot be evicted or replaced.
#[derive(Debug)]
pub struct PrefixLeaseV1 {
    shared: Arc<PrefixCacheShared>,
    entry: Arc<PrefixCacheEntry>,
}

impl Drop for PrefixLeaseV1 {
    fn drop(&mut self) {
        let result =
            self.entry
                .readers
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                    current.checked_sub(1)
                });
        debug_assert!(result.is_ok(), "prefix lease reader count underflow");
    }
}

impl PrefixLeaseV1 {
    pub fn entry_id(&self) -> PrefixEntryIdV1 {
        PrefixEntryIdV1(self.entry.id)
    }

    pub fn key(&self) -> &PrefixCacheKeyV1 {
        &self.entry.key
    }

    pub fn value(&self) -> PrefixCacheValueV1 {
        self.entry.value
    }

    pub fn tokens(&self) -> &[u32] {
        self.entry.key.tokens()
    }

    pub fn redacted_digest(&self) -> [u8; 32] {
        self.entry.key.redacted_digest()
    }

    pub fn reader_count(&self) -> usize {
        self.entry.readers.load(Ordering::Acquire)
    }

    pub fn strong_cache_reference(&self) -> bool {
        Arc::strong_count(&self.shared) > 1
    }
}

/// Lookup result includes the requested/matched lengths so suffix execution
/// can begin at the returned lease without exposing the cache's internals.
#[derive(Debug)]
pub struct PrefixLookupResultV1 {
    lease: PrefixLeaseV1,
    requested_len: usize,
    matched_len: usize,
    kind: PrefixLookupKind,
}

impl PrefixLookupResultV1 {
    pub fn lease(&self) -> &PrefixLeaseV1 {
        &self.lease
    }

    pub fn into_lease(self) -> PrefixLeaseV1 {
        self.lease
    }

    pub const fn requested_len(&self) -> usize {
        self.requested_len
    }

    pub const fn matched_len(&self) -> usize {
        self.matched_len
    }

    pub const fn remaining_len(&self) -> usize {
        self.requested_len - self.matched_len
    }

    pub const fn kind(&self) -> PrefixLookupKind {
        self.kind
    }

    pub fn suffix<'a>(&self, requested_tokens: &'a [u32]) -> Option<&'a [u32]> {
        if requested_tokens.len() != self.requested_len
            || !requested_tokens.starts_with(self.lease.tokens())
        {
            return None;
        }
        Some(&requested_tokens[self.matched_len..])
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PrefixCacheError {
    EmptyIdentity,
    IdentityTooLarge,
    EmptyTokenSequence,
    InvalidKvLayout,
    InvalidQuota,
    ZeroLogicalTokens,
    LogicalLengthMismatch { key_tokens: u64, value_tokens: u64 },
    TokenLimitExceeded { tokens: u64 },
    QuotaExceeded,
    QuotaOverflow,
    AuditOverflow,
    ReaderOverflow,
    IdOverflow,
    ClockOverflow,
    EntryBusy,
    Poisoned,
}

impl fmt::Display for PrefixCacheError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyIdentity => formatter.write_str("prefix identity fields must be non-empty"),
            Self::IdentityTooLarge => write!(
                formatter,
                "prefix identity fields must not exceed {MAX_PREFIX_IDENTITY_BYTES} bytes"
            ),
            Self::EmptyTokenSequence => {
                formatter.write_str("prefix token sequence must be non-empty")
            }
            Self::InvalidKvLayout => {
                formatter.write_str("prefix KV layout dimensions must be non-zero")
            }
            Self::InvalidQuota => formatter.write_str("prefix cache quotas must be non-zero"),
            Self::ZeroLogicalTokens => {
                formatter.write_str("prefix value logical token count must be non-zero")
            }
            Self::LogicalLengthMismatch {
                key_tokens,
                value_tokens,
            } => write!(
                formatter,
                "prefix key has {key_tokens} tokens but value accounts for {value_tokens}"
            ),
            Self::TokenLimitExceeded { tokens } => write!(
                formatter,
                "prefix request has {tokens} tokens, over configured limit"
            ),
            Self::QuotaExceeded => formatter
                .write_str("prefix cache quota cannot admit entry without evicting a live lease"),
            Self::QuotaOverflow => formatter.write_str("prefix cache accounting overflowed"),
            Self::AuditOverflow => formatter.write_str("prefix cache audit counter overflowed"),
            Self::ReaderOverflow => formatter.write_str("prefix cache reader count overflowed"),
            Self::IdOverflow => formatter.write_str("prefix cache entry id overflowed"),
            Self::ClockOverflow => formatter.write_str("prefix cache LRU clock overflowed"),
            Self::EntryBusy => formatter.write_str("prefix cache entry has active readers"),
            Self::Poisoned => formatter.write_str("prefix cache lock was poisoned"),
        }
    }
}

impl std::error::Error for PrefixCacheError {}

fn digest_key(identity: &PrefixStateIdentityV1, tokens: &[u32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"sllm-prefix-key-v1\0");
    digest_bytes(&mut hasher, identity.model_lock_fingerprint());
    digest_bytes(&mut hasher, identity.derived_artifact_identity());
    digest_bytes(&mut hasher, identity.adapter_identity());
    digest_bytes(&mut hasher, identity.renderer_template_digest());
    digest_bytes(&mut hasher, identity.tokenizer_identity());
    hasher.update([encoding_tag(identity.kv_cache_encoding())]);
    hasher.update(identity.kv_layout().heads().to_le_bytes());
    hasher.update(identity.kv_layout().head_dim().to_le_bytes());
    digest_bytes(&mut hasher, identity.target_semantics());
    hasher.update(identity.context_policy_version().to_le_bytes());
    hasher.update((tokens.len() as u64).to_le_bytes());
    for token in tokens {
        hasher.update(token.to_le_bytes());
    }
    hasher.finalize().into()
}

fn digest_bytes(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

const fn encoding_tag(encoding: KvCacheEncoding) -> u8 {
    match encoding {
        KvCacheEncoding::Fp16 => 0,
        KvCacheEncoding::Fp8E4M3Fn => 1,
        KvCacheEncoding::Fp8E4M3FnStatic => 2,
        KvCacheEncoding::Nvfp4 => 3,
    }
}

#[cfg(test)]
mod atomicity_tests {
    use super::*;

    fn identity() -> PrefixStateIdentityV1 {
        PrefixStateIdentityV1::new(
            b"model",
            b"artifact",
            b"adapter",
            b"renderer",
            b"tokenizer",
            KvCacheEncoding::Fp16,
            PrefixKvLayoutV1::new(2, 64).unwrap(),
            b"gfx1030",
            1,
        )
        .unwrap()
    }

    fn key(tokens: &[u32]) -> PrefixCacheKeyV1 {
        PrefixCacheKeyV1::new(identity(), tokens).unwrap()
    }

    fn value(tokens: u64) -> PrefixCacheValueV1 {
        PrefixCacheValueV1::new(tokens, 16, [tokens as u8; 32]).unwrap()
    }

    #[test]
    fn publish_counter_failure_does_not_evict_existing_entry() {
        let cache = PrefixCacheV1::new(PrefixCacheConfigV1::new(1, 8, 32).unwrap());
        cache.publish(key(&[1]), value(1)).unwrap();
        {
            let mut state = cache.shared.state.lock().unwrap();
            state.next_id = u64::MAX;
        }
        assert_eq!(
            cache.publish(key(&[2]), value(1)),
            Err(PrefixCacheError::IdOverflow)
        );
        let state = cache.shared.state.lock().unwrap();
        assert_eq!(state.entries.len(), 1);
        assert_eq!(state.entries.values().next().unwrap().key.tokens(), [1]);
        assert_eq!(state.audit.evictions, 0);
    }

    #[test]
    fn lookup_counter_failure_does_not_leak_a_reader_or_audit_mutation() {
        let cache = PrefixCacheV1::new(PrefixCacheConfigV1::new(1, 8, 32).unwrap());
        cache.publish(key(&[1]), value(1)).unwrap();
        {
            let mut state = cache.shared.state.lock().unwrap();
            state.clock = u64::MAX;
        }
        assert!(matches!(
            cache.lookup(&identity(), &[1]),
            Err(PrefixCacheError::ClockOverflow)
        ));
        let state = cache.shared.state.lock().unwrap();
        let entry = state.entries.values().next().unwrap();
        assert_eq!(entry.readers.load(Ordering::Acquire), 0);
        assert_eq!(state.audit.lookups, 0);
        assert_eq!(state.audit.exact_hits, 0);
    }

    #[test]
    fn reader_overflow_does_not_wrap_or_publish_a_lookup() {
        let cache = PrefixCacheV1::new(PrefixCacheConfigV1::new(1, 8, 32).unwrap());
        cache.publish(key(&[1]), value(1)).unwrap();
        {
            let state = cache.shared.state.lock().unwrap();
            state
                .entries
                .values()
                .next()
                .unwrap()
                .readers
                .store(usize::MAX, Ordering::Release);
        }
        assert!(matches!(
            cache.lookup(&identity(), &[1]),
            Err(PrefixCacheError::ReaderOverflow)
        ));
        let state = cache.shared.state.lock().unwrap();
        let entry = state.entries.values().next().unwrap();
        assert_eq!(entry.readers.load(Ordering::Acquire), usize::MAX);
        assert_eq!(state.audit.lookups, 0);
        assert_eq!(state.audit.exact_hits, 0);
    }
}
