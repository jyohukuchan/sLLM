//! Bounded, model-neutral lifecycle ownership for dynamically loaded models.
//!
//! The registry in this module deliberately owns no model format or transport
//! details.  A caller supplies an offline loader and receives an RAII lease for
//! the existing [`ModelRegistryEntryV1`].

use crate::runtime::{BackendObservabilitySnapshotV1, ModelRegistryEntryV1};
use serde::Serialize;
use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, Condvar, Mutex, Weak};
use std::time::{Duration, Instant};

pub const MAX_CONFIGURED_ALIASES_V1: usize = 64;
pub const MAX_LOADED_MODELS_V1: usize = 16;
pub const MAX_IDENTITY_BYTES_V1: usize = 1024;

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct ModelLifecycleIdentityV1 {
    model_identity: String,
    plan_identity: String,
    adapter_identity: String,
}

impl ModelLifecycleIdentityV1 {
    pub fn new(
        model_identity: impl Into<String>,
        plan_identity: impl Into<String>,
        adapter_identity: impl Into<String>,
    ) -> Result<Self, ModelLifecycleErrorV1> {
        let value = Self {
            model_identity: model_identity.into(),
            plan_identity: plan_identity.into(),
            adapter_identity: adapter_identity.into(),
        };
        value.validate()?;
        Ok(value)
    }

    pub fn model_identity(&self) -> &str {
        &self.model_identity
    }
    pub fn plan_identity(&self) -> &str {
        &self.plan_identity
    }
    pub fn plan_digest(&self) -> &str {
        &self.plan_identity
    }
    pub fn adapter_identity(&self) -> &str {
        &self.adapter_identity
    }

    fn validate(&self) -> Result<(), ModelLifecycleErrorV1> {
        for value in [
            &self.model_identity,
            &self.plan_identity,
            &self.adapter_identity,
        ] {
            if value.is_empty()
                || value.len() > MAX_IDENTITY_BYTES_V1
                || value.contains('\0')
                || is_network_form(value)
            {
                return Err(ModelLifecycleErrorV1::InvalidDescriptor);
            }
        }
        if !is_sha256_digest(&self.plan_identity) {
            return Err(ModelLifecycleErrorV1::InvalidDescriptor);
        }
        Ok(())
    }
}

fn is_sha256_digest(value: &str) -> bool {
    value.len() == 71
        && value.starts_with("sha256:")
        && value[7..]
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelLifecycleDescriptorV1 {
    alias: String,
    identity: ModelLifecycleIdentityV1,
    declared_resident_bytes: u64,
}

impl ModelLifecycleDescriptorV1 {
    pub fn new(
        alias: impl Into<String>,
        model_identity: impl Into<String>,
        plan_identity: impl Into<String>,
        adapter_identity: impl Into<String>,
        declared_resident_bytes: u64,
    ) -> Result<Self, ModelLifecycleErrorV1> {
        let value = Self {
            alias: alias.into(),
            identity: ModelLifecycleIdentityV1::new(
                model_identity,
                plan_identity,
                adapter_identity,
            )?,
            declared_resident_bytes,
        };
        value.validate()?;
        Ok(value)
    }

    pub fn alias(&self) -> &str {
        &self.alias
    }
    pub fn identity(&self) -> &ModelLifecycleIdentityV1 {
        &self.identity
    }
    pub const fn declared_resident_bytes(&self) -> u64 {
        self.declared_resident_bytes
    }

    fn validate(&self) -> Result<(), ModelLifecycleErrorV1> {
        if self.alias.is_empty()
            || self.alias.len() > 128
            || self.alias.contains('\0')
            || is_network_form(&self.alias)
            || !self
                .alias
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
            || self.declared_resident_bytes == 0
        {
            return Err(ModelLifecycleErrorV1::InvalidDescriptor);
        }
        self.identity.validate()
    }
}

fn is_network_form(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    if lower.contains("://") || lower.starts_with("//") {
        return true;
    }
    lower.split_once(':').is_some_and(|(scheme, _)| {
        matches!(
            scheme,
            "http" | "https" | "ftp" | "file" | "tcp" | "udp" | "ws" | "wss" | "s3"
        )
    })
}

#[derive(Clone)]
pub struct ModelLifecycleLoadedV1 {
    owner: Arc<ModelRegistryEntryV1>,
    identity: ModelLifecycleIdentityV1,
    resident_bytes: u64,
}

impl ModelLifecycleLoadedV1 {
    pub fn new(
        owner: Arc<ModelRegistryEntryV1>,
        resident_bytes: u64,
        model_identity: impl Into<String>,
        plan_identity: impl Into<String>,
        adapter_identity: impl Into<String>,
    ) -> Result<Self, ModelLifecycleErrorV1> {
        if resident_bytes == 0 {
            return Err(ModelLifecycleErrorV1::InvalidDescriptor);
        }
        let model_identity = model_identity.into();
        if owner.lock_fingerprint() != model_identity {
            return Err(ModelLifecycleErrorV1::IdentityMismatch);
        }
        Ok(Self {
            owner,
            identity: ModelLifecycleIdentityV1::new(
                model_identity,
                plan_identity,
                adapter_identity,
            )?,
            resident_bytes,
        })
    }

    pub fn from_entry(
        owner: Arc<ModelRegistryEntryV1>,
        resident_bytes: u64,
        plan_identity: impl Into<String>,
        adapter_identity: impl Into<String>,
    ) -> Result<Self, ModelLifecycleErrorV1> {
        Self::new(
            owner.clone(),
            resident_bytes,
            owner.lock_fingerprint().to_owned(),
            plan_identity,
            adapter_identity,
        )
    }

    pub fn owner(&self) -> Arc<ModelRegistryEntryV1> {
        Arc::clone(&self.owner)
    }
    pub fn identity(&self) -> &ModelLifecycleIdentityV1 {
        &self.identity
    }
    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelLifecycleLoaderErrorV1 {
    Failed,
}

pub trait ModelLifecycleLoaderV1: Send + Sync + 'static {
    fn load(
        &self,
        descriptor: &ModelLifecycleDescriptorV1,
    ) -> Result<ModelLifecycleLoadedV1, ModelLifecycleLoaderErrorV1>;
    fn shutdown(&self, loaded: ModelLifecycleLoadedV1) -> Result<(), ModelLifecycleLoaderErrorV1>;
}

pub struct ModelLifecycleLoaderFnsV1<L, S> {
    load: L,
    shutdown: S,
}

impl<L, S> ModelLifecycleLoaderFnsV1<L, S> {
    pub fn new(load: L, shutdown: S) -> Self {
        Self { load, shutdown }
    }
}

impl<L, S, LE, SE> ModelLifecycleLoaderV1 for ModelLifecycleLoaderFnsV1<L, S>
where
    L: Fn(&ModelLifecycleDescriptorV1) -> Result<ModelLifecycleLoadedV1, LE>
        + Send
        + Sync
        + 'static,
    S: Fn(ModelLifecycleLoadedV1) -> Result<(), SE> + Send + Sync + 'static,
{
    fn load(
        &self,
        descriptor: &ModelLifecycleDescriptorV1,
    ) -> Result<ModelLifecycleLoadedV1, ModelLifecycleLoaderErrorV1> {
        (self.load)(descriptor).map_err(|_| ModelLifecycleLoaderErrorV1::Failed)
    }
    fn shutdown(&self, loaded: ModelLifecycleLoadedV1) -> Result<(), ModelLifecycleLoaderErrorV1> {
        (self.shutdown)(loaded).map_err(|_| ModelLifecycleLoaderErrorV1::Failed)
    }
}

pub fn model_lifecycle_loader_from_fns<L, S, LE, SE>(
    load: L,
    shutdown: S,
) -> Arc<dyn ModelLifecycleLoaderV1>
where
    L: Fn(&ModelLifecycleDescriptorV1) -> Result<ModelLifecycleLoadedV1, LE>
        + Send
        + Sync
        + 'static,
    S: Fn(ModelLifecycleLoadedV1) -> Result<(), SE> + Send + Sync + 'static,
{
    Arc::new(ModelLifecycleLoaderFnsV1::new(load, shutdown))
}

#[derive(Clone, Debug)]
pub struct ModelLifecycleConfigV1 {
    resident_byte_quota: u64,
    load_wait_timeout: Duration,
    drain_timeout: Duration,
}

impl ModelLifecycleConfigV1 {
    pub fn new(resident_byte_quota: u64) -> Result<Self, ModelLifecycleErrorV1> {
        if resident_byte_quota == 0 {
            return Err(ModelLifecycleErrorV1::InvalidConfig);
        }
        Ok(Self {
            resident_byte_quota,
            load_wait_timeout: Duration::from_secs(5),
            drain_timeout: Duration::from_secs(5),
        })
    }

    pub fn with_timeouts(
        mut self,
        load_wait_timeout: Duration,
        drain_timeout: Duration,
    ) -> Result<Self, ModelLifecycleErrorV1> {
        if load_wait_timeout.is_zero() || drain_timeout.is_zero() {
            return Err(ModelLifecycleErrorV1::InvalidConfig);
        }
        self.load_wait_timeout = load_wait_timeout;
        self.drain_timeout = drain_timeout;
        Ok(self)
    }
    pub const fn resident_byte_quota(&self) -> u64 {
        self.resident_byte_quota
    }
    pub const fn load_wait_timeout(&self) -> Duration {
        self.load_wait_timeout
    }
    pub const fn drain_timeout(&self) -> Duration {
        self.drain_timeout
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum ModelLifecycleStateV1 {
    Unloaded,
    Loading,
    Ready,
    Draining,
    Failed,
    Quarantined,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ModelLifecycleSnapshotV1 {
    pub alias: String,
    pub state: ModelLifecycleStateV1,
    pub active_leases: usize,
    pub resident_bytes: u64,
    pub last_used: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelLifecycleErrorV1 {
    InvalidConfig,
    InvalidDescriptor,
    TooManyConfiguredAliases,
    DuplicateAlias,
    AliasNotFound,
    AliasBusy,
    ModelLoading,
    ModelDraining,
    Quarantined,
    QuarantineNeedsClear,
    LoaderFailed,
    IdentityMismatch,
    LoadingTimeout,
    CapacityExceeded,
    QuotaExceeded,
    DrainTimeout,
    ShutdownFailed,
    StaleCompletion,
}

impl fmt::Display for ModelLifecycleErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "model lifecycle operation failed: {:?}", self)
    }
}
impl std::error::Error for ModelLifecycleErrorV1 {}

struct Record {
    descriptor: ModelLifecycleDescriptorV1,
    state: ModelLifecycleStateV1,
    generation: u64,
    active_leases: usize,
    resident_bytes: u64,
    last_used: u64,
    owner: Option<ModelLifecycleLoadedV1>,
    shutdown_started: bool,
}

struct RegistryState {
    records: BTreeMap<String, Record>,
    loaded_models: usize,
    resident_bytes: u64,
    clock: u64,
}

struct Inner {
    state: Mutex<RegistryState>,
    changed: Condvar,
    loader: Arc<dyn ModelLifecycleLoaderV1>,
    config: ModelLifecycleConfigV1,
}

#[derive(Clone)]
pub struct ModelLifecycleRegistryV1 {
    inner: Arc<Inner>,
}

pub struct ModelLifecycleLeaseV1 {
    inner: Weak<Inner>,
    alias: String,
    generation: u64,
    owner: Arc<ModelRegistryEntryV1>,
    identity: ModelLifecycleIdentityV1,
    resident_bytes: u64,
}

impl ModelLifecycleLeaseV1 {
    pub fn alias(&self) -> &str {
        &self.alias
    }
    pub const fn generation(&self) -> u64 {
        self.generation
    }
    pub fn owner(&self) -> Arc<ModelRegistryEntryV1> {
        Arc::clone(&self.owner)
    }
    pub fn identity(&self) -> &ModelLifecycleIdentityV1 {
        &self.identity
    }
    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }
}

impl Drop for ModelLifecycleLeaseV1 {
    fn drop(&mut self) {
        let Some(inner) = self.inner.upgrade() else {
            return;
        };
        let mut state = inner.state.lock().expect("lifecycle mutex poisoned");
        state.clock = state.clock.saturating_add(1);
        let now = state.clock;
        if let Some(record) = state.records.get_mut(&self.alias) {
            if (record.generation == self.generation
                || record.state == ModelLifecycleStateV1::Draining)
                && record.active_leases > 0
            {
                record.active_leases -= 1;
                record.last_used = now;
                inner.changed.notify_all();
            }
        }
    }
}

impl ModelLifecycleRegistryV1 {
    pub fn new<I>(
        descriptors: I,
        loader: Arc<dyn ModelLifecycleLoaderV1>,
        config: ModelLifecycleConfigV1,
    ) -> Result<Self, ModelLifecycleErrorV1>
    where
        I: IntoIterator<Item = ModelLifecycleDescriptorV1>,
    {
        let mut records = BTreeMap::new();
        for descriptor in descriptors {
            descriptor.validate()?;
            if records.len() >= MAX_CONFIGURED_ALIASES_V1 {
                return Err(ModelLifecycleErrorV1::TooManyConfiguredAliases);
            }
            if records.contains_key(descriptor.alias()) {
                return Err(ModelLifecycleErrorV1::DuplicateAlias);
            }
            let alias = descriptor.alias().to_owned();
            records.insert(
                alias,
                Record {
                    descriptor,
                    state: ModelLifecycleStateV1::Unloaded,
                    generation: 0,
                    active_leases: 0,
                    resident_bytes: 0,
                    last_used: 0,
                    owner: None,
                    shutdown_started: false,
                },
            );
        }
        Ok(Self {
            inner: Arc::new(Inner {
                state: Mutex::new(RegistryState {
                    records,
                    loaded_models: 0,
                    resident_bytes: 0,
                    clock: 0,
                }),
                changed: Condvar::new(),
                loader,
                config,
            }),
        })
    }

    pub fn new_with_fns<I, L, S, LE, SE>(
        descriptors: I,
        load: L,
        shutdown: S,
        config: ModelLifecycleConfigV1,
    ) -> Result<Self, ModelLifecycleErrorV1>
    where
        I: IntoIterator<Item = ModelLifecycleDescriptorV1>,
        L: Fn(&ModelLifecycleDescriptorV1) -> Result<ModelLifecycleLoadedV1, LE>
            + Send
            + Sync
            + 'static,
        S: Fn(ModelLifecycleLoadedV1) -> Result<(), SE> + Send + Sync + 'static,
    {
        Self::new(
            descriptors,
            model_lifecycle_loader_from_fns(load, shutdown),
            config,
        )
    }

    pub fn configured_aliases(&self) -> Vec<String> {
        self.inner
            .state
            .lock()
            .expect("lifecycle mutex poisoned")
            .records
            .keys()
            .cloned()
            .collect()
    }

    /// Adds a new unloaded alias to the bounded registry. The loader remains
    /// model-neutral and resolves the descriptor when the alias is loaded.
    pub fn register(
        &self,
        descriptor: ModelLifecycleDescriptorV1,
    ) -> Result<(), ModelLifecycleErrorV1> {
        descriptor.validate()?;
        let mut state = self.inner.state.lock().expect("lifecycle mutex poisoned");
        if state.records.len() >= MAX_CONFIGURED_ALIASES_V1 {
            return Err(ModelLifecycleErrorV1::TooManyConfiguredAliases);
        }
        if state.records.contains_key(descriptor.alias()) {
            return Err(ModelLifecycleErrorV1::DuplicateAlias);
        }
        let alias = descriptor.alias().to_owned();
        state.records.insert(
            alias,
            Record {
                descriptor,
                state: ModelLifecycleStateV1::Unloaded,
                generation: 0,
                active_leases: 0,
                resident_bytes: 0,
                last_used: 0,
                owner: None,
                shutdown_started: false,
            },
        );
        self.inner.changed.notify_all();
        Ok(())
    }

    /// Removes an idle non-resident alias. Loaded, loading, draining, leased,
    /// or cleanup-quarantined aliases must be made idle before removal.
    pub fn unregister(&self, alias: &str) -> Result<(), ModelLifecycleErrorV1> {
        let mut state = self.inner.state.lock().expect("lifecycle mutex poisoned");
        let record = state
            .records
            .get(alias)
            .ok_or(ModelLifecycleErrorV1::AliasNotFound)?;
        if record.active_leases > 0
            || record.resident_bytes > 0
            || matches!(
                record.state,
                ModelLifecycleStateV1::Loading
                    | ModelLifecycleStateV1::Ready
                    | ModelLifecycleStateV1::Draining
            )
        {
            return Err(ModelLifecycleErrorV1::AliasBusy);
        }
        state.records.remove(alias);
        self.inner.changed.notify_all();
        Ok(())
    }
    pub fn loaded_count(&self) -> usize {
        self.inner
            .state
            .lock()
            .expect("lifecycle mutex poisoned")
            .loaded_models
    }
    pub fn resident_bytes(&self) -> u64 {
        self.inner
            .state
            .lock()
            .expect("lifecycle mutex poisoned")
            .resident_bytes
    }

    pub fn snapshot(&self, alias: &str) -> Result<ModelLifecycleSnapshotV1, ModelLifecycleErrorV1> {
        let state = self.inner.state.lock().expect("lifecycle mutex poisoned");
        let record = state
            .records
            .get(alias)
            .ok_or(ModelLifecycleErrorV1::AliasNotFound)?;
        Ok(snapshot_of(alias, record))
    }
    pub fn snapshots(&self) -> Vec<ModelLifecycleSnapshotV1> {
        let state = self.inner.state.lock().expect("lifecycle mutex poisoned");
        state
            .records
            .iter()
            .map(|(alias, record)| snapshot_of(alias, record))
            .collect()
    }

    /// Returns runtime memory observations for models that currently own a
    /// loaded backend. Unloaded and failed records intentionally contribute no
    /// snapshot so metrics render their fixed zero-valued series.
    pub fn observability_snapshots(&self) -> Vec<(String, BackendObservabilitySnapshotV1)> {
        let state = self.inner.state.lock().expect("lifecycle mutex poisoned");
        state
            .records
            .iter()
            .filter_map(|(alias, record)| {
                record
                    .owner
                    .as_ref()
                    .map(|loaded| (alias.clone(), loaded.owner.observability_snapshot()))
            })
            .collect()
    }

    pub fn resolve(&self, alias: &str) -> Result<ModelLifecycleLeaseV1, ModelLifecycleErrorV1> {
        let deadline = Instant::now() + self.inner.config.load_wait_timeout;
        loop {
            let (descriptor, generation) = {
                let mut state = self.inner.state.lock().expect("lifecycle mutex poisoned");
                state.clock = state.clock.saturating_add(1);
                let now = state.clock;
                let record = state
                    .records
                    .get_mut(alias)
                    .ok_or(ModelLifecycleErrorV1::AliasNotFound)?;
                match record.state {
                    ModelLifecycleStateV1::Ready => {
                        let owner = record
                            .owner
                            .as_ref()
                            .ok_or(ModelLifecycleErrorV1::StaleCompletion)?
                            .owner();
                        record.active_leases += 1;
                        record.last_used = now;
                        return Ok(ModelLifecycleLeaseV1 {
                            inner: Arc::downgrade(&self.inner),
                            alias: alias.to_owned(),
                            generation: record.generation,
                            owner,
                            identity: record
                                .owner
                                .as_ref()
                                .expect("owner checked")
                                .identity()
                                .clone(),
                            resident_bytes: record.resident_bytes,
                        });
                    }
                    ModelLifecycleStateV1::Unloaded => {
                        record.state = ModelLifecycleStateV1::Loading;
                        record.generation = record.generation.wrapping_add(1);
                        (record.descriptor.clone(), record.generation)
                    }
                    ModelLifecycleStateV1::Loading => {
                        wait_for_change(&self.inner, state, deadline)?;
                        continue;
                    }
                    ModelLifecycleStateV1::Draining => {
                        return Err(ModelLifecycleErrorV1::ModelDraining);
                    }
                    ModelLifecycleStateV1::Failed => {
                        return Err(ModelLifecycleErrorV1::LoaderFailed);
                    }
                    ModelLifecycleStateV1::Quarantined => {
                        return Err(ModelLifecycleErrorV1::Quarantined);
                    }
                }
            };
            let loaded = match self.inner.loader.load(&descriptor) {
                Ok(loaded) => loaded,
                Err(_) => {
                    self.mark_failed(alias, generation, true, 0);
                    return Err(ModelLifecycleErrorV1::LoaderFailed);
                }
            };
            if loaded.identity() != descriptor.identity()
                || loaded.resident_bytes() != descriptor.declared_resident_bytes()
                || loaded.owner().alias() != descriptor.alias()
            {
                let cleanup_owner = loaded.clone();
                let shutdown = self.inner.loader.shutdown(loaded);
                self.finish_load_cleanup(alias, generation, cleanup_owner, shutdown.is_err(), true);
                return Err(ModelLifecycleErrorV1::IdentityMismatch);
            }
            return self.publish_loaded(alias, descriptor, generation, loaded);
        }
    }

    fn mark_failed(&self, alias: &str, generation: u64, quarantine: bool, resident_bytes: u64) {
        let mut state = self.inner.state.lock().expect("lifecycle mutex poisoned");
        if let Some(record) = state.records.get_mut(alias) {
            if record.state == ModelLifecycleStateV1::Draining && record.generation != generation {
                record.state = ModelLifecycleStateV1::Unloaded;
                self.inner.changed.notify_all();
            } else if record.state == ModelLifecycleStateV1::Loading
                && record.generation == generation
            {
                record.state = if quarantine {
                    ModelLifecycleStateV1::Quarantined
                } else {
                    ModelLifecycleStateV1::Failed
                };
                record.resident_bytes = resident_bytes;
                record.owner = None;
                if resident_bytes > 0 {
                    state.loaded_models += 1;
                    state.resident_bytes = state.resident_bytes.saturating_add(resident_bytes);
                }
                self.inner.changed.notify_all();
            }
        }
    }

    fn finish_load_cleanup(
        &self,
        alias: &str,
        generation: u64,
        owner: ModelLifecycleLoadedV1,
        cleanup_failed: bool,
        quarantine_on_success: bool,
    ) {
        let mut state = self.inner.state.lock().expect("lifecycle mutex poisoned");
        let target = state.records.get(alias).and_then(|record| {
            if record.state == ModelLifecycleStateV1::Loading && record.generation == generation {
                Some(if cleanup_failed || quarantine_on_success {
                    ModelLifecycleStateV1::Quarantined
                } else {
                    ModelLifecycleStateV1::Failed
                })
            } else if record.state == ModelLifecycleStateV1::Draining
                && record.generation != generation
            {
                Some(if cleanup_failed {
                    ModelLifecycleStateV1::Quarantined
                } else {
                    ModelLifecycleStateV1::Unloaded
                })
            } else {
                None
            }
        });
        let Some(target) = target else {
            return;
        };
        if cleanup_failed {
            state.loaded_models += 1;
            state.resident_bytes = state.resident_bytes.saturating_add(owner.resident_bytes());
        }
        let record = state.records.get_mut(alias).expect("record checked");
        record.state = target;
        record.owner = cleanup_failed.then_some(owner.clone());
        record.resident_bytes = if cleanup_failed {
            owner.resident_bytes()
        } else {
            0
        };
        record.shutdown_started = false;
        self.inner.changed.notify_all();
    }

    fn publish_loaded(
        &self,
        alias: &str,
        descriptor: ModelLifecycleDescriptorV1,
        generation: u64,
        loaded: ModelLifecycleLoadedV1,
    ) -> Result<ModelLifecycleLeaseV1, ModelLifecycleErrorV1> {
        loop {
            let victim = {
                let mut state = self.inner.state.lock().expect("lifecycle mutex poisoned");
                let current = state
                    .records
                    .get(alias)
                    .ok_or(ModelLifecycleErrorV1::AliasNotFound)?;
                if current.state != ModelLifecycleStateV1::Loading
                    || current.generation != generation
                {
                    None
                } else if loaded.resident_bytes() > self.inner.config.resident_byte_quota() {
                    drop(state);
                    let cleanup_owner = loaded.clone();
                    let shutdown = self.inner.loader.shutdown(loaded);
                    self.finish_load_cleanup(
                        alias,
                        generation,
                        cleanup_owner,
                        shutdown.is_err(),
                        false,
                    );
                    return Err(ModelLifecycleErrorV1::QuotaExceeded);
                } else if state.loaded_models < MAX_LOADED_MODELS_V1
                    && state.resident_bytes.saturating_add(loaded.resident_bytes())
                        <= self.inner.config.resident_byte_quota()
                {
                    state.clock = state.clock.saturating_add(1);
                    let now = state.clock;
                    {
                        let state_record = state.records.get_mut(alias).expect("record checked");
                        state_record.owner = Some(loaded.clone());
                        state_record.resident_bytes = loaded.resident_bytes();
                        state_record.state = ModelLifecycleStateV1::Ready;
                        state_record.shutdown_started = false;
                        state_record.last_used = now;
                        state_record.active_leases = 1;
                    }
                    state.loaded_models += 1;
                    state.resident_bytes += loaded.resident_bytes();
                    self.inner.changed.notify_all();
                    return Ok(ModelLifecycleLeaseV1 {
                        inner: Arc::downgrade(&self.inner),
                        alias: alias.to_owned(),
                        generation,
                        owner: loaded.owner(),
                        identity: loaded.identity().clone(),
                        resident_bytes: loaded.resident_bytes(),
                    });
                } else {
                    let candidate = state
                        .records
                        .iter()
                        .filter(|(name, record)| {
                            *name != alias
                                && record.state == ModelLifecycleStateV1::Ready
                                && record.active_leases == 0
                        })
                        .min_by_key(|(name, record)| (record.last_used, (*name).clone()))
                        .map(|(name, record)| (name.clone(), record.generation));
                    if let Some((victim_alias, _victim_generation)) = candidate {
                        let record = state
                            .records
                            .get_mut(&victim_alias)
                            .expect("candidate checked");
                        record.state = ModelLifecycleStateV1::Draining;
                        record.generation = record.generation.wrapping_add(1);
                        record.shutdown_started = true;
                        let owner = record.owner.take().expect("ready record has owner");
                        Some((victim_alias, record.generation, owner))
                    } else {
                        None
                    }
                }
            };
            let Some((victim_alias, victim_generation, victim_owner)) = victim else {
                if self
                    .inner
                    .state
                    .lock()
                    .expect("lifecycle mutex poisoned")
                    .records
                    .get(alias)
                    .is_some_and(|record| {
                        record.state != ModelLifecycleStateV1::Loading
                            || record.generation != generation
                    })
                {
                    let cleanup_owner = loaded.clone();
                    let shutdown = self.inner.loader.shutdown(loaded);
                    self.finish_load_cleanup(
                        alias,
                        generation,
                        cleanup_owner,
                        shutdown.is_err(),
                        false,
                    );
                    return Err(ModelLifecycleErrorV1::StaleCompletion);
                }
                let cleanup_owner = loaded.clone();
                let shutdown = self.inner.loader.shutdown(loaded);
                let state = self.inner.state.lock().expect("lifecycle mutex poisoned");
                let over_quota = state
                    .resident_bytes
                    .saturating_add(descriptor.declared_resident_bytes())
                    > self.inner.config.resident_byte_quota();
                drop(state);
                self.finish_load_cleanup(
                    alias,
                    generation,
                    cleanup_owner,
                    shutdown.is_err(),
                    false,
                );
                return Err(if over_quota {
                    ModelLifecycleErrorV1::QuotaExceeded
                } else {
                    ModelLifecycleErrorV1::CapacityExceeded
                });
            };
            let result = self.inner.loader.shutdown(victim_owner.clone());
            self.finish_shutdown(&victim_alias, victim_generation, victim_owner, result);
        }
    }

    fn finish_shutdown(
        &self,
        alias: &str,
        generation: u64,
        owner: ModelLifecycleLoadedV1,
        result: Result<(), ModelLifecycleLoaderErrorV1>,
    ) {
        let mut state = self.inner.state.lock().expect("lifecycle mutex poisoned");
        let matches = state.records.get(alias).is_some_and(|record| {
            record.state == ModelLifecycleStateV1::Draining && record.generation == generation
        });
        if !matches {
            return;
        }
        if result.is_ok() {
            state.loaded_models = state.loaded_models.saturating_sub(1);
            state.resident_bytes = state.resident_bytes.saturating_sub(owner.resident_bytes());
            let record = state.records.get_mut(alias).expect("record checked");
            record.resident_bytes = 0;
            record.owner = None;
            record.state = ModelLifecycleStateV1::Unloaded;
        } else {
            let record = state.records.get_mut(alias).expect("record checked");
            record.state = ModelLifecycleStateV1::Quarantined;
            record.resident_bytes = owner.resident_bytes();
            record.owner = Some(owner);
            record.shutdown_started = false;
        }
        if result.is_ok() {
            let record = state.records.get_mut(alias).expect("record checked");
            record.shutdown_started = false;
        }
        self.inner.changed.notify_all();
    }

    pub fn unload(&self, alias: &str) -> Result<(), ModelLifecycleErrorV1> {
        let deadline = Instant::now() + self.inner.config.drain_timeout;
        loop {
            let pending = {
                let mut state = self.inner.state.lock().expect("lifecycle mutex poisoned");
                let record = state
                    .records
                    .get_mut(alias)
                    .ok_or(ModelLifecycleErrorV1::AliasNotFound)?;
                match record.state {
                    ModelLifecycleStateV1::Unloaded | ModelLifecycleStateV1::Failed => {
                        return Ok(());
                    }
                    ModelLifecycleStateV1::Quarantined => {
                        return Err(ModelLifecycleErrorV1::Quarantined);
                    }
                    ModelLifecycleStateV1::Loading => {
                        record.state = ModelLifecycleStateV1::Draining;
                        record.generation = record.generation.wrapping_add(1);
                        self.inner.changed.notify_all();
                        None
                    }
                    ModelLifecycleStateV1::Ready => {
                        record.state = ModelLifecycleStateV1::Draining;
                        record.generation = record.generation.wrapping_add(1);
                        if record.active_leases == 0 {
                            record.shutdown_started = true;
                            Some((record.generation, record.owner.take().expect("ready owner")))
                        } else {
                            self.inner.changed.notify_all();
                            None
                        }
                    }
                    ModelLifecycleStateV1::Draining => {
                        if record.active_leases == 0 && !record.shutdown_started {
                            if let Some(owner) = record.owner.take() {
                                record.shutdown_started = true;
                                Some((record.generation, owner))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                }
            };
            if let Some((generation, owner)) = pending {
                let result = self.inner.loader.shutdown(owner.clone());
                self.finish_shutdown(alias, generation, owner, result);
                return if result.is_ok() {
                    Ok(())
                } else {
                    Err(ModelLifecycleErrorV1::ShutdownFailed)
                };
            }
            let state = self.inner.state.lock().expect("lifecycle mutex poisoned");
            wait_for_change(&self.inner, state, deadline).map_err(|error| match error {
                ModelLifecycleErrorV1::LoadingTimeout => ModelLifecycleErrorV1::DrainTimeout,
                other => other,
            })?;
        }
    }

    pub fn retry(&self, alias: &str) -> Result<(), ModelLifecycleErrorV1> {
        let mut state = self.inner.state.lock().expect("lifecycle mutex poisoned");
        let record = state
            .records
            .get_mut(alias)
            .ok_or(ModelLifecycleErrorV1::AliasNotFound)?;
        match record.state {
            ModelLifecycleStateV1::Failed => {
                record.state = ModelLifecycleStateV1::Unloaded;
                record.generation = record.generation.wrapping_add(1);
                self.inner.changed.notify_all();
                Ok(())
            }
            ModelLifecycleStateV1::Quarantined if record.resident_bytes == 0 => {
                record.state = ModelLifecycleStateV1::Unloaded;
                record.generation = record.generation.wrapping_add(1);
                self.inner.changed.notify_all();
                Ok(())
            }
            ModelLifecycleStateV1::Quarantined => Err(ModelLifecycleErrorV1::QuarantineNeedsClear),
            ModelLifecycleStateV1::Loading => Err(ModelLifecycleErrorV1::ModelLoading),
            ModelLifecycleStateV1::Draining => Err(ModelLifecycleErrorV1::ModelDraining),
            _ => Err(ModelLifecycleErrorV1::AliasBusy),
        }
    }

    pub fn clear_quarantine(&self, alias: &str) -> Result<(), ModelLifecycleErrorV1> {
        let owner = {
            let mut state = self.inner.state.lock().expect("lifecycle mutex poisoned");
            let (owner, resident_bytes) = {
                let record = state
                    .records
                    .get_mut(alias)
                    .ok_or(ModelLifecycleErrorV1::AliasNotFound)?;
                if record.state != ModelLifecycleStateV1::Quarantined {
                    return Err(ModelLifecycleErrorV1::AliasBusy);
                }
                if record.shutdown_started {
                    return Err(ModelLifecycleErrorV1::AliasBusy);
                }
                let owner = record.owner.take();
                if owner.is_some() {
                    record.shutdown_started = true;
                }
                (owner, record.resident_bytes)
            };
            let Some(owner) = owner else {
                state.loaded_models = state
                    .loaded_models
                    .saturating_sub(usize::from(resident_bytes > 0));
                state.resident_bytes = state.resident_bytes.saturating_sub(resident_bytes);
                let record = state.records.get_mut(alias).expect("record checked");
                record.resident_bytes = 0;
                record.state = ModelLifecycleStateV1::Unloaded;
                record.generation = record.generation.wrapping_add(1);
                self.inner.changed.notify_all();
                return Ok(());
            };
            owner
        };
        let result = self.inner.loader.shutdown(owner.clone());
        let mut state = self.inner.state.lock().expect("lifecycle mutex poisoned");
        if result.is_ok() {
            state.loaded_models = state.loaded_models.saturating_sub(1);
            state.resident_bytes = state.resident_bytes.saturating_sub(owner.resident_bytes());
            let record = state
                .records
                .get_mut(alias)
                .ok_or(ModelLifecycleErrorV1::AliasNotFound)?;
            record.owner = None;
            record.resident_bytes = 0;
            record.state = ModelLifecycleStateV1::Unloaded;
            record.generation = record.generation.wrapping_add(1);
            record.shutdown_started = false;
            self.inner.changed.notify_all();
            Ok(())
        } else {
            let record = state
                .records
                .get_mut(alias)
                .ok_or(ModelLifecycleErrorV1::AliasNotFound)?;
            record.owner = Some(owner);
            record.shutdown_started = false;
            self.inner.changed.notify_all();
            Err(ModelLifecycleErrorV1::ShutdownFailed)
        }
    }

    /// Load an alias into the bounded resident set, then release the lease.
    pub fn preload(&self, alias: &str) -> Result<(), ModelLifecycleErrorV1> {
        let lease = self.resolve(alias)?;
        drop(lease);
        Ok(())
    }

    /// Shutdown every currently idle Ready model in deterministic LRU order.
    /// Active leases are never evicted.  A shutdown failure quarantines that
    /// model and is returned after the callback has completed.
    pub fn evict_idle(&self) -> Result<usize, ModelLifecycleErrorV1> {
        let mut evicted = 0;
        loop {
            let pending = {
                let mut state = self.inner.state.lock().expect("lifecycle mutex poisoned");
                let candidate = state
                    .records
                    .iter()
                    .filter(|(_, record)| {
                        record.state == ModelLifecycleStateV1::Ready
                            && record.active_leases == 0
                            && !record.shutdown_started
                    })
                    .min_by_key(|(alias, record)| (record.last_used, (*alias).clone()))
                    .map(|(alias, _)| alias.clone());
                let Some(alias) = candidate else {
                    return Ok(evicted);
                };
                let record = state.records.get_mut(&alias).expect("candidate checked");
                record.state = ModelLifecycleStateV1::Draining;
                record.generation = record.generation.wrapping_add(1);
                record.shutdown_started = true;
                (
                    alias,
                    record.generation,
                    record.owner.take().expect("ready owner"),
                )
            };
            let (alias, generation, owner) = pending;
            let result = self.inner.loader.shutdown(owner.clone());
            let failed = result.is_err();
            self.finish_shutdown(&alias, generation, owner, result);
            evicted += 1;
            if failed {
                return Err(ModelLifecycleErrorV1::ShutdownFailed);
            }
        }
    }

    pub fn rebind(
        &self,
        descriptor: ModelLifecycleDescriptorV1,
    ) -> Result<(), ModelLifecycleErrorV1> {
        descriptor.validate()?;
        let mut state = self.inner.state.lock().expect("lifecycle mutex poisoned");
        let record = state
            .records
            .get_mut(descriptor.alias())
            .ok_or(ModelLifecycleErrorV1::AliasNotFound)?;
        if record.active_leases > 0
            || matches!(
                record.state,
                ModelLifecycleStateV1::Loading
                    | ModelLifecycleStateV1::Ready
                    | ModelLifecycleStateV1::Draining
            )
        {
            return Err(ModelLifecycleErrorV1::AliasBusy);
        }
        if record.state == ModelLifecycleStateV1::Quarantined && record.resident_bytes > 0 {
            return Err(ModelLifecycleErrorV1::QuarantineNeedsClear);
        }
        record.descriptor = descriptor;
        record.state = ModelLifecycleStateV1::Unloaded;
        record.generation = record.generation.wrapping_add(1);
        self.inner.changed.notify_all();
        Ok(())
    }
}

fn snapshot_of(alias: &str, record: &Record) -> ModelLifecycleSnapshotV1 {
    ModelLifecycleSnapshotV1 {
        alias: alias.to_owned(),
        state: record.state,
        active_leases: record.active_leases,
        resident_bytes: record.resident_bytes,
        last_used: record.last_used,
    }
}

fn wait_for_change(
    inner: &Inner,
    state: std::sync::MutexGuard<'_, RegistryState>,
    deadline: Instant,
) -> Result<(), ModelLifecycleErrorV1> {
    let remaining = deadline.saturating_duration_since(Instant::now());
    if remaining.is_zero() {
        return Err(ModelLifecycleErrorV1::LoadingTimeout);
    }
    let (_guard, result) = inner
        .changed
        .wait_timeout(state, remaining)
        .expect("lifecycle mutex poisoned");
    if result.timed_out() {
        Err(ModelLifecycleErrorV1::LoadingTimeout)
    } else {
        Ok(())
    }
}
