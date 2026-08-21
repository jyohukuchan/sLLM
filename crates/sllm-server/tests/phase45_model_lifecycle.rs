use sllm_frontend::GenerationCancellationV1;
use sllm_server::{
    BackendCompletionV1, BackendErrorV1, ChatCompletionRequestV1, ChatGenerationBackendV1,
    GenerationDeltaSinkV1, MAX_CONFIGURED_ALIASES_V1, MAX_LOADED_MODELS_V1, ModelLifecycleConfigV1,
    ModelLifecycleDescriptorV1, ModelLifecycleErrorV1, ModelLifecycleLoadedV1,
    ModelLifecycleRegistryV1, ModelLifecycleStateV1, ModelRegistryEntryV1,
};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::Duration;

struct NoopBackend;
impl ChatGenerationBackendV1 for NoopBackend {
    fn generate(
        &self,
        _request: &ChatCompletionRequestV1,
        _cancellation: &GenerationCancellationV1,
        _sink: &mut dyn GenerationDeltaSinkV1,
    ) -> Result<BackendCompletionV1, BackendErrorV1> {
        Err(BackendErrorV1::new("test backend is not executable"))
    }
}

fn descriptor(alias: &str, model: &str, bytes: u64) -> ModelLifecycleDescriptorV1 {
    ModelLifecycleDescriptorV1::new(
        alias,
        model,
        "sha256:1111111111111111111111111111111111111111111111111111111111111111",
        "adapter:none-v1",
        bytes,
    )
    .unwrap()
}

fn loaded(descriptor: &ModelLifecycleDescriptorV1) -> ModelLifecycleLoadedV1 {
    let owner = Arc::new(
        ModelRegistryEntryV1::new(
            descriptor.alias(),
            1,
            "phase45-test",
            descriptor.identity().model_identity(),
            Arc::new(NoopBackend),
        )
        .unwrap(),
    );
    ModelLifecycleLoadedV1::new(
        owner,
        descriptor.declared_resident_bytes(),
        descriptor.identity().model_identity(),
        descriptor.identity().plan_identity(),
        descriptor.identity().adapter_identity(),
    )
    .unwrap()
}

fn config(quota: u64) -> ModelLifecycleConfigV1 {
    ModelLifecycleConfigV1::new(quota)
        .unwrap()
        .with_timeouts(Duration::from_millis(100), Duration::from_millis(30))
        .unwrap()
}

#[test]
fn descriptor_is_offline_and_bounds_are_explicit() {
    assert!(
        ModelLifecycleDescriptorV1::new(
            "https://model",
            "sha256:a",
            "sha256:b",
            "adapter:none-v1",
            1
        )
        .is_err()
    );
    assert!(
        ModelLifecycleDescriptorV1::new("m", "https://model", "sha256:b", "adapter:none-v1", 1)
            .is_err()
    );
    assert!(
        ModelLifecycleDescriptorV1::new("m", "sha256:a", "sha256:b", "https://adapter", 1).is_err()
    );
    assert!(
        ModelLifecycleDescriptorV1::new("m", "sha256:a", "sha256:b", "adapter:none-v1", 0).is_err()
    );

    let descriptors = (0..=MAX_CONFIGURED_ALIASES_V1)
        .map(|index| descriptor(&format!("m{index}"), &format!("sha256:{index:064x}"), 1));
    let result = ModelLifecycleRegistryV1::new(
        descriptors,
        Arc::new(sllm_server::ModelLifecycleLoaderFnsV1::new(
            |d: &ModelLifecycleDescriptorV1| Ok::<_, ()>(loaded(d)),
            |_loaded: ModelLifecycleLoadedV1| Ok::<_, ()>(()),
        )),
        config(128),
    );
    assert!(matches!(
        result,
        Err(ModelLifecycleErrorV1::TooManyConfiguredAliases)
    ));
}

#[test]
fn same_alias_load_is_coalesced_and_lease_drop_is_exactly_once() {
    let loads = Arc::new(AtomicUsize::new(0));
    let loads_for_loader = Arc::clone(&loads);
    let registry = Arc::new(
        ModelLifecycleRegistryV1::new_with_fns(
            [descriptor(
                "m",
                "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                1,
            )],
            move |d: &ModelLifecycleDescriptorV1| {
                loads_for_loader.fetch_add(1, Ordering::SeqCst);
                thread::sleep(Duration::from_millis(15));
                Ok::<_, ()>(loaded(d))
            },
            |_loaded: ModelLifecycleLoadedV1| Ok::<_, ()>(()),
            config(8),
        )
        .unwrap(),
    );
    let leases: Vec<_> = (0..8)
        .map(|_| {
            let registry = Arc::clone(&registry);
            thread::spawn(move || registry.resolve("m").unwrap())
        })
        .map(|thread| thread.join().unwrap())
        .collect();
    assert_eq!(loads.load(Ordering::SeqCst), 1);
    assert_eq!(registry.snapshot("m").unwrap().active_leases, 8);
    drop(leases);
    assert_eq!(registry.snapshot("m").unwrap().active_leases, 0);

    // A failed bounded send must return ownership to the caller, whose Drop
    // path releases the lease exactly once.
    let lease = registry.resolve("m").unwrap();
    let failed_send = Err::<(), _>(lease);
    drop(failed_send);
    assert_eq!(registry.snapshot("m").unwrap().active_leases, 0);
}

#[test]
fn active_lease_drains_with_timeout_then_shutdowns() {
    let shutdowns = Arc::new(AtomicUsize::new(0));
    let shutdowns_for_loader = Arc::clone(&shutdowns);
    let registry = ModelLifecycleRegistryV1::new_with_fns(
        [descriptor(
            "m",
            "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            1,
        )],
        |d: &ModelLifecycleDescriptorV1| Ok::<_, ()>(loaded(d)),
        move |_loaded: ModelLifecycleLoadedV1| {
            shutdowns_for_loader.fetch_add(1, Ordering::SeqCst);
            Ok::<_, ()>(())
        },
        config(8),
    )
    .unwrap();
    let lease = registry.resolve("m").unwrap();
    assert_eq!(
        registry.unload("m"),
        Err(ModelLifecycleErrorV1::DrainTimeout)
    );
    assert_eq!(
        registry.snapshot("m").unwrap().state,
        ModelLifecycleStateV1::Draining
    );
    drop(lease);
    assert_eq!(registry.unload("m"), Ok(()));
    assert_eq!(shutdowns.load(Ordering::SeqCst), 1);
}

#[test]
fn loader_failure_after_concurrent_unload_wakes_drain() {
    let entered = Arc::new((Mutex::new(false), Condvar::new()));
    let release = Arc::new(AtomicBool::new(false));
    let entered_for_loader = Arc::clone(&entered);
    let release_for_loader = Arc::clone(&release);
    let registry = Arc::new(
        ModelLifecycleRegistryV1::new_with_fns(
            [descriptor(
                "race",
                "sha256:1212121212121212121212121212121212121212121212121212121212121212",
                1,
            )],
            move |_descriptor: &ModelLifecycleDescriptorV1| {
                let (lock, cv) = &*entered_for_loader;
                *lock.lock().unwrap() = true;
                cv.notify_all();
                while !release_for_loader.load(Ordering::Acquire) {
                    thread::sleep(Duration::from_millis(1));
                }
                Err::<ModelLifecycleLoadedV1, _>(())
            },
            |_loaded: ModelLifecycleLoadedV1| Ok::<_, ()>(()),
            config(8),
        )
        .unwrap(),
    );
    let pending = {
        let registry = Arc::clone(&registry);
        thread::spawn(move || registry.resolve("race"))
    };
    let (lock, cv) = &*entered;
    let mut started = lock.lock().unwrap();
    while !*started {
        started = cv.wait(started).unwrap();
    }
    drop(started);
    let unload = {
        let registry = Arc::clone(&registry);
        thread::spawn(move || registry.unload("race"))
    };
    thread::sleep(Duration::from_millis(5));
    release.store(true, Ordering::Release);
    assert!(matches!(
        pending.join().unwrap(),
        Err(ModelLifecycleErrorV1::LoaderFailed)
    ));
    assert_eq!(unload.join().unwrap(), Ok(()));
    assert_eq!(
        registry.snapshot("race").unwrap().state,
        ModelLifecycleStateV1::Unloaded
    );
}

#[test]
fn failure_quarantines_until_retry_or_explicit_clear() {
    let attempts = Arc::new(AtomicUsize::new(0));
    let attempts_for_loader = Arc::clone(&attempts);
    let registry = ModelLifecycleRegistryV1::new_with_fns(
        [descriptor(
            "m",
            "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
            1,
        )],
        move |d: &ModelLifecycleDescriptorV1| {
            if attempts_for_loader.fetch_add(1, Ordering::SeqCst) == 0 {
                Err::<ModelLifecycleLoadedV1, _>(())
            } else {
                Ok(loaded(d))
            }
        },
        |_loaded: ModelLifecycleLoadedV1| Ok::<_, ()>(()),
        config(8),
    )
    .unwrap();
    assert!(matches!(
        registry.resolve("m"),
        Err(ModelLifecycleErrorV1::LoaderFailed)
    ));
    assert_eq!(
        registry.snapshot("m").unwrap().state,
        ModelLifecycleStateV1::Quarantined
    );
    assert!(matches!(
        registry.resolve("m"),
        Err(ModelLifecycleErrorV1::Quarantined)
    ));
    registry.retry("m").unwrap();
    let lease = registry.resolve("m").unwrap();
    assert_eq!(
        lease.identity().model_identity(),
        "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
    );
}

#[test]
fn resident_quota_evicts_idle_lru_and_never_active_leases() {
    let shutdowns = Arc::new(AtomicUsize::new(0));
    let shutdowns_for_loader = Arc::clone(&shutdowns);
    let registry = ModelLifecycleRegistryV1::new_with_fns(
        [
            descriptor(
                "a",
                "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                1,
            ),
            descriptor(
                "b",
                "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                1,
            ),
            descriptor(
                "c",
                "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
                1,
            ),
        ],
        |d: &ModelLifecycleDescriptorV1| Ok::<_, ()>(loaded(d)),
        move |_loaded: ModelLifecycleLoadedV1| {
            shutdowns_for_loader.fetch_add(1, Ordering::SeqCst);
            Ok::<_, ()>(())
        },
        config(2),
    )
    .unwrap();
    let a = registry.resolve("a").unwrap();
    drop(a);
    let b = registry.resolve("b").unwrap();
    drop(b);
    let c = registry.resolve("c").unwrap();
    assert_eq!(registry.loaded_count(), MAX_LOADED_MODELS_V1.min(2));
    assert_eq!(
        registry.snapshot("a").unwrap().state,
        ModelLifecycleStateV1::Unloaded
    );
    assert_eq!(
        registry.snapshot("c").unwrap().state,
        ModelLifecycleStateV1::Ready
    );
    drop(c);
    assert_eq!(shutdowns.load(Ordering::SeqCst), 1);
}

#[test]
fn lease_drop_wakes_unload_waiter() {
    let registry = Arc::new(
        ModelLifecycleRegistryV1::new_with_fns(
            [descriptor(
                "m",
                "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
                1,
            )],
            |d: &ModelLifecycleDescriptorV1| Ok::<_, ()>(loaded(d)),
            |_loaded: ModelLifecycleLoadedV1| Ok::<_, ()>(()),
            config(8),
        )
        .unwrap(),
    );
    let lease = registry.resolve("m").unwrap();
    let waiter = {
        let registry = Arc::clone(&registry);
        thread::spawn(move || registry.unload("m"))
    };
    thread::sleep(Duration::from_millis(5));
    drop(lease);
    assert_eq!(waiter.join().unwrap(), Ok(()));
}

#[test]
fn preload_and_explicit_idle_eviction_are_bounded() {
    let registry = ModelLifecycleRegistryV1::new_with_fns(
        [descriptor(
            "m",
            "sha256:abababababababababababababababababababababababababababababababab",
            1,
        )],
        |d: &ModelLifecycleDescriptorV1| Ok::<_, ()>(loaded(d)),
        |_loaded: ModelLifecycleLoadedV1| Ok::<_, ()>(()),
        config(8),
    )
    .unwrap();
    registry.preload("m").unwrap();
    assert_eq!(registry.snapshot("m").unwrap().active_leases, 0);
    assert_eq!(registry.evict_idle(), Ok(1));
    assert_eq!(
        registry.snapshot("m").unwrap().state,
        ModelLifecycleStateV1::Unloaded
    );
}

#[test]
fn shutdown_failure_retains_owner_until_clear_retry_succeeds() {
    let shutdowns = Arc::new(AtomicUsize::new(0));
    let shutdowns_for_loader = Arc::clone(&shutdowns);
    let registry = ModelLifecycleRegistryV1::new_with_fns(
        [descriptor(
            "m",
            "sha256:cdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcd",
            1,
        )],
        |d: &ModelLifecycleDescriptorV1| Ok::<_, ()>(loaded(d)),
        move |_loaded: ModelLifecycleLoadedV1| {
            let attempt = shutdowns_for_loader.fetch_add(1, Ordering::SeqCst);
            if attempt < 2 { Err(()) } else { Ok(()) }
        },
        config(8),
    )
    .unwrap();
    drop(registry.resolve("m").unwrap());
    assert_eq!(
        registry.unload("m"),
        Err(ModelLifecycleErrorV1::ShutdownFailed)
    );
    assert_eq!(
        registry.snapshot("m").unwrap().state,
        ModelLifecycleStateV1::Quarantined
    );
    assert_eq!(registry.loaded_count(), 1);
    assert_eq!(
        registry.clear_quarantine("m"),
        Err(ModelLifecycleErrorV1::ShutdownFailed)
    );
    assert_eq!(
        registry.snapshot("m").unwrap().state,
        ModelLifecycleStateV1::Quarantined
    );
    registry.clear_quarantine("m").unwrap();
    assert_eq!(registry.loaded_count(), 0);
    assert_eq!(
        registry.snapshot("m").unwrap().state,
        ModelLifecycleStateV1::Unloaded
    );
    assert_eq!(shutdowns.load(Ordering::SeqCst), 3);
}

#[test]
fn quota_cleanup_failure_quarantines_loaded_owner() {
    let shutdowns = Arc::new(AtomicUsize::new(0));
    let shutdowns_for_loader = Arc::clone(&shutdowns);
    let registry = ModelLifecycleRegistryV1::new_with_fns(
        [descriptor(
            "oversize",
            "sha256:abababababababababababababababababababababababababababababababab",
            9,
        )],
        |d: &ModelLifecycleDescriptorV1| Ok::<_, ()>(loaded(d)),
        move |_loaded: ModelLifecycleLoadedV1| {
            let attempt = shutdowns_for_loader.fetch_add(1, Ordering::SeqCst);
            if attempt == 0 { Err(()) } else { Ok(()) }
        },
        config(8),
    )
    .unwrap();
    assert!(matches!(
        registry.resolve("oversize"),
        Err(ModelLifecycleErrorV1::QuotaExceeded)
    ));
    assert_eq!(
        registry.snapshot("oversize").unwrap().state,
        ModelLifecycleStateV1::Quarantined
    );
    assert_eq!(registry.resident_bytes(), 9);
    registry.clear_quarantine("oversize").unwrap();
    assert_eq!(registry.resident_bytes(), 0);
    assert_eq!(shutdowns.load(Ordering::SeqCst), 2);
}

#[test]
fn capacity_cleanup_failure_quarantines_loaded_owner() {
    let shutdowns = Arc::new(AtomicUsize::new(0));
    let shutdowns_for_loader = Arc::clone(&shutdowns);
    let descriptors = (0..=MAX_LOADED_MODELS_V1).map(|index| {
        descriptor(
            &format!("capacity-{index}"),
            &format!("sha256:{index:064x}"),
            1,
        )
    });
    let registry = ModelLifecycleRegistryV1::new_with_fns(
        descriptors,
        |d: &ModelLifecycleDescriptorV1| Ok::<_, ()>(loaded(d)),
        move |_loaded: ModelLifecycleLoadedV1| {
            let attempt = shutdowns_for_loader.fetch_add(1, Ordering::SeqCst);
            if attempt == 0 { Err(()) } else { Ok(()) }
        },
        config((MAX_LOADED_MODELS_V1 + 1) as u64),
    )
    .unwrap();
    let leases: Vec<_> = (0..MAX_LOADED_MODELS_V1)
        .map(|index| registry.resolve(&format!("capacity-{index}")))
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(registry.loaded_count(), MAX_LOADED_MODELS_V1);
    assert!(matches!(
        registry.resolve(&format!("capacity-{MAX_LOADED_MODELS_V1}")),
        Err(ModelLifecycleErrorV1::CapacityExceeded)
    ));
    let rejected_alias = format!("capacity-{MAX_LOADED_MODELS_V1}");
    assert_eq!(
        registry.snapshot(&rejected_alias).unwrap().state,
        ModelLifecycleStateV1::Quarantined
    );
    assert_eq!(
        registry.snapshot(&rejected_alias).unwrap().resident_bytes,
        1
    );
    assert_eq!(registry.loaded_count(), MAX_LOADED_MODELS_V1 + 1);
    registry.clear_quarantine(&rejected_alias).unwrap();
    assert_eq!(
        registry.snapshot(&rejected_alias).unwrap().state,
        ModelLifecycleStateV1::Unloaded
    );
    assert_eq!(registry.loaded_count(), MAX_LOADED_MODELS_V1);
    assert_eq!(shutdowns.load(Ordering::SeqCst), 2);
    drop(leases);
}

#[test]
fn stale_load_cleanup_failure_quarantines_loaded_owner() {
    let entered = Arc::new((Mutex::new(false), Condvar::new()));
    let release = Arc::new(AtomicBool::new(false));
    let entered_for_loader = Arc::clone(&entered);
    let release_for_loader = Arc::clone(&release);
    let shutdowns = Arc::new(AtomicUsize::new(0));
    let shutdowns_for_loader = Arc::clone(&shutdowns);
    let registry = Arc::new(
        ModelLifecycleRegistryV1::new_with_fns(
            [descriptor(
                "stale-cleanup",
                "sha256:abababababababababababababababababababababababababababababababab",
                1,
            )],
            move |d: &ModelLifecycleDescriptorV1| {
                let (lock, cv) = &*entered_for_loader;
                *lock.lock().unwrap() = true;
                cv.notify_all();
                while !release_for_loader.load(Ordering::Acquire) {
                    thread::sleep(Duration::from_millis(1));
                }
                Ok::<_, ()>(loaded(d))
            },
            move |_loaded: ModelLifecycleLoadedV1| {
                let attempt = shutdowns_for_loader.fetch_add(1, Ordering::SeqCst);
                if attempt == 0 { Err(()) } else { Ok(()) }
            },
            config(8),
        )
        .unwrap(),
    );
    let pending = {
        let registry = Arc::clone(&registry);
        thread::spawn(move || registry.resolve("stale-cleanup"))
    };
    let (lock, cv) = &*entered;
    let mut started = lock.lock().unwrap();
    while !*started {
        started = cv.wait(started).unwrap();
    }
    drop(started);
    let unload = {
        let registry = Arc::clone(&registry);
        thread::spawn(move || registry.unload("stale-cleanup"))
    };
    for _ in 0..100 {
        if registry.snapshot("stale-cleanup").unwrap().state == ModelLifecycleStateV1::Draining {
            break;
        }
        thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(
        registry.snapshot("stale-cleanup").unwrap().state,
        ModelLifecycleStateV1::Draining
    );
    release.store(true, Ordering::Release);
    assert!(matches!(
        pending.join().unwrap(),
        Err(ModelLifecycleErrorV1::StaleCompletion)
    ));
    assert_eq!(
        unload.join().unwrap(),
        Err(ModelLifecycleErrorV1::Quarantined)
    );
    assert_eq!(
        registry.snapshot("stale-cleanup").unwrap().state,
        ModelLifecycleStateV1::Quarantined
    );
    assert_eq!(registry.resident_bytes(), 1);
    registry.clear_quarantine("stale-cleanup").unwrap();
    assert_eq!(registry.resident_bytes(), 0);
    assert_eq!(registry.loaded_count(), 0);
    assert_eq!(shutdowns.load(Ordering::SeqCst), 2);
}

#[test]
fn rebind_rejects_loading_and_preserves_completion() {
    let entered = Arc::new((Mutex::new(false), Condvar::new()));
    let release = Arc::new(AtomicBool::new(false));
    let entered_for_loader = Arc::clone(&entered);
    let release_for_loader = Arc::clone(&release);
    let shutdowns = Arc::new(AtomicUsize::new(0));
    let shutdowns_for_loader = Arc::clone(&shutdowns);
    let old = descriptor(
        "m",
        "sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
        1,
    );
    let new = descriptor(
        "m",
        "sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        1,
    );
    let registry = Arc::new(
        ModelLifecycleRegistryV1::new_with_fns(
            [old.clone()],
            move |d: &ModelLifecycleDescriptorV1| {
                let (lock, cv) = &*entered_for_loader;
                *lock.lock().unwrap() = true;
                cv.notify_all();
                while !release_for_loader.load(Ordering::Acquire) {
                    thread::sleep(Duration::from_millis(1));
                }
                Ok::<_, ()>(loaded(d))
            },
            move |_loaded: ModelLifecycleLoadedV1| {
                shutdowns_for_loader.fetch_add(1, Ordering::SeqCst);
                Ok::<_, ()>(())
            },
            config(8),
        )
        .unwrap(),
    );
    let pending = {
        let registry = Arc::clone(&registry);
        thread::spawn(move || registry.resolve("m"))
    };
    let (lock, cv) = &*entered;
    let mut started = lock.lock().unwrap();
    while !*started {
        started = cv.wait(started).unwrap();
    }
    drop(started);
    assert_eq!(
        registry.rebind(new.clone()),
        Err(ModelLifecycleErrorV1::AliasBusy)
    );
    release.store(true, Ordering::Release);
    let old_lease = pending.join().unwrap().unwrap();
    assert_eq!(
        old_lease.identity().model_identity(),
        old.identity().model_identity()
    );
    drop(old_lease);
    registry.unload("m").unwrap();
    assert_eq!(shutdowns.load(Ordering::SeqCst), 1);
    registry.rebind(new).unwrap();
    let lease = registry.resolve("m").unwrap();
    assert_eq!(
        lease.identity().model_identity(),
        "sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
    );
    let owner = lease.owner();
    assert_eq!(owner.alias(), lease.alias());
    assert_eq!(owner.lock_fingerprint(), lease.identity().model_identity());
}
