//! C3a2 request-local FP16/FP8/NVFP4 KV ownership and append transactions.
//!
//! Native KV handles are deliberately kept opaque. The only Rust-visible
//! state is copied metadata; the two device allocations and their strides
//! remain native-owned. The sendable resource token is an erased-core
//! lifetime boundary, while the direct state/view owner types carry an Rc
//! marker so they cannot be moved or shared as thread-affine native views.

use std::collections::HashMap;
use std::marker::PhantomData;
use std::mem::size_of;
use std::ptr::NonNull;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock, Weak};
use std::time::Duration;

use sllm_core::{
    DType, Encoding, ExecutionSessionId, KvCacheEncoding, KvFp8PhysicalVariant, KvMemoryKind,
    KvStateAppendRequest, KvStateDescriptor, KvStateId, KvStateSnapshot, StateForkAuditV1,
    StateForkModeV1,
};
use sllm_hip_sys as sys;

use crate::Buffer;
use crate::rmsnorm::TensorBinding;
use crate::runtime::{
    CompletionState, Context, Queue, RuntimeError, RuntimeStatus, completion_from_opaque_token,
    enqueue_causal_completion_cleanup, enqueue_kv_completion_cleanup, enqueue_kv_state_cleanup,
    enqueue_kv_view_cleanup, ensure_ok, finalize_completion_after, gcn_arch_matches,
    logical_gcn_arch_name, release_causal_completion_once, release_kv_completion_once,
    release_kv_state_once, release_kv_view_once, result_error, sink,
};

const ERROR_CAPACITY: usize = 256;
const MAX_FINITE_TIMEOUT_MS: u32 = u32::MAX - 1;
const KERNEL_SYMBOL: &str = "kv_state.bf16_to_f16_token_major.v2";
const DEVICE_SYMBOL: &str = "sllm_kv_state_bf16_to_f16_token_major_v2";

pub(crate) fn native_kv_storage(
    descriptor: KvStateDescriptor,
    expected_target: Option<&str>,
) -> Result<(u32, u32, u32, u32), RuntimeError> {
    let target = expected_target.map(logical_gcn_arch_name);
    let storage = match descriptor.cache_encoding() {
        KvCacheEncoding::Fp16 => (
            sys::SLLM_TENSOR_DTYPE_F16,
            sys::SLLM_HIP_KV_ENCODING_FP16_V1,
            0,
            0,
        ),
        KvCacheEncoding::Fp8E4M3Fn => (
            sys::SLLM_TENSOR_DTYPE_F8_E4M3_FN,
            sys::SLLM_HIP_KV_ENCODING_FP8_V1,
            0,
            sys::SLLM_TENSOR_DTYPE_F32,
        ),
        KvCacheEncoding::Fp8E4M3FnStatic => (
            sys::SLLM_TENSOR_DTYPE_F8_E4M3_FN,
            sys::SLLM_HIP_KV_ENCODING_FP8_STATIC_V1,
            0,
            sys::SLLM_TENSOR_DTYPE_F32,
        ),
        KvCacheEncoding::Nvfp4 => (
            sys::SLLM_TENSOR_DTYPE_U8,
            sys::SLLM_HIP_KV_ENCODING_NVFP4_V1,
            16,
            sys::SLLM_TENSOR_DTYPE_F8_E4M3_FN,
        ),
        KvCacheEncoding::Fp8E4M3Block16 | KvCacheEncoding::Fp8E5M2Block16 => {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidKvStateDescriptor,
                "KV FP8 block16 has been retired; use standard OCP MXFP8 E4M3 or explicit FP16",
            ));
        }
        KvCacheEncoding::Mxfp8E4 | KvCacheEncoding::Mxfp8E5 => {
            let mxfp8 = descriptor.kv_mxfp8_descriptor().ok_or_else(|| {
                RuntimeError::local(
                    RuntimeStatus::InvalidKvStateDescriptor,
                    "KV MXFP8 encoding is missing its physical descriptor",
                )
            })?;
            let compatible = matches!(
                (target, mxfp8.physical_variant()),
                (None, _)
                    | (
                        Some("gfx1030" | "gfx1201" | "gfx942"),
                        KvFp8PhysicalVariant::OcpE4M3Fn
                    )
                    | (Some("gfx1030"), KvFp8PhysicalVariant::OcpE5M2)
            );
            if !compatible {
                return Err(RuntimeError::new(
                    RuntimeStatus::InvalidKvStateDescriptor,
                    format!(
                        "standard OCP MXFP8 physical variant {:?} is incompatible with target {}",
                        mxfp8.physical_variant(),
                        expected_target.unwrap_or("unspecified")
                    ),
                ));
            }
            let (dtype, encoding) = match mxfp8.physical_variant() {
                KvFp8PhysicalVariant::OcpE4M3Fn => (
                    sys::SLLM_TENSOR_DTYPE_F8_E4M3_FN,
                    sys::SLLM_HIP_KV_ENCODING_MXFP8_E4_V1,
                ),
                KvFp8PhysicalVariant::OcpE5M2 => (
                    sys::SLLM_TENSOR_DTYPE_F8_E5M2,
                    sys::SLLM_HIP_KV_ENCODING_MXFP8_E5_V1,
                ),
                KvFp8PhysicalVariant::E4M3FnuZ => unreachable!(),
            };
            (dtype, encoding, 32, sys::SLLM_TENSOR_DTYPE_U8)
        }
    };
    Ok(storage)
}

const RDNA_CONTIGUOUS_LONG_KV_MIN_TOKENS: u64 = 65_536;

fn selected_memory_kind_for_target(expected_target: Option<&str>, capacity_tokens: u64) -> u32 {
    if expected_target == Some("gfx942")
        || (matches!(expected_target, Some("gfx1030" | "gfx1201"))
            && capacity_tokens >= RDNA_CONTIGUOUS_LONG_KV_MIN_TOKENS)
    {
        sys::SLLM_HIP_KV_MEMORY_KIND_CONTIGUOUS_RESIDENT
    } else {
        sys::SLLM_HIP_KV_MEMORY_KIND_CAPABILITY_SELECTED
    }
}

fn selected_memory_kind(context: &Context, descriptor: KvStateDescriptor) -> u32 {
    if descriptor.sliding_window().is_some() {
        sys::SLLM_HIP_KV_MEMORY_KIND_VIRTUAL_CONTIGUOUS
    } else {
        selected_memory_kind_for_target(context.expected_target(), descriptor.capacity())
    }
}

struct KvStateInner {
    raw: usize,
    context: Context,
    session_id: ExecutionSessionId,
    state_id: KvStateId,
    descriptor: KvStateDescriptor,
    last_generation: AtomicU64,
}

type EvidenceKey = (u64, u64);
type EvidenceResourceMap = HashMap<EvidenceKey, Weak<KvStateInner>>;

static EVIDENCE_RESOURCES: OnceLock<Mutex<EvidenceResourceMap>> = OnceLock::new();

fn evidence_resources() -> &'static Mutex<EvidenceResourceMap> {
    EVIDENCE_RESOURCES.get_or_init(|| Mutex::new(EvidenceResourceMap::new()))
}

impl Drop for KvStateInner {
    fn drop(&mut self) {
        if let Ok(mut resources) = evidence_resources().lock() {
            resources.remove(&(self.session_id.raw(), self.state_id.raw()));
        }
        let Some(raw) = NonNull::new(self.raw as *mut sys::sllm_kv_state_t) else {
            return;
        };
        let (status, remaining) = release_kv_state_once(raw);
        if let Some(remaining) = remaining {
            enqueue_kv_state_cleanup(remaining, self.context.clone(), status);
        }
    }
}

/// Sendable opaque ownership token used by the erased core resource.
///
/// It contains no dereferenceable pointer or writable storage. The native
/// registry owns synchronization and handle validation; the final Arc drop
/// is the only point that releases the state handle.
#[derive(Clone)]
pub(crate) struct KvStateResource {
    inner: Arc<KvStateInner>,
}

impl KvStateResource {
    pub(crate) fn create(
        context: &Context,
        session_id: ExecutionSessionId,
        state_id: KvStateId,
        descriptor: KvStateDescriptor,
    ) -> Result<Self, RuntimeError> {
        if descriptor.capacity() > sys::SLLM_HIP_KV_MAX_CAPACITY {
            return Err(RuntimeError::local(
                RuntimeStatus::KvCapacityExceeded,
                "KV capacity exceeds the bounded native contract",
            ));
        }
        let context_raw = context.raw_handle()?;
        let (dtype, encoding, block_size, scale_dtype) =
            native_kv_storage(descriptor, context.expected_target())?;
        let static_scales = descriptor.static_fp8_scales();
        let mut reserved = [0_u32; 4];
        if let Some((key, value)) = static_scales {
            reserved[0] = key.to_bits();
            reserved[1] = value.to_bits();
        }
        if let Some(window) = descriptor.sliding_window() {
            reserved[2] = window as u32;
            reserved[3] = (window >> 32) as u32;
        }
        let info = sys::sllm_kv_state_create_info_v2_t {
            struct_size: size_of::<sys::sllm_kv_state_create_info_v2_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            create_info_version: if descriptor.sliding_window().is_some() {
                sys::SLLM_HIP_KV_STATE_CREATE_INFO_SLIDING_STATIC_FP8_VERSION
            } else if static_scales.is_some() {
                sys::SLLM_HIP_KV_STATE_CREATE_INFO_STATIC_FP8_VERSION
            } else {
                sys::SLLM_HIP_KV_STATE_CREATE_INFO_V2_VERSION
            },
            reserved0: 0,
            session_id: session_id.raw(),
            layer_id: descriptor.layer_id(),
            flags: 0,
            capacity_tokens: descriptor.capacity(),
            head_count: descriptor.layout().heads() as u32,
            head_dim: descriptor.layout().head_dim() as u32,
            memory_kind: selected_memory_kind(context, descriptor),
            layout: sys::SLLM_HIP_KV_LAYOUT_TOKEN_MAJOR,
            dtype,
            encoding,
            block_size,
            scale_dtype,
            reserved,
        };
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let mut raw_state = std::ptr::null_mut();
        let raw = unsafe {
            sys::sllm_kv_state_create_v2(
                context_raw.as_ptr(),
                &info,
                &mut raw_state,
                &mut error_sink,
            )
        };
        ensure_ok(raw, &error_buffer, error_sink.message_length)?;
        let raw_state = NonNull::new(raw_state).ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::InternalError,
                "native KV state create returned a null handle on success",
            )
        })?;
        let resource = Self {
            inner: Arc::new(KvStateInner {
                raw: raw_state.as_ptr() as usize,
                context: context.clone(),
                session_id,
                state_id,
                descriptor,
                last_generation: AtomicU64::new(0),
            }),
        };
        evidence_resources()
            .lock()
            .map_err(|_| {
                RuntimeError::local(
                    RuntimeStatus::InternalError,
                    "KV evidence resource registry is poisoned",
                )
            })?
            .insert(
                (session_id.raw(), state_id.raw()),
                Arc::downgrade(&resource.inner),
            );
        Ok(resource)
    }

    fn raw_handle(&self) -> Result<NonNull<sys::sllm_kv_state_t>, RuntimeError> {
        NonNull::new(self.inner.raw as *mut sys::sllm_kv_state_t).ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::InvalidHandle,
                "KV state was already released",
            )
        })
    }

    /// Forks the native state while preserving exact encoded planes.  Layout,
    /// encoding, and scales must match; capacity may grow for a reused prefix.
    pub(crate) fn fork(
        &self,
        state_id: KvStateId,
        descriptor: KvStateDescriptor,
    ) -> Result<(Self, StateForkAuditV1), RuntimeError> {
        if descriptor.layer_id() != self.inner.descriptor.layer_id()
            || descriptor.layout() != self.inner.descriptor.layout()
            || descriptor.cache_encoding() != self.inner.descriptor.cache_encoding()
            || descriptor.static_fp8_scales() != self.inner.descriptor.static_fp8_scales()
            || descriptor.kv_fp8_block16_descriptor()
                != self.inner.descriptor.kv_fp8_block16_descriptor()
            || descriptor.kv_mxfp8_descriptor() != self.inner.descriptor.kv_mxfp8_descriptor()
            || descriptor.sliding_window() != self.inner.descriptor.sliding_window()
        {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidKvStateDescriptor,
                "native KV fork requires an identical destination descriptor",
            ));
        }
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let mut info = sys::sllm_state_fork_info_t {
            struct_size: size_of::<sys::sllm_state_fork_info_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            info_version: sys::SLLM_HIP_STATE_FORK_INFO_VERSION,
            mode: 0,
            source_state_identity: 0,
            child_state_identity: 0,
            source_owned_bytes: 0,
            child_owned_bytes: 0,
            copied_bytes: 0,
            shared_bytes: 0,
            published_length: 0,
            page_bytes: 0,
            reserved: [0; 4],
        };
        let mut raw_child = std::ptr::null_mut();
        let (dtype, encoding, block_size, scale_dtype) =
            native_kv_storage(descriptor, self.inner.context.expected_target())?;
        let static_scales = descriptor.static_fp8_scales();
        let mut reserved = [0_u32; 4];
        if let Some((key, value)) = static_scales {
            reserved[0] = key.to_bits();
            reserved[1] = value.to_bits();
        }
        if let Some(window) = descriptor.sliding_window() {
            reserved[2] = window as u32;
            reserved[3] = (window >> 32) as u32;
        }
        let destination_info = sys::sllm_kv_state_create_info_v2_t {
            struct_size: size_of::<sys::sllm_kv_state_create_info_v2_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            create_info_version: if descriptor.sliding_window().is_some() {
                sys::SLLM_HIP_KV_STATE_CREATE_INFO_SLIDING_STATIC_FP8_VERSION
            } else if static_scales.is_some() {
                sys::SLLM_HIP_KV_STATE_CREATE_INFO_STATIC_FP8_VERSION
            } else {
                sys::SLLM_HIP_KV_STATE_CREATE_INFO_V2_VERSION
            },
            reserved0: 0,
            session_id: self.inner.session_id.raw(),
            layer_id: descriptor.layer_id(),
            flags: 0,
            capacity_tokens: descriptor.capacity(),
            head_count: descriptor.layout().heads() as u32,
            head_dim: descriptor.layout().head_dim() as u32,
            memory_kind: selected_memory_kind(&self.inner.context, descriptor),
            layout: sys::SLLM_HIP_KV_LAYOUT_TOKEN_MAJOR,
            dtype,
            encoding,
            block_size,
            scale_dtype,
            reserved,
        };
        let status = unsafe {
            sys::sllm_kv_state_fork(
                self.raw_handle()?.as_ptr(),
                &destination_info,
                &mut raw_child,
                &mut info,
                &mut error_sink,
            )
        };
        ensure_ok(status, &error_buffer, error_sink.message_length)?;
        let raw_child = NonNull::new(raw_child).ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::InternalError,
                "native KV fork returned a null child handle on success",
            )
        })?;
        let mode = match info.mode {
            sys::SLLM_HIP_STATE_FORK_MODE_SHARED_READ_ONLY_PAGES => {
                StateForkModeV1::SharedReadOnlyPages
            }
            sys::SLLM_HIP_STATE_FORK_MODE_DEVICE_COPY => StateForkModeV1::DeviceCopy,
            _ => {
                let mut child_handle = raw_child.as_ptr();
                let _ = unsafe { sys::sllm_kv_state_release(&mut child_handle, &mut error_sink) };
                return Err(RuntimeError::local(
                    RuntimeStatus::InvalidKvStateDescriptor,
                    "native KV fork returned an unknown mode",
                ));
            }
        };
        let shared_pages = info
            .page_bytes
            .checked_sub(1)
            .and_then(|_| info.shared_bytes.checked_add(info.page_bytes - 1))
            .map(|bytes| bytes / info.page_bytes.max(1))
            .unwrap_or(0);
        let audit = StateForkAuditV1::new(
            mode,
            info.published_length,
            shared_pages,
            info.copied_bytes,
            info.child_owned_bytes,
        )
        .map_err(|error| {
            let mut child_handle = raw_child.as_ptr();
            let _ = unsafe { sys::sllm_kv_state_release(&mut child_handle, &mut error_sink) };
            RuntimeError::new(
                RuntimeStatus::InvalidKvStateDescriptor,
                format!("native KV fork audit failed core validation: {error}"),
            )
        })?;
        let resource = Self {
            inner: Arc::new(KvStateInner {
                raw: raw_child.as_ptr() as usize,
                context: self.inner.context.clone(),
                session_id: self.inner.session_id,
                state_id,
                descriptor,
                last_generation: AtomicU64::new(self.inner.last_generation.load(Ordering::Acquire)),
            }),
        };
        evidence_resources()
            .lock()
            .map_err(|_| {
                RuntimeError::local(
                    RuntimeStatus::InternalError,
                    "KV evidence resource registry is poisoned",
                )
            })?
            .insert(
                (self.inner.session_id.raw(), state_id.raw()),
                Arc::downgrade(&resource.inner),
            );
        Ok((resource, audit))
    }

    /// Re-query post-COW ownership after a child append. The native query is
    /// authoritative for shared-page and destination-owned byte accounting.
    pub(crate) fn fork_query(&self) -> Result<StateForkAuditV1, RuntimeError> {
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let mut info = sys::sllm_state_fork_info_t {
            struct_size: size_of::<sys::sllm_state_fork_info_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            info_version: sys::SLLM_HIP_STATE_FORK_INFO_VERSION,
            mode: 0,
            source_state_identity: 0,
            child_state_identity: 0,
            source_owned_bytes: 0,
            child_owned_bytes: 0,
            copied_bytes: 0,
            shared_bytes: 0,
            published_length: 0,
            page_bytes: 0,
            reserved: [0; 4],
        };
        let status = unsafe {
            sys::sllm_kv_state_fork_query(self.raw_handle()?.as_ptr(), &mut info, &mut error_sink)
        };
        ensure_ok(status, &error_buffer, error_sink.message_length)?;
        let mode = match info.mode {
            sys::SLLM_HIP_STATE_FORK_MODE_SHARED_READ_ONLY_PAGES => {
                StateForkModeV1::SharedReadOnlyPages
            }
            sys::SLLM_HIP_STATE_FORK_MODE_DEVICE_COPY => StateForkModeV1::DeviceCopy,
            _ => {
                return Err(RuntimeError::local(
                    RuntimeStatus::InvalidKvStateDescriptor,
                    "native KV fork query returned an unknown mode",
                ));
            }
        };
        let shared_pages = if info.page_bytes == 0 {
            0
        } else {
            info.shared_bytes
                .saturating_add(info.page_bytes - 1)
                .checked_div(info.page_bytes)
                .unwrap_or(0)
        };
        StateForkAuditV1::new(
            mode,
            info.published_length,
            shared_pages,
            info.copied_bytes,
            info.child_owned_bytes,
        )
        .map_err(|error| {
            RuntimeError::new(
                RuntimeStatus::InvalidKvStateDescriptor,
                format!("native KV fork query audit failed core validation: {error}"),
            )
        })
    }

    pub(crate) fn snapshot(&self) -> Result<KvStateSnapshot, RuntimeError> {
        let view = NativeKvSnapshotOwner::create(self)?;
        let info = view.query()?;
        validate_view_info(
            &info,
            &self.inner.context,
            self.inner.raw,
            self.inner.session_id,
            self.inner.descriptor,
        )?;
        let previous = self.inner.last_generation.load(Ordering::Acquire);
        if info.generation < previous {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidKvStateDescriptor,
                "native KV snapshot generation moved backwards",
            ));
        }
        self.inner
            .last_generation
            .store(info.generation, Ordering::Release);
        let memory_kind = match info.memory_kind {
            sys::SLLM_HIP_KV_MEMORY_KIND_VIRTUAL_CONTIGUOUS => KvMemoryKind::VirtualContiguous,
            sys::SLLM_HIP_KV_MEMORY_KIND_CONTIGUOUS_RESIDENT => KvMemoryKind::ContiguousResident,
            _ => {
                return Err(RuntimeError::local(
                    RuntimeStatus::InvalidKvStateDescriptor,
                    "native KV memory provider is unknown",
                ));
            }
        };
        let retained_start = u64::from(info.reserved[2]) | (u64::from(info.reserved[3]) << 32);
        let retained_length = info.observed_length.saturating_sub(retained_start);
        let physical_memory = if self.inner.descriptor.sliding_window().is_some() {
            sllm_core::KvPhysicalMemorySnapshot::new_with_retention(
                memory_kind,
                self.inner.descriptor.capacity(),
                info.observed_length,
                info.physical_page_bytes,
                info.tokens_per_page,
                info.mapped_token_capacity,
                info.committed_bytes_per_plane,
                retained_start,
                retained_length,
            )
        } else {
            sllm_core::KvPhysicalMemorySnapshot::new_with_kind(
                memory_kind,
                self.inner.descriptor.capacity(),
                info.observed_length,
                info.physical_page_bytes,
                info.tokens_per_page,
                info.mapped_token_capacity,
                info.committed_bytes_per_plane,
            )
        }
        .map_err(|error| {
            RuntimeError::new(
                RuntimeStatus::InvalidKvStateDescriptor,
                format!("native KV physical-memory metadata failed core validation: {error}"),
            )
        })?;
        KvStateSnapshot::new_with_physical_memory(
            self.inner.session_id,
            self.inner.state_id,
            self.inner.descriptor,
            info.observed_length,
            physical_memory,
        )
        .map_err(|error| {
            RuntimeError::new(
                RuntimeStatus::InvalidKvStateDescriptor,
                format!("native KV snapshot failed core validation: {error}"),
            )
        })
    }

    pub(crate) fn rewind_last(
        &self,
        expected_length: u64,
        rewind_length: u64,
    ) -> Result<(), RuntimeError> {
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let status = unsafe {
            sys::sllm_kv_state_rewind_last(
                self.raw_handle()?.as_ptr(),
                expected_length,
                rewind_length,
                &mut error_sink,
            )
        };
        ensure_ok(status, &error_buffer, error_sink.message_length)
    }

    pub(crate) fn export_chunk(
        &self,
        plane: u32,
        byte_offset: u64,
        destination: &mut [u8],
        published_length: u64,
    ) -> Result<(), RuntimeError> {
        if destination.is_empty() {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidArgument,
                "KV export chunk must not be empty",
            ));
        }
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let sliding_window = self.inner.descriptor.sliding_window();
        let mut reserved = [0_u32; 4];
        if sliding_window.is_some() {
            reserved[0] = published_length as u32;
            reserved[1] = (published_length >> 32) as u32;
        }
        let chunk = sys::sllm_state_chunk_t {
            struct_size: size_of::<sys::sllm_state_chunk_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            info_version: if sliding_window.is_some() {
                sys::SLLM_HIP_STATE_IMAGE_SLIDING_VERSION
            } else {
                sys::SLLM_HIP_STATE_FORK_INFO_VERSION
            },
            plane,
            reserved0: sliding_window.unwrap_or(0) as u32,
            reserved1: (sliding_window.unwrap_or(0) >> 32) as u32,
            byte_offset,
            byte_length: destination.len() as u64,
            host_pointer: destination.as_mut_ptr().cast(),
            host_capacity: destination.len() as u64,
            reserved,
        };
        let status = unsafe {
            sys::sllm_kv_state_export(self.raw_handle()?.as_ptr(), &chunk, &mut error_sink)
        };
        ensure_ok(status, &error_buffer, error_sink.message_length)
    }

    pub(crate) fn import_chunk(
        &self,
        plane: u32,
        byte_offset: u64,
        source: &[u8],
        published_length: u64,
    ) -> Result<(), RuntimeError> {
        if source.is_empty() {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidArgument,
                "KV import chunk must not be empty",
            ));
        }
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let sliding_window = self.inner.descriptor.sliding_window();
        let mut reserved = [0_u32; 4];
        if sliding_window.is_some() {
            reserved[0] = published_length as u32;
            reserved[1] = (published_length >> 32) as u32;
        }
        let chunk = sys::sllm_state_chunk_t {
            struct_size: size_of::<sys::sllm_state_chunk_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            info_version: if sliding_window.is_some() {
                sys::SLLM_HIP_STATE_IMAGE_SLIDING_VERSION
            } else {
                sys::SLLM_HIP_STATE_FORK_INFO_VERSION
            },
            plane,
            reserved0: sliding_window.unwrap_or(0) as u32,
            reserved1: (sliding_window.unwrap_or(0) >> 32) as u32,
            byte_offset,
            byte_length: source.len() as u64,
            host_pointer: source.as_ptr().cast_mut().cast(),
            host_capacity: source.len() as u64,
            reserved,
        };
        let status = unsafe {
            sys::sllm_kv_state_import(self.raw_handle()?.as_ptr(), &chunk, &mut error_sink)
        };
        ensure_ok(status, &error_buffer, error_sink.message_length)
    }

    pub(crate) fn image_query(&self) -> Result<sys::sllm_state_image_info_t, RuntimeError> {
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let mut info = sys::sllm_state_image_info_t {
            struct_size: size_of::<sys::sllm_state_image_info_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            info_version: sys::SLLM_HIP_STATE_FORK_INFO_VERSION,
            reserved0: 0,
            session_id: 0,
            layer_id: 0,
            dtype: 0,
            encoding: 0,
            active_slot: 0,
            capacity_tokens: 0,
            published_length: 0,
            generation: 0,
            plane_count: 0,
            reserved: [0; 7],
        };
        let status = unsafe {
            sys::sllm_kv_state_image_query(self.raw_handle()?.as_ptr(), &mut info, &mut error_sink)
        };
        ensure_ok(status, &error_buffer, error_sink.message_length)?;
        Ok(info)
    }

    pub(crate) fn image_plane_size(&self, plane: u32) -> Result<u64, RuntimeError> {
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let mut size_bytes = 0_u64;
        let status = unsafe {
            sys::sllm_kv_state_image_plane_size(
                self.raw_handle()?.as_ptr(),
                plane,
                &mut size_bytes,
                &mut error_sink,
            )
        };
        ensure_ok(status, &error_buffer, error_sink.message_length)?;
        Ok(size_bytes)
    }

    pub(crate) fn import_finalize(
        &self,
        info: &sys::sllm_state_image_info_t,
    ) -> Result<(), RuntimeError> {
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let status = unsafe {
            sys::sllm_kv_state_import_finalize(self.raw_handle()?.as_ptr(), info, &mut error_sink)
        };
        ensure_ok(status, &error_buffer, error_sink.message_length)
    }

    pub(crate) fn readback(
        &self,
        plane: u32,
        byte_offset: u64,
        destination: &mut [u8],
    ) -> Result<(), RuntimeError> {
        let view = NativeKvSnapshotOwner::create(self)?;
        let info = view.query()?;
        validate_view_info(
            &info,
            &self.inner.context,
            self.inner.raw,
            self.inner.session_id,
            self.inner.descriptor,
        )?;
        view.readback(plane, byte_offset, destination)
    }

    pub(crate) fn append(
        &self,
        queue: &Queue,
        key: &TensorBinding,
        value: &TensorBinding,
        request: KvStateAppendRequest,
    ) -> Result<(KvAppendCompletion, KvAppendEvidence), RuntimeError> {
        if request.state_id() != self.inner.state_id
            || request.descriptor() != self.inner.descriptor
            || request.start_position() != request.expected_length()
            || request.end_position() > self.inner.descriptor.capacity()
        {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidKvAppendDescriptor,
                "Rust KV append metadata is not the admitted state request",
            ));
        }
        let key_raw = key.raw()?;
        let value_raw = value.raw()?;
        validate_append_binding(key, self.inner.descriptor)?;
        validate_append_binding(value, self.inner.descriptor)?;
        let descriptor = sys::sllm_kv_append_desc_t {
            struct_size: size_of::<sys::sllm_kv_append_desc_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            append_version: sys::SLLM_HIP_KV_STATE_VERSION,
            reserved0: 0,
            expected_length: request.expected_length(),
            start_position: request.start_position(),
            key_input: key_raw,
            value_input: value_raw,
            reserved: [0; 4],
        };
        let state_raw = self.raw_handle()?;
        let queue_raw = queue.raw_handle()?;
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let mut raw_completion = std::ptr::null_mut();
        let mut append_info = empty_append_info();
        let raw = unsafe {
            sys::sllm_kv_state_append(
                state_raw.as_ptr(),
                queue_raw.as_ptr(),
                &descriptor,
                &mut raw_completion,
                &mut append_info,
                &mut error_sink,
            )
        };
        ensure_ok(raw, &error_buffer, error_sink.message_length)?;
        let raw_completion = NonNull::new(raw_completion).ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::InternalError,
                "native KV append returned a null completion on success",
            )
        })?;
        let completion = KvAppendCompletion {
            raw: Some(raw_completion.as_ptr() as usize),
            context: self.inner.context.clone(),
            queue: queue.clone(),
            key: key.buffer().clone(),
            value: value.buffer().clone(),
            state: self.clone(),
            terminal: false,
            canceled: false,
        };
        let evidence = match validate_append_info(
            &append_info,
            &self.inner.context,
            request,
            self.inner.descriptor,
        ) {
            Ok(evidence) => evidence,
            Err(error) => {
                drop(completion);
                return Err(error);
            }
        };
        Ok((completion, evidence))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn causal_attention(
        &self,
        queue: &Queue,
        query: &TensorBinding,
        output: &TensorBinding,
        start_position: u64,
        expected_kv_length: u64,
        sliding_window: Option<u64>,
        score_scale: Option<f32>,
    ) -> Result<(CausalAttentionCompletion, CausalAttentionEvidence), RuntimeError> {
        if sliding_window != self.inner.descriptor.sliding_window() {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidCausalAttentionDescriptor,
                "causal attention sliding window differs from the KV state descriptor",
            ));
        }
        validate_causal_attention_binding(query, self.inner.descriptor)?;
        validate_causal_attention_binding(output, self.inner.descriptor)?;
        let mut reserved = [0_u32; 4];
        if let Some(window) = sliding_window {
            reserved[0] = window as u32;
            reserved[1] = (window >> 32) as u32;
        }
        if let Some(scale) = score_scale {
            if !scale.is_finite() || scale <= 0.0 {
                return Err(RuntimeError::local(
                    RuntimeStatus::InvalidCausalAttentionDescriptor,
                    "causal attention score scale must be finite and positive",
                ));
            }
            reserved[2] = scale.to_bits();
        }
        let descriptor = sys::sllm_causal_attention_desc_t {
            struct_size: size_of::<sys::sllm_causal_attention_desc_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            op_version: if score_scale.is_some() {
                sys::SLLM_HIP_CAUSAL_ATTENTION_EXPLICIT_SCALE_VERSION
            } else if sliding_window.is_some() {
                sys::SLLM_HIP_CAUSAL_ATTENTION_SLIDING_VERSION
            } else {
                sys::SLLM_HIP_CAUSAL_ATTENTION_VERSION
            },
            reserved0: 0,
            start_position,
            expected_kv_length,
            kv_state: self.raw_handle()?.as_ptr(),
            query: query.raw()?,
            output: output.raw()?,
            reserved,
        };
        let mut dispatch_info = empty_causal_attention_info();
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let mut raw_completion = std::ptr::null_mut();
        let raw = unsafe {
            sys::sllm_causal_attention_execute(
                self.inner.context.raw_handle()?.as_ptr(),
                queue.raw_handle()?.as_ptr(),
                &descriptor,
                &mut raw_completion,
                &mut dispatch_info,
                &mut error_sink,
            )
        };
        ensure_ok(raw, &error_buffer, error_sink.message_length)?;
        let raw_completion = NonNull::new(raw_completion).ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::InternalError,
                "native causal attention returned a null completion on success",
            )
        })?;
        let completion = CausalAttentionCompletion {
            raw: Some(raw_completion.as_ptr() as usize),
            context: self.inner.context.clone(),
            queue: queue.clone(),
            _query: query.buffer().clone(),
            _output: output.buffer().clone(),
            state: self.clone(),
            terminal: false,
        };
        let evidence = match validate_causal_attention_info(
            &dispatch_info,
            &self.inner.context,
            start_position,
            expected_kv_length,
            self.inner.descriptor,
            u32::try_from(query.view().shape()[1]).map_err(|_| {
                RuntimeError::local(
                    RuntimeStatus::InvalidCausalAttentionDescriptor,
                    "causal attention query head count does not fit u32",
                )
            })?,
            sliding_window,
            score_scale,
        ) {
            Ok(evidence) => evidence,
            Err(error) => {
                drop(completion);
                return Err(error);
            }
        };
        Ok((completion, evidence))
    }

    /// Enqueues causal attention behind an already-submitted append on the
    /// same queue. The append completion remains a separate owner; callers
    /// must retain it until this attention completion reaches a terminal
    /// state because native claim/release accounting is request-local.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn causal_attention_after_kv_append(
        &self,
        queue: &Queue,
        append: &KvAppendCompletion,
        query: &TensorBinding,
        output: &TensorBinding,
        start_position: u64,
        expected_kv_length: u64,
        sliding_window: Option<u64>,
        score_scale: Option<f32>,
    ) -> Result<(CausalAttentionCompletion, CausalAttentionEvidence), RuntimeError> {
        if append.state.inner.raw != self.inner.raw
            || append.context.raw_handle()?.as_ptr() != self.inner.context.raw_handle()?.as_ptr()
            || append.queue.raw_handle()?.as_ptr() != queue.raw_handle()?.as_ptr()
        {
            return Err(RuntimeError::local(
                RuntimeStatus::Busy,
                "KV append dependency does not belong to this state and queue",
            ));
        }
        if sliding_window != self.inner.descriptor.sliding_window() {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidCausalAttentionDescriptor,
                "causal attention sliding window differs from the KV state descriptor",
            ));
        }
        validate_causal_attention_binding(query, self.inner.descriptor)?;
        validate_causal_attention_binding(output, self.inner.descriptor)?;
        let mut reserved = [0_u32; 4];
        if let Some(window) = sliding_window {
            reserved[0] = window as u32;
            reserved[1] = (window >> 32) as u32;
        }
        if let Some(scale) = score_scale {
            if !scale.is_finite() || scale <= 0.0 {
                return Err(RuntimeError::local(
                    RuntimeStatus::InvalidCausalAttentionDescriptor,
                    "causal attention score scale must be finite and positive",
                ));
            }
            reserved[2] = scale.to_bits();
        }
        let descriptor = sys::sllm_causal_attention_desc_t {
            struct_size: size_of::<sys::sllm_causal_attention_desc_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            op_version: if score_scale.is_some() {
                sys::SLLM_HIP_CAUSAL_ATTENTION_EXPLICIT_SCALE_VERSION
            } else if sliding_window.is_some() {
                sys::SLLM_HIP_CAUSAL_ATTENTION_SLIDING_VERSION
            } else {
                sys::SLLM_HIP_CAUSAL_ATTENTION_VERSION
            },
            reserved0: 0,
            start_position,
            expected_kv_length,
            kv_state: self.raw_handle()?.as_ptr(),
            query: query.raw()?,
            output: output.raw()?,
            reserved,
        };
        let mut dispatch_info = empty_causal_attention_info();
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let mut raw_completion = std::ptr::null_mut();
        let raw = unsafe {
            sys::sllm_causal_attention_execute_after_kv_append(
                self.inner.context.raw_handle()?.as_ptr(),
                queue.raw_handle()?.as_ptr(),
                append.raw_handle()?.as_ptr(),
                &descriptor,
                &mut raw_completion,
                &mut dispatch_info,
                &mut error_sink,
            )
        };
        ensure_ok(raw, &error_buffer, error_sink.message_length)?;
        let raw_completion = NonNull::new(raw_completion).ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::InternalError,
                "native chained causal attention returned a null completion on success",
            )
        })?;
        let completion = CausalAttentionCompletion {
            raw: Some(raw_completion.as_ptr() as usize),
            context: self.inner.context.clone(),
            queue: queue.clone(),
            _query: query.buffer().clone(),
            _output: output.buffer().clone(),
            state: self.clone(),
            terminal: false,
        };
        let evidence = match validate_causal_attention_info(
            &dispatch_info,
            &self.inner.context,
            start_position,
            expected_kv_length,
            self.inner.descriptor,
            u32::try_from(query.view().shape()[1]).map_err(|_| {
                RuntimeError::local(
                    RuntimeStatus::InvalidCausalAttentionDescriptor,
                    "causal attention query head count does not fit u32",
                )
            })?,
            sliding_window,
            score_scale,
        ) {
            Ok(evidence) => evidence,
            Err(error) => {
                drop(completion);
                return Err(error);
            }
        };
        Ok((completion, evidence))
    }
}

pub(crate) fn resource_for_evidence(session_id: u64, state_id: u64) -> Option<KvStateResource> {
    evidence_resources()
        .lock()
        .ok()?
        .get(&(session_id, state_id))
        .and_then(Weak::upgrade)
        .map(|inner| KvStateResource { inner })
}

/// Direct native state owner. It is intentionally not Send or Sync.
#[derive(Clone)]
pub(crate) struct NativeKvStateOwner {
    _resource: KvStateResource,
    _not_send_sync: PhantomData<Rc<()>>,
}

impl NativeKvStateOwner {
    pub(crate) fn new(resource: KvStateResource) -> Self {
        Self {
            _resource: resource,
            _not_send_sync: PhantomData,
        }
    }
}

struct NativeKvSnapshotOwner {
    raw: Option<NonNull<sys::sllm_kv_view_t>>,
    context: Context,
    _state: NativeKvStateOwner,
    _not_send_sync: PhantomData<Rc<()>>,
}

impl NativeKvSnapshotOwner {
    fn create(state: &KvStateResource) -> Result<Self, RuntimeError> {
        let state_raw = state.raw_handle()?;
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let mut raw_view = std::ptr::null_mut();
        let raw = unsafe {
            sys::sllm_kv_state_snapshot(state_raw.as_ptr(), &mut raw_view, &mut error_sink)
        };
        ensure_ok(raw, &error_buffer, error_sink.message_length)?;
        let raw_view = NonNull::new(raw_view).ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::InternalError,
                "native KV snapshot returned a null view on success",
            )
        })?;
        Ok(Self {
            raw: Some(raw_view),
            context: state.inner.context.clone(),
            _state: NativeKvStateOwner::new(state.clone()),
            _not_send_sync: PhantomData,
        })
    }

    fn query(&self) -> Result<sys::sllm_kv_view_info_t, RuntimeError> {
        let raw = self.raw.ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::InvalidHandle,
                "KV snapshot view was released",
            )
        })?;
        let mut info = empty_view_info();
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let status = unsafe { sys::sllm_kv_view_query(raw.as_ptr(), &mut info, &mut error_sink) };
        ensure_ok(status, &error_buffer, error_sink.message_length)?;
        Ok(info)
    }

    fn readback(
        &self,
        plane: u32,
        byte_offset: u64,
        destination: &mut [u8],
    ) -> Result<(), RuntimeError> {
        if destination.is_empty() {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidArgument,
                "KV evidence readback destination is empty",
            ));
        }
        let raw = self.raw.ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::InvalidHandle,
                "KV snapshot view was released",
            )
        })?;
        let byte_length = u64::try_from(destination.len()).map_err(|_| {
            RuntimeError::local(
                RuntimeStatus::MetadataOverflow,
                "KV evidence readback destination is too large",
            )
        })?;
        let request = sys::evidence::sllm_hip_kv_readback_request_t {
            struct_size: size_of::<sys::evidence::sllm_hip_kv_readback_request_t>() as u32,
            abi_version: sys::evidence::SLLM_HIP_KV_EVIDENCE_ABI_VERSION,
            view: raw.as_ptr(),
            plane,
            reserved0: 0,
            byte_offset,
            byte_length,
            host_capacity: byte_length,
            host_output: destination.as_mut_ptr(),
            reserved: [0; 4],
        };
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let status = unsafe { sys::evidence::sllm_hip_kv_view_readback(&request, &mut error_sink) };
        ensure_ok(status, &error_buffer, error_sink.message_length)
    }
}

impl Drop for NativeKvSnapshotOwner {
    fn drop(&mut self) {
        let Some(raw) = self.raw.take() else {
            return;
        };
        let (status, remaining) = release_kv_view_once(raw);
        if let Some(remaining) = remaining {
            enqueue_kv_view_cleanup(remaining, self.context.clone(), status);
        }
    }
}

/// Metadata returned by one accepted native append.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KvAppendEvidence {
    pub dispatch_id: u64,
    pub dispatch_count: u32,
    pub kernel_id: u32,
    pub workgroup_size_x: u32,
    pub grid_size_x: u32,
    pub start_position: u64,
    pub token_count: u64,
    pub end_position: u64,
    pub commit_allowed: bool,
    pub fallback_allowed: bool,
    pub fallback_used: bool,
    pub kernel_symbol: String,
    pub device_symbol: String,
    pub target: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CausalAttentionEvidence {
    pub dispatch_id: u64,
    pub dispatch_count: u32,
    pub kernel_id: u32,
    pub workgroup_size_x: u32,
    pub grid_size_x: u32,
    pub query_count: u64,
    pub start_position: u64,
    pub committed_kv_length: u64,
    pub sliding_window: u64,
    pub retained_start: u64,
    pub score_scale_bits: u32,
    pub explicit_score_scale: bool,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub scale_denominator: u32,
    pub fallback_allowed: bool,
    pub fallback_used: bool,
    pub kernel_symbol: String,
    pub device_symbol: String,
    pub target: String,
}

/// Sendable append completion retaining every native dependency.
pub(crate) struct KvAppendCompletion {
    raw: Option<usize>,
    context: Context,
    queue: Queue,
    key: Buffer,
    value: Buffer,
    state: KvStateResource,
    terminal: bool,
    canceled: bool,
}

pub(crate) struct CausalAttentionCompletion {
    raw: Option<usize>,
    context: Context,
    queue: Queue,
    _query: Buffer,
    _output: Buffer,
    state: KvStateResource,
    terminal: bool,
}

impl CausalAttentionCompletion {
    pub(crate) fn query(&mut self) -> Result<CompletionState, RuntimeError> {
        self.call_completion(None)
    }

    pub(crate) fn wait(&mut self, timeout: Duration) -> Result<CompletionState, RuntimeError> {
        self.call_completion(Some(timeout))
    }

    pub(crate) fn finalize_after_token(
        &mut self,
        fence_token: u64,
    ) -> Result<CompletionState, RuntimeError> {
        let state = finalize_completion_after(
            self.raw_handle()?,
            completion_from_opaque_token(fence_token)?,
        )?;
        self.terminal = state != CompletionState::Pending;
        Ok(state)
    }

    pub(crate) fn kernel_elapsed_ns(&mut self) -> Result<u64, RuntimeError> {
        let raw = self.raw_handle()?;
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let mut timing = sys::sllm_completion_timing_t {
            struct_size: size_of::<sys::sllm_completion_timing_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            valid: 0,
            reserved0: 0,
            elapsed_ns: 0,
            reserved: [0; 4],
        };
        let status =
            unsafe { sys::sllm_completion_timing(raw.as_ptr(), &mut timing, &mut error_sink) };
        ensure_ok(status, &error_buffer, error_sink.message_length)?;
        if timing.valid != 1 || timing.elapsed_ns == 0 {
            return Err(RuntimeError::local(
                RuntimeStatus::HipRuntimeError,
                "causal attention completion timing was not positive",
            ));
        }
        Ok(timing.elapsed_ns)
    }

    fn raw_handle(&self) -> Result<NonNull<sys::sllm_completion_t>, RuntimeError> {
        let raw = self.raw.ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::InvalidHandle,
                "causal attention completion was already released",
            )
        })?;
        NonNull::new(raw as *mut sys::sllm_completion_t).ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::InvalidHandle,
                "causal attention completion had a null opaque handle",
            )
        })
    }

    fn call_completion(
        &mut self,
        timeout: Option<Duration>,
    ) -> Result<CompletionState, RuntimeError> {
        let raw = self.raw_handle()?;
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let mut result = completion_result();
        let status = unsafe {
            match timeout {
                Some(timeout) => sys::sllm_completion_wait(
                    raw.as_ptr(),
                    timeout_millis(timeout),
                    &mut result,
                    &mut error_sink,
                ),
                None => sys::sllm_completion_query(raw.as_ptr(), &mut result, &mut error_sink),
            }
        };
        let state = completion_state(result.state)?;
        if state != CompletionState::Pending {
            self.terminal = true;
        }
        let status = RuntimeStatus::from_raw(status);
        if status == RuntimeStatus::Ok {
            return Ok(state);
        }
        if state == CompletionState::Pending
            && matches!(status, RuntimeStatus::Pending | RuntimeStatus::Timeout)
        {
            return Err(result_error(
                status.raw(),
                &error_buffer,
                error_sink.message_length,
            ));
        }
        Err(result_error(
            status.raw(),
            &error_buffer,
            error_sink.message_length,
        ))
    }
}

impl Drop for CausalAttentionCompletion {
    fn drop(&mut self) {
        let Some(raw_value) = self.raw.take() else {
            return;
        };
        let Some(raw) = NonNull::new(raw_value as *mut sys::sllm_completion_t) else {
            return;
        };
        let (status, remaining) = release_causal_completion_once(raw);
        if let Some(remaining) = remaining {
            enqueue_causal_completion_cleanup(
                remaining,
                self.context.clone(),
                self.queue.clone(),
                self._query.clone(),
                self._output.clone(),
                self.state.clone(),
                status,
            );
        }
    }
}

impl KvAppendCompletion {
    pub(crate) fn query(&mut self) -> Result<CompletionState, RuntimeError> {
        self.call_completion(None)
    }

    pub(crate) fn wait(&mut self, timeout: Duration) -> Result<CompletionState, RuntimeError> {
        self.call_completion(Some(timeout))
    }

    pub(crate) fn finalize_after_token(
        &mut self,
        fence_token: u64,
    ) -> Result<CompletionState, RuntimeError> {
        let state = finalize_completion_after(
            self.raw_handle()?,
            completion_from_opaque_token(fence_token)?,
        )?;
        self.terminal = state != CompletionState::Pending;
        Ok(state)
    }

    pub(crate) fn cancel(&mut self) -> Result<(), RuntimeError> {
        if self.terminal || self.canceled {
            return Ok(());
        }
        let raw = self.raw_handle()?;
        let state = self.state.raw_handle()?;
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let status = unsafe {
            sys::sllm_kv_state_append_cancel(state.as_ptr(), raw.as_ptr(), &mut error_sink)
        };
        ensure_ok(status, &error_buffer, error_sink.message_length)?;
        self.canceled = true;
        Ok(())
    }

    fn raw_handle(&self) -> Result<NonNull<sys::sllm_completion_t>, RuntimeError> {
        let raw = self.raw.ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::InvalidHandle,
                "KV append completion was already released",
            )
        })?;
        NonNull::new(raw as *mut sys::sllm_completion_t).ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::InvalidHandle,
                "KV append completion had a null opaque handle",
            )
        })
    }

    fn call_completion(
        &mut self,
        timeout: Option<Duration>,
    ) -> Result<CompletionState, RuntimeError> {
        let raw = self.raw_handle()?;
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let mut result = completion_result();
        let status = unsafe {
            match timeout {
                Some(timeout) => sys::sllm_completion_wait(
                    raw.as_ptr(),
                    timeout_millis(timeout),
                    &mut result,
                    &mut error_sink,
                ),
                None => sys::sllm_completion_query(raw.as_ptr(), &mut result, &mut error_sink),
            }
        };
        let state = completion_state(result.state)?;
        if state != CompletionState::Pending {
            self.terminal = true;
        }
        let status = RuntimeStatus::from_raw(status);
        if status == RuntimeStatus::Ok {
            return Ok(state);
        }
        if state == CompletionState::Pending
            && matches!(status, RuntimeStatus::Pending | RuntimeStatus::Timeout)
        {
            return Err(result_error(
                status.raw(),
                &error_buffer,
                error_sink.message_length,
            ));
        }
        Err(result_error(
            status.raw(),
            &error_buffer,
            error_sink.message_length,
        ))
    }
}

impl Drop for KvAppendCompletion {
    fn drop(&mut self) {
        if !self.terminal && !self.canceled {
            let _ = self.cancel();
        }
        let Some(raw_value) = self.raw.take() else {
            return;
        };
        let Some(raw) = NonNull::new(raw_value as *mut sys::sllm_completion_t) else {
            return;
        };
        let (status, remaining) = release_kv_completion_once(raw);
        if let Some(remaining) = remaining {
            enqueue_kv_completion_cleanup(
                remaining,
                self.context.clone(),
                self.queue.clone(),
                self.key.clone(),
                self.value.clone(),
                self.state.clone(),
                status,
            );
        }
    }
}

fn empty_view_info() -> sys::sllm_kv_view_info_t {
    sys::sllm_kv_view_info_t {
        struct_size: size_of::<sys::sllm_kv_view_info_t>() as u32,
        abi_version: sys::SLLM_HIP_ABI_VERSION,
        info_version: sys::SLLM_HIP_KV_VIEW_INFO_VERSION,
        reserved0: 0,
        session_id: 0,
        layer_id: 0,
        dtype: 0,
        encoding: 0,
        head_count: 0,
        head_dim: 0,
        memory_kind: 0,
        layout: 0,
        reserved1: 0,
        capacity_tokens: 0,
        observed_length: 0,
        generation: 0,
        physical_page_bytes: 0,
        tokens_per_page: 0,
        mapped_token_capacity: 0,
        committed_bytes_per_plane: 0,
        context_identity: 0,
        state_identity: 0,
        k_stride_elements: [0; 3],
        v_stride_elements: [0; 3],
        reserved: [0; 4],
    }
}

fn empty_append_info() -> sys::sllm_kv_append_info_t {
    sys::sllm_kv_append_info_t {
        struct_size: size_of::<sys::sllm_kv_append_info_t>() as u32,
        abi_version: sys::SLLM_HIP_ABI_VERSION,
        info_version: sys::SLLM_HIP_KV_APPEND_INFO_VERSION,
        backend: 0,
        dispatch_id: 0,
        dispatch_count: 0,
        kernel_id: 0,
        workgroup_size_x: 0,
        grid_size_x: 0,
        start_position: 0,
        token_count: 0,
        end_position: 0,
        commit_allowed: 0,
        fallback_allowed: 0,
        fallback_used: 0,
        reserved0: 0,
        kernel_symbol: [0; 64],
        device_symbol: [0; 64],
        gcn_arch_name: [0; 64],
        reserved: [0; 8],
    }
}

fn empty_causal_attention_info() -> sys::sllm_causal_attention_dispatch_info_t {
    sys::sllm_causal_attention_dispatch_info_t {
        struct_size: size_of::<sys::sllm_causal_attention_dispatch_info_t>() as u32,
        abi_version: sys::SLLM_HIP_ABI_VERSION,
        info_version: sys::SLLM_HIP_CAUSAL_ATTENTION_DISPATCH_INFO_VERSION,
        backend: 0,
        dispatch_id: 0,
        dispatch_count: 0,
        kernel_id: 0,
        workgroup_size_x: 0,
        grid_size_x: 0,
        query_count: 0,
        start_position: 0,
        committed_kv_length: 0,
        q_heads: 0,
        kv_heads: 0,
        head_dim: 0,
        scale_denominator: 0,
        fallback_allowed: 0,
        fallback_used: 0,
        kernel_symbol: [0; 64],
        device_symbol: [0; 64],
        gcn_arch_name: [0; 64],
        reserved: [0; 8],
    }
}

fn validate_view_info(
    info: &sys::sllm_kv_view_info_t,
    context: &Context,
    raw_state: usize,
    session_id: ExecutionSessionId,
    descriptor: KvStateDescriptor,
) -> Result<(), RuntimeError> {
    let layout = descriptor.layout();
    let logical_head_dim = layout.head_dim() as u64;
    let physical_head_dim = match descriptor.cache_encoding() {
        KvCacheEncoding::Fp8E4M3Block16 | KvCacheEncoding::Fp8E5M2Block16 => {
            logical_head_dim.div_ceil(16) * 16
        }
        KvCacheEncoding::Mxfp8E4 | KvCacheEncoding::Mxfp8E5 => logical_head_dim.div_ceil(32) * 32,
        _ => logical_head_dim,
    };
    let token_stride = (layout.heads() as u64)
        .checked_mul(physical_head_dim)
        .ok_or_else(|| {
            RuntimeError::local(RuntimeStatus::MetadataOverflow, "KV stride overflow")
        })?;
    let expected_dtype = native_kv_storage(descriptor, context.expected_target())?.0;
    let expected_encoding = match descriptor.cache_encoding() {
        KvCacheEncoding::Fp16 => sys::SLLM_TENSOR_ENCODING_UNQUANTIZED,
        KvCacheEncoding::Fp8E4M3Fn | KvCacheEncoding::Fp8E4M3FnStatic => {
            sys::SLLM_TENSOR_ENCODING_FP8_OUTER_F32
        }
        KvCacheEncoding::Nvfp4 => sys::SLLM_TENSOR_ENCODING_NVFP4_BLOCK16_E4M3FN_F32,
        KvCacheEncoding::Fp8E4M3Block16 | KvCacheEncoding::Fp8E5M2Block16 => {
            sys::SLLM_TENSOR_ENCODING_FP8_BLOCK16_E8M0
        }
        KvCacheEncoding::Mxfp8E4 | KvCacheEncoding::Mxfp8E5 => {
            sys::SLLM_TENSOR_ENCODING_MXFP8_BLOCK32_E8M0
        }
    };
    let sliding_window = descriptor.sliding_window();
    let expected_info_version = if sliding_window.is_some() {
        sys::SLLM_HIP_KV_VIEW_INFO_SLIDING_VERSION
    } else {
        sys::SLLM_HIP_KV_VIEW_INFO_VERSION
    };
    let retained_start = info
        .observed_length
        .saturating_sub(sliding_window.unwrap_or(info.observed_length));
    let expected_reserved = if let Some(window) = sliding_window {
        [
            window as u32,
            (window >> 32) as u32,
            retained_start as u32,
            (retained_start >> 32) as u32,
        ]
    } else {
        [0; 4]
    };
    let physical_length_valid = if let Some(window) = sliding_window {
        info.mapped_token_capacity <= window.saturating_add(1)
            && info.observed_length.saturating_sub(retained_start) <= info.mapped_token_capacity
    } else {
        info.observed_length <= info.mapped_token_capacity
    };
    if info.struct_size != size_of::<sys::sllm_kv_view_info_t>() as u32
        || info.abi_version != sys::SLLM_HIP_ABI_VERSION
        || info.info_version != expected_info_version
        || info.session_id != session_id.raw()
        || info.layer_id != descriptor.layer_id()
        || info.dtype != expected_dtype
        || info.encoding != expected_encoding
        || info.head_count != layout.heads() as u32
        || info.head_dim != layout.head_dim() as u32
        || !matches!(
            info.memory_kind,
            sys::SLLM_HIP_KV_MEMORY_KIND_VIRTUAL_CONTIGUOUS
                | sys::SLLM_HIP_KV_MEMORY_KIND_CONTIGUOUS_RESIDENT
        )
        || info.layout != sys::SLLM_HIP_KV_LAYOUT_TOKEN_MAJOR
        || info.capacity_tokens != descriptor.capacity()
        || info.context_identity != context.raw_handle()?.as_ptr() as usize as u64
        || info.state_identity != raw_state as u64
        || info.k_stride_elements != [token_stride, physical_head_dim, 1]
        || info.v_stride_elements != [token_stride, physical_head_dim, 1]
        || info.physical_page_bytes == 0
        || info.tokens_per_page == 0
        || info.mapped_token_capacity > descriptor.capacity()
        || !physical_length_valid
        || info.committed_bytes_per_plane % info.physical_page_bytes != 0
        || info.reserved0 != 0
        || info.reserved1 != 0
        || info.reserved != expected_reserved
    {
        return Err(RuntimeError::local(
            RuntimeStatus::InvalidKvStateDescriptor,
            "native KV snapshot metadata differs from the descriptor layout",
        ));
    }
    if info.observed_length > descriptor.capacity() {
        return Err(RuntimeError::local(
            RuntimeStatus::KvCapacityExceeded,
            "native KV snapshot length exceeds capacity",
        ));
    }
    Ok(())
}

fn validate_append_binding(
    binding: &TensorBinding,
    descriptor: KvStateDescriptor,
) -> Result<(), RuntimeError> {
    let view = binding.view();
    let layout = descriptor.layout();
    if view.dtype() != DType::Bf16
        || view.encoding() != Encoding::Unquantized
        || view.shape().len() != 3
        || view.shape()[1] != layout.heads()
        || view.shape()[2] != layout.head_dim()
        || view.strides() != [layout.heads() * layout.head_dim(), layout.head_dim(), 1]
    {
        return Err(RuntimeError::local(
            RuntimeStatus::InvalidKvAppendDescriptor,
            "KV append input must match the contiguous descriptor layout",
        ));
    }
    Ok(())
}

fn validate_causal_attention_binding(
    binding: &TensorBinding,
    descriptor: KvStateDescriptor,
) -> Result<(), RuntimeError> {
    let view = binding.view();
    let layout = descriptor.layout();
    let q_heads = view.shape().get(1).copied().unwrap_or(0);
    if view.dtype() != DType::Bf16
        || view.encoding() != Encoding::Unquantized
        || view.shape().len() != 3
        || q_heads == 0
        || q_heads % layout.heads() != 0
        || !matches!(q_heads / layout.heads(), 2 | 4 | 6 | 8 | 16)
        || view.shape()[2] != layout.head_dim()
        || view.strides() != [q_heads * layout.head_dim(), layout.head_dim(), 1]
        || view.shape()[0] == 0
    {
        return Err(RuntimeError::local(
            RuntimeStatus::InvalidCausalAttentionDescriptor,
            "causal attention Q/output must match the contiguous KV descriptor layout",
        ));
    }
    Ok(())
}

fn validate_append_info(
    info: &sys::sllm_kv_append_info_t,
    context: &Context,
    request: KvStateAppendRequest,
    descriptor: KvStateDescriptor,
) -> Result<KvAppendEvidence, RuntimeError> {
    let observed_target = c_string(&info.gcn_arch_name);
    let target = logical_gcn_arch_name(&observed_target).to_owned();
    let expected_target = context.expected_target();
    let (expected_kernel_id, expected_kernel, expected_device) = match descriptor.cache_encoding() {
        KvCacheEncoding::Fp16 => (
            sys::SLLM_HIP_KV_KERNEL_ID_BF16_TO_F16_TOKEN_MAJOR_V2,
            KERNEL_SYMBOL,
            DEVICE_SYMBOL,
        ),
        KvCacheEncoding::Fp8E4M3Fn => (
            sys::SLLM_HIP_KV_KERNEL_ID_BF16_TO_FP8_TOKEN_MAJOR_V1,
            "kv_state.bf16_to_fp8_token_major.v1",
            "sllm_kv_state_bf16_to_fp8_token_major_v1",
        ),
        KvCacheEncoding::Fp8E4M3FnStatic => (
            sys::SLLM_HIP_KV_KERNEL_ID_BF16_TO_FP8_STATIC_TOKEN_MAJOR_V1,
            "kv_state.bf16_to_fp8_static_token_major.v1",
            "sllm_kv_state_bf16_to_fp8_token_major_v1",
        ),
        KvCacheEncoding::Nvfp4 => (
            sys::SLLM_HIP_KV_KERNEL_ID_BF16_TO_NVFP4_TOKEN_MAJOR_V1,
            "kv_state.bf16_to_nvfp4_token_major.v1",
            "sllm_kv_state_bf16_to_nvfp4_token_major_v1",
        ),
        KvCacheEncoding::Fp8E4M3Block16 => (
            sys::SLLM_HIP_KV_KERNEL_ID_BF16_TO_FP8_E4_BLOCK16_TOKEN_MAJOR_V2,
            "kv_state.bf16_to_fp8_e4_block16_token_major.v2",
            "sllm_kv_state_bf16_to_fp8_e4_block16_token_major_v2",
        ),
        KvCacheEncoding::Fp8E5M2Block16 => (
            sys::SLLM_HIP_KV_KERNEL_ID_BF16_TO_FP8_E5_BLOCK16_TOKEN_MAJOR_V2,
            "kv_state.bf16_to_fp8_e5_block16_token_major.v2",
            "sllm_kv_state_bf16_to_fp8_e5_block16_token_major_v2",
        ),
        KvCacheEncoding::Mxfp8E4 => (
            sys::SLLM_HIP_KV_KERNEL_ID_BF16_TO_MXFP8_E4_TOKEN_MAJOR_V1,
            "kv_state.bf16_to_mxfp8_e4_token_major.v1",
            "sllm_kv_state_bf16_to_mxfp8_e4_token_major_v1",
        ),
        KvCacheEncoding::Mxfp8E5 => (
            sys::SLLM_HIP_KV_KERNEL_ID_BF16_TO_MXFP8_E5_TOKEN_MAJOR_V1,
            "kv_state.bf16_to_mxfp8_e5_token_major.v1",
            "sllm_kv_state_bf16_to_mxfp8_e5_token_major_v1",
        ),
    };
    let expected_rows = request
        .token_count()
        .checked_mul(descriptor.layout().heads() as u64)
        .ok_or_else(|| RuntimeError::local(RuntimeStatus::MetadataOverflow, "KV grid overflow"))?;
    let expected_grid = if descriptor.cache_encoding() == KvCacheEncoding::Fp16 {
        expected_rows
            .checked_mul(descriptor.layout().head_dim() as u64)
            .and_then(|elements| {
                elements.checked_add(u64::from(sys::SLLM_HIP_KV_WORKGROUP_SIZE) - 1)
            })
            .map(|elements| elements / u64::from(sys::SLLM_HIP_KV_WORKGROUP_SIZE))
            .and_then(|value| u32::try_from(value).ok())
    } else {
        u32::try_from(expected_rows).ok()
    };
    if info.struct_size != size_of::<sys::sllm_kv_append_info_t>() as u32
        || info.abi_version != sys::SLLM_HIP_ABI_VERSION
        || info.info_version != sys::SLLM_HIP_KV_APPEND_INFO_VERSION
        || info.backend != sys::SLLM_BACKEND_HIP
        || info.dispatch_id == 0
        || info.dispatch_count != 1
        || info.kernel_id != expected_kernel_id
        || info.workgroup_size_x != sys::SLLM_HIP_KV_WORKGROUP_SIZE
        || Some(info.grid_size_x) != expected_grid
        || info.start_position != request.start_position()
        || info.token_count != request.token_count()
        || info.end_position != request.end_position()
        || info.commit_allowed != 1
        || info.fallback_allowed != 0
        || info.fallback_used != 0
        || c_string(&info.kernel_symbol) != expected_kernel
        || c_string(&info.device_symbol) != expected_device
        || info.reserved0 != 0
        || info.reserved != [0; 8]
        || info.end_position > descriptor.capacity()
        || expected_target.is_some_and(|expected| !gcn_arch_matches(expected, &observed_target))
    {
        return Err(RuntimeError::local(
            RuntimeStatus::InvalidKvAppendDescriptor,
            "native KV append metadata failed exact-target/no-fallback validation",
        ));
    }
    Ok(KvAppendEvidence {
        dispatch_id: info.dispatch_id,
        dispatch_count: info.dispatch_count,
        kernel_id: info.kernel_id,
        workgroup_size_x: info.workgroup_size_x,
        grid_size_x: info.grid_size_x,
        start_position: info.start_position,
        token_count: info.token_count,
        end_position: info.end_position,
        commit_allowed: info.commit_allowed == 1,
        fallback_allowed: info.fallback_allowed == 1,
        fallback_used: info.fallback_used == 1,
        kernel_symbol: c_string(&info.kernel_symbol),
        device_symbol: c_string(&info.device_symbol),
        target,
    })
}

fn decode_wave_split_q_preload_enabled(
    expected_target: Option<&str>,
    use_decode_wave_split: bool,
    q_preload_opt_in: Option<&std::ffi::OsStr>,
) -> bool {
    expected_target == Some("gfx1030")
        && use_decode_wave_split
        && q_preload_opt_in.is_none_or(|value| value == "1")
}

#[allow(clippy::too_many_arguments)]
fn decode_wave_split_fp16_pair_enabled(
    expected_target: Option<&str>,
    query_count: u64,
    committed_kv_length: u64,
    query_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    encoding: KvCacheEncoding,
    opt_in: Option<&std::ffi::OsStr>,
    force_baseline: bool,
) -> bool {
    !force_baseline
        && expected_target == Some("gfx1030")
        && opt_in.is_none_or(|value| value == "1")
        && query_count == 1
        && committed_kv_length >= 1024
        && query_heads == 16
        && kv_heads == 4
        && head_dim == 256
        && encoding == KvCacheEncoding::Fp16
}

#[allow(clippy::too_many_arguments)]
fn decode_gqa4_split_enabled(
    expected_target: Option<&str>,
    query_count: u64,
    committed_kv_length: u64,
    query_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    encoding: KvCacheEncoding,
    opt_in: Option<&std::ffi::OsStr>,
    force_baseline: bool,
) -> bool {
    !force_baseline
        && expected_target == Some("gfx1030")
        && opt_in.is_some_and(|value| value == "1")
        && query_count == 1
        && committed_kv_length >= 4096
        && query_heads == 16
        && kv_heads == 4
        && head_dim == 256
        && encoding == KvCacheEncoding::Fp16
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn decode_gqa4_split_p32_enabled(
    expected_target: Option<&str>,
    query_count: u64,
    committed_kv_length: u64,
    query_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    encoding: KvCacheEncoding,
    opt_in: Option<&std::ffi::OsStr>,
    force_baseline: bool,
) -> bool {
    decode_gqa4_split_p32_target_enabled(
        expected_target,
        query_count,
        committed_kv_length,
        query_heads,
        kv_heads,
        head_dim,
        encoding,
        opt_in,
        None,
        force_baseline,
    )
}

#[allow(clippy::too_many_arguments)]
fn decode_gqa4_split_p32_target_enabled(
    expected_target: Option<&str>,
    query_count: u64,
    committed_kv_length: u64,
    query_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    encoding: KvCacheEncoding,
    gfx1030_opt_in: Option<&std::ffi::OsStr>,
    gfx1201_opt_in: Option<&std::ffi::OsStr>,
    force_baseline: bool,
) -> bool {
    let target_opt_in = match expected_target {
        Some("gfx1030") => gfx1030_opt_in.is_none_or(|value| value == "1"),
        Some("gfx1201") => gfx1201_opt_in.is_none_or(|value| value == "1"),
        _ => false,
    };
    !force_baseline
        && target_opt_in
        && query_count == 1
        && committed_kv_length >= 4096
        && query_heads == 16
        && kv_heads == 4
        && head_dim == 256
        && encoding == KvCacheEncoding::Fp16
}

#[allow(clippy::too_many_arguments)]
fn decode_gqa6_split_p64_enabled(
    expected_target: Option<&str>,
    query_count: u64,
    committed_kv_length: u64,
    query_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    encoding: KvCacheEncoding,
    opt_in: Option<&std::ffi::OsStr>,
    force_baseline: bool,
) -> bool {
    !force_baseline
        && matches!(expected_target, Some("gfx1030" | "gfx1201"))
        && opt_in.is_some_and(|value| value == "1")
        && query_count == 1
        && ((expected_target == Some("gfx1030") && committed_kv_length >= 8192)
            || (expected_target == Some("gfx1201") && committed_kv_length >= 4096))
        && query_heads == 24
        && kv_heads == 4
        && head_dim == 256
        && encoding == KvCacheEncoding::Fp16
}

#[allow(clippy::too_many_arguments)]
fn decode_gqa6_split_p128_enabled(
    expected_target: Option<&str>,
    query_count: u64,
    committed_kv_length: u64,
    query_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    encoding: KvCacheEncoding,
    opt_in: Option<&std::ffi::OsStr>,
    force_baseline: bool,
) -> bool {
    !force_baseline
        && expected_target == Some("gfx1030")
        && opt_in.is_some_and(|value| value == "1")
        && query_count == 1
        && committed_kv_length >= 8192
        && query_heads == 24
        && kv_heads == 4
        && head_dim == 256
        && encoding == KvCacheEncoding::Fp16
}

#[allow(clippy::too_many_arguments)]
fn decode_gqa6_split_p32_enabled(
    expected_target: Option<&str>,
    query_count: u64,
    committed_kv_length: u64,
    query_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    encoding: KvCacheEncoding,
    opt_in: Option<&std::ffi::OsStr>,
    force_baseline: bool,
) -> bool {
    !force_baseline
        && matches!(expected_target, Some("gfx1030" | "gfx1201"))
        && opt_in.is_some_and(|value| value == "1")
        && query_count == 1
        && committed_kv_length >= 4096
        && query_heads == 24
        && kv_heads == 4
        && head_dim == 256
        && encoding == KvCacheEncoding::Fp16
}

#[allow(clippy::too_many_arguments)]
fn decode_wave_split_short_enabled(
    expected_target: Option<&str>,
    query_count: u64,
    committed_kv_length: u64,
    query_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    encoding: KvCacheEncoding,
    short_decode_opt_in: Option<&std::ffi::OsStr>,
) -> bool {
    expected_target == Some("gfx1030")
        && short_decode_opt_in.is_none_or(|value| value == "1")
        && query_count == 1
        && (32..1024).contains(&committed_kv_length)
        && query_heads == 16
        && kv_heads == 4
        && head_dim == 256
        && encoding == KvCacheEncoding::Fp16
}

fn decode_wave_split_short_q_preload_enabled(
    use_decode_wave_split_short: bool,
    short_q_preload_opt_in: Option<&std::ffi::OsStr>,
) -> bool {
    use_decode_wave_split_short && short_q_preload_opt_in.is_none_or(|value| value == "1")
}

#[allow(clippy::too_many_arguments)]
fn scaled_prefill_gemm_enabled(
    expected_target: Option<&str>,
    query_count: u64,
    query_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    encoding: KvCacheEncoding,
    opt_in: Option<&std::ffi::OsStr>,
    force_baseline: bool,
) -> bool {
    !force_baseline
        && expected_target == Some("gfx1030")
        && query_count >= 1024
        && query_heads == 16
        && kv_heads == 4
        && head_dim == 256
        && match encoding {
            KvCacheEncoding::Fp16 => opt_in.is_none_or(|value| value == "1"),
            KvCacheEncoding::Mxfp8E4 => opt_in.is_some_and(|value| value == "1"),
            _ => false,
        }
}

#[allow(clippy::too_many_arguments)]
fn long_prefill_v2_enabled(
    expected_target: Option<&str>,
    query_count: u64,
    query_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    encoding: KvCacheEncoding,
    opt_in: Option<&std::ffi::OsStr>,
    force_baseline: bool,
) -> bool {
    !force_baseline
        && expected_target == Some("gfx1030")
        && opt_in.is_some_and(|value| value == "1")
        && query_count >= 1024
        && query_heads == 16
        && kv_heads == 4
        && head_dim == 256
        && encoding == KvCacheEncoding::Fp16
}

#[allow(clippy::too_many_arguments)]
fn gqa6_qtile4_fp16_key_tile_enabled(
    expected_target: Option<&str>,
    query_count: u64,
    query_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    encoding: KvCacheEncoding,
    opt_ins: [Option<&std::ffi::OsStr>; 4],
    force_baseline: bool,
) -> Option<u32> {
    // Force baseline has precedence over every candidate opt-in.  Candidate
    // precedence for several enabled variables is the explicit K4 > K8 >
    // K16 > K32 order below.
    if force_baseline
        || !matches!(expected_target, Some("gfx1030" | "gfx1201"))
        || query_count < 128
        || query_heads != 24
        || kv_heads != 4
        || head_dim != 256
        || encoding != KvCacheEncoding::Fp16
    {
        return None;
    }
    for (key_tile, opt_in) in [4_u32, 8, 16, 32].into_iter().zip(opt_ins) {
        if opt_in.is_some_and(|value| value == "1") {
            return Some(key_tile);
        }
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn gqa6_blocksoftmax_enabled(
    expected_target: Option<&str>,
    query_count: u64,
    query_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    encoding: KvCacheEncoding,
    gfx1030_opt_in: Option<&std::ffi::OsStr>,
    gfx1201_opt_in: Option<&std::ffi::OsStr>,
    force_baseline: bool,
) -> bool {
    let target_opt_in = match expected_target {
        Some("gfx1030") => gfx1030_opt_in.is_some_and(|value| value == "1"),
        Some("gfx1201") => gfx1201_opt_in.is_some_and(|value| value == "1"),
        _ => false,
    };
    !force_baseline
        && target_opt_in
        && query_count >= 128
        && query_heads == 24
        && kv_heads == 4
        && head_dim == 256
        && encoding == KvCacheEncoding::Fp16
}

#[allow(clippy::too_many_arguments)]
fn gqa6_blocksoftmax_q8_enabled(
    expected_target: Option<&str>,
    query_count: u64,
    query_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    encoding: KvCacheEncoding,
    opt_in: Option<&std::ffi::OsStr>,
    force_baseline: bool,
) -> bool {
    !force_baseline
        && expected_target == Some("gfx1201")
        && opt_in.is_some_and(|value| value == "1")
        && query_count >= 128
        && query_heads == 24
        && kv_heads == 4
        && head_dim == 256
        && encoding == KvCacheEncoding::Fp16
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn gqa6_qtile4_k32_fp16_enabled(
    expected_target: Option<&str>,
    query_count: u64,
    query_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    encoding: KvCacheEncoding,
    opt_in: Option<&std::ffi::OsStr>,
    force_baseline: bool,
) -> bool {
    gqa6_qtile4_fp16_key_tile_enabled(
        expected_target,
        query_count,
        query_heads,
        kv_heads,
        head_dim,
        encoding,
        [None, None, None, opt_in],
        force_baseline,
    ) == Some(32)
}

fn implicit_attention_scale_evidence(head_dim: u32) -> (u32, u32, [u32; 8]) {
    let denominator = (head_dim as f32).sqrt() as u32;
    if u64::from(denominator) * u64::from(denominator) == u64::from(head_dim) {
        (denominator, 0, [0; 8])
    } else {
        let scale_bits = (1.0_f32 / (head_dim as f32).sqrt()).to_bits();
        (0, scale_bits, [0, 0, 0, 0, scale_bits, 1, 0, 0])
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_causal_attention_info(
    info: &sys::sllm_causal_attention_dispatch_info_t,
    context: &Context,
    start_position: u64,
    committed_kv_length: u64,
    descriptor: KvStateDescriptor,
    query_heads: u32,
    sliding_window: Option<u64>,
    score_scale: Option<f32>,
) -> Result<CausalAttentionEvidence, RuntimeError> {
    let observed_target = c_string(&info.gcn_arch_name);
    let target = logical_gcn_arch_name(&observed_target).to_owned();
    let query_count = committed_kv_length
        .checked_sub(start_position)
        .ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::CausalAttentionLengthMismatch,
                "causal attention evidence range underflowed",
            )
        })?;
    let expected_grid = query_count
        .checked_mul(u64::from(query_heads))
        .and_then(|value| u32::try_from(value).ok());
    let expected_target = context.expected_target();
    if let Some(scale) = score_scale {
        let window = sliding_window.unwrap_or(0);
        let retained_start = if window == 0 {
            0
        } else {
            committed_kv_length.saturating_sub(window)
        };
        let expected_reserved = [
            window as u32,
            (window >> 32) as u32,
            retained_start as u32,
            (retained_start >> 32) as u32,
            scale.to_bits(),
            1,
            0,
            0,
        ];
        let (kernel_id, kernel_symbol, device_symbol) = if window == 0 {
            (
                sys::SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_SCALED_STATIC_FP8_V1,
                "causal_attention.scaled_static_fp8_gqa.v1",
                "sllm_causal_attention_scaled_static_fp8_gqa_v1",
            )
        } else {
            (
                sys::SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_SLIDING_STATIC_FP8_V1,
                "causal_attention.sliding_static_fp8_gqa.v1",
                "sllm_causal_attention_sliding_static_fp8_gqa_v1",
            )
        };
        if descriptor.sliding_window() != sliding_window
            || descriptor.static_fp8_scales() != Some((1.0, 1.0))
            || info.struct_size != size_of::<sys::sllm_causal_attention_dispatch_info_t>() as u32
            || info.abi_version != sys::SLLM_HIP_ABI_VERSION
            || info.info_version != sys::SLLM_HIP_CAUSAL_ATTENTION_DISPATCH_INFO_VERSION
            || info.backend != sys::SLLM_BACKEND_HIP
            || info.dispatch_id == 0
            || info.dispatch_count != 1
            || info.kernel_id != kernel_id
            || info.workgroup_size_x != sys::SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE
            || Some(info.grid_size_x) != expected_grid
            || info.query_count != query_count
            || info.start_position != start_position
            || info.committed_kv_length != committed_kv_length
            || info.q_heads != query_heads
            || info.kv_heads != descriptor.layout().heads() as u32
            || info.head_dim != descriptor.layout().head_dim() as u32
            || info.scale_denominator != 0
            || info.fallback_allowed != 0
            || info.fallback_used != 0
            || c_string(&info.kernel_symbol) != kernel_symbol
            || c_string(&info.device_symbol) != device_symbol
            || info.reserved != expected_reserved
            || committed_kv_length > descriptor.capacity()
            || expected_target.is_some_and(|expected| !gcn_arch_matches(expected, &observed_target))
        {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidCausalAttentionDescriptor,
                "native explicit-scale causal attention metadata failed exact-target/no-fallback validation",
            ));
        }
        return Ok(CausalAttentionEvidence {
            dispatch_id: info.dispatch_id,
            dispatch_count: info.dispatch_count,
            kernel_id: info.kernel_id,
            workgroup_size_x: info.workgroup_size_x,
            grid_size_x: info.grid_size_x,
            query_count: info.query_count,
            start_position: info.start_position,
            committed_kv_length: info.committed_kv_length,
            sliding_window: window,
            retained_start,
            score_scale_bits: scale.to_bits(),
            explicit_score_scale: true,
            q_heads: info.q_heads,
            kv_heads: info.kv_heads,
            head_dim: info.head_dim,
            scale_denominator: info.scale_denominator,
            fallback_allowed: false,
            fallback_used: false,
            kernel_symbol: c_string(&info.kernel_symbol),
            device_symbol: c_string(&info.device_symbol),
            target,
        });
    }
    if let Some(window) = sliding_window {
        let retained_start = committed_kv_length.saturating_sub(window);
        let expected_reserved = [
            window as u32,
            (window >> 32) as u32,
            retained_start as u32,
            (retained_start >> 32) as u32,
            0,
            0,
            0,
            0,
        ];
        if descriptor.sliding_window() != Some(window)
            || info.struct_size != size_of::<sys::sllm_causal_attention_dispatch_info_t>() as u32
            || info.abi_version != sys::SLLM_HIP_ABI_VERSION
            || info.info_version != sys::SLLM_HIP_CAUSAL_ATTENTION_DISPATCH_INFO_VERSION
            || info.backend != sys::SLLM_BACKEND_HIP
            || info.dispatch_id == 0
            || info.dispatch_count != 1
            || info.kernel_id != sys::SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_SLIDING_STATIC_FP8_V1
            || info.workgroup_size_x != sys::SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE
            || Some(info.grid_size_x) != expected_grid
            || info.query_count != query_count
            || info.start_position != start_position
            || info.committed_kv_length != committed_kv_length
            || info.q_heads != query_heads
            || info.kv_heads != descriptor.layout().heads() as u32
            || info.head_dim != descriptor.layout().head_dim() as u32
            || info.scale_denominator != sys::SLLM_HIP_CAUSAL_ATTENTION_SCALE_DENOMINATOR
            || info.fallback_allowed != 0
            || info.fallback_used != 0
            || c_string(&info.kernel_symbol) != "causal_attention.sliding_static_fp8_gqa.v1"
            || c_string(&info.device_symbol) != "sllm_causal_attention_sliding_static_fp8_gqa_v1"
            || info.reserved != expected_reserved
            || committed_kv_length > descriptor.capacity()
            || expected_target.is_some_and(|expected| !gcn_arch_matches(expected, &observed_target))
        {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidCausalAttentionDescriptor,
                "native sliding causal attention metadata failed exact-target/no-fallback validation",
            ));
        }
        return Ok(CausalAttentionEvidence {
            dispatch_id: info.dispatch_id,
            dispatch_count: info.dispatch_count,
            kernel_id: info.kernel_id,
            workgroup_size_x: info.workgroup_size_x,
            grid_size_x: info.grid_size_x,
            query_count: info.query_count,
            start_position: info.start_position,
            committed_kv_length: info.committed_kv_length,
            sliding_window: window,
            retained_start,
            score_scale_bits: 0,
            explicit_score_scale: false,
            q_heads: info.q_heads,
            kv_heads: info.kv_heads,
            head_dim: info.head_dim,
            scale_denominator: info.scale_denominator,
            fallback_allowed: false,
            fallback_used: false,
            kernel_symbol: c_string(&info.kernel_symbol),
            device_symbol: c_string(&info.device_symbol),
            target,
        });
    }
    let use_gfx1201_wave_provider =
        expected_target == Some("gfx1201") && (query_count == 1 || query_count >= 32);
    let use_phase33_common_provider = matches!(expected_target, Some("gfx1030" | "gfx1201"));
    let force_baseline =
        std::env::var_os("SLLM_CAUSAL_ATTENTION_FORCE_BASELINE").is_some_and(|value| value == "1");
    let use_decode_wave_split_long = use_phase33_common_provider
        && query_count == 1
        && committed_kv_length >= 1024
        && descriptor.layout().head_dim() == 256;
    let short_decode_opt_in = std::env::var_os("SLLM_CAUSAL_ATTENTION_GFX1030_DECODE_WAVE_SHORT");
    let use_decode_wave_split_short = decode_wave_split_short_enabled(
        expected_target,
        query_count,
        committed_kv_length,
        query_heads,
        descriptor.layout().heads() as u32,
        descriptor.layout().head_dim() as u32,
        descriptor.cache_encoding(),
        short_decode_opt_in.as_deref(),
    );
    let use_decode_wave_split = use_decode_wave_split_long || use_decode_wave_split_short;
    let fp16_pair_opt_in = std::env::var_os("SLLM_CAUSAL_ATTENTION_GFX1030_DECODE_WAVE_FP16_PAIR");
    let gqa4_split_opt_in = std::env::var_os("SLLM_CAUSAL_ATTENTION_GFX1030_DECODE_GQA4_SPLIT");
    let gqa4_split_p32_opt_in =
        std::env::var_os("SLLM_CAUSAL_ATTENTION_GFX1030_DECODE_GQA4_SPLIT_P32");
    let gfx1201_gqa4_split_p32_opt_in =
        std::env::var_os("SLLM_CAUSAL_ATTENTION_GFX1201_DECODE_GQA4_SPLIT_P32");
    let gqa6_split_p64_opt_in = std::env::var_os("SLLM_CAUSAL_ATTENTION_GQA6_DECODE_SPLIT_P64");
    let gqa6_split_p128_opt_in = std::env::var_os("SLLM_CAUSAL_ATTENTION_GQA6_DECODE_SPLIT_P128");
    let gqa6_split_p32_opt_in = std::env::var_os("SLLM_CAUSAL_ATTENTION_GQA6_DECODE_SPLIT_P32");
    let gqa6_blocksoftmax_gfx1030_opt_in =
        std::env::var_os("SLLM_CAUSAL_ATTENTION_GQA6_PREFILL_BLOCKSOFTMAX_GFX1030");
    let gqa6_blocksoftmax_gfx1201_opt_in =
        std::env::var_os("SLLM_CAUSAL_ATTENTION_GQA6_PREFILL_BLOCKSOFTMAX_GFX1201");
    let gqa6_blocksoftmax_q8_gfx1201_opt_in =
        std::env::var_os("SLLM_CAUSAL_ATTENTION_GQA6_PREFILL_BLOCKSOFTMAX_Q8_GFX1201");
    let gqa6_rocblas_f32_opt_in =
        std::env::var_os("SLLM_CAUSAL_ATTENTION_GQA6_PREFILL_GFX1030_ROCBLAS_F32");
    let gfx1201_gqa6_rocblas_f32_opt_in =
        std::env::var_os("SLLM_CAUSAL_ATTENTION_GQA6_PREFILL_GFX1201_ROCBLAS_F32");
    let gfx1201_gqa6_rocblas_f16_tail_opt_in =
        std::env::var_os("SLLM_CAUSAL_ATTENTION_GQA6_PREFILL_GFX1201_ROCBLAS_F16_TAIL");
    let use_gqa6_rocblas_f32 = expected_target == Some("gfx1030")
        && !force_baseline
        && gqa6_rocblas_f32_opt_in
            .as_deref()
            .is_some_and(|value| value == "1")
        && query_count > 1
        && query_count <= u64::from(u32::MAX)
        && start_position
            .checked_add(query_count)
            .is_some_and(|end| end == committed_kv_length)
        && query_heads == 24
        && descriptor.layout().heads() == 4
        && descriptor.layout().head_dim() == 256
        && descriptor.cache_encoding() == KvCacheEncoding::Fp16;
    let use_gfx1201_gqa6_rocblas_f32 = expected_target == Some("gfx1201")
        && !force_baseline
        && gfx1201_gqa6_rocblas_f32_opt_in
            .as_deref()
            .is_some_and(|value| value == "1")
        && query_count > 1
        && query_count <= u64::from(u32::MAX)
        && start_position
            .checked_add(query_count)
            .is_some_and(|end| end == committed_kv_length)
        && query_heads == 24
        && descriptor.layout().heads() == 4
        && descriptor.layout().head_dim() == 256
        && descriptor.cache_encoding() == KvCacheEncoding::Fp16;
    let use_gfx1201_gqa6_rocblas_f16_tail = use_gfx1201_gqa6_rocblas_f32
        && start_position > 0
        && gfx1201_gqa6_rocblas_f16_tail_opt_in
            .as_deref()
            .is_some_and(|value| value == "1");
    let use_any_gqa6_rocblas_f32 = use_gqa6_rocblas_f32 || use_gfx1201_gqa6_rocblas_f32;
    let use_decode_gqa4_split_p32 = decode_gqa4_split_p32_target_enabled(
        expected_target,
        query_count,
        committed_kv_length,
        query_heads,
        descriptor.layout().heads() as u32,
        descriptor.layout().head_dim() as u32,
        descriptor.cache_encoding(),
        gqa4_split_p32_opt_in.as_deref(),
        gfx1201_gqa4_split_p32_opt_in.as_deref(),
        force_baseline,
    );
    let use_decode_gqa6_split_p128 = decode_gqa6_split_p128_enabled(
        expected_target,
        query_count,
        committed_kv_length,
        query_heads,
        descriptor.layout().heads() as u32,
        descriptor.layout().head_dim() as u32,
        descriptor.cache_encoding(),
        gqa6_split_p128_opt_in.as_deref(),
        force_baseline,
    );
    let use_decode_gqa6_split_p64 = decode_gqa6_split_p64_enabled(
        expected_target,
        query_count,
        committed_kv_length,
        query_heads,
        descriptor.layout().heads() as u32,
        descriptor.layout().head_dim() as u32,
        descriptor.cache_encoding(),
        gqa6_split_p64_opt_in.as_deref(),
        force_baseline,
    ) && !use_decode_gqa6_split_p128;
    let use_decode_gqa6_split_p32 = decode_gqa6_split_p32_enabled(
        expected_target,
        query_count,
        committed_kv_length,
        query_heads,
        descriptor.layout().heads() as u32,
        descriptor.layout().head_dim() as u32,
        descriptor.cache_encoding(),
        gqa6_split_p32_opt_in.as_deref(),
        force_baseline,
    ) && !use_decode_gqa6_split_p128
        && !use_decode_gqa6_split_p64;
    let use_prefill_gqa6_blocksoftmax_q8 = gqa6_blocksoftmax_q8_enabled(
        expected_target,
        query_count,
        query_heads,
        descriptor.layout().heads() as u32,
        descriptor.layout().head_dim() as u32,
        descriptor.cache_encoding(),
        gqa6_blocksoftmax_q8_gfx1201_opt_in.as_deref(),
        force_baseline,
    );
    let use_prefill_gqa6_blocksoftmax = gqa6_blocksoftmax_enabled(
        expected_target,
        query_count,
        query_heads,
        descriptor.layout().heads() as u32,
        descriptor.layout().head_dim() as u32,
        descriptor.cache_encoding(),
        gqa6_blocksoftmax_gfx1030_opt_in.as_deref(),
        gqa6_blocksoftmax_gfx1201_opt_in.as_deref(),
        force_baseline,
    ) && !use_prefill_gqa6_blocksoftmax_q8
        && !use_any_gqa6_rocblas_f32;
    let use_decode_gqa4_split = decode_gqa4_split_enabled(
        expected_target,
        query_count,
        committed_kv_length,
        query_heads,
        descriptor.layout().heads() as u32,
        descriptor.layout().head_dim() as u32,
        descriptor.cache_encoding(),
        gqa4_split_opt_in.as_deref(),
        force_baseline,
    ) && !use_decode_gqa4_split_p32
        && !use_decode_gqa6_split_p32
        && !use_decode_gqa6_split_p64;
    let use_decode_wave_split_fp16_pair = decode_wave_split_fp16_pair_enabled(
        expected_target,
        query_count,
        committed_kv_length,
        query_heads,
        descriptor.layout().heads() as u32,
        descriptor.layout().head_dim() as u32,
        descriptor.cache_encoding(),
        fp16_pair_opt_in.as_deref(),
        force_baseline,
    ) && !use_decode_gqa4_split
        && !use_decode_gqa4_split_p32
        && !use_decode_gqa6_split_p32
        && !use_decode_gqa6_split_p64;
    let q_preload_opt_in = std::env::var_os("SLLM_CAUSAL_ATTENTION_GFX1030_Q_PRELOAD");
    let use_decode_wave_split_q_preload_long = decode_wave_split_q_preload_enabled(
        expected_target,
        use_decode_wave_split_long,
        q_preload_opt_in.as_deref(),
    );
    let short_q_preload_opt_in =
        std::env::var_os("SLLM_CAUSAL_ATTENTION_GFX1030_DECODE_WAVE_SHORT_Q_PRELOAD");
    let use_decode_wave_split_q_preload_short = decode_wave_split_short_q_preload_enabled(
        use_decode_wave_split_short,
        short_q_preload_opt_in.as_deref(),
    );
    let use_decode_wave_split_q_preload =
        use_decode_wave_split_q_preload_long || use_decode_wave_split_q_preload_short;
    let use_prefill_gqa4 = use_phase33_common_provider
        && query_count >= 64
        && query_heads as usize / descriptor.layout().heads() == 4
        && descriptor.layout().head_dim() == 256;
    let gqa6_qtile4_opt_in = std::env::var_os("SLLM_CAUSAL_ATTENTION_GQA6_QTILE4");
    let use_prefill_gqa6_qtile4 = use_phase33_common_provider
        && query_count >= 128
        && query_heads as usize / descriptor.layout().heads() == 6
        && descriptor.layout().head_dim() == 256
        && gqa6_qtile4_opt_in
            .as_deref()
            .is_some_and(|value| value == "1")
        && !force_baseline
        && !use_prefill_gqa6_blocksoftmax_q8
        && !use_any_gqa6_rocblas_f32;
    let gqa6_qtile4_k4_fp16_opt_in = std::env::var_os("SLLM_CAUSAL_ATTENTION_GQA6_QTILE4_K4_FP16");
    let gqa6_qtile4_k8_fp16_opt_in = std::env::var_os("SLLM_CAUSAL_ATTENTION_GQA6_QTILE4_K8_FP16");
    let gqa6_qtile4_k16_fp16_opt_in =
        std::env::var_os("SLLM_CAUSAL_ATTENTION_GQA6_QTILE4_K16_FP16");
    let gqa6_qtile4_k32_fp16_opt_in =
        std::env::var_os("SLLM_CAUSAL_ATTENTION_GQA6_QTILE4_K32_FP16");
    let gqa6_qtile4_fp16_key_tile = gqa6_qtile4_fp16_key_tile_enabled(
        expected_target,
        query_count,
        query_heads,
        descriptor.layout().heads() as u32,
        descriptor.layout().head_dim() as u32,
        descriptor.cache_encoding(),
        [
            gqa6_qtile4_k4_fp16_opt_in.as_deref(),
            gqa6_qtile4_k8_fp16_opt_in.as_deref(),
            gqa6_qtile4_k16_fp16_opt_in.as_deref(),
            gqa6_qtile4_k32_fp16_opt_in.as_deref(),
        ],
        force_baseline || use_prefill_gqa6_blocksoftmax_q8 || use_any_gqa6_rocblas_f32,
    );
    let use_prefill_gqa6_qtile4_k4_fp16 = gqa6_qtile4_fp16_key_tile == Some(4);
    let use_prefill_gqa6_qtile4_k8_fp16 = gqa6_qtile4_fp16_key_tile == Some(8);
    let use_prefill_gqa6_qtile4_k16_fp16 = gqa6_qtile4_fp16_key_tile == Some(16);
    let use_prefill_gqa6_qtile4_k32_fp16 = gqa6_qtile4_fp16_key_tile == Some(32);
    let scaled_prefill_opt_in =
        std::env::var_os("SLLM_CAUSAL_ATTENTION_GFX1030_SCALED_PREFILL_GEMM");
    let long_prefill_v2_opt_in = std::env::var_os("SLLM_CAUSAL_ATTENTION_GFX1030_LONG_PREFILL_V2");
    let use_long_prefill_v2 = long_prefill_v2_enabled(
        expected_target,
        query_count,
        query_heads,
        descriptor.layout().heads() as u32,
        descriptor.layout().head_dim() as u32,
        descriptor.cache_encoding(),
        long_prefill_v2_opt_in.as_deref(),
        force_baseline,
    );
    let use_scaled_prefill_gemm = scaled_prefill_gemm_enabled(
        expected_target,
        query_count,
        query_heads,
        descriptor.layout().heads() as u32,
        descriptor.layout().head_dim() as u32,
        descriptor.cache_encoding(),
        scaled_prefill_opt_in.as_deref(),
        force_baseline,
    ) && !use_long_prefill_v2;
    let phase66_prefill_opt_in = std::env::var_os("SLLM_CAUSAL_ATTENTION_PHASE66_TILED_PREFILL");
    let phase66_prefill_requested = phase66_prefill_opt_in
        .as_deref()
        .is_some_and(|value| value == "1");
    let phase66_encoding = matches!(
        descriptor.cache_encoding(),
        KvCacheEncoding::Fp16 | KvCacheEncoding::Mxfp8E4
    );
    let phase66_shape = expected_target == Some("gfx1201")
        && query_heads == 16
        && descriptor.layout().heads() == 4
        && descriptor.layout().head_dim() == 256
        && phase66_encoding;
    let phase66_q4k1_control =
        phase66_prefill_requested && phase66_shape && query_count >= 64 && !force_baseline;
    let phase66_query_tile = if phase66_prefill_requested
        && phase66_shape
        && query_count >= 128
        && committed_kv_length >= query_count
        && !force_baseline
    {
        Some(if committed_kv_length >= 2_048 {
            8_u64
        } else {
            4_u64
        })
    } else {
        None
    };
    let phase66_key_tile = phase66_query_tile.map(|_| {
        if committed_kv_length >= 512 {
            8_u64
        } else {
            4_u64
        }
    });
    let use_prefill_gqa4_qtile4 = !use_any_gqa6_rocblas_f32
        && ((use_prefill_gqa4
            && (query_count >= 128 || phase66_q4k1_control)
            && !force_baseline
            && !use_scaled_prefill_gemm)
            || use_prefill_gqa6_qtile4
            || use_prefill_gqa6_qtile4_k4_fp16
            || use_prefill_gqa6_qtile4_k8_fp16
            || use_prefill_gqa6_qtile4_k16_fp16
            || use_prefill_gqa6_qtile4_k32_fp16
            || use_prefill_gqa6_blocksoftmax
            || use_prefill_gqa6_blocksoftmax_q8);
    let (expected_kernel_id, baseline_kernel, baseline_device) =
        if use_gfx1201_gqa6_rocblas_f16_tail {
            (
                sys::SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_GQA6_ROCBLAS_F16_TAIL_GFX1201_V1,
                "causal_attention.online_softmax_gqa.v2",
                "sllm_causal_attention_online_softmax_gqa_v2",
            )
        } else if use_gfx1201_gqa6_rocblas_f32 {
            (
                sys::SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_GQA6_ROCBLAS_F32_GFX1201_V1,
                "causal_attention.online_softmax_gqa.v2",
                "sllm_causal_attention_online_softmax_gqa_v2",
            )
        } else if use_decode_gqa6_split_p128 {
            (
                sys::SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_GQA6_SPLIT_P128_GFX1030_V1,
                "causal_attention.online_softmax_gqa.v2",
                "sllm_causal_attention_online_softmax_gqa_v2",
            )
        } else if descriptor.cache_encoding() == KvCacheEncoding::Fp16 {
            (
                sys::SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_ONLINE_SOFTMAX_V2,
                "causal_attention.online_softmax_gqa.v2",
                "sllm_causal_attention_online_softmax_gqa_v2",
            )
        } else {
            (
                sys::SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_PACKED_KV_V3,
                "causal_attention.online_softmax_gqa.packed_kv.v3",
                "sllm_causal_attention_online_softmax_gqa_packed_kv_v3",
            )
        };
    let (expected_kernel, expected_device) = if use_decode_gqa6_split_p128 {
        (
            "causal_attention.decode.gqa6_split_p128.fp16.v1",
            "sllm_causal_attention_decode_gqa6_split_p128_v1",
        )
    } else if use_decode_gqa6_split_p64 {
        (
            "causal_attention.decode.gqa6_split_p64.fp16.v1",
            "sllm_causal_attention_decode_gqa6_split_p64_v1",
        )
    } else if use_decode_gqa6_split_p32 {
        (
            "causal_attention.decode.gqa6_split_p32.fp16.v1",
            "sllm_causal_attention_decode_gqa6_split_p32_v1",
        )
    } else if use_prefill_gqa6_blocksoftmax_q8 {
        (
            "causal_attention.prefill.gqa6_blocksoftmax_q8.fp16.v1",
            "sllm_causal_attention_prefill_gqa6_blocksoftmax_q8_fp16_v1",
        )
    } else if use_prefill_gqa6_blocksoftmax {
        (
            "causal_attention.prefill.gqa6_blocksoftmax.fp16.v1",
            "sllm_causal_attention_prefill_gqa6_blocksoftmax_fp16_v1",
        )
    } else if use_decode_gqa4_split_p32 {
        (
            "causal_attention.decode.gqa4_tiled_split.p32.v1",
            "sllm_causal_attention_decode_gqa4_split_p32_v1",
        )
    } else if use_decode_gqa4_split {
        (
            "causal_attention.decode.gqa4_tiled_split.v1",
            "sllm_causal_attention_decode_gqa4_tiled_split_v1",
        )
    } else if use_decode_wave_split_fp16_pair {
        (
            "causal_attention.decode.wave8_split.fp16_pair.v1",
            "sllm_causal_attention_decode_wave8_split_fp16_pair_v1",
        )
    } else if use_decode_wave_split {
        if use_decode_wave_split_q_preload {
            (
                "causal_attention.decode.wave8_split.q_preload.v1",
                "sllm_causal_attention_decode_wave8_split_q_preload_v1",
            )
        } else {
            (
                "causal_attention.decode.wave8_split.v5",
                "sllm_causal_attention_decode_wave8_split_v5",
            )
        }
    } else if use_long_prefill_v2 {
        (
            "causal_attention.prefill.gfx1030_qtile8_split.v2",
            "sllm_causal_attention_prefill_gfx1030_qtile8_split_v2",
        )
    } else if use_gfx1201_gqa6_rocblas_f16_tail {
        (
            "causal_attention.prefill.gfx1201_rocblas_gqa6_f16_tail.v1",
            "sllm_causal_attention_prefill_gfx1201_rocblas_gqa6_f16_tail_v1",
        )
    } else if use_gfx1201_gqa6_rocblas_f32 {
        (
            "causal_attention.prefill.gfx1201_rocblas_gqa6_f32.v1",
            "sllm_causal_attention_prefill_gfx1201_rocblas_gqa6_f32_v1",
        )
    } else if use_gqa6_rocblas_f32 {
        (
            "causal_attention.prefill.gfx1030_rocblas_gqa6_f32.v1",
            "sllm_causal_attention_prefill_gfx1030_rocblas_gqa6_f32_v1",
        )
    } else if use_scaled_prefill_gemm {
        (
            "causal_attention.prefill.gfx1030_hipblas_scaled_fp16.v1",
            "sllm_causal_attention_prefill_gfx1030_hipblas_scaled_fp16_v1",
        )
    } else if phase66_query_tile == Some(4) && phase66_key_tile == Some(4) {
        (
            "causal_attention.prefill.typed_q4k4.v1",
            "sllm_causal_attention_prefill_typed_q4k4_v1",
        )
    } else if phase66_query_tile == Some(4) && phase66_key_tile == Some(8) {
        (
            "causal_attention.prefill.typed_q4k8.v1",
            "sllm_causal_attention_prefill_typed_q4k8_v1",
        )
    } else if phase66_query_tile == Some(8) && phase66_key_tile == Some(8) {
        (
            "causal_attention.prefill.typed_q8k8.v1",
            "sllm_causal_attention_prefill_typed_q8k8_v1",
        )
    } else if use_prefill_gqa4_qtile4 {
        if use_prefill_gqa6_qtile4_k4_fp16 {
            (
                "causal_attention.prefill.gqa6_qtile4_k4.fp16.v1",
                "sllm_causal_attention_prefill_gqa6_qtile4_k4_fp16_v1",
            )
        } else if use_prefill_gqa6_qtile4_k8_fp16 {
            (
                "causal_attention.prefill.gqa6_qtile4_k8.fp16.v1",
                "sllm_causal_attention_prefill_gqa6_qtile4_k8_fp16_v1",
            )
        } else if use_prefill_gqa6_qtile4_k16_fp16 {
            (
                "causal_attention.prefill.gqa6_qtile4_k16.fp16.v1",
                "sllm_causal_attention_prefill_gqa6_qtile4_k16_fp16_v1",
            )
        } else if use_prefill_gqa6_qtile4_k32_fp16 {
            (
                "causal_attention.prefill.gqa6_qtile4_k32.fp16.v1",
                "sllm_causal_attention_prefill_gqa6_qtile4_k32_fp16_v1",
            )
        } else if use_prefill_gqa6_qtile4 {
            (
                "causal_attention.prefill.gqa6_qtile4.v1",
                "sllm_causal_attention_prefill_gqa6_qtile4_v1",
            )
        } else {
            (
                "causal_attention.prefill.gqa4_qtile4.v7",
                "sllm_causal_attention_prefill_gqa4_qtile4_v7",
            )
        }
    } else if use_prefill_gqa4 {
        (
            "causal_attention.prefill.gqa4_shared.v6",
            "sllm_causal_attention_prefill_gqa4_shared_v6",
        )
    } else if use_gfx1201_wave_provider {
        if descriptor.cache_encoding() == KvCacheEncoding::Fp16 {
            (
                "causal_attention.online_softmax_gqa.gfx1201_wave.v4",
                "sllm_causal_attention_gfx1201_wave_v4",
            )
        } else {
            (
                "causal_attention.online_softmax_gqa.packed_kv.gfx1201_wave.v4",
                "sllm_causal_attention_packed_gfx1201_wave_v4",
            )
        }
    } else {
        (baseline_kernel, baseline_device)
    };
    let (expected_scale_denominator, implicit_scale_bits, expected_reserved) =
        implicit_attention_scale_evidence(descriptor.layout().head_dim() as u32);
    if info.struct_size != size_of::<sys::sllm_causal_attention_dispatch_info_t>() as u32
        || info.abi_version != sys::SLLM_HIP_ABI_VERSION
        || info.info_version != sys::SLLM_HIP_CAUSAL_ATTENTION_DISPATCH_INFO_VERSION
        || info.backend != sys::SLLM_BACKEND_HIP
        || info.dispatch_id == 0
        || info.dispatch_count
            != if use_gfx1201_gqa6_rocblas_f16_tail {
                5
            } else if use_any_gqa6_rocblas_f32 {
                7
            } else if use_decode_gqa4_split
                || use_decode_gqa4_split_p32
                || use_decode_gqa6_split_p128
                || use_decode_gqa6_split_p64
                || use_decode_gqa6_split_p32
                || use_long_prefill_v2
            {
                2
            } else {
                1
            }
        || info.kernel_id != expected_kernel_id
        || info.workgroup_size_x
            != if use_decode_gqa6_split_p128
                || use_decode_gqa6_split_p64
                || use_decode_gqa6_split_p32
            {
                192
            } else if use_decode_gqa4_split || use_decode_gqa4_split_p32 {
                128
            } else {
                sys::SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE
            }
        || Some(info.grid_size_x)
            != if use_decode_gqa6_split_p128 {
                Some(512)
            } else if use_decode_gqa6_split_p64 {
                Some(256)
            } else if use_decode_gqa6_split_p32 || use_decode_gqa4_split_p32 {
                Some(128)
            } else if use_decode_gqa4_split {
                Some(64)
            } else if use_any_gqa6_rocblas_f32 {
                query_count
                    .checked_mul(24)
                    .and_then(|value| u32::try_from(value).ok())
            } else if use_scaled_prefill_gemm {
                query_count
                    .checked_add(255)
                    .and_then(|value| (value / 256).checked_mul(descriptor.layout().heads() as u64))
                    .and_then(|value| u32::try_from(value).ok())
            } else if use_long_prefill_v2 {
                query_count
                    .checked_add(7)
                    .and_then(|value| {
                        (value / 8)
                            .checked_mul((descriptor.layout().heads() as u64).checked_mul(16)?)
                    })
                    .and_then(|value| u32::try_from(value).ok())
            } else if let Some(query_tile) = phase66_query_tile {
                query_count
                    .checked_add(query_tile - 1)
                    .and_then(|value| {
                        (value / query_tile).checked_mul(descriptor.layout().heads() as u64)
                    })
                    .and_then(|value| u32::try_from(value).ok())
            } else if use_prefill_gqa6_blocksoftmax_q8 {
                query_count
                    .checked_add(7)
                    .and_then(|value| (value / 8).checked_mul(descriptor.layout().heads() as u64))
                    .and_then(|value| u32::try_from(value).ok())
            } else if use_prefill_gqa4_qtile4 || use_prefill_gqa6_blocksoftmax {
                query_count
                    .checked_add(3)
                    .and_then(|value| (value / 4).checked_mul(descriptor.layout().heads() as u64))
                    .and_then(|value| u32::try_from(value).ok())
            } else if use_prefill_gqa4 {
                query_count
                    .checked_mul(descriptor.layout().heads() as u64)
                    .and_then(|value| u32::try_from(value).ok())
            } else {
                expected_grid
            }
        || info.query_count != query_count
        || info.start_position != start_position
        || info.committed_kv_length != committed_kv_length
        || info.q_heads != query_heads
        || info.kv_heads != descriptor.layout().heads() as u32
        || info.head_dim != descriptor.layout().head_dim() as u32
        || info.scale_denominator != expected_scale_denominator
        || info.fallback_allowed != 0
        || info.fallback_used != 0
        || c_string(&info.kernel_symbol) != expected_kernel
        || c_string(&info.device_symbol) != expected_device
        || info.reserved != expected_reserved
        || committed_kv_length > descriptor.capacity()
        || expected_target.is_some_and(|expected| !gcn_arch_matches(expected, &observed_target))
    {
        return Err(RuntimeError::local(
            RuntimeStatus::InvalidCausalAttentionDescriptor,
            "native causal attention metadata failed exact-target/no-fallback validation",
        ));
    }
    Ok(CausalAttentionEvidence {
        dispatch_id: info.dispatch_id,
        dispatch_count: info.dispatch_count,
        kernel_id: info.kernel_id,
        workgroup_size_x: info.workgroup_size_x,
        grid_size_x: info.grid_size_x,
        query_count: info.query_count,
        start_position: info.start_position,
        committed_kv_length: info.committed_kv_length,
        sliding_window: 0,
        retained_start: 0,
        score_scale_bits: implicit_scale_bits,
        explicit_score_scale: false,
        q_heads: info.q_heads,
        kv_heads: info.kv_heads,
        head_dim: info.head_dim,
        scale_denominator: info.scale_denominator,
        fallback_allowed: info.fallback_allowed != 0,
        fallback_used: info.fallback_used != 0,
        kernel_symbol: c_string(&info.kernel_symbol),
        device_symbol: c_string(&info.device_symbol),
        target,
    })
}

fn completion_result() -> sys::sllm_completion_result_t {
    sys::sllm_completion_result_t {
        struct_size: size_of::<sys::sllm_completion_result_t>() as u32,
        abi_version: sys::SLLM_HIP_ABI_VERSION,
        state: sys::SLLM_COMPLETION_STATE_PENDING,
        reserved0: 0,
        transfer_size_bytes: 0,
        available_bytes: 0,
        reserved: [0; 4],
    }
}

fn completion_state(raw: u32) -> Result<CompletionState, RuntimeError> {
    match raw {
        sys::SLLM_COMPLETION_STATE_PENDING => Ok(CompletionState::Pending),
        sys::SLLM_COMPLETION_STATE_SUCCESS => Ok(CompletionState::Success),
        sys::SLLM_COMPLETION_STATE_FAILURE => Ok(CompletionState::Failure),
        _ => Err(RuntimeError::local(
            RuntimeStatus::InternalError,
            "native KV completion returned an unknown state",
        )),
    }
}

fn timeout_millis(timeout: Duration) -> u32 {
    let millis = timeout.as_millis();
    if millis >= u128::from(MAX_FINITE_TIMEOUT_MS) {
        MAX_FINITE_TIMEOUT_MS
    } else {
        millis as u32
    }
}

fn c_string(value: &[std::ffi::c_char]) -> String {
    let length = value
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(value.len());
    value[..length]
        .iter()
        .map(|byte| *byte as u8 as char)
        .collect()
}

/// Exact BF16-to-FP16 round-to-nearest-even conversion used by the bounded
/// evidence oracle. It is independent of device storage.
pub fn bf16_to_f16_bits(bits: u16) -> u16 {
    f32_to_f16_bits(u32::from(bits) << 16)
}

fn f32_to_f16_bits(bits: u32) -> u16 {
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let fraction = bits & 0x7f_ff_ff;
    if exponent == 0xff {
        return sign | if fraction == 0 { 0x7c00 } else { 0x7e00 };
    }
    let unbiased = exponent - 127;
    if unbiased < -24 {
        return sign;
    }
    if unbiased < -14 {
        let shift = (-unbiased - 14) as u32;
        let mantissa = fraction | 0x80_0000;
        return sign | round_shift(mantissa, 13 + shift) as u16;
    }
    if unbiased > 15 {
        return sign | 0x7c00;
    }
    let half_exponent = (unbiased + 15) as u16;
    let rounded = round_shift(fraction, 13);
    if rounded >= 0x400 {
        let next_exponent = half_exponent + 1;
        return if next_exponent >= 0x1f {
            sign | 0x7c00
        } else {
            sign | (next_exponent << 10)
        };
    }
    sign | (half_exponent << 10) | rounded as u16
}

fn round_shift(value: u32, shift: u32) -> u32 {
    let truncated = value >> shift;
    let remainder = value & ((1_u32 << shift) - 1);
    let halfway = 1_u32 << (shift - 1);
    truncated + u32::from(remainder > halfway || (remainder == halfway && truncated & 1 != 0))
}

/// Exact expected placement in a native token-major [capacity, 4, 256] allocation.
pub fn expected_storage_offset(
    capacity: u64,
    start_position: u64,
    token: u64,
    head: u64,
    dim: u64,
) -> Option<u64> {
    if capacity == 0 || head >= 4 || dim >= 256 {
        return None;
    }
    start_position
        .checked_add(token)
        .filter(|position| *position < capacity)
        .and_then(|position| {
            position
                .checked_mul(4 * 256)?
                .checked_add(head.checked_mul(256)?)?
                .checked_add(dim)
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set_test_c_string<const N: usize>(destination: &mut [std::ffi::c_char; N], value: &str) {
        for (slot, byte) in destination.iter_mut().zip(value.bytes()) {
            *slot = byte as std::ffi::c_char;
        }
    }

    fn block16_descriptor(
        encoding: KvCacheEncoding,
        physical_variant: KvFp8PhysicalVariant,
    ) -> KvStateDescriptor {
        KvStateDescriptor::new_with_kv_fp8_block16(0, 17, 4, 257, encoding, physical_variant)
            .unwrap()
    }

    #[test]
    fn block16_native_storage_is_retired_for_every_target() {
        let ocp = block16_descriptor(
            KvCacheEncoding::Fp8E4M3Block16,
            KvFp8PhysicalVariant::OcpE4M3Fn,
        );
        let fnuz = block16_descriptor(
            KvCacheEncoding::Fp8E4M3Block16,
            KvFp8PhysicalVariant::E4M3FnuZ,
        );
        for (descriptor, target) in [
            (ocp, "gfx1201"),
            (ocp, "gfx1030"),
            (fnuz, "gfx942:sramecc+:xnack-"),
        ] {
            let error = native_kv_storage(descriptor, Some(target)).unwrap_err();
            assert_eq!(error.status(), RuntimeStatus::InvalidKvStateDescriptor);
            assert!(error.message().contains("retired"));
        }
    }

    #[test]
    fn implicit_attention_scale_evidence_is_exact_for_square_and_non_square_head_dims() {
        assert_eq!(implicit_attention_scale_evidence(256), (16, 0, [0; 8]));
        for head_dim in [128_u32, 512] {
            let (denominator, scale_bits, reserved) = implicit_attention_scale_evidence(head_dim);
            assert_eq!(denominator, 0);
            assert_eq!(scale_bits, (1.0_f32 / (head_dim as f32).sqrt()).to_bits());
            assert_eq!(reserved, [0, 0, 0, 0, scale_bits, 1, 0, 0]);
        }
    }

    #[test]
    fn explicit_score_scale_evidence_is_exact_for_full_and_sliding_static_fp8() {
        let context = Context::test_without_native();
        for (sliding_window, start, committed) in [
            (None, 0_u64, 1023_u64),
            (Some(1024), 0, 1024),
            (Some(1024), 1024, 1025),
        ] {
            let descriptor = if let Some(window) = sliding_window {
                KvStateDescriptor::new_with_static_fp8_sliding(0, 262_144, 4, 256, window).unwrap()
            } else {
                KvStateDescriptor::new_with_static_fp8(0, 262_144, 4, 256, 1.0, 1.0).unwrap()
            };
            let mut info = empty_causal_attention_info();
            info.backend = sys::SLLM_BACKEND_HIP;
            info.dispatch_id = 7;
            info.dispatch_count = 1;
            info.kernel_id = if sliding_window.is_some() {
                sys::SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_SLIDING_STATIC_FP8_V1
            } else {
                sys::SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_SCALED_STATIC_FP8_V1
            };
            info.workgroup_size_x = sys::SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE;
            info.query_count = committed - start;
            info.grid_size_x = u32::try_from(info.query_count * 16).unwrap();
            info.start_position = start;
            info.committed_kv_length = committed;
            info.q_heads = 16;
            info.kv_heads = 4;
            info.head_dim = 256;
            info.scale_denominator = 0;
            let window = sliding_window.unwrap_or(0);
            let retained_start = if window == 0 {
                0
            } else {
                committed.saturating_sub(window)
            };
            info.reserved = [
                window as u32,
                (window >> 32) as u32,
                retained_start as u32,
                (retained_start >> 32) as u32,
                1.0_f32.to_bits(),
                1,
                0,
                0,
            ];
            if sliding_window.is_some() {
                set_test_c_string(
                    &mut info.kernel_symbol,
                    "causal_attention.sliding_static_fp8_gqa.v1",
                );
                set_test_c_string(
                    &mut info.device_symbol,
                    "sllm_causal_attention_sliding_static_fp8_gqa_v1",
                );
            } else {
                set_test_c_string(
                    &mut info.kernel_symbol,
                    "causal_attention.scaled_static_fp8_gqa.v1",
                );
                set_test_c_string(
                    &mut info.device_symbol,
                    "sllm_causal_attention_scaled_static_fp8_gqa_v1",
                );
            }
            let evidence = validate_causal_attention_info(
                &info,
                &context,
                start,
                committed,
                descriptor,
                16,
                sliding_window,
                Some(1.0),
            )
            .unwrap();
            assert_eq!(evidence.score_scale_bits, 1.0_f32.to_bits());
            assert!(evidence.explicit_score_scale);
            assert!(!evidence.fallback_allowed);
            assert!(!evidence.fallback_used);

            let mut fallback = info;
            fallback.fallback_used = 1;
            assert!(
                validate_causal_attention_info(
                    &fallback,
                    &context,
                    start,
                    committed,
                    descriptor,
                    16,
                    sliding_window,
                    Some(1.0),
                )
                .is_err()
            );
            let mut wrong_scale = info;
            wrong_scale.reserved[4] = (1.0_f32 / 16.0).to_bits();
            assert!(
                validate_causal_attention_info(
                    &wrong_scale,
                    &context,
                    start,
                    committed,
                    descriptor,
                    16,
                    sliding_window,
                    Some(1.0),
                )
                .is_err()
            );
        }
    }

    #[test]
    fn standard_mxfp8_e4_native_storage_supports_initial_amd_targets() {
        let e4 = KvStateDescriptor::new_with_kv_mxfp8(
            0,
            17,
            4,
            257,
            KvCacheEncoding::Mxfp8E4,
            KvFp8PhysicalVariant::OcpE4M3Fn,
        )
        .unwrap();
        let e5 = KvStateDescriptor::new_with_kv_mxfp8(
            0,
            17,
            4,
            257,
            KvCacheEncoding::Mxfp8E5,
            KvFp8PhysicalVariant::OcpE5M2,
        )
        .unwrap();
        for target in ["gfx1030", "gfx1201", "gfx942:sramecc+:xnack-"] {
            assert_eq!(
                native_kv_storage(e4, Some(target)).unwrap().0,
                sys::SLLM_TENSOR_DTYPE_F8_E4M3_FN
            );
        }
        assert_eq!(
            native_kv_storage(e5, Some("gfx1030")).unwrap().0,
            sys::SLLM_TENSOR_DTYPE_F8_E5M2
        );
        for (descriptor, target) in [
            (e4, "unknown"),
            (e5, "gfx1201"),
            (e5, "gfx942:sramecc+:xnack-"),
        ] {
            let error = native_kv_storage(descriptor, Some(target)).unwrap_err();
            assert_eq!(error.status(), RuntimeStatus::InvalidKvStateDescriptor);
            assert!(error.message().contains("standard OCP MXFP8"));
        }
    }

    #[test]
    fn long_rdna_and_gfx942_use_only_the_fixed_contiguous_provider() {
        assert_eq!(
            selected_memory_kind_for_target(Some("gfx942"), 1),
            sys::SLLM_HIP_KV_MEMORY_KIND_CONTIGUOUS_RESIDENT
        );
        assert_eq!(
            selected_memory_kind_for_target(Some("gfx1030"), 65_535),
            sys::SLLM_HIP_KV_MEMORY_KIND_CAPABILITY_SELECTED
        );
        assert_eq!(
            selected_memory_kind_for_target(Some("gfx1030"), 65_536),
            sys::SLLM_HIP_KV_MEMORY_KIND_CONTIGUOUS_RESIDENT
        );
        assert_eq!(
            selected_memory_kind_for_target(Some("gfx1030"), 65_537),
            sys::SLLM_HIP_KV_MEMORY_KIND_CONTIGUOUS_RESIDENT
        );
        assert_eq!(
            selected_memory_kind_for_target(Some("gfx1201"), 65_535),
            sys::SLLM_HIP_KV_MEMORY_KIND_CAPABILITY_SELECTED
        );
        assert_eq!(
            selected_memory_kind_for_target(Some("gfx1201"), 65_536),
            sys::SLLM_HIP_KV_MEMORY_KIND_CONTIGUOUS_RESIDENT
        );
        assert_eq!(
            selected_memory_kind_for_target(Some("gfx1201"), 65_537),
            sys::SLLM_HIP_KV_MEMORY_KIND_CONTIGUOUS_RESIDENT
        );
        assert_eq!(
            selected_memory_kind_for_target(None, 131_072),
            sys::SLLM_HIP_KV_MEMORY_KIND_CAPABILITY_SELECTED
        );
    }

    #[test]
    fn phase50_gfx942_never_selects_rdna_causal_candidates() {
        let opt_ins = [
            None,
            Some(std::ffi::OsStr::new("1")),
            Some(std::ffi::OsStr::new("0")),
            Some(std::ffi::OsStr::new("unknown")),
        ];
        for opt_in in opt_ins {
            for committed_kv_length in [4095, 4096, 4097] {
                assert!(!decode_wave_split_fp16_pair_enabled(
                    Some("gfx942"),
                    1,
                    committed_kv_length,
                    16,
                    4,
                    256,
                    KvCacheEncoding::Fp16,
                    opt_in,
                    false,
                ));
                assert!(!decode_gqa4_split_enabled(
                    Some("gfx942"),
                    1,
                    committed_kv_length,
                    16,
                    4,
                    256,
                    KvCacheEncoding::Fp16,
                    opt_in,
                    false,
                ));
                assert!(!decode_gqa4_split_p32_enabled(
                    Some("gfx942"),
                    1,
                    committed_kv_length,
                    16,
                    4,
                    256,
                    KvCacheEncoding::Fp16,
                    opt_in,
                    false,
                ));
            }
            for committed_kv_length in [31, 32, 33, 1023] {
                assert!(!decode_wave_split_short_enabled(
                    Some("gfx942"),
                    1,
                    committed_kv_length,
                    16,
                    4,
                    256,
                    KvCacheEncoding::Fp16,
                    opt_in,
                ));
            }
            assert!(!decode_wave_split_q_preload_enabled(
                Some("gfx942"),
                true,
                opt_in,
            ));
            assert!(!decode_wave_split_q_preload_enabled(
                Some("gfx1201"),
                true,
                opt_in,
            ));
            for query_count in [1023, 1024, 1025, 4096, 10_001] {
                assert!(!scaled_prefill_gemm_enabled(
                    Some("gfx942"),
                    query_count,
                    16,
                    4,
                    256,
                    KvCacheEncoding::Fp16,
                    opt_in,
                    false,
                ));
                assert!(!long_prefill_v2_enabled(
                    Some("gfx942"),
                    query_count,
                    16,
                    4,
                    256,
                    KvCacheEncoding::Fp16,
                    opt_in,
                    false,
                ));
            }
        }

        // A force-baseline request must remain safe even if a candidate
        // opt-in is present; gfx942 is rejected before this fallback branch.
        assert!(!decode_wave_split_fp16_pair_enabled(
            Some("gfx942"),
            1,
            4096,
            16,
            4,
            256,
            KvCacheEncoding::Fp16,
            Some(std::ffi::OsStr::new("1")),
            true,
        ));
        assert!(!decode_gqa4_split_enabled(
            Some("gfx942"),
            1,
            4096,
            16,
            4,
            256,
            KvCacheEncoding::Fp16,
            Some(std::ffi::OsStr::new("1")),
            true,
        ));
        assert!(!decode_gqa4_split_p32_enabled(
            Some("gfx942"),
            1,
            4096,
            16,
            4,
            256,
            KvCacheEncoding::Fp16,
            Some(std::ffi::OsStr::new("1")),
            true,
        ));
        assert!(!scaled_prefill_gemm_enabled(
            Some("gfx942"),
            1024,
            16,
            4,
            256,
            KvCacheEncoding::Fp16,
            Some(std::ffi::OsStr::new("1")),
            true,
        ));
        assert!(!long_prefill_v2_enabled(
            Some("gfx942"),
            1024,
            16,
            4,
            256,
            KvCacheEncoding::Fp16,
            Some(std::ffi::OsStr::new("1")),
            true,
        ));
    }

    #[test]
    fn decode_q_preload_guard_defaults_on_and_accepts_only_explicit_disable() {
        assert!(decode_wave_split_q_preload_enabled(
            Some("gfx1030"),
            true,
            None
        ));
        assert!(decode_wave_split_q_preload_enabled(
            Some("gfx1030"),
            true,
            Some(std::ffi::OsStr::new("1"))
        ));
        assert!(!decode_wave_split_q_preload_enabled(
            Some("gfx1030"),
            true,
            Some(std::ffi::OsStr::new("0"))
        ));
        assert!(!decode_wave_split_q_preload_enabled(
            Some("gfx1030"),
            true,
            Some(std::ffi::OsStr::new("invalid"))
        ));
        assert!(!decode_wave_split_q_preload_enabled(
            Some("gfx1201"),
            true,
            None
        ));
        assert!(!decode_wave_split_q_preload_enabled(
            Some("gfx1030"),
            false,
            None
        ));
    }

    #[test]
    fn decode_fp16_pair_guard_defaults_on_for_long_gfx1030_shape_and_force_safe() {
        let enabled = Some(std::ffi::OsStr::new("1"));
        assert!(decode_wave_split_fp16_pair_enabled(
            Some("gfx1030"),
            1,
            1024,
            16,
            4,
            256,
            KvCacheEncoding::Fp16,
            enabled,
            false,
        ));
        assert!(decode_wave_split_fp16_pair_enabled(
            Some("gfx1030"),
            1,
            100_000,
            16,
            4,
            256,
            KvCacheEncoding::Fp16,
            enabled,
            false,
        ));
        for (committed_kv_length, query_count) in [(1023, 1), (1024, 2)] {
            assert!(!decode_wave_split_fp16_pair_enabled(
                Some("gfx1030"),
                query_count,
                committed_kv_length,
                16,
                4,
                256,
                KvCacheEncoding::Fp16,
                enabled,
                false,
            ));
        }
        assert!(decode_wave_split_fp16_pair_enabled(
            Some("gfx1030"),
            1,
            1024,
            16,
            4,
            256,
            KvCacheEncoding::Fp16,
            None,
            false,
        ));
        for opt_in in [
            Some(std::ffi::OsStr::new("0")),
            Some(std::ffi::OsStr::new("unknown")),
        ] {
            assert!(!decode_wave_split_fp16_pair_enabled(
                Some("gfx1030"),
                1,
                1024,
                16,
                4,
                256,
                KvCacheEncoding::Fp16,
                opt_in,
                false,
            ));
        }
        assert!(!decode_wave_split_fp16_pair_enabled(
            Some("gfx1201"),
            1,
            1024,
            16,
            4,
            256,
            KvCacheEncoding::Fp16,
            enabled,
            false,
        ));
        for (query_heads, kv_heads, head_dim, encoding) in [
            (8, 4, 256, KvCacheEncoding::Fp16),
            (16, 8, 256, KvCacheEncoding::Fp16),
            (16, 4, 128, KvCacheEncoding::Fp16),
            (16, 4, 256, KvCacheEncoding::Fp8E4M3Fn),
        ] {
            assert!(!decode_wave_split_fp16_pair_enabled(
                Some("gfx1030"),
                1,
                1024,
                query_heads,
                kv_heads,
                head_dim,
                encoding,
                enabled,
                false,
            ));
        }
        assert!(!decode_wave_split_fp16_pair_enabled(
            Some("gfx1030"),
            1,
            1024,
            16,
            4,
            256,
            KvCacheEncoding::Fp16,
            enabled,
            true,
        ));
    }

    #[test]
    fn decode_gqa4_split_guard_requires_exact_opt_in_and_force_safe() {
        let enabled = Some(std::ffi::OsStr::new("1"));
        assert!(decode_gqa4_split_enabled(
            Some("gfx1030"),
            1,
            4096,
            16,
            4,
            256,
            KvCacheEncoding::Fp16,
            enabled,
            false,
        ));
        assert!(!decode_gqa4_split_enabled(
            Some("gfx1030"),
            1,
            4096,
            16,
            4,
            256,
            KvCacheEncoding::Fp16,
            None,
            false,
        ));
        assert!(!decode_gqa4_split_enabled(
            Some("gfx1030"),
            1,
            4096,
            16,
            4,
            256,
            KvCacheEncoding::Fp16,
            Some(std::ffi::OsStr::new("0")),
            false,
        ));
        assert!(!decode_gqa4_split_enabled(
            Some("gfx1030"),
            1,
            4096,
            16,
            4,
            256,
            KvCacheEncoding::Fp16,
            enabled,
            true,
        ));
        for (committed_kv_length, query_count) in [(1023, 1), (1024, 1), (1025, 1), (4095, 1)] {
            assert!(!decode_gqa4_split_enabled(
                Some("gfx1030"),
                query_count,
                committed_kv_length,
                16,
                4,
                256,
                KvCacheEncoding::Fp16,
                enabled,
                false,
            ));
        }
        for (target, query_count, committed_kv_length, query_heads, kv_heads, head_dim, encoding) in [
            (Some("gfx1201"), 1, 4096, 16, 4, 256, KvCacheEncoding::Fp16),
            (Some("gfx1030"), 2, 4096, 16, 4, 256, KvCacheEncoding::Fp16),
            (Some("gfx1030"), 1, 4096, 8, 4, 256, KvCacheEncoding::Fp16),
            (Some("gfx1030"), 1, 4096, 16, 8, 256, KvCacheEncoding::Fp16),
            (Some("gfx1030"), 1, 4096, 16, 4, 128, KvCacheEncoding::Fp16),
            (
                Some("gfx1030"),
                1,
                4096,
                16,
                4,
                256,
                KvCacheEncoding::Fp8E4M3Fn,
            ),
        ] {
            assert!(!decode_gqa4_split_enabled(
                target,
                query_count,
                committed_kv_length,
                query_heads,
                kv_heads,
                head_dim,
                encoding,
                enabled,
                false,
            ));
        }
    }

    #[test]
    fn decode_gqa4_split_partition_guards_share_shape_and_threshold_contract() {
        let enabled = Some(std::ffi::OsStr::new("1"));
        for opt_in in [enabled, Some(std::ffi::OsStr::new("0")), None] {
            assert_eq!(
                decode_gqa4_split_enabled(
                    Some("gfx1030"),
                    1,
                    4096,
                    16,
                    4,
                    256,
                    KvCacheEncoding::Fp16,
                    opt_in,
                    false,
                ),
                opt_in == enabled,
            );
        }
        for committed_kv_length in [1023, 1024, 1025, 4095] {
            assert!(!decode_gqa4_split_enabled(
                Some("gfx1030"),
                1,
                committed_kv_length,
                16,
                4,
                256,
                KvCacheEncoding::Fp16,
                enabled,
                false,
            ));
        }
    }

    #[test]
    fn decode_gqa4_split_p32_guard_is_default_on_with_explicit_rollback() {
        let enabled = Some(std::ffi::OsStr::new("1"));
        let rollback = Some(std::ffi::OsStr::new("0"));
        let unknown = Some(std::ffi::OsStr::new("unexpected"));
        for opt_in in [None, enabled] {
            assert!(decode_gqa4_split_p32_enabled(
                Some("gfx1030"),
                1,
                4096,
                16,
                4,
                256,
                KvCacheEncoding::Fp16,
                opt_in,
                false,
            ));
        }
        for opt_in in [rollback, unknown] {
            assert!(!decode_gqa4_split_p32_enabled(
                Some("gfx1030"),
                1,
                4096,
                16,
                4,
                256,
                KvCacheEncoding::Fp16,
                opt_in,
                false,
            ));
        }
        assert!(!decode_gqa4_split_p32_enabled(
            Some("gfx1030"),
            1,
            4096,
            16,
            4,
            256,
            KvCacheEncoding::Fp16,
            enabled,
            true,
        ));
        for committed_kv_length in [4095, 4096, 4097] {
            assert_eq!(
                decode_gqa4_split_p32_enabled(
                    Some("gfx1030"),
                    1,
                    committed_kv_length,
                    16,
                    4,
                    256,
                    KvCacheEncoding::Fp16,
                    None,
                    false,
                ),
                committed_kv_length >= 4096,
            );
        }
        assert!(decode_gqa4_split_p32_target_enabled(
            Some("gfx1201"),
            1,
            4096,
            16,
            4,
            256,
            KvCacheEncoding::Fp16,
            None,
            None,
            false,
        ));
        for (target, query_count, query_heads, kv_heads, head_dim, encoding) in [
            (Some("gfx942"), 1, 16, 4, 256, KvCacheEncoding::Fp16),
            (Some("gfx1030"), 2, 16, 4, 256, KvCacheEncoding::Fp16),
            (Some("gfx1030"), 1, 8, 4, 256, KvCacheEncoding::Fp16),
            (Some("gfx1030"), 1, 16, 8, 256, KvCacheEncoding::Fp16),
            (Some("gfx1030"), 1, 16, 4, 128, KvCacheEncoding::Fp16),
            (Some("gfx1030"), 1, 16, 4, 256, KvCacheEncoding::Fp8E4M3Fn),
        ] {
            assert!(!decode_gqa4_split_p32_enabled(
                target,
                query_count,
                4096,
                query_heads,
                kv_heads,
                head_dim,
                encoding,
                None,
                false,
            ));
        }
    }

    #[test]
    fn decode_gqa6_split_p32_guard_is_explicit_opt_in_target_scoped_and_force_safe() {
        let enabled = Some(std::ffi::OsStr::new("1"));
        assert!(decode_gqa6_split_p32_enabled(
            Some("gfx1030"),
            1,
            4096,
            24,
            4,
            256,
            KvCacheEncoding::Fp16,
            enabled,
            false,
        ));
        assert!(decode_gqa6_split_p32_enabled(
            Some("gfx1201"),
            1,
            8192,
            24,
            4,
            256,
            KvCacheEncoding::Fp16,
            enabled,
            false,
        ));
        for opt_in in [
            None,
            Some(std::ffi::OsStr::new("0")),
            Some(std::ffi::OsStr::new("unknown")),
        ] {
            assert!(!decode_gqa6_split_p32_enabled(
                Some("gfx1030"),
                1,
                4096,
                24,
                4,
                256,
                KvCacheEncoding::Fp16,
                opt_in,
                false,
            ));
        }
        assert!(!decode_gqa6_split_p32_enabled(
            Some("gfx1030"),
            1,
            4096,
            24,
            4,
            256,
            KvCacheEncoding::Fp16,
            enabled,
            true,
        ));
        for (target, query_count, committed_kv_length, query_heads, kv_heads, head_dim, encoding) in [
            (Some("gfx942"), 1, 4096, 24, 4, 256, KvCacheEncoding::Fp16),
            (Some("gfx1030"), 2, 4096, 24, 4, 256, KvCacheEncoding::Fp16),
            (Some("gfx1030"), 1, 4095, 24, 4, 256, KvCacheEncoding::Fp16),
            (Some("gfx1030"), 1, 4096, 16, 4, 256, KvCacheEncoding::Fp16),
            (Some("gfx1030"), 1, 4096, 24, 8, 256, KvCacheEncoding::Fp16),
            (Some("gfx1030"), 1, 4096, 24, 4, 128, KvCacheEncoding::Fp16),
            (
                Some("gfx1030"),
                1,
                4096,
                24,
                4,
                256,
                KvCacheEncoding::Fp8E4M3Fn,
            ),
        ] {
            assert!(!decode_gqa6_split_p32_enabled(
                target,
                query_count,
                committed_kv_length,
                query_heads,
                kv_heads,
                head_dim,
                encoding,
                enabled,
                false,
            ));
        }
    }

    #[test]
    fn decode_gqa6_split_p64_guard_is_explicit_opt_in_target_scoped_and_force_safe() {
        let enabled = Some(std::ffi::OsStr::new("1"));
        assert!(decode_gqa6_split_p64_enabled(
            Some("gfx1030"),
            1,
            8192,
            24,
            4,
            256,
            KvCacheEncoding::Fp16,
            enabled,
            false,
        ));
        assert!(decode_gqa6_split_p64_enabled(
            Some("gfx1201"),
            1,
            8192,
            24,
            4,
            256,
            KvCacheEncoding::Fp16,
            enabled,
            false,
        ));
        assert!(!decode_gqa6_split_p64_enabled(
            Some("gfx1030"),
            1,
            8191,
            24,
            4,
            256,
            KvCacheEncoding::Fp16,
            enabled,
            false,
        ));
        for opt_in in [
            None,
            Some(std::ffi::OsStr::new("0")),
            Some(std::ffi::OsStr::new("unknown")),
        ] {
            assert!(!decode_gqa6_split_p64_enabled(
                Some("gfx1030"),
                1,
                4096,
                24,
                4,
                256,
                KvCacheEncoding::Fp16,
                opt_in,
                false,
            ));
        }
        assert!(!decode_gqa6_split_p64_enabled(
            Some("gfx1030"),
            1,
            4096,
            24,
            4,
            256,
            KvCacheEncoding::Fp16,
            enabled,
            true,
        ));
        for (target, query_count, committed_kv_length, query_heads, kv_heads, head_dim, encoding) in [
            (Some("gfx942"), 1, 4096, 24, 4, 256, KvCacheEncoding::Fp16),
            (Some("gfx1030"), 2, 4096, 24, 4, 256, KvCacheEncoding::Fp16),
            (Some("gfx1030"), 1, 4095, 24, 4, 256, KvCacheEncoding::Fp16),
            (Some("gfx1030"), 1, 4096, 16, 4, 256, KvCacheEncoding::Fp16),
            (Some("gfx1030"), 1, 4096, 24, 8, 256, KvCacheEncoding::Fp16),
            (Some("gfx1030"), 1, 4096, 24, 4, 128, KvCacheEncoding::Fp16),
            (
                Some("gfx1030"),
                1,
                4096,
                24,
                4,
                256,
                KvCacheEncoding::Fp8E4M3Fn,
            ),
        ] {
            assert!(!decode_gqa6_split_p64_enabled(
                target,
                query_count,
                committed_kv_length,
                query_heads,
                kv_heads,
                head_dim,
                encoding,
                enabled,
                false,
            ));
        }
        let p32 = decode_gqa6_split_p32_enabled(
            Some("gfx1030"),
            1,
            4096,
            24,
            4,
            256,
            KvCacheEncoding::Fp16,
            enabled,
            false,
        );
        let p64 = decode_gqa6_split_p64_enabled(
            Some("gfx1030"),
            1,
            8192,
            24,
            4,
            256,
            KvCacheEncoding::Fp16,
            enabled,
            false,
        );
        assert!(p32);
        assert!(p64);
        assert!(!(p32 && !p64));
    }

    #[test]
    fn decode_gqa6_split_p128_guard_is_gfx1030_long_context_opt_in_only() {
        let enabled = Some(std::ffi::OsStr::new("1"));
        assert!(decode_gqa6_split_p128_enabled(
            Some("gfx1030"),
            1,
            8192,
            24,
            4,
            256,
            KvCacheEncoding::Fp16,
            enabled,
            false,
        ));
        for (target, length, query_count, query_heads, kv_heads, head_dim, encoding) in [
            (Some("gfx1201"), 8192, 1, 24, 4, 256, KvCacheEncoding::Fp16),
            (Some("gfx1030"), 8191, 1, 24, 4, 256, KvCacheEncoding::Fp16),
            (Some("gfx1030"), 8192, 2, 24, 4, 256, KvCacheEncoding::Fp16),
            (Some("gfx1030"), 8192, 1, 16, 4, 256, KvCacheEncoding::Fp16),
            (Some("gfx1030"), 8192, 1, 24, 8, 256, KvCacheEncoding::Fp16),
            (Some("gfx1030"), 8192, 1, 24, 4, 128, KvCacheEncoding::Fp16),
            (
                Some("gfx1030"),
                8192,
                1,
                24,
                4,
                256,
                KvCacheEncoding::Fp8E4M3Fn,
            ),
        ] {
            assert!(!decode_gqa6_split_p128_enabled(
                target,
                query_count,
                length,
                query_heads,
                kv_heads,
                head_dim,
                encoding,
                enabled,
                false,
            ));
        }
        for opt_in in [
            None,
            Some(std::ffi::OsStr::new("0")),
            Some(std::ffi::OsStr::new("unknown")),
        ] {
            assert!(!decode_gqa6_split_p128_enabled(
                Some("gfx1030"),
                1,
                8192,
                24,
                4,
                256,
                KvCacheEncoding::Fp16,
                opt_in,
                false,
            ));
        }
        assert!(!decode_gqa6_split_p128_enabled(
            Some("gfx1030"),
            1,
            8192,
            24,
            4,
            256,
            KvCacheEncoding::Fp16,
            enabled,
            true,
        ));
    }

    #[test]
    fn decode_gqa4_split_p32_gfx1201_is_default_on_and_target_scoped() {
        let enabled = Some(std::ffi::OsStr::new("1"));
        let disabled = Some(std::ffi::OsStr::new("0"));
        let unknown = Some(std::ffi::OsStr::new("unknown"));
        let select = |target,
                      committed_kv_length,
                      query_count,
                      query_heads,
                      kv_heads,
                      head_dim,
                      encoding,
                      gfx1030_opt_in,
                      gfx1201_opt_in,
                      force_baseline| {
            decode_gqa4_split_p32_target_enabled(
                target,
                query_count,
                committed_kv_length,
                query_heads,
                kv_heads,
                head_dim,
                encoding,
                gfx1030_opt_in,
                gfx1201_opt_in,
                force_baseline,
            )
        };

        assert!(select(
            Some("gfx1201"),
            4096,
            1,
            16,
            4,
            256,
            KvCacheEncoding::Fp16,
            None,
            None,
            false,
        ));
        for opt_in in [disabled, unknown] {
            assert!(!select(
                Some("gfx1201"),
                4096,
                1,
                16,
                4,
                256,
                KvCacheEncoding::Fp16,
                enabled,
                opt_in,
                false,
            ));
        }
        for committed_kv_length in [4095, 4096, 4097] {
            assert_eq!(
                select(
                    Some("gfx1201"),
                    committed_kv_length,
                    1,
                    16,
                    4,
                    256,
                    KvCacheEncoding::Fp16,
                    None,
                    enabled,
                    false,
                ),
                committed_kv_length >= 4096,
            );
        }
        assert!(!select(
            Some("gfx1201"),
            4096,
            1,
            16,
            4,
            256,
            KvCacheEncoding::Fp16,
            None,
            enabled,
            true,
        ));
        for (query_count, query_heads, kv_heads, head_dim, encoding) in [
            (2, 16, 4, 256, KvCacheEncoding::Fp16),
            (1, 8, 4, 256, KvCacheEncoding::Fp16),
            (1, 16, 8, 256, KvCacheEncoding::Fp16),
            (1, 16, 4, 128, KvCacheEncoding::Fp16),
            (1, 16, 4, 256, KvCacheEncoding::Fp8E4M3Fn),
        ] {
            assert!(!select(
                Some("gfx1201"),
                4096,
                query_count,
                query_heads,
                kv_heads,
                head_dim,
                encoding,
                None,
                enabled,
                false,
            ));
        }
        assert!(select(
            Some("gfx1201"),
            4096,
            1,
            16,
            4,
            256,
            KvCacheEncoding::Fp16,
            enabled,
            None,
            false,
        ));
        assert!(!select(
            Some("gfx942"),
            4096,
            1,
            16,
            4,
            256,
            KvCacheEncoding::Fp16,
            None,
            enabled,
            false,
        ));
        assert!(!select(
            Some("unknown"),
            4096,
            1,
            16,
            4,
            256,
            KvCacheEncoding::Fp16,
            None,
            enabled,
            false,
        ));
    }

    #[test]
    fn decode_short_wave_guard_is_exact_target_shape_encoding_and_default_on() {
        let enabled = Some(std::ffi::OsStr::new("1"));
        assert!(decode_wave_split_short_enabled(
            Some("gfx1030"),
            1,
            32,
            16,
            4,
            256,
            KvCacheEncoding::Fp16,
            enabled,
        ));
        assert!(decode_wave_split_short_enabled(
            Some("gfx1030"),
            1,
            128,
            16,
            4,
            256,
            KvCacheEncoding::Fp16,
            None,
        ));
        assert!(decode_wave_split_short_enabled(
            Some("gfx1030"),
            1,
            1023,
            16,
            4,
            256,
            KvCacheEncoding::Fp16,
            enabled,
        ));
        for (query_count, committed_kv_length) in [(1, 31), (1, 1024), (2, 128)] {
            assert!(!decode_wave_split_short_enabled(
                Some("gfx1030"),
                query_count,
                committed_kv_length,
                16,
                4,
                256,
                KvCacheEncoding::Fp16,
                enabled,
            ));
        }
        assert!(!decode_wave_split_short_enabled(
            Some("gfx1201"),
            1,
            128,
            16,
            4,
            256,
            KvCacheEncoding::Fp16,
            enabled,
        ));
        assert!(!decode_wave_split_short_enabled(
            Some("gfx1030"),
            1,
            128,
            16,
            4,
            256,
            KvCacheEncoding::Fp8E4M3Fn,
            enabled,
        ));
        assert!(!decode_wave_split_short_enabled(
            Some("gfx1030"),
            1,
            128,
            8,
            4,
            256,
            KvCacheEncoding::Fp16,
            enabled,
        ));
        assert!(!decode_wave_split_short_enabled(
            Some("gfx1030"),
            1,
            128,
            16,
            4,
            256,
            KvCacheEncoding::Fp16,
            Some(std::ffi::OsStr::new("0")),
        ));
        assert!(!decode_wave_split_short_enabled(
            Some("gfx1030"),
            1,
            128,
            16,
            4,
            256,
            KvCacheEncoding::Fp16,
            Some(std::ffi::OsStr::new("unknown")),
        ));
        assert!(!decode_wave_split_short_q_preload_enabled(
            false,
            Some(std::ffi::OsStr::new("1")),
        ));
        assert!(!decode_wave_split_short_q_preload_enabled(
            true,
            Some(std::ffi::OsStr::new("0")),
        ));
        assert!(decode_wave_split_short_q_preload_enabled(true, None));
        assert!(!decode_wave_split_short_q_preload_enabled(
            true,
            Some(std::ffi::OsStr::new("unknown")),
        ));
        assert!(decode_wave_split_short_q_preload_enabled(
            true,
            Some(std::ffi::OsStr::new("1")),
        ));
    }

    #[test]
    fn scaled_prefill_gemm_guard_keeps_fp16_default_on_and_mxfp8_explicit() {
        let enabled = Some(std::ffi::OsStr::new("1"));
        assert!(scaled_prefill_gemm_enabled(
            Some("gfx1030"),
            1024,
            16,
            4,
            256,
            KvCacheEncoding::Fp16,
            enabled,
            false,
        ));
        assert!(scaled_prefill_gemm_enabled(
            Some("gfx1030"),
            1024,
            16,
            4,
            256,
            KvCacheEncoding::Mxfp8E4,
            enabled,
            false,
        ));
        assert!(!scaled_prefill_gemm_enabled(
            Some("gfx1030"),
            1023,
            16,
            4,
            256,
            KvCacheEncoding::Fp16,
            enabled,
            false,
        ));
        for query_count in [1024, 1025, 4096, 10001, 100000] {
            assert!(scaled_prefill_gemm_enabled(
                Some("gfx1030"),
                query_count,
                16,
                4,
                256,
                KvCacheEncoding::Fp16,
                enabled,
                false,
            ));
        }
        assert!(scaled_prefill_gemm_enabled(
            Some("gfx1030"),
            1024,
            16,
            4,
            256,
            KvCacheEncoding::Fp16,
            None,
            false,
        ));
        assert!(!scaled_prefill_gemm_enabled(
            Some("gfx1030"),
            1024,
            16,
            4,
            256,
            KvCacheEncoding::Mxfp8E4,
            None,
            false,
        ));
        for value in ["0", "unknown"] {
            assert!(!scaled_prefill_gemm_enabled(
                Some("gfx1030"),
                1024,
                16,
                4,
                256,
                KvCacheEncoding::Fp16,
                Some(std::ffi::OsStr::new(value)),
                false,
            ));
        }
        for (target, query_heads, kv_heads, head_dim, encoding) in [
            (Some("gfx1201"), 16, 4, 256, KvCacheEncoding::Fp16),
            (Some("gfx1030"), 8, 4, 256, KvCacheEncoding::Fp16),
            (Some("gfx1030"), 16, 8, 256, KvCacheEncoding::Fp16),
            (Some("gfx1030"), 16, 4, 128, KvCacheEncoding::Fp16),
            (Some("gfx1030"), 16, 4, 256, KvCacheEncoding::Fp8E4M3Fn),
        ] {
            assert!(!scaled_prefill_gemm_enabled(
                target,
                1024,
                query_heads,
                kv_heads,
                head_dim,
                encoding,
                enabled,
                false,
            ));
        }
        assert!(!scaled_prefill_gemm_enabled(
            Some("gfx1030"),
            1024,
            16,
            4,
            256,
            KvCacheEncoding::Fp16,
            enabled,
            true,
        ));
    }

    #[test]
    fn long_prefill_v2_guard_is_explicit_and_matches_native_shape() {
        let enabled = Some(std::ffi::OsStr::new("1"));
        for query_count in [1024, 4096, 10_001, 100_000] {
            assert!(long_prefill_v2_enabled(
                Some("gfx1030"),
                query_count,
                16,
                4,
                256,
                KvCacheEncoding::Fp16,
                enabled,
                false,
            ));
        }
        assert!(!long_prefill_v2_enabled(
            Some("gfx1030"),
            1024,
            16,
            4,
            256,
            KvCacheEncoding::Fp16,
            None,
            false,
        ));
        assert!(!long_prefill_v2_enabled(
            Some("gfx1030"),
            1024,
            16,
            4,
            256,
            KvCacheEncoding::Fp16,
            enabled,
            true,
        ));
        for (target, query_count, query_heads, kv_heads, head_dim, encoding) in [
            (Some("gfx1201"), 1024, 16, 4, 256, KvCacheEncoding::Fp16),
            (Some("gfx1030"), 1023, 16, 4, 256, KvCacheEncoding::Fp16),
            (Some("gfx1030"), 1024, 8, 4, 256, KvCacheEncoding::Fp16),
            (Some("gfx1030"), 1024, 16, 8, 256, KvCacheEncoding::Fp16),
            (Some("gfx1030"), 1024, 16, 4, 128, KvCacheEncoding::Fp16),
            (
                Some("gfx1030"),
                1024,
                16,
                4,
                256,
                KvCacheEncoding::Fp8E4M3Fn,
            ),
        ] {
            assert!(!long_prefill_v2_enabled(
                target,
                query_count,
                query_heads,
                kv_heads,
                head_dim,
                encoding,
                enabled,
                false,
            ));
        }
    }

    #[test]
    fn gqa6_blocksoftmax_guard_is_target_scoped_and_force_safe() {
        let enabled = Some(std::ffi::OsStr::new("1"));
        let disabled = Some(std::ffi::OsStr::new("0"));
        assert!(gqa6_blocksoftmax_enabled(
            Some("gfx1030"),
            128,
            24,
            4,
            256,
            KvCacheEncoding::Fp16,
            enabled,
            None,
            false,
        ));
        assert!(gqa6_blocksoftmax_enabled(
            Some("gfx1201"),
            129,
            24,
            4,
            256,
            KvCacheEncoding::Fp16,
            None,
            enabled,
            false,
        ));
        assert!(!gqa6_blocksoftmax_enabled(
            Some("gfx1030"),
            128,
            24,
            4,
            256,
            KvCacheEncoding::Fp16,
            enabled,
            None,
            true,
        ));
        for (target, gfx1030, gfx1201) in [
            (Some("gfx942"), enabled, enabled),
            (Some("gfx1030"), disabled, enabled),
            (Some("gfx1201"), enabled, disabled),
        ] {
            assert!(!gqa6_blocksoftmax_enabled(
                target,
                128,
                24,
                4,
                256,
                KvCacheEncoding::Fp16,
                gfx1030,
                gfx1201,
                false,
            ));
        }
        for (query_count, query_heads, kv_heads, head_dim, encoding) in [
            (127, 24, 4, 256, KvCacheEncoding::Fp16),
            (128, 16, 4, 256, KvCacheEncoding::Fp16),
            (128, 24, 8, 256, KvCacheEncoding::Fp16),
            (128, 24, 4, 128, KvCacheEncoding::Fp16),
            (128, 24, 4, 256, KvCacheEncoding::Mxfp8E4),
        ] {
            assert!(!gqa6_blocksoftmax_enabled(
                Some("gfx1030"),
                query_count,
                query_heads,
                kv_heads,
                head_dim,
                encoding,
                enabled,
                None,
                false,
            ));
        }
    }

    #[test]
    fn gqa6_blocksoftmax_q8_guard_is_gfx1201_only_and_exact() {
        let enabled = Some(std::ffi::OsStr::new("1"));
        assert!(gqa6_blocksoftmax_q8_enabled(
            Some("gfx1201"),
            128,
            24,
            4,
            256,
            KvCacheEncoding::Fp16,
            enabled,
            false,
        ));
        assert!(!gqa6_blocksoftmax_q8_enabled(
            Some("gfx1030"),
            128,
            24,
            4,
            256,
            KvCacheEncoding::Fp16,
            enabled,
            false,
        ));
        assert!(!gqa6_blocksoftmax_q8_enabled(
            Some("gfx1201"),
            128,
            24,
            4,
            256,
            KvCacheEncoding::Fp16,
            enabled,
            true,
        ));
        for (query_count, query_heads, kv_heads, head_dim, encoding) in [
            (127, 24, 4, 256, KvCacheEncoding::Fp16),
            (128, 16, 4, 256, KvCacheEncoding::Fp16),
            (128, 24, 8, 256, KvCacheEncoding::Fp16),
            (128, 24, 4, 128, KvCacheEncoding::Fp16),
            (128, 24, 4, 256, KvCacheEncoding::Mxfp8E4),
        ] {
            assert!(!gqa6_blocksoftmax_q8_enabled(
                Some("gfx1201"),
                query_count,
                query_heads,
                kv_heads,
                head_dim,
                encoding,
                enabled,
                false,
            ));
        }
    }

    #[test]
    fn gqa6_qtile4_k32_fp16_guard_is_exact_and_force_rollback() {
        let enabled = Some(std::ffi::OsStr::new("1"));
        for target in [Some("gfx1030"), Some("gfx1201")] {
            for query_count in [128, 129, 4096, 9435] {
                assert!(gqa6_qtile4_k32_fp16_enabled(
                    target,
                    query_count,
                    24,
                    4,
                    256,
                    KvCacheEncoding::Fp16,
                    enabled,
                    false,
                ));
            }
        }
        assert!(!gqa6_qtile4_k32_fp16_enabled(
            Some("gfx1030"),
            127,
            24,
            4,
            256,
            KvCacheEncoding::Fp16,
            enabled,
            false,
        ));
        for (target, query_heads, kv_heads, head_dim, encoding) in [
            (Some("gfx942"), 24, 4, 256, KvCacheEncoding::Fp16),
            (Some("gfx1030"), 16, 4, 256, KvCacheEncoding::Fp16),
            (Some("gfx1030"), 24, 8, 256, KvCacheEncoding::Fp16),
            (Some("gfx1030"), 24, 4, 128, KvCacheEncoding::Fp16),
            (Some("gfx1030"), 24, 4, 256, KvCacheEncoding::Mxfp8E4),
        ] {
            assert!(!gqa6_qtile4_k32_fp16_enabled(
                target,
                128,
                query_heads,
                kv_heads,
                head_dim,
                encoding,
                enabled,
                false,
            ));
        }
        for opt_in in [
            None,
            Some(std::ffi::OsStr::new("0")),
            Some(std::ffi::OsStr::new("unknown")),
        ] {
            assert!(!gqa6_qtile4_k32_fp16_enabled(
                Some("gfx1201"),
                128,
                24,
                4,
                256,
                KvCacheEncoding::Fp16,
                opt_in,
                false,
            ));
        }
        assert!(!gqa6_qtile4_k32_fp16_enabled(
            Some("gfx1030"),
            9435,
            24,
            4,
            256,
            KvCacheEncoding::Fp16,
            enabled,
            true,
        ));
    }

    #[test]
    fn gqa6_qtile4_fp16_key_tile_precedence_is_explicit() {
        let enabled = Some(std::ffi::OsStr::new("1"));
        let disabled = Some(std::ffi::OsStr::new("0"));
        for target in [Some("gfx1030"), Some("gfx1201")] {
            for (key_tile, opt_ins) in [
                (4, [enabled, None, None, None]),
                (8, [None, enabled, None, None]),
                (16, [None, None, enabled, None]),
                (32, [None, None, None, enabled]),
            ] {
                assert_eq!(
                    gqa6_qtile4_fp16_key_tile_enabled(
                        target,
                        128,
                        24,
                        4,
                        256,
                        KvCacheEncoding::Fp16,
                        opt_ins,
                        false,
                    ),
                    Some(key_tile)
                );
            }
        }
        assert_eq!(
            gqa6_qtile4_fp16_key_tile_enabled(
                Some("gfx1030"),
                128,
                24,
                4,
                256,
                KvCacheEncoding::Fp16,
                [enabled, enabled, enabled, enabled],
                false,
            ),
            Some(4)
        );
        assert_eq!(
            gqa6_qtile4_fp16_key_tile_enabled(
                Some("gfx1030"),
                128,
                24,
                4,
                256,
                KvCacheEncoding::Fp16,
                [disabled, disabled, disabled, disabled],
                false,
            ),
            None
        );
        assert_eq!(
            gqa6_qtile4_fp16_key_tile_enabled(
                Some("gfx1030"),
                128,
                24,
                4,
                256,
                KvCacheEncoding::Fp16,
                [enabled, enabled, enabled, enabled],
                true,
            ),
            None
        );
        assert_eq!(
            gqa6_qtile4_fp16_key_tile_enabled(
                Some("gfx942"),
                128,
                24,
                4,
                256,
                KvCacheEncoding::Fp16,
                [enabled, enabled, enabled, enabled],
                false,
            ),
            None
        );
    }

    #[test]
    fn fp16_oracle_covers_special_and_rounding_cases() {
        assert_eq!(bf16_to_f16_bits(0x0000), 0x0000);
        assert_eq!(bf16_to_f16_bits(0x8000), 0x8000);
        assert_eq!(bf16_to_f16_bits(0x0001), 0x0000);
        assert_eq!(bf16_to_f16_bits(0x3880), 0x0400);
        assert_eq!(bf16_to_f16_bits(0x7f80), 0x7c00);
        assert_eq!(bf16_to_f16_bits(0xff80), 0xfc00);
        assert_eq!(bf16_to_f16_bits(0x7fc1), 0x7e00);
        assert_eq!(bf16_to_f16_bits(0x3f80), 0x3c00);
        assert_eq!(bf16_to_f16_bits(0x3f81), 0x3c08);
        assert_eq!(bf16_to_f16_bits(0x477f), 0x7bf8);
        assert_eq!(bf16_to_f16_bits(0x4780), 0x7c00);
    }

    #[test]
    fn placement_is_token_major_and_rejects_boundaries() {
        assert_eq!(
            expected_storage_offset(257, 17, 3, 0, 0),
            Some(20 * 4 * 256)
        );
        assert_eq!(
            expected_storage_offset(257, 17, 3, 1, 255),
            Some(20 * 4 * 256 + 256 + 255)
        );
        assert_eq!(expected_storage_offset(257, 257, 0, 0, 0), None);
        assert_eq!(expected_storage_offset(257, 0, 0, 4, 0), None);
    }

    #[test]
    fn direct_native_owners_are_not_send_or_sync() {
        static_assertions::assert_not_impl_any!(NativeKvStateOwner: Send, Sync);
        static_assertions::assert_not_impl_any!(NativeKvSnapshotOwner: Send, Sync);
        static_assertions::assert_impl_all!(KvStateResource: Send, Sync);
        static_assertions::assert_impl_all!(KvAppendCompletion: Send);
        static_assertions::assert_impl_all!(CausalAttentionCompletion: Send);
    }

    #[test]
    fn abi_layout_fields_have_expected_rust_sizes() {
        assert_eq!(size_of::<sys::sllm_kv_state_create_info_t>(), 48);
        assert_eq!(size_of::<sys::sllm_kv_state_create_info_v2_t>(), 88);
        assert_eq!(size_of::<sys::sllm_kv_view_info_t>(), 192);
        assert_eq!(size_of::<sys::sllm_kv_append_desc_t>(), 416);
        assert_eq!(size_of::<sys::sllm_kv_append_info_t>(), 304);
        assert_eq!(size_of::<sys::sllm_causal_attention_desc_t>(), 424);
        assert_eq!(size_of::<sys::sllm_causal_attention_dispatch_info_t>(), 312);
    }
}
