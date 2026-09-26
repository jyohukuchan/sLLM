//! C3a2 request-local FP16/FP8/NVFP4 KV ownership and append transactions.
//!
//! Native KV handles are deliberately kept opaque. The only Rust-visible
//! state is copied metadata; the two device allocations and their strides
//! remain native-owned. The sendable resource token is an erased-core
//! lifetime boundary, while the direct state/view owner types carry an Rc
//! marker so they cannot be moved or shared as thread-affine native views.

use std::collections::HashMap;
use std::mem::size_of;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock, Weak};
use std::time::Duration;

use sllm_core::{
    DType, Encoding, ExecutionSessionId, ExecutionStateImageV2, KV_PAGED_IMAGE_TOKEN_BLOCK_SIZE,
    KV_PAGED_INVALID_BLOCK_ID, KV_PAGED_INVALID_TAG, KV_PAGED_PHYSICAL_LAYOUT_VERSION,
    KV_PAGED_PLANE_COUNT, KV_PAGED_RING_SLOT_COUNT, KV_PAGED_TOKEN_BLOCK_SIZE, KvCacheEncoding,
    KvFp8PhysicalVariant, KvPagedImageMetadataV1, KvPagedImageTopologyV1,
    KvPagedPhysicalMemorySnapshot, KvPagedRingTableV1, KvStateAppendRequest, KvStateDescriptor,
    KvStateId, KvStateSnapshot, StateForkAuditV1, StateLayerMetadataV1, StateOwnerKindV1,
};
use sllm_hip_sys as sys;

use crate::Buffer;
use crate::rmsnorm::TensorBinding;
use crate::runtime::{
    CompletionState, Context, Queue, RuntimeError, RuntimeStatus, completion_from_opaque_token,
    enqueue_causal_completion_cleanup, enqueue_kv_completion_cleanup, enqueue_kv_state_cleanup,
    ensure_ok, finalize_completion_after, gcn_arch_matches, logical_gcn_arch_name,
    release_causal_completion_once, release_kv_completion_once, release_kv_state_once,
    result_error, sink,
};

const ERROR_CAPACITY: usize = 256;
const MAX_FINITE_TIMEOUT_MS: u32 = u32::MAX - 1;
const KERNEL_SYMBOL: &str = "kv_state.bf16_to_f16_token_major.v2";
const DEVICE_SYMBOL: &str = "sllm_kv_state_bf16_to_f16_token_major_v2";
const WHOLE_DECODE_CAPTURE_MAX_ROWS: u64 = 9;

fn operation_range_admitted(
    start: u64,
    rows: u64,
    end: u64,
    capacity: u64,
    capture_projected: bool,
) -> bool {
    if rows == 0 || start.checked_add(rows) != Some(end) {
        return false;
    }
    if end <= capacity {
        return true;
    }
    capture_projected
        && rows <= WHOLE_DECODE_CAPTURE_MAX_ROWS
        && capacity
            .checked_add(WHOLE_DECODE_CAPTURE_MAX_ROWS - 1)
            .is_some_and(|limit| start < limit && end <= limit)
}

#[allow(dead_code)]
fn paged_native_storage(
    context: &Context,
    descriptor: KvStateDescriptor,
) -> Result<(u32, u32, u32, u32), RuntimeError> {
    let target = context.expected_target().map(logical_gcn_arch_name);
    if !matches!(target, Some("gfx1030" | "gfx1201")) {
        return Err(RuntimeError::new(
            RuntimeStatus::Unsupported,
            format!(
                "paged KV adapter requires exact gfx1030 or gfx1201 (got {})",
                target.unwrap_or("unspecified")
            ),
        ));
    }
    if let Some(window) = descriptor.sliding_window() {
        if window != 1024 || descriptor.cache_encoding() != KvCacheEncoding::Fp8E4M3FnStatic {
            return Err(RuntimeError::new(
                RuntimeStatus::Unsupported,
                "paged sliding KV requires static FP8 E4 with a 1024-token window".to_owned(),
            ));
        }
    }
    match descriptor.cache_encoding() {
        KvCacheEncoding::Fp16 => Ok((
            sys::SLLM_TENSOR_DTYPE_F16,
            sys::SLLM_HIP_KV_ENCODING_FP16_V1,
            0,
            0,
        )),
        KvCacheEncoding::Fp8E4M3Fn => Ok((
            sys::SLLM_TENSOR_DTYPE_F8_E4M3_FN,
            sys::SLLM_HIP_KV_ENCODING_FP8_V1,
            0,
            sys::SLLM_TENSOR_DTYPE_F32,
        )),
        KvCacheEncoding::Fp8E4M3FnStatic => {
            if descriptor.static_fp8_scales().is_none() {
                return Err(RuntimeError::new(
                    RuntimeStatus::InvalidKvStateDescriptor,
                    "paged KV static FP8 E4 requires binary32 key/value scales".to_owned(),
                ));
            }
            Ok((
                sys::SLLM_TENSOR_DTYPE_F8_E4M3_FN,
                sys::SLLM_HIP_KV_ENCODING_FP8_STATIC_V1,
                0,
                sys::SLLM_TENSOR_DTYPE_F32,
            ))
        }
        KvCacheEncoding::Nvfp4 => Ok((
            sys::SLLM_TENSOR_DTYPE_U8,
            sys::SLLM_HIP_KV_ENCODING_NVFP4_V1,
            16,
            sys::SLLM_TENSOR_DTYPE_F8_E4M3_FN,
        )),
        KvCacheEncoding::Mxfp8E4 => {
            if !matches!(
                descriptor.kv_mxfp8_descriptor(),
                Some(mxfp8) if mxfp8.physical_variant() == KvFp8PhysicalVariant::OcpE4M3Fn
            ) {
                return Err(RuntimeError::new(
                    RuntimeStatus::Unsupported,
                    "paged KV MXFP8 E4 requires the standard OCP E4M3FN physical variant"
                        .to_owned(),
                ));
            }
            Ok((
                sys::SLLM_TENSOR_DTYPE_F8_E4M3_FN,
                sys::SLLM_HIP_KV_ENCODING_MXFP8_E4_V1,
                32,
                sys::SLLM_TENSOR_DTYPE_U8,
            ))
        }
        KvCacheEncoding::Mxfp8E5 => {
            if !matches!(
                descriptor.kv_mxfp8_descriptor(),
                Some(mxfp8) if mxfp8.physical_variant() == KvFp8PhysicalVariant::OcpE5M2
            ) || target != Some("gfx1030")
            {
                return Err(RuntimeError::new(
                    RuntimeStatus::Unsupported,
                    "paged KV MXFP8 E5 requires the exact gfx1030 OCP E5M2 recipe".to_owned(),
                ));
            }
            Ok((
                sys::SLLM_TENSOR_DTYPE_F8_E5M2,
                sys::SLLM_HIP_KV_ENCODING_MXFP8_E5_V1,
                32,
                sys::SLLM_TENSOR_DTYPE_U8,
            ))
        }
        _ => Err(RuntimeError::new(
            RuntimeStatus::Unsupported,
            format!(
                "paged KV adapter does not support {} yet",
                descriptor.cache_encoding().canonical_name()
            ),
        )),
    }
}

fn paged_static_scale_bits(descriptor: KvStateDescriptor) -> (u32, u32) {
    descriptor
        .static_fp8_scales()
        .map(|(key, value)| (key.to_bits(), value.to_bits()))
        .unwrap_or((0, 0))
}

/// Reserve enough physical IDs for one parent and one child to each diverge
/// to their full logical capacity. The native paged table stores IDs as u32,
/// so keep the strict `UINT32_MAX` rejection in the Rust adapter as well.
fn paged_pool_capacities(
    capacity_tokens: u64,
    sliding_window: bool,
) -> Result<(u64, u64), RuntimeError> {
    let required_logical_blocks = capacity_tokens
        .checked_add(u64::from(KV_PAGED_TOKEN_BLOCK_SIZE) - 1)
        .ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::MetadataOverflow,
                "paged KV logical table capacity overflowed",
            )
        })?
        / u64::from(KV_PAGED_TOKEN_BLOCK_SIZE);
    // A sliding state publishes its nine-slot ring through the same logical
    // table capacity field used by non-sliding states.  The ring has one spare
    // slot beyond the 1024-token retention window, so a 1024-token descriptor
    // still needs nine addressable table entries.
    let logical_table_capacity = if sliding_window {
        required_logical_blocks.max(KV_PAGED_RING_SLOT_COUNT as u64)
    } else {
        required_logical_blocks
    };
    let max_physical_blocks = logical_table_capacity.checked_mul(2).ok_or_else(|| {
        RuntimeError::local(
            RuntimeStatus::MetadataOverflow,
            "paged KV parent-child physical block capacity overflowed",
        )
    })?;
    if max_physical_blocks >= u64::from(u32::MAX) {
        return Err(RuntimeError::local(
            RuntimeStatus::KvCapacityExceeded,
            "paged KV parent-child physical block capacity exceeds the u32 ID domain",
        ));
    }
    Ok((logical_table_capacity, max_physical_blocks))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)]
enum KvStateStorageKind {
    Legacy,
    Paged,
}

struct KvStateInner {
    raw: usize,
    context: Context,
    session_id: ExecutionSessionId,
    state_id: KvStateId,
    descriptor: KvStateDescriptor,
    storage_kind: KvStateStorageKind,
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
    pub(crate) fn create_paged(
        context: &Context,
        session_id: ExecutionSessionId,
        state_id: KvStateId,
        descriptor: KvStateDescriptor,
    ) -> Result<Self, RuntimeError> {
        if descriptor.capacity() > sys::SLLM_HIP_KV_MAX_CAPACITY {
            return Err(RuntimeError::local(
                RuntimeStatus::KvCapacityExceeded,
                "paged KV capacity exceeds the bounded native contract",
            ));
        }
        let context_raw = context.raw_handle()?;
        let (dtype, encoding, quantization_block_size, scale_dtype) =
            paged_native_storage(context, descriptor)?;
        let (logical_table_capacity, max_physical_blocks) =
            paged_pool_capacities(descriptor.capacity(), descriptor.sliding_window().is_some())?;
        let (static_key_scale_bits, static_value_scale_bits) = paged_static_scale_bits(descriptor);
        let info = sys::sllm_kv_state_paged_create_info_t {
            struct_size: size_of::<sys::sllm_kv_state_paged_create_info_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            create_info_version: sys::SLLM_HIP_KV_PAGED_CREATE_INFO_VERSION,
            reserved0: 0,
            session_id: session_id.raw(),
            layer_id: descriptor.layer_id(),
            flags: 0,
            capacity_tokens: descriptor.capacity(),
            head_count: descriptor.layout().heads() as u32,
            head_dim: descriptor.layout().head_dim() as u32,
            memory_kind: sys::SLLM_HIP_KV_MEMORY_KIND_PAGED,
            layout: sys::SLLM_HIP_KV_LAYOUT_TOKEN_MAJOR,
            dtype,
            encoding,
            scale_dtype,
            quantization_block_size,
            token_block_size: KV_PAGED_TOKEN_BLOCK_SIZE,
            physical_layout_version: KV_PAGED_PHYSICAL_LAYOUT_VERSION,
            logical_table_capacity,
            max_physical_blocks,
            sliding_window_tokens: descriptor.sliding_window().unwrap_or(0),
            static_key_scale_bits,
            static_value_scale_bits,
            reserved: [0; 4],
        };
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let mut raw_state = std::ptr::null_mut();
        let raw = unsafe {
            sys::sllm_kv_state_create_paged(
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
                "native paged KV state create returned a null handle on success",
            )
        })?;
        let resource = Self {
            inner: Arc::new(KvStateInner {
                raw: raw_state.as_ptr() as usize,
                context: context.clone(),
                session_id,
                state_id,
                descriptor,
                storage_kind: KvStateStorageKind::Paged,
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

    #[allow(dead_code)]
    fn ensure_paged_storage(&self, operation: &str) -> Result<(), RuntimeError> {
        if self.inner.storage_kind != KvStateStorageKind::Paged {
            return Err(RuntimeError::new(
                RuntimeStatus::Unsupported,
                format!("legacy KV state does not support paged image {operation}"),
            ));
        }
        Ok(())
    }

    /// Queries the additive Paged image ABI. This is deliberately separate
    /// from the legacy image query so callers cannot reinterpret VMM fields as
    /// Paged topology.
    #[allow(dead_code)]
    pub(crate) fn paged_image_query(
        &self,
    ) -> Result<sys::sllm_kv_paged_image_info_t, RuntimeError> {
        self.ensure_paged_storage("query")?;
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let mut info = sys::sllm_kv_paged_image_info_t {
            struct_size: size_of::<sys::sllm_kv_paged_image_info_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            image_version: sys::SLLM_HIP_KV_PAGED_IMAGE_VERSION,
            flags: 0,
            byte_order: sys::SLLM_HIP_KV_PAGED_IMAGE_ENDIAN_LITTLE,
            session_id: 0,
            layer_id: 0,
            dtype: 0,
            encoding: 0,
            head_count: 0,
            head_dim: 0,
            layout: 0,
            token_block_size: 0,
            physical_layout_version: 0,
            capacity_tokens: 0,
            published_length: 0,
            generation: 0,
            retained_start: 0,
            retained_length: 0,
            sliding_window_tokens: 0,
            logical_table_capacity: 0,
            physical_block_count: 0,
            plane_count: 0,
            ring_slot_count: 0,
            table_entry_width: 0,
            reserved0: 0,
            plane_block_stride: [0; 6],
            plane_bytes: [0; 6],
            static_key_scale_bits: 0,
            static_value_scale_bits: 0,
            reserved: [0; 8],
        };
        let status = unsafe {
            sys::sllm_kv_state_paged_image_query(
                self.raw_handle()?.as_ptr(),
                &mut info,
                &mut error_sink,
            )
        };
        ensure_ok(status, &error_buffer, error_sink.message_length)?;
        self.validate_paged_image_info(&info)?;
        Ok(info)
    }

    #[allow(dead_code)]
    pub(crate) fn paged_image_section_size(
        &self,
        section: u32,
        plane: u32,
    ) -> Result<u64, RuntimeError> {
        self.ensure_paged_storage("section query")?;
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let mut size_bytes = 0_u64;
        let status = unsafe {
            sys::sllm_kv_state_paged_image_section_size(
                self.raw_handle()?.as_ptr(),
                section,
                plane,
                &mut size_bytes,
                &mut error_sink,
            )
        };
        ensure_ok(status, &error_buffer, error_sink.message_length)?;
        Ok(size_bytes)
    }

    #[allow(dead_code)]
    pub(crate) fn paged_image_export_chunk(
        &self,
        section: u32,
        plane: u32,
        byte_offset: u64,
        destination: &mut [u8],
    ) -> Result<(), RuntimeError> {
        self.ensure_paged_storage("export")?;
        if destination.is_empty() {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidArgument,
                "Paged image export chunk must not be empty",
            ));
        }
        let byte_length = u64::try_from(destination.len()).map_err(|_| {
            RuntimeError::local(
                RuntimeStatus::MetadataOverflow,
                "Paged image export chunk length does not fit u64",
            )
        })?;
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let chunk = sys::sllm_kv_paged_image_chunk_t {
            struct_size: size_of::<sys::sllm_kv_paged_image_chunk_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            image_version: sys::SLLM_HIP_KV_PAGED_IMAGE_VERSION,
            section,
            plane,
            reserved0: 0,
            byte_offset,
            byte_length,
            host_pointer: destination.as_mut_ptr().cast(),
            host_capacity: byte_length,
            reserved: [0; 4],
        };
        let status = unsafe {
            sys::sllm_kv_state_paged_image_export(
                self.raw_handle()?.as_ptr(),
                &chunk,
                &mut error_sink,
            )
        };
        ensure_ok(status, &error_buffer, error_sink.message_length)
    }

    #[allow(dead_code)]
    pub(crate) fn paged_image_import_chunk(
        &self,
        section: u32,
        plane: u32,
        byte_offset: u64,
        source: &[u8],
    ) -> Result<(), RuntimeError> {
        self.ensure_paged_storage("import")?;
        if source.is_empty() {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidArgument,
                "Paged image import chunk must not be empty",
            ));
        }
        let byte_length = u64::try_from(source.len()).map_err(|_| {
            RuntimeError::local(
                RuntimeStatus::MetadataOverflow,
                "Paged image import chunk length does not fit u64",
            )
        })?;
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let chunk = sys::sllm_kv_paged_image_chunk_t {
            struct_size: size_of::<sys::sllm_kv_paged_image_chunk_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            image_version: sys::SLLM_HIP_KV_PAGED_IMAGE_VERSION,
            section,
            plane,
            reserved0: 0,
            byte_offset,
            byte_length,
            host_pointer: source.as_ptr().cast_mut().cast(),
            host_capacity: byte_length,
            reserved: [0; 4],
        };
        let status = unsafe {
            sys::sllm_kv_state_paged_image_import(
                self.raw_handle()?.as_ptr(),
                &chunk,
                &mut error_sink,
            )
        };
        ensure_ok(status, &error_buffer, error_sink.message_length)
    }

    #[allow(dead_code)]
    pub(crate) fn paged_image_import_finalize(
        &self,
        info: &sys::sllm_kv_paged_image_info_t,
    ) -> Result<(), RuntimeError> {
        self.ensure_paged_storage("import finalize")?;
        self.validate_paged_image_info(info)?;
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let status = unsafe {
            sys::sllm_kv_state_paged_image_import_finalize(
                self.raw_handle()?.as_ptr(),
                info,
                &mut error_sink,
            )
        };
        ensure_ok(status, &error_buffer, error_sink.message_length)
    }

    #[allow(dead_code)]
    fn validate_paged_image_info(
        &self,
        info: &sys::sllm_kv_paged_image_info_t,
    ) -> Result<(), RuntimeError> {
        let descriptor = self.inner.descriptor;
        let observed_target = self
            .inner
            .context
            .expected_target()
            .map(logical_gcn_arch_name)
            .unwrap_or("");
        if !matches!(observed_target, "gfx1030" | "gfx1201") {
            return Err(RuntimeError::new(
                RuntimeStatus::Unsupported,
                "Paged image requires exact gfx1030 or gfx1201 target".to_owned(),
            ));
        }
        let (expected_dtype, expected_encoding, _, _) =
            paged_native_storage(&self.inner.context, descriptor)?;
        let expected_flags = (descriptor.sliding_window().is_some() as u32
            * sys::SLLM_HIP_KV_PAGED_IMAGE_FLAG_SLIDING)
            | (descriptor.static_fp8_scales().is_some() as u32
                * sys::SLLM_HIP_KV_PAGED_IMAGE_FLAG_STATIC_SCALES);
        let expected_planes = match descriptor.cache_encoding() {
            KvCacheEncoding::Fp16 | KvCacheEncoding::Fp8E4M3FnStatic => 2,
            KvCacheEncoding::Fp8E4M3Fn | KvCacheEncoding::Mxfp8E4 | KvCacheEncoding::Mxfp8E5 => 4,
            KvCacheEncoding::Nvfp4 => 6,
            KvCacheEncoding::Fp8E4M3Block16 | KvCacheEncoding::Fp8E5M2Block16 => 0,
        };
        if info.struct_size != size_of::<sys::sllm_kv_paged_image_info_t>() as u32
            || info.abi_version != sys::SLLM_HIP_ABI_VERSION
            || info.image_version != sys::SLLM_HIP_KV_PAGED_IMAGE_VERSION
            || info.byte_order != sys::SLLM_HIP_KV_PAGED_IMAGE_ENDIAN_LITTLE
            || info.flags != expected_flags
            || info.session_id != self.inner.session_id.raw()
            || info.layer_id != descriptor.layer_id()
            || info.dtype != expected_dtype
            || info.encoding != expected_encoding
            || info.head_count != descriptor.layout().heads() as u32
            || info.head_dim != descriptor.layout().head_dim() as u32
            || info.layout != sys::SLLM_HIP_KV_LAYOUT_TOKEN_MAJOR
            || info.token_block_size != KV_PAGED_IMAGE_TOKEN_BLOCK_SIZE
            || info.physical_layout_version != KV_PAGED_PHYSICAL_LAYOUT_VERSION
            || info.capacity_tokens != descriptor.capacity()
            || info.published_length > info.capacity_tokens
            || info.plane_count != expected_planes
            || info.table_entry_width != sys::SLLM_HIP_KV_PAGED_IMAGE_TABLE_ENTRY_U32
            || info.reserved0 != 0
            || info.reserved.iter().any(|value| *value != 0)
        {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidKvStateDescriptor,
                "native Paged image metadata does not match the descriptor",
            ));
        }
        if descriptor.sliding_window().is_some() {
            if info.ring_slot_count != KV_PAGED_RING_SLOT_COUNT as u32
                || info.sliding_window_tokens != descriptor.sliding_window().unwrap_or(0)
            {
                return Err(RuntimeError::local(
                    RuntimeStatus::InvalidKvStateDescriptor,
                    "native Paged image ring metadata does not match the descriptor",
                ));
            }
        } else if info.ring_slot_count != 0 || info.sliding_window_tokens != 0 {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidKvStateDescriptor,
                "non-sliding Paged image contains ring metadata",
            ));
        }
        if descriptor
            .static_fp8_scales()
            .map(|(key, value)| (key.to_bits(), value.to_bits()))
            != Some((info.static_key_scale_bits, info.static_value_scale_bits))
            && descriptor.static_fp8_scales().is_some()
        {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidKvStateDescriptor,
                "native Paged image static scales differ from the descriptor",
            ));
        }
        if descriptor.static_fp8_scales().is_none()
            && (info.static_key_scale_bits != 0 || info.static_value_scale_bits != 0)
        {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidKvStateDescriptor,
                "native Paged image has unexpected static scales",
            ));
        }
        for plane in expected_planes as usize..KV_PAGED_PLANE_COUNT {
            if info.plane_block_stride[plane] != 0 || info.plane_bytes[plane] != 0 {
                return Err(RuntimeError::local(
                    RuntimeStatus::InvalidKvStateDescriptor,
                    "native Paged image has bytes for an absent plane",
                ));
            }
        }
        Ok(())
    }

    #[allow(dead_code)]
    pub(crate) fn paged_image_metadata(
        &self,
        info: &sys::sllm_kv_paged_image_info_t,
        logical_table_bytes: Option<&[u8]>,
        ring_table_bytes: Option<&[u8]>,
        ring_tag_bytes: Option<&[u8]>,
    ) -> Result<KvPagedImageMetadataV1, RuntimeError> {
        self.validate_paged_image_info(info)?;
        let descriptor = self.inner.descriptor;
        let topology = if descriptor.sliding_window().is_some() {
            let table = ring_table_bytes.ok_or_else(|| {
                RuntimeError::local(
                    RuntimeStatus::InvalidKvStateDescriptor,
                    "Paged image is missing the sliding ring table",
                )
            })?;
            let tags = ring_tag_bytes.ok_or_else(|| {
                RuntimeError::local(
                    RuntimeStatus::InvalidKvStateDescriptor,
                    "Paged image is missing the sliding ring tags",
                )
            })?;
            if table.len() != KV_PAGED_RING_SLOT_COUNT * 4
                || tags.len() != KV_PAGED_RING_SLOT_COUNT * 8
            {
                return Err(RuntimeError::local(
                    RuntimeStatus::InvalidKvStateDescriptor,
                    "Paged image ring section size is invalid",
                ));
            }
            let mut block_ids = [KV_PAGED_INVALID_BLOCK_ID; KV_PAGED_RING_SLOT_COUNT];
            let mut absolute_tags = [KV_PAGED_INVALID_TAG; KV_PAGED_RING_SLOT_COUNT];
            for slot in 0..KV_PAGED_RING_SLOT_COUNT {
                let table_start = slot * 4;
                let tag_start = slot * 8;
                block_ids[slot] = u32::from_le_bytes(
                    table[table_start..table_start + 4]
                        .try_into()
                        .expect("ring table width"),
                );
                absolute_tags[slot] = u64::from_le_bytes(
                    tags[tag_start..tag_start + 8]
                        .try_into()
                        .expect("ring tag width"),
                );
            }
            KvPagedImageTopologyV1::SlidingRing(KvPagedRingTableV1::new(block_ids, absolute_tags))
        } else {
            let table = logical_table_bytes.ok_or_else(|| {
                RuntimeError::local(
                    RuntimeStatus::InvalidKvStateDescriptor,
                    "Paged image is missing the logical table",
                )
            })?;
            let expected_len = usize::try_from(info.logical_table_capacity)
                .ok()
                .and_then(|count| count.checked_mul(4))
                .ok_or_else(|| {
                    RuntimeError::local(
                        RuntimeStatus::MetadataOverflow,
                        "Paged image logical table size overflowed",
                    )
                })?;
            if table.len() != expected_len {
                return Err(RuntimeError::local(
                    RuntimeStatus::InvalidKvStateDescriptor,
                    "Paged image logical table section size is invalid",
                ));
            }
            let mut entries = Vec::with_capacity(info.logical_table_capacity as usize);
            for chunk in table.chunks_exact(4) {
                entries.push(u32::from_le_bytes(chunk.try_into().expect("table width")));
            }
            KvPagedImageTopologyV1::LogicalTable(entries)
        };
        KvPagedImageMetadataV1::new(
            descriptor,
            info.published_length,
            info.generation,
            info.retained_start,
            info.sliding_window_tokens,
            info.physical_block_count,
            info.plane_block_stride,
            topology,
        )
        .map_err(|error| {
            RuntimeError::new(
                RuntimeStatus::InvalidKvStateDescriptor,
                format!("Paged image topology failed core validation: {error}"),
            )
        })
    }

    #[allow(dead_code)]
    fn paged_image_export_section(
        &self,
        section: u32,
        plane: u32,
        size: u64,
    ) -> Result<Vec<u8>, RuntimeError> {
        if size == 0 {
            return Ok(Vec::new());
        }
        let size_usize = usize::try_from(size).map_err(|_| {
            RuntimeError::local(
                RuntimeStatus::MetadataOverflow,
                "Paged image section exceeds host usize",
            )
        })?;
        let mut output = vec![0_u8; size_usize];
        let chunk_limit = usize::try_from(sys::SLLM_HIP_STATE_CHUNK_MAX_BYTES).map_err(|_| {
            RuntimeError::local(
                RuntimeStatus::MetadataOverflow,
                "Paged image chunk limit exceeds host usize",
            )
        })?;
        let mut offset = 0usize;
        while offset < output.len() {
            let end = offset.saturating_add(chunk_limit).min(output.len());
            self.paged_image_export_chunk(section, plane, offset as u64, &mut output[offset..end])?;
            offset = end;
        }
        Ok(output)
    }

    #[allow(dead_code)]
    pub(crate) fn export_paged_image_v2(&self) -> Result<ExecutionStateImageV2, RuntimeError> {
        let info = self.paged_image_query()?;
        let logical_table = if info.ring_slot_count == 0 {
            Some(self.paged_image_export_section(
                sys::SLLM_HIP_KV_PAGED_IMAGE_SECTION_LOGICAL_TABLE,
                0,
                self.paged_image_section_size(
                    sys::SLLM_HIP_KV_PAGED_IMAGE_SECTION_LOGICAL_TABLE,
                    0,
                )?,
            )?)
        } else {
            None
        };
        let ring_tags = if info.ring_slot_count != 0 {
            Some(self.paged_image_export_section(
                sys::SLLM_HIP_KV_PAGED_IMAGE_SECTION_RING_TAGS,
                0,
                self.paged_image_section_size(sys::SLLM_HIP_KV_PAGED_IMAGE_SECTION_RING_TAGS, 0)?,
            )?)
        } else {
            None
        };
        let ring_table = if info.ring_slot_count != 0 {
            Some(self.paged_image_export_section(
                sys::SLLM_HIP_KV_PAGED_IMAGE_SECTION_RING_TABLE,
                0,
                self.paged_image_section_size(sys::SLLM_HIP_KV_PAGED_IMAGE_SECTION_RING_TABLE, 0)?,
            )?)
        } else {
            None
        };
        let mut planes = std::array::from_fn(|_| Vec::new());
        for (index, plane) in planes.iter_mut().enumerate() {
            if index < info.plane_count as usize {
                *plane = self.paged_image_export_section(
                    sys::SLLM_HIP_KV_PAGED_IMAGE_SECTION_PLANE,
                    (index + 1) as u32,
                    self.paged_image_section_size(
                        sys::SLLM_HIP_KV_PAGED_IMAGE_SECTION_PLANE,
                        (index + 1) as u32,
                    )?,
                )?;
            }
        }
        let paged_metadata = self.paged_image_metadata(
            &info,
            logical_table.as_deref(),
            ring_table.as_deref(),
            ring_tags.as_deref(),
        )?;
        ExecutionStateImageV2::new(
            StateLayerMetadataV1 {
                owner: StateOwnerKindV1::Kv,
                layer_id: info.layer_id,
                published_length: info.published_length,
                generation: info.generation,
                active_slot: None,
            },
            paged_metadata,
            planes,
        )
        .map_err(|error| {
            RuntimeError::new(
                RuntimeStatus::InvalidKvStateDescriptor,
                format!("Paged V2 image validation failed: {error}"),
            )
        })
    }

    #[allow(dead_code)]
    fn paged_image_import_section(
        &self,
        section: u32,
        plane: u32,
        bytes: &[u8],
    ) -> Result<(), RuntimeError> {
        if bytes.is_empty() {
            return Ok(());
        }
        let chunk_limit = usize::try_from(sys::SLLM_HIP_STATE_CHUNK_MAX_BYTES).map_err(|_| {
            RuntimeError::local(
                RuntimeStatus::MetadataOverflow,
                "Paged image chunk limit exceeds host usize",
            )
        })?;
        let mut offset = 0usize;
        while offset < bytes.len() {
            let end = offset.saturating_add(chunk_limit).min(bytes.len());
            self.paged_image_import_chunk(section, plane, offset as u64, &bytes[offset..end])?;
            offset = end;
        }
        Ok(())
    }

    #[allow(dead_code)]
    pub(crate) fn import_paged_image_v2(
        &self,
        image: &ExecutionStateImageV2,
    ) -> Result<(), RuntimeError> {
        let destination = self.paged_image_query()?;
        if destination.published_length != 0 {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidKvStateDescriptor,
                "Paged V2 image import requires an empty destination",
            ));
        }
        let paged_metadata = image.paged_metadata();
        if image.metadata().layer_id != self.inner.descriptor.layer_id()
            || paged_metadata.descriptor() != self.inner.descriptor
            || paged_metadata.descriptor().layer_id() != destination.layer_id
        {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidKvStateDescriptor,
                "Paged V2 image descriptor does not match the destination",
            ));
        }
        let mut info = destination;
        let descriptor = self.inner.descriptor;
        info.flags = (u32::from(descriptor.sliding_window().is_some())
            * sys::SLLM_HIP_KV_PAGED_IMAGE_FLAG_SLIDING)
            | (u32::from(descriptor.static_fp8_scales().is_some())
                * sys::SLLM_HIP_KV_PAGED_IMAGE_FLAG_STATIC_SCALES);
        info.published_length = paged_metadata.observed_length();
        info.generation = paged_metadata.generation();
        info.retained_start = paged_metadata.retained_start();
        info.retained_length = info.published_length - info.retained_start;
        info.sliding_window_tokens = paged_metadata.sliding_window();
        info.physical_block_count = paged_metadata.physical_block_capacity();
        info.plane_block_stride = paged_metadata.plane_strides();
        info.plane_bytes = std::array::from_fn(|index| image.planes()[index].len() as u64);
        info.static_key_scale_bits = descriptor
            .static_fp8_scales()
            .map_or(0, |(key, _)| key.to_bits());
        info.static_value_scale_bits = descriptor
            .static_fp8_scales()
            .map_or(0, |(_, value)| value.to_bits());
        match paged_metadata.topology() {
            KvPagedImageTopologyV1::LogicalTable(table) => {
                let mut bytes = Vec::with_capacity(table.len() * 4);
                for entry in table {
                    bytes.extend_from_slice(&entry.to_le_bytes());
                }
                info.ring_slot_count = 0;
                self.paged_image_import_section(
                    sys::SLLM_HIP_KV_PAGED_IMAGE_SECTION_LOGICAL_TABLE,
                    0,
                    &bytes,
                )?;
            }
            KvPagedImageTopologyV1::SlidingRing(ring) => {
                let logical_capacity = descriptor
                    .capacity()
                    .checked_add(u64::from(KV_PAGED_TOKEN_BLOCK_SIZE) - 1)
                    .ok_or_else(|| {
                        RuntimeError::local(
                            RuntimeStatus::MetadataOverflow,
                            "Paged sliding logical table capacity overflowed",
                        )
                    })?
                    / u64::from(KV_PAGED_TOKEN_BLOCK_SIZE);
                let logical_capacity = logical_capacity.max(KV_PAGED_RING_SLOT_COUNT as u64);
                let logical_capacity = usize::try_from(logical_capacity).map_err(|_| {
                    RuntimeError::local(
                        RuntimeStatus::MetadataOverflow,
                        "Paged sliding logical table capacity exceeds host usize",
                    )
                })?;
                let mut logical = vec![KV_PAGED_INVALID_BLOCK_ID; logical_capacity];
                logical[..KV_PAGED_RING_SLOT_COUNT].copy_from_slice(&ring.block_ids());
                let mut logical_bytes = Vec::with_capacity(logical.len() * 4);
                for entry in logical {
                    logical_bytes.extend_from_slice(&entry.to_le_bytes());
                }
                let mut table = Vec::with_capacity(KV_PAGED_RING_SLOT_COUNT * 4);
                let mut tags = Vec::with_capacity(KV_PAGED_RING_SLOT_COUNT * 8);
                for entry in ring.block_ids() {
                    table.extend_from_slice(&entry.to_le_bytes());
                }
                for tag in ring.absolute_tags() {
                    tags.extend_from_slice(&tag.to_le_bytes());
                }
                info.ring_slot_count = KV_PAGED_RING_SLOT_COUNT as u32;
                self.paged_image_import_section(
                    sys::SLLM_HIP_KV_PAGED_IMAGE_SECTION_LOGICAL_TABLE,
                    0,
                    &logical_bytes,
                )?;
                self.paged_image_import_section(
                    sys::SLLM_HIP_KV_PAGED_IMAGE_SECTION_RING_TABLE,
                    0,
                    &table,
                )?;
                self.paged_image_import_section(
                    sys::SLLM_HIP_KV_PAGED_IMAGE_SECTION_RING_TAGS,
                    0,
                    &tags,
                )?;
            }
        }
        for (index, plane) in image.planes().iter().enumerate() {
            self.paged_image_import_section(
                sys::SLLM_HIP_KV_PAGED_IMAGE_SECTION_PLANE,
                (index + 1) as u32,
                plane,
            )?;
        }
        self.paged_image_import_finalize(&info)
    }

    /// Forks the native state while preserving exact encoded planes.  Layout,
    /// encoding, and scales must match; capacity may grow for a reused prefix.
    pub(crate) fn fork(
        &self,
        state_id: KvStateId,
        descriptor: KvStateDescriptor,
    ) -> Result<(Self, StateForkAuditV1), RuntimeError> {
        self.fork_paged(state_id, descriptor)
    }

    /// Fork a paged state through the additive ABI. The parent pool and its
    /// physical blocks are shared; legacy VMM page accounting is never used.
    fn fork_paged(
        &self,
        state_id: KvStateId,
        descriptor: KvStateDescriptor,
    ) -> Result<(Self, StateForkAuditV1), RuntimeError> {
        if self.inner.storage_kind != KvStateStorageKind::Paged {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidKvStateDescriptor,
                "legacy KV state must use the legacy fork adapter",
            ));
        }
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
                "native paged KV fork requires an identical destination recipe",
            ));
        }
        let (dtype, encoding, quantization_block_size, scale_dtype) =
            paged_native_storage(&self.inner.context, descriptor)?;
        let (logical_table_capacity, max_physical_blocks) =
            paged_pool_capacities(descriptor.capacity(), descriptor.sliding_window().is_some())?;
        let (static_key_scale_bits, static_value_scale_bits) = paged_static_scale_bits(descriptor);
        let destination_info = sys::sllm_kv_state_paged_create_info_t {
            struct_size: size_of::<sys::sllm_kv_state_paged_create_info_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            create_info_version: sys::SLLM_HIP_KV_PAGED_CREATE_INFO_VERSION,
            reserved0: 0,
            session_id: self.inner.session_id.raw(),
            layer_id: descriptor.layer_id(),
            flags: 0,
            capacity_tokens: descriptor.capacity(),
            head_count: descriptor.layout().heads() as u32,
            head_dim: descriptor.layout().head_dim() as u32,
            memory_kind: sys::SLLM_HIP_KV_MEMORY_KIND_PAGED,
            layout: sys::SLLM_HIP_KV_LAYOUT_TOKEN_MAJOR,
            dtype,
            encoding,
            scale_dtype,
            quantization_block_size,
            token_block_size: KV_PAGED_TOKEN_BLOCK_SIZE,
            physical_layout_version: KV_PAGED_PHYSICAL_LAYOUT_VERSION,
            logical_table_capacity,
            max_physical_blocks,
            sliding_window_tokens: descriptor.sliding_window().unwrap_or(0),
            static_key_scale_bits,
            static_value_scale_bits,
            reserved: [0; 4],
        };
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let mut info = empty_paged_fork_info();
        let mut raw_child = std::ptr::null_mut();
        let status = unsafe {
            sys::sllm_kv_state_fork_paged(
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
                "native paged KV fork returned a null child handle on success",
            )
        })?;
        let (expected_source_blocks, _) = paged_pool_capacities(
            self.inner.descriptor.capacity(),
            self.inner.descriptor.sliding_window().is_some(),
        )?;
        let valid_info = info.info_version == sys::SLLM_HIP_KV_PAGED_STATE_FORK_INFO_VERSION
            && info.token_block_size == KV_PAGED_TOKEN_BLOCK_SIZE
            && info.physical_layout_version == KV_PAGED_PHYSICAL_LAYOUT_VERSION
            && info.source_logical_table_capacity == expected_source_blocks
            && info.child_logical_table_capacity == logical_table_capacity
            && info.source_physical_blocks == info.shared_physical_blocks
            && info.child_physical_blocks == info.shared_physical_blocks
            && info.copied_physical_blocks == 0
            && info.shared_physical_blocks != 0
            && info.published_length != 0
            && info.source_physical_blocks <= expected_source_blocks;
        if !valid_info {
            let mut child_handle = raw_child.as_ptr();
            let _ = unsafe { sys::sllm_kv_state_release(&mut child_handle, &mut error_sink) };
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidKvStateDescriptor,
                "native paged KV fork returned inconsistent block accounting",
            ));
        }
        let audit = StateForkAuditV1::new_paged(
            info.published_length,
            info.shared_physical_blocks,
            info.shared_bytes,
            info.copied_bytes,
            info.child_owned_bytes,
        )
        .map_err(|error| {
            let mut child_handle = raw_child.as_ptr();
            let _ = unsafe { sys::sllm_kv_state_release(&mut child_handle, &mut error_sink) };
            RuntimeError::new(
                RuntimeStatus::InvalidKvStateDescriptor,
                format!("native paged KV fork audit failed core validation: {error}"),
            )
        })?;
        let resource = Self {
            inner: Arc::new(KvStateInner {
                raw: raw_child.as_ptr() as usize,
                context: self.inner.context.clone(),
                session_id: self.inner.session_id,
                state_id,
                descriptor,
                storage_kind: KvStateStorageKind::Paged,
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
    /// authoritative for shared-page or shared-block and destination-owned
    /// byte accounting.
    pub(crate) fn fork_query(&self) -> Result<StateForkAuditV1, RuntimeError> {
        self.fork_query_paged()
    }

    fn fork_query_paged(&self) -> Result<StateForkAuditV1, RuntimeError> {
        if self.inner.storage_kind != KvStateStorageKind::Paged {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidKvStateDescriptor,
                "legacy KV state must use the legacy fork query adapter",
            ));
        }
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let mut info = empty_paged_fork_info();
        let status = unsafe {
            sys::sllm_kv_state_fork_query_paged(
                self.raw_handle()?.as_ptr(),
                &mut info,
                &mut error_sink,
            )
        };
        ensure_ok(status, &error_buffer, error_sink.message_length)?;
        if info.info_version != sys::SLLM_HIP_KV_PAGED_STATE_FORK_INFO_VERSION
            || info.token_block_size != KV_PAGED_TOKEN_BLOCK_SIZE
            || info.physical_layout_version != KV_PAGED_PHYSICAL_LAYOUT_VERSION
            || info.source_state_identity != info.child_state_identity
            || info.source_logical_table_capacity != info.child_logical_table_capacity
            || info.source_physical_blocks != info.child_physical_blocks
            || info.shared_physical_blocks > info.child_physical_blocks
            || info.shared_bytes.checked_add(info.child_owned_bytes)
                != Some(info.source_owned_bytes)
            || info.published_length == 0
        {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidKvStateDescriptor,
                "native paged KV fork query returned inconsistent block accounting",
            ));
        }
        StateForkAuditV1::new_paged(
            info.published_length,
            info.shared_physical_blocks,
            info.shared_bytes,
            info.copied_bytes,
            info.child_owned_bytes,
        )
        .map_err(|error| {
            RuntimeError::new(
                RuntimeStatus::InvalidKvStateDescriptor,
                format!("native paged KV fork query audit failed core validation: {error}"),
            )
        })
    }

    pub(crate) fn snapshot(&self) -> Result<KvStateSnapshot, RuntimeError> {
        self.paged_snapshot()
    }

    /// Queries paged-pool metadata through the additive ABI. Legacy VMM view
    /// fields are never read for this path.
    pub(crate) fn paged_snapshot(&self) -> Result<KvStateSnapshot, RuntimeError> {
        if self.inner.storage_kind != KvStateStorageKind::Paged {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidKvStateDescriptor,
                "legacy KV state must use the legacy snapshot adapter",
            ));
        }
        let state_raw = self.raw_handle()?;
        let mut info = empty_paged_view_info();
        let mut error_buffer = [0_u8; ERROR_CAPACITY];
        let mut error_sink = sink(&mut error_buffer);
        let status = unsafe {
            sys::sllm_kv_state_query_paged(state_raw.as_ptr(), &mut info, &mut error_sink)
        };
        ensure_ok(status, &error_buffer, error_sink.message_length)?;
        validate_paged_view_info(
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
                "native paged KV snapshot generation moved backwards",
            ));
        }
        self.inner
            .last_generation
            .store(info.generation, Ordering::Release);
        let physical = KvPagedPhysicalMemorySnapshot::new(
            info.capacity_tokens,
            info.observed_length,
            info.token_block_size,
            info.physical_layout_version,
            info.logical_table_capacity,
            info.max_physical_blocks,
            info.allocated_physical_blocks,
            info.committed_bytes_per_plane,
            info.committed_bytes_total,
        )
        .map_err(|error| {
            RuntimeError::new(
                RuntimeStatus::InvalidKvStateDescriptor,
                format!("native paged KV physical metadata failed core validation: {error}"),
            )
        })?;
        KvStateSnapshot::new_with_paged_physical_memory(
            self.inner.session_id,
            self.inner.state_id,
            self.inner.descriptor,
            info.observed_length,
            physical,
        )
        .map_err(|error| {
            RuntimeError::new(
                RuntimeStatus::InvalidKvStateDescriptor,
                format!("native paged KV snapshot failed core validation: {error}"),
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

    pub(crate) fn readback(
        &self,
        _plane: u32,
        _byte_offset: u64,
        _destination: &mut [u8],
    ) -> Result<(), RuntimeError> {
        Err(RuntimeError::new(
            RuntimeStatus::Unsupported,
            "legacy KV readback is retired; Paged KV has no V1 view adapter".to_owned(),
        ))
    }

    pub(crate) fn append(
        &self,
        queue: &Queue,
        key: &TensorBinding,
        value: &TensorBinding,
        request: KvStateAppendRequest,
    ) -> Result<(KvAppendCompletion, KvAppendEvidence), RuntimeError> {
        let capture_projected = crate::graph_span::whole_decode_capture_active_on(queue)?;
        if request.state_id() != self.inner.state_id
            || request.descriptor() != self.inner.descriptor
            || request.start_position() != request.expected_length()
            || !operation_range_admitted(
                request.start_position(),
                request.token_count(),
                request.end_position(),
                self.inner.descriptor.capacity(),
                capture_projected,
            )
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
        let evidence = match if self.inner.storage_kind == KvStateStorageKind::Paged {
            validate_paged_append_info(
                &append_info,
                &self.inner.context,
                request,
                self.inner.descriptor,
                capture_projected,
            )
        } else {
            validate_append_info(
                &append_info,
                &self.inner.context,
                request,
                self.inner.descriptor,
                capture_projected,
            )
        } {
            Ok(evidence) => evidence,
            Err(error) => {
                let mut completion = completion;
                let _ = completion.wait(Duration::from_secs(30));
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
        let capture_projected = crate::graph_span::whole_decode_capture_active_on(queue)?;
        if capture_projected && (sliding_window.is_some() || score_scale.is_some()) {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidCausalAttentionDescriptor,
                "whole-decode capture supports only unscaled full causal attention",
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
        let evidence = match validate_attention_info_for_storage(
            &dispatch_info,
            &self.inner.context,
            self.inner.storage_kind,
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
            capture_projected,
        ) {
            Ok(evidence) => evidence,
            Err(error) => {
                let mut completion = completion;
                let _ = completion.wait(Duration::from_secs(30));
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
        if crate::graph_span::whole_decode_capture_active_on(queue)? {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidCausalAttentionDescriptor,
                "whole-decode capture does not use the eager append-attention chain",
            ));
        }
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
        let evidence = match validate_attention_info_for_storage(
            &dispatch_info,
            &self.inner.context,
            self.inner.storage_kind,
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
            false,
        ) {
            Ok(evidence) => evidence,
            Err(error) => {
                let mut completion = completion;
                let _ = completion.wait(Duration::from_secs(30));
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
    pub(crate) fn capture_marker(
        &mut self,
        capture: &mut crate::graph_span::WholeDecodeCapture,
    ) -> Result<(), RuntimeError> {
        capture.capture_opaque_completion(&mut self.raw)
    }

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
    pub(crate) fn capture_marker(
        &mut self,
        capture: &mut crate::graph_span::WholeDecodeCapture,
    ) -> Result<(), RuntimeError> {
        capture.capture_opaque_completion(&mut self.raw)
    }

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

fn empty_paged_view_info() -> sys::sllm_kv_paged_view_info_t {
    sys::sllm_kv_paged_view_info_t {
        struct_size: size_of::<sys::sllm_kv_paged_view_info_t>() as u32,
        abi_version: sys::SLLM_HIP_ABI_VERSION,
        info_version: sys::SLLM_HIP_KV_PAGED_VIEW_INFO_VERSION,
        reserved0: 0,
        session_id: 0,
        layer_id: 0,
        dtype: 0,
        encoding: 0,
        head_count: 0,
        head_dim: 0,
        memory_kind: 0,
        layout: 0,
        token_block_size: 0,
        physical_layout_version: 0,
        reserved1: 0,
        capacity_tokens: 0,
        observed_length: 0,
        generation: 0,
        logical_table_capacity: 0,
        max_physical_blocks: 0,
        allocated_physical_blocks: 0,
        committed_bytes_per_plane: [0; KV_PAGED_PLANE_COUNT],
        committed_bytes_total: 0,
        context_identity: 0,
        state_identity: 0,
        reserved: [0; 4],
    }
}

fn empty_paged_fork_info() -> sys::sllm_kv_paged_state_fork_info_t {
    sys::sllm_kv_paged_state_fork_info_t {
        struct_size: size_of::<sys::sllm_kv_paged_state_fork_info_t>() as u32,
        abi_version: sys::SLLM_HIP_ABI_VERSION,
        info_version: sys::SLLM_HIP_KV_PAGED_STATE_FORK_INFO_VERSION,
        reserved0: 0,
        source_state_identity: 0,
        child_state_identity: 0,
        source_owned_bytes: 0,
        child_owned_bytes: 0,
        copied_bytes: 0,
        shared_bytes: 0,
        published_length: 0,
        token_block_size: 0,
        physical_layout_version: 0,
        source_logical_table_capacity: 0,
        child_logical_table_capacity: 0,
        source_physical_blocks: 0,
        child_physical_blocks: 0,
        copied_physical_blocks: 0,
        shared_physical_blocks: 0,
        committed_bytes_total: 0,
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

#[allow(dead_code)]
fn validate_paged_view_info(
    info: &sys::sllm_kv_paged_view_info_t,
    context: &Context,
    raw_state: usize,
    session_id: ExecutionSessionId,
    descriptor: KvStateDescriptor,
) -> Result<(), RuntimeError> {
    let (expected_dtype, expected_encoding, _, _) = paged_native_storage(context, descriptor)?;
    let layout = descriptor.layout();
    let required_logical_blocks = descriptor
        .capacity()
        .checked_add(u64::from(KV_PAGED_TOKEN_BLOCK_SIZE) - 1)
        .ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::MetadataOverflow,
                "paged KV logical table capacity overflowed",
            )
        })?
        / u64::from(KV_PAGED_TOKEN_BLOCK_SIZE);
    if info.struct_size != size_of::<sys::sllm_kv_paged_view_info_t>() as u32
        || info.abi_version != sys::SLLM_HIP_ABI_VERSION
        || info.info_version != sys::SLLM_HIP_KV_PAGED_VIEW_INFO_VERSION
        || info.reserved0 != 0
        || info.session_id != session_id.raw()
        || info.layer_id != descriptor.layer_id()
        || info.dtype != expected_dtype
        || info.encoding != expected_encoding
        || info.head_count != layout.heads() as u32
        || info.head_dim != layout.head_dim() as u32
        || info.memory_kind != sys::SLLM_HIP_KV_MEMORY_KIND_PAGED
        || info.layout != sys::SLLM_HIP_KV_LAYOUT_TOKEN_MAJOR
        || info.token_block_size != KV_PAGED_TOKEN_BLOCK_SIZE
        || info.physical_layout_version != KV_PAGED_PHYSICAL_LAYOUT_VERSION
        || info.capacity_tokens != descriptor.capacity()
        || info.observed_length > info.capacity_tokens
        || info.logical_table_capacity < required_logical_blocks
        || info.max_physical_blocks < info.logical_table_capacity
        || info.allocated_physical_blocks > info.max_physical_blocks
        || info.context_identity != context.raw_handle()?.as_ptr() as usize as u64
        || info.state_identity != raw_state as u64
        || info.reserved1 != 0
        || info.reserved.iter().any(|value| *value != 0)
    {
        return Err(RuntimeError::local(
            RuntimeStatus::InvalidKvStateDescriptor,
            "native paged KV view metadata does not match the Rust descriptor",
        ));
    }
    // The core constructor performs the checked sum and observed-length block
    // boundary validation. Calling it here also makes malformed native counts
    // fail before they become an externally visible snapshot.
    KvPagedPhysicalMemorySnapshot::new(
        info.capacity_tokens,
        info.observed_length,
        info.token_block_size,
        info.physical_layout_version,
        info.logical_table_capacity,
        info.max_physical_blocks,
        info.allocated_physical_blocks,
        info.committed_bytes_per_plane,
        info.committed_bytes_total,
    )
    .map_err(|error| {
        RuntimeError::new(
            RuntimeStatus::InvalidKvStateDescriptor,
            format!("native paged KV view metadata failed core validation: {error}"),
        )
    })?;
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

fn paged_append_recipe(
    encoding: KvCacheEncoding,
) -> Result<(u32, &'static str, &'static str), RuntimeError> {
    let recipe = match encoding {
        KvCacheEncoding::Fp16 => (
            sys::SLLM_HIP_KV_KERNEL_ID_BF16_TO_PAGED_F16_V1,
            "kv_state.bf16_to_paged_f16.v1",
            "sllm_kv_state_bf16_to_paged_f16_v1",
        ),
        KvCacheEncoding::Fp8E4M3Fn => (
            sys::SLLM_HIP_KV_KERNEL_ID_BF16_TO_PAGED_FP8_E4_V1,
            "kv_state.bf16_to_paged_fp8.v1",
            "sllm_kv_state_bf16_to_paged_fp8_v1",
        ),
        KvCacheEncoding::Fp8E4M3FnStatic => (
            sys::SLLM_HIP_KV_KERNEL_ID_BF16_TO_PAGED_FP8_STATIC_E4_V1,
            "kv_state.bf16_to_paged_fp8_static.v1",
            "sllm_kv_state_bf16_to_paged_fp8_static_v1",
        ),
        KvCacheEncoding::Nvfp4 => (
            sys::SLLM_HIP_KV_KERNEL_ID_BF16_TO_PAGED_NVFP4_V1,
            "kv_state.bf16_to_paged_nvfp4.v1",
            "sllm_kv_state_bf16_to_paged_nvfp4_v1",
        ),
        KvCacheEncoding::Mxfp8E4 => (
            sys::SLLM_HIP_KV_KERNEL_ID_BF16_TO_PAGED_MXFP8_E4_V1,
            "kv_state.bf16_to_paged_mxfp8_e4.v1",
            "sllm_kv_state_bf16_to_paged_mxfp8_e4_v1",
        ),
        KvCacheEncoding::Mxfp8E5 => (
            sys::SLLM_HIP_KV_KERNEL_ID_BF16_TO_PAGED_MXFP8_E5_V1,
            "kv_state.bf16_to_paged_mxfp8_e5.v1",
            "sllm_kv_state_bf16_to_paged_mxfp8_e5_v1",
        ),
        KvCacheEncoding::Fp8E4M3Block16 | KvCacheEncoding::Fp8E5M2Block16 => {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidKvAppendDescriptor,
                "paged KV append received a retired FP8 block16 encoding",
            ));
        }
    };
    Ok(recipe)
}

fn paged_target_matches(context: &Context, observed_target: &str) -> bool {
    let Some(expected_target) = context.expected_target().map(logical_gcn_arch_name) else {
        return false;
    };
    matches!(expected_target, "gfx1030" | "gfx1201")
        && matches!(observed_target, "gfx1030" | "gfx1201")
        && expected_target == observed_target
}

fn validate_paged_append_info(
    info: &sys::sllm_kv_append_info_t,
    context: &Context,
    request: KvStateAppendRequest,
    descriptor: KvStateDescriptor,
    capture_projected: bool,
) -> Result<KvAppendEvidence, RuntimeError> {
    let observed_target = c_string(&info.gcn_arch_name);
    let target = logical_gcn_arch_name(&observed_target).to_owned();
    let (expected_kernel_id, expected_kernel, expected_device) =
        paged_append_recipe(descriptor.cache_encoding())?;
    let expected_rows = request
        .token_count()
        .checked_mul(descriptor.layout().heads() as u64)
        .ok_or_else(|| {
            RuntimeError::local(RuntimeStatus::MetadataOverflow, "paged KV grid overflow")
        })?;
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
    let target_supported = paged_target_matches(context, &observed_target)
        && !(descriptor.cache_encoding() == KvCacheEncoding::Mxfp8E5
            && context.expected_target().map(logical_gcn_arch_name) != Some("gfx1030"));
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
        || !operation_range_admitted(
            info.start_position,
            info.token_count,
            info.end_position,
            descriptor.capacity(),
            capture_projected,
        )
        || !target_supported
    {
        return Err(RuntimeError::local(
            RuntimeStatus::InvalidKvAppendDescriptor,
            "native paged KV append metadata failed exact-provider/no-fallback validation",
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

fn validate_append_info(
    info: &sys::sllm_kv_append_info_t,
    context: &Context,
    request: KvStateAppendRequest,
    descriptor: KvStateDescriptor,
    capture_projected: bool,
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
        || !operation_range_admitted(
            info.start_position,
            info.token_count,
            info.end_position,
            descriptor.capacity(),
            capture_projected,
        )
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
    capture_projected: bool,
) -> Result<CausalAttentionEvidence, RuntimeError> {
    let staged_decode_opt_in = std::env::var_os("SLLM_CAUSAL_ATTENTION_GFX1030_DECODE_WAVE_STAGED");
    let staged32_decode_opt_in = std::env::var_os("SLLM_CAUSAL_ATTENTION_DECODE_WAVE_STAGED32");
    validate_causal_attention_info_impl(
        info,
        context,
        start_position,
        committed_kv_length,
        descriptor,
        query_heads,
        sliding_window,
        score_scale,
        staged_decode_opt_in.as_deref(),
        staged32_decode_opt_in.as_deref(),
        capture_projected,
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_paged_attention_info(
    info: &sys::sllm_causal_attention_dispatch_info_t,
    context: &Context,
    start_position: u64,
    committed_kv_length: u64,
    descriptor: KvStateDescriptor,
    query_heads: u32,
    sliding_window: Option<u64>,
    score_scale: Option<f32>,
    capture_projected: bool,
) -> Result<CausalAttentionEvidence, RuntimeError> {
    let query_count = committed_kv_length
        .checked_sub(start_position)
        .ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::CausalAttentionLengthMismatch,
                "paged causal attention evidence range underflowed",
            )
        })?;
    let observed_target = c_string(&info.gcn_arch_name);
    let target = logical_gcn_arch_name(&observed_target).to_owned();
    let target_supported = paged_target_matches(context, &observed_target)
        && !(descriptor.cache_encoding() == KvCacheEncoding::Mxfp8E5
            && context.expected_target().map(logical_gcn_arch_name) != Some("gfx1030"));
    let kv_heads = descriptor.layout().heads() as u32;
    let head_dim = descriptor.layout().head_dim() as u32;
    let mxfp8_gqa6 = descriptor.cache_encoding() == KvCacheEncoding::Mxfp8E4
        && kv_heads == 4
        && head_dim == 256
        && query_heads == 24;
    let mxfp8_gqa6_gfx1201_wave_prefill =
        mxfp8_gqa6 && target == "gfx1201" && query_count > 5 && query_count < 128;
    let mxfp8_gqa6_gfx1201_qtile4_prefill =
        mxfp8_gqa6 && target == "gfx1201" && query_count >= 128 && start_position < 1024;
    let fp16_reviewed = descriptor.cache_encoding() == KvCacheEncoding::Fp16
        && kv_heads == 4
        && head_dim == 256
        && matches!(query_heads, 16 | 24);
    let fp16_gqa4_shared_prefill =
        fp16_reviewed && query_count >= 64 && query_heads == 16 && target_supported;
    // Gemma4's Paged sliding provider is intentionally exact: the native
    // ring kernel has a fixed 9-slot/1024-token contract and is reviewed only
    // for the 16Q/8KV, head_dim=256 static-FP8 shape on these two targets.
    let paged_sliding_static_fp8 = descriptor.cache_encoding() == KvCacheEncoding::Fp8E4M3FnStatic
        && descriptor.static_fp8_scales() == Some((1.0, 1.0))
        && descriptor.sliding_window() == Some(1024)
        && sliding_window == Some(1024)
        && score_scale == Some(1.0)
        && kv_heads == 8
        && head_dim == 256
        && query_heads == 16;
    // Gemma4 full-attention layers use the generic Paged provider with the
    // reviewed static-FP8 16Q/2KV, head_dim=512 shape and unit score scale.
    let paged_full_static_fp8 = descriptor.cache_encoding() == KvCacheEncoding::Fp8E4M3FnStatic
        && descriptor.static_fp8_scales() == Some((1.0, 1.0))
        && descriptor.sliding_window().is_none()
        && sliding_window.is_none()
        && score_scale == Some(1.0)
        && kv_heads == 2
        && head_dim == 512
        && query_heads == 16;
    let (
        expected_kernel_id,
        expected_kernel,
        expected_device,
        expected_dispatch_count,
        expected_workgroup,
        expected_grid,
    ) = if paged_sliding_static_fp8 {
        let grid = query_count
            .checked_mul(u64::from(query_heads))
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| {
                RuntimeError::local(
                    RuntimeStatus::MetadataOverflow,
                    "paged sliding static-FP8 grid overflowed",
                )
            })?;
        (
            sys::SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_PAGED_SLIDING_STATIC_FP8_V1,
            "causal_attention.paged_sliding_static_fp8_ring.v1",
            "sllm_causal_attention_paged_sliding_static_fp8_ring_v1",
            1,
            sys::SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE,
            grid,
        )
    } else if paged_full_static_fp8 {
        let grid = query_count
            .checked_mul(u64::from(query_heads))
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| {
                RuntimeError::local(
                    RuntimeStatus::MetadataOverflow,
                    "paged full static-FP8 grid overflowed",
                )
            })?;
        (
            sys::SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_PAGED_GENERIC_FORMATS_V1,
            "causal_attention.paged.generic_formats.v1",
            "sllm_causal_attention_paged_generic_formats_v1",
            1,
            sys::SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE,
            grid,
        )
    } else if mxfp8_gqa6 {
        if query_count <= 5 {
            let splits = if committed_kv_length >= 8192 { 128 } else { 32 };
            if query_count == 3
                && committed_kv_length >= 8192
                && target_supported
                && sliding_window.is_none()
                && score_scale.is_none()
            {
                let grid = 4_u64
                    .checked_mul(splits)
                    .and_then(|value| u32::try_from(value).ok())
                    .ok_or_else(|| {
                        RuntimeError::local(
                            RuntimeStatus::MetadataOverflow,
                            "paged GQA6 C1 decode grid overflowed",
                        )
                    })?;
                (
                    sys::SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_PAGED_DECODE_GQA6_C1_M3_V1,
                    "causal_attention.paged_decode.gqa6_c1_m3.mxfp8_e4.v1",
                    "sllm_causal_attention_paged_decode_gqa6_c1_m3_mxfp8_e4_v1",
                    2,
                    192,
                    grid,
                )
            } else {
                let grid = query_count
                    .checked_mul(if target == "gfx1030" { 4 } else { 24 })
                    .and_then(|value| value.checked_mul(splits))
                    .and_then(|value| u32::try_from(value).ok())
                    .ok_or_else(|| {
                        RuntimeError::local(
                            RuntimeStatus::MetadataOverflow,
                            "paged GQA6 decode grid overflowed",
                        )
                    })?;
                (
                    sys::SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_PAGED_DECODE_GQA6_V1,
                    "causal_attention.paged_decode.gqa6_m1_m5.mxfp8_e4.v1",
                    "sllm_causal_attention_paged_decode_gqa6_m1_m5_mxfp8_e4_v1",
                    2,
                    if target == "gfx1030" { 192 } else { 32 },
                    grid,
                )
            }
        } else if mxfp8_gqa6_gfx1201_wave_prefill {
            let grid = query_count
                .checked_mul(u64::from(query_heads))
                .and_then(|value| u32::try_from(value).ok())
                .ok_or_else(|| {
                    RuntimeError::local(
                        RuntimeStatus::MetadataOverflow,
                        "paged gfx1201 GQA6 wave prefill grid overflowed",
                    )
                })?;
            (
                sys::SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_PAGED_PREFILL_GQA6_V1,
                "causal_attention.paged_prefill.gqa6_wave.gfx1201.mxfp8_e4.v1",
                // The native symbol is 65 bytes; the ABI field retains the
                // first 63 bytes plus the terminating NUL.
                "sllm_causal_attention_paged_prefill_gqa6_wave_gfx1201_mxfp8_e4_",
                1,
                sys::SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE,
                grid,
            )
        } else if mxfp8_gqa6_gfx1201_qtile4_prefill {
            let grid = query_count
                .checked_mul(u64::from(query_heads))
                .and_then(|value| u32::try_from(value).ok())
                .ok_or_else(|| {
                    RuntimeError::local(
                        RuntimeStatus::MetadataOverflow,
                        "paged gfx1201 GQA6 qtile4 prefill grid overflowed",
                    )
                })?;
            (
                sys::SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_PAGED_PREFILL_GQA6_V1,
                "causal_attention.paged_prefill.gqa6_qtile4.mxfp8_e4.v1",
                "sllm_causal_attention_paged_prefill_gqa6_qtile4_mxfp8_e4_v1",
                1,
                sys::SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE,
                grid,
            )
        } else {
            let grid = query_count
                .checked_add(7)
                .and_then(|value| (value / 8).checked_mul(4))
                .and_then(|value| u32::try_from(value).ok())
                .ok_or_else(|| {
                    RuntimeError::local(
                        RuntimeStatus::MetadataOverflow,
                        "paged GQA6 prefill grid overflowed",
                    )
                })?;
            (
                sys::SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_PAGED_PREFILL_GQA6_V1,
                "causal_attention.paged_prefill.gqa6_qtile8.mxfp8_e4.v1",
                "sllm_causal_attention_paged_prefill_gqa6_qtile8_mxfp8_e4_v1",
                1,
                512,
                grid,
            )
        }
    } else if fp16_gqa4_shared_prefill {
        (
            sys::SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_PAGED_PREFILL_FP16_V1,
            "causal_attention.paged_prefill.gqa4_shared.v1",
            "sllm_causal_attention_paged_prefill_gqa4_shared_v1",
            1,
            sys::SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE,
            query_count
                .checked_mul(u64::from(query_heads))
                .and_then(|value| u32::try_from(value).ok())
                .ok_or_else(|| {
                    RuntimeError::local(
                        RuntimeStatus::MetadataOverflow,
                        "paged FP16 GQA4 prefill grid overflowed",
                    )
                })?,
        )
    } else if fp16_reviewed {
        if query_count <= 5 {
            (
                sys::SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_PAGED_DECODE_FP16_V1,
                "causal_attention.paged_decode.fp16_gqa.v1",
                "sllm_causal_attention_paged_decode_fp16_gqa_v1",
                1,
                sys::SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE,
                query_count
                    .checked_mul(u64::from(query_heads))
                    .and_then(|value| u32::try_from(value).ok())
                    .ok_or_else(|| {
                        RuntimeError::local(
                            RuntimeStatus::MetadataOverflow,
                            "paged FP16 decode grid overflowed",
                        )
                    })?,
            )
        } else {
            (
                sys::SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_PAGED_PREFILL_FP16_V1,
                "causal_attention.paged_prefill.fp16_gqa.v1",
                "sllm_causal_attention_paged_prefill_fp16_gqa_v1",
                1,
                sys::SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE,
                query_count
                    .checked_mul(u64::from(query_heads))
                    .and_then(|value| u32::try_from(value).ok())
                    .ok_or_else(|| {
                        RuntimeError::local(
                            RuntimeStatus::MetadataOverflow,
                            "paged FP16 prefill grid overflowed",
                        )
                    })?,
            )
        }
    } else {
        let grid = query_count
            .checked_mul(u64::from(query_heads))
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| {
                RuntimeError::local(
                    RuntimeStatus::MetadataOverflow,
                    "paged generic attention grid overflowed",
                )
            })?;
        (
            sys::SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_PAGED_GENERIC_FORMATS_V1,
            "causal_attention.paged.generic_formats.v1",
            "sllm_causal_attention_paged_generic_formats_v1",
            1,
            sys::SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE,
            grid,
        )
    };
    let (expected_scale_denominator, implicit_scale_bits, expected_reserved) =
        if paged_sliding_static_fp8 || paged_full_static_fp8 {
            let retained_start = committed_kv_length.saturating_sub(1024);
            let window = if paged_sliding_static_fp8 { 1024 } else { 0 };
            (
                0,
                1.0_f32.to_bits(),
                [
                    window,
                    0,
                    if paged_sliding_static_fp8 {
                        retained_start as u32
                    } else {
                        0
                    },
                    if paged_sliding_static_fp8 {
                        (retained_start >> 32) as u32
                    } else {
                        0
                    },
                    1.0_f32.to_bits(),
                    1,
                    0,
                    0,
                ],
            )
        } else {
            implicit_attention_scale_evidence(head_dim)
        };
    if query_count == 0
        || (!paged_sliding_static_fp8
            && !paged_full_static_fp8
            && (sliding_window.is_some() || score_scale.is_some()))
        || !target_supported
        || info.struct_size != size_of::<sys::sllm_causal_attention_dispatch_info_t>() as u32
        || info.abi_version != sys::SLLM_HIP_ABI_VERSION
        || info.info_version != sys::SLLM_HIP_CAUSAL_ATTENTION_DISPATCH_INFO_VERSION
        || info.backend != sys::SLLM_BACKEND_HIP
        || info.dispatch_id == 0
        || info.dispatch_count != expected_dispatch_count
        || info.kernel_id != expected_kernel_id
        || info.workgroup_size_x != expected_workgroup
        || info.grid_size_x != expected_grid
        || info.query_count != query_count
        || info.start_position != start_position
        || info.committed_kv_length != committed_kv_length
        || info.q_heads != query_heads
        || info.kv_heads != kv_heads
        || info.head_dim != head_dim
        || info.scale_denominator != expected_scale_denominator
        || info.fallback_allowed != 0
        || info.fallback_used != 0
        || c_string(&info.kernel_symbol) != expected_kernel
        || c_string(&info.device_symbol) != expected_device
        || info.reserved != expected_reserved
        || !operation_range_admitted(
            start_position,
            query_count,
            committed_kv_length,
            descriptor.capacity(),
            capture_projected,
        )
    {
        return Err(RuntimeError::local(
            RuntimeStatus::InvalidCausalAttentionDescriptor,
            "native paged causal attention metadata failed exact-provider/no-fallback validation",
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
        sliding_window: if paged_sliding_static_fp8 { 1024 } else { 0 },
        retained_start: if paged_sliding_static_fp8 {
            committed_kv_length.saturating_sub(1024)
        } else {
            0
        },
        score_scale_bits: implicit_scale_bits,
        explicit_score_scale: paged_sliding_static_fp8 || paged_full_static_fp8,
        q_heads: info.q_heads,
        kv_heads: info.kv_heads,
        head_dim: info.head_dim,
        scale_denominator: info.scale_denominator,
        fallback_allowed: false,
        fallback_used: false,
        kernel_symbol: c_string(&info.kernel_symbol),
        device_symbol: c_string(&info.device_symbol),
        target,
    })
}

#[allow(clippy::too_many_arguments)]
fn validate_attention_info_for_storage(
    info: &sys::sllm_causal_attention_dispatch_info_t,
    context: &Context,
    storage_kind: KvStateStorageKind,
    start_position: u64,
    committed_kv_length: u64,
    descriptor: KvStateDescriptor,
    query_heads: u32,
    sliding_window: Option<u64>,
    score_scale: Option<f32>,
    capture_projected: bool,
) -> Result<CausalAttentionEvidence, RuntimeError> {
    match storage_kind {
        KvStateStorageKind::Legacy => validate_causal_attention_info(
            info,
            context,
            start_position,
            committed_kv_length,
            descriptor,
            query_heads,
            sliding_window,
            score_scale,
            capture_projected,
        ),
        KvStateStorageKind::Paged => validate_paged_attention_info(
            info,
            context,
            start_position,
            committed_kv_length,
            descriptor,
            query_heads,
            sliding_window,
            score_scale,
            capture_projected,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn validate_causal_attention_info_with_staged_opt_in(
    info: &sys::sllm_causal_attention_dispatch_info_t,
    context: &Context,
    start_position: u64,
    committed_kv_length: u64,
    descriptor: KvStateDescriptor,
    query_heads: u32,
    sliding_window: Option<u64>,
    score_scale: Option<f32>,
    staged_decode_opt_in: Option<&std::ffi::OsStr>,
) -> Result<CausalAttentionEvidence, RuntimeError> {
    validate_causal_attention_info_impl(
        info,
        context,
        start_position,
        committed_kv_length,
        descriptor,
        query_heads,
        sliding_window,
        score_scale,
        staged_decode_opt_in,
        None,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn validate_causal_attention_info_with_staged_opt_ins(
    info: &sys::sllm_causal_attention_dispatch_info_t,
    context: &Context,
    start_position: u64,
    committed_kv_length: u64,
    descriptor: KvStateDescriptor,
    query_heads: u32,
    sliding_window: Option<u64>,
    score_scale: Option<f32>,
    staged_decode_opt_in: Option<&std::ffi::OsStr>,
    staged32_decode_opt_in: Option<&std::ffi::OsStr>,
) -> Result<CausalAttentionEvidence, RuntimeError> {
    validate_causal_attention_info_impl(
        info,
        context,
        start_position,
        committed_kv_length,
        descriptor,
        query_heads,
        sliding_window,
        score_scale,
        staged_decode_opt_in,
        staged32_decode_opt_in,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_causal_attention_info_impl(
    info: &sys::sllm_causal_attention_dispatch_info_t,
    context: &Context,
    start_position: u64,
    committed_kv_length: u64,
    descriptor: KvStateDescriptor,
    query_heads: u32,
    sliding_window: Option<u64>,
    score_scale: Option<f32>,
    staged_decode_opt_in: Option<&std::ffi::OsStr>,
    staged32_decode_opt_in: Option<&std::ffi::OsStr>,
    capture_projected: bool,
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
            || !operation_range_admitted(
                start_position,
                query_count,
                committed_kv_length,
                descriptor.capacity(),
                capture_projected,
            )
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
            || !operation_range_admitted(
                start_position,
                query_count,
                committed_kv_length,
                descriptor.capacity(),
                capture_projected,
            )
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
    let use_decode_wave_split_staged = !force_baseline
        && expected_target == Some("gfx1030")
        && staged_decode_opt_in.is_some_and(|value| value == "1")
        && (1..=4).contains(&query_count)
        && committed_kv_length >= 1024
        && query_heads == 24
        && descriptor.layout().heads() == 4
        && descriptor.layout().head_dim() == 256
        && descriptor.cache_encoding() == KvCacheEncoding::Mxfp8E4;
    let use_decode_wave_split_staged32 = !force_baseline
        && matches!(expected_target, Some("gfx1030" | "gfx1201"))
        && (staged32_decode_opt_in.is_some_and(|value| value == "1")
            || (staged32_decode_opt_in.is_none() && !use_decode_wave_split_staged))
        && (1..=9).contains(&query_count)
        && committed_kv_length >= 1
        && query_heads == 24
        && descriptor.layout().heads() == 4
        && descriptor.layout().head_dim() == 256
        && descriptor.cache_encoding() == KvCacheEncoding::Mxfp8E4;
    let use_decode_wave_split_gqa_shared = use_decode_wave_split_staged32
        && expected_target == Some("gfx1030")
        && (1..=3).contains(&query_count);
    let use_decode_wave_split128 = use_decode_wave_split_staged32
        && query_count <= 3
        && committed_kv_length >= 8192
        && (expected_target == Some("gfx1201") || use_decode_wave_split_gqa_shared);
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
    let use_prefill_gqa6_qtile8_w16 = use_phase33_common_provider
        && matches!(expected_target, Some("gfx1030" | "gfx1201"))
        && query_count >= 128
        && start_position >= 1024
        && query_heads == 24
        && descriptor.layout().heads() == 4
        && descriptor.layout().head_dim() == 256
        && descriptor.cache_encoding() == KvCacheEncoding::Mxfp8E4
        && gqa6_qtile4_opt_in.is_none()
        && !force_baseline
        && !use_prefill_gqa6_blocksoftmax_q8
        && !use_any_gqa6_rocblas_f32;
    let use_prefill_gqa6_qtile4 = use_phase33_common_provider
        && query_count >= 128
        && query_heads as usize / descriptor.layout().heads() == 6
        && descriptor.layout().head_dim() == 256
        // MXFP8 E4 uses the already verified format-neutral qtile4 provider
        // by default.  An explicit value other than "1" remains a rollback;
        // FP16 and other encodings retain their existing opt-in behavior.
        && match descriptor.cache_encoding() {
            KvCacheEncoding::Mxfp8E4 => {
                gqa6_qtile4_opt_in
                    .as_deref()
                    .is_some_and(|value| value == "1")
                    || (gqa6_qtile4_opt_in.is_none()
                        && query_heads == 24
                        && descriptor.layout().heads() == 4
                        && !use_prefill_gqa6_qtile8_w16)
            }
            _ => gqa6_qtile4_opt_in
                .as_deref()
                .is_some_and(|value| value == "1"),
        }
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
    let use_scaled_prefill_gemm = scaled_prefill_gemm_enabled(
        expected_target,
        query_count,
        query_heads,
        descriptor.layout().heads() as u32,
        descriptor.layout().head_dim() as u32,
        descriptor.cache_encoding(),
        scaled_prefill_opt_in.as_deref(),
        force_baseline,
    );
    let use_prefill_gqa4_qtile4 = !use_any_gqa6_rocblas_f32
        && ((use_prefill_gqa4
            && query_count >= 128
            && !force_baseline
            && !use_scaled_prefill_gemm)
            || use_prefill_gqa6_qtile8_w16
            || use_prefill_gqa6_qtile4
            || use_prefill_gqa6_qtile4_k4_fp16
            || use_prefill_gqa6_qtile4_k8_fp16
            || use_prefill_gqa6_qtile4_k16_fp16
            || use_prefill_gqa6_qtile4_k32_fp16
            || use_prefill_gqa6_blocksoftmax
            || use_prefill_gqa6_blocksoftmax_q8);
    let (expected_kernel_id, baseline_kernel, baseline_device) = if use_decode_wave_split_staged32 {
        (
            sys::SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_DECODE_WAVE_SPLIT_STAGED_V1,
            "causal_attention.online_softmax_gqa.packed_kv.v3",
            "sllm_causal_attention_online_softmax_gqa_packed_kv_v3",
        )
    } else if use_decode_wave_split_staged {
        (
            sys::SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_DECODE_WAVE_SPLIT_STAGED_GFX1030_V1,
            "causal_attention.decode.wave8_split.staged.gfx1030.v1",
            "sllm_causal_attention_decode_wave8_split_staged_gfx1030_v1",
        )
    } else if use_gfx1201_gqa6_rocblas_f16_tail {
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
    let (expected_kernel, expected_device) = if use_decode_wave_split_staged32 {
        (
            "causal_attention.decode.wave32_split.staged.v1",
            "sllm_causal_attention_decode_wave32_split_staged_v1",
        )
    } else if use_decode_wave_split_staged {
        (
            "causal_attention.decode.wave8_split.staged.gfx1030.v1",
            "sllm_causal_attention_decode_wave8_split_staged_gfx1030_v1",
        )
    } else if use_decode_gqa6_split_p128 {
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
        } else if use_prefill_gqa6_qtile8_w16 {
            (
                "causal_attention.prefill.gqa6_qtile8_w16.mxfp8.v1",
                "sllm_causal_attention_prefill_gqa6_qtile8_w16_mxfp8_v1",
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
            != if use_decode_wave_split_staged32 || use_decode_wave_split_staged {
                2
            } else if use_gfx1201_gqa6_rocblas_f16_tail {
                5
            } else if use_any_gqa6_rocblas_f32 {
                7
            } else if use_decode_gqa4_split
                || use_decode_gqa4_split_p32
                || use_decode_gqa6_split_p128
                || use_decode_gqa6_split_p64
                || use_decode_gqa6_split_p32
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
            } else if use_prefill_gqa6_qtile8_w16 {
                512
            } else {
                sys::SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE
            }
        || Some(info.grid_size_x)
            != if use_decode_wave_split_staged32 {
                query_count
                    .checked_mul(if use_decode_wave_split_gqa_shared {
                        4
                    } else {
                        24
                    })
                    .and_then(|value| {
                        value.checked_mul(if use_decode_wave_split128 { 128 } else { 32 })
                    })
                    .and_then(|value| u32::try_from(value).ok())
            } else if use_decode_wave_split_staged {
                query_count
                    .checked_mul(24)
                    .and_then(|value| value.checked_mul(8))
                    .and_then(|value| u32::try_from(value).ok())
            } else if use_decode_gqa6_split_p128 {
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
            } else if use_prefill_gqa6_blocksoftmax_q8 {
                query_count
                    .checked_add(7)
                    .and_then(|value| (value / 8).checked_mul(descriptor.layout().heads() as u64))
                    .and_then(|value| u32::try_from(value).ok())
            } else if use_prefill_gqa6_qtile8_w16 {
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
        || !operation_range_admitted(
            start_position,
            query_count,
            committed_kv_length,
            descriptor.capacity(),
            capture_projected,
        )
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

    #[test]
    fn paged_pool_capacity_allows_one_full_parent_child_divergence() {
        assert_eq!(paged_pool_capacities(1, false).unwrap(), (1, 2));
        assert_eq!(paged_pool_capacities(127, false).unwrap(), (1, 2));
        assert_eq!(paged_pool_capacities(128, false).unwrap(), (1, 2));
        assert_eq!(paged_pool_capacities(129, false).unwrap(), (2, 4));
        assert_eq!(paged_pool_capacities(257, false).unwrap(), (3, 6));
        assert_eq!(
            paged_pool_capacities(u64::MAX, false).unwrap_err().status(),
            RuntimeStatus::MetadataOverflow
        );
        let over_u32_parent_capacity = (u64::from(u32::MAX) / 2 + 1) * 128;
        assert_eq!(
            paged_pool_capacities(over_u32_parent_capacity, false)
                .unwrap_err()
                .status(),
            RuntimeStatus::KvCapacityExceeded
        );
    }

    #[test]
    fn paged_sliding_pool_capacity_reserves_the_nine_slot_ring() {
        for capacity in [1023_u64, 1024, 1025, 1152] {
            assert_eq!(paged_pool_capacities(capacity, true).unwrap(), (9, 18));
        }
        assert_eq!(paged_pool_capacities(1024, false).unwrap(), (8, 16));
    }

    #[test]
    fn paged_adapter_admits_native_create_recipes_including_sliding_static_fp8() {
        let fp16 =
            KvStateDescriptor::new_with_storage(0, 257, 4, 256, KvCacheEncoding::Fp16).unwrap();
        let dynamic_fp8 =
            KvStateDescriptor::new_with_storage(0, 257, 4, 256, KvCacheEncoding::Fp8E4M3Fn)
                .unwrap();
        let static_fp8 =
            KvStateDescriptor::new_with_static_fp8(0, 257, 4, 256, 1.25, 0.75).unwrap();
        let nvfp4 =
            KvStateDescriptor::new_with_storage(0, 257, 4, 256, KvCacheEncoding::Nvfp4).unwrap();
        let mxfp8 = KvStateDescriptor::new_with_kv_mxfp8(
            0,
            257,
            4,
            256,
            KvCacheEncoding::Mxfp8E4,
            KvFp8PhysicalVariant::OcpE4M3Fn,
        )
        .unwrap();
        let mxfp8_e5 = KvStateDescriptor::new_with_kv_mxfp8(
            0,
            257,
            4,
            256,
            KvCacheEncoding::Mxfp8E5,
            KvFp8PhysicalVariant::OcpE5M2,
        )
        .unwrap();
        for target in ["gfx1030", "gfx1201"] {
            let context = Context::test_without_native_for_target(target);
            assert_eq!(
                paged_native_storage(&context, fp16).unwrap(),
                (
                    sys::SLLM_TENSOR_DTYPE_F16,
                    sys::SLLM_HIP_KV_ENCODING_FP16_V1,
                    0,
                    0
                )
            );
            assert_eq!(
                paged_native_storage(&context, dynamic_fp8).unwrap(),
                (
                    sys::SLLM_TENSOR_DTYPE_F8_E4M3_FN,
                    sys::SLLM_HIP_KV_ENCODING_FP8_V1,
                    0,
                    sys::SLLM_TENSOR_DTYPE_F32
                )
            );
            assert_eq!(
                paged_native_storage(&context, static_fp8).unwrap(),
                (
                    sys::SLLM_TENSOR_DTYPE_F8_E4M3_FN,
                    sys::SLLM_HIP_KV_ENCODING_FP8_STATIC_V1,
                    0,
                    sys::SLLM_TENSOR_DTYPE_F32
                )
            );
            assert_eq!(
                paged_static_scale_bits(static_fp8),
                (1.25_f32.to_bits(), 0.75_f32.to_bits())
            );
            assert_eq!(
                paged_native_storage(&context, nvfp4).unwrap(),
                (
                    sys::SLLM_TENSOR_DTYPE_U8,
                    sys::SLLM_HIP_KV_ENCODING_NVFP4_V1,
                    16,
                    sys::SLLM_TENSOR_DTYPE_F8_E4M3_FN
                )
            );
            assert_eq!(
                paged_native_storage(&context, mxfp8).unwrap(),
                (
                    sys::SLLM_TENSOR_DTYPE_F8_E4M3_FN,
                    sys::SLLM_HIP_KV_ENCODING_MXFP8_E4_V1,
                    32,
                    sys::SLLM_TENSOR_DTYPE_U8
                )
            );
            let e5 = paged_native_storage(&context, mxfp8_e5);
            if target == "gfx1030" {
                assert_eq!(
                    e5.unwrap(),
                    (
                        sys::SLLM_TENSOR_DTYPE_F8_E5M2,
                        sys::SLLM_HIP_KV_ENCODING_MXFP8_E5_V1,
                        32,
                        sys::SLLM_TENSOR_DTYPE_U8
                    )
                );
            } else {
                assert_eq!(e5.unwrap_err().status(), RuntimeStatus::Unsupported);
            }
        }
        for target in [Some("gfx942"), Some("gfx1030:sramecc+:xnack-"), None] {
            let context = match target {
                Some(target) => Context::test_without_native_for_target(target),
                None => Context::test_without_native(),
            };
            assert_eq!(
                paged_native_storage(&context, fp16).unwrap_err().status(),
                RuntimeStatus::Unsupported
            );
        }
        let unsupported = block16_descriptor(
            KvCacheEncoding::Fp8E4M3Block16,
            KvFp8PhysicalVariant::OcpE4M3Fn,
        );
        assert_eq!(
            paged_native_storage(
                &Context::test_without_native_for_target("gfx1030"),
                unsupported
            )
            .unwrap_err()
            .status(),
            RuntimeStatus::Unsupported
        );

        let sliding =
            KvStateDescriptor::new_with_static_fp8_sliding(0, 2048, 4, 256, 1024).unwrap();
        assert_eq!(
            paged_native_storage(&Context::test_without_native_for_target("gfx1030"), sliding)
                .unwrap(),
            (
                sys::SLLM_TENSOR_DTYPE_F8_E4M3_FN,
                sys::SLLM_HIP_KV_ENCODING_FP8_STATIC_V1,
                0,
                sys::SLLM_TENSOR_DTYPE_F32
            )
        );
        assert_eq!(
            paged_native_storage(&Context::test_without_native_for_target("gfx1201"), sliding)
                .unwrap(),
            (
                sys::SLLM_TENSOR_DTYPE_F8_E4M3_FN,
                sys::SLLM_HIP_KV_ENCODING_FP8_STATIC_V1,
                0,
                sys::SLLM_TENSOR_DTYPE_F32
            )
        );
    }

    #[test]
    fn paged_view_info_initializes_additive_counts_without_vmm_fields() {
        let info = empty_paged_view_info();
        assert_eq!(
            info.struct_size,
            size_of::<sys::sllm_kv_paged_view_info_t>() as u32
        );
        assert_eq!(info.info_version, sys::SLLM_HIP_KV_PAGED_VIEW_INFO_VERSION);
        assert_eq!(info.token_block_size, 0);
        assert_eq!(info.committed_bytes_per_plane, [0; KV_PAGED_PLANE_COUNT]);
        assert_eq!(info.committed_bytes_total, 0);
    }

    #[test]
    fn projected_operation_range_requires_bounded_capture_admission() {
        assert!(!operation_range_admitted(20, 3, 23, 22, false));
        assert!(operation_range_admitted(20, 3, 23, 22, true));
        assert!(operation_range_admitted(21, 9, 30, 22, true));
        assert!(!operation_range_admitted(21, 10, 31, 22, true));
        assert!(!operation_range_admitted(22, 9, 31, 22, true));
        assert!(!operation_range_admitted(20, 3, 24, 22, true));
        assert!(operation_range_admitted(19, 3, 22, 22, false));
    }

    #[test]
    fn capture_projected_causal_evidence_requires_scope_flag() {
        let descriptor = KvStateDescriptor::new_with_kv_mxfp8(
            0,
            22,
            4,
            256,
            KvCacheEncoding::Mxfp8E4,
            KvFp8PhysicalVariant::OcpE4M3Fn,
        )
        .unwrap();
        let context = Context::test_without_native_for_target("gfx1030");
        let mut info = empty_causal_attention_info();
        info.backend = sys::SLLM_BACKEND_HIP;
        info.dispatch_id = 1;
        info.dispatch_count = 2;
        info.kernel_id = sys::SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_DECODE_WAVE_SPLIT_STAGED_V1;
        info.workgroup_size_x = sys::SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE;
        info.grid_size_x = 9 * 24 * 32;
        info.query_count = 9;
        info.start_position = 21;
        info.committed_kv_length = 30;
        info.q_heads = 24;
        info.kv_heads = 4;
        info.head_dim = 256;
        info.scale_denominator = 16;
        set_test_c_string(
            &mut info.kernel_symbol,
            "causal_attention.decode.wave32_split.staged.v1",
        );
        set_test_c_string(
            &mut info.device_symbol,
            "sllm_causal_attention_decode_wave32_split_staged_v1",
        );
        set_test_c_string(&mut info.gcn_arch_name, "gfx1030");
        assert!(
            validate_causal_attention_info_impl(
                &info, &context, 21, 30, descriptor, 24, None, None, None, None, false,
            )
            .is_err()
        );
        assert!(
            validate_causal_attention_info_impl(
                &info, &context, 21, 30, descriptor, 24, None, None, None, None, true,
            )
            .is_ok()
        );
    }

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
                false,
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
                    false,
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
                    false,
                )
                .is_err()
            );
        }
    }

    #[test]
    fn mxfp8_e4_packed_attention_metadata_accepts_chain_shape() {
        let context = Context::test_without_native();
        let descriptor = KvStateDescriptor::new_with_kv_mxfp8(
            0,
            65,
            4,
            256,
            KvCacheEncoding::Mxfp8E4,
            KvFp8PhysicalVariant::OcpE4M3Fn,
        )
        .unwrap();
        let mut info = empty_causal_attention_info();
        info.backend = sys::SLLM_BACKEND_HIP;
        info.dispatch_id = 11;
        info.dispatch_count = 1;
        info.kernel_id = sys::SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_PACKED_KV_V3;
        info.workgroup_size_x = sys::SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE;
        info.grid_size_x = 24;
        info.query_count = 1;
        info.start_position = 32;
        info.committed_kv_length = 33;
        info.q_heads = 24;
        info.kv_heads = 4;
        info.head_dim = 256;
        info.scale_denominator = 16;
        set_test_c_string(
            &mut info.kernel_symbol,
            "causal_attention.online_softmax_gqa.packed_kv.v3",
        );
        set_test_c_string(
            &mut info.device_symbol,
            "sllm_causal_attention_online_softmax_gqa_packed_kv_v3",
        );
        set_test_c_string(&mut info.gcn_arch_name, "gfx1030");

        let evidence = validate_causal_attention_info(
            &info, &context, 32, 33, descriptor, 24, None, None, false,
        )
        .unwrap();
        assert_eq!(evidence.dispatch_count, 1);
        assert_eq!(evidence.query_count, 1);
        assert_eq!(evidence.start_position, 32);
        assert_eq!(evidence.committed_kv_length, 33);
        assert_eq!(
            evidence.kernel_id,
            sys::SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_PACKED_KV_V3
        );
        assert!(!evidence.fallback_allowed);
        assert!(!evidence.fallback_used);
    }

    #[test]
    fn paged_attention_metadata_requires_native_provider_identity() {
        let context = Context::test_without_native_for_target("gfx1030");
        let cases = [
            (
                KvStateDescriptor::new_with_kv_mxfp8(
                    0,
                    65,
                    4,
                    256,
                    KvCacheEncoding::Mxfp8E4,
                    KvFp8PhysicalVariant::OcpE4M3Fn,
                )
                .unwrap(),
                24,
                128,
                2,
                192,
                "causal_attention.paged_decode.gqa6_m1_m5.mxfp8_e4.v1",
                "sllm_causal_attention_paged_decode_gqa6_m1_m5_mxfp8_e4_v1",
                sys::SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_PAGED_DECODE_GQA6_V1,
            ),
            (
                KvStateDescriptor::new_with_storage(0, 65, 4, 256, KvCacheEncoding::Fp16).unwrap(),
                16,
                16,
                1,
                sys::SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE,
                "causal_attention.paged_decode.fp16_gqa.v1",
                "sllm_causal_attention_paged_decode_fp16_gqa_v1",
                sys::SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_PAGED_DECODE_FP16_V1,
            ),
            (
                KvStateDescriptor::new_with_storage(0, 65, 4, 256, KvCacheEncoding::Fp8E4M3Fn)
                    .unwrap(),
                8,
                8,
                1,
                sys::SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE,
                "causal_attention.paged.generic_formats.v1",
                "sllm_causal_attention_paged_generic_formats_v1",
                sys::SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_PAGED_GENERIC_FORMATS_V1,
            ),
        ];
        for (
            descriptor,
            query_heads,
            grid_size,
            dispatch_count,
            workgroup_size,
            kernel_symbol,
            device_symbol,
            kernel_id,
        ) in cases
        {
            let mut info = empty_causal_attention_info();
            info.backend = sys::SLLM_BACKEND_HIP;
            info.dispatch_id = 1;
            info.dispatch_count = dispatch_count;
            info.kernel_id = kernel_id;
            info.workgroup_size_x = workgroup_size;
            info.grid_size_x = grid_size;
            info.query_count = 1;
            info.start_position = 32;
            info.committed_kv_length = 33;
            info.q_heads = query_heads;
            info.kv_heads = 4;
            info.head_dim = 256;
            info.scale_denominator = 16;
            set_test_c_string(&mut info.kernel_symbol, kernel_symbol);
            set_test_c_string(&mut info.device_symbol, device_symbol);
            set_test_c_string(&mut info.gcn_arch_name, "gfx1030");
            let evidence = validate_attention_info_for_storage(
                &info,
                &context,
                KvStateStorageKind::Paged,
                32,
                33,
                descriptor,
                query_heads,
                None,
                None,
                false,
            )
            .unwrap();
            assert_eq!(evidence.kernel_id, kernel_id);
            assert_eq!(evidence.kernel_symbol, kernel_symbol);
            assert_eq!(evidence.device_symbol, device_symbol);
            assert!(!evidence.fallback_allowed);
            assert!(!evidence.fallback_used);

            let mut wrong_provider = info;
            wrong_provider.kernel_id = sys::SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_ONLINE_SOFTMAX_V2;
            assert!(
                validate_attention_info_for_storage(
                    &wrong_provider,
                    &context,
                    KvStateStorageKind::Paged,
                    32,
                    33,
                    descriptor,
                    query_heads,
                    None,
                    None,
                    false,
                )
                .is_err()
            );
        }
    }

    #[test]
    fn paged_c1_m3_dispatch_metadata_is_exact_and_boundary_gated() {
        let descriptor = KvStateDescriptor::new_with_kv_mxfp8(
            0,
            8193,
            4,
            256,
            KvCacheEncoding::Mxfp8E4,
            KvFp8PhysicalVariant::OcpE4M3Fn,
        )
        .unwrap();
        for target in ["gfx1030", "gfx1201"] {
            let context = Context::test_without_native_for_target(target);
            let mut info = empty_causal_attention_info();
            info.backend = sys::SLLM_BACKEND_HIP;
            info.dispatch_id = 118;
            info.dispatch_count = 2;
            info.kernel_id = sys::SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_PAGED_DECODE_GQA6_C1_M3_V1;
            info.workgroup_size_x = 192;
            info.grid_size_x = 4 * 128;
            info.query_count = 3;
            info.start_position = 8190;
            info.committed_kv_length = 8193;
            info.q_heads = 24;
            info.kv_heads = 4;
            info.head_dim = 256;
            info.scale_denominator = 16;
            set_test_c_string(
                &mut info.kernel_symbol,
                "causal_attention.paged_decode.gqa6_c1_m3.mxfp8_e4.v1",
            );
            set_test_c_string(
                &mut info.device_symbol,
                "sllm_causal_attention_paged_decode_gqa6_c1_m3_mxfp8_e4_v1",
            );
            set_test_c_string(&mut info.gcn_arch_name, target);
            let validate = |candidate: &sys::sllm_causal_attention_dispatch_info_t| {
                validate_attention_info_for_storage(
                    candidate,
                    &context,
                    KvStateStorageKind::Paged,
                    candidate.start_position,
                    candidate.committed_kv_length,
                    descriptor,
                    24,
                    None,
                    None,
                    false,
                )
            };
            assert!(validate(&info).is_ok(), "{target}");

            let mut below_threshold = info;
            below_threshold.start_position = 8188;
            below_threshold.committed_kv_length = 8191;
            assert!(validate(&below_threshold).is_err(), "{target}");

            let mut wrong_rows = info;
            wrong_rows.query_count = 2;
            wrong_rows.start_position = 8191;
            assert!(validate(&wrong_rows).is_err(), "{target}");

            assert!(
                validate_attention_info_for_storage(
                    &info,
                    &context,
                    KvStateStorageKind::Paged,
                    8190,
                    8193,
                    descriptor,
                    24,
                    None,
                    Some(1.0 / 16.0),
                    false,
                )
                .is_err(),
                "{target}"
            );
        }
    }

    #[test]
    fn paged_sliding_static_fp8_metadata_is_exact_on_reviewed_targets() {
        let descriptor =
            KvStateDescriptor::new_with_static_fp8_sliding(0, 262_144, 8, 256, 1024).unwrap();
        for target in ["gfx1030", "gfx1201"] {
            let context = Context::test_without_native_for_target(target);
            for (start, committed) in [(0_u64, 1023_u64), (0, 1024), (1024, 1025)] {
                let query_count = committed - start;
                let mut info = empty_causal_attention_info();
                info.backend = sys::SLLM_BACKEND_HIP;
                info.dispatch_id = 23;
                info.dispatch_count = 1;
                info.kernel_id =
                    sys::SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_PAGED_SLIDING_STATIC_FP8_V1;
                info.workgroup_size_x = sys::SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE;
                info.grid_size_x = u32::try_from(query_count * 16).unwrap();
                info.query_count = query_count;
                info.start_position = start;
                info.committed_kv_length = committed;
                info.q_heads = 16;
                info.kv_heads = 8;
                info.head_dim = 256;
                info.scale_denominator = 0;
                let retained_start = committed.saturating_sub(1024);
                info.reserved = [
                    1024,
                    0,
                    retained_start as u32,
                    (retained_start >> 32) as u32,
                    1.0_f32.to_bits(),
                    1,
                    0,
                    0,
                ];
                set_test_c_string(
                    &mut info.kernel_symbol,
                    "causal_attention.paged_sliding_static_fp8_ring.v1",
                );
                set_test_c_string(
                    &mut info.device_symbol,
                    "sllm_causal_attention_paged_sliding_static_fp8_ring_v1",
                );
                set_test_c_string(&mut info.gcn_arch_name, target);

                let evidence = validate_attention_info_for_storage(
                    &info,
                    &context,
                    KvStateStorageKind::Paged,
                    start,
                    committed,
                    descriptor,
                    16,
                    Some(1024),
                    Some(1.0),
                    false,
                )
                .unwrap_or_else(|error| panic!("{target} {start}->{committed}: {error:?}"));
                assert_eq!(
                    evidence.kernel_id,
                    sys::SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_PAGED_SLIDING_STATIC_FP8_V1
                );
                assert_eq!(evidence.sliding_window, 1024);
                assert_eq!(evidence.retained_start, retained_start);
                assert_eq!(evidence.score_scale_bits, 1.0_f32.to_bits());
                assert!(evidence.explicit_score_scale);
                assert!(!evidence.fallback_allowed);
                assert!(!evidence.fallback_used);

                let mut wrong_scale = info;
                wrong_scale.reserved[4] = (1.0_f32 / 16.0).to_bits();
                assert!(
                    validate_attention_info_for_storage(
                        &wrong_scale,
                        &context,
                        KvStateStorageKind::Paged,
                        start,
                        committed,
                        descriptor,
                        16,
                        Some(1024),
                        Some(1.0),
                        false,
                    )
                    .is_err()
                );
            }
        }
    }

    #[test]
    fn paged_full_static_fp8_metadata_accepts_gemma_unit_scale() {
        let descriptor =
            KvStateDescriptor::new_with_static_fp8(0, 262_144, 2, 512, 1.0, 1.0).unwrap();
        for target in ["gfx1030", "gfx1201"] {
            let context = Context::test_without_native_for_target(target);
            let mut info = empty_causal_attention_info();
            info.backend = sys::SLLM_BACKEND_HIP;
            info.dispatch_id = 31;
            info.dispatch_count = 1;
            info.kernel_id = sys::SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_PAGED_GENERIC_FORMATS_V1;
            info.workgroup_size_x = sys::SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE;
            info.grid_size_x = 16;
            info.query_count = 1;
            info.start_position = 17;
            info.committed_kv_length = 18;
            info.q_heads = 16;
            info.kv_heads = 2;
            info.head_dim = 512;
            info.scale_denominator = 0;
            info.reserved = [0, 0, 0, 0, 1.0_f32.to_bits(), 1, 0, 0];
            set_test_c_string(
                &mut info.kernel_symbol,
                "causal_attention.paged.generic_formats.v1",
            );
            set_test_c_string(
                &mut info.device_symbol,
                "sllm_causal_attention_paged_generic_formats_v1",
            );
            set_test_c_string(&mut info.gcn_arch_name, target);
            let evidence = validate_attention_info_for_storage(
                &info,
                &context,
                KvStateStorageKind::Paged,
                17,
                18,
                descriptor,
                16,
                None,
                Some(1.0),
                false,
            )
            .unwrap_or_else(|error| panic!("{target}: {error:?}"));
            assert_eq!(evidence.kernel_id, info.kernel_id);
            assert_eq!(evidence.score_scale_bits, 1.0_f32.to_bits());
            assert!(evidence.explicit_score_scale);
            assert!(!evidence.fallback_allowed);
            assert!(!evidence.fallback_used);
        }
    }

    #[test]
    fn paged_full_static_fp8_nonunit_scales_use_implicit_score_metadata() {
        let descriptor =
            KvStateDescriptor::new_with_static_fp8(0, 262_144, 2, 512, 0.5, 0.75).unwrap();
        let context = Context::test_without_native_for_target("gfx1201");
        let (scale_denominator, scale_bits, reserved) = implicit_attention_scale_evidence(512);
        let mut info = empty_causal_attention_info();
        info.backend = sys::SLLM_BACKEND_HIP;
        info.dispatch_id = 32;
        info.dispatch_count = 1;
        info.kernel_id = sys::SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_PAGED_GENERIC_FORMATS_V1;
        info.workgroup_size_x = sys::SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE;
        info.grid_size_x = 16;
        info.query_count = 1;
        info.start_position = 17;
        info.committed_kv_length = 18;
        info.q_heads = 16;
        info.kv_heads = 2;
        info.head_dim = 512;
        info.scale_denominator = scale_denominator;
        info.reserved = reserved;
        set_test_c_string(
            &mut info.kernel_symbol,
            "causal_attention.paged.generic_formats.v1",
        );
        set_test_c_string(
            &mut info.device_symbol,
            "sllm_causal_attention_paged_generic_formats_v1",
        );
        set_test_c_string(&mut info.gcn_arch_name, "gfx1201");
        let evidence = validate_attention_info_for_storage(
            &info,
            &context,
            KvStateStorageKind::Paged,
            17,
            18,
            descriptor,
            16,
            None,
            None,
            false,
        )
        .unwrap();
        assert_eq!(evidence.score_scale_bits, scale_bits);
        assert!(!evidence.explicit_score_scale);
        assert!(!evidence.fallback_allowed);
        assert!(!evidence.fallback_used);
    }

    #[test]
    fn paged_gqa6_prefill_metadata_uses_target_specific_wave_provider() {
        let descriptor = KvStateDescriptor::new_with_kv_mxfp8(
            0,
            1024,
            4,
            256,
            KvCacheEncoding::Mxfp8E4,
            KvFp8PhysicalVariant::OcpE4M3Fn,
        )
        .unwrap();
        for (target, workgroup, grid, kernel_symbol, device_symbol) in [
            (
                "gfx1030",
                512,
                4,
                "causal_attention.paged_prefill.gqa6_qtile8.mxfp8_e4.v1",
                "sllm_causal_attention_paged_prefill_gqa6_qtile8_mxfp8_e4_v1",
            ),
            (
                "gfx1201",
                256,
                144,
                "causal_attention.paged_prefill.gqa6_wave.gfx1201.mxfp8_e4.v1",
                "sllm_causal_attention_paged_prefill_gqa6_wave_gfx1201_mxfp8_e4_",
            ),
        ] {
            let context = Context::test_without_native_for_target(target);
            let mut info = empty_causal_attention_info();
            info.backend = sys::SLLM_BACKEND_HIP;
            info.dispatch_id = 17;
            info.dispatch_count = 1;
            info.kernel_id = sys::SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_PAGED_PREFILL_GQA6_V1;
            info.workgroup_size_x = workgroup;
            info.grid_size_x = grid;
            info.query_count = 6;
            info.start_position = 128;
            info.committed_kv_length = 134;
            info.q_heads = 24;
            info.kv_heads = 4;
            info.head_dim = 256;
            info.scale_denominator = 16;
            set_test_c_string(&mut info.kernel_symbol, kernel_symbol);
            set_test_c_string(&mut info.device_symbol, device_symbol);
            set_test_c_string(&mut info.gcn_arch_name, target);
            let evidence = validate_attention_info_for_storage(
                &info,
                &context,
                KvStateStorageKind::Paged,
                128,
                134,
                descriptor,
                24,
                None,
                None,
                false,
            )
            .unwrap_or_else(|error| panic!("target {target}: {error:?}"));
            assert_eq!(evidence.workgroup_size_x, workgroup);
            assert_eq!(evidence.grid_size_x, grid);
            assert_eq!(evidence.kernel_symbol, kernel_symbol);
            assert_eq!(evidence.device_symbol, device_symbol);
        }
    }

    #[test]
    fn paged_fp16_gqa4_prefill_metadata_accepts_legacy_shared_provider_shape() {
        let descriptor =
            KvStateDescriptor::new_with_storage(0, 512, 4, 256, KvCacheEncoding::Fp16).unwrap();
        for target in ["gfx1030", "gfx1201"] {
            for query_count in [65_u64, 85, 265] {
                let context = Context::test_without_native_for_target(target);
                let mut info = empty_causal_attention_info();
                info.backend = sys::SLLM_BACKEND_HIP;
                info.dispatch_id = 18;
                info.dispatch_count = 1;
                info.kernel_id = sys::SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_PAGED_PREFILL_FP16_V1;
                info.workgroup_size_x = sys::SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE;
                info.grid_size_x = (query_count * 16) as u32;
                info.query_count = query_count;
                info.start_position = 0;
                info.committed_kv_length = query_count;
                info.q_heads = 16;
                info.kv_heads = 4;
                info.head_dim = 256;
                info.scale_denominator = 16;
                set_test_c_string(
                    &mut info.kernel_symbol,
                    "causal_attention.paged_prefill.gqa4_shared.v1",
                );
                set_test_c_string(
                    &mut info.device_symbol,
                    "sllm_causal_attention_paged_prefill_gqa4_shared_v1",
                );
                set_test_c_string(&mut info.gcn_arch_name, target);
                let evidence = validate_attention_info_for_storage(
                    &info,
                    &context,
                    KvStateStorageKind::Paged,
                    0,
                    query_count,
                    descriptor,
                    16,
                    None,
                    None,
                    false,
                )
                .unwrap_or_else(|error| panic!("target {target} M={query_count}: {error:?}"));
                assert_eq!(
                    evidence.kernel_symbol,
                    "causal_attention.paged_prefill.gqa4_shared.v1"
                );
                assert_eq!(
                    evidence.device_symbol,
                    "sllm_causal_attention_paged_prefill_gqa4_shared_v1"
                );
                assert_eq!(evidence.grid_size_x, query_count as u32 * 16);
            }
        }
    }

    #[test]
    fn paged_gfx1201_prefill_metadata_keeps_the_long_qtile_gate() {
        let descriptor = KvStateDescriptor::new_with_kv_mxfp8(
            0,
            2048,
            4,
            256,
            KvCacheEncoding::Mxfp8E4,
            KvFp8PhysicalVariant::OcpE4M3Fn,
        )
        .unwrap();
        let context = Context::test_without_native_for_target("gfx1201");
        for (start, workgroup, grid, kernel, device) in [
            (
                0,
                256,
                128 * 24,
                "causal_attention.paged_prefill.gqa6_qtile4.mxfp8_e4.v1",
                "sllm_causal_attention_paged_prefill_gqa6_qtile4_mxfp8_e4_v1",
            ),
            (
                1024,
                512,
                (128 / 8) * 4,
                "causal_attention.paged_prefill.gqa6_qtile8.mxfp8_e4.v1",
                "sllm_causal_attention_paged_prefill_gqa6_qtile8_mxfp8_e4_v1",
            ),
        ] {
            let mut info = empty_causal_attention_info();
            info.backend = sys::SLLM_BACKEND_HIP;
            info.dispatch_id = 17;
            info.dispatch_count = 1;
            info.kernel_id = sys::SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_PAGED_PREFILL_GQA6_V1;
            info.workgroup_size_x = workgroup;
            info.grid_size_x = grid;
            info.query_count = 128;
            info.start_position = start;
            info.committed_kv_length = start + 128;
            info.q_heads = 24;
            info.kv_heads = 4;
            info.head_dim = 256;
            info.scale_denominator = 16;
            set_test_c_string(&mut info.kernel_symbol, kernel);
            set_test_c_string(&mut info.device_symbol, device);
            set_test_c_string(&mut info.gcn_arch_name, "gfx1201");
            let evidence = validate_attention_info_for_storage(
                &info,
                &context,
                KvStateStorageKind::Paged,
                start,
                start + 128,
                descriptor,
                24,
                None,
                None,
                false,
            )
            .unwrap_or_else(|error| panic!("start {start}: {error:?}"));
            assert_eq!(evidence.kernel_symbol, kernel);
        }
    }

    #[test]
    fn mxfp8_e4_qtile8_w16_metadata_accepts_long_gfx1030_gfx1201_prefix() {
        let descriptor = KvStateDescriptor::new_with_kv_mxfp8(
            0,
            16384,
            4,
            256,
            KvCacheEncoding::Mxfp8E4,
            KvFp8PhysicalVariant::OcpE4M3Fn,
        )
        .unwrap();
        for target in ["gfx1030", "gfx1201"] {
            let context = Context::test_without_native_for_target(target);
            for query_count in [128_u64, 129, 1024] {
                let start_position = 1024;
                let committed_kv_length = start_position + query_count;
                let mut info = empty_causal_attention_info();
                info.backend = sys::SLLM_BACKEND_HIP;
                info.dispatch_id = 8500 + query_count;
                info.dispatch_count = 1;
                info.kernel_id = sys::SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_PACKED_KV_V3;
                info.workgroup_size_x = 512;
                info.grid_size_x = u32::try_from(query_count.div_ceil(8) * 4).unwrap();
                info.query_count = query_count;
                info.start_position = start_position;
                info.committed_kv_length = committed_kv_length;
                info.q_heads = 24;
                info.kv_heads = 4;
                info.head_dim = 256;
                info.scale_denominator = 16;
                set_test_c_string(
                    &mut info.kernel_symbol,
                    "causal_attention.prefill.gqa6_qtile8_w16.mxfp8.v1",
                );
                set_test_c_string(
                    &mut info.device_symbol,
                    "sllm_causal_attention_prefill_gqa6_qtile8_w16_mxfp8_v1",
                );
                set_test_c_string(&mut info.gcn_arch_name, target);

                let evidence = validate_causal_attention_info(
                    &info,
                    &context,
                    start_position,
                    committed_kv_length,
                    descriptor,
                    24,
                    None,
                    None,
                    false,
                )
                .unwrap();
                assert_eq!(evidence.workgroup_size_x, 512);
                assert_eq!(evidence.grid_size_x, info.grid_size_x);
                assert_eq!(
                    evidence.kernel_symbol,
                    "causal_attention.prefill.gqa6_qtile8_w16.mxfp8.v1"
                );
                assert_eq!(
                    evidence.device_symbol,
                    "sllm_causal_attention_prefill_gqa6_qtile8_w16_mxfp8_v1"
                );
            }
        }
    }

    #[test]
    fn mxfp8_e4_staged_decode_metadata_accepts_exact_opt_in_and_rejects_invalid() {
        let context = Context::test_without_native_for_target("gfx1030");
        let descriptor = KvStateDescriptor::new_with_kv_mxfp8(
            0,
            8192,
            4,
            256,
            KvCacheEncoding::Mxfp8E4,
            KvFp8PhysicalVariant::OcpE4M3Fn,
        )
        .unwrap();
        for query_count in 1..=4_u64 {
            let start_position = 1024;
            let committed_kv_length = start_position + query_count;
            let mut info = empty_causal_attention_info();
            info.backend = sys::SLLM_BACKEND_HIP;
            info.dispatch_id = 80 + query_count;
            info.dispatch_count = 2;
            info.kernel_id =
                sys::SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_DECODE_WAVE_SPLIT_STAGED_GFX1030_V1;
            info.workgroup_size_x = sys::SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE;
            info.grid_size_x = u32::try_from(query_count * 24 * 8).unwrap();
            info.query_count = query_count;
            info.start_position = start_position;
            info.committed_kv_length = committed_kv_length;
            info.q_heads = 24;
            info.kv_heads = 4;
            info.head_dim = 256;
            info.scale_denominator = 16;
            set_test_c_string(
                &mut info.kernel_symbol,
                "causal_attention.decode.wave8_split.staged.gfx1030.v1",
            );
            set_test_c_string(
                &mut info.device_symbol,
                "sllm_causal_attention_decode_wave8_split_staged_gfx1030_v1",
            );
            set_test_c_string(&mut info.gcn_arch_name, "gfx1030");

            let evidence = validate_causal_attention_info_with_staged_opt_in(
                &info,
                &context,
                start_position,
                committed_kv_length,
                descriptor,
                24,
                None,
                None,
                Some(std::ffi::OsStr::new("1")),
            )
            .unwrap();
            assert_eq!(evidence.kernel_id, info.kernel_id);
            assert_eq!(evidence.dispatch_count, 2);
            assert_eq!(evidence.grid_size_x, query_count as u32 * 24 * 8);
            assert_eq!(evidence.query_count, query_count);

            assert!(
                validate_causal_attention_info_with_staged_opt_in(
                    &info,
                    &context,
                    start_position,
                    committed_kv_length,
                    descriptor,
                    24,
                    None,
                    None,
                    None,
                )
                .is_err()
            );

            let mut fallback = info;
            fallback.fallback_used = 1;
            assert!(
                validate_causal_attention_info_with_staged_opt_in(
                    &fallback,
                    &context,
                    start_position,
                    committed_kv_length,
                    descriptor,
                    24,
                    None,
                    None,
                    Some(std::ffi::OsStr::new("1")),
                )
                .is_err()
            );
        }
    }

    #[test]
    fn mxfp8_e4_staged32_metadata_accepts_short_rows_on_both_targets() {
        let descriptor = KvStateDescriptor::new_with_kv_mxfp8(
            0,
            8192,
            4,
            256,
            KvCacheEncoding::Mxfp8E4,
            KvFp8PhysicalVariant::OcpE4M3Fn,
        )
        .unwrap();
        for target in ["gfx1030", "gfx1201"] {
            let context = Context::test_without_native_for_target(target);
            for query_count in 1..=10_u64 {
                let start_position = 0;
                let committed_kv_length = start_position + query_count;
                let mut info = empty_causal_attention_info();
                info.backend = sys::SLLM_BACKEND_HIP;
                info.dispatch_id = 9300 + query_count;
                info.dispatch_count = 2;
                info.kernel_id =
                    sys::SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_DECODE_WAVE_SPLIT_STAGED_V1;
                info.workgroup_size_x = sys::SLLM_HIP_CAUSAL_ATTENTION_WORKGROUP_SIZE;
                let grid_heads = if target == "gfx1030" && query_count <= 3 {
                    4
                } else {
                    24
                };
                info.grid_size_x = u32::try_from(query_count * grid_heads * 32).unwrap();
                info.query_count = query_count;
                info.start_position = start_position;
                info.committed_kv_length = committed_kv_length;
                info.q_heads = 24;
                info.kv_heads = 4;
                info.head_dim = 256;
                info.scale_denominator = 16;
                set_test_c_string(
                    &mut info.kernel_symbol,
                    "causal_attention.decode.wave32_split.staged.v1",
                );
                set_test_c_string(
                    &mut info.device_symbol,
                    "sllm_causal_attention_decode_wave32_split_staged_v1",
                );
                set_test_c_string(&mut info.gcn_arch_name, target);

                let evidence = validate_causal_attention_info_with_staged_opt_ins(
                    &info,
                    &context,
                    start_position,
                    committed_kv_length,
                    descriptor,
                    24,
                    None,
                    None,
                    None,
                    None,
                );
                if query_count == 10 {
                    assert!(evidence.is_err());
                    continue;
                }
                let evidence = evidence.unwrap();
                assert_eq!(evidence.kernel_id, info.kernel_id);
                assert_eq!(evidence.dispatch_count, 2);
                assert_eq!(
                    evidence.grid_size_x,
                    query_count as u32 * grid_heads as u32 * 32
                );
                assert_eq!(evidence.query_count, query_count);
            }
        }
    }

    #[test]
    fn mxfp8_e4_split128_metadata_uses_long_context_and_query_scope() {
        let descriptor = KvStateDescriptor::new_with_kv_mxfp8(
            0,
            10000,
            4,
            256,
            KvCacheEncoding::Mxfp8E4,
            KvFp8PhysicalVariant::OcpE4M3Fn,
        )
        .unwrap();
        for target in ["gfx1030", "gfx1201"] {
            let context = Context::test_without_native_for_target(target);
            for length in [8191_u64, 8192, 8193] {
                for m in 1..=4_u64 {
                    let shared = target == "gfx1030" && m <= 3;
                    let heads = if shared { 4 } else { 24 };
                    let splits = if length >= 8192 && m <= 3 { 128 } else { 32 };
                    let mut info = empty_causal_attention_info();
                    info.backend = sys::SLLM_BACKEND_HIP;
                    info.dispatch_id = 93128;
                    info.dispatch_count = 2;
                    info.kernel_id =
                        sys::SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_DECODE_WAVE_SPLIT_STAGED_V1;
                    info.workgroup_size_x = 256;
                    info.grid_size_x = (m * heads * splits) as u32;
                    info.query_count = m;
                    info.start_position = length - m;
                    info.committed_kv_length = length;
                    info.q_heads = 24;
                    info.kv_heads = 4;
                    info.head_dim = 256;
                    info.scale_denominator = 16;
                    set_test_c_string(
                        &mut info.kernel_symbol,
                        "causal_attention.decode.wave32_split.staged.v1",
                    );
                    set_test_c_string(
                        &mut info.device_symbol,
                        "sllm_causal_attention_decode_wave32_split_staged_v1",
                    );
                    set_test_c_string(&mut info.gcn_arch_name, target);
                    assert!(
                        validate_causal_attention_info_with_staged_opt_ins(
                            &info,
                            &context,
                            length - m,
                            length,
                            descriptor,
                            24,
                            None,
                            None,
                            None,
                            None,
                        )
                        .is_ok()
                    );
                }
            }
        }
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
    fn decode_gqa6_split_p64_guard_requires_explicit_opt_in_and_rollback() {
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
        for (target, committed_kv_length) in [(Some("gfx1030"), 8192), (Some("gfx1201"), 4096)] {
            assert!(!decode_gqa6_split_p64_enabled(
                target,
                1,
                committed_kv_length,
                24,
                4,
                256,
                KvCacheEncoding::Fp16,
                None,
                false,
            ));
            for opt_in in ["0", "unknown"] {
                assert!(!decode_gqa6_split_p64_enabled(
                    target,
                    1,
                    committed_kv_length,
                    24,
                    4,
                    256,
                    KvCacheEncoding::Fp16,
                    Some(std::ffi::OsStr::new(opt_in)),
                    false,
                ));
            }
        }
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
        assert!(!decode_gqa6_split_p64_enabled(
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
    fn resource_sendability_contract_is_preserved() {
        static_assertions::assert_impl_all!(KvStateResource: Send, Sync);
        static_assertions::assert_impl_all!(KvAppendCompletion: Send);
        static_assertions::assert_impl_all!(CausalAttentionCompletion: Send);
    }

    #[test]
    fn abi_layout_fields_have_expected_rust_sizes() {
        assert_eq!(size_of::<sys::sllm_kv_append_desc_t>(), 416);
        assert_eq!(size_of::<sys::sllm_kv_append_info_t>(), 304);
        assert_eq!(size_of::<sys::sllm_causal_attention_desc_t>(), 424);
        assert_eq!(size_of::<sys::sllm_causal_attention_dispatch_info_t>(), 312);
        assert_eq!(size_of::<sys::sllm_kv_paged_state_fork_info_t>(), 152);
        let info = empty_paged_fork_info();
        assert_eq!(
            info.struct_size as usize,
            size_of::<sys::sllm_kv_paged_state_fork_info_t>()
        );
        assert_eq!(
            info.info_version,
            sys::SLLM_HIP_KV_PAGED_STATE_FORK_INFO_VERSION
        );
        assert_eq!(info.shared_physical_blocks, 0);
    }
}
