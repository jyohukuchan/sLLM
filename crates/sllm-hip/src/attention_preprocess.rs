//! Safe C3a1 attention-preprocess preparation and asynchronous execution.

use std::mem::size_of;
use std::ptr::NonNull;
use std::sync::Arc;

use sllm_core::{AttentionPreprocessContract, SemanticOpDescriptor, SemanticOpKind};
use sllm_hip_sys as sys;

use crate::runtime::{
    Completion, CompletionState, Context, Queue, RuntimeError, RuntimeStatus,
    enqueue_attention_preprocess_cleanup, ensure_ok, release_attention_preprocess_plan_once, sink,
};
use crate::{HipBackend, TensorBinding};

#[derive(Clone, Debug)]
pub struct AttentionPreprocessDescriptor {
    packed_q_gate: TensorBinding,
    k: TensorBinding,
    q_raw_scale: TensorBinding,
    k_raw_scale: TensorBinding,
    positions: TensorBinding,
    q_output: TensorBinding,
    gate_output: TensorBinding,
    k_output: TensorBinding,
    semantic: Arc<SemanticOpDescriptor>,
}

impl AttentionPreprocessDescriptor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        packed_q_gate: TensorBinding,
        k: TensorBinding,
        q_raw_scale: TensorBinding,
        k_raw_scale: TensorBinding,
        positions: TensorBinding,
        q_output: TensorBinding,
        gate_output: TensorBinding,
        k_output: TensorBinding,
        contract: AttentionPreprocessContract,
    ) -> Result<Self, sllm_core::OpError> {
        let semantic = Arc::new(SemanticOpDescriptor::new_attention_preprocess(
            vec![
                packed_q_gate.view().clone(),
                k.view().clone(),
                q_raw_scale.view().clone(),
                k_raw_scale.view().clone(),
                positions.view().clone(),
            ],
            vec![
                q_output.view().clone(),
                gate_output.view().clone(),
                k_output.view().clone(),
            ],
            contract,
        )?);
        Ok(Self {
            packed_q_gate,
            k,
            q_raw_scale,
            k_raw_scale,
            positions,
            q_output,
            gate_output,
            k_output,
            semantic,
        })
    }

    pub fn semantic(&self) -> &SemanticOpDescriptor {
        self.semantic.as_ref()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_validated_semantic(
        semantic: Arc<SemanticOpDescriptor>,
        packed_q_gate: TensorBinding,
        k: TensorBinding,
        q_raw_scale: TensorBinding,
        k_raw_scale: TensorBinding,
        positions: TensorBinding,
        q_output: TensorBinding,
        gate_output: TensorBinding,
        k_output: TensorBinding,
    ) -> Result<Self, RuntimeError> {
        semantic.validate().map_err(|error| {
            RuntimeError::new(
                RuntimeStatus::InvalidAttentionPreprocessDescriptor,
                format!("invalid validated attention preprocess descriptor: {error}"),
            )
        })?;
        if semantic.kind() != SemanticOpKind::AttentionPreprocess
            || semantic.inputs().len() != 5
            || semantic.outputs().len() != 3
        {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidAttentionPreprocessDescriptor,
                "semantic descriptor is not a canonical attention preprocess operation",
            ));
        }
        let input_views = [
            packed_q_gate.view(),
            k.view(),
            q_raw_scale.view(),
            k_raw_scale.view(),
            positions.view(),
        ];
        if input_views
            .iter()
            .zip(semantic.inputs())
            .any(|(actual, expected)| *actual != expected)
        {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidAttentionPreprocessDescriptor,
                "bound HIP input tensor views differ from the core semantic descriptor",
            ));
        }
        let output_views = [q_output.view(), gate_output.view(), k_output.view()];
        if output_views
            .iter()
            .zip(semantic.outputs())
            .any(|(actual, expected)| *actual != expected)
        {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidAttentionPreprocessDescriptor,
                "bound HIP output tensor views differ from the core semantic descriptor",
            ));
        }
        Ok(Self {
            packed_q_gate,
            k,
            q_raw_scale,
            k_raw_scale,
            positions,
            q_output,
            gate_output,
            k_output,
            semantic,
        })
    }

    fn raw(&self) -> Result<sys::sllm_attention_preprocess_desc_t, RuntimeError> {
        let contract = self
            .semantic
            .attention_preprocess_contract()
            .ok_or_else(|| {
                RuntimeError::local(
                    RuntimeStatus::InvalidAttentionPreprocessDescriptor,
                    "attention preprocess contract is absent",
                )
            })?;
        Ok(raw_descriptor(
            contract,
            self.packed_q_gate.raw()?,
            self.k.raw()?,
            self.q_raw_scale.raw()?,
            self.k_raw_scale.raw()?,
            self.positions.raw()?,
            self.q_output.raw()?,
            self.gate_output.raw()?,
            self.k_output.raw()?,
        ))
    }
}

#[allow(clippy::too_many_arguments)]
fn raw_descriptor(
    contract: AttentionPreprocessContract,
    packed_q_gate: sys::sllm_tensor_binding_t,
    k: sys::sllm_tensor_binding_t,
    q_raw_scale: sys::sllm_tensor_binding_t,
    k_raw_scale: sys::sllm_tensor_binding_t,
    positions: sys::sllm_tensor_binding_t,
    q_output: sys::sllm_tensor_binding_t,
    gate_output: sys::sllm_tensor_binding_t,
    k_output: sys::sllm_tensor_binding_t,
) -> sys::sllm_attention_preprocess_desc_t {
    sys::sllm_attention_preprocess_desc_t {
        struct_size: size_of::<sys::sllm_attention_preprocess_desc_t>() as u32,
        abi_version: sys::SLLM_HIP_ABI_VERSION,
        op_version: sys::SLLM_HIP_ATTENTION_PREPROCESS_VERSION,
        start_position: contract.start_position(),
        reserved: [0; 4],
        packed_q_gate,
        k,
        q_raw_scale,
        k_raw_scale,
        positions,
        q_output,
        gate_output,
        k_output,
    }
}

struct PreparedAttentionPreprocessOwners {
    context: Context,
    descriptor: AttentionPreprocessDescriptor,
}

pub(crate) struct PreparedAttentionPreprocessState {
    raw: NonNull<sys::sllm_attention_preprocess_plan_t>,
    owners: PreparedAttentionPreprocessOwners,
}

// SAFETY: the native plan is an opaque registry token. Native transitions are
// serialized by the public registry/accounting lock and the retained owner
// graph is immutable.
unsafe impl Send for PreparedAttentionPreprocessState {}
unsafe impl Sync for PreparedAttentionPreprocessState {}

impl Drop for PreparedAttentionPreprocessState {
    fn drop(&mut self) {
        let (status, remaining) = release_attention_preprocess_plan_once(self.raw);
        if let Some(remaining) = remaining {
            enqueue_attention_preprocess_cleanup(
                remaining,
                self.owners.context.clone(),
                self.owners.descriptor.clone(),
                status,
            );
        }
    }
}

#[derive(Clone)]
pub struct PreparedAttentionPreprocess {
    state: Arc<PreparedAttentionPreprocessState>,
}

// SAFETY: the state has the Send/Sync guarantees documented above.
unsafe impl Send for PreparedAttentionPreprocess {}
unsafe impl Sync for PreparedAttentionPreprocess {}

impl std::fmt::Debug for PreparedAttentionPreprocess {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PreparedAttentionPreprocess")
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AttentionPreprocessDispatchInfo {
    pub abi_version: u32,
    pub info_version: u32,
    pub dispatch_id: u64,
    pub dispatch_count: u32,
    pub kernel_id: u32,
    pub workgroup_size_x: u32,
    pub grid_size_x: u32,
    pub m: u64,
    pub q_heads: u32,
    pub k_heads: u32,
    pub q_head_dim: u32,
    pub k_head_dim: u32,
    pub rotary_dim: u32,
    pub start_position: u32,
    pub backend: u32,
    pub fallback_allowed: bool,
    pub fallback_used: bool,
    pub kernel_symbol: String,
    pub device_symbol: String,
    pub gcn_arch_name: String,
}

fn read_c_string(value: &[core::ffi::c_char]) -> String {
    let length = value
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(value.len());
    value[..length]
        .iter()
        .map(|byte| *byte as u8)
        .map(char::from)
        .collect()
}

fn dispatch_info_from_raw(
    info: &sys::sllm_attention_preprocess_dispatch_info_t,
) -> AttentionPreprocessDispatchInfo {
    AttentionPreprocessDispatchInfo {
        abi_version: info.abi_version,
        info_version: info.info_version,
        dispatch_id: info.dispatch_id,
        dispatch_count: info.dispatch_count,
        kernel_id: info.kernel_id,
        workgroup_size_x: info.workgroup_size_x,
        grid_size_x: info.grid_size_x,
        m: info.m,
        q_heads: info.q_heads,
        k_heads: info.k_heads,
        q_head_dim: info.q_head_dim,
        k_head_dim: info.k_head_dim,
        rotary_dim: info.rotary_dim,
        start_position: info.start_position,
        backend: info.backend,
        fallback_allowed: info.fallback_allowed != 0,
        fallback_used: info.fallback_used != 0,
        kernel_symbol: read_c_string(&info.kernel_symbol),
        device_symbol: read_c_string(&info.device_symbol),
        gcn_arch_name: read_c_string(&info.gcn_arch_name),
    }
}

pub struct AttentionPreprocessSubmission {
    completion: Completion,
    _plan: Arc<PreparedAttentionPreprocessState>,
}

impl std::fmt::Debug for AttentionPreprocessSubmission {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AttentionPreprocessSubmission")
            .finish_non_exhaustive()
    }
}

impl AttentionPreprocessSubmission {
    pub fn query(&mut self) -> Result<CompletionState, RuntimeError> {
        self.completion.query()
    }

    pub fn wait(&mut self, timeout: std::time::Duration) -> Result<CompletionState, RuntimeError> {
        self.completion.wait(timeout)
    }

    pub fn kernel_elapsed_ns(&mut self) -> Result<u64, RuntimeError> {
        self.completion.kernel_elapsed_ns()
    }
}

impl HipBackend {
    pub fn prepare_attention_preprocess(
        &self,
        context: &Context,
        descriptor: AttentionPreprocessDescriptor,
    ) -> Result<PreparedAttentionPreprocess, RuntimeError> {
        let raw_descriptor = descriptor.raw()?;
        let mut raw_plan = std::ptr::null_mut();
        let mut buffer = [0_u8; 256];
        let mut error_sink = sink(&mut buffer);
        let status = unsafe {
            sys::sllm_attention_preprocess_prepare(
                context.raw_handle()?.as_ptr(),
                &raw_descriptor,
                &mut raw_plan,
                &mut error_sink,
            )
        };
        ensure_ok(status, &buffer, error_sink.message_length)?;
        let raw = NonNull::new(raw_plan).ok_or_else(|| {
            RuntimeError::new(
                RuntimeStatus::InternalError,
                "native attention preprocess prepare returned a null plan on success".to_owned(),
            )
        })?;
        Ok(PreparedAttentionPreprocess {
            state: Arc::new(PreparedAttentionPreprocessState {
                raw,
                owners: PreparedAttentionPreprocessOwners {
                    context: context.clone(),
                    descriptor,
                },
            }),
        })
    }
}

impl PreparedAttentionPreprocess {
    pub fn execute(
        &self,
        queue: &Queue,
    ) -> Result<
        (
            AttentionPreprocessSubmission,
            AttentionPreprocessDispatchInfo,
        ),
        RuntimeError,
    > {
        let mut info = sys::sllm_attention_preprocess_dispatch_info_t {
            struct_size: size_of::<sys::sllm_attention_preprocess_dispatch_info_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            info_version: sys::SLLM_HIP_ATTENTION_PREPROCESS_DISPATCH_INFO_VERSION,
            backend: 0,
            dispatch_id: 0,
            dispatch_count: 0,
            kernel_id: 0,
            workgroup_size_x: 0,
            grid_size_x: 0,
            m: 0,
            q_heads: 0,
            k_heads: 0,
            q_head_dim: 0,
            k_head_dim: 0,
            rotary_dim: 0,
            start_position: 0,
            fallback_allowed: 0,
            fallback_used: 0,
            kernel_symbol: [0; sys::SLLM_HIP_ATTENTION_PREPROCESS_KERNEL_SYMBOL_MAX as usize],
            device_symbol: [0; sys::SLLM_HIP_ATTENTION_PREPROCESS_DEVICE_SYMBOL_MAX as usize],
            gcn_arch_name: [0; sys::SLLM_HIP_MAX_GCN_ARCH_NAME as usize],
            reserved: [0; 8],
        };
        let mut error_buffer = [0_u8; 256];
        let mut error_sink = sink(&mut error_buffer);
        let mut raw_completion = std::ptr::null_mut();
        let status = unsafe {
            sys::sllm_attention_preprocess_execute(
                self.state.raw.as_ptr(),
                queue.raw_handle()?.as_ptr(),
                &mut raw_completion,
                &mut info,
                &mut error_sink,
            )
        };
        ensure_ok(status, &error_buffer, error_sink.message_length)?;
        let raw_completion = NonNull::new(raw_completion).ok_or_else(|| {
            RuntimeError::new(
                RuntimeStatus::InternalError,
                "native attention preprocess execute returned a null completion on success"
                    .to_owned(),
            )
        })?;
        let dispatch_info = dispatch_info_from_raw(&info);
        let completion = Completion::from_native(
            raw_completion,
            &self.state.owners.context,
            queue,
            self.state.owners.descriptor.packed_q_gate.buffer(),
            0,
            false,
        );
        Ok((
            AttentionPreprocessSubmission {
                completion,
                _plan: Arc::clone(&self.state),
            },
            dispatch_info,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Buffer;
    use sllm_core::{AttentionPreprocessPositionMode, DType, TensorView};
    use std::mem::{align_of, offset_of};
    use std::thread;

    fn raw_binding(
        offset: u64,
        rank: u32,
        shape: [u64; 8],
        strides: [u64; 8],
        dtype: u32,
    ) -> sys::sllm_tensor_binding_t {
        sys::sllm_tensor_binding_t {
            struct_size: size_of::<sys::sllm_tensor_binding_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            buffer: NonNull::<sys::sllm_buffer_t>::dangling().as_ptr(),
            byte_offset: offset,
            dtype,
            encoding: sys::SLLM_TENSOR_ENCODING_UNQUANTIZED,
            rank,
            reserved0: 0,
            shape,
            stride_elements: strides,
            reserved: [0; 2],
        }
    }

    fn descriptor_fixture() -> (Context, AttentionPreprocessDescriptor) {
        let context = Context::test_without_native();
        let packed = Buffer::test_without_native(&context);
        let k = Buffer::test_without_native(&context);
        let q_scale = Buffer::test_without_native(&context);
        let k_scale = Buffer::test_without_native(&context);
        let positions = Buffer::test_without_native(&context);
        let q_output = Buffer::test_without_native(&context);
        let gate_output = Buffer::test_without_native(&context);
        let k_output = Buffer::test_without_native(&context);
        let contract = AttentionPreprocessContract::new_qwen3_5(
            AttentionPreprocessPositionMode::Prefill,
            0,
            3,
        )
        .expect("valid attention preprocess contract");
        let descriptor = AttentionPreprocessDescriptor::new(
            packed.binding(TensorView::contiguous(DType::Bf16, &[3, 16, 512]).unwrap()),
            k.binding(TensorView::contiguous(DType::Bf16, &[3, 4, 256]).unwrap()),
            q_scale.binding(TensorView::contiguous(DType::Bf16, &[16, 256]).unwrap()),
            k_scale.binding(TensorView::contiguous(DType::Bf16, &[4, 256]).unwrap()),
            positions.binding(TensorView::contiguous(DType::I32, &[3]).unwrap()),
            q_output.binding(TensorView::contiguous(DType::Bf16, &[3, 16, 256]).unwrap()),
            gate_output.binding(TensorView::contiguous(DType::Bf16, &[3, 16, 256]).unwrap()),
            k_output.binding(TensorView::contiguous(DType::Bf16, &[3, 4, 256]).unwrap()),
            contract,
        )
        .expect("valid attention preprocess descriptor");
        (context, descriptor)
    }

    #[test]
    fn semantic_lowering_preserves_all_roles_and_explicit_start_position() {
        let (_context, descriptor) = descriptor_fixture();
        let semantic = descriptor.semantic();
        assert_eq!(semantic.kind(), SemanticOpKind::AttentionPreprocess);
        assert_eq!(semantic.inputs().len(), 5);
        assert_eq!(semantic.outputs().len(), 3);
        assert_eq!(semantic.inputs()[0].shape(), &[3, 16, 512]);
        assert_eq!(semantic.inputs()[4].dtype(), DType::I32);
        assert_eq!(semantic.outputs()[2].shape(), &[3, 4, 256]);
        assert_eq!(
            semantic
                .attention_preprocess_contract()
                .expect("contract retained")
                .start_position(),
            0
        );
    }

    #[test]
    fn raw_abi_fields_and_layout_are_exact() {
        let contract = AttentionPreprocessContract::new_qwen3_5(
            AttentionPreprocessPositionMode::DecodeContinuation,
            257,
            3,
        )
        .expect("valid continuation contract");
        let mut packed_shape = [0; 8];
        packed_shape[..3].copy_from_slice(&[3, 16, 512]);
        let mut packed_stride = [0; 8];
        packed_stride[..3].copy_from_slice(&[8192, 512, 1]);
        let mut k_shape = [0; 8];
        k_shape[..3].copy_from_slice(&[3, 4, 256]);
        let mut k_stride = [0; 8];
        k_stride[..3].copy_from_slice(&[1024, 256, 1]);
        let mut q_scale_shape = [0; 8];
        q_scale_shape[..2].copy_from_slice(&[16, 256]);
        let mut q_scale_stride = [0; 8];
        q_scale_stride[..2].copy_from_slice(&[256, 1]);
        let mut k_scale_shape = [0; 8];
        k_scale_shape[..2].copy_from_slice(&[4, 256]);
        let mut k_scale_stride = [0; 8];
        k_scale_stride[..2].copy_from_slice(&[256, 1]);
        let mut position_shape = [0; 8];
        position_shape[0] = 3;
        let mut position_stride = [0; 8];
        position_stride[0] = 1;
        let raw = raw_descriptor(
            contract,
            raw_binding(
                8,
                3,
                packed_shape,
                packed_stride,
                sys::SLLM_TENSOR_DTYPE_BF16,
            ),
            raw_binding(16, 3, k_shape, k_stride, sys::SLLM_TENSOR_DTYPE_BF16),
            raw_binding(
                24,
                2,
                q_scale_shape,
                q_scale_stride,
                sys::SLLM_TENSOR_DTYPE_BF16,
            ),
            raw_binding(
                32,
                2,
                k_scale_shape,
                k_scale_stride,
                sys::SLLM_TENSOR_DTYPE_BF16,
            ),
            raw_binding(
                40,
                1,
                position_shape,
                position_stride,
                sys::SLLM_TENSOR_DTYPE_I32,
            ),
            raw_binding(
                48,
                3,
                packed_shape,
                packed_stride,
                sys::SLLM_TENSOR_DTYPE_BF16,
            ),
            raw_binding(
                56,
                3,
                packed_shape,
                packed_stride,
                sys::SLLM_TENSOR_DTYPE_BF16,
            ),
            raw_binding(64, 3, k_shape, k_stride, sys::SLLM_TENSOR_DTYPE_BF16),
        );
        assert_eq!(
            raw.struct_size as usize,
            size_of::<sys::sllm_attention_preprocess_desc_t>()
        );
        assert_eq!(raw.abi_version, sys::SLLM_HIP_ABI_VERSION);
        assert_eq!(raw.op_version, sys::SLLM_HIP_ATTENTION_PREPROCESS_VERSION);
        assert_eq!(raw.start_position, 257);
        assert_eq!(raw.reserved, [0; 4]);
        assert_eq!(raw.packed_q_gate.byte_offset, 8);
        assert_eq!(raw.k.byte_offset, 16);
        assert_eq!(raw.q_raw_scale.rank, 2);
        assert_eq!(raw.positions.dtype, sys::SLLM_TENSOR_DTYPE_I32);
        assert_eq!(raw.q_output.shape, packed_shape);
        assert_eq!(raw.k_output.shape, k_shape);
        assert_eq!(size_of::<sys::sllm_attention_preprocess_desc_t>(), 1504);
        assert_eq!(align_of::<sys::sllm_attention_preprocess_desc_t>(), 8);
        assert_eq!(
            offset_of!(sys::sllm_attention_preprocess_desc_t, start_position),
            12
        );
        assert_eq!(
            offset_of!(sys::sllm_attention_preprocess_desc_t, packed_q_gate),
            32
        );
        assert_eq!(
            offset_of!(sys::sllm_attention_preprocess_desc_t, k_output),
            32 + 7 * 184
        );
        assert_eq!(
            size_of::<sys::sllm_attention_preprocess_dispatch_info_t>(),
            304
        );
        assert_eq!(
            align_of::<sys::sllm_attention_preprocess_dispatch_info_t>(),
            8
        );
    }

    #[test]
    fn dispatch_info_maps_all_native_fields() {
        let mut info = sys::sllm_attention_preprocess_dispatch_info_t {
            struct_size: size_of::<sys::sllm_attention_preprocess_dispatch_info_t>() as u32,
            abi_version: 1,
            info_version: 1,
            backend: sys::SLLM_BACKEND_HIP,
            dispatch_id: 9,
            dispatch_count: 1,
            kernel_id: 1,
            workgroup_size_x: 1,
            grid_size_x: 60,
            m: 3,
            q_heads: 16,
            k_heads: 4,
            q_head_dim: 256,
            k_head_dim: 256,
            rotary_dim: 64,
            start_position: 257,
            fallback_allowed: 0,
            fallback_used: 0,
            kernel_symbol: [0; 64],
            device_symbol: [0; 64],
            gcn_arch_name: [0; 64],
            reserved: [0; 8],
        };
        for (destination, source) in info.kernel_symbol[..4].iter_mut().zip(b"test") {
            *destination = *source as core::ffi::c_char;
        }
        for (destination, source) in info.device_symbol[..6].iter_mut().zip(b"device") {
            *destination = *source as core::ffi::c_char;
        }
        for (destination, source) in info.gcn_arch_name[..7].iter_mut().zip(b"gfx1201") {
            *destination = *source as core::ffi::c_char;
        }
        let mapped = dispatch_info_from_raw(&info);
        assert_eq!(mapped.m, 3);
        assert_eq!(mapped.start_position, 257);
        assert_eq!(mapped.kernel_symbol, "test");
        assert_eq!(mapped.device_symbol, "device");
        assert_eq!(mapped.gcn_arch_name, "gfx1201");
        assert!(!mapped.fallback_allowed);
        assert!(!mapped.fallback_used);
    }

    #[test]
    fn prepared_attention_preprocess_is_send_sync_and_non_consuming_failure_retains_owner_graph() {
        let _serial = crate::runtime::CLEANUP_TEST_SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        static_assertions::assert_impl_all!(PreparedAttentionPreprocess: Send, Sync);
        static_assertions::assert_impl_all!(AttentionPreprocessSubmission: Send, Sync);
        let before = Context::durable_quarantine_count();
        let (context, descriptor) = descriptor_fixture();
        crate::runtime::force_attention_preprocess_plan_release_for_test(
            RuntimeStatus::InternalError,
            false,
        );
        let prepared = PreparedAttentionPreprocess {
            state: Arc::new(PreparedAttentionPreprocessState {
                raw: NonNull::dangling(),
                owners: PreparedAttentionPreprocessOwners {
                    context,
                    descriptor,
                },
            }),
        };
        let shared = Arc::new(prepared);
        let worker_shared = Arc::clone(&shared);
        let worker = thread::spawn(move || drop(worker_shared));
        drop(shared);
        worker
            .join()
            .expect("attention preprocess owner drop must not panic");
        assert_eq!(Context::durable_quarantine_count(), before + 1);
        crate::runtime::clear_forced_attention_preprocess_plan_release_for_test();
    }

    #[test]
    fn status_mapping_has_new_codes_without_reassigning_existing_codes() {
        assert_eq!(
            RuntimeStatus::from_raw(sys::SLLM_STATUS_INVALID_ATTENTION_PREPROCESS_DESCRIPTOR),
            RuntimeStatus::InvalidAttentionPreprocessDescriptor
        );
        assert_eq!(
            RuntimeStatus::from_raw(sys::SLLM_STATUS_POSITION_PAYLOAD_MISMATCH),
            RuntimeStatus::PositionPayloadMismatch
        );
        assert_eq!(
            RuntimeStatus::InvalidAttentionPreprocessDescriptor.raw(),
            0x119
        );
        assert_eq!(RuntimeStatus::PositionPayloadMismatch.raw(), 0x11a);
        assert_eq!(RuntimeStatus::InvalidMatmulDescriptor.raw(), 0x118);
    }
}
