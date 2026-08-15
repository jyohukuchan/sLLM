//! Safe RMSNorm preparation and asynchronous baseline execution wrappers.

use std::mem::size_of;
use std::ptr::NonNull;
use std::sync::Arc;

use sllm_core::{
    DType, Encoding, Fp8ResidentRepresentation, Fp8ScaleGranularity, RmsNormScaleMode,
    SemanticOpDescriptor, TensorView,
};
use sllm_hip_sys as sys;

use crate::runtime::{
    Completion, CompletionState, Context, Queue, RuntimeError, RuntimeStatus,
    enqueue_rmsnorm_cleanup, ensure_ok, release_rmsnorm_plan_once, sink,
};
use crate::{Buffer, HipBackend};

#[derive(Clone, Debug)]
pub struct TensorBinding {
    buffer: Buffer,
    view: TensorView,
}

impl TensorBinding {
    pub(crate) fn from_buffer(buffer: Buffer, view: TensorView) -> Self {
        Self { buffer, view }
    }

    pub fn view(&self) -> &TensorView {
        &self.view
    }

    pub(crate) fn raw(&self) -> Result<sys::sllm_tensor_binding_t, RuntimeError> {
        let rank = u32::try_from(self.view.shape().len()).map_err(|_| {
            RuntimeError::local(
                RuntimeStatus::InvalidTensorBinding,
                "tensor rank does not fit ABI",
            )
        })?;
        if rank > sys::SLLM_HIP_TENSOR_MAX_RANK {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidTensorBinding,
                "tensor rank exceeds the public RMSNorm ABI",
            ));
        }
        let mut shape = [0_u64; 8];
        let mut strides = [0_u64; 8];
        for (index, (&extent, &stride)) in self
            .view
            .shape()
            .iter()
            .zip(self.view.strides())
            .enumerate()
        {
            shape[index] = u64::try_from(extent).map_err(|_| {
                RuntimeError::local(
                    RuntimeStatus::MetadataOverflow,
                    "tensor extent does not fit ABI",
                )
            })?;
            strides[index] = u64::try_from(stride).map_err(|_| {
                RuntimeError::local(
                    RuntimeStatus::MetadataOverflow,
                    "tensor stride does not fit ABI",
                )
            })?;
        }
        Ok(sys::sllm_tensor_binding_t {
            struct_size: size_of::<sys::sllm_tensor_binding_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            buffer: self.buffer.raw_handle()?.as_ptr(),
            byte_offset: self.view.byte_offset(),
            dtype: match self.view.dtype() {
                DType::Bf16 => sys::SLLM_TENSOR_DTYPE_BF16,
                DType::F32 => sys::SLLM_TENSOR_DTYPE_F32,
                DType::F8E4M3Fn => sys::SLLM_TENSOR_DTYPE_F8_E4M3_FN,
                DType::F8E4M3FnuZ => sys::SLLM_TENSOR_DTYPE_F8_E4M3_FNUZ,
                DType::U8 => sys::SLLM_TENSOR_DTYPE_U8,
                DType::I32 => sys::SLLM_TENSOR_DTYPE_I32,
                _ => u32::MAX,
            },
            encoding: match self.view.encoding() {
                Encoding::Unquantized => sys::SLLM_TENSOR_ENCODING_UNQUANTIZED,
                Encoding::Nvfp4 {
                    block_size: 16,
                    scale_dtype: DType::F8E4M3Fn,
                } => sys::SLLM_TENSOR_ENCODING_NVFP4_BLOCK16_E4M3FN_F32,
                Encoding::Nvfp4 { .. } => u32::MAX,
                Encoding::Fp8Scaled {
                    granularity: Fp8ScaleGranularity::OuterDimension,
                    scale_dtype: DType::F32,
                    resident: Fp8ResidentRepresentation::PackedBytes,
                } => sys::SLLM_TENSOR_ENCODING_FP8_OUTER_F32,
                Encoding::Fp8Scaled { .. } => u32::MAX,
            },
            rank,
            reserved0: 0,
            shape,
            stride_elements: strides,
            reserved: [0; 2],
        })
    }

    pub(crate) fn buffer(&self) -> &Buffer {
        &self.buffer
    }
}

#[derive(Clone, Debug)]
pub struct RmsNormDescriptor {
    activation: TensorBinding,
    raw_scale: TensorBinding,
    output: TensorBinding,
    semantic: Arc<SemanticOpDescriptor>,
}

impl RmsNormDescriptor {
    pub fn new(
        activation: TensorBinding,
        raw_scale: TensorBinding,
        output: TensorBinding,
        epsilon: f32,
    ) -> Result<Self, sllm_core::OpError> {
        let semantic = Arc::new(SemanticOpDescriptor::new_rms_norm(
            vec![activation.view.clone(), raw_scale.view.clone()],
            vec![output.view.clone()],
            epsilon,
            RmsNormScaleMode::OffsetOne,
        )?);
        Ok(Self {
            activation,
            raw_scale,
            output,
            semantic,
        })
    }

    pub fn semantic(&self) -> &SemanticOpDescriptor {
        self.semantic.as_ref()
    }

    /// Lower one already-validated core semantic descriptor without
    /// reconstructing a descriptor at the HIP boundary.  The identity of the
    /// `Arc` is deliberately retained by the prepared owner graph.
    pub(crate) fn from_validated_semantic(
        semantic: Arc<SemanticOpDescriptor>,
        activation: TensorBinding,
        raw_scale: TensorBinding,
        output: TensorBinding,
    ) -> Result<Self, RuntimeError> {
        semantic.validate().map_err(|error| {
            RuntimeError::new(
                RuntimeStatus::InvalidRmsNormDescriptor,
                format!("invalid validated semantic RMSNorm descriptor: {error}"),
            )
        })?;
        if semantic.kind() != sllm_core::SemanticOpKind::RmsNorm
            || semantic.inputs().len() != 2
            || semantic.outputs().len() != 1
        {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidRmsNormDescriptor,
                "semantic descriptor is not a canonical RMSNorm operation",
            ));
        }
        if activation.view != semantic.inputs()[0]
            || raw_scale.view != semantic.inputs()[1]
            || output.view != semantic.outputs()[0]
        {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidRmsNormDescriptor,
                "bound HIP tensor views differ from the core semantic descriptor",
            ));
        }
        Ok(Self {
            activation,
            raw_scale,
            output,
            semantic,
        })
    }

    fn raw(&self) -> Result<sys::sllm_rmsnorm_desc_t, RuntimeError> {
        let contract = self.semantic.rms_norm_contract().ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::InvalidRmsNormDescriptor,
                "RMSNorm contract is absent",
            )
        })?;
        Ok(sys::sllm_rmsnorm_desc_t {
            struct_size: size_of::<sys::sllm_rmsnorm_desc_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            op_version: sys::SLLM_HIP_RMSNORM_VERSION,
            accumulation_dtype: sys::SLLM_RMSNORM_ACCUMULATION_F32,
            scale_mode: match contract.scale_mode() {
                RmsNormScaleMode::OffsetOne => sys::SLLM_RMSNORM_SCALE_MODE_OFFSET_ONE,
                RmsNormScaleMode::Direct => sys::SLLM_RMSNORM_SCALE_MODE_DIRECT,
            },
            alias_policy: sys::SLLM_RMSNORM_ALIAS_POLICY_REJECT_OVERLAP,
            epsilon_bits: contract.epsilon().bits(),
            reserved: [0; 3],
            activation: self.activation.raw()?,
            raw_scale: self.raw_scale.raw()?,
            output: self.output.raw()?,
        })
    }
}

/// An opaque, prepared native plan. It strongly owns the Context and all
/// bound buffers through the descriptor until its native release succeeds.
struct PreparedRmsNormOwners {
    context: Context,
    descriptor: RmsNormDescriptor,
}

pub(crate) struct PreparedRmsNormState {
    pub(crate) raw: NonNull<sys::sllm_rmsnorm_plan_t>,
    owners: PreparedRmsNormOwners,
}

// SAFETY: the native plan is an opaque token. Native registry lookup,
// execution, and release serialize transitions through the native registry
// and Context accounting lock; the retained Rust owner graph is immutable.
unsafe impl Send for PreparedRmsNormState {}
// SAFETY: shared state exposes no mutable native operation without the
// serialized FFI transition described above.
unsafe impl Sync for PreparedRmsNormState {}

impl Drop for PreparedRmsNormState {
    fn drop(&mut self) {
        let (status, remaining) = release_rmsnorm_plan_once(self.raw);
        if let Some(remaining) = remaining {
            enqueue_rmsnorm_cleanup(
                remaining,
                self.owners.context.clone(),
                self.owners.descriptor.clone(),
                status,
            );
        }
    }
}

#[derive(Clone)]
pub struct PreparedRmsNorm {
    state: std::sync::Arc<PreparedRmsNormState>,
}

// SAFETY: the native plan is an opaque token. Native registry lookup and
// release are serialized by the native registry and Context accounting lock.
unsafe impl Send for PreparedRmsNorm {}
// SAFETY: shared references expose only immutable ownership access.
unsafe impl Sync for PreparedRmsNorm {}

impl std::fmt::Debug for PreparedRmsNorm {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PreparedRmsNorm")
            .finish_non_exhaustive()
    }
}

impl Drop for PreparedRmsNorm {
    fn drop(&mut self) {}
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RmsNormDispatchInfo {
    pub abi_version: u32,
    pub info_version: u32,
    pub dispatch_id: u64,
    pub dispatch_count: u32,
    pub kernel_id: u32,
    pub workgroup_size_x: u32,
    pub grid_size_x: u32,
    pub row_count: u64,
    pub normalized_size: u64,
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
        .collect::<Vec<_>>()
        .into_iter()
        .map(char::from)
        .collect()
}

fn dispatch_info_from_raw(info: &sys::sllm_rmsnorm_dispatch_info_t) -> RmsNormDispatchInfo {
    RmsNormDispatchInfo {
        abi_version: info.abi_version,
        info_version: info.info_version,
        dispatch_id: info.dispatch_id,
        dispatch_count: info.dispatch_count,
        kernel_id: info.kernel_id,
        workgroup_size_x: info.workgroup_size_x,
        grid_size_x: info.grid_size_x,
        row_count: info.row_count,
        normalized_size: info.normalized_size,
        backend: info.backend,
        fallback_allowed: info.fallback_allowed != 0,
        fallback_used: info.fallback_used != 0,
        kernel_symbol: read_c_string(&info.kernel_symbol),
        device_symbol: read_c_string(&info.device_symbol),
        gcn_arch_name: read_c_string(&info.gcn_arch_name),
    }
}

pub struct RmsNormSubmission {
    completion: Completion,
    _plan: std::sync::Arc<PreparedRmsNormState>,
}

impl std::fmt::Debug for RmsNormSubmission {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RmsNormSubmission")
            .finish_non_exhaustive()
    }
}

impl RmsNormSubmission {
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
    /// Validate and capture an RMSNorm plan.  This does not execute RMSNorm.
    pub fn prepare_rms_norm(
        &self,
        context: &Context,
        descriptor: RmsNormDescriptor,
    ) -> Result<PreparedRmsNorm, RuntimeError> {
        let raw_descriptor = descriptor.raw()?;
        let mut raw_plan = std::ptr::null_mut();
        let mut buffer = [0_u8; 256];
        let mut error_sink = sink(&mut buffer);
        let status = unsafe {
            sys::sllm_rmsnorm_prepare(
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
                "native RMSNorm prepare returned a null plan on success".to_owned(),
            )
        })?;
        Ok(PreparedRmsNorm {
            state: std::sync::Arc::new(PreparedRmsNormState {
                raw,
                owners: PreparedRmsNormOwners {
                    context: context.clone(),
                    descriptor,
                },
            }),
        })
    }
}

impl PreparedRmsNorm {
    /// Enqueue the reusable baseline RMSNorm plan on `queue`.
    ///
    /// The native completion retains the plan, queue, and all three bound
    /// buffers until it reaches a terminal state. The Rust plan Arc keeps the
    /// complete owner graph alive as well when this submission is dropped
    /// before the caller observes completion.
    pub fn execute(
        &self,
        queue: &Queue,
    ) -> Result<(RmsNormSubmission, RmsNormDispatchInfo), RuntimeError> {
        let mut info = sys::sllm_rmsnorm_dispatch_info_t {
            struct_size: size_of::<sys::sllm_rmsnorm_dispatch_info_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            info_version: sys::SLLM_HIP_RMSNORM_DISPATCH_INFO_VERSION,
            backend: 0,
            dispatch_id: 0,
            dispatch_count: 0,
            kernel_id: 0,
            workgroup_size_x: 0,
            grid_size_x: 0,
            row_count: 0,
            normalized_size: 0,
            fallback_allowed: 0,
            fallback_used: 0,
            kernel_symbol: [0; sys::SLLM_HIP_RMSNORM_KERNEL_SYMBOL_MAX as usize],
            device_symbol: [0; sys::SLLM_HIP_RMSNORM_DEVICE_SYMBOL_MAX as usize],
            gcn_arch_name: [0; 64],
            reserved: [0; 8],
        };
        let mut error_buffer = [0_u8; 256];
        let mut error_sink = sink(&mut error_buffer);
        let mut raw_completion = std::ptr::null_mut();
        let status = unsafe {
            sys::sllm_rmsnorm_execute(
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
                "native RMSNorm execute returned a null completion on success".to_owned(),
            )
        })?;
        let dispatch_info = dispatch_info_from_raw(&info);
        let completion = Completion::from_native(
            raw_completion,
            &self.state.owners.context,
            queue,
            &self.state.owners.descriptor.activation.buffer,
            0,
            false,
        );
        Ok((
            RmsNormSubmission {
                completion,
                _plan: std::sync::Arc::clone(&self.state),
            },
            dispatch_info,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    fn raw_binding(
        offset: u64,
        rank: u32,
        shape: [u64; 8],
        strides: [u64; 8],
    ) -> sys::sllm_tensor_binding_t {
        sys::sllm_tensor_binding_t {
            struct_size: size_of::<sys::sllm_tensor_binding_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            buffer: NonNull::<sys::sllm_buffer_t>::dangling().as_ptr(),
            byte_offset: offset,
            dtype: sys::SLLM_TENSOR_DTYPE_BF16,
            encoding: sys::SLLM_TENSOR_ENCODING_UNQUANTIZED,
            rank,
            reserved0: 0,
            shape,
            stride_elements: strides,
            reserved: [0; 2],
        }
    }

    fn raw_descriptor() -> sys::sllm_rmsnorm_desc_t {
        let mut activation_shape = [0; 8];
        activation_shape[..2].copy_from_slice(&[2, 3]);
        let mut activation_strides = [0; 8];
        activation_strides[..2].copy_from_slice(&[3, 1]);
        let mut scale_shape = [0; 8];
        scale_shape[0] = 3;
        let mut scale_strides = [0; 8];
        scale_strides[0] = 1;
        sys::sllm_rmsnorm_desc_t {
            struct_size: size_of::<sys::sllm_rmsnorm_desc_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            op_version: sys::SLLM_HIP_RMSNORM_VERSION,
            accumulation_dtype: sys::SLLM_RMSNORM_ACCUMULATION_F32,
            scale_mode: sys::SLLM_RMSNORM_SCALE_MODE_OFFSET_ONE,
            alias_policy: sys::SLLM_RMSNORM_ALIAS_POLICY_REJECT_OVERLAP,
            epsilon_bits: 1.0e-6_f32.to_bits(),
            reserved: [0; 3],
            activation: raw_binding(0, 2, activation_shape, activation_strides),
            raw_scale: raw_binding(16, 1, scale_shape, scale_strides),
            output: raw_binding(32, 2, activation_shape, activation_strides),
        }
    }

    fn owner_fixture() -> (Context, RmsNormDescriptor) {
        let context = Context::test_without_native();
        let activation_buffer = Buffer::test_without_native(&context);
        let scale_buffer = Buffer::test_without_native(&context);
        let output_buffer = Buffer::test_without_native(&context);
        let activation = activation_buffer
            .binding(sllm_core::TensorView::contiguous(DType::Bf16, &[2, 3]).unwrap());
        let raw_scale =
            scale_buffer.binding(sllm_core::TensorView::contiguous(DType::Bf16, &[3]).unwrap());
        let output =
            output_buffer.binding(sllm_core::TensorView::contiguous(DType::Bf16, &[2, 3]).unwrap());
        let descriptor = RmsNormDescriptor::new(activation, raw_scale, output, 1.0e-6)
            .expect("test RMSNorm descriptor must be semantically valid");
        (context, descriptor)
    }

    #[test]
    fn prepared_rmsnorm_is_send_sync_and_unique_drop_is_thread_safe() {
        let _serial = crate::runtime::CLEANUP_TEST_SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        static_assertions::assert_impl_all!(PreparedRmsNorm: Send, Sync);
        let (context, descriptor) = owner_fixture();
        crate::runtime::force_rmsnorm_plan_release_for_test(RuntimeStatus::InternalError, true);
        let prepared = PreparedRmsNorm {
            state: Arc::new(PreparedRmsNormState {
                raw: NonNull::dangling(),
                owners: PreparedRmsNormOwners {
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
            .expect("PreparedRmsNorm drop worker must not panic");
        crate::runtime::clear_forced_rmsnorm_plan_release_for_test();
    }

    #[test]
    fn prepared_rmsnorm_nonconsuming_failure_retains_complete_owner_graph() {
        let _serial = crate::runtime::CLEANUP_TEST_SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let before = Context::durable_quarantine_count();
        let (context, descriptor) = owner_fixture();
        crate::runtime::force_rmsnorm_plan_release_for_test(RuntimeStatus::InternalError, false);
        drop(PreparedRmsNorm {
            state: Arc::new(PreparedRmsNormState {
                raw: NonNull::dangling(),
                owners: PreparedRmsNormOwners {
                    context,
                    descriptor,
                },
            }),
        });
        assert_eq!(Context::durable_quarantine_count(), before + 1);
        crate::runtime::clear_forced_rmsnorm_plan_release_for_test();
    }

    #[test]
    fn descriptor_requires_bound_metadata_but_stub_never_prepares() {
        assert_eq!(size_of::<sys::sllm_tensor_binding_t>(), 184);
        assert_eq!(size_of::<sys::sllm_rmsnorm_desc_t>(), 592);
        let mut message = [0_u8; 256];
        let mut error_sink = sink(&mut message);
        let mut plan = std::ptr::null_mut();
        let mut descriptor = raw_descriptor();
        let status = unsafe {
            sys::sllm_rmsnorm_prepare(
                NonNull::<sys::sllm_context_t>::dangling().as_ptr(),
                &descriptor,
                &mut plan,
                &mut error_sink,
            )
        };
        assert_eq!(
            RuntimeStatus::from_raw(status),
            RuntimeStatus::HipUnavailable
        );
        assert!(plan.is_null());
        descriptor.output.byte_offset = 1;
        let status = unsafe {
            sys::sllm_rmsnorm_prepare(
                NonNull::<sys::sllm_context_t>::dangling().as_ptr(),
                &descriptor,
                &mut plan,
                &mut error_sink,
            )
        };
        assert_eq!(
            RuntimeStatus::from_raw(status),
            RuntimeStatus::MisalignedOffset
        );
        let status = unsafe { sys::sllm_rmsnorm_plan_release(&mut plan, &mut error_sink) };
        assert_eq!(
            RuntimeStatus::from_raw(status),
            RuntimeStatus::InvalidArgument
        );
    }
}
