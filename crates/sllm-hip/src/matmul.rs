//! Safe BF16 matmul preparation and asynchronous baseline execution wrappers.

use std::mem::size_of;
use std::ptr::NonNull;
use std::sync::Arc;

use sllm_core::{SemanticOpDescriptor, SemanticOpKind};
use sllm_hip_sys as sys;

use crate::runtime::{
    Completion, CompletionState, Context, Queue, RuntimeError, RuntimeStatus,
    enqueue_matmul_cleanup, ensure_ok, release_matmul_plan_once, sink,
};
use crate::{HipBackend, TensorBinding};

#[derive(Clone, Debug)]
pub struct MatmulDescriptor {
    activation: TensorBinding,
    weight: TensorBinding,
    output: TensorBinding,
    semantic: Arc<SemanticOpDescriptor>,
}

impl MatmulDescriptor {
    pub fn new(
        activation: TensorBinding,
        weight: TensorBinding,
        output: TensorBinding,
    ) -> Result<Self, sllm_core::OpError> {
        let semantic = Arc::new(SemanticOpDescriptor::new(
            SemanticOpKind::Matmul,
            vec![activation.view().clone(), weight.view().clone()],
            vec![output.view().clone()],
        )?);
        Ok(Self {
            activation,
            weight,
            output,
            semantic,
        })
    }

    pub fn semantic(&self) -> &SemanticOpDescriptor {
        self.semantic.as_ref()
    }

    pub(crate) fn from_validated_semantic(
        semantic: Arc<SemanticOpDescriptor>,
        activation: TensorBinding,
        weight: TensorBinding,
        output: TensorBinding,
    ) -> Result<Self, RuntimeError> {
        semantic.validate().map_err(|error| {
            RuntimeError::new(
                RuntimeStatus::InvalidMatmulDescriptor,
                format!("invalid validated matmul descriptor: {error}"),
            )
        })?;
        if semantic.kind() != SemanticOpKind::Matmul
            || semantic.inputs().len() != 2
            || semantic.outputs().len() != 1
        {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidMatmulDescriptor,
                "semantic descriptor is not a canonical matmul operation",
            ));
        }
        if activation.view() != &semantic.inputs()[0]
            || weight.view() != &semantic.inputs()[1]
            || output.view() != &semantic.outputs()[0]
        {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidMatmulDescriptor,
                "bound HIP tensor views differ from the core matmul descriptor",
            ));
        }
        Ok(Self {
            activation,
            weight,
            output,
            semantic,
        })
    }

    fn raw(&self) -> Result<sys::sllm_matmul_desc_t, RuntimeError> {
        let op_version = if matches!(
            self.weight.view().dtype(),
            sllm_core::DType::F8E4M3Fn | sllm_core::DType::F8E4M3FnuZ
        ) {
            sys::SLLM_HIP_MATMUL_FP8_VERSION
        } else {
            sys::SLLM_HIP_MATMUL_VERSION
        };
        Ok(sys::sllm_matmul_desc_t {
            struct_size: size_of::<sys::sllm_matmul_desc_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            op_version,
            reserved: [0; 5],
            activation: self.activation.raw()?,
            weight: self.weight.raw()?,
            output: self.output.raw()?,
        })
    }
}

struct PreparedMatmulOwners {
    context: Context,
    descriptor: MatmulDescriptor,
}

struct PreparedMatmulState {
    raw: NonNull<sys::sllm_matmul_plan_t>,
    owners: PreparedMatmulOwners,
}

// SAFETY: the native plan is an opaque registry token. Native transitions are
// serialized by the public registry/accounting lock and the retained owner
// graph is immutable.
unsafe impl Send for PreparedMatmulState {}
unsafe impl Sync for PreparedMatmulState {}

impl Drop for PreparedMatmulState {
    fn drop(&mut self) {
        let (status, remaining) = release_matmul_plan_once(self.raw);
        if let Some(remaining) = remaining {
            enqueue_matmul_cleanup(
                remaining,
                self.owners.context.clone(),
                self.owners.descriptor.clone(),
                status,
            );
        }
    }
}

#[derive(Clone)]
pub struct PreparedMatmul {
    state: Arc<PreparedMatmulState>,
}

// SAFETY: the state has the Send/Sync guarantees documented above.
unsafe impl Send for PreparedMatmul {}
unsafe impl Sync for PreparedMatmul {}

impl std::fmt::Debug for PreparedMatmul {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PreparedMatmul")
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MatmulDispatchInfo {
    pub abi_version: u32,
    pub info_version: u32,
    pub dispatch_id: u64,
    pub dispatch_count: u32,
    pub kernel_id: u32,
    pub workgroup_size_x: u32,
    pub grid_size_x: u32,
    pub m: u64,
    pub k: u64,
    pub n: u64,
    pub output_elements: u64,
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

fn dispatch_info_from_raw(info: &sys::sllm_matmul_dispatch_info_t) -> MatmulDispatchInfo {
    MatmulDispatchInfo {
        abi_version: info.abi_version,
        info_version: info.info_version,
        dispatch_id: info.dispatch_id,
        dispatch_count: info.dispatch_count,
        kernel_id: info.kernel_id,
        workgroup_size_x: info.workgroup_size_x,
        grid_size_x: info.grid_size_x,
        m: info.m,
        k: info.k,
        n: info.n,
        output_elements: info.output_elements,
        backend: info.backend,
        fallback_allowed: info.fallback_allowed != 0,
        fallback_used: info.fallback_used != 0,
        kernel_symbol: read_c_string(&info.kernel_symbol),
        device_symbol: read_c_string(&info.device_symbol),
        gcn_arch_name: read_c_string(&info.gcn_arch_name),
    }
}

pub struct MatmulSubmission {
    completion: Completion,
    _plan: Arc<PreparedMatmulState>,
}

impl std::fmt::Debug for MatmulSubmission {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MatmulSubmission")
            .finish_non_exhaustive()
    }
}

impl MatmulSubmission {
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
    pub fn prepare_matmul(
        &self,
        context: &Context,
        descriptor: MatmulDescriptor,
    ) -> Result<PreparedMatmul, RuntimeError> {
        let raw_descriptor = descriptor.raw()?;
        let mut raw_plan = std::ptr::null_mut();
        let mut buffer = [0_u8; 256];
        let mut error_sink = sink(&mut buffer);
        let status = unsafe {
            sys::sllm_matmul_prepare(
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
                "native matmul prepare returned a null plan on success".to_owned(),
            )
        })?;
        Ok(PreparedMatmul {
            state: Arc::new(PreparedMatmulState {
                raw,
                owners: PreparedMatmulOwners {
                    context: context.clone(),
                    descriptor,
                },
            }),
        })
    }
}

impl PreparedMatmul {
    pub fn execute(
        &self,
        queue: &Queue,
    ) -> Result<(MatmulSubmission, MatmulDispatchInfo), RuntimeError> {
        let mut info = sys::sllm_matmul_dispatch_info_t {
            struct_size: size_of::<sys::sllm_matmul_dispatch_info_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            info_version: sys::SLLM_HIP_MATMUL_DISPATCH_INFO_VERSION,
            backend: 0,
            dispatch_id: 0,
            dispatch_count: 0,
            kernel_id: 0,
            workgroup_size_x: 0,
            grid_size_x: 0,
            fallback_allowed: 0,
            fallback_used: 0,
            m: 0,
            k: 0,
            n: 0,
            output_elements: 0,
            kernel_symbol: [0; sys::SLLM_HIP_MATMUL_KERNEL_SYMBOL_MAX as usize],
            device_symbol: [0; sys::SLLM_HIP_MATMUL_DEVICE_SYMBOL_MAX as usize],
            gcn_arch_name: [0; sys::SLLM_HIP_MAX_GCN_ARCH_NAME as usize],
            reserved: [0; 8],
        };
        let mut error_buffer = [0_u8; 256];
        let mut error_sink = sink(&mut error_buffer);
        let mut raw_completion = std::ptr::null_mut();
        let status = unsafe {
            sys::sllm_matmul_execute(
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
                "native matmul execute returned a null completion on success".to_owned(),
            )
        })?;
        let dispatch_info = dispatch_info_from_raw(&info);
        let completion = Completion::from_native(
            raw_completion,
            &self.state.owners.context,
            queue,
            self.state.owners.descriptor.activation.buffer(),
            0,
            false,
        );
        Ok((
            MatmulSubmission {
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
    use sllm_core::{DType, TensorView};
    use std::thread;

    fn descriptor_fixture() -> (Context, MatmulDescriptor) {
        let context = Context::test_without_native();
        let activation = Buffer::test_without_native(&context)
            .binding(TensorView::contiguous(DType::Bf16, &[3, 5]).unwrap());
        let weight = Buffer::test_without_native(&context)
            .binding(TensorView::contiguous(DType::Bf16, &[7, 5]).unwrap());
        let output = Buffer::test_without_native(&context)
            .binding(TensorView::contiguous(DType::Bf16, &[3, 7]).unwrap());
        let descriptor = MatmulDescriptor::new(activation, weight, output)
            .expect("test matmul descriptor must be semantically valid");
        (context, descriptor)
    }

    #[test]
    fn descriptor_lowering_preserves_core_matmul_roles_and_layout() {
        let (_context, descriptor) = descriptor_fixture();
        assert_eq!(descriptor.semantic().kind(), SemanticOpKind::Matmul);
        assert_eq!(descriptor.semantic().inputs()[0].shape(), &[3, 5]);
        assert_eq!(descriptor.semantic().inputs()[1].shape(), &[7, 5]);
        assert_eq!(descriptor.semantic().outputs()[0].shape(), &[3, 7]);
        assert_eq!(size_of::<sys::sllm_matmul_desc_t>(), 584);
    }

    #[test]
    fn dispatch_conversion_exposes_shape_and_no_fallback() {
        let mut info = sys::sllm_matmul_dispatch_info_t {
            struct_size: size_of::<sys::sllm_matmul_dispatch_info_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            info_version: sys::SLLM_HIP_MATMUL_DISPATCH_INFO_VERSION,
            backend: sys::SLLM_BACKEND_HIP,
            dispatch_id: 9,
            dispatch_count: 1,
            kernel_id: sys::SLLM_HIP_MATMUL_KERNEL_ID_BASELINE_BF16_FP32_V1,
            workgroup_size_x: sys::SLLM_HIP_MATMUL_WORKGROUP_SIZE,
            grid_size_x: 2,
            fallback_allowed: 0,
            fallback_used: 0,
            m: 3,
            k: 5,
            n: 7,
            output_elements: 21,
            kernel_symbol: [0; 64],
            device_symbol: [0; 64],
            gcn_arch_name: [0; 64],
            reserved: [0; 8],
        };
        info.kernel_symbol[0] = b'k' as core::ffi::c_char;
        info.gcn_arch_name[0] = b'g' as core::ffi::c_char;
        let dispatch = dispatch_info_from_raw(&info);
        assert_eq!((dispatch.m, dispatch.k, dispatch.n), (3, 5, 7));
        assert_eq!(dispatch.output_elements, 21);
        assert!(!dispatch.fallback_allowed);
        assert!(!dispatch.fallback_used);
        assert_eq!(dispatch.kernel_symbol, "k");
        assert_eq!(dispatch.gcn_arch_name, "g");
    }

    #[test]
    fn invalid_native_matmul_status_maps_to_typed_runtime_status() {
        assert_eq!(
            RuntimeStatus::from_raw(sys::SLLM_STATUS_INVALID_MATMUL_DESCRIPTOR),
            RuntimeStatus::InvalidMatmulDescriptor
        );
        assert_eq!(
            RuntimeStatus::InvalidMatmulDescriptor.raw(),
            sys::SLLM_STATUS_INVALID_MATMUL_DESCRIPTOR
        );
    }

    #[test]
    fn prepared_matmul_retains_owner_graph_until_threaded_drop() {
        let _serial = crate::runtime::CLEANUP_TEST_SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        static_assertions::assert_impl_all!(PreparedMatmul: Send, Sync);
        let (context, descriptor) = descriptor_fixture();
        crate::runtime::force_matmul_plan_release_for_test(RuntimeStatus::Ok, true);
        let prepared = PreparedMatmul {
            state: Arc::new(PreparedMatmulState {
                raw: NonNull::dangling(),
                owners: PreparedMatmulOwners {
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
            .expect("PreparedMatmul drop worker must not panic");
        crate::runtime::clear_forced_matmul_plan_release_for_test();
    }

    #[test]
    fn prepared_matmul_nonconsuming_failure_retains_complete_owner_graph() {
        let _serial = crate::runtime::CLEANUP_TEST_SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let before = Context::durable_quarantine_count();
        let (context, descriptor) = descriptor_fixture();
        crate::runtime::force_matmul_plan_release_for_test(RuntimeStatus::InternalError, false);
        drop(PreparedMatmul {
            state: Arc::new(PreparedMatmulState {
                raw: NonNull::dangling(),
                owners: PreparedMatmulOwners {
                    context,
                    descriptor,
                },
            }),
        });
        assert_eq!(Context::durable_quarantine_count(), before + 1);
        crate::runtime::clear_forced_matmul_plan_release_for_test();
    }
}
