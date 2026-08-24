//! Fixed-role Qwen3.5 GDN projection bundle ABI wrapper.

use std::mem::size_of;
use std::ptr::NonNull;
use std::sync::Arc;

use sllm_core::{SemanticOpDescriptor, SemanticOpKind};
use sllm_hip_sys as sys;

use crate::runtime::{
    Completion, CompletionState, Context, Queue, RuntimeError, RuntimeStatus, ensure_ok, sink,
};
use crate::{HipBackend, TensorBinding};

#[derive(Clone)]
pub struct GdnProjectionBundleDescriptor {
    activation: TensorBinding,
    weights: [TensorBinding; 4],
    outputs: [TensorBinding; 4],
    _semantic: Arc<SemanticOpDescriptor>,
}

impl GdnProjectionBundleDescriptor {
    pub(crate) fn from_validated_semantic(
        semantic: Arc<SemanticOpDescriptor>,
        activation: TensorBinding,
        weights: [TensorBinding; 4],
        outputs: [TensorBinding; 4],
    ) -> Result<Self, RuntimeError> {
        semantic.validate().map_err(|error| {
            RuntimeError::new(RuntimeStatus::InvalidMatmulDescriptor, error.to_string())
        })?;
        if semantic.kind() != SemanticOpKind::GdnProjectionBundle
            || semantic.inputs().len() != 5
            || semantic.outputs().len() != 4
            || activation.view() != &semantic.inputs()[0]
            || weights
                .iter()
                .enumerate()
                .any(|(i, b)| b.view() != &semantic.inputs()[i + 1])
            || outputs
                .iter()
                .enumerate()
                .any(|(i, b)| b.view() != &semantic.outputs()[i])
        {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidMatmulDescriptor,
                "invalid GDN projection bundle bindings",
            ));
        }
        Ok(Self {
            activation,
            weights,
            outputs,
            _semantic: semantic,
        })
    }

    fn raw(&self) -> Result<sys::sllm_gdn_projection_bundle_desc_t, RuntimeError> {
        let weights = self
            .weights
            .iter()
            .map(TensorBinding::raw)
            .collect::<Result<Vec<_>, _>>()?;
        let outputs = self
            .outputs
            .iter()
            .map(TensorBinding::raw)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(sys::sllm_gdn_projection_bundle_desc_t {
            struct_size: size_of::<sys::sllm_gdn_projection_bundle_desc_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            op_version: sys::SLLM_HIP_GDN_PROJECTION_BUNDLE_VERSION,
            reserved: [0; 5],
            activation: self.activation.raw()?,
            weights: weights.try_into().map_err(|_| {
                RuntimeError::local(
                    RuntimeStatus::InvalidMatmulDescriptor,
                    "weight ABI array length mismatch",
                )
            })?,
            outputs: outputs.try_into().map_err(|_| {
                RuntimeError::local(
                    RuntimeStatus::InvalidMatmulDescriptor,
                    "output ABI array length mismatch",
                )
            })?,
        })
    }
}

struct Owners {
    context: Context,
    descriptor: GdnProjectionBundleDescriptor,
}
struct State {
    raw: NonNull<sys::sllm_gdn_projection_bundle_plan_t>,
    owners: Owners,
}
unsafe impl Send for State {}
unsafe impl Sync for State {}
impl Drop for State {
    fn drop(&mut self) {
        let mut raw = self.raw.as_ptr();
        let mut error = [0_u8; 256];
        let mut error_sink = sink(&mut error);
        let _ = unsafe { sys::sllm_gdn_projection_bundle_plan_release(&mut raw, &mut error_sink) };
    }
}

#[derive(Clone)]
pub struct PreparedGdnProjectionBundle {
    state: Arc<State>,
}
unsafe impl Send for PreparedGdnProjectionBundle {}
unsafe impl Sync for PreparedGdnProjectionBundle {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GdnProjectionBundleDispatchInfo {
    pub dispatch_id: u64,
    pub dispatch_count: u32,
    pub grid_size_x: u32,
    pub workgroup_size_x: u32,
    pub fallback_used: u32,
    pub m: u64,
    pub widths: [u32; 4],
    pub gcn_arch_name: String,
}

fn dispatch_info(
    raw: &sys::sllm_gdn_projection_bundle_dispatch_info_t,
) -> GdnProjectionBundleDispatchInfo {
    let end = raw
        .gcn_arch_name
        .iter()
        .position(|b| *b == 0)
        .unwrap_or(raw.gcn_arch_name.len());
    GdnProjectionBundleDispatchInfo {
        dispatch_id: raw.dispatch_id,
        dispatch_count: raw.dispatch_count,
        grid_size_x: raw.grid_size_x,
        workgroup_size_x: raw.workgroup_size_x,
        fallback_used: raw.fallback_used,
        m: raw.m,
        widths: raw.widths,
        gcn_arch_name: raw.gcn_arch_name[..end]
            .iter()
            .map(|b| *b as u8)
            .collect::<Vec<_>>()
            .into_iter()
            .map(char::from)
            .collect(),
    }
}

impl HipBackend {
    pub fn prepare_gdn_projection_bundle(
        &self,
        context: &Context,
        descriptor: GdnProjectionBundleDescriptor,
    ) -> Result<PreparedGdnProjectionBundle, RuntimeError> {
        let raw_desc = descriptor.raw()?;
        let mut raw_plan = std::ptr::null_mut();
        let mut error = [0_u8; 256];
        let mut sink = sink(&mut error);
        let status = unsafe {
            sys::sllm_gdn_projection_bundle_prepare(
                context.raw_handle()?.as_ptr(),
                &raw_desc,
                &mut raw_plan,
                &mut sink,
            )
        };
        ensure_ok(status, &error, sink.message_length)?;
        let raw = NonNull::new(raw_plan).ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::InternalError,
                "bundle prepare returned null plan",
            )
        })?;
        Ok(PreparedGdnProjectionBundle {
            state: Arc::new(State {
                raw,
                owners: Owners {
                    context: context.clone(),
                    descriptor,
                },
            }),
        })
    }
}

impl PreparedGdnProjectionBundle {
    pub fn execute(
        &self,
        queue: &Queue,
    ) -> Result<
        (
            GdnProjectionBundleSubmission,
            GdnProjectionBundleDispatchInfo,
        ),
        RuntimeError,
    > {
        let mut info = sys::sllm_gdn_projection_bundle_dispatch_info_t {
            struct_size: size_of::<sys::sllm_gdn_projection_bundle_dispatch_info_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            info_version: sys::SLLM_HIP_GDN_PROJECTION_BUNDLE_DISPATCH_INFO_VERSION,
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
            widths: [0; 4],
            reserved0: 0,
            kernel_symbol: [0; sys::SLLM_HIP_MATMUL_KERNEL_SYMBOL_MAX as usize],
            device_symbol: [0; sys::SLLM_HIP_MATMUL_DEVICE_SYMBOL_MAX as usize],
            gcn_arch_name: [0; sys::SLLM_HIP_MAX_GCN_ARCH_NAME as usize],
            reserved: [0; 8],
        };
        let mut error = [0_u8; 256];
        let mut sink = sink(&mut error);
        let mut raw_completion = std::ptr::null_mut();
        let status = unsafe {
            sys::sllm_gdn_projection_bundle_execute(
                self.state.raw.as_ptr(),
                queue.raw_handle()?.as_ptr(),
                &mut raw_completion,
                &mut info,
                &mut sink,
            )
        };
        ensure_ok(status, &error, sink.message_length)?;
        let raw_completion = NonNull::new(raw_completion).ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::InternalError,
                "bundle execute returned null completion",
            )
        })?;
        let completion = Completion::from_native(
            raw_completion,
            &self.state.owners.context,
            queue,
            self.state.owners.descriptor.activation.buffer(),
            0,
            false,
        );
        Ok((
            GdnProjectionBundleSubmission {
                completion,
                _plan: Arc::clone(&self.state),
            },
            dispatch_info(&info),
        ))
    }
}

pub struct GdnProjectionBundleSubmission {
    completion: Completion,
    _plan: Arc<State>,
}
impl GdnProjectionBundleSubmission {
    pub fn query(&mut self) -> Result<CompletionState, RuntimeError> {
        self.completion.query()
    }
    pub fn wait(&mut self, timeout: std::time::Duration) -> Result<CompletionState, RuntimeError> {
        self.completion.wait(timeout)
    }
    pub(crate) fn kernel_elapsed_ns(&mut self) -> Result<u64, RuntimeError> {
        self.completion.kernel_elapsed_ns()
    }
    pub(crate) fn finalize_after_token(
        &mut self,
        token: u64,
    ) -> Result<CompletionState, RuntimeError> {
        self.completion.finalize_after_token(token)
    }
}
