//! Fixed-role Qwen3.5 MLP gate/up/SiLU bundle ABI wrapper.

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
pub struct MlpGateUpSiluBundleDescriptor {
    activation: TensorBinding,
    gate_weight: TensorBinding,
    up_weight: TensorBinding,
    gate_output: TensorBinding,
    up_output: TensorBinding,
    silu_output: TensorBinding,
    _semantic: Arc<SemanticOpDescriptor>,
}

impl MlpGateUpSiluBundleDescriptor {
    pub(crate) fn from_validated_semantic(
        semantic: Arc<SemanticOpDescriptor>,
        activation: TensorBinding,
        gate_weight: TensorBinding,
        up_weight: TensorBinding,
        gate_output: TensorBinding,
        up_output: TensorBinding,
        silu_output: TensorBinding,
    ) -> Result<Self, RuntimeError> {
        semantic.validate().map_err(|error| {
            RuntimeError::new(RuntimeStatus::InvalidMatmulDescriptor, error.to_string())
        })?;
        if semantic.kind() != SemanticOpKind::MlpGateUpSiluBundle
            || semantic.inputs().len() != 3
            || semantic.outputs().len() != 3
            || activation.view() != &semantic.inputs()[0]
            || gate_weight.view() != &semantic.inputs()[1]
            || up_weight.view() != &semantic.inputs()[2]
            || gate_output.view() != &semantic.outputs()[0]
            || up_output.view() != &semantic.outputs()[1]
            || silu_output.view() != &semantic.outputs()[2]
        {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidMatmulDescriptor,
                "invalid MLP gate/up/SiLU bundle bindings",
            ));
        }
        Ok(Self {
            activation,
            gate_weight,
            up_weight,
            gate_output,
            up_output,
            silu_output,
            _semantic: semantic,
        })
    }

    fn raw(&self) -> Result<sys::sllm_mlp_gate_up_silu_bundle_desc_t, RuntimeError> {
        Ok(sys::sllm_mlp_gate_up_silu_bundle_desc_t {
            struct_size: size_of::<sys::sllm_mlp_gate_up_silu_bundle_desc_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            op_version: sys::SLLM_HIP_MLP_GATE_UP_SILU_BUNDLE_VERSION,
            reserved: [0; 5],
            activation: self.activation.raw()?,
            gate_weight: self.gate_weight.raw()?,
            up_weight: self.up_weight.raw()?,
            gate_output: self.gate_output.raw()?,
            up_output: self.up_output.raw()?,
            silu_output: self.silu_output.raw()?,
        })
    }
}

struct Owners {
    context: Context,
    descriptor: MlpGateUpSiluBundleDescriptor,
}
struct State {
    raw: NonNull<sys::sllm_mlp_gate_up_silu_bundle_plan_t>,
    owners: Owners,
}
unsafe impl Send for State {}
unsafe impl Sync for State {}
impl Drop for State {
    fn drop(&mut self) {
        let mut raw = self.raw.as_ptr();
        let mut error = [0_u8; 256];
        let mut error_sink = sink(&mut error);
        let _ =
            unsafe { sys::sllm_mlp_gate_up_silu_bundle_plan_release(&mut raw, &mut error_sink) };
    }
}

#[derive(Clone)]
pub struct PreparedMlpGateUpSiluBundle {
    state: Arc<State>,
}
unsafe impl Send for PreparedMlpGateUpSiluBundle {}
unsafe impl Sync for PreparedMlpGateUpSiluBundle {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MlpGateUpSiluBundleDispatchInfo {
    pub dispatch_id: u64,
    pub dispatch_count: u32,
    pub kernel_id: u32,
    pub grid_size_x: u32,
    pub workgroup_size_x: u32,
    pub fallback_used: u32,
    pub m: u64,
    pub k: u64,
    pub n: u64,
    pub kernel_symbol: String,
    pub device_symbol: String,
    pub gcn_arch_name: String,
}

fn dispatch_info(
    raw: &sys::sllm_mlp_gate_up_silu_bundle_dispatch_info_t,
) -> MlpGateUpSiluBundleDispatchInfo {
    fn fixed_string(values: &[std::ffi::c_char]) -> String {
        let end = values
            .iter()
            .position(|value| *value == 0)
            .unwrap_or(values.len());
        values[..end]
            .iter()
            .map(|value| *value as u8 as char)
            .collect()
    }
    let end = raw
        .gcn_arch_name
        .iter()
        .position(|b| *b == 0)
        .unwrap_or(raw.gcn_arch_name.len());
    MlpGateUpSiluBundleDispatchInfo {
        dispatch_id: raw.dispatch_id,
        dispatch_count: raw.dispatch_count,
        kernel_id: raw.kernel_id,
        grid_size_x: raw.grid_size_x,
        workgroup_size_x: raw.workgroup_size_x,
        fallback_used: raw.fallback_used,
        m: raw.m,
        k: raw.k,
        n: raw.n,
        kernel_symbol: fixed_string(&raw.kernel_symbol),
        device_symbol: fixed_string(&raw.device_symbol),
        gcn_arch_name: raw.gcn_arch_name[..end]
            .iter()
            .map(|b| *b as u8)
            .map(char::from)
            .collect(),
    }
}

impl HipBackend {
    pub fn prepare_mlp_gate_up_silu_bundle(
        &self,
        context: &Context,
        descriptor: MlpGateUpSiluBundleDescriptor,
    ) -> Result<PreparedMlpGateUpSiluBundle, RuntimeError> {
        let raw_desc = descriptor.raw()?;
        let mut raw_plan = std::ptr::null_mut();
        let mut error = [0_u8; 256];
        let mut error_sink = sink(&mut error);
        let status = unsafe {
            sys::sllm_mlp_gate_up_silu_bundle_prepare(
                context.raw_handle()?.as_ptr(),
                &raw_desc,
                &mut raw_plan,
                &mut error_sink,
            )
        };
        ensure_ok(status, &error, error_sink.message_length)?;
        let raw = NonNull::new(raw_plan).ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::InternalError,
                "MLP gate/up/SiLU bundle prepare returned null plan",
            )
        })?;
        Ok(PreparedMlpGateUpSiluBundle {
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

impl PreparedMlpGateUpSiluBundle {
    pub fn execute(
        &self,
        queue: &Queue,
    ) -> Result<
        (
            MlpGateUpSiluBundleSubmission,
            MlpGateUpSiluBundleDispatchInfo,
        ),
        RuntimeError,
    > {
        let mut info = sys::sllm_mlp_gate_up_silu_bundle_dispatch_info_t {
            struct_size: size_of::<sys::sllm_mlp_gate_up_silu_bundle_dispatch_info_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            info_version: sys::SLLM_HIP_MLP_GATE_UP_SILU_BUNDLE_DISPATCH_INFO_VERSION,
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
        let mut error = [0_u8; 256];
        let mut error_sink = sink(&mut error);
        let mut raw_completion = std::ptr::null_mut();
        let status = unsafe {
            sys::sllm_mlp_gate_up_silu_bundle_execute(
                self.state.raw.as_ptr(),
                queue.raw_handle()?.as_ptr(),
                &mut raw_completion,
                &mut info,
                &mut error_sink,
            )
        };
        ensure_ok(status, &error, error_sink.message_length)?;
        let raw_completion = NonNull::new(raw_completion).ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::InternalError,
                "MLP gate/up/SiLU bundle execute returned null completion",
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
            MlpGateUpSiluBundleSubmission {
                completion,
                _plan: Arc::clone(&self.state),
            },
            dispatch_info(&info),
        ))
    }
}

pub struct MlpGateUpSiluBundleSubmission {
    completion: Completion,
    _plan: Arc<State>,
}
impl MlpGateUpSiluBundleSubmission {
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
