//! Exact Qwen3.8 projection-pack ABI wrapper for the bounded NVFP4 and FP8
//! GDN roles.

use std::mem::size_of;
use std::ptr::NonNull;
use std::sync::Arc;

use sllm_core::{
    Qwen38ProjectionPackContractV1, Qwen38ProjectionPackRoleV1, SemanticOpDescriptor,
    SemanticOpKind,
};
use sllm_hip_sys as sys;

use crate::runtime::{
    Completion, CompletionState, Context, Queue, RuntimeError, RuntimeStatus, ensure_ok, sink,
};
use crate::{HipBackend, TensorBinding};

#[derive(Clone)]
pub struct Qwen38ProjectionPack2Descriptor {
    activation: TensorBinding,
    gate_weight: TensorBinding,
    up_weight: TensorBinding,
    gate_output: TensorBinding,
    up_output: TensorBinding,
    role: Qwen38ProjectionPackRoleV1,
    input_global_scale_f32_bits: u32,
    _semantic: Arc<SemanticOpDescriptor>,
}

impl Qwen38ProjectionPack2Descriptor {
    pub(crate) fn from_validated_semantic(
        semantic: Arc<SemanticOpDescriptor>,
        activation: TensorBinding,
        gate_weight: TensorBinding,
        up_weight: TensorBinding,
        gate_output: TensorBinding,
        up_output: TensorBinding,
    ) -> Result<Self, RuntimeError> {
        semantic.validate().map_err(|error| {
            RuntimeError::new(RuntimeStatus::InvalidMatmulDescriptor, error.to_string())
        })?;
        let contract = semantic.qwen38_projection_pack_contract().ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::InvalidMatmulDescriptor,
                "Qwen3.8 projection-pack contract is absent",
            )
        })?;
        let role = contract.role();
        if semantic.kind() != SemanticOpKind::Qwen38ProjectionPack2
            || semantic.inputs().len() != 3
            || semantic.outputs().len() != 2
            || !matches!(
                role,
                Qwen38ProjectionPackRoleV1::Nvfp4MlpGateUp | Qwen38ProjectionPackRoleV1::Fp8GdnQkvZ
            )
            || activation.view() != &semantic.inputs()[0]
            || gate_weight.view() != &semantic.inputs()[1]
            || up_weight.view() != &semantic.inputs()[2]
            || gate_output.view() != &semantic.outputs()[0]
            || up_output.view() != &semantic.outputs()[1]
        {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidMatmulDescriptor,
                "invalid Qwen3.8 projection-pack role or bindings",
            ));
        }
        // The native FP8 GDN provider is deliberately decode-only.  Keep the
        // semantic contract broad enough for graph planning, but fail closed
        // at this ABI boundary for any shape outside its exact M=1 contract.
        if role == Qwen38ProjectionPackRoleV1::Fp8GdnQkvZ
            && (activation.view().shape()
                != [1, Qwen38ProjectionPackContractV1::HIDDEN_SIZE as usize]
                || contract.input_global_scale_f32_bits() != 0)
        {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidMatmulDescriptor,
                "Qwen3.8 FP8 GDN projection-pack requires M=1 and input scale 0",
            ));
        }
        Ok(Self {
            activation,
            gate_weight,
            up_weight,
            gate_output,
            up_output,
            role,
            input_global_scale_f32_bits: contract.input_global_scale_f32_bits(),
            _semantic: semantic,
        })
    }

    fn raw(&self) -> Result<sys::sllm_qwen38_projection_pack2_desc_t, RuntimeError> {
        Ok(sys::sllm_qwen38_projection_pack2_desc_t {
            struct_size: size_of::<sys::sllm_qwen38_projection_pack2_desc_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            op_version: sys::SLLM_HIP_QWEN38_PROJECTION_PACK2_VERSION,
            role: match self.role {
                Qwen38ProjectionPackRoleV1::Nvfp4MlpGateUp => {
                    sys::SLLM_HIP_QWEN38_PROJECTION_PACK2_ROLE_NVFP4_MLP_GATE_UP
                }
                Qwen38ProjectionPackRoleV1::Fp8GdnQkvZ => {
                    sys::SLLM_HIP_QWEN38_PROJECTION_PACK2_ROLE_FP8_GDN_QKV_Z
                }
                Qwen38ProjectionPackRoleV1::Fp8MlpGateUp
                | Qwen38ProjectionPackRoleV1::Fp8FullAttentionQkv => {
                    return Err(RuntimeError::local(
                        RuntimeStatus::InvalidMatmulDescriptor,
                        "unsupported Qwen3.8 projection-pack role for HIP ABI",
                    ));
                }
            },
            input_global_scale_f32_bits: self.input_global_scale_f32_bits,
            reserved: [0; 3],
            activation: self.activation.raw()?,
            gate_weight: self.gate_weight.raw()?,
            up_weight: self.up_weight.raw()?,
            gate_output: self.gate_output.raw()?,
            up_output: self.up_output.raw()?,
        })
    }
}

struct Owners {
    context: Context,
    descriptor: Qwen38ProjectionPack2Descriptor,
}

struct State {
    raw: NonNull<sys::sllm_qwen38_projection_pack2_plan_t>,
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
            unsafe { sys::sllm_qwen38_projection_pack2_plan_release(&mut raw, &mut error_sink) };
    }
}

#[derive(Clone)]
pub struct PreparedQwen38ProjectionPack2 {
    state: Arc<State>,
}

unsafe impl Send for PreparedQwen38ProjectionPack2 {}
unsafe impl Sync for PreparedQwen38ProjectionPack2 {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen38ProjectionPack2DispatchInfo {
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
    pub workspace_bytes: u64,
    pub role: u32,
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
    info: &sys::sllm_qwen38_projection_pack2_dispatch_info_t,
) -> Qwen38ProjectionPack2DispatchInfo {
    Qwen38ProjectionPack2DispatchInfo {
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
        workspace_bytes: info.workspace_bytes,
        role: info.role,
        backend: info.backend,
        fallback_allowed: info.fallback_allowed != 0,
        fallback_used: info.fallback_used != 0,
        kernel_symbol: read_c_string(&info.kernel_symbol),
        device_symbol: read_c_string(&info.device_symbol),
        gcn_arch_name: read_c_string(&info.gcn_arch_name),
    }
}

impl HipBackend {
    pub fn prepare_qwen38_projection_pack2(
        &self,
        context: &Context,
        descriptor: Qwen38ProjectionPack2Descriptor,
    ) -> Result<PreparedQwen38ProjectionPack2, RuntimeError> {
        let raw_descriptor = descriptor.raw()?;
        let mut raw_plan = std::ptr::null_mut();
        let mut error = [0_u8; 256];
        let mut error_sink = sink(&mut error);
        let status = unsafe {
            sys::sllm_qwen38_projection_pack2_prepare(
                context.raw_handle()?.as_ptr(),
                &raw_descriptor,
                &mut raw_plan,
                &mut error_sink,
            )
        };
        ensure_ok(status, &error, error_sink.message_length)?;
        let raw = NonNull::new(raw_plan).ok_or_else(|| {
            RuntimeError::local(
                RuntimeStatus::InternalError,
                "Qwen3.8 projection-pack prepare returned null plan",
            )
        })?;
        Ok(PreparedQwen38ProjectionPack2 {
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

impl PreparedQwen38ProjectionPack2 {
    pub(crate) fn raw_plan_handle(&self) -> *const sys::sllm_qwen38_projection_pack2_plan_t {
        self.state.raw.as_ptr()
    }

    pub fn execute(
        &self,
        queue: &Queue,
    ) -> Result<
        (
            Qwen38ProjectionPack2Submission,
            Qwen38ProjectionPack2DispatchInfo,
        ),
        RuntimeError,
    > {
        let mut info = sys::sllm_qwen38_projection_pack2_dispatch_info_t {
            struct_size: size_of::<sys::sllm_qwen38_projection_pack2_dispatch_info_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            info_version: sys::SLLM_HIP_QWEN38_PROJECTION_PACK2_DISPATCH_INFO_VERSION,
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
            workspace_bytes: 0,
            role: 0,
            reserved0: 0,
            kernel_symbol: [0; sys::SLLM_HIP_MATMUL_KERNEL_SYMBOL_MAX as usize],
            device_symbol: [0; sys::SLLM_HIP_MATMUL_DEVICE_SYMBOL_MAX as usize],
            gcn_arch_name: [0; sys::SLLM_HIP_MAX_GCN_ARCH_NAME as usize],
            reserved: [0; 8],
        };
        let mut error = [0_u8; 256];
        let mut error_sink = sink(&mut error);
        let mut raw_completion = std::ptr::null_mut();
        let status = unsafe {
            sys::sllm_qwen38_projection_pack2_execute(
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
                "Qwen3.8 projection-pack execute returned null completion",
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
            Qwen38ProjectionPack2Submission {
                completion,
                _plan: Arc::clone(&self.state),
            },
            dispatch_info_from_raw(&info),
        ))
    }
}

pub struct Qwen38ProjectionPack2Submission {
    completion: Completion,
    _plan: Arc<State>,
}

impl Qwen38ProjectionPack2Submission {
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
