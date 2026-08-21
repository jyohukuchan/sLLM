//! Safe prepared categorical selection over one BF16 logits row.

use std::mem::size_of;
use std::ptr::NonNull;
use std::sync::Arc;

use sllm_hip_sys as sys;

use crate::runtime::{
    Completion, CompletionState, Context, Queue, RuntimeError, RuntimeStatus, ensure_ok,
    release_token_selector_plan_once, sink,
};
use crate::{HipBackend, TensorBinding};

#[derive(Clone, Debug)]
pub struct TokenSelectorDescriptor {
    logits: TensorBinding,
    additive_logits: TensorBinding,
    valid_mask: TensorBinding,
    output: TensorBinding,
    vocab_size: u64,
    temperature: f32,
    seed: u64,
    counter: u64,
}

impl TokenSelectorDescriptor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        logits: TensorBinding,
        additive_logits: TensorBinding,
        valid_mask: TensorBinding,
        output: TensorBinding,
        vocab_size: u64,
        temperature: f32,
        seed: u64,
        counter: u64,
    ) -> Result<Self, RuntimeError> {
        if vocab_size == 0 || !temperature.is_finite() || temperature <= 0.0 {
            return Err(RuntimeError::local(
                RuntimeStatus::InvalidArgument,
                "token selector vocabulary or temperature is invalid",
            ));
        }
        Ok(Self {
            logits,
            additive_logits,
            valid_mask,
            output,
            vocab_size,
            temperature,
            seed,
            counter,
        })
    }

    pub fn logits(&self) -> &TensorBinding {
        &self.logits
    }

    pub fn output(&self) -> &TensorBinding {
        &self.output
    }

    fn raw(&self) -> Result<sys::sllm_token_selector_desc_t, RuntimeError> {
        Ok(sys::sllm_token_selector_desc_t {
            struct_size: size_of::<sys::sllm_token_selector_desc_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            op_version: sys::SLLM_HIP_TOKEN_SELECTOR_VERSION,
            reserved: [0; 4],
            logits: self.logits.raw()?,
            additive_logits: self.additive_logits.raw()?,
            valid_mask: self.valid_mask.raw()?,
            output: self.output.raw()?,
            vocab_size: self.vocab_size,
            temperature: self.temperature,
            seed: self.seed,
            counter: self.counter,
        })
    }
}

struct PreparedTokenSelectorOwners {
    context: Context,
    descriptor: TokenSelectorDescriptor,
}

struct PreparedTokenSelectorState {
    raw: NonNull<sys::sllm_token_selector_plan_t>,
    owners: PreparedTokenSelectorOwners,
}

unsafe impl Send for PreparedTokenSelectorState {}
unsafe impl Sync for PreparedTokenSelectorState {}

impl Drop for PreparedTokenSelectorState {
    fn drop(&mut self) {
        let _ = release_token_selector_plan_once(self.raw);
    }
}

#[derive(Clone)]
pub struct PreparedTokenSelector {
    state: Arc<PreparedTokenSelectorState>,
}

unsafe impl Send for PreparedTokenSelector {}
unsafe impl Sync for PreparedTokenSelector {}

impl std::fmt::Debug for PreparedTokenSelector {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PreparedTokenSelector")
            .field("vocab_size", &self.state.owners.descriptor.vocab_size)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TokenSelectorRecord {
    pub token_id: i32,
    pub status: u32,
    pub logprob: f32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenSelectorDispatchInfo {
    pub abi_version: u32,
    pub info_version: u32,
    pub dispatch_id: u64,
    pub dispatch_count: u32,
    pub kernel_id: u32,
    pub workgroup_size_x: u32,
    pub grid_size_x: u32,
    pub vocab_size: u64,
    pub fallback_allowed: bool,
    pub fallback_used: bool,
    pub result_status: u32,
    pub token_id: i32,
    pub backend: u32,
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
    info: &sys::sllm_token_selector_dispatch_info_t,
) -> TokenSelectorDispatchInfo {
    TokenSelectorDispatchInfo {
        abi_version: info.abi_version,
        info_version: info.info_version,
        dispatch_id: info.dispatch_id,
        dispatch_count: info.dispatch_count,
        kernel_id: info.kernel_id,
        workgroup_size_x: info.workgroup_size_x,
        grid_size_x: info.grid_size_x,
        vocab_size: info.vocab_size,
        fallback_allowed: info.fallback_allowed != 0,
        fallback_used: info.fallback_used != 0,
        result_status: info.result_status,
        token_id: info.token_id,
        backend: info.backend,
        kernel_symbol: read_c_string(&info.kernel_symbol),
        device_symbol: read_c_string(&info.device_symbol),
        gcn_arch_name: read_c_string(&info.gcn_arch_name),
    }
}

pub struct TokenSelectorSubmission {
    completion: Completion,
    plan: Arc<PreparedTokenSelectorState>,
}

impl std::fmt::Debug for TokenSelectorSubmission {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TokenSelectorSubmission")
            .finish_non_exhaustive()
    }
}

impl TokenSelectorSubmission {
    pub fn query(&mut self) -> Result<CompletionState, RuntimeError> {
        self.completion.query()
    }

    pub fn wait(&mut self, timeout: std::time::Duration) -> Result<CompletionState, RuntimeError> {
        self.completion.wait(timeout)
    }

    pub fn kernel_elapsed_ns(&mut self) -> Result<u64, RuntimeError> {
        self.completion.kernel_elapsed_ns()
    }

    pub(crate) fn finalize_after_token(
        &mut self,
        fence_token: u64,
    ) -> Result<CompletionState, RuntimeError> {
        self.completion.finalize_after_token(fence_token)
    }

    pub fn read_record(&mut self, queue: &Queue) -> Result<TokenSelectorRecord, RuntimeError> {
        if !matches!(
            self.completion.wait(std::time::Duration::from_secs(30))?,
            CompletionState::Success
        ) {
            return Err(RuntimeError::local(
                RuntimeStatus::NotReady,
                "token selector completion is not successful",
            ));
        }
        let output = self.plan.owners.descriptor.output.buffer().clone();
        let mut copy = queue.copy_to_host(
            &output,
            u64::from(sys::SLLM_HIP_TOKEN_SELECTOR_OUTPUT_BYTES),
            self.plan.owners.descriptor.output.view().byte_offset(),
        )?;
        copy.wait(std::time::Duration::from_secs(30))?;
        let mut bytes = [0_u8; sys::SLLM_HIP_TOKEN_SELECTOR_OUTPUT_BYTES as usize];
        copy.read_into(&mut bytes)?;
        let token_id = i32::from_ne_bytes(bytes[0..4].try_into().expect("record token id"));
        let status = u32::from_ne_bytes(bytes[4..8].try_into().expect("record status"));
        let logprob = f32::from_bits(u32::from_ne_bytes(
            bytes[8..12].try_into().expect("record logprob"),
        ));
        Ok(TokenSelectorRecord {
            token_id,
            status,
            logprob,
        })
    }
}

impl HipBackend {
    pub fn prepare_token_selector(
        &self,
        context: &Context,
        descriptor: TokenSelectorDescriptor,
    ) -> Result<PreparedTokenSelector, RuntimeError> {
        let raw_descriptor = descriptor.raw()?;
        let mut raw_plan = std::ptr::null_mut();
        let mut error_buffer = [0_u8; 256];
        let mut error_sink = sink(&mut error_buffer);
        let status = unsafe {
            sys::sllm_token_selector_prepare(
                context.raw_handle()?.as_ptr(),
                &raw_descriptor,
                &mut raw_plan,
                &mut error_sink,
            )
        };
        ensure_ok(status, &error_buffer, error_sink.message_length)?;
        let raw = NonNull::new(raw_plan).ok_or_else(|| {
            RuntimeError::new(
                RuntimeStatus::InternalError,
                "native token selector prepare returned a null plan on success".to_owned(),
            )
        })?;
        Ok(PreparedTokenSelector {
            state: Arc::new(PreparedTokenSelectorState {
                raw,
                owners: PreparedTokenSelectorOwners {
                    context: context.clone(),
                    descriptor,
                },
            }),
        })
    }
}

impl PreparedTokenSelector {
    pub fn execute(
        &self,
        queue: &Queue,
    ) -> Result<(TokenSelectorSubmission, TokenSelectorDispatchInfo), RuntimeError> {
        let mut info = sys::sllm_token_selector_dispatch_info_t {
            struct_size: size_of::<sys::sllm_token_selector_dispatch_info_t>() as u32,
            abi_version: sys::SLLM_HIP_ABI_VERSION,
            info_version: sys::SLLM_HIP_TOKEN_SELECTOR_DISPATCH_INFO_VERSION,
            backend: 0,
            dispatch_id: 0,
            dispatch_count: 0,
            kernel_id: 0,
            workgroup_size_x: 0,
            grid_size_x: 0,
            vocab_size: 0,
            fallback_allowed: 0,
            fallback_used: 0,
            result_status: 0,
            token_id: -1,
            kernel_symbol: [0; sys::SLLM_HIP_TOKEN_SELECTOR_KERNEL_SYMBOL_MAX as usize],
            device_symbol: [0; sys::SLLM_HIP_TOKEN_SELECTOR_DEVICE_SYMBOL_MAX as usize],
            gcn_arch_name: [0; sys::SLLM_HIP_MAX_GCN_ARCH_NAME as usize],
            reserved: [0; 8],
        };
        let mut raw_completion = std::ptr::null_mut();
        let mut error_buffer = [0_u8; 256];
        let mut error_sink = sink(&mut error_buffer);
        let status = unsafe {
            sys::sllm_token_selector_execute(
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
                "native token selector execute returned a null completion on success".to_owned(),
            )
        })?;
        let dispatch = dispatch_info_from_raw(&info);
        let completion = Completion::from_native(
            raw_completion,
            &self.state.owners.context,
            queue,
            self.state.owners.descriptor.output.buffer(),
            0,
            false,
        );
        Ok((
            TokenSelectorSubmission {
                completion,
                plan: Arc::clone(&self.state),
            },
            dispatch,
        ))
    }
}
