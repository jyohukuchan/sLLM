// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

use crate::host_bytes::{decode_f32_le_values, encode_f32_to_bytes};
use crate::loader::{
    PassthroughBf16ResidentData, PassthroughPayloadVerification,
    load_named_passthrough_bf16_resident, read_named_passthrough_f32,
    verify_named_passthrough_payload,
};
use crate::sq_reference::sq8_f32_le_sha256;
use crate::sq8_layer_oracle::{QWEN3_14B_HIDDEN_SIZE, QWEN3_14B_RMS_NORM_EPSILON};
use crate::sq8_layer_runtime::QWEN3_14B_SQ8_REQUIRED_HIP_KERNEL_ENV;
use crate::sq8_stack_runtime::{Qwen3Sq8StackRuntime, Sq8StackExecutionReport};
use sha2::{Digest, Sha256};
use std::path::Path;
use ullm_runtime_sys::{
    DeviceInfo, RuntimeBuffer, RuntimeContext, RuntimeStream, matvec_bf16_f32,
    segmented_rmsnorm_f32,
};

pub const QWEN3_14B_VOCAB_SIZE: usize = 151_936;
pub const QWEN3_14B_FINAL_NORM_TENSOR: &str = "model.norm.weight";
pub const QWEN3_14B_LM_HEAD_TENSOR: &str = "lm_head.weight";
pub const QWEN3_14B_SQ8_MODEL_HEAD_REQUIRED_HIP_KERNEL_ENV: [&str; 2] = [
    "ULLM_REQUIRE_HIP_RMSNORM_KERNEL",
    "ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL",
];

const BF16_DTYPE: &str = "BF16";
const R9700_RUNTIME_NAME: &str = "AMD Radeon Graphics";
const R9700_MEMORY_BYTES_MIN: u64 = 30 * 1024 * 1024 * 1024;
const R9700_MEMORY_BYTES_MAX: u64 = 34 * 1024 * 1024 * 1024;
const QWEN3_14B_FINAL_NORM_SHAPE: [u64; 1] = [QWEN3_14B_HIDDEN_SIZE as u64];
const QWEN3_14B_LM_HEAD_SHAPE: [u64; 2] =
    [QWEN3_14B_VOCAB_SIZE as u64, QWEN3_14B_HIDDEN_SIZE as u64];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sq8ModelHeadRuntimeStatus {
    Ready,
    OutputReady,
    Poisoned,
}

#[derive(Debug)]
enum Sq8ModelHeadRuntimeState {
    Ready,
    OutputReady,
    Poisoned(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sq8ModelHeadDeviceIdentity {
    pub device_id: i32,
    pub backend: String,
    pub name: String,
    pub gcn_arch_name: String,
    pub compute_major: i32,
    pub compute_minor: i32,
    pub total_global_mem: u64,
}

impl Sq8ModelHeadDeviceIdentity {
    fn from_runtime(value: DeviceInfo) -> Self {
        Self {
            device_id: value.device_id,
            backend: value.backend,
            name: value.name,
            gcn_arch_name: value.gcn_arch_name,
            compute_major: value.compute_major,
            compute_minor: value.compute_minor,
            total_global_mem: value.total_global_mem,
        }
    }

    fn validate_r9700(&self) -> Result<(), String> {
        let arch = self.gcn_arch_name.split(':').next().unwrap_or_default();
        if self.backend != "hip"
            || self.name != R9700_RUNTIME_NAME
            || (!arch.is_empty() && arch != "gfx1201")
            || self.compute_major != 12
            || self.compute_minor != 0
            || !(R9700_MEMORY_BYTES_MIN..=R9700_MEMORY_BYTES_MAX).contains(&self.total_global_mem)
        {
            return Err(format!(
                "Qwen3-14B SQ8 model head requires the canonical R9700/gfx1201 HIP identity, got backend={} name={} arch={} compute={}.{} memory={}",
                self.backend,
                self.name,
                self.gcn_arch_name,
                self.compute_major,
                self.compute_minor,
                self.total_global_mem
            ));
        }
        Ok(())
    }
}

pub fn validate_qwen3_14b_sq8_r9700_device_info(info: &DeviceInfo) -> Result<(), String> {
    Sq8ModelHeadDeviceIdentity::from_runtime(info.clone()).validate_r9700()
}

impl Sq8ModelHeadRuntimeState {
    fn status(&self) -> Sq8ModelHeadRuntimeStatus {
        match self {
            Self::Ready => Sq8ModelHeadRuntimeStatus::Ready,
            Self::OutputReady => Sq8ModelHeadRuntimeStatus::OutputReady,
            Self::Poisoned(_) => Sq8ModelHeadRuntimeStatus::Poisoned,
        }
    }

    fn ensure_usable(&self) -> Result<(), String> {
        match self {
            Self::Ready | Self::OutputReady => Ok(()),
            Self::Poisoned(reason) => Err(format!(
                "Qwen3-14B SQ8 model head is permanently poisoned: {reason}"
            )),
        }
    }

    fn poison(&mut self, reason: String) {
        if !matches!(self, Self::Poisoned(_)) {
            *self = Self::Poisoned(reason);
        }
    }

    fn poison_reason(&self) -> Option<&str> {
        match self {
            Self::Poisoned(reason) => Some(reason),
            Self::Ready | Self::OutputReady => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sq8ModelHeadPayloadIdentity {
    pub tensor_name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub elements: u64,
    pub payload_bytes: u64,
    pub payload_sha256: String,
    pub processed_chunks: u64,
}

impl Sq8ModelHeadPayloadIdentity {
    fn validate(
        &self,
        expected_name: &str,
        expected_shape: &[u64],
        expected_elements: u64,
        expected_bytes: u64,
    ) -> Result<(), String> {
        if self.tensor_name != expected_name {
            return Err(format!(
                "SQ8 model-head tensor name mismatch: expected={expected_name} actual={}",
                self.tensor_name
            ));
        }
        if self.dtype != BF16_DTYPE {
            return Err(format!(
                "SQ8 model-head tensor {expected_name} dtype mismatch: expected={BF16_DTYPE} actual={}",
                self.dtype
            ));
        }
        if self.shape != expected_shape {
            return Err(format!(
                "SQ8 model-head tensor {expected_name} shape mismatch: expected={expected_shape:?} actual={:?}",
                self.shape
            ));
        }
        if self.elements != expected_elements || self.payload_bytes != expected_bytes {
            return Err(format!(
                "SQ8 model-head tensor {expected_name} size mismatch: expected_elements={expected_elements} actual_elements={} expected_bytes={expected_bytes} actual_bytes={}",
                self.elements, self.payload_bytes
            ));
        }
        validate_sha256(&self.payload_sha256).map_err(|err| {
            format!("SQ8 model-head tensor {expected_name} payload hash is invalid: {err}")
        })?;
        if self.processed_chunks == 0 {
            return Err(format!(
                "SQ8 model-head tensor {expected_name} must have at least one processed chunk"
            ));
        }
        Ok(())
    }
}

impl From<PassthroughPayloadVerification> for Sq8ModelHeadPayloadIdentity {
    fn from(value: PassthroughPayloadVerification) -> Self {
        Self {
            tensor_name: value.tensor_name,
            dtype: value.dtype,
            shape: value.shape,
            elements: value.elements,
            payload_bytes: value.payload_bytes,
            payload_sha256: value.payload_sha256,
            processed_chunks: value.verified_chunks,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sq8ModelHeadTensorHealth {
    pub elements: usize,
    pub all_finite: bool,
    pub minimum: f32,
    pub maximum: f32,
    pub f32_le_sha256: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sq8ModelHeadExecutionReport {
    pub device: Sq8ModelHeadDeviceIdentity,
    pub stack_execution: Sq8StackExecutionReport,
    pub stack_artifact_content_sha256: String,
    pub sequence_len: usize,
    pub selected_row: usize,
    pub final_norm: Sq8ModelHeadPayloadIdentity,
    pub lm_head: Sq8ModelHeadPayloadIdentity,
    pub final_hidden_health: Sq8ModelHeadTensorHealth,
    pub logits_health: Sq8ModelHeadTensorHealth,
    pub stack_kernel_guard_checks: usize,
    pub head_kernel_guard_checks: usize,
    pub d2d_copy_count: usize,
    pub rmsnorm_call_count: usize,
    pub bf16_matvec_call_count: usize,
    pub result_readback_count: usize,
    pub execution_synchronization_count: usize,
    pub fallback_used: bool,
    pub host_staging_used: bool,
}

impl Sq8ModelHeadExecutionReport {
    pub fn validate_contract(&self) -> Result<(), String> {
        self.device.validate_r9700()?;
        self.stack_execution.validate_optimized_promotion()?;
        if self.stack_execution.artifact_content_sha256 != self.stack_artifact_content_sha256
            || self.stack_execution.sequence_len != self.sequence_len
        {
            return Err(
                "Qwen3-14B SQ8 model-head report is not bound to its optimized stack execution"
                    .into(),
            );
        }
        if !matches!(self.sequence_len, 1 | 2 | 4 | 8 | 16 | 32 | 128) {
            return Err(format!(
                "Qwen3-14B SQ8 model-head report has unmeasured sequence length {}",
                self.sequence_len
            ));
        }
        if self.selected_row != self.sequence_len - 1 {
            return Err(format!(
                "Qwen3-14B SQ8 model-head selected row mismatch: expected={} actual={}",
                self.sequence_len - 1,
                self.selected_row
            ));
        }
        validate_sha256(&self.stack_artifact_content_sha256).map_err(|err| {
            format!("Qwen3-14B SQ8 model-head stack artifact hash is invalid: {err}")
        })?;
        self.final_norm.validate(
            QWEN3_14B_FINAL_NORM_TENSOR,
            &QWEN3_14B_FINAL_NORM_SHAPE,
            QWEN3_14B_HIDDEN_SIZE as u64,
            bf16_bytes(QWEN3_14B_HIDDEN_SIZE)? as u64,
        )?;
        self.lm_head.validate(
            QWEN3_14B_LM_HEAD_TENSOR,
            &QWEN3_14B_LM_HEAD_SHAPE,
            checked_elements(QWEN3_14B_VOCAB_SIZE, QWEN3_14B_HIDDEN_SIZE, "LM head")? as u64,
            bf16_bytes(checked_elements(
                QWEN3_14B_VOCAB_SIZE,
                QWEN3_14B_HIDDEN_SIZE,
                "LM head",
            )?)? as u64,
        )?;
        validate_health_contract(
            &self.final_hidden_health,
            QWEN3_14B_HIDDEN_SIZE,
            "final hidden",
        )?;
        validate_health_contract(&self.logits_health, QWEN3_14B_VOCAB_SIZE, "logits")?;
        if self.stack_kernel_guard_checks != QWEN3_14B_SQ8_REQUIRED_HIP_KERNEL_ENV.len()
            || self.head_kernel_guard_checks
                != QWEN3_14B_SQ8_MODEL_HEAD_REQUIRED_HIP_KERNEL_ENV.len()
        {
            return Err(format!(
                "Qwen3-14B SQ8 model-head guard count mismatch: expected_stack={} actual_stack={} expected_head={} actual_head={}",
                QWEN3_14B_SQ8_REQUIRED_HIP_KERNEL_ENV.len(),
                self.stack_kernel_guard_checks,
                QWEN3_14B_SQ8_MODEL_HEAD_REQUIRED_HIP_KERNEL_ENV.len(),
                self.head_kernel_guard_checks
            ));
        }
        if self.d2d_copy_count != 1
            || self.rmsnorm_call_count != 1
            || self.bf16_matvec_call_count != 1
            || self.result_readback_count != 2
            || self.execution_synchronization_count != 1
        {
            return Err(format!(
                "Qwen3-14B SQ8 model-head operation count mismatch: d2d={} rmsnorm={} bf16_matvec={} readback={} sync={}",
                self.d2d_copy_count,
                self.rmsnorm_call_count,
                self.bf16_matvec_call_count,
                self.result_readback_count,
                self.execution_synchronization_count
            ));
        }
        if self.fallback_used || self.host_staging_used {
            return Err(
                "Qwen3-14B SQ8 model-head execution requires no fallback or host staging".into(),
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sq8ModelHeadResult {
    pub final_hidden: Vec<f32>,
    pub logits: Vec<f32>,
    pub report: Sq8ModelHeadExecutionReport,
}

impl Sq8ModelHeadResult {
    pub fn validate_contract(&self) -> Result<(), String> {
        self.report.validate_contract()?;
        let final_hidden_health = validate_sq8_model_head_tensor_health(
            &self.final_hidden,
            QWEN3_14B_HIDDEN_SIZE,
            "final hidden",
        )?;
        let logits_health =
            validate_sq8_model_head_tensor_health(&self.logits, QWEN3_14B_VOCAB_SIZE, "logits")?;
        if final_hidden_health != self.report.final_hidden_health {
            return Err("Qwen3-14B SQ8 model-head final-hidden health/hash mismatch".into());
        }
        if logits_health != self.report.logits_health {
            return Err("Qwen3-14B SQ8 model-head logits health/hash mismatch".into());
        }
        Ok(())
    }
}

/// Owns fixed Qwen3-14B final normalization and BF16 language-model-head storage.
///
/// Model loading may use bounded host staging. Each public execution consumes a completed,
/// promotion-valid stack output on the same R9700, executes without kernel fallback/host staging,
/// and synchronizes once before returning host-owned results. A partial execution failure
/// permanently poisons this instance.
#[derive(Debug)]
pub struct Qwen3Sq8ModelHeadRuntime {
    device: Sq8ModelHeadDeviceIdentity,
    final_norm_identity: Sq8ModelHeadPayloadIdentity,
    lm_head_identity: Sq8ModelHeadPayloadIdentity,
    final_norm_f32: RuntimeBuffer,
    lm_head_bf16: RuntimeBuffer,
    selected_hidden: RuntimeBuffer,
    final_hidden: RuntimeBuffer,
    logits: RuntimeBuffer,
    state: Sq8ModelHeadRuntimeState,
}

impl Qwen3Sq8ModelHeadRuntime {
    pub fn load(
        context: &mut RuntimeContext,
        stream: &mut RuntimeStream,
        package_path: impl AsRef<Path>,
        chunk_bytes: usize,
    ) -> Result<Self, String> {
        let package_path = package_path.as_ref();
        if chunk_bytes == 0 {
            return Err("Qwen3-14B SQ8 model-head chunk_bytes must be greater than zero".into());
        }
        let device = Sq8ModelHeadDeviceIdentity::from_runtime(context.device_info()?);
        device.validate_r9700()?;

        let load_result = (|| {
            let norm_verification = verify_named_passthrough_payload(
                package_path,
                QWEN3_14B_FINAL_NORM_TENSOR,
                BF16_DTYPE,
                &QWEN3_14B_FINAL_NORM_SHAPE,
                chunk_bytes,
            )?;
            let final_norm =
                read_named_passthrough_f32(package_path, QWEN3_14B_FINAL_NORM_TENSOR, chunk_bytes)?;
            if final_norm.dtype != BF16_DTYPE
                || final_norm.shape != QWEN3_14B_FINAL_NORM_SHAPE
                || final_norm.values.len() != QWEN3_14B_HIDDEN_SIZE
            {
                return Err(format!(
                    "Qwen3-14B SQ8 final norm changed after verification: dtype={} shape={:?} elements={}",
                    final_norm.dtype,
                    final_norm.shape,
                    final_norm.values.len()
                ));
            }
            validate_sq8_model_head_tensor_health(
                &final_norm.values,
                QWEN3_14B_HIDDEN_SIZE,
                "final norm weight",
            )?;
            let decoded_norm_sha256 = bf16_values_sha256(&final_norm.values);
            if decoded_norm_sha256 != norm_verification.payload_sha256 {
                return Err(format!(
                    "Qwen3-14B SQ8 final norm checksum changed after verification: verified={} decoded={decoded_norm_sha256}",
                    norm_verification.payload_sha256
                ));
            }
            let final_norm_bytes = encode_f32_to_bytes(&final_norm.values);
            let mut final_norm_f32 = context
                .alloc_buffer(final_norm_bytes.len())
                .map_err(|err| format!("failed to allocate Qwen3-14B final norm: {err}"))?;
            final_norm_f32
                .copy_from_host(0, &final_norm_bytes, Some(&mut *stream))
                .map_err(|err| format!("failed to upload Qwen3-14B final norm: {err}"))?;
            stream.synchronize().map_err(|err| {
                format!("failed to synchronize Qwen3-14B final norm upload: {err}")
            })?;
            drop(final_norm_bytes);
            drop(final_norm);

            let lm_head = load_named_passthrough_bf16_resident(
                context,
                stream,
                package_path,
                QWEN3_14B_LM_HEAD_TENSOR,
                &QWEN3_14B_LM_HEAD_SHAPE,
                chunk_bytes,
            )?;
            let lm_head_identity = resident_identity(&lm_head);
            let final_norm_identity = Sq8ModelHeadPayloadIdentity::from(norm_verification);
            final_norm_identity.validate(
                QWEN3_14B_FINAL_NORM_TENSOR,
                &QWEN3_14B_FINAL_NORM_SHAPE,
                QWEN3_14B_HIDDEN_SIZE as u64,
                bf16_bytes(QWEN3_14B_HIDDEN_SIZE)? as u64,
            )?;
            lm_head_identity.validate(
                QWEN3_14B_LM_HEAD_TENSOR,
                &QWEN3_14B_LM_HEAD_SHAPE,
                checked_elements(QWEN3_14B_VOCAB_SIZE, QWEN3_14B_HIDDEN_SIZE, "LM head")? as u64,
                bf16_bytes(checked_elements(
                    QWEN3_14B_VOCAB_SIZE,
                    QWEN3_14B_HIDDEN_SIZE,
                    "LM head",
                )?)? as u64,
            )?;

            let selected_hidden = alloc_f32(context, QWEN3_14B_HIDDEN_SIZE, "selected hidden")?;
            let final_hidden = alloc_f32(context, QWEN3_14B_HIDDEN_SIZE, "final hidden")?;
            let logits = alloc_f32(context, QWEN3_14B_VOCAB_SIZE, "logits")?;

            Ok(Self {
                device,
                final_norm_identity,
                lm_head_identity,
                final_norm_f32,
                lm_head_bf16: lm_head.buffer,
                selected_hidden,
                final_hidden,
                logits,
                state: Sq8ModelHeadRuntimeState::Ready,
            })
        })();

        match load_result {
            Ok(runtime) => Ok(runtime),
            Err(operation_error) => Err(load_error_after_stream_recovery(stream, operation_error)),
        }
    }

    pub fn status(&self) -> Sq8ModelHeadRuntimeStatus {
        self.state.status()
    }

    pub fn device_identity(&self) -> &Sq8ModelHeadDeviceIdentity {
        &self.device
    }

    pub fn poison_reason(&self) -> Option<&str> {
        self.state.poison_reason()
    }

    pub fn final_norm_identity(&self) -> &Sq8ModelHeadPayloadIdentity {
        &self.final_norm_identity
    }

    pub fn lm_head_identity(&self) -> &Sq8ModelHeadPayloadIdentity {
        &self.lm_head_identity
    }

    /// Applies final normalization and the BF16 language-model head to the stack's last row.
    ///
    /// The stack must already have completed and synchronized. This method then enqueues one D2D
    /// row copy, one RMSNorm, one BF16 matvec, and two result readbacks before one synchronization.
    pub fn run_synchronized(
        &mut self,
        stack: &Qwen3Sq8StackRuntime,
        stream: &mut RuntimeStream,
    ) -> Result<Sq8ModelHeadResult, String> {
        self.validate_execution_preconditions(stack)?;
        let sequence_len = stack.config().sequence_len;
        let selected_row = sequence_len - 1;
        let selected_row_offset = selected_row_offset_bytes(sequence_len)?;
        let hidden_bytes = f32_bytes(QWEN3_14B_HIDDEN_SIZE)?;
        let logits_bytes = f32_bytes(QWEN3_14B_VOCAB_SIZE)?;
        let (stack_hidden, stack_execution) = stack.resident_optimized_output()?;
        let stack_execution = stack_execution.clone();

        let mut final_hidden_host = zeroed_host_bytes(hidden_bytes, "final hidden readback")?;
        let mut logits_host = zeroed_host_bytes(logits_bytes, "logits readback")?;

        let operation_result = (|| {
            self.selected_hidden.copy_from_buffer(
                0,
                stack_hidden,
                selected_row_offset,
                hidden_bytes,
                Some(&mut *stream),
            )?;
            segmented_rmsnorm_f32(
                &self.selected_hidden,
                &self.final_norm_f32,
                1,
                QWEN3_14B_HIDDEN_SIZE,
                QWEN3_14B_RMS_NORM_EPSILON,
                &mut self.final_hidden,
                Some(&mut *stream),
            )?;
            matvec_bf16_f32(
                &self.lm_head_bf16,
                &self.final_hidden,
                QWEN3_14B_VOCAB_SIZE,
                QWEN3_14B_HIDDEN_SIZE,
                &mut self.logits,
                Some(&mut *stream),
            )?;
            self.final_hidden
                .copy_to_host(0, &mut final_hidden_host, Some(&mut *stream))?;
            self.logits
                .copy_to_host(0, &mut logits_host, Some(&mut *stream))?;
            Ok::<(), String>(())
        })();
        if let Err(operation_error) = operation_result {
            return Err(self.poison_after_stream_recovery(stream, operation_error));
        }
        if let Err(sync_error) = stream.synchronize() {
            return Err(self.poison_with_error(format!(
                "failed to synchronize Qwen3-14B SQ8 model-head execution: {sync_error}"
            )));
        }

        let final_hidden = decode_f32_le_values(&final_hidden_host);
        let logits = decode_f32_le_values(&logits_host);
        let result = Sq8ModelHeadResult {
            report: Sq8ModelHeadExecutionReport {
                device: self.device.clone(),
                stack_execution,
                stack_artifact_content_sha256: stack.artifact_content_sha256().to_string(),
                sequence_len,
                selected_row,
                final_norm: self.final_norm_identity.clone(),
                lm_head: self.lm_head_identity.clone(),
                final_hidden_health: match validate_sq8_model_head_tensor_health(
                    &final_hidden,
                    QWEN3_14B_HIDDEN_SIZE,
                    "final hidden",
                ) {
                    Ok(health) => health,
                    Err(err) => return Err(self.poison_with_error(err)),
                },
                logits_health: match validate_sq8_model_head_tensor_health(
                    &logits,
                    QWEN3_14B_VOCAB_SIZE,
                    "logits",
                ) {
                    Ok(health) => health,
                    Err(err) => return Err(self.poison_with_error(err)),
                },
                stack_kernel_guard_checks: QWEN3_14B_SQ8_REQUIRED_HIP_KERNEL_ENV.len(),
                head_kernel_guard_checks: QWEN3_14B_SQ8_MODEL_HEAD_REQUIRED_HIP_KERNEL_ENV.len(),
                d2d_copy_count: 1,
                rmsnorm_call_count: 1,
                bf16_matvec_call_count: 1,
                result_readback_count: 2,
                execution_synchronization_count: 1,
                fallback_used: false,
                host_staging_used: false,
            },
            final_hidden,
            logits,
        };
        if let Err(err) = result.validate_contract() {
            return Err(self.poison_with_error(format!(
                "Qwen3-14B SQ8 model-head result validation failed: {err}"
            )));
        }
        self.state = Sq8ModelHeadRuntimeState::OutputReady;
        Ok(result)
    }

    fn validate_runtime_contract(&self) -> Result<(), String> {
        self.state.ensure_usable()?;
        self.device.validate_r9700()?;
        self.final_norm_identity.validate(
            QWEN3_14B_FINAL_NORM_TENSOR,
            &QWEN3_14B_FINAL_NORM_SHAPE,
            QWEN3_14B_HIDDEN_SIZE as u64,
            bf16_bytes(QWEN3_14B_HIDDEN_SIZE)? as u64,
        )?;
        self.lm_head_identity.validate(
            QWEN3_14B_LM_HEAD_TENSOR,
            &QWEN3_14B_LM_HEAD_SHAPE,
            checked_elements(QWEN3_14B_VOCAB_SIZE, QWEN3_14B_HIDDEN_SIZE, "LM head")? as u64,
            bf16_bytes(checked_elements(
                QWEN3_14B_VOCAB_SIZE,
                QWEN3_14B_HIDDEN_SIZE,
                "LM head",
            )?)? as u64,
        )?;
        validate_buffer_size(
            &self.final_norm_f32,
            f32_bytes(QWEN3_14B_HIDDEN_SIZE)?,
            "final norm",
        )?;
        validate_buffer_size(
            &self.lm_head_bf16,
            bf16_bytes(checked_elements(
                QWEN3_14B_VOCAB_SIZE,
                QWEN3_14B_HIDDEN_SIZE,
                "LM head",
            )?)?,
            "LM head",
        )?;
        validate_buffer_size(
            &self.selected_hidden,
            f32_bytes(QWEN3_14B_HIDDEN_SIZE)?,
            "selected hidden",
        )?;
        validate_buffer_size(
            &self.final_hidden,
            f32_bytes(QWEN3_14B_HIDDEN_SIZE)?,
            "final hidden",
        )?;
        validate_buffer_size(&self.logits, f32_bytes(QWEN3_14B_VOCAB_SIZE)?, "logits")
    }

    fn validate_execution_preconditions(&self, stack: &Qwen3Sq8StackRuntime) -> Result<(), String> {
        self.validate_runtime_contract()?;
        let config = stack.config();
        config.validate()?;
        if config.position_offset != 0 {
            return Err("Qwen3-14B SQ8 model head requires stack position_offset=0".into());
        }
        let (stack_hidden, stack_execution) = stack.resident_optimized_output()?;
        if stack_execution.artifact_content_sha256 != stack.artifact_content_sha256()
            || stack_execution.sequence_len != config.sequence_len
        {
            return Err(
                "Qwen3-14B SQ8 model head received an inconsistent optimized stack output".into(),
            );
        }
        validate_buffer_size(
            stack_hidden,
            f32_bytes(checked_elements(
                config.sequence_len,
                QWEN3_14B_HIDDEN_SIZE,
                "stack hidden",
            )?)?,
            "stack hidden",
        )?;
        validate_hip_only_guards()
    }

    fn poison_after_stream_recovery(
        &mut self,
        stream: &mut RuntimeStream,
        operation_error: String,
    ) -> String {
        let error = match stream.synchronize() {
            Ok(()) => operation_error,
            Err(sync_error) => format!(
                "{operation_error}; subsequent Qwen3-14B SQ8 model-head stream recovery failed: {sync_error}"
            ),
        };
        self.poison_with_error(error)
    }

    fn poison_with_error(&mut self, error: String) -> String {
        self.state.poison(error.clone());
        error
    }
}

pub fn validate_sq8_model_head_tensor_health(
    values: &[f32],
    expected_elements: usize,
    label: &str,
) -> Result<Sq8ModelHeadTensorHealth, String> {
    if values.len() != expected_elements {
        return Err(format!(
            "Qwen3-14B SQ8 model-head {label} element mismatch: expected={expected_elements} actual={}",
            values.len()
        ));
    }
    let first = values
        .first()
        .copied()
        .ok_or_else(|| format!("Qwen3-14B SQ8 model-head {label} must not be empty"))?;
    if !first.is_finite() {
        return Err(format!(
            "Qwen3-14B SQ8 model-head {label} contains non-finite value {first} at index 0"
        ));
    }
    let mut minimum = first;
    let mut maximum = first;
    for (index, value) in values.iter().copied().enumerate().skip(1) {
        if !value.is_finite() {
            return Err(format!(
                "Qwen3-14B SQ8 model-head {label} contains non-finite value {value} at index {index}"
            ));
        }
        minimum = minimum.min(value);
        maximum = maximum.max(value);
    }
    Ok(Sq8ModelHeadTensorHealth {
        elements: values.len(),
        all_finite: true,
        minimum,
        maximum,
        f32_le_sha256: sq8_f32_le_sha256(values)
            .map_err(|err| format!("failed to hash Qwen3-14B SQ8 model-head {label}: {err}"))?,
    })
}

fn validate_health_contract(
    health: &Sq8ModelHeadTensorHealth,
    expected_elements: usize,
    label: &str,
) -> Result<(), String> {
    if health.elements != expected_elements || !health.all_finite {
        return Err(format!(
            "Qwen3-14B SQ8 model-head {label} health mismatch: expected_elements={expected_elements} actual_elements={} all_finite={}",
            health.elements, health.all_finite
        ));
    }
    if !health.minimum.is_finite() || !health.maximum.is_finite() || health.minimum > health.maximum
    {
        return Err(format!(
            "Qwen3-14B SQ8 model-head {label} has invalid finite range [{}, {}]",
            health.minimum, health.maximum
        ));
    }
    validate_sha256(&health.f32_le_sha256)
        .map_err(|err| format!("Qwen3-14B SQ8 model-head {label} hash is invalid: {err}"))
}

fn resident_identity(value: &PassthroughBf16ResidentData) -> Sq8ModelHeadPayloadIdentity {
    Sq8ModelHeadPayloadIdentity {
        tensor_name: value.tensor_name.clone(),
        dtype: BF16_DTYPE.to_string(),
        shape: value.shape.clone(),
        elements: value.elements,
        payload_bytes: value.payload_bytes,
        payload_sha256: value.payload_sha256.clone(),
        processed_chunks: value.upload_chunks,
    }
}

fn validate_hip_only_guards() -> Result<(), String> {
    for name in QWEN3_14B_SQ8_REQUIRED_HIP_KERNEL_ENV
        .into_iter()
        .chain(QWEN3_14B_SQ8_MODEL_HEAD_REQUIRED_HIP_KERNEL_ENV)
    {
        if std::env::var_os(name).is_none() {
            return Err(format!(
                "Qwen3-14B SQ8 model-head execution requires {name}=1 to forbid HIP host-staging fallback"
            ));
        }
    }
    Ok(())
}

fn validate_buffer_size(
    buffer: &RuntimeBuffer,
    expected_bytes: usize,
    label: &str,
) -> Result<(), String> {
    let actual_bytes = buffer
        .size()
        .map_err(|err| format!("failed to inspect Qwen3-14B SQ8 model-head {label}: {err}"))?;
    if actual_bytes != expected_bytes {
        return Err(format!(
            "Qwen3-14B SQ8 model-head {label} buffer size mismatch: expected={expected_bytes} actual={actual_bytes}"
        ));
    }
    Ok(())
}

fn alloc_f32(
    context: &mut RuntimeContext,
    elements: usize,
    label: &str,
) -> Result<RuntimeBuffer, String> {
    context
        .alloc_buffer(f32_bytes(elements)?)
        .map_err(|err| format!("failed to allocate Qwen3-14B SQ8 model-head {label}: {err}"))
}

fn zeroed_host_bytes(bytes: usize, label: &str) -> Result<Vec<u8>, String> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(bytes)
        .map_err(|err| format!("failed to allocate Qwen3-14B SQ8 {label}: {err}"))?;
    values.resize(bytes, 0);
    Ok(values)
}

fn checked_elements(rows: usize, cols: usize, label: &str) -> Result<usize, String> {
    rows.checked_mul(cols)
        .ok_or_else(|| format!("Qwen3-14B SQ8 model-head {label} element count overflows"))
}

fn selected_row_offset_bytes(sequence_len: usize) -> Result<usize, String> {
    if !matches!(sequence_len, 1 | 2 | 4 | 8 | 16 | 32 | 128) {
        return Err(format!(
            "Qwen3-14B SQ8 model-head selected row has unmeasured sequence length {sequence_len}"
        ));
    }
    f32_bytes(checked_elements(
        sequence_len - 1,
        QWEN3_14B_HIDDEN_SIZE,
        "selected hidden row offset",
    )?)
}

fn f32_bytes(elements: usize) -> Result<usize, String> {
    elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "Qwen3-14B SQ8 model-head F32 byte size overflows".into())
}

fn bf16_bytes(elements: usize) -> Result<usize, String> {
    elements
        .checked_mul(std::mem::size_of::<u16>())
        .ok_or_else(|| "Qwen3-14B SQ8 model-head BF16 byte size overflows".into())
}

fn validate_sha256(value: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        return Err(format!("expected a lowercase SHA-256 digest, got {value}"));
    }
    Ok(())
}

fn bf16_values_sha256(values: &[f32]) -> String {
    let mut digest = Sha256::new();
    for value in values {
        digest.update(((value.to_bits() >> 16) as u16).to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn load_error_after_stream_recovery(stream: &mut RuntimeStream, operation_error: String) -> String {
    match stream.synchronize() {
        Ok(()) => operation_error,
        Err(sync_error) => format!(
            "{operation_error}; subsequent Qwen3-14B SQ8 model-head load recovery failed: {sync_error}"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sq8_layer_runtime::{
        Sq8LayerExecutionProfile, Sq8LayerExecutionReport, Sq8LayerProjectionExecution,
    };
    use crate::sq8_stack_runtime::{Sq8StackExecutionMode, Sq8StackInputOrigin};
    use ullm_runtime_sys::Sq8CkImplementation;

    fn r9700_identity() -> Sq8ModelHeadDeviceIdentity {
        Sq8ModelHeadDeviceIdentity {
            device_id: 0,
            backend: "hip".to_string(),
            name: "AMD Radeon Graphics".to_string(),
            gcn_arch_name: "gfx1201".to_string(),
            compute_major: 12,
            compute_minor: 0,
            total_global_mem: 34_208_743_424,
        }
    }

    fn optimized_stack_execution() -> Sq8StackExecutionReport {
        let hidden =
            Sq8LayerProjectionExecution::Ck(Sq8CkImplementation::MemV1DefaultTile16x128x128);
        let gate_up =
            Sq8LayerProjectionExecution::Ck(Sq8CkImplementation::MemV1KPaddingTile16x128x256);
        let down = Sq8LayerProjectionExecution::Ck(Sq8CkImplementation::MemV1DefaultTile16x128x256);
        let layer = Sq8LayerExecutionReport {
            profile: Sq8LayerExecutionProfile::Rdna4W8a8BlockCk,
            q: hidden,
            k: hidden,
            v: hidden,
            o: hidden,
            gate: gate_up,
            up: gate_up,
            down,
            activation_quantizations: 4,
            projection_calls: 7,
            fallback_used: false,
        };
        Sq8StackExecutionReport {
            profile: Sq8LayerExecutionProfile::Rdna4W8a8BlockCk,
            mode: Sq8StackExecutionMode::SynchronizedResident,
            input_origin: Sq8StackInputOrigin::PreviouslyUploadedResident,
            sequence_len: 8,
            position_offset: 0,
            artifact_content_sha256: "0".repeat(64),
            layer_reports: std::array::from_fn(|_| layer.clone()),
            activation_quantizations: 160,
            projection_calls: 280,
            d2d_copy_count: 40,
            input_upload_synchronization_count: 0,
            execution_synchronization_count: 1,
            host_readback_count: 0,
            fallback_used: false,
            host_staging_used: false,
        }
    }

    fn payload_identity(name: &str, shape: Vec<u64>, elements: u64) -> Sq8ModelHeadPayloadIdentity {
        Sq8ModelHeadPayloadIdentity {
            tensor_name: name.to_string(),
            dtype: BF16_DTYPE.to_string(),
            shape,
            elements,
            payload_bytes: elements * 2,
            payload_sha256: "1".repeat(64),
            processed_chunks: 1,
        }
    }

    fn health(elements: usize) -> Sq8ModelHeadTensorHealth {
        Sq8ModelHeadTensorHealth {
            elements,
            all_finite: true,
            minimum: -1.0,
            maximum: 1.0,
            f32_le_sha256: "2".repeat(64),
        }
    }

    fn valid_report() -> Sq8ModelHeadExecutionReport {
        let lm_elements =
            checked_elements(QWEN3_14B_VOCAB_SIZE, QWEN3_14B_HIDDEN_SIZE, "test").unwrap();
        Sq8ModelHeadExecutionReport {
            device: r9700_identity(),
            stack_execution: optimized_stack_execution(),
            stack_artifact_content_sha256: "0".repeat(64),
            sequence_len: 8,
            selected_row: 7,
            final_norm: payload_identity(
                QWEN3_14B_FINAL_NORM_TENSOR,
                QWEN3_14B_FINAL_NORM_SHAPE.to_vec(),
                QWEN3_14B_HIDDEN_SIZE as u64,
            ),
            lm_head: payload_identity(
                QWEN3_14B_LM_HEAD_TENSOR,
                QWEN3_14B_LM_HEAD_SHAPE.to_vec(),
                lm_elements as u64,
            ),
            final_hidden_health: health(QWEN3_14B_HIDDEN_SIZE),
            logits_health: health(QWEN3_14B_VOCAB_SIZE),
            stack_kernel_guard_checks: QWEN3_14B_SQ8_REQUIRED_HIP_KERNEL_ENV.len(),
            head_kernel_guard_checks: QWEN3_14B_SQ8_MODEL_HEAD_REQUIRED_HIP_KERNEL_ENV.len(),
            d2d_copy_count: 1,
            rmsnorm_call_count: 1,
            bf16_matvec_call_count: 1,
            result_readback_count: 2,
            execution_synchronization_count: 1,
            fallback_used: false,
            host_staging_used: false,
        }
    }

    #[test]
    fn tensor_health_validates_range_and_hash() {
        let values = [-2.0_f32, 0.0, 3.5];
        let health = validate_sq8_model_head_tensor_health(&values, 3, "test").unwrap();
        assert!(health.all_finite);
        assert_eq!(health.elements, 3);
        assert_eq!(health.minimum, -2.0);
        assert_eq!(health.maximum, 3.5);
        assert_eq!(health.f32_le_sha256.len(), 64);
    }

    #[test]
    fn tensor_health_rejects_wrong_length_and_nonfinite_values() {
        assert!(validate_sq8_model_head_tensor_health(&[1.0], 2, "test").is_err());
        assert!(validate_sq8_model_head_tensor_health(&[1.0, f32::NAN], 2, "test").is_err());
        assert!(validate_sq8_model_head_tensor_health(&[f32::INFINITY], 1, "test").is_err());
    }

    #[test]
    fn report_enforces_fixed_model_and_operation_contract() {
        let report = valid_report();
        report.validate_contract().unwrap();

        let mut runtime_api_identity = report.clone();
        runtime_api_identity.device.gcn_arch_name.clear();
        runtime_api_identity.validate_contract().unwrap();

        let mut bad = report.clone();
        bad.selected_row = 6;
        assert!(bad.validate_contract().is_err());

        let mut bad = report.clone();
        bad.bf16_matvec_call_count = 0;
        assert!(bad.validate_contract().is_err());

        let mut bad = report.clone();
        bad.host_staging_used = true;
        assert!(bad.validate_contract().is_err());

        let mut bad = report.clone();
        bad.device.gcn_arch_name = "gfx1030".to_string();
        assert!(bad.validate_contract().is_err());

        let mut bad = report.clone();
        bad.device.gcn_arch_name.clear();
        bad.device.name = "HIP device 0".to_string();
        assert!(bad.validate_contract().is_err());

        let mut bad = report.clone();
        bad.device.gcn_arch_name.clear();
        bad.device.total_global_mem = R9700_MEMORY_BYTES_MIN - 1;
        assert!(bad.validate_contract().is_err());

        let mut bad = report.clone();
        bad.stack_execution.mode = Sq8StackExecutionMode::LayerwiseAuditNonTimed;
        bad.stack_execution.execution_synchronization_count = 40;
        bad.stack_execution.host_readback_count = 40;
        bad.stack_execution.host_staging_used = true;
        assert!(bad.stack_execution.validate_contract().is_ok());
        assert!(bad.validate_contract().is_err());

        let mut bad = report;
        bad.lm_head.tensor_name = "model.lm_head.weight".into();
        assert!(bad.validate_contract().is_err());
    }

    #[test]
    fn sha256_contract_is_lowercase_and_exact_length() {
        assert!(validate_sha256(&"0".repeat(64)).is_ok());
        assert!(validate_sha256(&"f".repeat(64)).is_ok());
        assert!(validate_sha256(&"F".repeat(64)).is_err());
        assert!(validate_sha256(&"0".repeat(63)).is_err());
    }

    #[test]
    fn decoded_bf16_hash_uses_original_little_endian_payload_bits() {
        let values = [1.0_f32, -2.0_f32, 0.5_f32];
        let mut digest = Sha256::new();
        digest.update(0x3f80_u16.to_le_bytes());
        digest.update(0xc000_u16.to_le_bytes());
        digest.update(0x3f00_u16.to_le_bytes());
        assert_eq!(
            bf16_values_sha256(&values),
            format!("{:x}", digest.finalize())
        );
    }

    #[test]
    fn selected_row_offset_uses_the_last_prompt_row() {
        for sequence_len in [1, 2, 4, 8, 16, 32, 128] {
            assert_eq!(
                selected_row_offset_bytes(sequence_len).unwrap(),
                (sequence_len - 1) * QWEN3_14B_HIDDEN_SIZE * std::mem::size_of::<f32>()
            );
        }
        assert!(selected_row_offset_bytes(0).is_err());
        assert!(selected_row_offset_bytes(3).is_err());
        assert_eq!(
            QWEN3_14B_LM_HEAD_SHAPE,
            [QWEN3_14B_VOCAB_SIZE as u64, QWEN3_14B_HIDDEN_SIZE as u64]
        );
    }

    #[test]
    fn model_head_poison_state_is_permanent() {
        let mut state = Sq8ModelHeadRuntimeState::Ready;
        state.poison("stream failure".to_string());
        assert_eq!(state.status(), Sq8ModelHeadRuntimeStatus::Poisoned);
        assert_eq!(state.poison_reason(), Some("stream failure"));
        assert!(state.ensure_usable().is_err());
        state.poison("replacement".to_string());
        assert_eq!(state.poison_reason(), Some("stream failure"));
    }
}
