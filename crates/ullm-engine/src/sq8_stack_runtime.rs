// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

use crate::decoder::PagedDecodeState;
use crate::host_bytes::{decode_f32_le_values, encode_f32_to_bytes};
use crate::sq_canonical::Sq8CanonicalArtifact;
use crate::sq8_layer_oracle::{
    QWEN3_14B_HEAD_DIM, QWEN3_14B_HIDDEN_SIZE, QWEN3_14B_INTERMEDIATE_SIZE, QWEN3_14B_KV_HEADS,
    QWEN3_14B_VALUE_DIM,
};
use crate::sq8_layer_runtime::{
    QWEN3_14B_SQ8_LAYER_ACTIVATION_QUANTIZATIONS, QWEN3_14B_SQ8_LAYER_PROJECTIONS,
    QWEN3_14B_SQ8_PAGED_REQUIRED_HIP_KERNEL_ENV, QWEN3_14B_SQ8_REQUIRED_HIP_KERNEL_ENV,
    Qwen3Sq8LayerConfig, Qwen3Sq8LayerNormValues, Qwen3Sq8LayerWeights, Qwen3Sq8LayerWorkspace,
    Sq8LayerExecutionProfile, Sq8LayerExecutionReport, Sq8LayerProjectionExecution,
    load_qwen3_14b_sq8_layer_weights, qwen3_sq8_layer_tensor_names, validate_norm_values,
};
use ullm_runtime_sys::{RuntimeBuffer, RuntimeContext, RuntimeStream, Sq8CkImplementation};

pub const QWEN3_14B_SQ8_STACK_LAYERS: usize = 40;
pub const QWEN3_14B_SQ8_STACK_PROJECTIONS: usize =
    QWEN3_14B_SQ8_STACK_LAYERS * QWEN3_14B_SQ8_LAYER_PROJECTIONS;
pub const QWEN3_14B_SQ8_STACK_ACTIVATION_QUANTIZATIONS: usize =
    QWEN3_14B_SQ8_STACK_LAYERS * QWEN3_14B_SQ8_LAYER_ACTIVATION_QUANTIZATIONS;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sq8PagedStackPhase {
    Prefill,
    Decode,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sq8PagedStackExecutionReport {
    pub phase: Sq8PagedStackPhase,
    pub position: usize,
    pub stack: Sq8StackExecutionReport,
    pub cache_lengths: [usize; QWEN3_14B_SQ8_STACK_LAYERS],
    pub kv_write_calls: usize,
    pub paged_attention_calls: usize,
    pub input_d2d_copy_count: usize,
}

impl Sq8PagedStackExecutionReport {
    pub fn validate_contract(&self) -> Result<(), String> {
        self.stack.validate_optimized_promotion()?;
        if self.stack.host_staging_used || self.stack.host_readback_count != 0 {
            return Err("SQ8 paged stack report rejects host staging/readback".into());
        }
        let expected_cache_len = match self.phase {
            Sq8PagedStackPhase::Prefill => {
                if self.position != 0 || self.stack.sequence_len <= 1 {
                    return Err(
                        "SQ8 paged prefill report requires position=0 and sequence_len>1".into(),
                    );
                }
                let expected_writes = self
                    .stack
                    .sequence_len
                    .checked_mul(QWEN3_14B_SQ8_STACK_LAYERS)
                    .ok_or_else(|| "SQ8 paged prefill KV write count overflows".to_string())?;
                if self.kv_write_calls != expected_writes
                    || self.paged_attention_calls != 0
                    || self.input_d2d_copy_count != 0
                {
                    return Err(format!(
                        "SQ8 paged prefill counters mismatch: kv_writes={} paged_attention={} input_d2d={}",
                        self.kv_write_calls, self.paged_attention_calls, self.input_d2d_copy_count
                    ));
                }
                self.stack.sequence_len
            }
            Sq8PagedStackPhase::Decode => {
                if self.stack.sequence_len != 1 {
                    return Err("SQ8 paged decode report requires M=1".into());
                }
                if self.kv_write_calls != QWEN3_14B_SQ8_STACK_LAYERS
                    || self.paged_attention_calls != QWEN3_14B_SQ8_STACK_LAYERS
                    || self.input_d2d_copy_count != 1
                {
                    return Err(format!(
                        "SQ8 paged decode counters mismatch: kv_writes={} paged_attention={} input_d2d={}",
                        self.kv_write_calls, self.paged_attention_calls, self.input_d2d_copy_count
                    ));
                }
                self.position
                    .checked_add(1)
                    .ok_or_else(|| "SQ8 paged decode cache length overflows".to_string())?
            }
        };
        if let Some((layer_index, actual)) = self
            .cache_lengths
            .iter()
            .copied()
            .enumerate()
            .find(|(_, actual)| *actual != expected_cache_len)
        {
            return Err(format!(
                "SQ8 paged stack layer {layer_index} cache length mismatch: expected={expected_cache_len} actual={actual}"
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sq8StackExecutionMode {
    SynchronizedResident,
    LayerwiseAuditNonTimed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sq8StackInputOrigin {
    PreviouslyUploadedResident,
    SynchronizedHostUploadBeforeExecution,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sq8StackRuntimeStatus {
    NeedsInput,
    InputReady,
    OutputReady,
    Poisoned,
}

#[derive(Debug)]
enum Sq8StackResidentState {
    NeedsInput,
    InputReady,
    OutputReady,
    Poisoned(String),
}

impl Sq8StackResidentState {
    fn status(&self) -> Sq8StackRuntimeStatus {
        match self {
            Self::NeedsInput => Sq8StackRuntimeStatus::NeedsInput,
            Self::InputReady => Sq8StackRuntimeStatus::InputReady,
            Self::OutputReady => Sq8StackRuntimeStatus::OutputReady,
            Self::Poisoned(_) => Sq8StackRuntimeStatus::Poisoned,
        }
    }

    fn ensure_usable(&self) -> Result<(), String> {
        match self {
            Self::Poisoned(reason) => Err(format!(
                "Qwen3-14B SQ8 stack is permanently poisoned: {reason}"
            )),
            Self::NeedsInput | Self::InputReady | Self::OutputReady => Ok(()),
        }
    }

    fn require_input_ready(&self) -> Result<(), String> {
        self.ensure_usable()?;
        if matches!(self, Self::InputReady) {
            Ok(())
        } else {
            Err(
                "Qwen3-14B SQ8 stack requires a fresh synchronized resident input; completed output cannot be rerun as layer-0 input"
                    .into(),
            )
        }
    }

    fn require_output_ready(&self) -> Result<(), String> {
        self.ensure_usable()?;
        if matches!(self, Self::OutputReady) {
            Ok(())
        } else {
            Err("Qwen3-14B SQ8 stack has no completed resident output".into())
        }
    }

    fn mark_needs_input(&mut self) -> Result<(), String> {
        self.ensure_usable()?;
        *self = Self::NeedsInput;
        Ok(())
    }

    fn mark_input_ready(&mut self) -> Result<(), String> {
        self.ensure_usable()?;
        *self = Self::InputReady;
        Ok(())
    }

    fn mark_output_ready(&mut self) -> Result<(), String> {
        self.ensure_usable()?;
        *self = Self::OutputReady;
        Ok(())
    }

    fn poison(&mut self, reason: String) {
        if !matches!(self, Self::Poisoned(_)) {
            *self = Self::Poisoned(reason);
        }
    }

    fn poison_reason(&self) -> Option<&str> {
        match self {
            Self::Poisoned(reason) => Some(reason),
            Self::NeedsInput | Self::InputReady | Self::OutputReady => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sq8StackExecutionReport {
    pub profile: Sq8LayerExecutionProfile,
    pub mode: Sq8StackExecutionMode,
    pub input_origin: Sq8StackInputOrigin,
    pub sequence_len: usize,
    pub position_offset: usize,
    pub artifact_content_sha256: String,
    pub layer_reports: [Sq8LayerExecutionReport; QWEN3_14B_SQ8_STACK_LAYERS],
    pub activation_quantizations: usize,
    pub projection_calls: usize,
    pub d2d_copy_count: usize,
    pub input_upload_synchronization_count: usize,
    pub execution_synchronization_count: usize,
    pub host_readback_count: usize,
    pub fallback_used: bool,
    pub host_staging_used: bool,
}

impl Sq8StackExecutionReport {
    pub fn all_ck(&self) -> bool {
        self.layer_reports
            .iter()
            .all(Sq8LayerExecutionReport::all_ck)
    }

    pub fn all_reference_hip(&self) -> bool {
        self.layer_reports
            .iter()
            .all(Sq8LayerExecutionReport::all_reference_hip)
    }

    pub fn validate_contract(&self) -> Result<(), String> {
        if !matches!(self.sequence_len, 1 | 2 | 4 | 8 | 16 | 32 | 128) {
            return Err(format!(
                "Qwen3-14B SQ8 stack report has unmeasured M {}",
                self.sequence_len
            ));
        }
        if self.position_offset != 0 {
            return Err("Qwen3-14B SQ8 stack report requires position_offset=0".into());
        }
        validate_sha256(&self.artifact_content_sha256)?;
        if self.projection_calls != QWEN3_14B_SQ8_STACK_PROJECTIONS {
            return Err(format!(
                "Qwen3-14B SQ8 stack report projection count mismatch: expected={} actual={}",
                QWEN3_14B_SQ8_STACK_PROJECTIONS, self.projection_calls
            ));
        }
        if self.d2d_copy_count != QWEN3_14B_SQ8_STACK_LAYERS {
            return Err(format!(
                "Qwen3-14B SQ8 stack report D2D count mismatch: expected={} actual={}",
                QWEN3_14B_SQ8_STACK_LAYERS, self.d2d_copy_count
            ));
        }
        match self.input_origin {
            Sq8StackInputOrigin::PreviouslyUploadedResident => {
                if self.input_upload_synchronization_count != 0 {
                    return Err(
                        "resident SQ8 stack input report must not count a host upload synchronization"
                            .into(),
                    );
                }
            }
            Sq8StackInputOrigin::SynchronizedHostUploadBeforeExecution => {
                if self.input_upload_synchronization_count != 1 {
                    return Err(
                        "host SQ8 stack input report must count exactly one pre-execution synchronization"
                            .into(),
                    );
                }
            }
        }
        match self.mode {
            Sq8StackExecutionMode::SynchronizedResident => {
                if self.execution_synchronization_count != 1
                    || self.host_readback_count != 0
                    || self.host_staging_used
                {
                    return Err(
                        "resident SQ8 stack execution must have one final synchronization and no host staging/readback"
                        .into(),
                    );
                }
            }
            Sq8StackExecutionMode::LayerwiseAuditNonTimed => {
                if self.execution_synchronization_count != QWEN3_14B_SQ8_STACK_LAYERS
                    || self.host_readback_count != QWEN3_14B_SQ8_STACK_LAYERS
                    || !self.host_staging_used
                {
                    return Err(
                        "layerwise SQ8 stack audit must have 40 synchronization/readback operations and host staging"
                        .into(),
                    );
                }
            }
        }
        if self.layer_reports.iter().any(|report| {
            report.profile != self.profile
                || report.projection_calls != QWEN3_14B_SQ8_LAYER_PROJECTIONS
        }) {
            return Err("Qwen3-14B SQ8 stack contains an inconsistent layer report".into());
        }
        if self.profile == Sq8LayerExecutionProfile::Rdna4W8a8BlockCk {
            for (layer_index, report) in self.layer_reports.iter().enumerate() {
                validate_measured_ck_dispatch(self.sequence_len, layer_index, report)?;
            }
        }
        let expected_layer_quantizations = match self.profile {
            Sq8LayerExecutionProfile::Rdna4W8a8BlockCk => {
                QWEN3_14B_SQ8_LAYER_ACTIVATION_QUANTIZATIONS
            }
            Sq8LayerExecutionProfile::ReferenceW8a16Block2d => 0,
        };
        if let Some((layer_index, report)) = self
            .layer_reports
            .iter()
            .enumerate()
            .find(|(_, report)| report.activation_quantizations != expected_layer_quantizations)
        {
            return Err(format!(
                "Qwen3-14B SQ8 stack layer {layer_index} activation quantization count mismatch: expected={expected_layer_quantizations} actual={}",
                report.activation_quantizations
            ));
        }
        let activation_quantizations =
            self.layer_reports
                .iter()
                .try_fold(0_usize, |total, report| {
                    total
                        .checked_add(report.activation_quantizations)
                        .ok_or_else(|| {
                            "Qwen3-14B SQ8 stack activation quantization count overflows"
                                .to_string()
                        })
                })?;
        if activation_quantizations != self.activation_quantizations {
            return Err(format!(
                "Qwen3-14B SQ8 stack activation quantization count mismatch: layers={activation_quantizations} report={}",
                self.activation_quantizations
            ));
        }
        let layer_fallback = self.layer_reports.iter().any(|report| report.fallback_used);
        if self.fallback_used != layer_fallback {
            return Err("Qwen3-14B SQ8 stack fallback report is inconsistent".into());
        }
        match self.profile {
            Sq8LayerExecutionProfile::Rdna4W8a8BlockCk => {
                if !self.all_ck()
                    || self.activation_quantizations != QWEN3_14B_SQ8_STACK_ACTIVATION_QUANTIZATIONS
                    || self.fallback_used
                {
                    return Err(
                        "optimized SQ8 stack report requires 280 CK projections, 160 activation quantizations, and no fallback"
                            .into(),
                    );
                }
            }
            Sq8LayerExecutionProfile::ReferenceW8a16Block2d => {
                if !self.all_reference_hip()
                    || self.activation_quantizations != 0
                    || self.fallback_used
                {
                    return Err(
                        "reference SQ8 stack report requires all HIP projections and no fallback"
                            .into(),
                    );
                }
            }
        }
        Ok(())
    }

    pub fn validate_optimized_promotion(&self) -> Result<(), String> {
        self.validate_contract()?;
        if self.profile != Sq8LayerExecutionProfile::Rdna4W8a8BlockCk {
            return Err("SQ8 stack optimized promotion requires the RDNA4 CK profile".into());
        }
        if self.mode != Sq8StackExecutionMode::SynchronizedResident {
            return Err("SQ8 stack optimized promotion rejects non-timed audit execution".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sq8StackLayerAudit {
    pub layer_index: usize,
    pub output: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sq8StackLayerwiseAudit {
    pub report: Sq8StackExecutionReport,
    pub layers: [Sq8StackLayerAudit; QWEN3_14B_SQ8_STACK_LAYERS],
}

/// Owns the complete Qwen3-14B SQ8 decoder stack and its shared activation storage.
///
/// Every public execution method takes `&mut self` and returns only after synchronizing its
/// stream. Safe Rust therefore cannot run this stack concurrently on a second stream.
#[derive(Debug)]
pub struct Qwen3Sq8StackRuntime {
    config: Qwen3Sq8LayerConfig,
    artifact_content_sha256: String,
    weights: Box<[Qwen3Sq8LayerWeights; QWEN3_14B_SQ8_STACK_LAYERS]>,
    workspace: Qwen3Sq8LayerWorkspace,
    resident_hidden: RuntimeBuffer,
    state: Sq8StackResidentState,
    last_execution_report: Option<Sq8StackExecutionReport>,
}

#[derive(Debug)]
pub(crate) struct Qwen3Sq8PagedDecodeRuntime {
    config: Qwen3Sq8LayerConfig,
    workspace: Qwen3Sq8LayerWorkspace,
    resident_hidden: RuntimeBuffer,
    output_ready: bool,
    poison_reason: Option<String>,
    last_execution_report: Option<Sq8PagedStackExecutionReport>,
}

impl Qwen3Sq8PagedDecodeRuntime {
    pub(crate) fn allocate(context: &mut RuntimeContext) -> Result<Self, String> {
        let config = Qwen3Sq8LayerConfig::qwen3_14b(1, 0)?;
        let workspace = Qwen3Sq8LayerWorkspace::allocate(context, config)
            .map_err(|err| format!("failed to allocate Qwen3-14B SQ8 decode workspace: {err}"))?;
        let resident_hidden = context.alloc_buffer(hidden_bytes(config)?).map_err(|err| {
            format!("failed to allocate Qwen3-14B SQ8 decode hidden state: {err}")
        })?;
        Ok(Self {
            config,
            workspace,
            resident_hidden,
            output_ready: false,
            poison_reason: None,
            last_execution_report: None,
        })
    }

    pub(crate) fn resident_hidden_buffer(&self) -> Result<&RuntimeBuffer, String> {
        self.ensure_usable()?;
        if !self.output_ready {
            return Err("Qwen3-14B SQ8 decode has no completed resident output".into());
        }
        Ok(&self.resident_hidden)
    }

    pub(crate) fn last_execution_report(&self) -> Option<&Sq8PagedStackExecutionReport> {
        self.last_execution_report.as_ref()
    }

    fn ensure_usable(&self) -> Result<(), String> {
        match &self.poison_reason {
            Some(reason) => Err(format!(
                "Qwen3-14B SQ8 decode runtime is permanently poisoned: {reason}"
            )),
            None => Ok(()),
        }
    }

    fn poison(&mut self, reason: String) {
        if self.poison_reason.is_none() {
            self.poison_reason = Some(reason);
        }
        self.output_ready = false;
        self.last_execution_report = None;
    }
}

impl Qwen3Sq8StackRuntime {
    /// Loads the fixed 40-layer stack one layer at a time.
    ///
    /// `norms` is consumed in layer order. Each layer's host norms are dropped immediately after
    /// that layer has been uploaded, while all projection weights remain resident.
    pub fn load(
        context: &mut RuntimeContext,
        stream: &mut RuntimeStream,
        artifact: &Sq8CanonicalArtifact,
        sequence_len: usize,
        norms: Vec<Qwen3Sq8LayerNormValues>,
        upload_chunk_bytes: usize,
    ) -> Result<Self, String> {
        let config = Qwen3Sq8LayerConfig::qwen3_14b(sequence_len, 0)?;
        validate_norm_layer_count(norms.len())?;
        if upload_chunk_bytes == 0 {
            return Err("Qwen3-14B SQ8 stack upload chunk size must be greater than zero".into());
        }
        validate_stack_artifact(artifact)?;
        for (layer_index, layer_norms) in norms.iter().enumerate() {
            validate_norm_values(layer_norms).map_err(|err| {
                format!("Qwen3-14B SQ8 stack layer {layer_index} norm validation failed: {err}")
            })?;
        }

        let load_result = (|| {
            let mut loaded = Vec::with_capacity(QWEN3_14B_SQ8_STACK_LAYERS);
            for (layer_index, layer_norms) in norms.into_iter().enumerate() {
                let weights = load_qwen3_14b_sq8_layer_weights(
                    context,
                    stream,
                    artifact,
                    layer_index,
                    &layer_norms,
                    upload_chunk_bytes,
                )
                .map_err(|err| {
                    format!("failed to load Qwen3-14B SQ8 stack layer {layer_index}: {err}")
                })?;
                loaded.push(weights);
                drop(layer_norms);
            }
            boxed_layer_array(loaded)
        })();
        let weights = match load_result {
            Ok(weights) => weights,
            Err(err) => return Err(load_error_after_stream_recovery(stream, err)),
        };

        validate_loaded_layer_indices(weights.iter().map(|weights| weights.layer_index))?;
        let workspace = Qwen3Sq8LayerWorkspace::allocate(context, config)
            .map_err(|err| format!("failed to allocate Qwen3-14B SQ8 stack workspace: {err}"))?;
        let resident_hidden = context
            .alloc_buffer(hidden_bytes(config)?)
            .map_err(|err| format!("failed to allocate Qwen3-14B SQ8 stack hidden state: {err}"))?;

        Ok(Self {
            config,
            artifact_content_sha256: artifact.manifest().integrity.content_sha256.clone(),
            weights,
            workspace,
            resident_hidden,
            state: Sq8StackResidentState::NeedsInput,
            last_execution_report: None,
        })
    }

    pub fn config(&self) -> Qwen3Sq8LayerConfig {
        self.config
    }

    pub fn artifact_content_sha256(&self) -> &str {
        &self.artifact_content_sha256
    }

    pub fn layer_count(&self) -> usize {
        self.weights.len()
    }

    pub fn status(&self) -> Sq8StackRuntimeStatus {
        self.state.status()
    }

    pub fn poison_reason(&self) -> Option<&str> {
        self.state.poison_reason()
    }

    pub fn last_execution_report(&self) -> Option<&Sq8StackExecutionReport> {
        self.last_execution_report.as_ref()
    }

    /// Returns the completed hidden state for crate-internal same-stream composition.
    pub(crate) fn resident_hidden_buffer(&self) -> Result<&RuntimeBuffer, String> {
        self.state.require_output_ready()?;
        Ok(&self.resident_hidden)
    }

    pub(crate) fn resident_optimized_output(
        &self,
    ) -> Result<(&RuntimeBuffer, &Sq8StackExecutionReport), String> {
        self.state.require_output_ready()?;
        let report = self.last_execution_report.as_ref().ok_or_else(|| {
            "Qwen3-14B SQ8 stack completed output has no bound execution report".to_string()
        })?;
        report.validate_optimized_promotion()?;
        Ok((&self.resident_hidden, report))
    }

    /// Reads the completed output and synchronizes before returning host ownership.
    pub fn read_output_synchronized(
        &mut self,
        stream: &mut RuntimeStream,
    ) -> Result<Vec<f32>, String> {
        self.validate_runtime_contract()?;
        self.state.require_output_ready()?;
        let mut host = vec![0_u8; hidden_bytes(self.config)?];
        let copy_result = self
            .resident_hidden_buffer()?
            .copy_to_host(0, &mut host, Some(&mut *stream))
            .map_err(|err| format!("failed to read Qwen3-14B SQ8 stack output: {err}"));
        let sync_result = stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize Qwen3-14B SQ8 stack output read: {err}"));
        match (copy_result, sync_result) {
            (Ok(()), Ok(())) => {}
            (Err(operation_error), Ok(())) => {
                return Err(self.poison_with_error(operation_error));
            }
            (Ok(()), Err(sync_error)) => {
                return Err(self.poison_with_error(sync_error));
            }
            (Err(operation_error), Err(sync_error)) => {
                return Err(self.poison_with_error(format!(
                    "{operation_error}; subsequent stream synchronization also failed: {sync_error}"
                )));
            }
        }
        let output = decode_f32_le_values(&host);
        if let Some((index, value)) = output
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(self.poison_with_error(format!(
                "Qwen3-14B SQ8 stack output contains non-finite value {value} at index {index}"
            )));
        }
        Ok(output)
    }

    /// Uploads and synchronizes host input before any stack execution can begin.
    pub fn upload_host_input_synchronized(
        &mut self,
        input: &[f32],
        stream: &mut RuntimeStream,
    ) -> Result<(), String> {
        self.validate_runtime_contract()?;
        validate_host_input(input, self.config)?;
        let bytes = encode_f32_to_bytes(input);
        self.state.mark_needs_input()?;
        self.last_execution_report = None;
        let copy_result = self
            .resident_hidden
            .copy_from_host(0, &bytes, Some(&mut *stream))
            .map_err(|err| format!("failed to upload Qwen3-14B SQ8 stack input: {err}"));
        let sync_result = stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize Qwen3-14B SQ8 stack input: {err}"));
        match (copy_result, sync_result) {
            (Ok(()), Ok(())) => self.state.mark_input_ready(),
            (Err(operation_error), Ok(())) => {
                let error = self.poison_with_error(operation_error);
                Err(error)
            }
            (Ok(()), Err(sync_error)) => {
                let error = self.poison_with_error(sync_error);
                Err(error)
            }
            (Err(operation_error), Err(sync_error)) => {
                let error = self.poison_with_error(format!(
                    "{operation_error}; subsequent stream synchronization also failed: {sync_error}"
                ));
                Err(error)
            }
        }
    }

    /// Copies one complete device-resident input into the stack and synchronizes before execution.
    pub(crate) fn upload_device_input_synchronized(
        &mut self,
        input: &RuntimeBuffer,
        stream: &mut RuntimeStream,
    ) -> Result<(), String> {
        self.validate_runtime_contract()?;
        let expected_bytes = hidden_bytes(self.config)?;
        let actual_bytes = input.size()?;
        if actual_bytes != expected_bytes {
            return Err(format!(
                "Qwen3-14B SQ8 stack device input byte size mismatch: expected={expected_bytes} actual={actual_bytes}"
            ));
        }
        self.state.mark_needs_input()?;
        self.last_execution_report = None;
        let copy_result = self
            .resident_hidden
            .copy_from_buffer(0, input, 0, expected_bytes, Some(&mut *stream))
            .map_err(|err| format!("failed to copy Qwen3-14B SQ8 stack device input: {err}"));
        let sync_result = stream.synchronize().map_err(|err| {
            format!("failed to synchronize Qwen3-14B SQ8 stack device input: {err}")
        });
        match (copy_result, sync_result) {
            (Ok(()), Ok(())) => self.state.mark_input_ready(),
            (Err(operation_error), Ok(())) => Err(self.poison_with_error(operation_error)),
            (Ok(()), Err(sync_error)) => Err(self.poison_with_error(sync_error)),
            (Err(operation_error), Err(sync_error)) => Err(self.poison_with_error(format!(
                "{operation_error}; subsequent stream synchronization also failed: {sync_error}"
            ))),
        }
    }

    /// Runs all 40 resident layers after a synchronized host upload.
    ///
    /// The upload and its synchronization are outside the layer execution interval. The layer
    /// interval contains exactly 40 output-to-hidden D2D copies and one final synchronization.
    pub fn run_host_input_synchronized(
        &mut self,
        input: &[f32],
        profile: Sq8LayerExecutionProfile,
        stream: &mut RuntimeStream,
    ) -> Result<Sq8StackExecutionReport, String> {
        self.upload_host_input_synchronized(input, stream)?;
        let mut report = self.run_uploaded_synchronized(profile, stream)?;
        report.input_origin = Sq8StackInputOrigin::SynchronizedHostUploadBeforeExecution;
        report.input_upload_synchronization_count = 1;
        if let Err(err) = report.validate_contract() {
            return Err(self.poison_with_error(format!(
                "Qwen3-14B SQ8 stack host-input report validation failed: {err}"
            )));
        }
        self.last_execution_report = Some(report.clone());
        Ok(report)
    }

    /// Runs the promotion path and rejects any non-CK projection or fallback report.
    pub fn run_host_input_optimized_synchronized(
        &mut self,
        input: &[f32],
        stream: &mut RuntimeStream,
    ) -> Result<Sq8StackExecutionReport, String> {
        let report = self.run_host_input_synchronized(
            input,
            Sq8LayerExecutionProfile::Rdna4W8a8BlockCk,
            stream,
        )?;
        if let Err(err) = report.validate_optimized_promotion() {
            return Err(self.poison_with_error(format!(
                "Qwen3-14B SQ8 stack optimized promotion validation failed: {err}"
            )));
        }
        Ok(report)
    }

    /// Enqueues 40 layers and 40 D2D copies on one stream, then synchronizes exactly once.
    pub fn run_uploaded_synchronized(
        &mut self,
        profile: Sq8LayerExecutionProfile,
        stream: &mut RuntimeStream,
    ) -> Result<Sq8StackExecutionReport, String> {
        self.validate_execution_preconditions()?;
        self.state.mark_needs_input()?;
        self.last_execution_report = None;
        let mut layer_reports = Vec::with_capacity(QWEN3_14B_SQ8_STACK_LAYERS);

        for layer_index in 0..QWEN3_14B_SQ8_STACK_LAYERS {
            let result = self.enqueue_layer_and_copy(layer_index, profile, stream);
            match result {
                Ok(report) => layer_reports.push(report),
                Err(err) => return Err(self.poison_after_stream_recovery(stream, err)),
            }
        }

        if let Err(err) = stream.synchronize() {
            return Err(self.poison_with_error(format!(
                "failed to synchronize Qwen3-14B SQ8 stack execution: {err}"
            )));
        }
        let report = match build_report(
            self.config,
            &self.artifact_content_sha256,
            profile,
            Sq8StackExecutionMode::SynchronizedResident,
            Sq8StackInputOrigin::PreviouslyUploadedResident,
            layer_reports,
            0,
            1,
            0,
            false,
        ) {
            Ok(report) => report,
            Err(err) => {
                return Err(self.poison_with_error(format!(
                    "Qwen3-14B SQ8 stack execution report validation failed: {err}"
                )));
            }
        };
        self.state.mark_output_ready()?;
        self.last_execution_report = Some(report.clone());
        Ok(report)
    }

    /// Runs an already uploaded resident input through the promotion path.
    pub fn run_uploaded_optimized_synchronized(
        &mut self,
        stream: &mut RuntimeStream,
    ) -> Result<Sq8StackExecutionReport, String> {
        let report =
            self.run_uploaded_synchronized(Sq8LayerExecutionProfile::Rdna4W8a8BlockCk, stream)?;
        if let Err(err) = report.validate_optimized_promotion() {
            return Err(self.poison_with_error(format!(
                "Qwen3-14B SQ8 stack optimized promotion validation failed: {err}"
            )));
        }
        Ok(report)
    }

    pub(crate) fn run_uploaded_paged_prefill_optimized_synchronized(
        &mut self,
        caches: &mut [PagedDecodeState],
        stream: &mut RuntimeStream,
    ) -> Result<Sq8PagedStackExecutionReport, String> {
        self.validate_execution_preconditions()?;
        validate_paged_cache_layer_count(caches)?;
        validate_paged_hip_only_guards()?;
        if let Some((layer_index, cache)) = caches
            .iter()
            .enumerate()
            .find(|(_, cache)| cache.written_len() != 0)
        {
            return Err(format!(
                "Qwen3-14B SQ8 paged prefill layer {layer_index} cache is not empty: written_len={}",
                cache.written_len()
            ));
        }

        self.state.mark_needs_input()?;
        self.last_execution_report = None;
        let profile = Sq8LayerExecutionProfile::Rdna4W8a8BlockCk;
        let mut layer_reports = Vec::with_capacity(QWEN3_14B_SQ8_STACK_LAYERS);
        for (layer_index, cache) in caches.iter_mut().enumerate() {
            match self.enqueue_layer_and_copy_paged_prefill(layer_index, profile, cache, stream) {
                Ok(report) => layer_reports.push(report),
                Err(err) => return Err(self.poison_after_stream_recovery(stream, err)),
            }
        }
        if let Err(err) = stream.synchronize() {
            return Err(self.poison_with_error(format!(
                "failed to synchronize Qwen3-14B SQ8 paged prefill: {err}"
            )));
        }
        let stack = build_report(
            self.config,
            &self.artifact_content_sha256,
            profile,
            Sq8StackExecutionMode::SynchronizedResident,
            Sq8StackInputOrigin::PreviouslyUploadedResident,
            layer_reports,
            0,
            1,
            0,
            false,
        )
        .map_err(|err| {
            self.poison_with_error(format!(
                "Qwen3-14B SQ8 paged prefill report validation failed: {err}"
            ))
        })?;
        let report = Sq8PagedStackExecutionReport {
            phase: Sq8PagedStackPhase::Prefill,
            position: 0,
            cache_lengths: cache_lengths_array(caches)?,
            kv_write_calls: self.config.sequence_len * QWEN3_14B_SQ8_STACK_LAYERS,
            paged_attention_calls: 0,
            input_d2d_copy_count: 0,
            stack: stack.clone(),
        };
        if let Err(err) = report.validate_contract() {
            return Err(self.poison_with_error(format!(
                "Qwen3-14B SQ8 paged prefill contract failed: {err}"
            )));
        }
        self.state.mark_output_ready()?;
        self.last_execution_report = Some(stack);
        Ok(report)
    }

    pub(crate) fn run_paged_decode_optimized_synchronized(
        &mut self,
        decode: &mut Qwen3Sq8PagedDecodeRuntime,
        input: &RuntimeBuffer,
        position: usize,
        caches: &mut [PagedDecodeState],
        stream: &mut RuntimeStream,
    ) -> Result<Sq8PagedStackExecutionReport, String> {
        self.validate_runtime_contract()?;
        self.state.require_output_ready()?;
        decode.ensure_usable()?;
        validate_paged_cache_layer_count(caches)?;
        validate_paged_hip_only_guards()?;
        if input.size()? != hidden_bytes(decode.config)? {
            return Err(format!(
                "Qwen3-14B SQ8 paged decode input byte size mismatch: expected={} actual={}",
                hidden_bytes(decode.config)?,
                input.size()?
            ));
        }
        if let Some((layer_index, cache)) = caches
            .iter()
            .enumerate()
            .find(|(_, cache)| cache.written_len() != position)
        {
            return Err(format!(
                "Qwen3-14B SQ8 paged decode layer {layer_index} cache position mismatch: expected={position} actual={}",
                cache.written_len()
            ));
        }

        decode.output_ready = false;
        decode.last_execution_report = None;
        if let Err(err) = decode.resident_hidden.copy_from_buffer(
            0,
            input,
            0,
            hidden_bytes(decode.config)?,
            Some(&mut *stream),
        ) {
            let error = self.poison_after_stream_recovery(
                stream,
                format!("failed to copy Qwen3-14B SQ8 decode input D2D: {err}"),
            );
            decode.poison(error.clone());
            return Err(error);
        }

        let profile = Sq8LayerExecutionProfile::Rdna4W8a8BlockCk;
        let mut layer_reports = Vec::with_capacity(QWEN3_14B_SQ8_STACK_LAYERS);
        for (layer_index, cache) in caches.iter_mut().enumerate() {
            match self.enqueue_paged_decode_layer_and_copy(
                decode,
                layer_index,
                profile,
                position,
                cache,
                stream,
            ) {
                Ok(report) => layer_reports.push(report),
                Err(err) => {
                    let error = self.poison_after_stream_recovery(stream, err);
                    decode.poison(error.clone());
                    return Err(error);
                }
            }
        }
        if let Err(err) = stream.synchronize() {
            let error = self.poison_with_error(format!(
                "failed to synchronize Qwen3-14B SQ8 paged decode: {err}"
            ));
            decode.poison(error.clone());
            return Err(error);
        }
        let stack = match build_report(
            decode.config,
            &self.artifact_content_sha256,
            profile,
            Sq8StackExecutionMode::SynchronizedResident,
            Sq8StackInputOrigin::PreviouslyUploadedResident,
            layer_reports,
            0,
            1,
            0,
            false,
        ) {
            Ok(report) => report,
            Err(err) => {
                let error = self.poison_with_error(format!(
                    "Qwen3-14B SQ8 paged decode report validation failed: {err}"
                ));
                decode.poison(error.clone());
                return Err(error);
            }
        };
        let report = Sq8PagedStackExecutionReport {
            phase: Sq8PagedStackPhase::Decode,
            position,
            cache_lengths: cache_lengths_array(caches)?,
            kv_write_calls: QWEN3_14B_SQ8_STACK_LAYERS,
            paged_attention_calls: QWEN3_14B_SQ8_STACK_LAYERS,
            input_d2d_copy_count: 1,
            stack,
        };
        if let Err(err) = report.validate_contract() {
            let error = self
                .poison_with_error(format!("Qwen3-14B SQ8 paged decode contract failed: {err}"));
            decode.poison(error.clone());
            return Err(error);
        }
        decode.output_ready = true;
        decode.last_execution_report = Some(report.clone());
        Ok(report)
    }

    /// Runs a non-timed audit and returns every layer output on the host.
    ///
    /// This path intentionally synchronizes and stages one output after every layer. It must not
    /// be used for stack performance measurements.
    pub fn run_host_input_layerwise_audit(
        &mut self,
        input: &[f32],
        profile: Sq8LayerExecutionProfile,
        stream: &mut RuntimeStream,
    ) -> Result<Sq8StackLayerwiseAudit, String> {
        self.upload_host_input_synchronized(input, stream)?;
        self.validate_execution_preconditions()?;
        let output_bytes = hidden_bytes(self.config)?;
        self.state.mark_needs_input()?;
        self.last_execution_report = None;
        let mut layer_reports = Vec::with_capacity(QWEN3_14B_SQ8_STACK_LAYERS);
        let mut layer_audits = Vec::with_capacity(QWEN3_14B_SQ8_STACK_LAYERS);

        for layer_index in 0..QWEN3_14B_SQ8_STACK_LAYERS {
            let report = match self.enqueue_layer_and_copy(layer_index, profile, stream) {
                Ok(report) => report,
                Err(err) => return Err(self.poison_after_stream_recovery(stream, err)),
            };
            let mut host = vec![0_u8; output_bytes];
            let copy_result = self
                .resident_hidden
                .copy_to_host(0, &mut host, Some(&mut *stream))
                .map_err(|err| {
                    format!("failed to read Qwen3-14B SQ8 stack layer {layer_index}: {err}")
                });
            let sync_result = stream.synchronize().map_err(|err| {
                format!("failed to synchronize Qwen3-14B SQ8 stack layer {layer_index}: {err}")
            });
            match (copy_result, sync_result) {
                (Ok(()), Ok(())) => {}
                (Err(operation_error), Ok(())) => {
                    return Err(self.poison_with_error(operation_error));
                }
                (Ok(()), Err(sync_error)) => {
                    return Err(self.poison_with_error(sync_error));
                }
                (Err(operation_error), Err(sync_error)) => {
                    return Err(self.poison_with_error(format!(
                        "{operation_error}; subsequent stream synchronization also failed: {sync_error}"
                    )));
                }
            }
            let output = decode_f32_le_values(&host);
            if let Err(err) = validate_finite_layer_output(layer_index, &output) {
                return Err(self.poison_with_error(err));
            }
            layer_reports.push(report);
            layer_audits.push(Sq8StackLayerAudit {
                layer_index,
                output,
            });
        }

        let report = match build_report(
            self.config,
            &self.artifact_content_sha256,
            profile,
            Sq8StackExecutionMode::LayerwiseAuditNonTimed,
            Sq8StackInputOrigin::SynchronizedHostUploadBeforeExecution,
            layer_reports,
            1,
            QWEN3_14B_SQ8_STACK_LAYERS,
            QWEN3_14B_SQ8_STACK_LAYERS,
            true,
        ) {
            Ok(report) => report,
            Err(err) => {
                return Err(self.poison_with_error(format!(
                    "Qwen3-14B SQ8 stack audit report validation failed: {err}"
                )));
            }
        };
        let converted: Result<
            [Sq8StackLayerAudit; QWEN3_14B_SQ8_STACK_LAYERS],
            Vec<Sq8StackLayerAudit>,
        > = layer_audits.try_into();
        let layers = match converted {
            Ok(layers) => layers,
            Err(layers) => {
                return Err(self.poison_with_error(format!(
                    "Qwen3-14B SQ8 stack audit layer count mismatch: expected={} actual={}",
                    QWEN3_14B_SQ8_STACK_LAYERS,
                    layers.len()
                )));
            }
        };
        self.state.mark_output_ready()?;
        self.last_execution_report = Some(report.clone());
        Ok(Sq8StackLayerwiseAudit { report, layers })
    }

    fn enqueue_layer_and_copy(
        &mut self,
        layer_index: usize,
        profile: Sq8LayerExecutionProfile,
        stream: &mut RuntimeStream,
    ) -> Result<Sq8LayerExecutionReport, String> {
        let weights = self.weights.get(layer_index).ok_or_else(|| {
            format!("Qwen3-14B SQ8 stack layer index {layer_index} is out of range")
        })?;
        if weights.layer_index != layer_index {
            return Err(format!(
                "Qwen3-14B SQ8 stack layer order mismatch: slot={layer_index} weight={}",
                weights.layer_index
            ));
        }
        let report = self
            .workspace
            .enqueue(weights, &self.resident_hidden, profile, stream)
            .map_err(|err| format!("Qwen3-14B SQ8 stack layer {layer_index} failed: {err}"))?;
        self.resident_hidden
            .copy_from_buffer(
                0,
                self.workspace.output_buffer(),
                0,
                hidden_bytes(self.config)?,
                Some(&mut *stream),
            )
            .map_err(|err| {
                format!("failed to copy Qwen3-14B SQ8 stack layer {layer_index} output D2D: {err}")
            })?;
        Ok(report)
    }

    fn enqueue_layer_and_copy_paged_prefill(
        &mut self,
        layer_index: usize,
        profile: Sq8LayerExecutionProfile,
        cache: &mut PagedDecodeState,
        stream: &mut RuntimeStream,
    ) -> Result<Sq8LayerExecutionReport, String> {
        let weights = self.weights.get(layer_index).ok_or_else(|| {
            format!("Qwen3-14B SQ8 stack layer index {layer_index} is out of range")
        })?;
        if weights.layer_index != layer_index {
            return Err(format!(
                "Qwen3-14B SQ8 stack layer order mismatch: slot={layer_index} weight={}",
                weights.layer_index
            ));
        }
        let report = self
            .workspace
            .enqueue_prefill_with_paged_kv(weights, &self.resident_hidden, profile, cache, stream)
            .map_err(|err| {
                format!("Qwen3-14B SQ8 paged prefill layer {layer_index} failed: {err}")
            })?;
        self.resident_hidden
            .copy_from_buffer(
                0,
                self.workspace.output_buffer(),
                0,
                hidden_bytes(self.config)?,
                Some(&mut *stream),
            )
            .map_err(|err| {
                format!(
                    "failed to copy Qwen3-14B SQ8 paged prefill layer {layer_index} output D2D: {err}"
                )
            })?;
        Ok(report)
    }

    fn enqueue_paged_decode_layer_and_copy(
        &self,
        decode: &mut Qwen3Sq8PagedDecodeRuntime,
        layer_index: usize,
        profile: Sq8LayerExecutionProfile,
        position: usize,
        cache: &mut PagedDecodeState,
        stream: &mut RuntimeStream,
    ) -> Result<Sq8LayerExecutionReport, String> {
        let weights = self.weights.get(layer_index).ok_or_else(|| {
            format!("Qwen3-14B SQ8 stack layer index {layer_index} is out of range")
        })?;
        if weights.layer_index != layer_index {
            return Err(format!(
                "Qwen3-14B SQ8 stack layer order mismatch: slot={layer_index} weight={}",
                weights.layer_index
            ));
        }
        let report = decode
            .workspace
            .enqueue_paged_decode(
                weights,
                &decode.resident_hidden,
                profile,
                position,
                cache,
                stream,
            )
            .map_err(|err| {
                format!("Qwen3-14B SQ8 paged decode layer {layer_index} failed: {err}")
            })?;
        decode
            .resident_hidden
            .copy_from_buffer(
                0,
                decode.workspace.output_buffer(),
                0,
                hidden_bytes(decode.config)?,
                Some(&mut *stream),
            )
            .map_err(|err| {
                format!(
                    "failed to copy Qwen3-14B SQ8 paged decode layer {layer_index} output D2D: {err}"
                )
            })?;
        Ok(report)
    }

    fn validate_runtime_contract(&self) -> Result<(), String> {
        self.state.ensure_usable()?;
        self.config.validate()?;
        if self.config.position_offset != 0 {
            return Err("Qwen3-14B SQ8 stack requires position_offset=0".into());
        }
        validate_loaded_layer_indices(self.weights.iter().map(|weights| weights.layer_index))?;
        validate_sha256(&self.artifact_content_sha256)?;
        if self.workspace.config() != self.config {
            return Err("Qwen3-14B SQ8 stack workspace configuration mismatch".into());
        }
        let expected_bytes = hidden_bytes(self.config)?;
        let actual_bytes = self.resident_hidden.size()?;
        if actual_bytes != expected_bytes {
            return Err(format!(
                "Qwen3-14B SQ8 stack hidden buffer size mismatch: expected={expected_bytes} actual={actual_bytes}"
            ));
        }
        Ok(())
    }

    fn validate_execution_preconditions(&self) -> Result<(), String> {
        self.validate_runtime_contract()?;
        self.state.require_input_ready()?;
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
                "{operation_error}; subsequent Qwen3-14B SQ8 stack stream recovery failed: {sync_error}"
            ),
        };
        self.poison_with_error(error)
    }

    fn poison_with_error(&mut self, error: String) -> String {
        self.last_execution_report = None;
        self.state.poison(error.clone());
        error
    }
}

#[allow(clippy::too_many_arguments)]
fn build_report(
    config: Qwen3Sq8LayerConfig,
    artifact_content_sha256: &str,
    profile: Sq8LayerExecutionProfile,
    mode: Sq8StackExecutionMode,
    input_origin: Sq8StackInputOrigin,
    layer_reports: Vec<Sq8LayerExecutionReport>,
    input_upload_synchronization_count: usize,
    execution_synchronization_count: usize,
    host_readback_count: usize,
    host_staging_used: bool,
) -> Result<Sq8StackExecutionReport, String> {
    let layer_reports: [Sq8LayerExecutionReport; QWEN3_14B_SQ8_STACK_LAYERS] =
        layer_reports.try_into().map_err(|reports: Vec<_>| {
            format!(
                "Qwen3-14B SQ8 stack report layer count mismatch: expected={} actual={}",
                QWEN3_14B_SQ8_STACK_LAYERS,
                reports.len()
            )
        })?;
    for (layer_index, report) in layer_reports.iter().enumerate() {
        if report.profile != profile {
            return Err(format!(
                "Qwen3-14B SQ8 stack layer {layer_index} report profile mismatch"
            ));
        }
        if report.projection_calls != QWEN3_14B_SQ8_LAYER_PROJECTIONS {
            return Err(format!(
                "Qwen3-14B SQ8 stack layer {layer_index} projection count mismatch: expected={} actual={}",
                QWEN3_14B_SQ8_LAYER_PROJECTIONS, report.projection_calls
            ));
        }
    }
    let activation_quantizations = layer_reports.iter().try_fold(0_usize, |total, report| {
        total
            .checked_add(report.activation_quantizations)
            .ok_or_else(|| {
                "Qwen3-14B SQ8 stack activation quantization count overflows".to_string()
            })
    })?;
    let projection_calls = layer_reports.iter().try_fold(0_usize, |total, report| {
        total
            .checked_add(report.projection_calls)
            .ok_or_else(|| "Qwen3-14B SQ8 stack projection count overflows".to_string())
    })?;
    let fallback_used = layer_reports.iter().any(|report| report.fallback_used);
    let report = Sq8StackExecutionReport {
        profile,
        mode,
        input_origin,
        sequence_len: config.sequence_len,
        position_offset: config.position_offset,
        artifact_content_sha256: artifact_content_sha256.to_string(),
        layer_reports,
        activation_quantizations,
        projection_calls,
        d2d_copy_count: QWEN3_14B_SQ8_STACK_LAYERS,
        input_upload_synchronization_count,
        execution_synchronization_count,
        host_readback_count,
        fallback_used,
        host_staging_used,
    };
    report.validate_contract()?;
    Ok(report)
}

fn validate_measured_ck_dispatch(
    m: usize,
    layer_index: usize,
    report: &Sq8LayerExecutionReport,
) -> Result<(), String> {
    if !matches!(m, 1 | 2 | 4 | 8 | 16 | 32 | 128) {
        return Err(format!(
            "Qwen3-14B SQ8 stack CK dispatch has unmeasured M {m}"
        ));
    }
    let hidden = Sq8CkImplementation::MemV1DefaultTile16x128x128;
    let (gate_up, down) = if m == 128 {
        (
            Sq8CkImplementation::MemV1DefaultTile16x256x128,
            Sq8CkImplementation::MemV1DefaultTile16x128x128,
        )
    } else {
        (
            Sq8CkImplementation::MemV1KPaddingTile16x128x256,
            Sq8CkImplementation::MemV1DefaultTile16x128x256,
        )
    };
    for (projection, actual, expected) in [
        ("q", report.q, hidden),
        ("k", report.k, hidden),
        ("v", report.v, hidden),
        ("o", report.o, hidden),
        ("gate", report.gate, gate_up),
        ("up", report.up, gate_up),
        ("down", report.down, down),
    ] {
        let expected = Sq8LayerProjectionExecution::Ck(expected);
        if actual != expected {
            return Err(format!(
                "Qwen3-14B SQ8 stack layer {layer_index} {projection} dispatch mismatch for M={m}: expected={expected:?} actual={actual:?}"
            ));
        }
    }
    Ok(())
}

fn validate_stack_artifact(artifact: &Sq8CanonicalArtifact) -> Result<(), String> {
    let tensors = &artifact.manifest().quantized_tensors;
    if tensors.len() != QWEN3_14B_SQ8_STACK_PROJECTIONS {
        return Err(format!(
            "Qwen3-14B SQ8 stack artifact projection count mismatch: expected={} actual={}",
            QWEN3_14B_SQ8_STACK_PROJECTIONS,
            tensors.len()
        ));
    }
    for layer_index in 0..QWEN3_14B_SQ8_STACK_LAYERS {
        let [q, k, v, o, gate, up, down] = qwen3_sq8_layer_tensor_names(layer_index);
        for (name, expected_shape) in [
            (q, [QWEN3_14B_HIDDEN_SIZE, QWEN3_14B_HIDDEN_SIZE]),
            (
                k,
                [
                    QWEN3_14B_KV_HEADS * QWEN3_14B_HEAD_DIM,
                    QWEN3_14B_HIDDEN_SIZE,
                ],
            ),
            (
                v,
                [
                    QWEN3_14B_KV_HEADS * QWEN3_14B_VALUE_DIM,
                    QWEN3_14B_HIDDEN_SIZE,
                ],
            ),
            (o, [QWEN3_14B_HIDDEN_SIZE, QWEN3_14B_HIDDEN_SIZE]),
            (gate, [QWEN3_14B_INTERMEDIATE_SIZE, QWEN3_14B_HIDDEN_SIZE]),
            (up, [QWEN3_14B_INTERMEDIATE_SIZE, QWEN3_14B_HIDDEN_SIZE]),
            (down, [QWEN3_14B_HIDDEN_SIZE, QWEN3_14B_INTERMEDIATE_SIZE]),
        ] {
            let pair = artifact.tensor_pair(&name)?;
            let expected = [expected_shape[0] as u64, expected_shape[1] as u64];
            if pair.shape != expected {
                return Err(format!(
                    "Qwen3-14B SQ8 stack tensor {name} shape mismatch: expected={expected:?} actual={:?}",
                    pair.shape
                ));
            }
        }
    }
    validate_sha256(&artifact.manifest().integrity.content_sha256)
}

fn validate_norm_layer_count(actual: usize) -> Result<(), String> {
    if actual != QWEN3_14B_SQ8_STACK_LAYERS {
        return Err(format!(
            "Qwen3-14B SQ8 stack norm layer count mismatch: expected={} actual={actual}",
            QWEN3_14B_SQ8_STACK_LAYERS
        ));
    }
    Ok(())
}

fn validate_paged_cache_layer_count(caches: &[PagedDecodeState]) -> Result<(), String> {
    if caches.len() != QWEN3_14B_SQ8_STACK_LAYERS {
        return Err(format!(
            "Qwen3-14B SQ8 paged KV layer count mismatch: expected={} actual={}",
            QWEN3_14B_SQ8_STACK_LAYERS,
            caches.len()
        ));
    }
    Ok(())
}

fn cache_lengths_array(
    caches: &[PagedDecodeState],
) -> Result<[usize; QWEN3_14B_SQ8_STACK_LAYERS], String> {
    validate_paged_cache_layer_count(caches)?;
    let lengths = caches
        .iter()
        .map(PagedDecodeState::written_len)
        .collect::<Vec<_>>();
    lengths.try_into().map_err(|lengths: Vec<_>| {
        format!(
            "Qwen3-14B SQ8 paged KV length count mismatch: expected={} actual={}",
            QWEN3_14B_SQ8_STACK_LAYERS,
            lengths.len()
        )
    })
}

fn validate_loaded_layer_indices(indices: impl IntoIterator<Item = usize>) -> Result<(), String> {
    let mut count = 0_usize;
    for (expected, actual) in indices.into_iter().enumerate() {
        if expected >= QWEN3_14B_SQ8_STACK_LAYERS {
            return Err(format!(
                "Qwen3-14B SQ8 stack has more than {} layers",
                QWEN3_14B_SQ8_STACK_LAYERS
            ));
        }
        if actual != expected {
            return Err(format!(
                "Qwen3-14B SQ8 stack layer order mismatch: slot={expected} weight={actual}"
            ));
        }
        count += 1;
    }
    if count != QWEN3_14B_SQ8_STACK_LAYERS {
        return Err(format!(
            "Qwen3-14B SQ8 stack weight layer count mismatch: expected={} actual={count}",
            QWEN3_14B_SQ8_STACK_LAYERS
        ));
    }
    Ok(())
}

fn boxed_layer_array(
    layers: Vec<Qwen3Sq8LayerWeights>,
) -> Result<Box<[Qwen3Sq8LayerWeights; QWEN3_14B_SQ8_STACK_LAYERS]>, String> {
    let actual = layers.len();
    layers.into_boxed_slice().try_into().map_err(|_| {
        format!(
            "Qwen3-14B SQ8 stack loaded layer count mismatch: expected={} actual={actual}",
            QWEN3_14B_SQ8_STACK_LAYERS
        )
    })
}

fn hidden_bytes(config: Qwen3Sq8LayerConfig) -> Result<usize, String> {
    config
        .sequence_len
        .checked_mul(QWEN3_14B_HIDDEN_SIZE)
        .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| "Qwen3-14B SQ8 stack hidden byte size overflows".to_string())
}

fn validate_host_input(input: &[f32], config: Qwen3Sq8LayerConfig) -> Result<(), String> {
    let expected_bytes = hidden_bytes(config)?;
    let expected_elements = expected_bytes / std::mem::size_of::<f32>();
    if input.len() != expected_elements {
        return Err(format!(
            "Qwen3-14B SQ8 stack input length mismatch: expected={expected_elements} actual={}",
            input.len()
        ));
    }
    if let Some((index, value)) = input
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!(
            "Qwen3-14B SQ8 stack input contains non-finite value {value} at index {index}"
        ));
    }
    Ok(())
}

fn validate_finite_layer_output(layer_index: usize, output: &[f32]) -> Result<(), String> {
    if let Some((index, value)) = output
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!(
            "Qwen3-14B SQ8 stack layer {layer_index} output contains non-finite value {value} at index {index}"
        ));
    }
    Ok(())
}

fn validate_hip_only_guards() -> Result<(), String> {
    let missing = QWEN3_14B_SQ8_REQUIRED_HIP_KERNEL_ENV
        .into_iter()
        .filter(|name| std::env::var_os(name).is_none())
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(format!(
            "Qwen3-14B SQ8 stack requires HIP-only primitive execution; set these no-staging guards before process start: {}",
            missing.join(",")
        ));
    }
    Ok(())
}

fn validate_paged_hip_only_guards() -> Result<(), String> {
    let missing = QWEN3_14B_SQ8_PAGED_REQUIRED_HIP_KERNEL_ENV
        .into_iter()
        .filter(|name| std::env::var_os(name).is_none())
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(format!(
            "Qwen3-14B SQ8 paged stack requires HIP-only KV primitives; set these no-staging guards before process start: {}",
            missing.join(",")
        ));
    }
    Ok(())
}

fn validate_sha256(value: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte))
    {
        return Err("Qwen3-14B SQ8 stack artifact content SHA-256 is invalid".into());
    }
    Ok(())
}

fn load_error_after_stream_recovery(stream: &mut RuntimeStream, operation_error: String) -> String {
    match stream.synchronize() {
        Ok(()) => operation_error,
        Err(sync_error) => format!(
            "{operation_error}; subsequent Qwen3-14B SQ8 stack stream recovery failed: {sync_error}"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ullm_runtime_sys::{Sq8CkImplementation, SqFp8ExecutionPath};

    fn reference_report(fallback_used: bool) -> Sq8LayerExecutionReport {
        use crate::sq8_layer_runtime::Sq8LayerProjectionExecution::Reference;
        let hip = Reference(SqFp8ExecutionPath::HipKernel);
        Sq8LayerExecutionReport {
            profile: Sq8LayerExecutionProfile::ReferenceW8a16Block2d,
            q: hip,
            k: hip,
            v: hip,
            o: hip,
            gate: hip,
            up: hip,
            down: hip,
            activation_quantizations: 0,
            projection_calls: 7,
            fallback_used,
        }
    }

    fn ck_report(m: usize) -> Sq8LayerExecutionReport {
        use crate::sq8_layer_runtime::Sq8LayerProjectionExecution::Ck;
        let hidden = Ck(Sq8CkImplementation::MemV1DefaultTile16x128x128);
        let (gate_up, down) = if m == 128 {
            (
                Ck(Sq8CkImplementation::MemV1DefaultTile16x256x128),
                Ck(Sq8CkImplementation::MemV1DefaultTile16x128x128),
            )
        } else {
            (
                Ck(Sq8CkImplementation::MemV1KPaddingTile16x128x256),
                Ck(Sq8CkImplementation::MemV1DefaultTile16x128x256),
            )
        };
        Sq8LayerExecutionReport {
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
        }
    }

    #[test]
    fn stack_contract_is_fixed_to_ordered_40_layers() {
        assert!(validate_loaded_layer_indices(0..40).is_ok());
        assert!(validate_loaded_layer_indices(0..39).is_err());
        assert!(validate_loaded_layer_indices(0..41).is_err());
        let mut reordered = (0..40).collect::<Vec<_>>();
        reordered.swap(8, 9);
        assert!(validate_loaded_layer_indices(reordered).is_err());
    }

    #[test]
    fn stack_contract_rejects_wrong_norm_count() {
        assert!(validate_norm_layer_count(40).is_ok());
        assert!(validate_norm_layer_count(39).is_err());
        assert!(validate_norm_layer_count(41).is_err());
    }

    #[test]
    fn stack_hidden_bytes_cover_only_measured_m_and_detect_bad_input() {
        for m in [1, 2, 4, 8, 16, 32, 128] {
            let config = Qwen3Sq8LayerConfig::qwen3_14b(m, 0).unwrap();
            assert_eq!(hidden_bytes(config).unwrap(), m * 5120 * 4);
            let mut input = vec![0.0_f32; m * 5120];
            assert!(validate_host_input(&input, config).is_ok());
            input[0] = f32::NAN;
            assert!(validate_host_input(&input, config).is_err());
        }
        assert!(Qwen3Sq8LayerConfig::qwen3_14b(64, 0).is_err());
        assert!(Qwen3Sq8LayerConfig::qwen3_14b(8, 1).is_err());
    }

    #[test]
    fn stack_report_requires_40_complete_layer_reports() {
        let config = Qwen3Sq8LayerConfig::qwen3_14b(8, 0).unwrap();
        let hash = "a".repeat(64);
        assert!(
            build_report(
                config,
                &hash,
                Sq8LayerExecutionProfile::ReferenceW8a16Block2d,
                Sq8StackExecutionMode::SynchronizedResident,
                Sq8StackInputOrigin::PreviouslyUploadedResident,
                vec![reference_report(false); 39],
                0,
                1,
                0,
                false,
            )
            .is_err()
        );
        let report = build_report(
            config,
            &hash,
            Sq8LayerExecutionProfile::ReferenceW8a16Block2d,
            Sq8StackExecutionMode::SynchronizedResident,
            Sq8StackInputOrigin::PreviouslyUploadedResident,
            vec![reference_report(false); 40],
            0,
            1,
            0,
            false,
        )
        .unwrap();
        assert_eq!(report.layer_reports.len(), 40);
        assert_eq!(report.projection_calls, 280);
        assert_eq!(report.activation_quantizations, 0);
        assert_eq!(report.d2d_copy_count, 40);
        assert!(!report.fallback_used);
        assert!(report.all_reference_hip());
        assert!(!report.host_staging_used);

        let mut fallback_reports = vec![reference_report(false); 40];
        fallback_reports[17].fallback_used = true;
        assert!(
            build_report(
                config,
                &hash,
                Sq8LayerExecutionProfile::ReferenceW8a16Block2d,
                Sq8StackExecutionMode::SynchronizedResident,
                Sq8StackInputOrigin::PreviouslyUploadedResident,
                fallback_reports,
                0,
                1,
                0,
                false,
            )
            .is_err()
        );
    }

    #[test]
    fn optimized_promotion_requires_280_ck_and_160_quantizations() {
        let config = Qwen3Sq8LayerConfig::qwen3_14b(8, 0).unwrap();
        let hash = "b".repeat(64);
        let report = build_report(
            config,
            &hash,
            Sq8LayerExecutionProfile::Rdna4W8a8BlockCk,
            Sq8StackExecutionMode::SynchronizedResident,
            Sq8StackInputOrigin::PreviouslyUploadedResident,
            vec![ck_report(8); 40],
            0,
            1,
            0,
            false,
        )
        .unwrap();
        assert_eq!(report.projection_calls, 280);
        assert_eq!(report.activation_quantizations, 160);
        assert!(report.all_ck());
        assert!(report.validate_optimized_promotion().is_ok());

        let mut audit = report.clone();
        audit.mode = Sq8StackExecutionMode::LayerwiseAuditNonTimed;
        audit.execution_synchronization_count = 40;
        audit.host_readback_count = 40;
        audit.host_staging_used = true;
        assert!(audit.validate_contract().is_ok());
        assert!(audit.validate_optimized_promotion().is_err());
    }

    #[test]
    fn paged_stack_reports_bind_prefill_and_decode_cache_progress() {
        let prefill_stack = build_report(
            Qwen3Sq8LayerConfig::qwen3_14b(8, 0).unwrap(),
            &"b".repeat(64),
            Sq8LayerExecutionProfile::Rdna4W8a8BlockCk,
            Sq8StackExecutionMode::SynchronizedResident,
            Sq8StackInputOrigin::PreviouslyUploadedResident,
            vec![ck_report(8); 40],
            0,
            1,
            0,
            false,
        )
        .unwrap();
        let prefill = Sq8PagedStackExecutionReport {
            phase: Sq8PagedStackPhase::Prefill,
            position: 0,
            stack: prefill_stack,
            cache_lengths: [8; 40],
            kv_write_calls: 320,
            paged_attention_calls: 0,
            input_d2d_copy_count: 0,
        };
        assert!(prefill.validate_contract().is_ok());
        let mut wrong_prefill = prefill.clone();
        wrong_prefill.cache_lengths[17] = 7;
        assert!(wrong_prefill.validate_contract().is_err());

        let decode_stack = build_report(
            Qwen3Sq8LayerConfig::qwen3_14b(1, 0).unwrap(),
            &"b".repeat(64),
            Sq8LayerExecutionProfile::Rdna4W8a8BlockCk,
            Sq8StackExecutionMode::SynchronizedResident,
            Sq8StackInputOrigin::PreviouslyUploadedResident,
            vec![ck_report(1); 40],
            0,
            1,
            0,
            false,
        )
        .unwrap();
        let decode = Sq8PagedStackExecutionReport {
            phase: Sq8PagedStackPhase::Decode,
            position: 8,
            stack: decode_stack,
            cache_lengths: [9; 40],
            kv_write_calls: 40,
            paged_attention_calls: 40,
            input_d2d_copy_count: 1,
        };
        assert!(decode.validate_contract().is_ok());
        let mut wrong_decode = decode.clone();
        wrong_decode.input_d2d_copy_count = 0;
        assert!(wrong_decode.validate_contract().is_err());
    }

    #[test]
    fn paged_decode_runtime_allocates_exact_m1_hidden_on_cpu() {
        let mut context = RuntimeContext::create(0).unwrap();
        let decode = Qwen3Sq8PagedDecodeRuntime::allocate(&mut context).unwrap();
        assert_eq!(decode.config.sequence_len, 1);
        assert_eq!(decode.resident_hidden.size().unwrap(), 5120 * 4);
        assert!(decode.resident_hidden_buffer().is_err());
    }

    #[test]
    fn optimized_promotion_validates_m8_and_m128_dispatch_ids() {
        for m in [8, 128] {
            let config = Qwen3Sq8LayerConfig::qwen3_14b(m, 0).unwrap();
            let report = build_report(
                config,
                &"c".repeat(64),
                Sq8LayerExecutionProfile::Rdna4W8a8BlockCk,
                Sq8StackExecutionMode::SynchronizedResident,
                Sq8StackInputOrigin::PreviouslyUploadedResident,
                vec![ck_report(m); 40],
                0,
                1,
                0,
                false,
            )
            .unwrap();
            assert!(report.validate_optimized_promotion().is_ok());
        }

        let mut wrong_m128 = vec![ck_report(128); 40];
        wrong_m128[7].gate =
            Sq8LayerProjectionExecution::Ck(Sq8CkImplementation::MemV1KPaddingTile16x128x256);
        assert!(
            build_report(
                Qwen3Sq8LayerConfig::qwen3_14b(128, 0).unwrap(),
                &"d".repeat(64),
                Sq8LayerExecutionProfile::Rdna4W8a8BlockCk,
                Sq8StackExecutionMode::SynchronizedResident,
                Sq8StackInputOrigin::PreviouslyUploadedResident,
                wrong_m128,
                0,
                1,
                0,
                false,
            )
            .is_err()
        );
    }

    #[test]
    fn stack_artifact_hash_contract_is_lowercase_sha256() {
        assert!(validate_sha256(&"0".repeat(64)).is_ok());
        assert!(validate_sha256(&"f".repeat(64)).is_ok());
        assert!(validate_sha256(&"F".repeat(64)).is_err());
        assert!(validate_sha256(&"0".repeat(63)).is_err());
    }

    #[test]
    fn stack_state_keeps_prevalidation_errors_reusable() {
        let mut state = Sq8StackResidentState::OutputReady;
        let config = Qwen3Sq8LayerConfig::qwen3_14b(1, 0).unwrap();
        assert!(validate_host_input(&[], config).is_err());
        assert_eq!(state.status(), Sq8StackRuntimeStatus::OutputReady);
        assert!(state.ensure_usable().is_ok());
        state.mark_needs_input().unwrap();
        state.mark_input_ready().unwrap();
        assert_eq!(state.status(), Sq8StackRuntimeStatus::InputReady);
    }

    #[test]
    fn stack_state_rejects_completed_output_as_fresh_input() {
        let mut state = Sq8StackResidentState::InputReady;
        assert!(state.require_input_ready().is_ok());
        assert!(state.require_output_ready().is_err());
        state.mark_needs_input().unwrap();
        state.mark_output_ready().unwrap();
        assert_eq!(state.status(), Sq8StackRuntimeStatus::OutputReady);
        assert!(state.require_output_ready().is_ok());
        assert!(state.require_input_ready().is_err());
    }

    #[test]
    fn stack_state_poison_is_permanent() {
        let mut state = Sq8StackResidentState::NeedsInput;
        state.poison("stream synchronization failed".to_string());
        assert_eq!(state.status(), Sq8StackRuntimeStatus::Poisoned);
        assert_eq!(state.poison_reason(), Some("stream synchronization failed"));
        assert!(state.ensure_usable().is_err());
        assert!(state.mark_needs_input().is_err());
        assert!(state.mark_input_ready().is_err());
        assert!(state.mark_output_ready().is_err());
        state.poison("replacement reason".to_string());
        assert_eq!(state.poison_reason(), Some("stream synchronization failed"));
    }

    #[test]
    fn stack_uses_all_five_hip_only_guards() {
        assert_eq!(QWEN3_14B_SQ8_REQUIRED_HIP_KERNEL_ENV.len(), 5);
        assert_eq!(
            QWEN3_14B_SQ8_REQUIRED_HIP_KERNEL_ENV,
            [
                "ULLM_REQUIRE_HIP_RMSNORM_KERNEL",
                "ULLM_REQUIRE_HIP_ROPE_KERNEL",
                "ULLM_REQUIRE_HIP_CAUSAL_ATTN_KERNEL",
                "ULLM_REQUIRE_HIP_ADD_KERNEL",
                "ULLM_REQUIRE_HIP_SILU_MUL_KERNEL",
            ]
        );
    }
}
