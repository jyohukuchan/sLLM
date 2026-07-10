// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

use crate::loader::{
    PASSTHROUGH_MAX_STREAM_CHUNK_BYTES, PassthroughBf16ResidentData,
    load_named_passthrough_bf16_resident,
};
use crate::package::{PackageSummary, inspect_package};
use crate::sq8_layer_oracle::QWEN3_14B_HIDDEN_SIZE;
use crate::sq8_model_head_runtime::{
    QWEN3_14B_VOCAB_SIZE, validate_qwen3_14b_sq8_r9700_device_info,
};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use ullm_runtime_sys::{DeviceInfo, RuntimeBuffer, RuntimeContext, RuntimeStream, bf16_row_f32};

pub const QWEN3_14B_EMBED_TOKENS_TENSOR: &str = "model.embed_tokens.weight";
pub const QWEN3_14B_EMBED_TOKENS_SHAPE: [u64; 2] =
    [QWEN3_14B_VOCAB_SIZE as u64, QWEN3_14B_HIDDEN_SIZE as u64];
pub const QWEN3_14B_SQ8_EMBEDDING_REQUIRED_HIP_KERNEL_ENV: [&str; 1] =
    ["ULLM_REQUIRE_HIP_BF16_ROW_KERNEL"];

const BF16_DTYPE: &str = "BF16";
const PACKAGE_MANIFEST_FILE: &str = "manifest.json";
const MANIFEST_HASH_CHUNK_BYTES: usize = 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sq8EmbeddingRuntimeStatus {
    Ready,
    OutputEnqueued,
    OutputSynchronized,
    Poisoned,
}

#[derive(Debug)]
enum Sq8EmbeddingRuntimeState {
    Ready,
    OutputReady(Sq8EmbeddingExecutionReport),
    Poisoned(String),
}

impl Sq8EmbeddingRuntimeState {
    fn status(&self) -> Sq8EmbeddingRuntimeStatus {
        match self {
            Self::Ready => Sq8EmbeddingRuntimeStatus::Ready,
            Self::OutputReady(report) => match report.mode {
                Sq8EmbeddingExecutionMode::EnqueuedResident => {
                    Sq8EmbeddingRuntimeStatus::OutputEnqueued
                }
                Sq8EmbeddingExecutionMode::SynchronizedResident => {
                    Sq8EmbeddingRuntimeStatus::OutputSynchronized
                }
            },
            Self::Poisoned(_) => Sq8EmbeddingRuntimeStatus::Poisoned,
        }
    }

    fn ensure_usable(&self) -> Result<(), String> {
        match self {
            Self::Ready | Self::OutputReady(_) => Ok(()),
            Self::Poisoned(reason) => Err(format!(
                "Qwen3-14B SQ8 embedding runtime is permanently poisoned: {reason}"
            )),
        }
    }

    fn require_output(&self) -> Result<&Sq8EmbeddingExecutionReport, String> {
        self.ensure_usable()?;
        match self {
            Self::OutputReady(report) => Ok(report),
            Self::Ready => Err("Qwen3-14B SQ8 embedding runtime has no resident output".into()),
            Self::Poisoned(_) => unreachable!("poison was handled by ensure_usable"),
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
            Self::Ready | Self::OutputReady(_) => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sq8EmbeddingDeviceIdentity {
    pub device_id: i32,
    pub backend: String,
    pub name: String,
    pub gcn_arch_name: String,
    pub compute_major: i32,
    pub compute_minor: i32,
    pub total_global_mem: u64,
    pub flags: u32,
}

impl Sq8EmbeddingDeviceIdentity {
    fn from_runtime(value: DeviceInfo) -> Self {
        Self {
            device_id: value.device_id,
            backend: value.backend,
            name: value.name,
            gcn_arch_name: value.gcn_arch_name,
            compute_major: value.compute_major,
            compute_minor: value.compute_minor,
            total_global_mem: value.total_global_mem,
            flags: value.flags,
        }
    }

    fn validate_r9700(&self) -> Result<(), String> {
        validate_qwen3_14b_sq8_r9700_device_info(&DeviceInfo {
            device_id: self.device_id,
            backend: self.backend.clone(),
            name: self.name.clone(),
            total_global_mem: self.total_global_mem,
            compute_major: self.compute_major,
            compute_minor: self.compute_minor,
            gcn_arch_name: self.gcn_arch_name.clone(),
            flags: self.flags,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sq8EmbeddingPackageIdentity {
    pub canonical_package_dir: PathBuf,
    pub schema_version: Option<String>,
    pub source_model_dir: Option<String>,
    pub manifest_bytes: u64,
    pub manifest_sha256: String,
    pub quantized_tensors: usize,
    pub passthrough_tensors: usize,
    pub referenced_files: usize,
    pub referenced_file_bytes: u64,
    pub missing_referenced_files: usize,
}

impl Sq8EmbeddingPackageIdentity {
    pub fn validate_contract(&self) -> Result<(), String> {
        if !self.canonical_package_dir.is_absolute() {
            return Err(
                "SQ8 embedding package identity requires an absolute canonical path".into(),
            );
        }
        if self
            .schema_version
            .as_deref()
            .is_some_and(|value| value.is_empty())
        {
            return Err("SQ8 embedding package schema version must not be empty".into());
        }
        if self
            .source_model_dir
            .as_deref()
            .is_some_and(|value| value.is_empty())
        {
            return Err("SQ8 embedding package source model directory must not be empty".into());
        }
        if self.manifest_bytes == 0 {
            return Err("SQ8 embedding package manifest must not be empty".into());
        }
        validate_sha256(&self.manifest_sha256)
            .map_err(|err| format!("SQ8 embedding package manifest hash is invalid: {err}"))?;
        if self.passthrough_tensors == 0 || self.referenced_files == 0 {
            return Err(
                "SQ8 embedding package must contain passthrough tensors and referenced files"
                    .into(),
            );
        }
        if self.referenced_file_bytes == 0 || self.missing_referenced_files != 0 {
            return Err(format!(
                "SQ8 embedding package referenced-file contract failed: bytes={} missing={}",
                self.referenced_file_bytes, self.missing_referenced_files
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sq8EmbeddingPayloadIdentity {
    pub tensor_name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub elements: u64,
    pub payload_bytes: u64,
    pub payload_sha256: String,
    pub upload_chunks: u64,
}

impl Sq8EmbeddingPayloadIdentity {
    pub fn validate_contract(&self) -> Result<(), String> {
        if self.tensor_name != QWEN3_14B_EMBED_TOKENS_TENSOR {
            return Err(format!(
                "SQ8 embedding tensor name mismatch: expected={QWEN3_14B_EMBED_TOKENS_TENSOR} actual={}",
                self.tensor_name
            ));
        }
        if self.dtype != BF16_DTYPE {
            return Err(format!(
                "SQ8 embedding dtype mismatch: expected={BF16_DTYPE} actual={}",
                self.dtype
            ));
        }
        if self.shape != QWEN3_14B_EMBED_TOKENS_SHAPE {
            return Err(format!(
                "SQ8 embedding shape mismatch: expected={QWEN3_14B_EMBED_TOKENS_SHAPE:?} actual={:?}",
                self.shape
            ));
        }
        let expected_elements = embedding_elements()? as u64;
        let expected_bytes = bf16_bytes(embedding_elements()?)? as u64;
        if self.elements != expected_elements || self.payload_bytes != expected_bytes {
            return Err(format!(
                "SQ8 embedding payload size mismatch: expected_elements={expected_elements} actual_elements={} expected_bytes={expected_bytes} actual_bytes={}",
                self.elements, self.payload_bytes
            ));
        }
        validate_sha256(&self.payload_sha256)
            .map_err(|err| format!("SQ8 embedding payload hash is invalid: {err}"))?;
        if self.upload_chunks == 0 {
            return Err("SQ8 embedding payload must have at least one upload chunk".into());
        }
        Ok(())
    }
}

impl From<&PassthroughBf16ResidentData> for Sq8EmbeddingPayloadIdentity {
    fn from(value: &PassthroughBf16ResidentData) -> Self {
        Self {
            tensor_name: value.tensor_name.clone(),
            dtype: BF16_DTYPE.to_string(),
            shape: value.shape.clone(),
            elements: value.elements,
            payload_bytes: value.payload_bytes,
            payload_sha256: value.payload_sha256.clone(),
            upload_chunks: value.upload_chunks,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sq8EmbeddingLoadReport {
    pub package: Sq8EmbeddingPackageIdentity,
    pub payload: Sq8EmbeddingPayloadIdentity,
    pub requested_chunk_bytes: usize,
    pub maximum_staging_bytes: usize,
    pub upload_synchronization_count: u64,
    pub manifest_stability_checks: usize,
}

impl Sq8EmbeddingLoadReport {
    pub fn validate_contract(&self) -> Result<(), String> {
        self.package.validate_contract()?;
        self.payload.validate_contract()?;
        if self.requested_chunk_bytes == 0 {
            return Err("SQ8 embedding requested load chunk size must be nonzero".into());
        }
        let expected_staging =
            effective_load_chunk_bytes(self.payload.payload_bytes, self.requested_chunk_bytes)?;
        if self.maximum_staging_bytes != expected_staging
            || self.maximum_staging_bytes > PASSTHROUGH_MAX_STREAM_CHUNK_BYTES
        {
            return Err(format!(
                "SQ8 embedding bounded staging mismatch: expected={expected_staging} actual={} limit={PASSTHROUGH_MAX_STREAM_CHUNK_BYTES}",
                self.maximum_staging_bytes
            ));
        }
        let expected_chunks = self
            .payload
            .payload_bytes
            .div_ceil(self.maximum_staging_bytes as u64);
        if self.payload.upload_chunks != expected_chunks
            || self.upload_synchronization_count != expected_chunks
        {
            return Err(format!(
                "SQ8 embedding upload count mismatch: expected={expected_chunks} payload={} synchronizations={}",
                self.payload.upload_chunks, self.upload_synchronization_count
            ));
        }
        if self.manifest_stability_checks != 2 {
            return Err(format!(
                "SQ8 embedding package manifest must be stable across two checks, got {}",
                self.manifest_stability_checks
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sq8EmbeddingExecutionMode {
    EnqueuedResident,
    SynchronizedResident,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sq8EmbeddingExecutionReport {
    pub device: Sq8EmbeddingDeviceIdentity,
    pub load: Sq8EmbeddingLoadReport,
    pub mode: Sq8EmbeddingExecutionMode,
    pub token_id: usize,
    pub sequence_len: usize,
    pub output_elements: usize,
    pub kernel_guard_checks: usize,
    pub bf16_row_call_count: usize,
    pub execution_synchronization_count: usize,
    pub fallback_used: bool,
    pub host_staging_used: bool,
}

impl Sq8EmbeddingExecutionReport {
    pub fn validate_contract(&self) -> Result<(), String> {
        self.device.validate_r9700()?;
        self.load.validate_contract()?;
        validate_qwen3_14b_sq8_embedding_token_id(self.token_id)?;
        if self.sequence_len != 1 || self.output_elements != QWEN3_14B_HIDDEN_SIZE {
            return Err(format!(
                "SQ8 embedding output shape mismatch: expected=[1,{QWEN3_14B_HIDDEN_SIZE}] actual=[{},{}]",
                self.sequence_len, self.output_elements
            ));
        }
        if self.kernel_guard_checks != QWEN3_14B_SQ8_EMBEDDING_REQUIRED_HIP_KERNEL_ENV.len()
            || self.bf16_row_call_count != 1
        {
            return Err(format!(
                "SQ8 embedding operation count mismatch: guards={} bf16_row={}",
                self.kernel_guard_checks, self.bf16_row_call_count
            ));
        }
        let expected_synchronizations = match self.mode {
            Sq8EmbeddingExecutionMode::EnqueuedResident => 0,
            Sq8EmbeddingExecutionMode::SynchronizedResident => 1,
        };
        if self.execution_synchronization_count != expected_synchronizations {
            return Err(format!(
                "SQ8 embedding execution synchronization mismatch: expected={expected_synchronizations} actual={}",
                self.execution_synchronization_count
            ));
        }
        if self.fallback_used || self.host_staging_used {
            return Err("SQ8 embedding execution requires no fallback or host staging".into());
        }
        Ok(())
    }
}

/// Owns the exact Qwen3-14B BF16 token embedding matrix and one M=1 F32 output row.
///
/// Loading uses the shared bounded staging loader and verifies both the payload checksum and an
/// unchanged package manifest before/after upload. Inference requires the HIP BF16 row kernel;
/// it neither reads nor stages tensor data through the host. Any error after kernel submission
/// permanently poisons this instance.
#[derive(Debug)]
pub struct Qwen3Sq8EmbeddingRuntime {
    device: Sq8EmbeddingDeviceIdentity,
    load_report: Sq8EmbeddingLoadReport,
    embedding_bf16: RuntimeBuffer,
    output_f32: RuntimeBuffer,
    state: Sq8EmbeddingRuntimeState,
}

impl Qwen3Sq8EmbeddingRuntime {
    pub fn load(
        context: &mut RuntimeContext,
        stream: &mut RuntimeStream,
        package_path: impl AsRef<Path>,
        chunk_bytes: usize,
    ) -> Result<Self, String> {
        if chunk_bytes == 0 {
            return Err("Qwen3-14B SQ8 embedding chunk_bytes must be greater than zero".into());
        }
        let device_info = context.device_info()?;
        validate_qwen3_14b_sq8_r9700_device_info(&device_info)?;
        let device = Sq8EmbeddingDeviceIdentity::from_runtime(device_info);
        let package_before = inspect_embedding_package(package_path.as_ref())?;

        let load_result = (|| {
            let resident = load_named_passthrough_bf16_resident(
                context,
                stream,
                &package_before.canonical_package_dir,
                QWEN3_14B_EMBED_TOKENS_TENSOR,
                &QWEN3_14B_EMBED_TOKENS_SHAPE,
                chunk_bytes,
            )?;
            let payload = Sq8EmbeddingPayloadIdentity::from(&resident);
            let package_after = inspect_embedding_package(&package_before.canonical_package_dir)?;
            if package_after != package_before {
                return Err(format!(
                    "Qwen3-14B SQ8 embedding package changed during resident load: before_manifest={} after_manifest={}",
                    package_before.manifest_sha256, package_after.manifest_sha256
                ));
            }
            let load_report = Sq8EmbeddingLoadReport {
                package: package_before,
                payload,
                requested_chunk_bytes: chunk_bytes,
                maximum_staging_bytes: effective_load_chunk_bytes(
                    resident.payload_bytes,
                    chunk_bytes,
                )?,
                upload_synchronization_count: resident.upload_chunks,
                manifest_stability_checks: 2,
            };
            load_report.validate_contract()?;
            let output_f32 = context
                .alloc_buffer(f32_bytes(QWEN3_14B_HIDDEN_SIZE)?)
                .map_err(|err| {
                    format!("failed to allocate Qwen3-14B SQ8 embedding output: {err}")
                })?;
            validate_buffer_size(
                &resident.buffer,
                bf16_bytes(embedding_elements()?)?,
                "resident BF16 matrix",
            )?;
            validate_buffer_size(
                &output_f32,
                f32_bytes(QWEN3_14B_HIDDEN_SIZE)?,
                "M=1 F32 output",
            )?;
            Ok(Self {
                device,
                load_report,
                embedding_bf16: resident.buffer,
                output_f32,
                state: Sq8EmbeddingRuntimeState::Ready,
            })
        })();

        match load_result {
            Ok(runtime) => Ok(runtime),
            Err(operation_error) => Err(load_error_after_stream_recovery(stream, operation_error)),
        }
    }

    pub fn status(&self) -> Sq8EmbeddingRuntimeStatus {
        self.state.status()
    }

    pub fn device_identity(&self) -> &Sq8EmbeddingDeviceIdentity {
        &self.device
    }

    pub fn load_report(&self) -> &Sq8EmbeddingLoadReport {
        &self.load_report
    }

    pub fn poison_reason(&self) -> Option<&str> {
        self.state.poison_reason()
    }

    pub(crate) fn validate_serving_baseline(&self) -> Result<(), String> {
        self.validate_runtime_contract()?;
        if self.status() != Sq8EmbeddingRuntimeStatus::Ready {
            return Err(format!(
                "Qwen3-14B SQ8 serving embedding requires Ready, got {:?}",
                self.status()
            ));
        }
        Ok(())
    }

    pub(crate) fn validate_serving_preflight(&self) -> Result<(), String> {
        self.validate_serving_baseline()?;
        validate_hip_only_guards()
    }

    pub(crate) fn enqueue_serving_reset(
        &mut self,
        stream: &mut RuntimeStream,
    ) -> Result<(), String> {
        self.validate_runtime_contract()?;
        let bytes = self.output_f32.size().map_err(|err| {
            format!("failed to inspect Qwen3-14B SQ8 serving embedding output: {err}")
        })?;
        self.output_f32.zero(0, bytes, Some(stream)).map_err(|err| {
            format!("failed to enqueue Qwen3-14B SQ8 serving embedding reset: {err}")
        })
    }

    pub(crate) fn commit_serving_reset(&mut self) {
        self.state = Sq8EmbeddingRuntimeState::Ready;
    }

    /// Gathers one token embedding to the resident M=1 F32 output and synchronizes `stream`.
    /// No tensor value is copied to or from host memory.
    pub fn gather_token_synchronized(
        &mut self,
        token_id: usize,
        stream: &mut RuntimeStream,
    ) -> Result<Sq8EmbeddingExecutionReport, String> {
        let mut report = self.enqueue_token_resident(token_id, stream)?;
        if let Err(sync_error) = stream.synchronize() {
            return Err(self.poison_with_error(format!(
                "failed to synchronize Qwen3-14B SQ8 embedding gather: {sync_error}"
            )));
        }
        report.mode = Sq8EmbeddingExecutionMode::SynchronizedResident;
        report.execution_synchronization_count = 1;
        if let Err(err) = report.validate_contract() {
            return Err(self.poison_with_error(format!(
                "Qwen3-14B SQ8 synchronized embedding report validation failed: {err}"
            )));
        }
        self.state = Sq8EmbeddingRuntimeState::OutputReady(report.clone());
        Ok(report)
    }

    /// Enqueues one resident gather for use by the in-crate generation loop on the same stream.
    pub(crate) fn enqueue_token_resident(
        &mut self,
        token_id: usize,
        stream: &mut RuntimeStream,
    ) -> Result<Sq8EmbeddingExecutionReport, String> {
        self.validate_execution_preconditions(token_id)?;
        if let Err(operation_error) = bf16_row_f32(
            &self.embedding_bf16,
            QWEN3_14B_VOCAB_SIZE,
            QWEN3_14B_HIDDEN_SIZE,
            token_id,
            &mut self.output_f32,
            Some(&mut *stream),
        ) {
            return Err(self.poison_after_stream_recovery(stream, operation_error));
        }
        let report = Sq8EmbeddingExecutionReport {
            device: self.device.clone(),
            load: self.load_report.clone(),
            mode: Sq8EmbeddingExecutionMode::EnqueuedResident,
            token_id,
            sequence_len: 1,
            output_elements: QWEN3_14B_HIDDEN_SIZE,
            kernel_guard_checks: QWEN3_14B_SQ8_EMBEDDING_REQUIRED_HIP_KERNEL_ENV.len(),
            bf16_row_call_count: 1,
            execution_synchronization_count: 0,
            fallback_used: false,
            host_staging_used: false,
        };
        if let Err(err) = report.validate_contract() {
            return Err(self.poison_after_stream_recovery(
                stream,
                format!("Qwen3-14B SQ8 embedding report validation failed: {err}"),
            ));
        }
        self.state = Sq8EmbeddingRuntimeState::OutputReady(report.clone());
        Ok(report)
    }

    /// Returns the resident M=1 F32 row and the report for the gather that produced it.
    /// An enqueued output is only ordered for consumers submitted to the same stream.
    pub fn resident_output(
        &self,
    ) -> Result<(&RuntimeBuffer, &Sq8EmbeddingExecutionReport), String> {
        Ok((&self.output_f32, self.state.require_output()?))
    }

    fn validate_runtime_contract(&self) -> Result<(), String> {
        self.state.ensure_usable()?;
        self.device.validate_r9700()?;
        self.load_report.validate_contract()?;
        validate_buffer_size(
            &self.embedding_bf16,
            bf16_bytes(embedding_elements()?)?,
            "resident BF16 matrix",
        )?;
        validate_buffer_size(
            &self.output_f32,
            f32_bytes(QWEN3_14B_HIDDEN_SIZE)?,
            "M=1 F32 output",
        )
    }

    fn validate_execution_preconditions(&self, token_id: usize) -> Result<(), String> {
        self.validate_runtime_contract()?;
        validate_qwen3_14b_sq8_embedding_token_id(token_id)?;
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
                "{operation_error}; subsequent Qwen3-14B SQ8 embedding stream recovery failed: {sync_error}"
            ),
        };
        self.poison_with_error(error)
    }

    fn poison_with_error(&mut self, error: String) -> String {
        self.state.poison(error.clone());
        error
    }
}

pub fn validate_qwen3_14b_sq8_embedding_token_id(token_id: usize) -> Result<(), String> {
    if token_id >= QWEN3_14B_VOCAB_SIZE {
        return Err(format!(
            "Qwen3-14B SQ8 embedding token id {token_id} is out of range 0..{QWEN3_14B_VOCAB_SIZE}"
        ));
    }
    Ok(())
}

fn inspect_embedding_package(package_path: &Path) -> Result<Sq8EmbeddingPackageIdentity, String> {
    let canonical_package_dir = fs::canonicalize(package_path).map_err(|err| {
        format!(
            "failed to canonicalize Qwen3-14B SQ8 embedding package {}: {err}",
            package_path.display()
        )
    })?;
    if !fs::metadata(&canonical_package_dir)
        .map_err(|err| {
            format!(
                "failed to inspect Qwen3-14B SQ8 embedding package {}: {err}",
                canonical_package_dir.display()
            )
        })?
        .is_dir()
    {
        return Err(format!(
            "Qwen3-14B SQ8 embedding package is not a directory: {}",
            canonical_package_dir.display()
        ));
    }
    let summary = inspect_package(&canonical_package_dir)?;
    let manifest_path = canonical_manifest_path(&canonical_package_dir)?;
    let (manifest_bytes, manifest_sha256) = sha256_regular_file(&manifest_path)?;
    let identity = package_identity_from_summary(
        canonical_package_dir,
        summary,
        manifest_bytes,
        manifest_sha256,
    );
    identity.validate_contract()?;
    Ok(identity)
}

fn canonical_manifest_path(package_dir: &Path) -> Result<PathBuf, String> {
    let manifest = fs::canonicalize(package_dir.join(PACKAGE_MANIFEST_FILE))
        .map_err(|err| format!("failed to canonicalize Qwen3-14B SQ8 package manifest: {err}"))?;
    if !manifest.starts_with(package_dir) {
        return Err(format!(
            "Qwen3-14B SQ8 package manifest escapes package root: {}",
            manifest.display()
        ));
    }
    Ok(manifest)
}

fn package_identity_from_summary(
    canonical_package_dir: PathBuf,
    summary: PackageSummary,
    manifest_bytes: u64,
    manifest_sha256: String,
) -> Sq8EmbeddingPackageIdentity {
    Sq8EmbeddingPackageIdentity {
        canonical_package_dir,
        schema_version: summary.schema_version,
        source_model_dir: summary.source_model_dir,
        manifest_bytes,
        manifest_sha256,
        quantized_tensors: summary.quantized_tensors,
        passthrough_tensors: summary.passthrough_tensors,
        referenced_files: summary.referenced_files,
        referenced_file_bytes: summary.referenced_file_bytes,
        missing_referenced_files: summary.missing_referenced_files,
    }
}

fn sha256_regular_file(path: &Path) -> Result<(u64, String), String> {
    let mut file = File::open(path)
        .map_err(|err| format!("failed to open {} for SHA-256: {err}", path.display()))?;
    let opened_metadata = file
        .metadata()
        .map_err(|err| format!("failed to stat {} for SHA-256: {err}", path.display()))?;
    if !opened_metadata.is_file() || opened_metadata.len() == 0 {
        return Err(format!(
            "SHA-256 input must be a non-empty regular file: {}",
            path.display()
        ));
    }
    let opened_bytes = opened_metadata.len();
    let mut buffer = vec![0_u8; MANIFEST_HASH_CHUNK_BYTES];
    let mut digest = Sha256::new();
    let mut read_bytes = 0_u64;
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|err| format!("failed to hash {}: {err}", path.display()))?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
        read_bytes = read_bytes
            .checked_add(count as u64)
            .ok_or_else(|| format!("SHA-256 byte count overflows for {}", path.display()))?;
    }
    let final_bytes = file
        .metadata()
        .map_err(|err| format!("failed to re-stat {} after SHA-256: {err}", path.display()))?
        .len();
    if read_bytes != opened_bytes || final_bytes != opened_bytes {
        return Err(format!(
            "file changed while hashing {}: opened={opened_bytes} read={read_bytes} final={final_bytes}",
            path.display()
        ));
    }
    Ok((read_bytes, format!("{:x}", digest.finalize())))
}

fn validate_hip_only_guards() -> Result<(), String> {
    for name in QWEN3_14B_SQ8_EMBEDDING_REQUIRED_HIP_KERNEL_ENV {
        if std::env::var_os(name).is_none() {
            return Err(format!(
                "Qwen3-14B SQ8 embedding execution requires {name}=1 to forbid HIP host-staging fallback"
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
        .map_err(|err| format!("failed to inspect Qwen3-14B SQ8 embedding {label}: {err}"))?;
    if actual_bytes != expected_bytes {
        return Err(format!(
            "Qwen3-14B SQ8 embedding {label} buffer size mismatch: expected={expected_bytes} actual={actual_bytes}"
        ));
    }
    Ok(())
}

fn embedding_elements() -> Result<usize, String> {
    QWEN3_14B_VOCAB_SIZE
        .checked_mul(QWEN3_14B_HIDDEN_SIZE)
        .ok_or_else(|| "Qwen3-14B SQ8 embedding element count overflows".into())
}

fn effective_load_chunk_bytes(payload_bytes: u64, requested: usize) -> Result<usize, String> {
    if payload_bytes == 0 || requested == 0 {
        return Err("SQ8 embedding payload and requested chunk sizes must be nonzero".into());
    }
    usize::try_from(payload_bytes.min(requested.min(PASSTHROUGH_MAX_STREAM_CHUNK_BYTES) as u64))
        .map_err(|_| "SQ8 embedding effective chunk size does not fit usize".into())
}

fn f32_bytes(elements: usize) -> Result<usize, String> {
    elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "Qwen3-14B SQ8 embedding F32 byte size overflows".into())
}

fn bf16_bytes(elements: usize) -> Result<usize, String> {
    elements
        .checked_mul(std::mem::size_of::<u16>())
        .ok_or_else(|| "Qwen3-14B SQ8 embedding BF16 byte size overflows".into())
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

fn load_error_after_stream_recovery(stream: &mut RuntimeStream, operation_error: String) -> String {
    match stream.synchronize() {
        Ok(()) => operation_error,
        Err(sync_error) => format!(
            "{operation_error}; subsequent Qwen3-14B SQ8 embedding load recovery failed: {sync_error}"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r9700_identity() -> Sq8EmbeddingDeviceIdentity {
        Sq8EmbeddingDeviceIdentity {
            device_id: 0,
            backend: "hip".to_string(),
            name: "AMD Radeon Graphics".to_string(),
            gcn_arch_name: "gfx1201".to_string(),
            compute_major: 12,
            compute_minor: 0,
            total_global_mem: 34_208_743_424,
            flags: 0,
        }
    }

    fn package_identity() -> Sq8EmbeddingPackageIdentity {
        Sq8EmbeddingPackageIdentity {
            canonical_package_dir: PathBuf::from("/tmp/qwen3-14b-thin.ullm.d"),
            schema_version: Some("ullm-prototype-manifest-v0.1".to_string()),
            source_model_dir: Some("/models/Qwen3-14B-FP8".to_string()),
            manifest_bytes: 100,
            manifest_sha256: "1".repeat(64),
            quantized_tensors: 0,
            passthrough_tensors: 3,
            referenced_files: 3,
            referenced_file_bytes: 3_000_000_000,
            missing_referenced_files: 0,
        }
    }

    fn payload_identity(chunk_bytes: usize) -> Sq8EmbeddingPayloadIdentity {
        let elements = embedding_elements().unwrap();
        let payload_bytes = bf16_bytes(elements).unwrap() as u64;
        Sq8EmbeddingPayloadIdentity {
            tensor_name: QWEN3_14B_EMBED_TOKENS_TENSOR.to_string(),
            dtype: BF16_DTYPE.to_string(),
            shape: QWEN3_14B_EMBED_TOKENS_SHAPE.to_vec(),
            elements: elements as u64,
            payload_bytes,
            payload_sha256: "2".repeat(64),
            upload_chunks: payload_bytes.div_ceil(chunk_bytes as u64),
        }
    }

    fn load_report(chunk_bytes: usize) -> Sq8EmbeddingLoadReport {
        let payload = payload_identity(chunk_bytes);
        Sq8EmbeddingLoadReport {
            package: package_identity(),
            requested_chunk_bytes: chunk_bytes,
            maximum_staging_bytes: chunk_bytes,
            upload_synchronization_count: payload.upload_chunks,
            payload,
            manifest_stability_checks: 2,
        }
    }

    fn execution_report(mode: Sq8EmbeddingExecutionMode) -> Sq8EmbeddingExecutionReport {
        Sq8EmbeddingExecutionReport {
            device: r9700_identity(),
            load: load_report(1024 * 1024),
            mode,
            token_id: 353,
            sequence_len: 1,
            output_elements: QWEN3_14B_HIDDEN_SIZE,
            kernel_guard_checks: QWEN3_14B_SQ8_EMBEDDING_REQUIRED_HIP_KERNEL_ENV.len(),
            bf16_row_call_count: 1,
            execution_synchronization_count: usize::from(matches!(
                mode,
                Sq8EmbeddingExecutionMode::SynchronizedResident
            )),
            fallback_used: false,
            host_staging_used: false,
        }
    }

    #[test]
    fn token_id_validation_enforces_fixed_vocab() {
        validate_qwen3_14b_sq8_embedding_token_id(0).unwrap();
        validate_qwen3_14b_sq8_embedding_token_id(QWEN3_14B_VOCAB_SIZE - 1).unwrap();
        assert!(validate_qwen3_14b_sq8_embedding_token_id(QWEN3_14B_VOCAB_SIZE).is_err());
        assert!(validate_qwen3_14b_sq8_embedding_token_id(usize::MAX).is_err());
    }

    #[test]
    fn payload_contract_is_exact_qwen3_14b_bf16() {
        let identity = payload_identity(1024 * 1024);
        identity.validate_contract().unwrap();

        let mut bad = identity.clone();
        bad.tensor_name = "model.embedding.weight".into();
        assert!(bad.validate_contract().is_err());

        let mut bad = identity.clone();
        bad.dtype = "F16".into();
        assert!(bad.validate_contract().is_err());

        let mut bad = identity.clone();
        bad.shape[0] -= 1;
        assert!(bad.validate_contract().is_err());

        let mut bad = identity;
        bad.payload_sha256 = "A".repeat(64);
        assert!(bad.validate_contract().is_err());
    }

    #[test]
    fn load_report_binds_package_payload_and_bounded_staging() {
        let report = load_report(1024 * 1024);
        report.validate_contract().unwrap();

        let mut bad = report.clone();
        bad.package.manifest_sha256 = "3".repeat(63);
        assert!(bad.validate_contract().is_err());

        let mut bad = report.clone();
        bad.payload.upload_chunks -= 1;
        assert!(bad.validate_contract().is_err());

        let mut bad = report;
        bad.maximum_staging_bytes = PASSTHROUGH_MAX_STREAM_CHUNK_BYTES + 1;
        assert!(bad.validate_contract().is_err());
    }

    #[test]
    fn execution_report_requires_m1_kernel_only_no_staging() {
        for mode in [
            Sq8EmbeddingExecutionMode::EnqueuedResident,
            Sq8EmbeddingExecutionMode::SynchronizedResident,
        ] {
            execution_report(mode).validate_contract().unwrap();
        }

        let report = execution_report(Sq8EmbeddingExecutionMode::EnqueuedResident);
        let mut bad = report.clone();
        bad.sequence_len = 2;
        assert!(bad.validate_contract().is_err());

        let mut bad = report.clone();
        bad.execution_synchronization_count = 1;
        assert!(bad.validate_contract().is_err());

        let mut bad = report.clone();
        bad.fallback_used = true;
        assert!(bad.validate_contract().is_err());

        let mut bad = report;
        bad.host_staging_used = true;
        assert!(bad.validate_contract().is_err());
    }

    #[test]
    fn chunk_size_is_bounded_and_payload_limited() {
        assert_eq!(effective_load_chunk_bytes(8, 1024).unwrap(), 8);
        assert_eq!(
            effective_load_chunk_bytes(u64::MAX, usize::MAX).unwrap(),
            PASSTHROUGH_MAX_STREAM_CHUNK_BYTES
        );
        assert!(effective_load_chunk_bytes(0, 1).is_err());
        assert!(effective_load_chunk_bytes(1, 0).is_err());
    }

    #[test]
    fn poison_state_is_permanent() {
        let mut state = Sq8EmbeddingRuntimeState::Ready;
        state.poison("kernel failure".to_string());
        assert_eq!(state.status(), Sq8EmbeddingRuntimeStatus::Poisoned);
        assert_eq!(state.poison_reason(), Some("kernel failure"));
        assert!(state.ensure_usable().is_err());
        state.poison("replacement".to_string());
        assert_eq!(state.poison_reason(), Some("kernel failure"));
    }
}
