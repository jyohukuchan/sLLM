// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Resident AQ4/SQ8 package projection runtime shared by serving and CLI frontends.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::env;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::backend_dispatch::{
    BackendRequest, SQ8_0_PROJECTION_DISPATCH_PHASE, Sq8ProjectionFamily,
    Sq8ProjectionMatvecOperation as SqFp8ProjectionMatvecOperation,
    select_sq8_projection_implementation_id as select_backend_sq8_projection_implementation_id,
    sq8_0_projection_descriptor_family,
};
use crate::format_id::FORMAT_SQ8_0;
use crate::host_bytes::{decode_f32_le_values, encode_f32_to_bytes};
use crate::loader::{LoadOptions, WeightRegistry, materialize_config, matrix_shape_rows_cols};
use crate::package::{TensorSelector, select_tensor_payload_bundle};
use crate::qwen3_loader::Qwen3PackageSqOverlay;
use crate::sq::fp8_e4m3fn_to_f32;
use crate::sq_runtime::{Sq8ResidentRuntimeTensorRef, load_sq8_resident_tensor};
use crate::sq8_model_head_runtime::validate_qwen3_14b_sq8_r9700_device_info;

#[derive(Clone, Copy, Debug, Default)]
pub struct SqFp8ProjectionTelemetry {
    pub single_matvec_count: u64,
    pub batch_matvec_count: u64,
    pub pair_matvec_count: u64,
    pub triple_matvec_count: u64,
}

static SQ_FP8_SINGLE_MATVEC_COUNT: AtomicU64 = AtomicU64::new(0);
static SQ_FP8_BATCH_MATVEC_COUNT: AtomicU64 = AtomicU64::new(0);
static SQ_FP8_PAIR_MATVEC_COUNT: AtomicU64 = AtomicU64::new(0);
static SQ_FP8_TRIPLE_MATVEC_COUNT: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug)]
pub struct SqFp8ProjectionDispatch {
    pub operation: SqFp8ProjectionMatvecOperation,
    pub implementation_id: &'static str,
    pub family: Option<Sq8ProjectionFamily>,
}

impl SqFp8ProjectionDispatch {
    pub fn label(&self) -> &'static str {
        self.operation.label()
    }

    pub fn require_direct_family(&self, label: &str) -> Result<(), String> {
        match self.family {
            Some(Sq8ProjectionFamily::Direct) => Ok(()),
            None => Err(format!(
                "{label} SQ8_0 projection dispatch has no direct kernel family: operation={} implementation_id={}",
                self.operation.label(),
                self.implementation_id
            )),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SqFp8ProjectionDispatches {
    pub single: SqFp8ProjectionDispatch,
    pub batch: SqFp8ProjectionDispatch,
    pub pair: SqFp8ProjectionDispatch,
    pub triple: SqFp8ProjectionDispatch,
}

pub const SQ8_0_MODEL_ARCH_QWEN_FAMILY: &str = "Qwen3";

impl SqFp8ProjectionDispatches {
    pub fn from_info(info: &ullm_runtime_sys::DeviceInfo, model_arch: Option<&str>) -> Self {
        Self {
            single: sq_fp8_projection_dispatch(
                SqFp8ProjectionMatvecOperation::Single,
                info,
                model_arch,
            ),
            batch: sq_fp8_projection_dispatch(
                SqFp8ProjectionMatvecOperation::Batch,
                info,
                model_arch,
            ),
            pair: sq_fp8_projection_dispatch(
                SqFp8ProjectionMatvecOperation::Pair,
                info,
                model_arch,
            ),
            triple: sq_fp8_projection_dispatch(
                SqFp8ProjectionMatvecOperation::Triple,
                info,
                model_arch,
            ),
        }
    }

    fn for_operation(&self, operation: SqFp8ProjectionMatvecOperation) -> SqFp8ProjectionDispatch {
        match operation {
            SqFp8ProjectionMatvecOperation::Single => self.single,
            SqFp8ProjectionMatvecOperation::Batch => self.batch,
            SqFp8ProjectionMatvecOperation::Pair => self.pair,
            SqFp8ProjectionMatvecOperation::Triple => self.triple,
        }
    }
}

pub fn reset_sq_fp8_projection_telemetry() {
    SQ_FP8_SINGLE_MATVEC_COUNT.store(0, Ordering::Relaxed);
    SQ_FP8_BATCH_MATVEC_COUNT.store(0, Ordering::Relaxed);
    SQ_FP8_PAIR_MATVEC_COUNT.store(0, Ordering::Relaxed);
    SQ_FP8_TRIPLE_MATVEC_COUNT.store(0, Ordering::Relaxed);
}

pub fn snapshot_sq_fp8_projection_telemetry() -> SqFp8ProjectionTelemetry {
    SqFp8ProjectionTelemetry {
        single_matvec_count: SQ_FP8_SINGLE_MATVEC_COUNT.load(Ordering::Relaxed),
        batch_matvec_count: SQ_FP8_BATCH_MATVEC_COUNT.load(Ordering::Relaxed),
        pair_matvec_count: SQ_FP8_PAIR_MATVEC_COUNT.load(Ordering::Relaxed),
        triple_matvec_count: SQ_FP8_TRIPLE_MATVEC_COUNT.load(Ordering::Relaxed),
    }
}

fn record_sq_fp8_projection_dispatch(dispatch: SqFp8ProjectionDispatch) {
    match dispatch.operation {
        SqFp8ProjectionMatvecOperation::Single => {
            SQ_FP8_SINGLE_MATVEC_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        SqFp8ProjectionMatvecOperation::Batch => {
            SQ_FP8_BATCH_MATVEC_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        SqFp8ProjectionMatvecOperation::Pair => {
            SQ_FP8_PAIR_MATVEC_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        SqFp8ProjectionMatvecOperation::Triple => {
            SQ_FP8_TRIPLE_MATVEC_COUNT.fetch_add(1, Ordering::Relaxed);
        }
    }
}

fn runtime_device_gpu_arch(info: &ullm_runtime_sys::DeviceInfo) -> Option<&'static str> {
    let gcn_arch = info.gcn_arch_name.to_ascii_lowercase();
    let name = info.name.to_ascii_lowercase();
    if info.compute_major == 12 || gcn_arch.starts_with("gfx12") || name.contains("r9700") {
        Some("RDNA4")
    } else if info.compute_major == 10 || gcn_arch.starts_with("gfx10") || name.contains("v620") {
        Some("RDNA2")
    } else {
        None
    }
}

pub fn dispatchable_sq8_projection_gpu_name(info: &ullm_runtime_sys::DeviceInfo) -> Cow<'_, str> {
    if validate_qwen3_14b_sq8_r9700_device_info(info).is_ok() {
        Cow::Borrowed("Radeon_AI_PRO_R9700")
    } else {
        Cow::Borrowed(info.name.as_str())
    }
}

fn sq_fp8_projection_dispatch(
    operation: SqFp8ProjectionMatvecOperation,
    info: &ullm_runtime_sys::DeviceInfo,
    model_arch: Option<&str>,
) -> SqFp8ProjectionDispatch {
    let gpu_name = dispatchable_sq8_projection_gpu_name(info);
    let request = BackendRequest {
        operation: operation.operation_id(),
        phase: SQ8_0_PROJECTION_DISPATCH_PHASE,
        format_id: Some(FORMAT_SQ8_0),
        model_arch,
        gpu_arch: runtime_device_gpu_arch(info),
        gpu_name: Some(gpu_name.as_ref()),
    };
    let implementation_id = select_backend_sq8_projection_implementation_id(&request);
    SqFp8ProjectionDispatch {
        operation,
        implementation_id,
        family: sq8_0_projection_descriptor_family(implementation_id),
    }
}

pub fn select_sq_fp8_projection_implementation_id(
    operation: SqFp8ProjectionMatvecOperation,
    info: &ullm_runtime_sys::DeviceInfo,
    model_arch: Option<&str>,
) -> &'static str {
    sq_fp8_projection_dispatch(operation, info, model_arch).implementation_id
}

fn checked_f32_byte_len(elements: usize, label: &str) -> Result<usize, String> {
    elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| format!("{label} byte size overflows"))
}

pub fn package_aq4_f32_allocation_bytes(elements: u64) -> Result<u64, String> {
    elements
        .checked_mul(std::mem::size_of::<f32>() as u64)
        .ok_or_else(|| "AQ4 f32 allocation bytes overflow".to_string())
}

fn read_runtime_buffer_f32(
    buffer: &ullm_runtime_sys::RuntimeBuffer,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    elements: usize,
    label: &str,
) -> Result<Vec<f32>, String> {
    let mut bytes = vec![0_u8; checked_f32_byte_len(elements, label)?];
    buffer
        .copy_to_host(0, &mut bytes, Some(stream))
        .map_err(|err| format!("failed to copy {label} from runtime: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize {label} runtime copy: {err}"))?;
    Ok(decode_f32_le_values(&bytes))
}

fn env_flag_enabled(name: &str) -> bool {
    env::var(name)
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

pub struct PackageAq4ResidentMatvec {
    pub rows: usize,
    pub cols: usize,
    pub group_size: usize,
    pub tensor_scale: f32,
    pub scale_count: usize,
    row_scale_count: usize,
    projection_dispatches: SqFp8ProjectionDispatches,
    storage: PackageResidentMatvecStorage,
}

enum PackageResidentMatvecStorage {
    Aq4 {
        index_buffer: Arc<ullm_runtime_sys::RuntimeBuffer>,
        scale_buffer: Arc<ullm_runtime_sys::RuntimeBuffer>,
        codebook_buffer: Arc<ullm_runtime_sys::RuntimeBuffer>,
        scale_values_buffer: Arc<ullm_runtime_sys::RuntimeBuffer>,
        row_scale_buffer: Option<Arc<ullm_runtime_sys::RuntimeBuffer>>,
    },
    #[allow(dead_code)]
    F32 {
        matrix_buffer: Arc<ullm_runtime_sys::RuntimeBuffer>,
    },
    SqFp8 {
        payload_buffer: Arc<ullm_runtime_sys::RuntimeBuffer>,
        scale_buffer: Arc<ullm_runtime_sys::RuntimeBuffer>,
        scale_kind: u32,
        scale_block_cols: usize,
    },
}

struct PackageAq4StorageRef<'a> {
    index_buffer: &'a ullm_runtime_sys::RuntimeBuffer,
    scale_buffer: &'a ullm_runtime_sys::RuntimeBuffer,
    codebook_buffer: &'a ullm_runtime_sys::RuntimeBuffer,
    scale_values_buffer: &'a ullm_runtime_sys::RuntimeBuffer,
    row_scale_buffer: Option<&'a ullm_runtime_sys::RuntimeBuffer>,
}

#[derive(Default)]
pub struct PackageResidentSharedBufferRegistry {
    buffers: BTreeMap<String, Arc<ullm_runtime_sys::RuntimeBuffer>>,
}

impl PackageResidentSharedBufferRegistry {
    pub fn new() -> Self {
        Self {
            buffers: BTreeMap::new(),
        }
    }

    fn f32_buffer(
        &mut self,
        context: &mut ullm_runtime_sys::RuntimeContext,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        key: String,
        values: &[f32],
        label: &str,
    ) -> Result<Arc<ullm_runtime_sys::RuntimeBuffer>, String> {
        if let Some(buffer) = self.buffers.get(&key) {
            return Ok(buffer.clone());
        }
        let mut buffer = context
            .alloc_buffer(checked_f32_byte_len(values.len(), label)?)
            .map_err(|err| format!("failed to allocate shared {label}: {err}"))?;
        buffer
            .copy_from_host(0, &encode_f32_to_bytes(values), Some(stream))
            .map_err(|err| format!("failed to copy shared {label}: {err}"))?;
        let buffer = Arc::new(buffer);
        self.buffers.insert(key, buffer.clone());
        Ok(buffer)
    }
}

pub fn package_resident_f32_buffer(
    context: &mut ullm_runtime_sys::RuntimeContext,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    shared_buffers: &mut Option<&mut PackageResidentSharedBufferRegistry>,
    key: String,
    values: &[f32],
    label: &str,
) -> Result<Arc<ullm_runtime_sys::RuntimeBuffer>, String> {
    if let Some(shared) = shared_buffers.as_mut() {
        return shared.f32_buffer(context, stream, key, values, label);
    }
    let mut buffer = context
        .alloc_buffer(checked_f32_byte_len(values.len(), label)?)
        .map_err(|err| format!("failed to allocate {label}: {err}"))?;
    buffer
        .copy_from_host(0, &encode_f32_to_bytes(values), Some(stream))
        .map_err(|err| format!("failed to copy {label}: {err}"))?;
    Ok(Arc::new(buffer))
}

impl PackageAq4ResidentMatvec {
    pub fn load(
        context: &mut ullm_runtime_sys::RuntimeContext,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        registry: &mut WeightRegistry,
        path: &str,
        tensor_name: &str,
        chunk_bytes: usize,
    ) -> Result<Self, String> {
        let projection_dispatches = SqFp8ProjectionDispatches::from_info(
            &context
                .device_info()
                .map_err(|err| format!("failed to query runtime context device: {err}"))?,
            None,
        );
        Self::load_with_shared_buffers(
            context,
            stream,
            registry,
            None,
            path,
            tensor_name,
            chunk_bytes,
            projection_dispatches,
        )
    }

    fn load_with_shared_buffers(
        context: &mut ullm_runtime_sys::RuntimeContext,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        registry: &mut WeightRegistry,
        mut shared_buffers: Option<&mut PackageResidentSharedBufferRegistry>,
        path: &str,
        tensor_name: &str,
        chunk_bytes: usize,
        projection_dispatches: SqFp8ProjectionDispatches,
    ) -> Result<Self, String> {
        let selector = TensorSelector::Name(tensor_name.to_string());
        let bundle = select_tensor_payload_bundle(path, &selector)
            .map_err(|err| format!("failed to select tensor payloads for {tensor_name}: {err}"))?;
        let registry_index = if let Some((existing_index, _)) = registry
            .iter()
            .enumerate()
            .find(|(_, bundle)| bundle.tensor_name == tensor_name)
        {
            existing_index
        } else {
            registry
                .load_and_insert(
                    context,
                    stream,
                    &bundle,
                    LoadOptions {
                        chunk_bytes,
                        verify: true,
                    },
                )
                .map_err(|err| {
                    format!("failed to register tensor payloads for {tensor_name}: {err}")
                })?
        };
        let loaded = registry
            .get(registry_index)
            .ok_or_else(|| "registered tensor disappeared from weight registry".to_string())?;
        let materialize = materialize_config(loaded).map_err(|err| {
            format!(
                "failed to prepare AQ4 matvec config for {tensor_name} (registry index {registry_index}): {err}"
            )
        })?;
        let (rows, cols) = matrix_shape_rows_cols(&loaded.shape, materialize.elements)
            .map_err(|err| format!("invalid shape for {tensor_name}: {err}"))?;
        let scale_values_buffer = if let Some(shared) = shared_buffers.as_mut() {
            shared.f32_buffer(
                context,
                stream,
                format!("aq4-scale-values:{tensor_name}"),
                &materialize.scale_values,
                &format!("AQ4 scale values for {tensor_name}"),
            )?
        } else {
            let scale_bytes = usize::try_from(package_aq4_f32_allocation_bytes(
                u64::try_from(materialize.scale_values.len())
                    .map_err(|_| "AQ4 scale count does not fit u64".to_string())?,
            )?)
            .map_err(|_| "AQ4 scale allocation bytes do not fit usize".to_string())?;
            let mut buffer = context.alloc_buffer(scale_bytes).map_err(|err| {
                format!("failed to allocate AQ4 scale values for {tensor_name}: {err}")
            })?;
            buffer
                .copy_from_host(
                    0,
                    &encode_f32_to_bytes(&materialize.scale_values),
                    Some(stream),
                )
                .map_err(|err| {
                    format!("failed to copy AQ4 scale values for {tensor_name}: {err}")
                })?;
            Arc::new(buffer)
        };

        let mut row_scale_buffer = None;
        if !bundle.row_scale_overrides.is_empty() {
            let mut row_scales = vec![1.0_f32; rows];
            for entry in &bundle.row_scale_overrides {
                if entry.row_index >= rows || !entry.scale.is_finite() {
                    return Err(format!(
                        "invalid row scale override for {tensor_name} row {} scale {}",
                        entry.row_index, entry.scale
                    ));
                }
                row_scales[entry.row_index] *= entry.scale;
            }
            row_scale_buffer = Some(if let Some(shared) = shared_buffers.as_mut() {
                shared.f32_buffer(
                    context,
                    stream,
                    format!("aq4-row-scale:{tensor_name}"),
                    &row_scales,
                    &format!("AQ4 row scales for {tensor_name}"),
                )?
            } else {
                let row_scale_bytes = usize::try_from(package_aq4_f32_allocation_bytes(
                    u64::try_from(rows)
                        .map_err(|_| "AQ4 row-scale count does not fit u64".to_string())?,
                )?)
                .map_err(|_| "AQ4 row-scale allocation bytes do not fit usize".to_string())?;
                let mut buffer = context.alloc_buffer(row_scale_bytes).map_err(|err| {
                    format!("failed to allocate row scale buffer for {tensor_name}: {err}")
                })?;
                buffer
                    .copy_from_host(0, &encode_f32_to_bytes(&row_scales), Some(stream))
                    .map_err(|err| format!("failed to copy row scales for {tensor_name}: {err}"))?;
                Arc::new(buffer)
            });
        }

        Ok(Self {
            rows,
            cols,
            group_size: materialize.group_size,
            tensor_scale: materialize.tensor_scale,
            scale_count: materialize.scale_values.len(),
            row_scale_count: if row_scale_buffer.is_some() { rows } else { 0 },
            projection_dispatches,
            storage: PackageResidentMatvecStorage::Aq4 {
                index_buffer: loaded.index.buffer.clone(),
                scale_buffer: loaded.scale.buffer.clone(),
                codebook_buffer: loaded.codebook.buffer.clone(),
                scale_values_buffer,
                row_scale_buffer,
            },
        })
    }

    pub fn load_with_sq_overlay(
        context: &mut ullm_runtime_sys::RuntimeContext,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        registry: &mut WeightRegistry,
        shared_buffers: Option<&mut PackageResidentSharedBufferRegistry>,
        path: &str,
        tensor_name: &str,
        chunk_bytes: usize,
        sq_overlay: Option<&Qwen3PackageSqOverlay<'_>>,
    ) -> Result<Self, String> {
        let projection_dispatches = SqFp8ProjectionDispatches::from_info(
            &context
                .device_info()
                .map_err(|err| format!("failed to query runtime context device: {err}"))?,
            None,
        );
        if let Some(overlay) = sq_overlay {
            match load_sq8_resident_tensor(context, stream, overlay.artifact, tensor_name) {
                Ok(Some(resident)) => {
                    let rows = resident.rows;
                    let cols = resident.cols;
                    let scale_count = resident.scale_count;
                    let scale_kind = resident.scale_kind;
                    let scale_block_cols = resident.scale_block_cols;
                    return Ok(Self {
                        rows,
                        cols,
                        group_size: 0,
                        tensor_scale: 1.0,
                        scale_count,
                        row_scale_count: 0,
                        projection_dispatches,
                        storage: PackageResidentMatvecStorage::SqFp8 {
                            payload_buffer: Arc::new(resident.payload_buffer),
                            scale_buffer: Arc::new(resident.scale_buffer),
                            scale_kind,
                            scale_block_cols,
                        },
                    });
                }
                Ok(None) => {}
                Err(err) => {
                    return Err(format!(
                        "failed to load SQ FP8 overlay tensor {tensor_name}: {err}"
                    ));
                }
            }
        }
        Self::load_with_shared_buffers(
            context,
            stream,
            registry,
            shared_buffers,
            path,
            tensor_name,
            chunk_bytes,
            projection_dispatches,
        )
    }

    fn aq4_storage(&self, label: &str) -> Result<PackageAq4StorageRef<'_>, String> {
        match &self.storage {
            PackageResidentMatvecStorage::Aq4 {
                index_buffer,
                scale_buffer,
                codebook_buffer,
                scale_values_buffer,
                row_scale_buffer,
            } => Ok(PackageAq4StorageRef {
                index_buffer: index_buffer.as_ref(),
                scale_buffer: scale_buffer.as_ref(),
                codebook_buffer: codebook_buffer.as_ref(),
                scale_values_buffer: scale_values_buffer.as_ref(),
                row_scale_buffer: row_scale_buffer.as_deref(),
            }),
            PackageResidentMatvecStorage::F32 { .. } => Err(format!(
                "{label} requested AQ4 storage for SQ/F32 resident matrix"
            )),
            PackageResidentMatvecStorage::SqFp8 { .. } => Err(format!(
                "{label} requested AQ4 storage for SQ FP8 resident matrix"
            )),
        }
    }

    fn is_f32(&self) -> bool {
        !matches!(self.storage, PackageResidentMatvecStorage::Aq4 { .. })
    }

    fn sq_fp8_storage(&self) -> Option<Sq8ResidentRuntimeTensorRef<'_>> {
        match &self.storage {
            PackageResidentMatvecStorage::SqFp8 {
                payload_buffer,
                scale_buffer,
                scale_kind,
                scale_block_cols,
            } => Some(Sq8ResidentRuntimeTensorRef {
                payload_buffer: payload_buffer.as_ref(),
                scale_buffer: scale_buffer.as_ref(),
                scale_kind: *scale_kind,
                scale_block_cols: *scale_block_cols,
            }),
            PackageResidentMatvecStorage::Aq4 { .. } | PackageResidentMatvecStorage::F32 { .. } => {
                None
            }
        }
    }

    fn projection_dispatch(
        &self,
        operation: SqFp8ProjectionMatvecOperation,
    ) -> SqFp8ProjectionDispatch {
        self.projection_dispatches.for_operation(operation)
    }

    pub fn row_f32(
        &self,
        row_index: usize,
        output_buffer: &mut ullm_runtime_sys::RuntimeBuffer,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        label: &str,
    ) -> Result<(), String> {
        match &self.storage {
            PackageResidentMatvecStorage::Aq4 { .. } => {
                let aq4 = self.aq4_storage(label)?;
                ullm_runtime_sys::aq4_row_f32(
                    aq4.index_buffer,
                    aq4.scale_buffer,
                    aq4.codebook_buffer,
                    aq4.scale_values_buffer,
                    aq4.row_scale_buffer,
                    self.scale_count,
                    self.group_size,
                    self.tensor_scale,
                    self.row_scale_count,
                    self.rows,
                    self.cols,
                    row_index,
                    output_buffer,
                    Some(stream),
                )
                .map_err(|err| format!("failed to gather {label} AQ4 row: {err}"))
            }
            PackageResidentMatvecStorage::F32 { matrix_buffer } => {
                let offset = row_index
                    .checked_mul(self.cols)
                    .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
                    .ok_or_else(|| format!("{label} F32 row offset overflows"))?;
                let row_bytes = checked_f32_byte_len(self.cols, label)?;
                let mut bytes = vec![0_u8; row_bytes];
                matrix_buffer
                    .copy_to_host(offset, &mut bytes, Some(stream))
                    .map_err(|err| format!("failed to copy {label} F32 row from runtime: {err}"))?;
                stream
                    .synchronize()
                    .map_err(|err| format!("failed to synchronize {label} F32 row copy: {err}"))?;
                output_buffer
                    .copy_from_host(0, &bytes, Some(stream))
                    .map_err(|err| format!("failed to copy {label} F32 row to runtime: {err}"))
            }
            PackageResidentMatvecStorage::SqFp8 {
                payload_buffer,
                scale_buffer,
                scale_kind,
                scale_block_cols,
            } => {
                let row = self.sq_fp8_row_to_host_f32(
                    row_index,
                    payload_buffer.as_ref(),
                    scale_buffer.as_ref(),
                    *scale_kind,
                    *scale_block_cols,
                    stream,
                    label,
                )?;
                let output_bytes = checked_f32_byte_len(self.cols, label)?;
                let actual_bytes = output_buffer.size().map_err(|err| {
                    format!("failed to query {label} SQ FP8 row output size: {err}")
                })?;
                if actual_bytes < output_bytes {
                    return Err(format!(
                        "{label} SQ FP8 row output buffer is too small: got {actual_bytes} bytes expected at least {output_bytes}"
                    ));
                }
                output_buffer
                    .copy_from_host(0, &encode_f32_to_bytes(&row), Some(stream))
                    .map_err(|err| format!("failed to copy {label} SQ FP8 row to runtime: {err}"))
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn sq_fp8_row_to_host_f32(
        &self,
        row_index: usize,
        payload_buffer: &ullm_runtime_sys::RuntimeBuffer,
        scale_buffer: &ullm_runtime_sys::RuntimeBuffer,
        scale_kind: u32,
        scale_block_cols: usize,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        label: &str,
    ) -> Result<Vec<f32>, String> {
        if row_index >= self.rows {
            return Err(format!(
                "{label} SQ FP8 row index {row_index} is out of range for {} rows",
                self.rows
            ));
        }
        let blocks_per_row = if scale_kind == ullm_runtime_sys::SQ_FP8_SCALE_ROW_BLOCK {
            if scale_block_cols == 0 {
                return Err(format!(
                    "{label} SQ FP8 row_block scale_block_cols must be greater than zero"
                ));
            }
            self.cols
                .checked_add(scale_block_cols - 1)
                .and_then(|value| value.checked_div(scale_block_cols))
                .ok_or_else(|| format!("{label} SQ FP8 row_block count overflows"))?
        } else {
            1
        };
        let (scale_offset, scale_count) = match scale_kind {
            ullm_runtime_sys::SQ_FP8_SCALE_TENSOR => (0, 1),
            ullm_runtime_sys::SQ_FP8_SCALE_ROW => {
                let offset = row_index
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| format!("{label} SQ FP8 row scale offset overflows"))?;
                (offset, 1)
            }
            ullm_runtime_sys::SQ_FP8_SCALE_ROW_BLOCK => {
                let row_scale_index = row_index
                    .checked_mul(blocks_per_row)
                    .ok_or_else(|| format!("{label} SQ FP8 row_block scale index overflows"))?;
                let offset = row_scale_index
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| format!("{label} SQ FP8 row_block scale offset overflows"))?;
                (offset, blocks_per_row)
            }
            other => {
                return Err(format!(
                    "{label} SQ FP8 scale kind must be tensor(0), row(1), or row_block(2), got {other}"
                ));
            }
        };
        let payload_offset = row_index
            .checked_mul(self.cols)
            .ok_or_else(|| format!("{label} SQ FP8 row payload offset overflows"))?;
        let mut payload = vec![0_u8; self.cols];
        payload_buffer
            .copy_to_host(payload_offset, &mut payload, Some(stream))
            .map_err(|err| format!("failed to copy {label} SQ FP8 row payload: {err}"))?;
        let mut scale_bytes = vec![0_u8; checked_f32_byte_len(scale_count, "SQ FP8 row scale")?];
        scale_buffer
            .copy_to_host(scale_offset, &mut scale_bytes, Some(stream))
            .map_err(|err| format!("failed to copy {label} SQ FP8 row scales: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize {label} SQ FP8 row copy: {err}"))?;
        let scales = decode_f32_le_values(&scale_bytes);
        if scales.len() != scale_count {
            return Err(format!(
                "{label} SQ FP8 decoded scale count mismatch: got {} expected {scale_count}",
                scales.len()
            ));
        }
        let mut row = Vec::with_capacity(self.cols);
        for (col, payload_value) in payload.iter().copied().enumerate() {
            let scale = match scale_kind {
                ullm_runtime_sys::SQ_FP8_SCALE_TENSOR | ullm_runtime_sys::SQ_FP8_SCALE_ROW => {
                    scales[0]
                }
                ullm_runtime_sys::SQ_FP8_SCALE_ROW_BLOCK => scales[col / scale_block_cols],
                _ => unreachable!("validated SQ FP8 scale kind"),
            };
            let value = fp8_e4m3fn_to_f32(payload_value);
            if !value.is_finite() || !scale.is_finite() || scale <= 0.0 {
                return Err(format!(
                    "{label} SQ FP8 row {row_index} col {col} has invalid value={value} scale={scale}"
                ));
            }
            row.push(value * scale);
        }
        Ok(row)
    }

    pub fn matvec(
        &self,
        input_buffer: &ullm_runtime_sys::RuntimeBuffer,
        output_buffer: &mut ullm_runtime_sys::RuntimeBuffer,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        label: &str,
    ) -> Result<(), String> {
        match &self.storage {
            PackageResidentMatvecStorage::Aq4 { .. } => {
                let aq4 = self.aq4_storage(label)?;
                ullm_runtime_sys::aq4_matvec_f32(
                    aq4.index_buffer,
                    aq4.scale_buffer,
                    aq4.codebook_buffer,
                    aq4.scale_values_buffer,
                    input_buffer,
                    aq4.row_scale_buffer,
                    self.scale_count,
                    self.group_size,
                    self.tensor_scale,
                    self.row_scale_count,
                    self.rows,
                    self.cols,
                    output_buffer,
                    Some(stream),
                )
                .map_err(|err| format!("failed to run {label} AQ4 matvec: {err}"))
            }
            PackageResidentMatvecStorage::F32 { matrix_buffer } => ullm_runtime_sys::matvec_f32(
                matrix_buffer.as_ref(),
                input_buffer,
                self.rows,
                self.cols,
                output_buffer,
                Some(stream),
            )
            .map_err(|err| format!("failed to run {label} F32 matvec: {err}")),
            PackageResidentMatvecStorage::SqFp8 {
                payload_buffer,
                scale_buffer,
                scale_kind,
                scale_block_cols,
            } => {
                let dispatch = self.projection_dispatch(SqFp8ProjectionMatvecOperation::Single);
                dispatch.require_direct_family(label)?;
                ullm_runtime_sys::sq_fp8_matvec_f32(
                    payload_buffer.as_ref(),
                    scale_buffer.as_ref(),
                    input_buffer,
                    self.rows,
                    self.cols,
                    *scale_kind,
                    *scale_block_cols,
                    output_buffer,
                    Some(stream),
                )
                .map_err(|err| format!("failed to run {label} SQ FP8 matvec: {err}"))?;
                record_sq_fp8_projection_dispatch(dispatch);
                Ok(())
            }
        }
    }

    pub fn matvec_batch(
        &self,
        input_buffer: &ullm_runtime_sys::RuntimeBuffer,
        batch_count: usize,
        output_buffer: &mut ullm_runtime_sys::RuntimeBuffer,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        label: &str,
    ) -> Result<(), String> {
        match &self.storage {
            PackageResidentMatvecStorage::Aq4 { .. } => {
                let aq4 = self.aq4_storage(label)?;
                ullm_runtime_sys::aq4_matvec_batch_f32(
                    aq4.index_buffer,
                    aq4.scale_buffer,
                    aq4.codebook_buffer,
                    aq4.scale_values_buffer,
                    input_buffer,
                    aq4.row_scale_buffer,
                    self.scale_count,
                    self.group_size,
                    self.tensor_scale,
                    self.row_scale_count,
                    self.rows,
                    self.cols,
                    batch_count,
                    output_buffer,
                    Some(stream),
                )
                .map_err(|err| format!("failed to run {label} AQ4 matvec batch: {err}"))
            }
            PackageResidentMatvecStorage::SqFp8 {
                payload_buffer,
                scale_buffer,
                scale_kind,
                scale_block_cols,
            } => {
                let dispatch = self.projection_dispatch(SqFp8ProjectionMatvecOperation::Batch);
                dispatch.require_direct_family(label)?;
                ullm_runtime_sys::sq_fp8_matvec_batch_f32(
                    payload_buffer.as_ref(),
                    scale_buffer.as_ref(),
                    input_buffer,
                    self.rows,
                    self.cols,
                    *scale_kind,
                    *scale_block_cols,
                    batch_count,
                    output_buffer,
                    Some(stream),
                )
                .map_err(|err| format!("failed to run {label} SQ FP8 matvec batch: {err}"))?;
                record_sq_fp8_projection_dispatch(dispatch);
                Ok(())
            }
            PackageResidentMatvecStorage::F32 { .. } => {
                Err(format!("{label} F32 matvec batch is not implemented"))
            }
        }
    }

    pub fn matvec_top1(
        &self,
        input_buffer: &ullm_runtime_sys::RuntimeBuffer,
        partial_values_buffer: &mut ullm_runtime_sys::RuntimeBuffer,
        partial_indices_buffer: &mut ullm_runtime_sys::RuntimeBuffer,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        label: &str,
    ) -> Result<usize, String> {
        let aq4 = self.aq4_storage(label)?;
        ullm_runtime_sys::aq4_matvec_top1_f32(
            aq4.index_buffer,
            aq4.scale_buffer,
            aq4.codebook_buffer,
            aq4.scale_values_buffer,
            input_buffer,
            aq4.row_scale_buffer,
            self.scale_count,
            self.group_size,
            self.tensor_scale,
            self.row_scale_count,
            self.rows,
            self.cols,
            partial_values_buffer,
            partial_indices_buffer,
            Some(stream),
        )
        .map_err(|err| format!("failed to run {label} AQ4 matvec top1: {err}"))
    }

    pub fn matvec_add(
        &self,
        input_buffer: &ullm_runtime_sys::RuntimeBuffer,
        residual_buffer: &ullm_runtime_sys::RuntimeBuffer,
        output_buffer: &mut ullm_runtime_sys::RuntimeBuffer,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        label: &str,
    ) -> Result<(), String> {
        match &self.storage {
            PackageResidentMatvecStorage::Aq4 { .. } => {
                let aq4 = self.aq4_storage(label)?;
                ullm_runtime_sys::aq4_matvec_add_f32(
                    aq4.index_buffer,
                    aq4.scale_buffer,
                    aq4.codebook_buffer,
                    aq4.scale_values_buffer,
                    input_buffer,
                    residual_buffer,
                    aq4.row_scale_buffer,
                    self.scale_count,
                    self.group_size,
                    self.tensor_scale,
                    self.row_scale_count,
                    self.rows,
                    self.cols,
                    output_buffer,
                    Some(stream),
                )
                .map_err(|err| format!("failed to run {label} AQ4 matvec add: {err}"))
            }
            PackageResidentMatvecStorage::F32 { .. }
            | PackageResidentMatvecStorage::SqFp8 { .. } => {
                self.matvec(input_buffer, output_buffer, stream, label)?;
                let mut projected =
                    read_runtime_buffer_f32(output_buffer, stream, self.rows, label)?;
                let residual = read_runtime_buffer_f32(residual_buffer, stream, self.rows, label)?;
                for (left, right) in projected.iter_mut().zip(residual.iter()) {
                    *left += *right;
                }
                output_buffer
                    .copy_from_host(0, &encode_f32_to_bytes(&projected), Some(stream))
                    .map_err(|err| format!("failed to copy {label} F32 matvec add: {err}"))
            }
        }
    }

    pub fn matvec_pair_with(
        &self,
        right: &Self,
        input_buffer: &ullm_runtime_sys::RuntimeBuffer,
        left_output_buffer: &mut ullm_runtime_sys::RuntimeBuffer,
        right_output_buffer: &mut ullm_runtime_sys::RuntimeBuffer,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        label: &str,
    ) -> Result<(), String> {
        if self.cols != right.cols {
            return Err(format!(
                "{label} AQ4 matvec pair column mismatch: left=[{},{}] right=[{},{}]",
                self.rows, self.cols, right.rows, right.cols
            ));
        }
        if let (Some(left_sq), Some(right_sq)) = (self.sq_fp8_storage(), right.sq_fp8_storage()) {
            let dispatch = self.projection_dispatch(SqFp8ProjectionMatvecOperation::Pair);
            dispatch.require_direct_family(label)?;
            ullm_runtime_sys::sq_fp8_matvec_pair_f32(
                left_sq.payload_buffer,
                left_sq.scale_buffer,
                left_sq.scale_kind,
                left_sq.scale_block_cols,
                right_sq.payload_buffer,
                right_sq.scale_buffer,
                right_sq.scale_kind,
                right_sq.scale_block_cols,
                input_buffer,
                self.rows,
                right.rows,
                self.cols,
                left_output_buffer,
                right_output_buffer,
                Some(stream),
            )
            .map_err(|err| format!("failed to run {label} SQ FP8 matvec pair: {err}"))?;
            record_sq_fp8_projection_dispatch(dispatch);
            return Ok(());
        }
        if self.is_f32() || right.is_f32() {
            self.matvec(input_buffer, left_output_buffer, stream, label)?;
            return right.matvec(input_buffer, right_output_buffer, stream, label);
        }
        let left_aq4 = self.aq4_storage(label)?;
        let right_aq4 = right.aq4_storage(label)?;
        let result = ullm_runtime_sys::aq4_matvec_pair_f32(
            left_aq4.index_buffer,
            left_aq4.scale_buffer,
            left_aq4.codebook_buffer,
            left_aq4.scale_values_buffer,
            left_aq4.row_scale_buffer,
            self.scale_count,
            self.group_size,
            self.tensor_scale,
            self.row_scale_count,
            right_aq4.index_buffer,
            right_aq4.scale_buffer,
            right_aq4.codebook_buffer,
            right_aq4.scale_values_buffer,
            right_aq4.row_scale_buffer,
            right.scale_count,
            right.group_size,
            right.tensor_scale,
            right.row_scale_count,
            input_buffer,
            self.rows,
            right.rows,
            self.cols,
            left_output_buffer,
            right_output_buffer,
            Some(stream),
        );
        match result {
            Ok(()) => Ok(()),
            Err(err) if env_flag_enabled("ULLM_REQUIRE_HIP_AQ4_MATVEC_PAIR_KERNEL") => {
                Err(format!("failed to run {label} AQ4 matvec pair: {err}"))
            }
            Err(_) => {
                self.matvec(input_buffer, left_output_buffer, stream, label)?;
                right.matvec(input_buffer, right_output_buffer, stream, label)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn matvec_triple_with(
        &self,
        second: &Self,
        third: &Self,
        input_buffer: &ullm_runtime_sys::RuntimeBuffer,
        first_output_buffer: &mut ullm_runtime_sys::RuntimeBuffer,
        second_output_buffer: &mut ullm_runtime_sys::RuntimeBuffer,
        third_output_buffer: &mut ullm_runtime_sys::RuntimeBuffer,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        label: &str,
    ) -> Result<(), String> {
        if self.cols != second.cols || self.cols != third.cols {
            return Err(format!(
                "{label} AQ4 matvec triple column mismatch: first=[{},{}] second=[{},{}] third=[{},{}]",
                self.rows, self.cols, second.rows, second.cols, third.rows, third.cols
            ));
        }
        if let (Some(first_sq), Some(second_sq), Some(third_sq)) = (
            self.sq_fp8_storage(),
            second.sq_fp8_storage(),
            third.sq_fp8_storage(),
        ) {
            let dispatch = self.projection_dispatch(SqFp8ProjectionMatvecOperation::Triple);
            dispatch.require_direct_family(label)?;
            ullm_runtime_sys::sq_fp8_matvec_triple_f32(
                first_sq.payload_buffer,
                first_sq.scale_buffer,
                first_sq.scale_kind,
                first_sq.scale_block_cols,
                second_sq.payload_buffer,
                second_sq.scale_buffer,
                second_sq.scale_kind,
                second_sq.scale_block_cols,
                third_sq.payload_buffer,
                third_sq.scale_buffer,
                third_sq.scale_kind,
                third_sq.scale_block_cols,
                input_buffer,
                self.rows,
                second.rows,
                third.rows,
                self.cols,
                first_output_buffer,
                second_output_buffer,
                third_output_buffer,
                Some(stream),
            )
            .map_err(|err| format!("failed to run {label} SQ FP8 matvec triple: {err}"))?;
            record_sq_fp8_projection_dispatch(dispatch);
            return Ok(());
        }
        if self.is_f32() || second.is_f32() || third.is_f32() {
            self.matvec(input_buffer, first_output_buffer, stream, label)?;
            second.matvec(input_buffer, second_output_buffer, stream, label)?;
            return third.matvec(input_buffer, third_output_buffer, stream, label);
        }
        let first_aq4 = self.aq4_storage(label)?;
        let second_aq4 = second.aq4_storage(label)?;
        let third_aq4 = third.aq4_storage(label)?;
        let result = ullm_runtime_sys::aq4_matvec_triple_f32(
            first_aq4.index_buffer,
            first_aq4.scale_buffer,
            first_aq4.codebook_buffer,
            first_aq4.scale_values_buffer,
            first_aq4.row_scale_buffer,
            self.scale_count,
            self.group_size,
            self.tensor_scale,
            self.row_scale_count,
            second_aq4.index_buffer,
            second_aq4.scale_buffer,
            second_aq4.codebook_buffer,
            second_aq4.scale_values_buffer,
            second_aq4.row_scale_buffer,
            second.scale_count,
            second.group_size,
            second.tensor_scale,
            second.row_scale_count,
            third_aq4.index_buffer,
            third_aq4.scale_buffer,
            third_aq4.codebook_buffer,
            third_aq4.scale_values_buffer,
            third_aq4.row_scale_buffer,
            third.scale_count,
            third.group_size,
            third.tensor_scale,
            third.row_scale_count,
            input_buffer,
            self.rows,
            second.rows,
            third.rows,
            self.cols,
            first_output_buffer,
            second_output_buffer,
            third_output_buffer,
            Some(stream),
        );
        match result {
            Ok(()) => Ok(()),
            Err(err) if env_flag_enabled("ULLM_REQUIRE_HIP_AQ4_MATVEC_TRIPLE_KERNEL") => {
                Err(format!("failed to run {label} AQ4 matvec triple: {err}"))
            }
            Err(_) => {
                self.matvec_pair_with(
                    second,
                    input_buffer,
                    first_output_buffer,
                    second_output_buffer,
                    stream,
                    label,
                )?;
                third.matvec(input_buffer, third_output_buffer, stream, label)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn matvec_qkv_z_gate_beta_with(
        &self,
        z: &Self,
        a: &Self,
        b: &Self,
        input_buffer: &ullm_runtime_sys::RuntimeBuffer,
        a_log_buffer: &ullm_runtime_sys::RuntimeBuffer,
        dt_bias_buffer: &ullm_runtime_sys::RuntimeBuffer,
        qkv_output_buffer: &mut ullm_runtime_sys::RuntimeBuffer,
        z_output_buffer: &mut ullm_runtime_sys::RuntimeBuffer,
        gate_output_buffer: &mut ullm_runtime_sys::RuntimeBuffer,
        beta_output_buffer: &mut ullm_runtime_sys::RuntimeBuffer,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        label: &str,
    ) -> Result<(), String> {
        if self.cols != z.cols || self.cols != a.cols || self.cols != b.cols {
            return Err(format!(
                "{label} AQ4 qkv/z gate/beta column mismatch: qkv=[{},{}] z=[{},{}] a=[{},{}] b=[{},{}]",
                self.rows, self.cols, z.rows, z.cols, a.rows, a.cols, b.rows, b.cols
            ));
        }
        if a.rows != b.rows {
            return Err(format!(
                "{label} AQ4 qkv/z gate/beta head mismatch: a=[{},{}] b=[{},{}]",
                a.rows, a.cols, b.rows, b.cols
            ));
        }
        if self.is_f32() || z.is_f32() || a.is_f32() || b.is_f32() {
            self.matvec(input_buffer, qkv_output_buffer, stream, label)?;
            z.matvec(input_buffer, z_output_buffer, stream, label)?;
            return a.matvec_gate_beta_with(
                b,
                input_buffer,
                a_log_buffer,
                dt_bias_buffer,
                gate_output_buffer,
                beta_output_buffer,
                stream,
                label,
            );
        }
        let qkv_aq4 = self.aq4_storage(label)?;
        let z_aq4 = z.aq4_storage(label)?;
        let a_aq4 = a.aq4_storage(label)?;
        let b_aq4 = b.aq4_storage(label)?;
        let result = ullm_runtime_sys::aq4_matvec_qkv_z_gate_beta_f32(
            qkv_aq4.index_buffer,
            qkv_aq4.scale_buffer,
            qkv_aq4.codebook_buffer,
            qkv_aq4.scale_values_buffer,
            qkv_aq4.row_scale_buffer,
            self.scale_count,
            self.group_size,
            self.tensor_scale,
            self.row_scale_count,
            z_aq4.index_buffer,
            z_aq4.scale_buffer,
            z_aq4.codebook_buffer,
            z_aq4.scale_values_buffer,
            z_aq4.row_scale_buffer,
            z.scale_count,
            z.group_size,
            z.tensor_scale,
            z.row_scale_count,
            a_aq4.index_buffer,
            a_aq4.scale_buffer,
            a_aq4.codebook_buffer,
            a_aq4.scale_values_buffer,
            a_aq4.row_scale_buffer,
            a.scale_count,
            a.group_size,
            a.tensor_scale,
            a.row_scale_count,
            b_aq4.index_buffer,
            b_aq4.scale_buffer,
            b_aq4.codebook_buffer,
            b_aq4.scale_values_buffer,
            b_aq4.row_scale_buffer,
            b.scale_count,
            b.group_size,
            b.tensor_scale,
            b.row_scale_count,
            input_buffer,
            a_log_buffer,
            dt_bias_buffer,
            self.rows,
            z.rows,
            a.rows,
            self.cols,
            qkv_output_buffer,
            z_output_buffer,
            gate_output_buffer,
            beta_output_buffer,
            Some(stream),
        );
        match result {
            Ok(()) => Ok(()),
            Err(err) if env_flag_enabled("ULLM_REQUIRE_HIP_AQ4_MATVEC_QKV_Z_GATE_BETA_KERNEL") => {
                Err(format!("failed to run {label} AQ4 qkv/z gate/beta: {err}"))
            }
            Err(_) => {
                self.matvec_pair_with(
                    z,
                    input_buffer,
                    qkv_output_buffer,
                    z_output_buffer,
                    stream,
                    label,
                )?;
                a.matvec_gate_beta_with(
                    b,
                    input_buffer,
                    a_log_buffer,
                    dt_bias_buffer,
                    gate_output_buffer,
                    beta_output_buffer,
                    stream,
                    label,
                )
            }
        }
    }

    pub fn matvec_silu_mul_with(
        &self,
        up: &Self,
        input_buffer: &ullm_runtime_sys::RuntimeBuffer,
        output_buffer: &mut ullm_runtime_sys::RuntimeBuffer,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        label: &str,
    ) -> Result<(), String> {
        if self.rows != up.rows || self.cols != up.cols {
            return Err(format!(
                "{label} AQ4 fused MLP shape mismatch: gate=[{},{}] up=[{},{}]",
                self.rows, self.cols, up.rows, up.cols
            ));
        }
        if self.is_f32() || up.is_f32() {
            self.matvec(input_buffer, output_buffer, stream, label)?;
            let gate_values = read_runtime_buffer_f32(output_buffer, stream, self.rows, label)?;
            up.matvec(input_buffer, output_buffer, stream, label)?;
            let mut up_values = read_runtime_buffer_f32(output_buffer, stream, up.rows, label)?;
            for (value, gate) in up_values.iter_mut().zip(gate_values.iter()) {
                let sigmoid = 1.0_f32 / (1.0_f32 + (-*gate).exp());
                *value *= *gate * sigmoid;
            }
            output_buffer
                .copy_from_host(0, &encode_f32_to_bytes(&up_values), Some(stream))
                .map_err(|err| format!("failed to copy {label} F32 SiLU-mul result: {err}"))?;
            return Ok(());
        }
        let gate_aq4 = self.aq4_storage(label)?;
        let up_aq4 = up.aq4_storage(label)?;
        ullm_runtime_sys::aq4_matvec_silu_mul_f32(
            gate_aq4.index_buffer,
            gate_aq4.scale_buffer,
            gate_aq4.codebook_buffer,
            gate_aq4.scale_values_buffer,
            gate_aq4.row_scale_buffer,
            self.scale_count,
            self.group_size,
            self.tensor_scale,
            self.row_scale_count,
            up_aq4.index_buffer,
            up_aq4.scale_buffer,
            up_aq4.codebook_buffer,
            up_aq4.scale_values_buffer,
            up_aq4.row_scale_buffer,
            up.scale_count,
            up.group_size,
            up.tensor_scale,
            up.row_scale_count,
            input_buffer,
            self.rows,
            self.cols,
            output_buffer,
            Some(stream),
        )
        .map_err(|err| format!("failed to run {label} AQ4 fused matvec SiLU-mul: {err}"))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn matvec_gate_beta_with(
        &self,
        b: &Self,
        input_buffer: &ullm_runtime_sys::RuntimeBuffer,
        a_log_buffer: &ullm_runtime_sys::RuntimeBuffer,
        dt_bias_buffer: &ullm_runtime_sys::RuntimeBuffer,
        gate_output_buffer: &mut ullm_runtime_sys::RuntimeBuffer,
        beta_output_buffer: &mut ullm_runtime_sys::RuntimeBuffer,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        label: &str,
    ) -> Result<(), String> {
        if self.rows != b.rows || self.cols != b.cols {
            return Err(format!(
                "{label} AQ4 fused gate/beta shape mismatch: a=[{},{}] b=[{},{}]",
                self.rows, self.cols, b.rows, b.cols
            ));
        }
        if self.is_f32() || b.is_f32() {
            self.matvec(input_buffer, gate_output_buffer, stream, label)?;
            b.matvec(input_buffer, beta_output_buffer, stream, label)?;
            let a_values = read_runtime_buffer_f32(gate_output_buffer, stream, self.rows, label)?;
            let b_values = read_runtime_buffer_f32(beta_output_buffer, stream, b.rows, label)?;
            let a_log_values = read_runtime_buffer_f32(a_log_buffer, stream, self.rows, label)?;
            let dt_bias_values = read_runtime_buffer_f32(dt_bias_buffer, stream, self.rows, label)?;
            let mut gate_values = vec![0.0_f32; self.rows];
            let mut beta_values = vec![0.0_f32; self.rows];
            for index in 0..self.rows {
                let x = a_values[index] + dt_bias_values[index];
                let softplus = if x <= 20.0_f32 {
                    (1.0_f32 + x.exp()).ln()
                } else {
                    x
                };
                gate_values[index] = -a_log_values[index].exp() * softplus;
                beta_values[index] = 1.0_f32 / (1.0_f32 + (-b_values[index]).exp());
            }
            gate_output_buffer
                .copy_from_host(0, &encode_f32_to_bytes(&gate_values), Some(stream))
                .map_err(|err| format!("failed to copy {label} F32 gate output: {err}"))?;
            beta_output_buffer
                .copy_from_host(0, &encode_f32_to_bytes(&beta_values), Some(stream))
                .map_err(|err| format!("failed to copy {label} F32 beta output: {err}"))?;
            return Ok(());
        }
        let a_aq4 = self.aq4_storage(label)?;
        let b_aq4 = b.aq4_storage(label)?;
        ullm_runtime_sys::aq4_matvec_gate_beta_f32(
            a_aq4.index_buffer,
            a_aq4.scale_buffer,
            a_aq4.codebook_buffer,
            a_aq4.scale_values_buffer,
            a_aq4.row_scale_buffer,
            self.scale_count,
            self.group_size,
            self.tensor_scale,
            self.row_scale_count,
            b_aq4.index_buffer,
            b_aq4.scale_buffer,
            b_aq4.codebook_buffer,
            b_aq4.scale_values_buffer,
            b_aq4.row_scale_buffer,
            b.scale_count,
            b.group_size,
            b.tensor_scale,
            b.row_scale_count,
            input_buffer,
            a_log_buffer,
            dt_bias_buffer,
            self.rows,
            self.cols,
            gate_output_buffer,
            beta_output_buffer,
            Some(stream),
        )
        .map_err(|err| format!("failed to run {label} AQ4 fused gate/beta: {err}"))
    }
}
