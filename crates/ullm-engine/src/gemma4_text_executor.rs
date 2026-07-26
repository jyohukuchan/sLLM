// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Diagnostic-only, non-quantized Gemma4 text execution.
//!
//! `Gemma4TextExecutor` deliberately consumes the inspected source checkpoint
//! directly: BF16 safetensors weights are streamed to the existing BF16 x F32
//! matvec runtime primitive, while all activation-side math remains F32. It is
//! an architecture bring-up path, not an SQ8_0/AQ4_0 conversion or a serving
//! runtime. In particular, it owns one causal sequence and emits complete
//! layer outputs so `tools/architecture_hf_trace.py` can localize a mismatch.
//!
//! The implementation is intentionally fail-closed around the exact Hugging
//! Face branches needed by `google/gemma-4-E2B`: causal local/full attention,
//! direct-weight RMSNorm, PLE, tied embedding/head, and final logit soft-cap.

use crate::host_bytes::{decode_f32_le_values, encode_f32_to_bytes};
use crate::model_config::{
    DecoderLayerKind, Gemma4TextConfig, LoadedModelConfig, load_model_config_from_dir,
};
use serde_json::Value;
use std::collections::BTreeMap;
use std::env;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use ullm_runtime_sys::{
    DeviceInfo, RuntimeContext, RuntimeStream, device_count, device_info, matvec_bf16_f32,
};

pub const GEMMA4_TEXT_MODEL_FILE: &str = "model.safetensors";
pub const GEMMA4_TEXT_WEIGHT_PREFIX: &str = "model.language_model.";
pub const GEMMA4_TEXT_EMBED_TOKENS: &str = "model.language_model.embed_tokens.weight";
pub const GEMMA4_TEXT_EMBED_TOKENS_PER_LAYER: &str =
    "model.language_model.embed_tokens_per_layer.weight";
pub const GEMMA4_TEXT_PER_LAYER_MODEL_PROJECTION: &str =
    "model.language_model.per_layer_model_projection.weight";
pub const GEMMA4_TEXT_PER_LAYER_PROJECTION_NORM: &str =
    "model.language_model.per_layer_projection_norm.weight";
pub const GEMMA4_TEXT_FINAL_NORM: &str = "model.language_model.norm.weight";
pub const GEMMA4_TEXT_REQUIRED_HIP_KERNEL_ENV: &str = "ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL";

const SAFETENSORS_HEADER_LIMIT_BYTES: u64 = 128 * 1024 * 1024;
const R9700_RUNTIME_NAME: &str = "AMD Radeon Graphics";
const R9700_MEMORY_BYTES_MIN: u64 = 30 * 1024 * 1024 * 1024;
const R9700_MEMORY_BYTES_MAX: u64 = 34 * 1024 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gemma4TextDeviceIdentity {
    pub runtime_index: u32,
    pub device_id: i32,
    pub backend: String,
    pub name: String,
    pub gcn_arch_name: String,
    pub compute_major: i32,
    pub compute_minor: i32,
    pub total_global_mem: u64,
}

impl Gemma4TextDeviceIdentity {
    fn from_runtime(runtime_index: u32, info: DeviceInfo) -> Self {
        Self {
            runtime_index,
            device_id: info.device_id,
            backend: info.backend,
            name: info.name,
            gcn_arch_name: info.gcn_arch_name,
            compute_major: info.compute_major,
            compute_minor: info.compute_minor,
            total_global_mem: info.total_global_mem,
        }
    }

    fn validate_r9700(&self) -> Result<(), String> {
        let arch = self.gcn_arch_name.split(':').next().unwrap_or_default();
        if self.backend != "hip"
            || self.name != R9700_RUNTIME_NAME
            || !arch.eq_ignore_ascii_case("gfx1201")
            || self.compute_major != 12
            || self.compute_minor != 0
            || !(R9700_MEMORY_BYTES_MIN..=R9700_MEMORY_BYTES_MAX).contains(&self.total_global_mem)
        {
            return Err(format!(
                "Gemma4TextExecutor requires the canonical R9700/gfx1201 HIP identity, got runtime_index={} backend={} name={} arch={} compute={}.{} memory={}",
                self.runtime_index,
                self.backend,
                self.name,
                self.gcn_arch_name,
                self.compute_major,
                self.compute_minor,
                self.total_global_mem,
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Gemma4TextTop1 {
    pub token_id: u32,
    pub logit: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Gemma4TextStepTrace {
    pub input_token_ids: Vec<u32>,
    /// Flattened `[tokens, hidden]` F32 scaled word embeddings.
    pub embedding: Vec<f32>,
    /// One flattened `[tokens, hidden]` F32 output per decoder layer.
    pub layer_outputs: Vec<Vec<f32>>,
    /// Flattened `[tokens, hidden]` F32 final RMSNorm output.
    pub final_norm: Vec<f32>,
    /// F32 final-token logits after Gemma4's final soft-cap.
    pub logits_last: Vec<f32>,
    pub top1: Gemma4TextTop1,
}

#[derive(Debug, Clone)]
struct SafeTensorEntry {
    dtype: String,
    shape: Vec<usize>,
    data_start: u64,
    data_end: u64,
}

struct SafeTensorReader {
    path: PathBuf,
    file: File,
    payload_start: u64,
    entries: BTreeMap<String, SafeTensorEntry>,
}

impl SafeTensorReader {
    fn open(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let canonical_path = std::fs::canonicalize(path).map_err(|error| {
            format!(
                "failed to canonicalize safetensors {}: {error}",
                path.display()
            )
        })?;
        let mut file = File::open(&canonical_path).map_err(|error| {
            format!(
                "failed to open safetensors {}: {error}",
                canonical_path.display()
            )
        })?;
        let file_len = file
            .metadata()
            .map_err(|error| {
                format!(
                    "failed to stat safetensors {}: {error}",
                    canonical_path.display()
                )
            })?
            .len();
        if file_len < 8 {
            return Err(format!(
                "safetensors {} is shorter than its header length field",
                canonical_path.display()
            ));
        }
        let mut header_length_bytes = [0_u8; 8];
        file.read_exact(&mut header_length_bytes).map_err(|error| {
            format!(
                "failed to read safetensors header length {}: {error}",
                canonical_path.display()
            )
        })?;
        let header_length = u64::from_le_bytes(header_length_bytes);
        if header_length == 0 || header_length > SAFETENSORS_HEADER_LIMIT_BYTES {
            return Err(format!(
                "safetensors {} has invalid header length {header_length}",
                canonical_path.display()
            ));
        }
        let payload_start = 8_u64
            .checked_add(header_length)
            .ok_or_else(|| "safetensors header offset overflows u64".to_string())?;
        if payload_start > file_len {
            return Err(format!(
                "safetensors {} header extends beyond the file",
                canonical_path.display()
            ));
        }
        let header_size = usize::try_from(header_length)
            .map_err(|_| "safetensors header length exceeds host usize".to_string())?;
        let mut header_bytes = vec![0_u8; header_size];
        file.read_exact(&mut header_bytes).map_err(|error| {
            format!(
                "failed to read safetensors header {}: {error}",
                canonical_path.display()
            )
        })?;
        let header: Value = serde_json::from_slice(&header_bytes).map_err(|error| {
            format!(
                "failed to parse safetensors header {}: {error}",
                canonical_path.display()
            )
        })?;
        let object = header.as_object().ok_or_else(|| {
            format!(
                "safetensors header {} must be a JSON object",
                canonical_path.display()
            )
        })?;
        let payload_bytes = file_len
            .checked_sub(payload_start)
            .ok_or_else(|| "safetensors payload length underflow".to_string())?;
        let mut entries = BTreeMap::new();
        let mut intervals = Vec::new();
        for (name, value) in object {
            if name == "__metadata__" {
                continue;
            }
            let descriptor = value.as_object().ok_or_else(|| {
                format!(
                    "safetensors tensor {name} in {} must be an object",
                    canonical_path.display()
                )
            })?;
            let dtype = descriptor
                .get("dtype")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .ok_or_else(|| format!("safetensors tensor {name} has no dtype"))?
                .to_string();
            let shape = descriptor
                .get("shape")
                .and_then(Value::as_array)
                .ok_or_else(|| format!("safetensors tensor {name} has no shape"))?
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    let dimension = value.as_u64().ok_or_else(|| {
                        format!("safetensors tensor {name}.shape[{index}] is not an integer")
                    })?;
                    let dimension = usize::try_from(dimension).map_err(|_| {
                        format!("safetensors tensor {name}.shape[{index}] exceeds host usize")
                    })?;
                    if dimension == 0 {
                        return Err(format!("safetensors tensor {name}.shape[{index}] is zero"));
                    }
                    Ok(dimension)
                })
                .collect::<Result<Vec<_>, String>>()?;
            let offsets = descriptor
                .get("data_offsets")
                .and_then(Value::as_array)
                .ok_or_else(|| format!("safetensors tensor {name} has no data_offsets"))?;
            if offsets.len() != 2 {
                return Err(format!(
                    "safetensors tensor {name} data_offsets must have two entries"
                ));
            }
            let data_start = offsets[0].as_u64().ok_or_else(|| {
                format!("safetensors tensor {name} data_offsets[0] is not an integer")
            })?;
            let data_end = offsets[1].as_u64().ok_or_else(|| {
                format!("safetensors tensor {name} data_offsets[1] is not an integer")
            })?;
            if data_start > data_end || data_end > payload_bytes {
                return Err(format!(
                    "safetensors tensor {name} offsets [{data_start},{data_end}) exceed payload {payload_bytes}"
                ));
            }
            intervals.push((data_start, data_end, name.clone()));
            if entries
                .insert(
                    name.clone(),
                    SafeTensorEntry {
                        dtype,
                        shape,
                        data_start,
                        data_end,
                    },
                )
                .is_some()
            {
                return Err(format!("duplicate safetensors tensor name {name}"));
            }
        }
        if entries.is_empty() {
            return Err(format!(
                "safetensors {} contains no tensors",
                canonical_path.display()
            ));
        }
        intervals.sort_unstable_by_key(|(start, _, _)| *start);
        let mut previous_end = 0_u64;
        for (start, end, name) in intervals {
            if start < previous_end {
                return Err(format!(
                    "safetensors tensor {name} overlaps a preceding payload"
                ));
            }
            previous_end = end;
        }
        Ok(Self {
            path: canonical_path,
            file,
            payload_start,
            entries,
        })
    }

    fn require_bf16_shape(
        &self,
        name: &str,
        expected_shape: &[usize],
    ) -> Result<&SafeTensorEntry, String> {
        let entry = self.entries.get(name).ok_or_else(|| {
            format!(
                "Gemma4 safetensors {} is missing required tensor {name}",
                self.path.display()
            )
        })?;
        if entry.dtype != "BF16" {
            return Err(format!(
                "Gemma4 tensor {name} must be BF16, got {}",
                entry.dtype
            ));
        }
        if entry.shape != expected_shape {
            return Err(format!(
                "Gemma4 tensor {name} shape mismatch: expected {expected_shape:?}, got {:?}",
                entry.shape
            ));
        }
        let elements = expected_shape
            .iter()
            .try_fold(1_usize, |product, dimension| {
                product
                    .checked_mul(*dimension)
                    .ok_or_else(|| format!("Gemma4 tensor {name} element count overflows usize"))
            })?;
        let expected_bytes = elements
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or_else(|| format!("Gemma4 tensor {name} byte count overflows usize"))?;
        let actual_bytes = entry
            .data_end
            .checked_sub(entry.data_start)
            .ok_or_else(|| format!("Gemma4 tensor {name} has invalid offsets"))?;
        if actual_bytes
            != u64::try_from(expected_bytes)
                .map_err(|_| format!("Gemma4 tensor {name} expected byte count exceeds u64"))?
        {
            return Err(format!(
                "Gemma4 tensor {name} payload length mismatch: expected {expected_bytes}, got {actual_bytes}"
            ));
        }
        Ok(entry)
    }

    fn read_bf16(&mut self, name: &str, expected_shape: &[usize]) -> Result<Vec<u8>, String> {
        let entry = self.require_bf16_shape(name, expected_shape)?.clone();
        let byte_count = usize::try_from(entry.data_end - entry.data_start)
            .map_err(|_| format!("Gemma4 tensor {name} payload exceeds host usize"))?;
        let absolute_offset = self
            .payload_start
            .checked_add(entry.data_start)
            .ok_or_else(|| format!("Gemma4 tensor {name} file offset overflows"))?;
        self.file
            .seek(SeekFrom::Start(absolute_offset))
            .map_err(|error| format!("failed to seek Gemma4 tensor {name}: {error}"))?;
        let mut bytes = vec![0_u8; byte_count];
        self.file
            .read_exact(&mut bytes)
            .map_err(|error| format!("failed to read Gemma4 tensor {name}: {error}"))?;
        Ok(bytes)
    }

    fn read_bf16_row(
        &mut self,
        name: &str,
        rows: usize,
        columns: usize,
        row_index: usize,
    ) -> Result<Vec<f32>, String> {
        if row_index >= rows {
            return Err(format!(
                "Gemma4 tensor {name} row {row_index} is outside 0..{rows}"
            ));
        }
        let entry = self.require_bf16_shape(name, &[rows, columns])?.clone();
        let row_bytes = columns
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or_else(|| format!("Gemma4 tensor {name} row byte count overflows"))?;
        let row_offset = row_index
            .checked_mul(row_bytes)
            .ok_or_else(|| format!("Gemma4 tensor {name} row offset overflows"))?;
        let absolute_offset = self
            .payload_start
            .checked_add(entry.data_start)
            .and_then(|offset| offset.checked_add(u64::try_from(row_offset).ok()?))
            .ok_or_else(|| format!("Gemma4 tensor {name} row file offset overflows"))?;
        self.file
            .seek(SeekFrom::Start(absolute_offset))
            .map_err(|error| format!("failed to seek Gemma4 tensor {name} row: {error}"))?;
        let mut bytes = vec![0_u8; row_bytes];
        self.file
            .read_exact(&mut bytes)
            .map_err(|error| format!("failed to read Gemma4 tensor {name} row: {error}"))?;
        bf16_bytes_to_f32(&bytes, name)
    }

    fn read_bf16_vector(&mut self, name: &str, elements: usize) -> Result<Vec<f32>, String> {
        let bytes = self.read_bf16(name, &[elements])?;
        bf16_bytes_to_f32(&bytes, name)
    }
}

struct Bf16MatvecRuntime {
    context: RuntimeContext,
    stream: RuntimeStream,
    device: Gemma4TextDeviceIdentity,
}

impl Bf16MatvecRuntime {
    fn create() -> Result<Self, String> {
        if env::var(GEMMA4_TEXT_REQUIRED_HIP_KERNEL_ENV)
            .ok()
            .as_deref()
            != Some("1")
        {
            return Err(format!(
                "Gemma4TextExecutor requires {GEMMA4_TEXT_REQUIRED_HIP_KERNEL_ENV}=1 to forbid host-staging fallback"
            ));
        }
        let count = device_count()?;
        let mut candidates = Vec::new();
        for runtime_index in 0..count {
            let info = device_info(runtime_index).map_err(|error| {
                format!("failed to inspect runtime device {runtime_index}: {error}")
            })?;
            let identity = Gemma4TextDeviceIdentity::from_runtime(runtime_index, info);
            if identity.backend == "hip" {
                if identity.validate_r9700().is_ok() {
                    candidates.push(identity);
                } else if identity
                    .gcn_arch_name
                    .split(':')
                    .next()
                    .unwrap_or_default()
                    .eq_ignore_ascii_case("gfx1030")
                {
                    // Explicitly never select the passively cooled V620.
                    continue;
                }
            }
        }
        if candidates.len() != 1 {
            return Err(format!(
                "Gemma4TextExecutor requires exactly one selectable R9700/gfx1201, found {}",
                candidates.len()
            ));
        }
        let device = candidates.pop().expect("checked candidate count");
        let mut context = RuntimeContext::create(device.runtime_index)?;
        let context_device =
            Gemma4TextDeviceIdentity::from_runtime(device.runtime_index, context.device_info()?);
        context_device.validate_r9700()?;
        let stream = context.create_stream()?;
        Ok(Self {
            context,
            stream,
            device: context_device,
        })
    }

    fn device(&self) -> &Gemma4TextDeviceIdentity {
        &self.device
    }

    fn matvec(
        &mut self,
        matrix: &[u8],
        rows: usize,
        columns: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        if input.len() != columns {
            return Err(format!(
                "BF16 matvec input width mismatch: expected {columns}, got {}",
                input.len()
            ));
        }
        let expected_matrix_bytes = rows
            .checked_mul(columns)
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<u16>()))
            .ok_or_else(|| "BF16 matvec matrix byte count overflows".to_string())?;
        if matrix.len() != expected_matrix_bytes {
            return Err(format!(
                "BF16 matvec matrix byte length mismatch: expected {expected_matrix_bytes}, got {}",
                matrix.len()
            ));
        }
        let input_bytes = encode_f32_to_bytes(input);
        let output_bytes = rows
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "BF16 matvec output byte count overflows".to_string())?;
        let mut matrix_buffer = self.context.alloc_buffer(expected_matrix_bytes)?;
        let mut input_buffer = self.context.alloc_buffer(input_bytes.len())?;
        let mut output_buffer = self.context.alloc_buffer(output_bytes)?;
        matrix_buffer.copy_from_host(0, matrix, Some(&mut self.stream))?;
        input_buffer.copy_from_host(0, &input_bytes, Some(&mut self.stream))?;
        matvec_bf16_f32(
            &matrix_buffer,
            &input_buffer,
            rows,
            columns,
            &mut output_buffer,
            Some(&mut self.stream),
        )?;
        let mut host_output = vec![0_u8; output_bytes];
        output_buffer.copy_to_host(0, &mut host_output, Some(&mut self.stream))?;
        self.stream.synchronize()?;
        let output = decode_f32_le_values(&host_output);
        if output.len() != rows || output.iter().any(|value| !value.is_finite()) {
            return Err("Gemma4 BF16 matvec returned non-finite or malformed F32 output".into());
        }
        Ok(output)
    }
}

#[derive(Debug, Clone, Default)]
struct KvSequence {
    width: usize,
    key: Vec<f32>,
    value: Vec<f32>,
}

impl KvSequence {
    fn append(&mut self, key: &[f32], value: &[f32]) -> Result<(), String> {
        if key.len() != value.len() || key.is_empty() {
            return Err("Gemma4 KV append requires equal nonempty key/value widths".into());
        }
        if self.width == 0 {
            self.width = key.len();
        }
        if self.width != key.len() {
            return Err(format!(
                "Gemma4 KV cache width changed from {} to {}",
                self.width,
                key.len()
            ));
        }
        self.key.extend_from_slice(key);
        self.value.extend_from_slice(value);
        Ok(())
    }

    fn len(&self) -> Result<usize, String> {
        if self.width == 0
            || self.key.len() != self.value.len()
            || !self.key.len().is_multiple_of(self.width)
        {
            return Err("Gemma4 KV cache has invalid storage shape".into());
        }
        Ok(self.key.len() / self.width)
    }
}

#[derive(Debug)]
struct Gemma4KvCaches {
    per_layer: Vec<Option<KvSequence>>,
    shared_sliding: Option<KvSequence>,
    shared_full: Option<KvSequence>,
}

impl Gemma4KvCaches {
    fn new(layer_count: usize) -> Self {
        Self {
            per_layer: vec![None; layer_count],
            shared_sliding: None,
            shared_full: None,
        }
    }

    fn shared(&self, layer_kind: DecoderLayerKind) -> Option<&KvSequence> {
        match layer_kind {
            DecoderLayerKind::SlidingAttention => self.shared_sliding.as_ref(),
            DecoderLayerKind::FullAttention => self.shared_full.as_ref(),
            DecoderLayerKind::LinearAttention => None,
        }
    }

    fn store_shared(
        &mut self,
        layer_kind: DecoderLayerKind,
        source: KvSequence,
    ) -> Result<(), String> {
        match layer_kind {
            DecoderLayerKind::SlidingAttention => self.shared_sliding = Some(source),
            DecoderLayerKind::FullAttention => self.shared_full = Some(source),
            DecoderLayerKind::LinearAttention => {
                return Err("Gemma4 does not have linear-attention KV sharing".into());
            }
        }
        Ok(())
    }
}

/// One causal Gemma4 text sequence over source BF16 weights and F32
/// activations. `load` creates a new R9700 HIP context; callers should call
/// `reset` before an unrelated request.
pub struct Gemma4TextExecutor {
    source_model_dir: PathBuf,
    config_sha256: String,
    config: Gemma4TextConfig,
    weights: SafeTensorReader,
    matvec: Bf16MatvecRuntime,
    caches: Gemma4KvCaches,
    position: usize,
}

impl Gemma4TextExecutor {
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self, String> {
        let loaded = load_model_config_from_dir(model_dir)?;
        Self::from_loaded_config(loaded)
    }

    pub fn from_loaded_config(loaded: LoadedModelConfig) -> Result<Self, String> {
        let config = loaded.require_gemma4_text_executor()?.clone();
        let model_path = loaded.source_model_dir.join(GEMMA4_TEXT_MODEL_FILE);
        let weights = SafeTensorReader::open(&model_path)?;
        validate_checkpoint_contract(&weights, &config)?;
        let matvec = Bf16MatvecRuntime::create()?;
        Ok(Self {
            source_model_dir: loaded.source_model_dir,
            config_sha256: loaded.config_sha256,
            caches: Gemma4KvCaches::new(config.decoder.num_hidden_layers),
            config,
            weights,
            matvec,
            position: 0,
        })
    }

    pub fn config(&self) -> &Gemma4TextConfig {
        &self.config
    }

    pub fn source_model_dir(&self) -> &Path {
        &self.source_model_dir
    }

    pub fn config_sha256(&self) -> &str {
        &self.config_sha256
    }

    pub fn device(&self) -> &Gemma4TextDeviceIdentity {
        self.matvec.device()
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn reset(&mut self) {
        self.caches = Gemma4KvCaches::new(self.config.decoder.num_hidden_layers);
        self.position = 0;
    }

    /// Executes one causal input chunk. The chunk is evaluated token-by-token
    /// to preserve a simple explicit KV cache, but returned rows are laid out
    /// exactly as `[batch=1, tokens, width]` for the trace writer.
    pub fn execute_step(&mut self, input_token_ids: &[u32]) -> Result<Gemma4TextStepTrace, String> {
        if input_token_ids.is_empty() {
            return Err("Gemma4TextExecutor input token list must be nonempty".into());
        }
        let next_position = self
            .position
            .checked_add(input_token_ids.len())
            .ok_or_else(|| "Gemma4 sequence position overflows usize".to_string())?;
        if next_position > self.config.max_position_embeddings {
            return Err(format!(
                "Gemma4 input reaches position {next_position}, beyond max_position_embeddings={}",
                self.config.max_position_embeddings
            ));
        }
        let hidden = self.config.decoder.hidden_size;
        let layers = self.config.decoder.num_hidden_layers;
        let mut embedding = Vec::with_capacity(
            input_token_ids
                .len()
                .checked_mul(hidden)
                .ok_or_else(|| "Gemma4 embedding trace allocation overflows".to_string())?,
        );
        let mut layer_outputs = (0..layers)
            .map(|_| Vec::with_capacity(input_token_ids.len() * hidden))
            .collect::<Vec<_>>();
        let mut final_norm = Vec::with_capacity(input_token_ids.len() * hidden);
        let mut final_token_hidden = None;
        for token_id in input_token_ids {
            let token_index = usize::try_from(*token_id)
                .map_err(|_| "Gemma4 token ID does not fit usize".to_string())?;
            if token_index >= self.config.decoder.vocab_size {
                return Err(format!(
                    "Gemma4 token ID {token_id} is outside vocabulary 0..{}",
                    self.config.decoder.vocab_size
                ));
            }
            if token_index >= self.config.vocab_size_per_layer_input {
                return Err(format!(
                    "Gemma4 token ID {token_id} is outside PLE vocabulary 0..{}",
                    self.config.vocab_size_per_layer_input
                ));
            }
            let token = self.forward_token(*token_id)?;
            embedding.extend_from_slice(&token.embedding);
            for (layer_index, output) in token.layer_outputs.iter().enumerate() {
                layer_outputs[layer_index].extend_from_slice(output);
            }
            final_norm.extend_from_slice(&token.final_norm);
            final_token_hidden = Some(token.final_norm);
        }
        let final_token_hidden = final_token_hidden.expect("checked nonempty input");
        let logits_last = self.project_tied_logits(&final_token_hidden)?;
        let top1 = top1_from_logits(&logits_last)?;
        Ok(Gemma4TextStepTrace {
            input_token_ids: input_token_ids.to_vec(),
            embedding,
            layer_outputs,
            final_norm,
            logits_last,
            top1,
        })
    }

    fn forward_token(&mut self, token_id: u32) -> Result<TokenForward, String> {
        let token_index = usize::try_from(token_id)
            .map_err(|_| "Gemma4 token ID does not fit usize".to_string())?;
        let hidden = self.config.decoder.hidden_size;
        let layers = self.config.decoder.num_hidden_layers;
        let ple_dim = self.config.hidden_size_per_layer_input;
        let embedding_scale = (hidden as f32).sqrt();
        let mut embedding = self.weights.read_bf16_row(
            GEMMA4_TEXT_EMBED_TOKENS,
            self.config.decoder.vocab_size,
            hidden,
            token_index,
        )?;
        scale_in_place(&mut embedding, embedding_scale, "Gemma4 embedding scale")?;
        let per_layer_inputs = self.compute_ple(token_index, &embedding)?;
        if per_layer_inputs.len() != layers * ple_dim {
            return Err("Gemma4 PLE result has an invalid packed width".into());
        }
        let mut hidden_states = embedding.clone();
        let mut layer_outputs = Vec::with_capacity(layers);
        let position = self.position;
        for layer_index in 0..layers {
            let ple_start = layer_index
                .checked_mul(ple_dim)
                .ok_or_else(|| "Gemma4 PLE layer offset overflows".to_string())?;
            let ple_end = ple_start
                .checked_add(ple_dim)
                .ok_or_else(|| "Gemma4 PLE layer end overflows".to_string())?;
            hidden_states = self.forward_layer(
                layer_index,
                &hidden_states,
                &per_layer_inputs[ple_start..ple_end],
                position,
            )?;
            layer_outputs.push(hidden_states.clone());
        }
        let final_weight = self
            .weights
            .read_bf16_vector(GEMMA4_TEXT_FINAL_NORM, hidden)?;
        let final_norm = rms_norm(
            &hidden_states,
            Some(&final_weight),
            self.config.decoder.rms_norm_eps,
        )?;
        self.position = self
            .position
            .checked_add(1)
            .ok_or_else(|| "Gemma4 sequence position overflows usize".to_string())?;
        Ok(TokenForward {
            embedding,
            layer_outputs,
            final_norm,
        })
    }

    fn compute_ple(&mut self, token_index: usize, embedding: &[f32]) -> Result<Vec<f32>, String> {
        let layers = self.config.decoder.num_hidden_layers;
        let ple_dim = self.config.hidden_size_per_layer_input;
        let packed_width = layers
            .checked_mul(ple_dim)
            .ok_or_else(|| "Gemma4 PLE packed width overflows".to_string())?;
        let mut token_identity = self.weights.read_bf16_row(
            GEMMA4_TEXT_EMBED_TOKENS_PER_LAYER,
            self.config.vocab_size_per_layer_input,
            packed_width,
            token_index,
        )?;
        scale_in_place(
            &mut token_identity,
            (ple_dim as f32).sqrt(),
            "Gemma4 PLE token embedding scale",
        )?;
        let mut projected = self.matmul_named(
            GEMMA4_TEXT_PER_LAYER_MODEL_PROJECTION,
            packed_width,
            self.config.decoder.hidden_size,
            embedding,
        )?;
        scale_in_place(
            &mut projected,
            (self.config.decoder.hidden_size as f32).sqrt().recip(),
            "Gemma4 PLE model projection scale",
        )?;
        let norm_weight = self
            .weights
            .read_bf16_vector(GEMMA4_TEXT_PER_LAYER_PROJECTION_NORM, ple_dim)?;
        let combine_scale = 2.0_f32.powf(-0.5);
        for layer_index in 0..layers {
            let start = layer_index
                .checked_mul(ple_dim)
                .ok_or_else(|| "Gemma4 PLE slice offset overflows".to_string())?;
            let end = start
                .checked_add(ple_dim)
                .ok_or_else(|| "Gemma4 PLE slice end overflows".to_string())?;
            let normalized = rms_norm(
                &projected[start..end],
                Some(&norm_weight),
                self.config.decoder.rms_norm_eps,
            )?;
            for (index, value) in projected[start..end].iter_mut().enumerate() {
                *value = (normalized[index] + token_identity[start + index]) * combine_scale;
            }
        }
        finite_slice(&projected, "Gemma4 PLE")?;
        Ok(projected)
    }

    fn forward_layer(
        &mut self,
        layer_index: usize,
        hidden_states: &[f32],
        per_layer_input: &[f32],
        position: usize,
    ) -> Result<Vec<f32>, String> {
        let hidden = self.config.decoder.hidden_size;
        if hidden_states.len() != hidden {
            return Err(format!(
                "Gemma4 layer {layer_index} input width mismatch: expected {hidden}, got {}",
                hidden_states.len()
            ));
        }
        if per_layer_input.len() != self.config.hidden_size_per_layer_input {
            return Err(format!(
                "Gemma4 layer {layer_index} PLE width mismatch: expected {}, got {}",
                self.config.hidden_size_per_layer_input,
                per_layer_input.len()
            ));
        }
        let residual_attention = hidden_states.to_vec();
        let input_norm_weight = self
            .weights
            .read_bf16_vector(&layer_tensor(layer_index, "input_layernorm.weight"), hidden)?;
        let input_norm = rms_norm(
            hidden_states,
            Some(&input_norm_weight),
            self.config.decoder.rms_norm_eps,
        )?;
        let attention = self.forward_attention(layer_index, &input_norm, position)?;
        let post_attention_weight = self.weights.read_bf16_vector(
            &layer_tensor(layer_index, "post_attention_layernorm.weight"),
            hidden,
        )?;
        let post_attention = rms_norm(
            &attention,
            Some(&post_attention_weight),
            self.config.decoder.rms_norm_eps,
        )?;
        let attention_residual = add_vectors(
            &residual_attention,
            &post_attention,
            "Gemma4 attention residual",
        )?;

        let residual_mlp = attention_residual.clone();
        let pre_feedforward_weight = self.weights.read_bf16_vector(
            &layer_tensor(layer_index, "pre_feedforward_layernorm.weight"),
            hidden,
        )?;
        let feedforward_input = rms_norm(
            &attention_residual,
            Some(&pre_feedforward_weight),
            self.config.decoder.rms_norm_eps,
        )?;
        let mlp = self.forward_mlp(layer_index, &feedforward_input)?;
        let post_feedforward_weight = self.weights.read_bf16_vector(
            &layer_tensor(layer_index, "post_feedforward_layernorm.weight"),
            hidden,
        )?;
        let post_feedforward = rms_norm(
            &mlp,
            Some(&post_feedforward_weight),
            self.config.decoder.rms_norm_eps,
        )?;
        let mlp_residual = add_vectors(&residual_mlp, &post_feedforward, "Gemma4 MLP residual")?;

        let ple_gate = self.matmul_named(
            &layer_tensor(layer_index, "per_layer_input_gate.weight"),
            self.config.hidden_size_per_layer_input,
            hidden,
            &mlp_residual,
        )?;
        let mut ple_product = gelu_pytorch_tanh(&ple_gate)?;
        multiply_in_place(&mut ple_product, per_layer_input, "Gemma4 PLE gate product")?;
        let ple_projection = self.matmul_named(
            &layer_tensor(layer_index, "per_layer_projection.weight"),
            hidden,
            self.config.hidden_size_per_layer_input,
            &ple_product,
        )?;
        let post_ple_weight = self.weights.read_bf16_vector(
            &layer_tensor(layer_index, "post_per_layer_input_norm.weight"),
            hidden,
        )?;
        let post_ple = rms_norm(
            &ple_projection,
            Some(&post_ple_weight),
            self.config.decoder.rms_norm_eps,
        )?;
        let mut output = add_vectors(&mlp_residual, &post_ple, "Gemma4 PLE residual")?;
        let layer_scalar = self
            .weights
            .read_bf16_vector(&layer_tensor(layer_index, "layer_scalar"), 1)?[0];
        scale_in_place(&mut output, layer_scalar, "Gemma4 layer scalar")?;
        Ok(output)
    }

    fn forward_attention(
        &mut self,
        layer_index: usize,
        hidden_states: &[f32],
        position: usize,
    ) -> Result<Vec<f32>, String> {
        let layer_kind = *self
            .config
            .layer_types
            .get(layer_index)
            .ok_or_else(|| format!("Gemma4 layer type missing at {layer_index}"))?;
        let head_dim = match layer_kind {
            DecoderLayerKind::SlidingAttention => self.config.local_head_dim,
            DecoderLayerKind::FullAttention => self.config.global_head_dim,
            DecoderLayerKind::LinearAttention => {
                return Err("Gemma4TextExecutor does not implement linear attention".into());
            }
        };
        let q_heads = self.config.decoder.num_attention_heads;
        let kv_heads = self.config.decoder.num_key_value_heads;
        let q_width = q_heads
            .checked_mul(head_dim)
            .ok_or_else(|| "Gemma4 q width overflows".to_string())?;
        let kv_width = kv_heads
            .checked_mul(head_dim)
            .ok_or_else(|| "Gemma4 KV width overflows".to_string())?;
        let hidden = self.config.decoder.hidden_size;
        let q_raw = self.matmul_named(
            &layer_tensor(layer_index, "self_attn.q_proj.weight"),
            q_width,
            hidden,
            hidden_states,
        )?;
        let q_norm_weight = self.weights.read_bf16_vector(
            &layer_tensor(layer_index, "self_attn.q_norm.weight"),
            head_dim,
        )?;
        let mut query = rms_norm_heads(
            &q_raw,
            q_heads,
            head_dim,
            Some(&q_norm_weight),
            self.config.decoder.rms_norm_eps,
        )?;
        apply_gemma4_rope_in_place(
            &mut query,
            q_heads,
            head_dim,
            layer_kind,
            position,
            &self.config,
        )?;

        let first_shared = self
            .config
            .decoder
            .num_hidden_layers
            .checked_sub(self.config.num_kv_shared_layers)
            .ok_or_else(|| "Gemma4 KV shared layer count exceeds layer count".to_string())?;
        let is_shared_layer = layer_index >= first_shared && self.config.num_kv_shared_layers > 0;
        let cache = if is_shared_layer {
            self.caches.shared(layer_kind).cloned().ok_or_else(|| {
                format!(
                    "Gemma4 layer {layer_index} needs shared {} K/V before its source layer ran",
                    layer_kind.as_str()
                )
            })?
        } else {
            let key_raw = self.matmul_named(
                &layer_tensor(layer_index, "self_attn.k_proj.weight"),
                kv_width,
                hidden,
                hidden_states,
            )?;
            let value_raw = self.matmul_named(
                &layer_tensor(layer_index, "self_attn.v_proj.weight"),
                kv_width,
                hidden,
                hidden_states,
            )?;
            let k_norm_weight = self.weights.read_bf16_vector(
                &layer_tensor(layer_index, "self_attn.k_norm.weight"),
                head_dim,
            )?;
            let mut key = rms_norm_heads(
                &key_raw,
                kv_heads,
                head_dim,
                Some(&k_norm_weight),
                self.config.decoder.rms_norm_eps,
            )?;
            apply_gemma4_rope_in_place(
                &mut key,
                kv_heads,
                head_dim,
                layer_kind,
                position,
                &self.config,
            )?;
            let value = rms_norm_heads(
                &value_raw,
                kv_heads,
                head_dim,
                None,
                self.config.decoder.rms_norm_eps,
            )?;
            let source_cache = {
                let layer_cache = self
                    .caches
                    .per_layer
                    .get_mut(layer_index)
                    .ok_or_else(|| format!("Gemma4 KV cache has no layer {layer_index}"))?
                    .get_or_insert_with(KvSequence::default);
                layer_cache.append(&key, &value)?;
                layer_cache.clone()
            };
            if self.should_store_shared_kv(layer_index) {
                self.caches.store_shared(layer_kind, source_cache.clone())?;
            }
            source_cache
        };
        let attention_output = causal_attention(
            &query,
            &cache,
            q_heads,
            kv_heads,
            head_dim,
            matches!(layer_kind, DecoderLayerKind::SlidingAttention)
                .then_some(self.config.sliding_window),
        )?;
        self.matmul_named(
            &layer_tensor(layer_index, "self_attn.o_proj.weight"),
            hidden,
            q_width,
            &attention_output,
        )
    }

    fn should_store_shared_kv(&self, layer_index: usize) -> bool {
        let first_shared = self
            .config
            .decoder
            .num_hidden_layers
            .saturating_sub(self.config.num_kv_shared_layers);
        if layer_index >= first_shared || self.config.num_kv_shared_layers == 0 {
            return false;
        }
        let kind = self.config.layer_types[layer_index];
        self.config.layer_types[layer_index + 1..first_shared]
            .iter()
            .all(|candidate| *candidate != kind)
    }

    fn forward_mlp(&mut self, layer_index: usize, input: &[f32]) -> Result<Vec<f32>, String> {
        let first_shared = self
            .config
            .decoder
            .num_hidden_layers
            .checked_sub(self.config.num_kv_shared_layers)
            .ok_or_else(|| "Gemma4 KV shared layer count exceeds layer count".to_string())?;
        let use_double_wide =
            self.config.use_double_wide_mlp && layer_index >= first_shared && first_shared > 0;
        let intermediate = self
            .config
            .dense_mlp
            .intermediate_size
            .checked_mul(if use_double_wide { 2 } else { 1 })
            .ok_or_else(|| "Gemma4 MLP intermediate width overflows".to_string())?;
        let hidden = self.config.decoder.hidden_size;
        let gate = self.matmul_named(
            &layer_tensor(layer_index, "mlp.gate_proj.weight"),
            intermediate,
            hidden,
            input,
        )?;
        let up = self.matmul_named(
            &layer_tensor(layer_index, "mlp.up_proj.weight"),
            intermediate,
            hidden,
            input,
        )?;
        let mut activated = gelu_pytorch_tanh(&gate)?;
        multiply_in_place(&mut activated, &up, "Gemma4 gated MLP product")?;
        self.matmul_named(
            &layer_tensor(layer_index, "mlp.down_proj.weight"),
            hidden,
            intermediate,
            &activated,
        )
    }

    fn project_tied_logits(&mut self, final_hidden: &[f32]) -> Result<Vec<f32>, String> {
        let mut logits = self.matmul_named(
            GEMMA4_TEXT_EMBED_TOKENS,
            self.config.decoder.vocab_size,
            self.config.decoder.hidden_size,
            final_hidden,
        )?;
        let cap = self.config.final_logit_softcapping;
        for value in &mut logits {
            *value = (*value / cap).tanh() * cap;
        }
        finite_slice(&logits, "Gemma4 final soft-capped logits")?;
        Ok(logits)
    }

    fn matmul_named(
        &mut self,
        name: &str,
        rows: usize,
        columns: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        finite_slice(input, &format!("Gemma4 matvec input {name}"))?;
        let matrix = self.weights.read_bf16(name, &[rows, columns])?;
        self.matvec.matvec(&matrix, rows, columns, input)
    }
}

#[derive(Debug)]
struct TokenForward {
    embedding: Vec<f32>,
    layer_outputs: Vec<Vec<f32>>,
    final_norm: Vec<f32>,
}

fn validate_checkpoint_contract(
    weights: &SafeTensorReader,
    config: &Gemma4TextConfig,
) -> Result<(), String> {
    config.validate_nonquantized_executor()?;
    let hidden = config.decoder.hidden_size;
    let layers = config.decoder.num_hidden_layers;
    let ple_dim = config.hidden_size_per_layer_input;
    let packed_ple = layers
        .checked_mul(ple_dim)
        .ok_or_else(|| "Gemma4 PLE packed width overflows".to_string())?;
    weights.require_bf16_shape(
        GEMMA4_TEXT_EMBED_TOKENS,
        &[config.decoder.vocab_size, hidden],
    )?;
    weights.require_bf16_shape(
        GEMMA4_TEXT_EMBED_TOKENS_PER_LAYER,
        &[config.vocab_size_per_layer_input, packed_ple],
    )?;
    weights.require_bf16_shape(
        GEMMA4_TEXT_PER_LAYER_MODEL_PROJECTION,
        &[packed_ple, hidden],
    )?;
    weights.require_bf16_shape(GEMMA4_TEXT_PER_LAYER_PROJECTION_NORM, &[ple_dim])?;
    weights.require_bf16_shape(GEMMA4_TEXT_FINAL_NORM, &[hidden])?;

    let first_shared = layers
        .checked_sub(config.num_kv_shared_layers)
        .ok_or_else(|| "Gemma4 KV shared layer count exceeds layer count".to_string())?;
    for layer_index in 0..layers {
        let layer_kind = config.layer_types[layer_index];
        let head_dim = match layer_kind {
            DecoderLayerKind::SlidingAttention => config.local_head_dim,
            DecoderLayerKind::FullAttention => config.global_head_dim,
            DecoderLayerKind::LinearAttention => {
                return Err("Gemma4 checkpoint has unsupported linear-attention layer".into());
            }
        };
        let q_width = config
            .decoder
            .num_attention_heads
            .checked_mul(head_dim)
            .ok_or_else(|| "Gemma4 q width overflows".to_string())?;
        let kv_width = config
            .decoder
            .num_key_value_heads
            .checked_mul(head_dim)
            .ok_or_else(|| "Gemma4 KV width overflows".to_string())?;
        for norm in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight",
            "post_feedforward_layernorm.weight",
            "post_per_layer_input_norm.weight",
        ] {
            weights.require_bf16_shape(&layer_tensor(layer_index, norm), &[hidden])?;
        }
        weights.require_bf16_shape(&layer_tensor(layer_index, "layer_scalar"), &[1])?;
        weights.require_bf16_shape(
            &layer_tensor(layer_index, "self_attn.q_proj.weight"),
            &[q_width, hidden],
        )?;
        weights.require_bf16_shape(
            &layer_tensor(layer_index, "self_attn.q_norm.weight"),
            &[head_dim],
        )?;
        weights.require_bf16_shape(
            &layer_tensor(layer_index, "self_attn.o_proj.weight"),
            &[hidden, q_width],
        )?;
        // HF ignores the physical K/V tensors of shared layers.  Require the
        // tensors only for layers whose modules actually own those projections.
        if layer_index < first_shared {
            weights.require_bf16_shape(
                &layer_tensor(layer_index, "self_attn.k_proj.weight"),
                &[kv_width, hidden],
            )?;
            weights.require_bf16_shape(
                &layer_tensor(layer_index, "self_attn.v_proj.weight"),
                &[kv_width, hidden],
            )?;
            weights.require_bf16_shape(
                &layer_tensor(layer_index, "self_attn.k_norm.weight"),
                &[head_dim],
            )?;
        }
        let double_wide =
            config.use_double_wide_mlp && layer_index >= first_shared && first_shared > 0;
        let intermediate = config
            .dense_mlp
            .intermediate_size
            .checked_mul(if double_wide { 2 } else { 1 })
            .ok_or_else(|| "Gemma4 MLP intermediate width overflows".to_string())?;
        weights.require_bf16_shape(
            &layer_tensor(layer_index, "mlp.gate_proj.weight"),
            &[intermediate, hidden],
        )?;
        weights.require_bf16_shape(
            &layer_tensor(layer_index, "mlp.up_proj.weight"),
            &[intermediate, hidden],
        )?;
        weights.require_bf16_shape(
            &layer_tensor(layer_index, "mlp.down_proj.weight"),
            &[hidden, intermediate],
        )?;
        weights.require_bf16_shape(
            &layer_tensor(layer_index, "per_layer_input_gate.weight"),
            &[ple_dim, hidden],
        )?;
        weights.require_bf16_shape(
            &layer_tensor(layer_index, "per_layer_projection.weight"),
            &[hidden, ple_dim],
        )?;
    }
    Ok(())
}

fn layer_tensor(layer_index: usize, suffix: &str) -> String {
    format!("{GEMMA4_TEXT_WEIGHT_PREFIX}layers.{layer_index}.{suffix}")
}

fn bf16_bytes_to_f32(bytes: &[u8], label: &str) -> Result<Vec<f32>, String> {
    if !bytes.len().is_multiple_of(std::mem::size_of::<u16>()) {
        return Err(format!("{label} BF16 payload has an odd byte length"));
    }
    let values = bytes
        .chunks_exact(std::mem::size_of::<u16>())
        .map(|chunk| {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            f32::from_bits(u32::from(bits) << 16)
        })
        .collect::<Vec<_>>();
    finite_slice(&values, label)?;
    Ok(values)
}

fn finite_slice(values: &[f32], label: &str) -> Result<(), String> {
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!(
            "{label} contains non-finite value at {index}: {value}"
        ));
    }
    Ok(())
}

fn scale_in_place(values: &mut [f32], scale: f32, label: &str) -> Result<(), String> {
    if !scale.is_finite() {
        return Err(format!("{label} is non-finite: {scale}"));
    }
    for value in values.iter_mut() {
        *value *= scale;
    }
    finite_slice(values, label)
}

fn add_vectors(left: &[f32], right: &[f32], label: &str) -> Result<Vec<f32>, String> {
    if left.len() != right.len() {
        return Err(format!(
            "{label} width mismatch: left={} right={}",
            left.len(),
            right.len()
        ));
    }
    let output = left
        .iter()
        .zip(right)
        .map(|(left, right)| left + right)
        .collect::<Vec<_>>();
    finite_slice(&output, label)?;
    Ok(output)
}

fn multiply_in_place(values: &mut [f32], other: &[f32], label: &str) -> Result<(), String> {
    if values.len() != other.len() {
        return Err(format!(
            "{label} width mismatch: left={} right={}",
            values.len(),
            other.len()
        ));
    }
    for (value, other) in values.iter_mut().zip(other) {
        *value *= *other;
    }
    finite_slice(values, label)
}

fn rms_norm(values: &[f32], weight: Option<&[f32]>, epsilon: f32) -> Result<Vec<f32>, String> {
    if values.is_empty() || !epsilon.is_finite() || epsilon <= 0.0 {
        return Err("Gemma4 RMSNorm needs nonempty input and finite positive epsilon".into());
    }
    if let Some(weight) = weight
        && weight.len() != values.len()
    {
        return Err(format!(
            "Gemma4 RMSNorm weight width mismatch: values={} weight={}",
            values.len(),
            weight.len()
        ));
    }
    finite_slice(values, "Gemma4 RMSNorm input")?;
    if let Some(weight) = weight {
        finite_slice(weight, "Gemma4 RMSNorm weight")?;
    }
    let mut mean_squared = 0.0_f32;
    for value in values {
        mean_squared += value * value;
    }
    mean_squared = mean_squared / values.len() as f32 + epsilon;
    let inverse = mean_squared.powf(-0.5);
    if !inverse.is_finite() {
        return Err("Gemma4 RMSNorm inverse RMS is non-finite".into());
    }
    let output: Vec<f32> = match weight {
        Some(weight) => values
            .iter()
            .zip(weight)
            .map(|(value, weight)| value * inverse * weight)
            .collect(),
        None => values.iter().map(|value| value * inverse).collect(),
    };
    finite_slice(&output, "Gemma4 RMSNorm output")?;
    Ok(output)
}

fn rms_norm_heads(
    values: &[f32],
    heads: usize,
    head_dim: usize,
    weight: Option<&[f32]>,
    epsilon: f32,
) -> Result<Vec<f32>, String> {
    let expected = heads
        .checked_mul(head_dim)
        .ok_or_else(|| "Gemma4 RMSNorm head shape overflows".to_string())?;
    if values.len() != expected {
        return Err(format!(
            "Gemma4 head RMSNorm value width mismatch: expected {expected}, got {}",
            values.len()
        ));
    }
    if let Some(weight) = weight
        && weight.len() != head_dim
    {
        return Err(format!(
            "Gemma4 head RMSNorm weight width mismatch: expected {head_dim}, got {}",
            weight.len()
        ));
    }
    let mut output = Vec::with_capacity(expected);
    for head in values.chunks_exact(head_dim) {
        output.extend_from_slice(&rms_norm(head, weight, epsilon)?);
    }
    Ok(output)
}

fn gelu_pytorch_tanh(values: &[f32]) -> Result<Vec<f32>, String> {
    finite_slice(values, "Gemma4 GELU input")?;
    // Match Transformers' GELUTanh.forward operation order and its literal
    // 0.7978845608 coefficient, rather than substituting a mathematically
    // equivalent sqrt(2/pi) expression with a different F32 rounding path.
    let coefficient = 0.797_884_560_8_f32;
    let output = values
        .iter()
        .map(|value| {
            0.5 * value * (1.0 + (value * coefficient * (1.0 + 0.044_715 * value * value)).tanh())
        })
        .collect::<Vec<_>>();
    finite_slice(&output, "Gemma4 GELU output")?;
    Ok(output)
}

fn apply_gemma4_rope_in_place(
    values: &mut [f32],
    heads: usize,
    head_dim: usize,
    layer_kind: DecoderLayerKind,
    position: usize,
    config: &Gemma4TextConfig,
) -> Result<(), String> {
    if !head_dim.is_multiple_of(2) {
        return Err(format!("Gemma4 RoPE head_dim must be even, got {head_dim}"));
    }
    let expected = heads
        .checked_mul(head_dim)
        .ok_or_else(|| "Gemma4 RoPE shape overflows".to_string())?;
    if values.len() != expected {
        return Err(format!(
            "Gemma4 RoPE input width mismatch: expected {expected}, got {}",
            values.len()
        ));
    }
    let rope = match layer_kind {
        DecoderLayerKind::SlidingAttention => &config.sliding_rope,
        DecoderLayerKind::FullAttention => &config.full_rope,
        DecoderLayerKind::LinearAttention => {
            return Err("Gemma4 does not implement linear-attention RoPE".into());
        }
    };
    let half = head_dim / 2;
    let active_pairs = match layer_kind {
        DecoderLayerKind::SlidingAttention => half,
        DecoderLayerKind::FullAttention => {
            let partial = rope.partial_rotary_factor.unwrap_or(1.0);
            ((partial * head_dim as f32) / 2.0).floor() as usize
        }
        DecoderLayerKind::LinearAttention => unreachable!("rejected above"),
    };
    if active_pairs > half {
        return Err(format!(
            "Gemma4 RoPE active pair count {active_pairs} exceeds head half-width {half}"
        ));
    }
    let mut cosine = Vec::with_capacity(half);
    let mut sine = Vec::with_capacity(half);
    for pair in 0..half {
        let inverse_frequency = if pair < active_pairs {
            let exponent = (2 * pair) as f32 / head_dim as f32;
            rope.rope_theta.powf(exponent).recip()
        } else {
            0.0
        };
        let angle = inverse_frequency * position as f32;
        cosine.push(angle.cos());
        sine.push(angle.sin());
    }
    for head in values.chunks_exact_mut(head_dim) {
        for pair in 0..half {
            let first = head[pair];
            let second = head[half + pair];
            head[pair] = first * cosine[pair] - second * sine[pair];
            head[half + pair] = second * cosine[pair] + first * sine[pair];
        }
    }
    finite_slice(values, "Gemma4 RoPE output")
}

fn causal_attention(
    query: &[f32],
    cache: &KvSequence,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    sliding_window: Option<usize>,
) -> Result<Vec<f32>, String> {
    if q_heads == 0 || kv_heads == 0 || head_dim == 0 || !q_heads.is_multiple_of(kv_heads) {
        return Err("Gemma4 attention has invalid head geometry".into());
    }
    let expected_q_width = q_heads
        .checked_mul(head_dim)
        .ok_or_else(|| "Gemma4 attention query width overflows".to_string())?;
    let expected_kv_width = kv_heads
        .checked_mul(head_dim)
        .ok_or_else(|| "Gemma4 attention KV width overflows".to_string())?;
    if query.len() != expected_q_width || cache.width != expected_kv_width {
        return Err(format!(
            "Gemma4 attention width mismatch: query={} expected_q={} cache={} expected_kv={}",
            query.len(),
            expected_q_width,
            cache.width,
            expected_kv_width
        ));
    }
    let cache_len = cache.len()?;
    if cache_len == 0 {
        return Err("Gemma4 attention has no causal KV entries".into());
    }
    let first_key = match sliding_window {
        Some(window) => {
            if window == 0 {
                return Err("Gemma4 sliding attention window must be nonzero".into());
            }
            cache_len.saturating_sub(window)
        }
        None => 0,
    };
    let key_count = cache_len
        .checked_sub(first_key)
        .ok_or_else(|| "Gemma4 attention key range underflow".to_string())?;
    let groups = q_heads / kv_heads;
    let mut output = vec![0.0_f32; expected_q_width];
    for q_head in 0..q_heads {
        let query_head = &query[q_head * head_dim..(q_head + 1) * head_dim];
        let kv_head = q_head / groups;
        let mut scores = Vec::with_capacity(key_count);
        let mut maximum = f32::NEG_INFINITY;
        for key_index in first_key..cache_len {
            let base = key_index
                .checked_mul(cache.width)
                .and_then(|offset| offset.checked_add(kv_head * head_dim))
                .ok_or_else(|| "Gemma4 attention key offset overflows".to_string())?;
            let mut score = 0.0_f32;
            for channel in 0..head_dim {
                score += query_head[channel] * cache.key[base + channel];
            }
            maximum = maximum.max(score);
            scores.push(score);
        }
        let mut denominator = 0.0_f32;
        for score in &mut scores {
            *score = (*score - maximum).exp();
            denominator += *score;
        }
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err("Gemma4 attention softmax denominator is invalid".into());
        }
        let output_head = &mut output[q_head * head_dim..(q_head + 1) * head_dim];
        for (relative_key, probability) in scores.into_iter().enumerate() {
            let key_index = first_key + relative_key;
            let base = key_index
                .checked_mul(cache.width)
                .and_then(|offset| offset.checked_add(kv_head * head_dim))
                .ok_or_else(|| "Gemma4 attention value offset overflows".to_string())?;
            let probability = probability / denominator;
            for channel in 0..head_dim {
                output_head[channel] += probability * cache.value[base + channel];
            }
        }
    }
    finite_slice(&output, "Gemma4 attention output")?;
    Ok(output)
}

fn top1_from_logits(logits: &[f32]) -> Result<Gemma4TextTop1, String> {
    let Some((&first, rest)) = logits.split_first() else {
        return Err("Gemma4 top-1 needs nonempty logits".into());
    };
    if !first.is_finite() {
        return Err("Gemma4 top-1 first logit is non-finite".into());
    }
    let mut token_id = 0_usize;
    let mut logit = first;
    for (index, value) in rest.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(format!("Gemma4 top-1 logit at {} is non-finite", index + 1));
        }
        // PyTorch argmax returns the first maximal index; retain that tie rule.
        if value > logit {
            token_id = index + 1;
            logit = value;
        }
    }
    Ok(Gemma4TextTop1 {
        token_id: u32::try_from(token_id)
            .map_err(|_| "Gemma4 vocabulary index does not fit u32".to_string())?,
        logit,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_config::{
        AttentionConfig, DecoderShapeConfig, DenseMlpConfig, GemmaRopeConfig,
        RmsNormWeightConvention,
    };

    fn config() -> Gemma4TextConfig {
        Gemma4TextConfig {
            decoder: DecoderShapeConfig {
                model_type: "gemma4_text".into(),
                hidden_size: 8,
                num_hidden_layers: 2,
                num_attention_heads: 2,
                num_key_value_heads: 1,
                head_dim: 4,
                rms_norm_eps: 1e-6,
                vocab_size: 16,
                tie_word_embeddings: true,
            },
            attention: AttentionConfig {
                bias: false,
                dropout: 0.0,
            },
            dense_mlp: DenseMlpConfig {
                activation: "gelu_pytorch_tanh".into(),
                intermediate_size: 16,
            },
            layer_types: vec![
                DecoderLayerKind::SlidingAttention,
                DecoderLayerKind::FullAttention,
            ],
            local_head_dim: 4,
            global_head_dim: 4,
            num_global_key_value_heads: None,
            sliding_window: 2,
            sliding_rope: GemmaRopeConfig {
                rope_type: "default".into(),
                rope_theta: 10_000.0,
                partial_rotary_factor: None,
            },
            full_rope: GemmaRopeConfig {
                rope_type: "proportional".into(),
                rope_theta: 1_000_000.0,
                partial_rotary_factor: Some(0.25),
            },
            attention_k_eq_v: false,
            num_kv_shared_layers: 1,
            use_double_wide_mlp: true,
            hidden_size_per_layer_input: 2,
            vocab_size_per_layer_input: 16,
            final_logit_softcapping: 30.0,
            max_position_embeddings: 64,
            use_bidirectional_attention: None,
            enable_moe_block: false,
            norm_weight_convention: RmsNormWeightConvention::DirectWeight,
        }
    }

    #[test]
    fn bf16_conversion_preserves_expected_values() {
        let values = bf16_bytes_to_f32(&[0x80, 0x3f, 0x20, 0xc0], "test").unwrap();
        assert_eq!(values, vec![1.0, -2.5]);
    }

    #[test]
    fn direct_rms_norm_does_not_add_one_to_weights() {
        let output = rms_norm(&[3.0, 4.0], Some(&[2.0, 3.0]), 0.0 + 1e-6).unwrap();
        let inverse = (12.5_f32 + 1e-6).powf(-0.5);
        assert!((output[0] - 3.0 * inverse * 2.0).abs() < 1e-6);
        assert!((output[1] - 4.0 * inverse * 3.0).abs() < 1e-6);
    }

    #[test]
    fn proportional_rope_leaves_unrotated_channels_unchanged() {
        let config = config();
        let mut values = vec![1.0, 2.0, 3.0, 4.0];
        apply_gemma4_rope_in_place(
            &mut values,
            1,
            4,
            DecoderLayerKind::FullAttention,
            7,
            &config,
        )
        .unwrap();
        // 25% of a four-wide head yields zero active pairs under HF's
        // `int(partial * head_dim // 2)` rule.
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn sliding_attention_keeps_only_the_recent_window() {
        let mut cache = KvSequence {
            width: 1,
            key: vec![1.0, 1.0, 1.0],
            value: vec![1.0, 2.0, 5.0],
        };
        cache.append(&[1.0], &[9.0]).unwrap();
        let output = causal_attention(&[1.0], &cache, 1, 1, 1, Some(2)).unwrap();
        assert!(output[0] > 6.0);
        assert!(output[0] < 9.1);
    }
}
