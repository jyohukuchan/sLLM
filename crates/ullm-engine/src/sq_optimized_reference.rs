// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

use crate::sq::fp8_e4m3fn_to_f32;
use crate::sq_canonical::{Sq8CanonicalArtifact, Sq8CanonicalTensorPair};
use crate::sq_reference::sq8_f32_le_sha256;
use memchr::memchr2;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::OnceLock;
use std::thread;

pub const SQ8_DYNAMIC_ACTIVATION_BLOCK_COLS: usize = 128;
pub const SQ8_OCP_E4M3_MAX_FINITE: f32 = 448.0;
pub const SQ8_MIN_POSITIVE_F32_SCALE: f32 = f32::from_bits(1);
pub const SQ8_OPTIMIZED_REFERENCE_WEIGHT_CHUNK_BYTES: usize = 1024 * 1024;
pub const SQ8_OPTIMIZED_REFERENCE_SCALE_CHUNK_BYTES: usize = 1024 * 1024;
pub const SQ8_OPTIMIZED_REFERENCE_MIN_PARALLEL_ROWS: usize = 8;
pub const SQ8_OPTIMIZED_REFERENCE_MAX_THREADS: usize = 16;
pub const SQ8_OPTIMIZED_RELATIVE_L2_THRESHOLD: f64 = 5.0e-3;
pub const SQ8_OPTIMIZED_COSINE_THRESHOLD: f64 = 0.9999;

static SQ8_OCP_E4M3_VALUES: OnceLock<[f32; 256]> = OnceLock::new();

#[derive(Debug, Clone, PartialEq)]
pub struct Sq8DynamicActivation {
    rows: usize,
    cols: usize,
    blocks_per_row: usize,
    values: Vec<u8>,
    scales: Vec<f32>,
    input_f32_le_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Sq8DynamicActivationHashes {
    pub input_f32_le_sha256: String,
    pub encoded_bytes_sha256: String,
    pub scales_f32_le_sha256: String,
}

impl Sq8DynamicActivation {
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn block_cols(&self) -> usize {
        SQ8_DYNAMIC_ACTIVATION_BLOCK_COLS
    }

    pub fn blocks_per_row(&self) -> usize {
        self.blocks_per_row
    }

    pub fn scale_shape(&self) -> [usize; 2] {
        [self.rows, self.blocks_per_row]
    }

    pub fn values(&self) -> &[u8] {
        &self.values
    }

    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    pub fn input_f32_le_sha256(&self) -> &str {
        &self.input_f32_le_sha256
    }

    pub fn hashes(&self) -> Result<Sq8DynamicActivationHashes, String> {
        Ok(Sq8DynamicActivationHashes {
            input_f32_le_sha256: self.input_f32_le_sha256.clone(),
            encoded_bytes_sha256: sq8_bytes_sha256(&self.values),
            scales_f32_le_sha256: sq8_f32_le_sha256(&self.scales)?,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sq8OptimizedReferenceProjection {
    pub tensor: String,
    pub output_rows: usize,
    pub output_cols: usize,
    pub cpu_worker_threads: usize,
    pub output: Vec<f32>,
}

impl Sq8OptimizedReferenceProjection {
    pub fn output_f32_le_sha256(&self) -> Result<String, String> {
        sq8_f32_le_sha256(&self.output)
    }
}

pub fn sq8_bytes_sha256(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}

pub fn decode_sq8_ocp_e4m3(byte: u8) -> Result<f32, String> {
    let value = sq8_ocp_e4m3_values()[usize::from(byte)];
    if !value.is_finite() {
        return Err(format!(
            "SQ8 OCP E4M3 byte 0x{byte:02x} encodes a non-finite value"
        ));
    }
    Ok(value)
}

pub fn encode_sq8_ocp_e4m3_rne(value: f32) -> Result<u8, String> {
    if !value.is_finite() {
        return Err(format!(
            "SQ8 OCP E4M3 RNE encode requires a finite value, got {value}"
        ));
    }

    let sign = if value.is_sign_negative() { 0x80 } else { 0 };
    let magnitude = value.abs();
    if magnitude >= SQ8_OCP_E4M3_MAX_FINITE {
        return Ok(sign | 0x7e);
    }

    let values = &sq8_ocp_e4m3_values()[..127];
    let upper_index = values.partition_point(|candidate| *candidate < magnitude);
    if values[upper_index] == magnitude {
        return Ok(sign | upper_index as u8);
    }
    debug_assert!(upper_index > 0);
    let lower_index = upper_index - 1;
    let lower_distance = f64::from(magnitude) - f64::from(values[lower_index]);
    let upper_distance = f64::from(values[upper_index]) - f64::from(magnitude);
    let magnitude_byte = if lower_distance < upper_distance {
        lower_index as u8
    } else if upper_distance < lower_distance {
        upper_index as u8
    } else if lower_index & 1 == 0 {
        // Adjacent finite encodings are ordered, so encoding parity is the significand LSB.
        lower_index as u8
    } else {
        upper_index as u8
    };
    Ok(sign | magnitude_byte)
}

fn sq8_ocp_e4m3_values() -> &'static [f32; 256] {
    SQ8_OCP_E4M3_VALUES.get_or_init(|| std::array::from_fn(|index| fp8_e4m3fn_to_f32(index as u8)))
}

pub fn quantize_sq8_dynamic_activation(
    input: &[f32],
    rows: usize,
    cols: usize,
) -> Result<Sq8DynamicActivation, String> {
    if rows == 0 || cols == 0 {
        return Err(format!(
            "SQ8 dynamic activation shape must be non-zero, got [{rows},{cols}]"
        ));
    }
    let elements = rows
        .checked_mul(cols)
        .ok_or_else(|| format!("SQ8 dynamic activation shape [{rows},{cols}] overflows usize"))?;
    if input.len() != elements {
        return Err(format!(
            "SQ8 dynamic activation input length mismatch for [{rows},{cols}]: expected={elements} actual={}",
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
            "SQ8 dynamic activation contains non-finite value {value} at index {index}"
        ));
    }
    let input_f32_le_sha256 = sq8_f32_le_sha256(input)?;

    let blocks_per_row = cols.div_ceil(SQ8_DYNAMIC_ACTIVATION_BLOCK_COLS);
    let scale_elements = rows.checked_mul(blocks_per_row).ok_or_else(|| {
        format!("SQ8 dynamic activation scale shape [{rows},{blocks_per_row}] overflows usize")
    })?;
    let mut values = vec![0_u8; elements];
    let mut scales = Vec::with_capacity(scale_elements);

    for row in 0..rows {
        let row_start = row * cols;
        for block_col in 0..blocks_per_row {
            let start_col = block_col * SQ8_DYNAMIC_ACTIVATION_BLOCK_COLS;
            let end_col = (start_col + SQ8_DYNAMIC_ACTIVATION_BLOCK_COLS).min(cols);
            let block = &input[row_start + start_col..row_start + end_col];
            let max_abs = block.iter().copied().map(f32::abs).fold(0.0_f32, f32::max);
            let scale = if max_abs == 0.0 {
                1.0
            } else {
                // Keep the dequant multiplier representable when absmax/448 underflows F32.
                (max_abs / SQ8_OCP_E4M3_MAX_FINITE).max(SQ8_MIN_POSITIVE_F32_SCALE)
            };
            if !scale.is_finite() || scale <= 0.0 {
                return Err(format!(
                    "SQ8 dynamic activation produced invalid scale {scale} at [{row},{block_col}]"
                ));
            }
            scales.push(scale);
            for col in start_col..end_col {
                values[row_start + col] = encode_sq8_ocp_e4m3_rne(input[row_start + col] / scale)?;
            }
        }
    }

    debug_assert_eq!(scales.len(), scale_elements);
    Ok(Sq8DynamicActivation {
        rows,
        cols,
        blocks_per_row,
        values,
        scales,
        input_f32_le_sha256,
    })
}

pub fn run_sq8_optimized_reference_projection(
    artifact: &Sq8CanonicalArtifact,
    tensor_name: &str,
    activation: &Sq8DynamicActivation,
) -> Result<Sq8OptimizedReferenceProjection, String> {
    run_sq8_optimized_reference_projection_with_chunk_bytes(
        artifact,
        tensor_name,
        activation,
        SQ8_OPTIMIZED_REFERENCE_WEIGHT_CHUNK_BYTES,
    )
}

fn run_sq8_optimized_reference_projection_with_chunk_bytes(
    artifact: &Sq8CanonicalArtifact,
    tensor_name: &str,
    activation: &Sq8DynamicActivation,
    weight_chunk_bytes: usize,
) -> Result<Sq8OptimizedReferenceProjection, String> {
    if weight_chunk_bytes == 0 {
        return Err(
            "SQ8 optimized reference weight chunk size must be greater than zero".to_string(),
        );
    }
    let pair = artifact.tensor_pair(tensor_name)?;
    let weight_rows = usize::try_from(pair.shape[0]).map_err(|_| {
        format!(
            "SQ8 optimized reference tensor {} rows do not fit usize",
            pair.name
        )
    })?;
    let weight_cols = usize::try_from(pair.shape[1]).map_err(|_| {
        format!(
            "SQ8 optimized reference tensor {} cols do not fit usize",
            pair.name
        )
    })?;
    if activation.cols != weight_cols {
        return Err(format!(
            "SQ8 optimized reference tensor {} activation K mismatch: expected={weight_cols} actual={}",
            pair.name, activation.cols
        ));
    }
    if pair.scale.block_shape
        != [
            SQ8_DYNAMIC_ACTIVATION_BLOCK_COLS as u64,
            SQ8_DYNAMIC_ACTIVATION_BLOCK_COLS as u64,
        ]
    {
        return Err(format!(
            "SQ8 optimized reference tensor {} requires canonical 128x128 weight blocks, got {:?}",
            pair.name, pair.scale.block_shape
        ));
    }

    let weight_elements = weight_rows.checked_mul(weight_cols).ok_or_else(|| {
        format!(
            "SQ8 optimized reference tensor {} shape overflows usize",
            pair.name
        )
    })?;
    let output_elements = activation.rows.checked_mul(weight_rows).ok_or_else(|| {
        format!(
            "SQ8 optimized reference tensor {} output shape [{},{}] overflows usize",
            pair.name, activation.rows, weight_rows
        )
    })?;
    let weight_scale_rows = usize::try_from(pair.scale.shape[0]).map_err(|_| {
        format!(
            "SQ8 optimized reference tensor {} scale rows do not fit usize",
            pair.name
        )
    })?;
    let weight_scale_cols = usize::try_from(pair.scale.shape[1]).map_err(|_| {
        format!(
            "SQ8 optimized reference tensor {} scale cols do not fit usize",
            pair.name
        )
    })?;
    let expected_weight_scale_rows = weight_rows.div_ceil(SQ8_DYNAMIC_ACTIVATION_BLOCK_COLS);
    let expected_weight_scale_cols = weight_cols.div_ceil(SQ8_DYNAMIC_ACTIVATION_BLOCK_COLS);
    if weight_scale_rows != expected_weight_scale_rows
        || weight_scale_cols != expected_weight_scale_cols
    {
        return Err(format!(
            "SQ8 optimized reference tensor {} scale shape mismatch: expected=[{expected_weight_scale_rows},{expected_weight_scale_cols}] actual=[{weight_scale_rows},{weight_scale_cols}]",
            pair.name
        ));
    }
    if activation.blocks_per_row != expected_weight_scale_cols {
        return Err(format!(
            "SQ8 optimized reference tensor {} activation scale columns mismatch: expected={expected_weight_scale_cols} actual={}",
            pair.name, activation.blocks_per_row
        ));
    }

    let weight_scales =
        artifact.read_tensor_scales_f32(&pair.name, SQ8_OPTIMIZED_REFERENCE_SCALE_CHUNK_BYTES)?;
    let expected_weight_scales = weight_scale_rows
        .checked_mul(weight_scale_cols)
        .ok_or_else(|| {
            format!(
                "SQ8 optimized reference tensor {} scale shape overflows usize",
                pair.name
            )
        })?;
    if weight_scales.len() != expected_weight_scales {
        return Err(format!(
            "SQ8 optimized reference tensor {} scale element mismatch: expected={expected_weight_scales} actual={}",
            pair.name,
            weight_scales.len()
        ));
    }

    let paths = artifact.tensor_payload_paths(&pair.name)?;
    let (output, cpu_worker_threads) = stream_optimized_reference_projection(
        &paths.weight,
        pair,
        activation,
        &weight_scales,
        weight_rows,
        weight_cols,
        weight_elements,
        weight_scale_cols,
        output_elements,
        weight_chunk_bytes,
    )?;
    Ok(Sq8OptimizedReferenceProjection {
        tensor: pair.name.clone(),
        output_rows: activation.rows,
        output_cols: weight_rows,
        cpu_worker_threads,
        output,
    })
}

#[allow(clippy::too_many_arguments)]
fn stream_optimized_reference_projection(
    weight_path: &Path,
    pair: &Sq8CanonicalTensorPair,
    activation: &Sq8DynamicActivation,
    weight_scales: &[f32],
    weight_rows: usize,
    weight_cols: usize,
    weight_elements: usize,
    weight_scale_cols: usize,
    output_elements: usize,
    weight_chunk_bytes: usize,
) -> Result<(Vec<f32>, usize), String> {
    let activation_values = dequantize_dynamic_activation_f64(activation)?;
    let cpu_worker_threads = optimized_reference_worker_count(activation.rows);
    let mut weight_file = File::open(weight_path).map_err(|err| {
        format!(
            "failed to open SQ8 optimized reference weight {}: {err}",
            weight_path.display()
        )
    })?;
    let opened_bytes = weight_file
        .metadata()
        .map_err(|err| {
            format!(
                "failed to stat opened SQ8 optimized reference weight {}: {err}",
                weight_path.display()
            )
        })?
        .len();
    if opened_bytes != pair.weight.bytes || opened_bytes != weight_elements as u64 {
        return Err(format!(
            "SQ8 optimized reference tensor {} weight length mismatch before read: manifest={} shape={} file={opened_bytes}",
            pair.name, pair.weight.bytes, weight_elements
        ));
    }

    let chunk_bytes = weight_chunk_bytes.min(weight_elements).max(1);
    let mut buffer = vec![0_u8; chunk_bytes];
    let mut reconstructed_weight_buffer = if cpu_worker_threads == 1 {
        Vec::new()
    } else {
        Vec::with_capacity(chunk_bytes)
    };
    let mut accumulators = vec![0.0_f64; output_elements];
    let mut digest = Sha256::new();
    let mut flat_offset = 0_usize;
    while flat_offset < weight_elements {
        let read_len = (weight_elements - flat_offset).min(buffer.len());
        weight_file
            .read_exact(&mut buffer[..read_len])
            .map_err(|err| {
                format!(
                    "failed to read SQ8 optimized reference tensor {} weight at byte {flat_offset}: {err}",
                    pair.name
                )
            })?;
        let chunk = &buffer[..read_len];
        digest.update(chunk);
        if let Some(local_index) = memchr2(0x7f, 0xff, chunk) {
            return Err(format!(
                "SQ8 optimized reference tensor {} contains non-finite E4M3 byte at offset {}",
                pair.name,
                flat_offset + local_index
            ));
        }

        if cpu_worker_threads == 1 {
            accumulate_raw_weight_chunk(
                chunk,
                flat_offset,
                &activation_values,
                weight_scales,
                weight_rows,
                weight_cols,
                weight_scale_cols,
                &mut accumulators,
            );
        } else {
            reconstruct_weight_chunk_f64(
                chunk,
                flat_offset,
                weight_scales,
                weight_cols,
                weight_scale_cols,
                &mut reconstructed_weight_buffer,
            );
            accumulate_reconstructed_weight_chunk_parallel(
                &reconstructed_weight_buffer,
                flat_offset,
                &activation_values,
                activation.rows,
                weight_rows,
                weight_cols,
                cpu_worker_threads,
                &mut accumulators,
            )?;
        }
        flat_offset += read_len;
    }

    let mut trailing = [0_u8; 1];
    let trailing_bytes = weight_file.read(&mut trailing).map_err(|err| {
        format!(
            "failed to verify EOF for SQ8 optimized reference tensor {} after {} bytes: {err}",
            pair.name, pair.weight.bytes
        )
    })?;
    if trailing_bytes != 0 {
        return Err(format!(
            "SQ8 optimized reference tensor {} weight has trailing data after {} bytes",
            pair.name, pair.weight.bytes
        ));
    }
    let final_bytes = weight_file
        .metadata()
        .map_err(|err| {
            format!(
                "failed to re-stat opened SQ8 optimized reference weight {}: {err}",
                weight_path.display()
            )
        })?
        .len();
    if final_bytes != opened_bytes || final_bytes != pair.weight.bytes {
        return Err(format!(
            "SQ8 optimized reference tensor {} weight size changed during read: manifest={} before={opened_bytes} after={final_bytes}",
            pair.name, pair.weight.bytes
        ));
    }
    let actual_sha256 = format!("{:x}", digest.finalize());
    if actual_sha256 != pair.weight.sha256 {
        return Err(format!(
            "SQ8 optimized reference tensor {} weight checksum mismatch: manifest={} file={actual_sha256}",
            pair.name, pair.weight.sha256
        ));
    }

    let output = accumulators
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            if !value.is_finite() {
                return Err(format!(
                    "SQ8 optimized reference tensor {} F64 accumulator is non-finite at output index {index}",
                    pair.name
                ));
            }
            let output = value as f32;
            if !output.is_finite() {
                return Err(format!(
                    "SQ8 optimized reference tensor {} F32 output is non-finite at output index {index}",
                    pair.name
                ));
            }
            Ok(output)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((output, cpu_worker_threads))
}

fn optimized_reference_worker_count(activation_rows: usize) -> usize {
    if activation_rows < SQ8_OPTIMIZED_REFERENCE_MIN_PARALLEL_ROWS {
        return 1;
    }
    let available = thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1);
    let requested = available
        .min(activation_rows)
        .clamp(1, SQ8_OPTIMIZED_REFERENCE_MAX_THREADS);
    let rows_per_worker = activation_rows.div_ceil(requested);
    activation_rows.div_ceil(rows_per_worker)
}

fn dequantize_dynamic_activation_f64(
    activation: &Sq8DynamicActivation,
) -> Result<Vec<f64>, String> {
    let mut dequantized = Vec::with_capacity(activation.values.len());
    let decode_table = sq8_ocp_e4m3_values();
    for row in 0..activation.rows {
        for col in 0..activation.cols {
            let scale_index =
                row * activation.blocks_per_row + col / SQ8_DYNAMIC_ACTIVATION_BLOCK_COLS;
            let scale = activation.scales[scale_index];
            let value = decode_table[usize::from(activation.values[row * activation.cols + col])];
            let reconstructed = f64::from(value) * f64::from(scale);
            if !reconstructed.is_finite() {
                return Err(format!(
                    "SQ8 dynamic activation reconstruction is non-finite at [{row},{col}]"
                ));
            }
            dequantized.push(reconstructed);
        }
    }
    Ok(dequantized)
}

#[allow(clippy::too_many_arguments)]
fn accumulate_raw_weight_chunk(
    weight_chunk: &[u8],
    flat_offset: usize,
    activation_values: &[f64],
    weight_scales: &[f32],
    weight_rows: usize,
    weight_cols: usize,
    weight_scale_cols: usize,
    accumulators: &mut [f64],
) {
    let decode_table = sq8_ocp_e4m3_values();
    for (local_index, weight_byte) in weight_chunk.iter().copied().enumerate() {
        let flat_index = flat_offset + local_index;
        let weight_row = flat_index / weight_cols;
        let col = flat_index % weight_cols;
        let block_col = col / SQ8_DYNAMIC_ACTIVATION_BLOCK_COLS;
        let weight_scale_index =
            (weight_row / SQ8_DYNAMIC_ACTIVATION_BLOCK_COLS) * weight_scale_cols + block_col;
        let weight = f64::from(decode_table[usize::from(weight_byte)])
            * f64::from(weight_scales[weight_scale_index]);
        for activation_row in 0..activation_values.len() / weight_cols {
            accumulators[activation_row * weight_rows + weight_row] +=
                activation_values[activation_row * weight_cols + col] * weight;
        }
    }
}

fn reconstruct_weight_chunk_f64(
    weight_chunk: &[u8],
    flat_offset: usize,
    weight_scales: &[f32],
    weight_cols: usize,
    weight_scale_cols: usize,
    reconstructed: &mut Vec<f64>,
) {
    reconstructed.clear();
    let decode_table = sq8_ocp_e4m3_values();
    reconstructed.extend(weight_chunk.iter().copied().enumerate().map(
        |(local_index, weight_byte)| {
            let flat_index = flat_offset + local_index;
            let weight_row = flat_index / weight_cols;
            let col = flat_index % weight_cols;
            let scale_index = (weight_row / SQ8_DYNAMIC_ACTIVATION_BLOCK_COLS) * weight_scale_cols
                + col / SQ8_DYNAMIC_ACTIVATION_BLOCK_COLS;
            f64::from(decode_table[usize::from(weight_byte)])
                * f64::from(weight_scales[scale_index])
        },
    ));
}

#[allow(clippy::too_many_arguments)]
fn accumulate_reconstructed_weight_chunk_parallel(
    weight_chunk: &[f64],
    flat_offset: usize,
    activation_values: &[f64],
    activation_rows: usize,
    weight_rows: usize,
    weight_cols: usize,
    worker_threads: usize,
    accumulators: &mut [f64],
) -> Result<(), String> {
    let rows_per_worker = activation_rows.div_ceil(worker_threads);
    let accumulator_elements_per_worker = rows_per_worker * weight_rows;
    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_threads);
        for (worker_index, accumulator_rows) in accumulators
            .chunks_mut(accumulator_elements_per_worker)
            .enumerate()
        {
            let activation_row_start = worker_index * rows_per_worker;
            handles.push(scope.spawn(move || {
                let activation_row_count = accumulator_rows.len() / weight_rows;
                for (local_index, weight) in weight_chunk.iter().copied().enumerate() {
                    let flat_index = flat_offset + local_index;
                    let weight_row = flat_index / weight_cols;
                    let col = flat_index % weight_cols;
                    for local_activation_row in 0..activation_row_count {
                        let activation_row = activation_row_start + local_activation_row;
                        accumulator_rows[local_activation_row * weight_rows + weight_row] +=
                            activation_values[activation_row * weight_cols + col] * weight;
                    }
                }
            }));
        }
        let mut worker_panicked = false;
        for handle in handles {
            worker_panicked |= handle.join().is_err();
        }
        if worker_panicked {
            return Err("SQ8 optimized reference worker thread panicked".to_string());
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format_id::FORMAT_SQ8_0;
    use crate::sq_canonical::{
        SQ8_CANONICAL_ARTIFACT_KIND, SQ8_CANONICAL_ARTIFACT_SCHEMA_VERSION,
        SQ8_CANONICAL_IMPORT_MODE, SQ8_CANONICAL_RAW_ENCODING, SQ8_CANONICAL_SCALE_DTYPE,
        SQ8_CANONICAL_SCALE_LAYOUT, SQ8_CANONICAL_SCALE_ORDER, SQ8_CANONICAL_SCALE_SEMANTIC,
        SQ8_CANONICAL_WEIGHT_DTYPE, read_sq8_canonical_artifact,
    };
    use serde_json::{Value, json};
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    const TEST_TENSOR: &str = "model.layers.0.self_attn.q_proj.weight";

    struct TestArtifact {
        root: PathBuf,
    }

    impl Drop for TestArtifact {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        let mut digest = Sha256::new();
        digest.update(bytes);
        format!("{:x}", digest.finalize())
    }

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * 2);
        for value in values {
            bytes.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
        }
        bytes
    }

    fn set_content_sha256(manifest: &mut Value) {
        let mut content = manifest.clone();
        content.as_object_mut().unwrap().remove("integrity");
        manifest["integrity"]["content_sha256"] =
            json!(sha256_hex(&serde_json::to_vec(&content).unwrap()));
    }

    fn write_test_artifact() -> TestArtifact {
        let root = std::env::temp_dir().join(format!(
            "ullm-engine-sq8-optimized-reference-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(root.join("weights")).unwrap();
        fs::create_dir_all(root.join("scales")).unwrap();
        let weight = vec![0x38_u8; 3 * 129];
        let scale = bf16_bytes(&[2.0, 3.0]);
        fs::write(root.join("weights/q.f8_e4m3"), &weight).unwrap();
        fs::write(root.join("scales/q.bf16"), &scale).unwrap();
        let mut manifest = json!({
            "schema_version": SQ8_CANONICAL_ARTIFACT_SCHEMA_VERSION,
            "artifact_kind": SQ8_CANONICAL_ARTIFACT_KIND,
            "format_id": FORMAT_SQ8_0,
            "source": {
                "model_name": "test-sq8-optimized-reference",
                "config_file": "config.json",
                "config_sha256": "0".repeat(64),
                "index_file": "model.safetensors.index.json",
                "index_sha256": "1".repeat(64),
                "quantization": {
                    "quant_method": "fp8",
                    "format": "e4m3",
                    "activation_scheme": "dynamic",
                    "weight_block_shape": [128, 128]
                }
            },
            "import": {
                "mode": SQ8_CANONICAL_IMPORT_MODE,
                "encoding": SQ8_CANONICAL_RAW_ENCODING
            },
            "integrity": {"content_sha256": "2".repeat(64)},
            "coverage": {
                "scope": "selected_tensors",
                "source_tensor_count": 2,
                "source_fp8_weight_count": 1,
                "source_scale_count": 1,
                "paired_tensor_count": 1,
                "selected_pair_count": 1,
                "unpaired_tensor_count": 0,
                "passthrough_tensor_count": 0
            },
            "storage": {
                "weight_payload_bytes": weight.len(),
                "scale_payload_bytes": scale.len(),
                "total_payload_bytes": weight.len() + scale.len()
            },
            "quantized_tensors": [{
                "name": TEST_TENSOR,
                "family": "attn_q",
                "shape": [3, 129],
                "elements": 3 * 129,
                "weight": {
                    "dtype": SQ8_CANONICAL_WEIGHT_DTYPE,
                    "encoding": SQ8_CANONICAL_RAW_ENCODING,
                    "file": "weights/q.f8_e4m3",
                    "bytes": weight.len(),
                    "sha256": sha256_hex(&weight),
                    "source_file": "model.safetensors"
                },
                "scale": {
                    "name": "model.layers.0.self_attn.q_proj.weight_scale_inv",
                    "dtype": SQ8_CANONICAL_SCALE_DTYPE,
                    "encoding": SQ8_CANONICAL_RAW_ENCODING,
                    "file": "scales/q.bf16",
                    "shape": [1, 2],
                    "elements": 2,
                    "bytes": scale.len(),
                    "sha256": sha256_hex(&scale),
                    "source_file": "model.safetensors",
                    "layout": SQ8_CANONICAL_SCALE_LAYOUT,
                    "block_shape": [128, 128],
                    "order": SQ8_CANONICAL_SCALE_ORDER,
                    "semantic": SQ8_CANONICAL_SCALE_SEMANTIC
                }
            }],
            "passthrough_tensors": []
        });
        set_content_sha256(&mut manifest);
        fs::write(
            root.join("sq_manifest.json"),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
        TestArtifact { root }
    }

    #[test]
    fn ocp_e4m3_decode_covers_signed_zero_subnormal_and_max() {
        assert_eq!(
            decode_sq8_ocp_e4m3(0x00).unwrap().to_bits(),
            0.0_f32.to_bits()
        );
        assert_eq!(
            decode_sq8_ocp_e4m3(0x80).unwrap().to_bits(),
            (-0.0_f32).to_bits()
        );
        assert_eq!(decode_sq8_ocp_e4m3(0x01).unwrap(), 2.0_f32.powi(-9));
        assert_eq!(decode_sq8_ocp_e4m3(0x38).unwrap(), 1.0);
        assert_eq!(decode_sq8_ocp_e4m3(0x7e).unwrap(), 448.0);
        assert!(
            decode_sq8_ocp_e4m3(0x7f)
                .unwrap_err()
                .contains("non-finite")
        );
        assert!(
            decode_sq8_ocp_e4m3(0xff)
                .unwrap_err()
                .contains("non-finite")
        );
    }

    #[test]
    fn ocp_e4m3_encode_uses_round_to_nearest_even_and_saturates() {
        assert_eq!(encode_sq8_ocp_e4m3_rne(2.0_f32.powi(-10)).unwrap(), 0x00);
        assert_eq!(
            encode_sq8_ocp_e4m3_rne(3.0 * 2.0_f32.powi(-10)).unwrap(),
            0x02
        );
        assert_eq!(encode_sq8_ocp_e4m3_rne(1.0625).unwrap(), 0x38);
        assert_eq!(encode_sq8_ocp_e4m3_rne(1.1875).unwrap(), 0x3a);
        assert_eq!(encode_sq8_ocp_e4m3_rne(448.0).unwrap(), 0x7e);
        assert_eq!(encode_sq8_ocp_e4m3_rne(f32::MAX).unwrap(), 0x7e);
        assert_eq!(encode_sq8_ocp_e4m3_rne(-f32::MAX).unwrap(), 0xfe);
        assert_eq!(encode_sq8_ocp_e4m3_rne(-0.0).unwrap(), 0x80);
    }

    #[test]
    fn ocp_e4m3_encode_rejects_nonfinite_values() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(
                encode_sq8_ocp_e4m3_rne(value)
                    .unwrap_err()
                    .contains("finite")
            );
        }
    }

    #[test]
    fn ocp_e4m3_finite_encodings_roundtrip_exactly() {
        for byte in 0_u8..=u8::MAX {
            if matches!(byte, 0x7f | 0xff) {
                continue;
            }
            let decoded = decode_sq8_ocp_e4m3(byte).unwrap();
            assert_eq!(encode_sq8_ocp_e4m3_rne(decoded).unwrap(), byte);
        }
    }

    #[test]
    fn dynamic_activation_uses_independent_row_k128_scales() {
        let mut input = vec![0.0_f32; 2 * 129];
        input[0] = 448.0;
        input[1] = 224.0;
        input[128] = -2.0;
        input[129] = -0.0;
        let quantized = quantize_sq8_dynamic_activation(&input, 2, 129).unwrap();
        assert_eq!(quantized.rows(), 2);
        assert_eq!(quantized.cols(), 129);
        assert_eq!(quantized.block_cols(), 128);
        assert_eq!(quantized.blocks_per_row(), 2);
        assert_eq!(quantized.scale_shape(), [2, 2]);
        assert_eq!(quantized.scales(), &[1.0, 2.0 / 448.0, 1.0, 1.0]);
        assert_eq!(quantized.values()[0], 0x7e);
        assert_eq!(quantized.values()[1], 0x76);
        assert_eq!(quantized.values()[128], 0xfe);
        assert_eq!(quantized.values()[129], 0x80);
        assert!(quantized.values()[130..].iter().all(|byte| *byte == 0));
    }

    #[test]
    fn dynamic_activation_keeps_the_smallest_finite_f32_representable() {
        let minimum = f32::from_bits(1);
        let quantized = quantize_sq8_dynamic_activation(&[minimum, -minimum], 1, 2).unwrap();
        assert_eq!(quantized.scales(), &[SQ8_MIN_POSITIVE_F32_SCALE]);
        assert_eq!(quantized.values(), &[0x38, 0xb8]);
        for (byte, expected) in quantized.values().iter().zip([minimum, -minimum]) {
            assert_eq!(
                decode_sq8_ocp_e4m3(*byte).unwrap() * quantized.scales()[0],
                expected
            );
        }
    }

    #[test]
    fn dynamic_activation_handles_largest_finite_f32_without_nonfinite_scale() {
        let quantized = quantize_sq8_dynamic_activation(&[f32::MAX, -f32::MAX], 1, 2).unwrap();
        assert!(quantized.scales()[0].is_finite());
        assert!(quantized.scales()[0] > 0.0);
        assert_eq!(quantized.values(), &[0x7e, 0xfe]);
        assert!(
            (decode_sq8_ocp_e4m3(quantized.values()[0]).unwrap() as f64
                * f64::from(quantized.scales()[0]))
            .is_finite()
        );
    }

    #[test]
    fn existing_reference_activation_fixture_has_public_quantization_hashes() {
        let input = crate::sq_reference::sq8_reference_activation(2 * 129);
        let quantized = quantize_sq8_dynamic_activation(&input, 2, 129).unwrap();
        let hashes = quantized.hashes().unwrap();
        assert_eq!(
            hashes.input_f32_le_sha256,
            crate::sq_reference::sq8_f32_le_sha256(&input).unwrap()
        );
        assert_eq!(
            hashes.encoded_bytes_sha256,
            sq8_bytes_sha256(quantized.values())
        );
        assert_eq!(
            hashes.scales_f32_le_sha256,
            crate::sq_reference::sq8_f32_le_sha256(quantized.scales()).unwrap()
        );
        assert_eq!(quantized.input_f32_le_sha256(), hashes.input_f32_le_sha256);
        assert!(serde_json::to_value(&hashes).is_ok());
    }

    #[test]
    fn dynamic_activation_rejects_nonfinite_and_invalid_shapes() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let err = quantize_sq8_dynamic_activation(&[value], 1, 1).unwrap_err();
            assert!(err.contains("non-finite"), "{err}");
        }
        assert!(
            quantize_sq8_dynamic_activation(&[], 0, 1)
                .unwrap_err()
                .contains("non-zero")
        );
        assert!(
            quantize_sq8_dynamic_activation(&[], 1, 0)
                .unwrap_err()
                .contains("non-zero")
        );
        assert!(
            quantize_sq8_dynamic_activation(&[1.0], 1, 2)
                .unwrap_err()
                .contains("length mismatch")
        );
        assert!(
            quantize_sq8_dynamic_activation(&[], usize::MAX, 2)
                .unwrap_err()
                .contains("overflows")
        );
    }

    #[test]
    fn optimized_oracle_streams_weight_once_and_returns_m_by_n_f32() {
        let fixture = write_test_artifact();
        let artifact = read_sq8_canonical_artifact(&fixture.root).unwrap();
        let mut input = vec![1.0_f32; 2 * 129];
        input[128] = 2.0;
        input[129..].fill(0.0);
        let activation = quantize_sq8_dynamic_activation(&input, 2, 129).unwrap();
        let projection = run_sq8_optimized_reference_projection_with_chunk_bytes(
            &artifact,
            TEST_TENSOR,
            &activation,
            17,
        )
        .unwrap();
        assert_eq!(projection.tensor, TEST_TENSOR);
        assert_eq!(projection.output_rows, 2);
        assert_eq!(projection.output_cols, 3);
        assert_eq!(projection.output.len(), 6);
        assert_eq!(
            projection.output_f32_le_sha256().unwrap(),
            crate::sq_reference::sq8_f32_le_sha256(&projection.output).unwrap()
        );
        for value in &projection.output[..3] {
            assert!((*value - 262.0).abs() <= 1.0e-5, "{value}");
        }
        assert_eq!(&projection.output[3..], &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn optimized_oracle_parallel_rows_keep_independent_accumulators() {
        let fixture = write_test_artifact();
        let artifact = read_sq8_canonical_artifact(&fixture.root).unwrap();
        let rows = SQ8_OPTIMIZED_REFERENCE_MIN_PARALLEL_ROWS;
        let mut input = vec![0.0_f32; rows * 129];
        for row in 0..rows {
            let value = (row + 1) as f32;
            input[row * 129..row * 129 + 128].fill(value);
            input[row * 129 + 128] = 2.0 * value;
        }
        let activation = quantize_sq8_dynamic_activation(&input, rows, 129).unwrap();
        let projection =
            run_sq8_optimized_reference_projection(&artifact, TEST_TENSOR, &activation).unwrap();
        assert_eq!(
            projection.cpu_worker_threads,
            optimized_reference_worker_count(rows)
        );
        for row in 0..rows {
            let expected = 262.0 * (row + 1) as f32;
            for value in &projection.output[row * 3..row * 3 + 3] {
                assert!((*value - expected).abs() <= 1.0e-3, "{value} != {expected}");
            }
        }
    }

    #[test]
    fn optimized_oracle_rejects_shape_mismatch_and_weight_mutation() {
        let fixture = write_test_artifact();
        let artifact = read_sq8_canonical_artifact(&fixture.root).unwrap();
        let wrong_k = quantize_sq8_dynamic_activation(&[1.0; 128], 1, 128).unwrap();
        let err =
            run_sq8_optimized_reference_projection(&artifact, TEST_TENSOR, &wrong_k).unwrap_err();
        assert!(err.contains("activation K mismatch"), "{err}");

        let activation = quantize_sq8_dynamic_activation(&[1.0; 129], 1, 129).unwrap();
        let weight_path = fixture.root.join("weights/q.f8_e4m3");
        let mut weight = fs::read(&weight_path).unwrap();
        weight[0] = 0x40;
        fs::write(weight_path, weight).unwrap();
        let err = run_sq8_optimized_reference_projection(&artifact, TEST_TENSOR, &activation)
            .unwrap_err();
        assert!(err.contains("weight checksum mismatch"), "{err}");
    }
}
