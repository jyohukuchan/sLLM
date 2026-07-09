// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

use crate::sq::fp8_e4m3fn_to_f32;
use crate::sq_canonical::{Sq8CanonicalArtifact, Sq8CanonicalTensorPair};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Read;
use std::path::Path;

pub const SQ8_REFERENCE_WEIGHT_CHUNK_BYTES: usize = 1024 * 1024;
pub const SQ8_REFERENCE_SCALE_CHUNK_BYTES: usize = 1024 * 1024;
pub const SQ8_CORRECTNESS_MAX_ABS_THRESHOLD: f64 = 2.0e-5;
pub const SQ8_CORRECTNESS_RELATIVE_L2_THRESHOLD: f64 = 1.0e-5;
pub const SQ8_CORRECTNESS_COSINE_THRESHOLD: f64 = 0.999_999;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Sq8ReferenceImplementationProfile {
    ReferenceW8a16Block2d,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Sq8CorrectnessExecutionPath {
    CpuStreamingReference,
    RuntimeCpuReference,
    RuntimeHipKernel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Sq8CorrectnessFallbackState {
    NotApplicable,
    NotUsed,
    Used,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Sq8CorrectnessThresholds {
    pub max_abs: f64,
    pub relative_l2: f64,
    pub cosine_similarity: f64,
}

pub const SQ8_CORRECTNESS_THRESHOLDS: Sq8CorrectnessThresholds = Sq8CorrectnessThresholds {
    max_abs: SQ8_CORRECTNESS_MAX_ABS_THRESHOLD,
    relative_l2: SQ8_CORRECTNESS_RELATIVE_L2_THRESHOLD,
    cosine_similarity: SQ8_CORRECTNESS_COSINE_THRESHOLD,
};

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Sq8CorrectnessMetrics {
    pub element_count: usize,
    pub nonfinite_count: usize,
    pub mse: f64,
    pub max_abs: f64,
    pub relative_l2: f64,
    pub cosine_similarity: f64,
}

impl Sq8CorrectnessMetrics {
    pub fn passes_fixed_thresholds(&self) -> bool {
        self.nonfinite_count == 0
            && self.mse.is_finite()
            && self.max_abs.is_finite()
            && self.relative_l2.is_finite()
            && self.cosine_similarity.is_finite()
            && self.max_abs <= SQ8_CORRECTNESS_THRESHOLDS.max_abs
            && self.relative_l2 <= SQ8_CORRECTNESS_THRESHOLDS.relative_l2
            && self.cosine_similarity >= SQ8_CORRECTNESS_THRESHOLDS.cosine_similarity
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Sq8CorrectnessReport {
    pub artifact_content_sha256: String,
    pub tensor: String,
    pub input_f32_le_sha256: String,
    pub implementation_profile: Sq8ReferenceImplementationProfile,
    pub execution_path: Sq8CorrectnessExecutionPath,
    pub fallback_state: Sq8CorrectnessFallbackState,
    pub metrics: Sq8CorrectnessMetrics,
    pub thresholds: Sq8CorrectnessThresholds,
    pub passed: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sq8ReferenceProjection {
    pub tensor: String,
    pub input_f32_le_sha256: String,
    pub output: Vec<f32>,
}

pub fn sq8_reference_activation(elements: usize) -> Vec<f32> {
    (0..elements)
        .map(|index| {
            let reduced_index = index % 257;
            let residue = (73 * reduced_index + 19) % 257;
            (residue as i32 - 128) as f32 / 256.0_f32
        })
        .collect()
}

pub fn sq8_f32_le_sha256(values: &[f32]) -> Result<String, String> {
    let mut digest = Sha256::new();
    for (index, value) in values.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(format!(
                "SQ8 F32 input contains non-finite value {value} at index {index}"
            ));
        }
        digest.update(value.to_le_bytes());
    }
    Ok(format!("{:x}", digest.finalize()))
}

pub fn run_sq8_reference_projection(
    artifact: &Sq8CanonicalArtifact,
    tensor_name: &str,
    input: &[f32],
) -> Result<Sq8ReferenceProjection, String> {
    run_sq8_reference_projection_with_chunk_bytes(
        artifact,
        tensor_name,
        input,
        SQ8_REFERENCE_WEIGHT_CHUNK_BYTES,
    )
}

fn run_sq8_reference_projection_with_chunk_bytes(
    artifact: &Sq8CanonicalArtifact,
    tensor_name: &str,
    input: &[f32],
    weight_chunk_bytes: usize,
) -> Result<Sq8ReferenceProjection, String> {
    if weight_chunk_bytes == 0 {
        return Err("SQ8 reference weight chunk size must be greater than zero".to_string());
    }
    let pair = artifact.tensor_pair(tensor_name)?;
    let rows = usize::try_from(pair.shape[0])
        .map_err(|_| format!("SQ8 reference tensor {} rows do not fit usize", pair.name))?;
    let cols = usize::try_from(pair.shape[1])
        .map_err(|_| format!("SQ8 reference tensor {} cols do not fit usize", pair.name))?;
    let elements = rows
        .checked_mul(cols)
        .ok_or_else(|| format!("SQ8 reference tensor {} shape overflows usize", pair.name))?;
    if input.len() != cols {
        return Err(format!(
            "SQ8 reference tensor {} input length mismatch: expected={cols} actual={}",
            pair.name,
            input.len()
        ));
    }
    let input_f32_le_sha256 = sq8_f32_le_sha256(input)?;
    let block_rows = usize::try_from(pair.scale.block_shape[0]).map_err(|_| {
        format!(
            "SQ8 reference tensor {} block rows do not fit usize",
            pair.name
        )
    })?;
    let block_cols = usize::try_from(pair.scale.block_shape[1]).map_err(|_| {
        format!(
            "SQ8 reference tensor {} block cols do not fit usize",
            pair.name
        )
    })?;
    let scale_rows = usize::try_from(pair.scale.shape[0]).map_err(|_| {
        format!(
            "SQ8 reference tensor {} scale rows do not fit usize",
            pair.name
        )
    })?;
    let scale_cols = usize::try_from(pair.scale.shape[1]).map_err(|_| {
        format!(
            "SQ8 reference tensor {} scale cols do not fit usize",
            pair.name
        )
    })?;
    let expected_scale_elements = scale_rows
        .checked_mul(scale_cols)
        .ok_or_else(|| format!("SQ8 reference tensor {} scale shape overflows", pair.name))?;
    let scales = artifact.read_tensor_scales_f32(&pair.name, SQ8_REFERENCE_SCALE_CHUNK_BYTES)?;
    if scales.len() != expected_scale_elements {
        return Err(format!(
            "SQ8 reference tensor {} scale element mismatch: expected={expected_scale_elements} actual={}",
            pair.name,
            scales.len()
        ));
    }
    let paths = artifact.tensor_payload_paths(&pair.name)?;
    let output = stream_reference_weight_matvec(
        &paths.weight,
        pair,
        input,
        &scales,
        rows,
        cols,
        elements,
        block_rows,
        block_cols,
        scale_cols,
        weight_chunk_bytes,
    )?;
    Ok(Sq8ReferenceProjection {
        tensor: pair.name.clone(),
        input_f32_le_sha256,
        output,
    })
}

#[allow(clippy::too_many_arguments)]
fn stream_reference_weight_matvec(
    weight_path: &Path,
    pair: &Sq8CanonicalTensorPair,
    input: &[f32],
    scales: &[f32],
    rows: usize,
    cols: usize,
    elements: usize,
    block_rows: usize,
    block_cols: usize,
    scale_cols: usize,
    weight_chunk_bytes: usize,
) -> Result<Vec<f32>, String> {
    let mut weight_file = File::open(weight_path).map_err(|err| {
        format!(
            "failed to open SQ8 reference weight {}: {err}",
            weight_path.display()
        )
    })?;
    let opened_bytes = weight_file
        .metadata()
        .map_err(|err| {
            format!(
                "failed to stat opened SQ8 reference weight {}: {err}",
                weight_path.display()
            )
        })?
        .len();
    if opened_bytes != pair.weight.bytes || opened_bytes != elements as u64 {
        return Err(format!(
            "SQ8 reference tensor {} weight length mismatch before read: manifest={} shape={} file={opened_bytes}",
            pair.name, pair.weight.bytes, elements
        ));
    }

    let chunk_bytes = weight_chunk_bytes.min(elements).max(1);
    let mut buffer = vec![0_u8; chunk_bytes];
    let mut accumulators = vec![0.0_f64; rows];
    let mut digest = Sha256::new();
    let mut flat_offset = 0_usize;
    while flat_offset < elements {
        let read_len = (elements - flat_offset).min(buffer.len());
        weight_file
            .read_exact(&mut buffer[..read_len])
            .map_err(|err| {
                format!(
                    "failed to read SQ8 reference tensor {} weight at byte {flat_offset}: {err}",
                    pair.name
                )
            })?;
        let chunk = &buffer[..read_len];
        digest.update(chunk);
        if let Some(local_index) = memchr::memchr2(0x7f, 0xff, chunk) {
            return Err(format!(
                "SQ8 reference tensor {} contains non-finite E4M3 byte at offset {}",
                pair.name,
                flat_offset + local_index
            ));
        }
        for (local_index, byte) in chunk.iter().copied().enumerate() {
            let flat_index = flat_offset + local_index;
            let row = flat_index / cols;
            let col = flat_index % cols;
            let scale_index = (row / block_rows)
                .checked_mul(scale_cols)
                .and_then(|value| value.checked_add(col / block_cols))
                .ok_or_else(|| {
                    format!(
                        "SQ8 reference tensor {} scale index overflows at [{row},{col}]",
                        pair.name
                    )
                })?;
            let scale = *scales.get(scale_index).ok_or_else(|| {
                format!(
                    "SQ8 reference tensor {} scale index {scale_index} is out of range at [{row},{col}]",
                    pair.name
                )
            })?;
            let weight = fp8_e4m3fn_to_f32(byte);
            if !weight.is_finite() || !scale.is_finite() || !input[col].is_finite() {
                return Err(format!(
                    "SQ8 reference tensor {} encountered non-finite operand at [{row},{col}]",
                    pair.name
                ));
            }
            accumulators[row] += f64::from(weight) * f64::from(scale) * f64::from(input[col]);
        }
        flat_offset += read_len;
    }

    let mut trailing = [0_u8; 1];
    let trailing_bytes = weight_file.read(&mut trailing).map_err(|err| {
        format!(
            "failed to verify EOF for SQ8 reference tensor {} after {} bytes: {err}",
            pair.name, pair.weight.bytes
        )
    })?;
    if trailing_bytes != 0 {
        return Err(format!(
            "SQ8 reference tensor {} weight has trailing data after {} bytes",
            pair.name, pair.weight.bytes
        ));
    }
    let final_bytes = weight_file
        .metadata()
        .map_err(|err| {
            format!(
                "failed to re-stat opened SQ8 reference weight {}: {err}",
                weight_path.display()
            )
        })?
        .len();
    if final_bytes != opened_bytes || final_bytes != pair.weight.bytes {
        return Err(format!(
            "SQ8 reference tensor {} weight size changed during read: manifest={} before={opened_bytes} after={final_bytes}",
            pair.name, pair.weight.bytes
        ));
    }
    let actual_sha256 = format!("{:x}", digest.finalize());
    if actual_sha256 != pair.weight.sha256 {
        return Err(format!(
            "SQ8 reference tensor {} weight checksum mismatch: manifest={} file={actual_sha256}",
            pair.name, pair.weight.sha256
        ));
    }

    accumulators
        .into_iter()
        .enumerate()
        .map(|(row, value)| {
            if !value.is_finite() {
                return Err(format!(
                    "SQ8 reference tensor {} F64 accumulator is non-finite at row {row}",
                    pair.name
                ));
            }
            let output = value as f32;
            if !output.is_finite() {
                return Err(format!(
                    "SQ8 reference tensor {} F32 output is non-finite at row {row}",
                    pair.name
                ));
            }
            Ok(output)
        })
        .collect()
}

pub fn compare_sq8_correctness(
    reference: &[f32],
    actual: &[f32],
) -> Result<Sq8CorrectnessMetrics, String> {
    if reference.is_empty() {
        return Err("SQ8 correctness comparison requires non-empty outputs".to_string());
    }
    if reference.len() != actual.len() {
        return Err(format!(
            "SQ8 correctness output length mismatch: reference={} actual={}",
            reference.len(),
            actual.len()
        ));
    }
    let nonfinite_count = reference
        .iter()
        .chain(actual)
        .filter(|value| !value.is_finite())
        .count();
    if nonfinite_count != 0 {
        return Ok(Sq8CorrectnessMetrics {
            element_count: reference.len(),
            nonfinite_count,
            mse: f64::MAX,
            max_abs: f64::MAX,
            relative_l2: f64::MAX,
            cosine_similarity: -1.0,
        });
    }

    let mut squared_error = 0.0_f64;
    let mut max_abs = 0.0_f64;
    let mut reference_squared = 0.0_f64;
    let mut actual_squared = 0.0_f64;
    let mut dot = 0.0_f64;
    for (reference_value, actual_value) in reference.iter().zip(actual) {
        let reference_value = f64::from(*reference_value);
        let actual_value = f64::from(*actual_value);
        let difference = actual_value - reference_value;
        squared_error += difference * difference;
        max_abs = max_abs.max(difference.abs());
        reference_squared += reference_value * reference_value;
        actual_squared += actual_value * actual_value;
        dot += reference_value * actual_value;
    }
    let mse = squared_error / reference.len() as f64;
    let relative_l2 = if reference_squared == 0.0 {
        if squared_error == 0.0 { 0.0 } else { f64::MAX }
    } else {
        (squared_error / reference_squared).sqrt()
    };
    let cosine_similarity = if reference_squared == 0.0 && actual_squared == 0.0 {
        1.0
    } else if reference_squared == 0.0 || actual_squared == 0.0 {
        0.0
    } else {
        (dot / (reference_squared.sqrt() * actual_squared.sqrt())).clamp(-1.0, 1.0)
    };
    Ok(Sq8CorrectnessMetrics {
        element_count: reference.len(),
        nonfinite_count,
        mse,
        max_abs,
        relative_l2,
        cosine_similarity,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn build_sq8_correctness_report(
    artifact: &Sq8CanonicalArtifact,
    tensor_name: &str,
    input: &[f32],
    reference: &[f32],
    actual: &[f32],
    execution_path: Sq8CorrectnessExecutionPath,
    fallback_state: Sq8CorrectnessFallbackState,
) -> Result<Sq8CorrectnessReport, String> {
    let pair = artifact.tensor_pair(tensor_name)?;
    let expected_input = usize::try_from(pair.shape[1])
        .map_err(|_| format!("SQ8 correctness tensor {} cols do not fit usize", pair.name))?;
    let expected_output = usize::try_from(pair.shape[0])
        .map_err(|_| format!("SQ8 correctness tensor {} rows do not fit usize", pair.name))?;
    if input.len() != expected_input {
        return Err(format!(
            "SQ8 correctness tensor {} input length mismatch: expected={expected_input} actual={}",
            pair.name,
            input.len()
        ));
    }
    if reference.len() != expected_output || actual.len() != expected_output {
        return Err(format!(
            "SQ8 correctness tensor {} output length mismatch: expected={expected_output} reference={} actual={}",
            pair.name,
            reference.len(),
            actual.len()
        ));
    }
    let metrics = compare_sq8_correctness(reference, actual)?;
    let passed =
        metrics.passes_fixed_thresholds() && fallback_state != Sq8CorrectnessFallbackState::Used;
    Ok(Sq8CorrectnessReport {
        artifact_content_sha256: artifact.manifest().integrity.content_sha256.clone(),
        tensor: pair.name.clone(),
        input_f32_le_sha256: sq8_f32_le_sha256(input)?,
        implementation_profile: Sq8ReferenceImplementationProfile::ReferenceW8a16Block2d,
        execution_path,
        fallback_state,
        metrics,
        thresholds: SQ8_CORRECTNESS_THRESHOLDS,
        passed,
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

    fn write_edge_artifact() -> TestArtifact {
        let root = std::env::temp_dir().join(format!(
            "ullm-engine-sq8-reference-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(root.join("weights")).unwrap();
        fs::create_dir_all(root.join("scales")).unwrap();
        let weight = vec![0x38_u8; 129 * 129];
        let scale = bf16_bytes(&[1.0, 2.0, 3.0, 4.0]);
        fs::write(root.join("weights/q.f8_e4m3"), &weight).unwrap();
        fs::write(root.join("scales/q.bf16"), &scale).unwrap();
        let mut manifest = json!({
            "schema_version": SQ8_CANONICAL_ARTIFACT_SCHEMA_VERSION,
            "artifact_kind": SQ8_CANONICAL_ARTIFACT_KIND,
            "format_id": FORMAT_SQ8_0,
            "source": {
                "model_name": "test-sq8-reference",
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
                "shape": [129, 129],
                "elements": 129 * 129,
                "weight": {
                    "dtype": SQ8_CANONICAL_WEIGHT_DTYPE,
                    "encoding": SQ8_CANONICAL_RAW_ENCODING,
                    "file": "weights/q.f8_e4m3",
                    "bytes": weight.len(),
                    "sha256": sha256_hex(&weight),
                    "source_file": "model-00001-of-00001.safetensors"
                },
                "scale": {
                    "name": "model.layers.0.self_attn.q_proj.weight_scale_inv",
                    "dtype": SQ8_CANONICAL_SCALE_DTYPE,
                    "encoding": SQ8_CANONICAL_RAW_ENCODING,
                    "file": "scales/q.bf16",
                    "shape": [2, 2],
                    "elements": 4,
                    "bytes": scale.len(),
                    "sha256": sha256_hex(&scale),
                    "source_file": "model-00001-of-00001.safetensors",
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
    fn fixed_activation_has_stable_f32_le_hash() {
        let activation = sq8_reference_activation(129);
        assert_eq!(activation[0], -109.0 / 256.0);
        assert_eq!(activation[1], -36.0 / 256.0);
        assert_eq!(activation[128], -17.0 / 256.0);
        assert_eq!(
            sq8_f32_le_sha256(&activation).unwrap(),
            "1fdf28379ac33408541790b897c54e5ce728f555884f9119ddecb388ae967307"
        );
        assert_eq!(
            sq8_f32_le_sha256(&sq8_reference_activation(5120)).unwrap(),
            "93f05449d07327c1237992938233030f1058dbe965504e343c8ae656dbe2e781"
        );
    }

    #[test]
    fn streaming_oracle_handles_129_by_129_edge_blocks() {
        let fixture = write_edge_artifact();
        let artifact = read_sq8_canonical_artifact(&fixture.root).unwrap();
        let input = vec![1.0_f32; 129];
        let projection =
            run_sq8_reference_projection_with_chunk_bytes(&artifact, TEST_TENSOR, &input, 17)
                .unwrap();
        assert_eq!(projection.output.len(), 129);
        assert!(projection.output[..128].iter().all(|value| *value == 130.0));
        assert_eq!(projection.output[128], 388.0);
    }

    #[test]
    fn streaming_oracle_rejects_payload_change_after_artifact_read() {
        let fixture = write_edge_artifact();
        let artifact = read_sq8_canonical_artifact(&fixture.root).unwrap();
        let path = fixture.root.join("weights/q.f8_e4m3");
        let mut weight = fs::read(&path).unwrap();
        weight[0] = 0x40;
        fs::write(path, weight).unwrap();
        let err =
            run_sq8_reference_projection(&artifact, TEST_TENSOR, &vec![1.0; 129]).unwrap_err();
        assert!(err.contains("weight checksum mismatch"), "{err}");
    }

    #[test]
    fn fixed_threshold_gate_is_fail_closed() {
        let passing = Sq8CorrectnessMetrics {
            element_count: 1,
            nonfinite_count: 0,
            mse: 0.0,
            max_abs: SQ8_CORRECTNESS_MAX_ABS_THRESHOLD,
            relative_l2: SQ8_CORRECTNESS_RELATIVE_L2_THRESHOLD,
            cosine_similarity: SQ8_CORRECTNESS_COSINE_THRESHOLD,
        };
        assert!(passing.passes_fixed_thresholds());
        for failing in [
            Sq8CorrectnessMetrics {
                max_abs: SQ8_CORRECTNESS_MAX_ABS_THRESHOLD + f64::EPSILON,
                ..passing.clone()
            },
            Sq8CorrectnessMetrics {
                relative_l2: SQ8_CORRECTNESS_RELATIVE_L2_THRESHOLD + f64::EPSILON,
                ..passing.clone()
            },
            Sq8CorrectnessMetrics {
                cosine_similarity: SQ8_CORRECTNESS_COSINE_THRESHOLD - f64::EPSILON,
                ..passing.clone()
            },
            Sq8CorrectnessMetrics {
                nonfinite_count: 1,
                ..passing
            },
        ] {
            assert!(!failing.passes_fixed_thresholds());
        }
    }

    #[test]
    fn correctness_metrics_and_report_are_typed_and_serializable() {
        let fixture = write_edge_artifact();
        let artifact = read_sq8_canonical_artifact(&fixture.root).unwrap();
        let input = vec![1.0_f32; 129];
        let reference = vec![1.0_f32; 129];
        let report = build_sq8_correctness_report(
            &artifact,
            TEST_TENSOR,
            &input,
            &reference,
            &reference,
            Sq8CorrectnessExecutionPath::RuntimeHipKernel,
            Sq8CorrectnessFallbackState::NotUsed,
        )
        .unwrap();
        assert!(report.passed);
        let value = serde_json::to_value(&report).unwrap();
        assert_eq!(value["implementation_profile"], "reference_w8a16_block2d");
        assert_eq!(value["execution_path"], "runtime_hip_kernel");
        assert_eq!(value["fallback_state"], "not_used");
        assert_eq!(value["thresholds"]["max_abs"], 2.0e-5);

        let fallback_report = build_sq8_correctness_report(
            &artifact,
            TEST_TENSOR,
            &input,
            &reference,
            &reference,
            Sq8CorrectnessExecutionPath::RuntimeCpuReference,
            Sq8CorrectnessFallbackState::Used,
        )
        .unwrap();
        assert!(fallback_report.metrics.passes_fixed_thresholds());
        assert!(!fallback_report.passed);
        let fallback_value = serde_json::to_value(&fallback_report).unwrap();
        assert_eq!(fallback_value["execution_path"], "runtime_cpu_reference");
        assert_eq!(fallback_value["fallback_state"], "used");
    }

    #[test]
    fn metrics_count_nonfinite_and_gate_rejects_them() {
        let metrics = compare_sq8_correctness(&[1.0, f32::NAN], &[1.0, f32::INFINITY]).unwrap();
        assert_eq!(metrics.nonfinite_count, 2);
        assert!(!metrics.passes_fixed_thresholds());
        assert!(serde_json::to_value(&metrics).is_ok());
    }
}
