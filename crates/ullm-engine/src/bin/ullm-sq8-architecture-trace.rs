// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Diagnostic-only Qwen3-14B SQ8_0 trace producer.
//!
//! This writes the independent `ullm.architecture_trace.v1` interchange
//! format consumed by `tools/architecture_hf_trace.py`.  It deliberately does
//! not use the FP32 corpus, numerical-gate, campaign, worker, or service
//! paths.  The trace is limited to one M=1 forward because the existing SQ8_0
//! serving instrumentation exposes one host row per layer.

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{Seek, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;
use ullm_engine::loader::read_named_passthrough_f32_rows;
use ullm_engine::model_config::{ModelConfig, load_model_config_from_package};
use ullm_engine::sq_canonical::read_sq8_canonical_artifact;
use ullm_engine::sq8_embedding_runtime::{
    QWEN3_14B_EMBED_TOKENS_TENSOR, QWEN3_14B_SQ8_EMBEDDING_REQUIRED_HIP_KERNEL_ENV,
};
use ullm_engine::sq8_layer_oracle::QWEN3_14B_HIDDEN_SIZE;
use ullm_engine::sq8_layer_runtime::{
    QWEN3_14B_SQ8_PAGED_REQUIRED_HIP_KERNEL_ENV, QWEN3_14B_SQ8_REQUIRED_HIP_KERNEL_ENV,
};
use ullm_engine::sq8_model_head_runtime::{
    QWEN3_14B_SQ8_MODEL_HEAD_REQUIRED_HIP_KERNEL_ENV, QWEN3_14B_VOCAB_SIZE,
    validate_qwen3_14b_sq8_r9700_device_info,
};
use ullm_engine::sq8_serving_runtime::{
    Qwen3Sq8ServingSession, Sq8ServingPrefillMode, load_qwen3_14b_sq8_serving_norms,
};
use ullm_runtime_sys::{RuntimeContext, device_count, device_info};

const SCHEMA_VERSION: &str = "ullm.architecture_trace.v1";
const STEP_ID: &str = "step-0000";
const UPLOAD_CHUNK_BYTES: usize = 16 * 1024 * 1024;
const LAYER_COUNT: usize = 40;

#[derive(Debug)]
struct Options {
    artifact: PathBuf,
    package: PathBuf,
    token_id: usize,
    output: PathBuf,
}

#[derive(Debug)]
struct TraceArray {
    name: String,
    shape: Vec<usize>,
    values: Vec<f32>,
}

#[derive(Debug)]
struct ZipEntry {
    name: String,
    crc32: u32,
    size: u32,
    offset: u32,
}

fn usage() -> &'static str {
    "usage: ullm-sq8-architecture-trace --artifact PATH --package PATH --token-id ID --output PATH"
}

fn main() -> ExitCode {
    match parse_options().and_then(run) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("ullm-sq8-architecture-trace: {error}");
            ExitCode::from(2)
        }
    }
}

fn parse_options() -> Result<Options, String> {
    let mut artifact = None;
    let mut package = None;
    let mut token_id = None;
    let mut output = None;
    let mut arguments = env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--artifact" => {
                if artifact
                    .replace(PathBuf::from(next_argument("--artifact", &mut arguments)?))
                    .is_some()
                {
                    return Err(format!(
                        "--artifact was supplied more than once; {}",
                        usage()
                    ));
                }
            }
            "--package" => {
                if package
                    .replace(PathBuf::from(next_argument("--package", &mut arguments)?))
                    .is_some()
                {
                    return Err(format!(
                        "--package was supplied more than once; {}",
                        usage()
                    ));
                }
            }
            "--token-id" => {
                let raw = next_argument("--token-id", &mut arguments)?;
                let parsed = raw.parse::<usize>().map_err(|_| {
                    format!("--token-id must be a non-negative integer, got {raw:?}")
                })?;
                if token_id.replace(parsed).is_some() {
                    return Err(format!(
                        "--token-id was supplied more than once; {}",
                        usage()
                    ));
                }
            }
            "--output" => {
                if output
                    .replace(PathBuf::from(next_argument("--output", &mut arguments)?))
                    .is_some()
                {
                    return Err(format!("--output was supplied more than once; {}", usage()));
                }
            }
            "--help" | "-h" => return Err(usage().to_string()),
            _ => return Err(format!("unknown argument {argument:?}; {}", usage())),
        }
    }
    Ok(Options {
        artifact: artifact.ok_or_else(|| format!("--artifact is required; {}", usage()))?,
        package: package.ok_or_else(|| format!("--package is required; {}", usage()))?,
        token_id: token_id.ok_or_else(|| format!("--token-id is required; {}", usage()))?,
        output: output.ok_or_else(|| format!("--output is required; {}", usage()))?,
    })
}

fn next_argument(
    name: &str,
    arguments: &mut impl Iterator<Item = String>,
) -> Result<String, String> {
    arguments
        .next()
        .ok_or_else(|| format!("{name} requires a value; {}", usage()))
}

fn run(options: Options) -> Result<(), String> {
    if options.output.exists() {
        return Err(format!(
            "output already exists; refusing to overwrite {}",
            options.output.display()
        ));
    }
    if options.token_id >= QWEN3_14B_VOCAB_SIZE {
        return Err(format!(
            "token ID {} is outside Qwen3-14B vocabulary 0..{QWEN3_14B_VOCAB_SIZE}",
            options.token_id
        ));
    }
    require_hip_kernel_guards()?;

    let loaded_config = load_model_config_from_package(&options.package)?;
    let resident_descriptor = loaded_config
        .resident_descriptor()
        .and_then(|descriptor| {
            descriptor.require_qwen3_14b_sq8_0()?;
            Ok(descriptor)
        })
        .map_err(|error| format!("Qwen3 SQ8_0 trace config rejection: {error}"))?;
    debug_assert_eq!(resident_descriptor.layers.len(), LAYER_COUNT);
    let model_type = match &loaded_config.model {
        ModelConfig::Qwen3(config) => config.decoder.model_type.clone(),
        _ => unreachable!("require_qwen3_full_attention returned a non-Qwen3 config"),
    };

    let embedding = read_named_passthrough_f32_rows(
        &options.package,
        QWEN3_14B_EMBED_TOKENS_TENSOR,
        &[options.token_id],
    )?;
    if embedding.columns != QWEN3_14B_HIDDEN_SIZE || embedding.values.len() != QWEN3_14B_HIDDEN_SIZE
    {
        return Err(format!(
            "embedding trace shape mismatch: columns={} values={} expected={QWEN3_14B_HIDDEN_SIZE}",
            embedding.columns,
            embedding.values.len()
        ));
    }

    let runtime_index = isolated_r9700_device()?;
    // Verify the caller's device isolation before the canonical artifact is
    // expanded.  A wrong HIP visibility setting must fail in milliseconds,
    // rather than after reading the complete SQ8_0 artifact.
    let artifact = read_sq8_canonical_artifact(&options.artifact)?;
    let mut context = RuntimeContext::create(runtime_index)?;
    let device = context.device_info()?;
    validate_qwen3_14b_sq8_r9700_device_info(&device)?;
    let mut stream = context.create_stream()?;
    let norms = load_qwen3_14b_sq8_serving_norms(&options.package, UPLOAD_CHUNK_BYTES)
        .map_err(|error| error.to_string())?;
    let started = Instant::now();
    let mut session = Qwen3Sq8ServingSession::load_with_prefill_mode(
        &mut context,
        &mut stream,
        &artifact,
        &options.package,
        norms,
        UPLOAD_CHUNK_BYTES,
        Sq8ServingPrefillMode::SequentialM1,
    )
    .map_err(|error| error.to_string())?;
    session
        .start_teacher_forced_capture_for_testing(
            "architecture-trace-step-0000",
            vec![options.token_id],
            1,
            &mut stream,
        )
        .map_err(|error| error.to_string())?;
    let capture = session
        .advance_teacher_forced_capture_for_testing(None, true, true, &mut stream)
        .map_err(|error| error.to_string())?
        .ok_or_else(|| "teacher-forced capture did not produce output".to_string())?;
    if capture.input_token_id != options.token_id || capture.position != 0 {
        return Err(format!(
            "teacher-forced trace identity mismatch: token={} position={} expected_token={} expected_position=0",
            capture.input_token_id, capture.position, options.token_id
        ));
    }
    if capture.final_hidden.len() != QWEN3_14B_HIDDEN_SIZE
        || capture.logits.len() != QWEN3_14B_VOCAB_SIZE
    {
        return Err(format!(
            "teacher-forced trace head shape mismatch: final_hidden={} logits={}",
            capture.final_hidden.len(),
            capture.logits.len()
        ));
    }
    let layers = capture
        .layers
        .ok_or_else(|| "teacher-forced trace omitted layer outputs".to_string())?;
    if layers.len() != LAYER_COUNT {
        return Err(format!(
            "teacher-forced trace layer count mismatch: expected={LAYER_COUNT} actual={}",
            layers.len()
        ));
    }

    let mut arrays = Vec::with_capacity(LAYER_COUNT + 3);
    arrays.push(TraceArray {
        name: "embedding".to_string(),
        shape: vec![1, 1, QWEN3_14B_HIDDEN_SIZE],
        values: embedding.values,
    });
    for (layer_index, values) in layers.into_iter().enumerate() {
        arrays.push(TraceArray {
            name: format!("layer-{layer_index:04}"),
            shape: vec![1, 1, QWEN3_14B_HIDDEN_SIZE],
            values,
        });
    }
    arrays.push(TraceArray {
        name: "final-norm".to_string(),
        shape: vec![1, 1, QWEN3_14B_HIDDEN_SIZE],
        values: capture.final_hidden,
    });
    arrays.push(TraceArray {
        name: "logits-last".to_string(),
        shape: vec![1, QWEN3_14B_VOCAB_SIZE],
        values: capture.logits,
    });
    for array in &arrays {
        validate_trace_array(array)?;
    }

    let tensor_names = arrays
        .iter()
        .map(|array| array.name.clone())
        .collect::<Vec<_>>();
    let tensor_shapes = arrays
        .iter()
        .map(|array| (array.name.clone(), json!(array.shape)))
        .collect::<serde_json::Map<_, _>>();
    let elapsed_seconds = started.elapsed().as_secs_f64();
    publish_trace(
        &options.output,
        &arrays,
        json!({
            "schema_version": SCHEMA_VERSION,
            "producer": "ullm-sq8-architecture-trace",
            "model_dir": loaded_config.source_model_dir,
            "config_sha256": loaded_config.config_sha256,
            "architectures": [loaded_config.architecture_kind().architecture_name()],
            "model_type": model_type,
            "weight_format": "SQ8_0 with BF16 passthrough embedding/head",
            "compute_dtype": "float32",
            "device": {
                "runtime_index": runtime_index,
                "device_id": device.device_id,
                "backend": device.backend,
                "name": device.name,
                "gcn_arch_name": device.gcn_arch_name,
            },
            "initial_token_ids": [options.token_id],
            "generated_token_ids": [capture.top1.token_id],
            "candidate_top1_logit": capture.top1.logit,
            "load_and_run_elapsed_seconds": elapsed_seconds,
            "steps": [{
                "id": STEP_ID,
                "input_token_ids": [options.token_id],
                "greedy_next_token_id": capture.top1.token_id,
                "elapsed_seconds": elapsed_seconds,
                "tensor_names": tensor_names,
                "tensor_shapes": tensor_shapes,
            }],
        }),
    )?;
    println!(
        "captured {} layers x 1 SQ8_0 GPU step to {} (top1={} elapsed={elapsed_seconds:.1}s)",
        LAYER_COUNT,
        options.output.display(),
        capture.top1.token_id,
    );
    Ok(())
}

fn require_hip_kernel_guards() -> Result<(), String> {
    let mut names = QWEN3_14B_SQ8_REQUIRED_HIP_KERNEL_ENV
        .into_iter()
        .chain(QWEN3_14B_SQ8_PAGED_REQUIRED_HIP_KERNEL_ENV)
        .chain(QWEN3_14B_SQ8_MODEL_HEAD_REQUIRED_HIP_KERNEL_ENV)
        .chain(QWEN3_14B_SQ8_EMBEDDING_REQUIRED_HIP_KERNEL_ENV)
        .collect::<Vec<_>>();
    names.sort_unstable();
    names.dedup();
    let missing = names
        .into_iter()
        .filter(|name| env::var(name).ok().as_deref() != Some("1"))
        .collect::<Vec<_>>();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "SQ8_0 architecture trace requires HIP guards equal to 1: {}",
            missing.join(", ")
        ))
    }
}

fn isolated_r9700_device() -> Result<u32, String> {
    let mut devices = Vec::new();
    for runtime_index in 1..device_count()? {
        let info = device_info(runtime_index).map_err(|error| {
            format!("failed to inspect runtime device {runtime_index}: {error}")
        })?;
        if info.backend == "hip" {
            devices.push((runtime_index, info));
        }
    }
    if devices.len() != 1 {
        return Err(format!(
            "SQ8_0 architecture trace requires exactly one isolated HIP device, found {}",
            devices.len()
        ));
    }
    let (runtime_index, info) = devices.pop().expect("checked exactly one device");
    validate_qwen3_14b_sq8_r9700_device_info(&info)?;
    if info.device_id != 0 {
        return Err(format!(
            "SQ8_0 architecture trace requires isolated HIP device ID 0, got {}",
            info.device_id
        ));
    }
    Ok(runtime_index)
}

fn validate_trace_array(array: &TraceArray) -> Result<(), String> {
    let expected = array.shape.iter().try_fold(1_usize, |product, dimension| {
        product
            .checked_mul(*dimension)
            .ok_or_else(|| format!("{} shape product overflows", array.name))
    })?;
    if array.values.len() != expected {
        return Err(format!(
            "{} values length mismatch: expected={expected} actual={}",
            array.name,
            array.values.len()
        ));
    }
    if let Some((index, value)) = array
        .values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!(
            "{} contains non-finite value at {index}: {value}",
            array.name
        ));
    }
    Ok(())
}

fn publish_trace(output: &Path, arrays: &[TraceArray], mut metadata: Value) -> Result<(), String> {
    let parent = output
        .parent()
        .ok_or_else(|| format!("trace output has no parent: {}", output.display()))?;
    fs::create_dir_all(parent).map_err(|error| {
        format!(
            "failed to create trace parent {}: {error}",
            parent.display()
        )
    })?;
    fs::create_dir(output).map_err(|error| {
        format!(
            "failed to create trace output {}: {error}",
            output.display()
        )
    })?;
    let tensor_path = output.join("tensors.npz");
    write_npz(&tensor_path, arrays)?;
    let tensor_sha256 = sha256_file(&tensor_path)?;
    let object = metadata
        .as_object_mut()
        .ok_or_else(|| "trace metadata is not an object".to_string())?;
    object.insert("tensors_file".to_string(), json!("tensors.npz"));
    object.insert("tensors_sha256".to_string(), json!(tensor_sha256));
    let metadata_path = output.join("metadata.json");
    let bytes = serde_json::to_vec_pretty(&metadata)
        .map_err(|error| format!("failed to serialize trace metadata: {error}"))?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&metadata_path)
        .map_err(|error| format!("failed to create {}: {error}", metadata_path.display()))?;
    file.write_all(&bytes)
        .and_then(|_| file.write_all(b"\n"))
        .and_then(|_| file.sync_all())
        .map_err(|error| format!("failed to write {}: {error}", metadata_path.display()))
}

fn write_npz(path: &Path, arrays: &[TraceArray]) -> Result<(), String> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("failed to create {}: {error}", path.display()))?;
    let mut entries = Vec::with_capacity(arrays.len());
    for array in arrays {
        let name = format!("{STEP_ID}__{}.npy", array.name);
        let payload = npy_f32(&array.shape, &array.values)?;
        let size = u32::try_from(payload.len())
            .map_err(|_| format!("{} NPY payload exceeds ZIP32 limit", array.name))?;
        let offset = u32::try_from(
            file.stream_position()
                .map_err(|error| format!("failed to inspect ZIP offset: {error}"))?,
        )
        .map_err(|_| "NPZ local header offset exceeds ZIP32 limit".to_string())?;
        let crc32 = crc32(&payload);
        write_u32(&mut file, 0x0403_4b50)?;
        write_u16(&mut file, 20)?;
        write_u16(&mut file, 0)?;
        write_u16(&mut file, 0)?;
        write_u16(&mut file, 0)?;
        write_u16(&mut file, 0)?;
        write_u32(&mut file, crc32)?;
        write_u32(&mut file, size)?;
        write_u32(&mut file, size)?;
        write_u16(
            &mut file,
            u16::try_from(name.len()).map_err(|_| "NPZ entry name exceeds ZIP16 limit")?,
        )?;
        write_u16(&mut file, 0)?;
        file.write_all(name.as_bytes())
            .and_then(|_| file.write_all(&payload))
            .map_err(|error| format!("failed to write NPZ entry {name}: {error}"))?;
        entries.push(ZipEntry {
            name,
            crc32,
            size,
            offset,
        });
    }
    let central_offset = u32::try_from(
        file.stream_position()
            .map_err(|error| format!("failed to inspect NPZ central offset: {error}"))?,
    )
    .map_err(|_| "NPZ central offset exceeds ZIP32 limit".to_string())?;
    for entry in &entries {
        write_u32(&mut file, 0x0201_4b50)?;
        write_u16(&mut file, 20)?;
        write_u16(&mut file, 20)?;
        write_u16(&mut file, 0)?;
        write_u16(&mut file, 0)?;
        write_u16(&mut file, 0)?;
        write_u16(&mut file, 0)?;
        write_u32(&mut file, entry.crc32)?;
        write_u32(&mut file, entry.size)?;
        write_u32(&mut file, entry.size)?;
        write_u16(
            &mut file,
            u16::try_from(entry.name.len()).map_err(|_| "NPZ entry name exceeds ZIP16 limit")?,
        )?;
        write_u16(&mut file, 0)?;
        write_u16(&mut file, 0)?;
        write_u16(&mut file, 0)?;
        write_u16(&mut file, 0)?;
        write_u32(&mut file, 0)?;
        write_u32(&mut file, entry.offset)?;
        file.write_all(entry.name.as_bytes())
            .map_err(|error| format!("failed to write NPZ central entry: {error}"))?;
    }
    let central_end = u32::try_from(
        file.stream_position()
            .map_err(|error| format!("failed to inspect NPZ central end: {error}"))?,
    )
    .map_err(|_| "NPZ central end exceeds ZIP32 limit".to_string())?;
    let central_size = central_end
        .checked_sub(central_offset)
        .ok_or_else(|| "NPZ central directory offsets are invalid".to_string())?;
    let count = u16::try_from(entries.len()).map_err(|_| "NPZ has too many entries")?;
    write_u32(&mut file, 0x0605_4b50)?;
    write_u16(&mut file, 0)?;
    write_u16(&mut file, 0)?;
    write_u16(&mut file, count)?;
    write_u16(&mut file, count)?;
    write_u32(&mut file, central_size)?;
    write_u32(&mut file, central_offset)?;
    write_u16(&mut file, 0)?;
    file.sync_all()
        .map_err(|error| format!("failed to sync {}: {error}", path.display()))
}

fn npy_f32(shape: &[usize], values: &[f32]) -> Result<Vec<u8>, String> {
    let dimensions = match shape {
        [] => return Err("NPY scalar trace tensors are unsupported".to_string()),
        [dimension] => format!("{dimension},"),
        dimensions => dimensions
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", "),
    };
    let base_header =
        format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({dimensions}), }}");
    let header_prefix_bytes = 10_usize;
    let padding = (16 - (header_prefix_bytes + base_header.len() + 1) % 16) % 16;
    let header = format!("{base_header}{}\n", " ".repeat(padding));
    let header_length = u16::try_from(header.len())
        .map_err(|_| "NPY header exceeds version 1.0 length limit".to_string())?;
    let values_bytes = values
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "NPY tensor byte length overflows".to_string())?;
    let mut bytes = Vec::with_capacity(header_prefix_bytes + header.len() + values_bytes);
    bytes.extend_from_slice(b"\x93NUMPY");
    bytes.extend_from_slice(&[1, 0]);
    bytes.extend_from_slice(&header_length.to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Ok(bytes)
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut value = !0_u32;
    for byte in bytes {
        value ^= u32::from(*byte);
        for _ in 0..8 {
            value = if value & 1 == 0 {
                value >> 1
            } else {
                (value >> 1) ^ 0xedb8_8320
            };
        }
    }
    !value
}

fn write_u16(file: &mut File, value: u16) -> Result<(), String> {
    file.write_all(&value.to_le_bytes())
        .map_err(|error| format!("failed to write ZIP u16: {error}"))
}

fn write_u32(file: &mut File, value: u32) -> Result<(), String> {
    file.write_all(&value.to_le_bytes())
        .map_err(|error| format!("failed to write ZIP u32: {error}"))
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let bytes =
        fs::read(path).map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn npy_f32_header_is_aligned_and_little_endian() {
        let bytes = npy_f32(&[1, 1, 2], &[1.5, -2.0]).unwrap();
        assert_eq!(&bytes[..6], b"\x93NUMPY");
        assert_eq!(&bytes[6..8], &[1, 0]);
        let header_length = u16::from_le_bytes(bytes[8..10].try_into().unwrap()) as usize;
        assert_eq!((10 + header_length) % 16, 0);
        assert!(
            std::str::from_utf8(&bytes[10..10 + header_length])
                .unwrap()
                .contains("'shape': (1, 1, 2)")
        );
        assert_eq!(
            f32::from_le_bytes(
                bytes[10 + header_length..14 + header_length]
                    .try_into()
                    .unwrap()
            ),
            1.5
        );
    }

    #[test]
    fn crc32_matches_zip_reference_vector() {
        assert_eq!(crc32(b"123456789"), 0xcbf4_3926);
    }
}
