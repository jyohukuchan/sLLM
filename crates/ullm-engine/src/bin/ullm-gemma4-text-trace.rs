// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Emits a diagnostic-only Gemma4 text trace in the architecture-trace schema.
//!
//! The output is intentionally consumed by `tools/architecture_hf_trace.py`.
//! It does not use a campaign, reference corpus, quantization artifact, or
//! serving session.

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{Seek, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;
use ullm_engine::gemma4_text_executor::{Gemma4TextExecutor, Gemma4TextStepTrace};

const SCHEMA_VERSION: &str = "ullm.architecture_trace.v1";
const MAX_NEW_TOKENS: usize = 4;

#[derive(Debug)]
struct Options {
    model_dir: PathBuf,
    token_ids: Vec<u32>,
    new_tokens: usize,
    output: PathBuf,
}

#[derive(Debug)]
struct TraceArray {
    key: String,
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
    "usage: ullm-gemma4-text-trace --model-dir PATH --token-ids ID[,ID...] --new-tokens 1..4 --output PATH"
}

fn main() -> ExitCode {
    match parse_options().and_then(run) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("ullm-gemma4-text-trace: {error}");
            ExitCode::from(2)
        }
    }
}

fn parse_options() -> Result<Options, String> {
    let mut model_dir = None;
    let mut token_ids = None;
    let mut new_tokens = None;
    let mut output = None;
    let mut arguments = env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--model-dir" => {
                if model_dir
                    .replace(PathBuf::from(next_argument("--model-dir", &mut arguments)?))
                    .is_some()
                {
                    return Err(format!(
                        "--model-dir was supplied more than once; {}",
                        usage()
                    ));
                }
            }
            "--token-ids" => {
                let raw = next_argument("--token-ids", &mut arguments)?;
                if token_ids.replace(parse_token_ids(&raw)?).is_some() {
                    return Err(format!(
                        "--token-ids was supplied more than once; {}",
                        usage()
                    ));
                }
            }
            "--new-tokens" => {
                let raw = next_argument("--new-tokens", &mut arguments)?;
                let value = raw.parse::<usize>().map_err(|_| {
                    format!("--new-tokens must be an integer in 1..={MAX_NEW_TOKENS}, got {raw:?}")
                })?;
                if !(1..=MAX_NEW_TOKENS).contains(&value) {
                    return Err(format!(
                        "--new-tokens must be in 1..={MAX_NEW_TOKENS}, got {value}"
                    ));
                }
                if new_tokens.replace(value).is_some() {
                    return Err(format!(
                        "--new-tokens was supplied more than once; {}",
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
        model_dir: model_dir.ok_or_else(|| format!("--model-dir is required; {}", usage()))?,
        token_ids: token_ids.ok_or_else(|| format!("--token-ids is required; {}", usage()))?,
        new_tokens: new_tokens.unwrap_or(1),
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

fn parse_token_ids(raw: &str) -> Result<Vec<u32>, String> {
    let mut result = Vec::new();
    for item in raw.split(',') {
        let item = item.trim();
        if item.is_empty() {
            return Err("--token-ids contains an empty item".into());
        }
        let token_id = item
            .parse::<u32>()
            .map_err(|_| format!("--token-ids has invalid non-negative integer {item:?}"))?;
        result.push(token_id);
    }
    if result.is_empty() {
        return Err("--token-ids must contain at least one ID".into());
    }
    Ok(result)
}

fn run(options: Options) -> Result<(), String> {
    if options.output.exists() {
        return Err(format!(
            "output already exists; refusing to overwrite {}",
            options.output.display()
        ));
    }
    let started = Instant::now();
    let mut executor = Gemma4TextExecutor::load(&options.model_dir)?;
    let config = executor.config().clone();
    let mut arrays = Vec::new();
    let mut steps = Vec::with_capacity(options.new_tokens);
    let mut generated_token_ids = Vec::with_capacity(options.new_tokens);
    let mut input = options.token_ids.clone();
    for step_index in 0..options.new_tokens {
        let step_id = format!("step-{step_index:04}");
        let step_started = Instant::now();
        let trace = executor.execute_step(&input)?;
        let elapsed_seconds = step_started.elapsed().as_secs_f64();
        let tensor_names = append_step_arrays(&mut arrays, &step_id, &trace, &config)?;
        let tensor_shapes = arrays
            .iter()
            .filter(|array| array.key.starts_with(&format!("{step_id}__")))
            .map(|array| {
                (
                    array
                        .key
                        .strip_prefix(&format!("{step_id}__"))
                        .expect("selected by exact prefix")
                        .to_string(),
                    json!(array.shape),
                )
            })
            .collect::<serde_json::Map<_, _>>();
        generated_token_ids.push(trace.top1.token_id);
        steps.push(json!({
            "id": step_id,
            "input_token_ids": input,
            "greedy_next_token_id": trace.top1.token_id,
            "candidate_top1_logit": trace.top1.logit,
            "elapsed_seconds": elapsed_seconds,
            "tensor_names": tensor_names,
            "tensor_shapes": tensor_shapes,
        }));
        input = vec![trace.top1.token_id];
    }
    let device = executor.device();
    let elapsed_seconds = started.elapsed().as_secs_f64();
    publish_trace(
        &options.output,
        &arrays,
        json!({
            "schema_version": SCHEMA_VERSION,
            "producer": "ullm-gemma4-text-executor",
            "model_dir": executor.source_model_dir(),
            "config_sha256": executor.config_sha256(),
            "architectures": ["Gemma4ForConditionalGeneration"],
            "model_type": "gemma4",
            "weight_format": "BF16 safetensors source weights",
            "compute_dtype": "float32 activations",
            "device": {
                "runtime_index": device.runtime_index,
                "device_id": device.device_id,
                "backend": &device.backend,
                "name": &device.name,
                "gcn_arch_name": &device.gcn_arch_name,
            },
            "initial_token_ids": options.token_ids,
            "generated_token_ids": generated_token_ids,
            "load_and_run_elapsed_seconds": elapsed_seconds,
            "steps": steps,
        }),
    )?;
    println!(
        "captured {} Gemma4 layers x {} step(s) to {} (generated={:?} elapsed={elapsed_seconds:.1}s)",
        config.decoder.num_hidden_layers,
        options.new_tokens,
        options.output.display(),
        generated_token_ids,
    );
    Ok(())
}

fn append_step_arrays(
    arrays: &mut Vec<TraceArray>,
    step_id: &str,
    trace: &Gemma4TextStepTrace,
    config: &ullm_engine::model_config::Gemma4TextConfig,
) -> Result<Vec<String>, String> {
    let tokens = trace.input_token_ids.len();
    let hidden = config.decoder.hidden_size;
    let expected_hidden_values = tokens
        .checked_mul(hidden)
        .ok_or_else(|| "trace hidden shape overflows".to_string())?;
    if trace.embedding.len() != expected_hidden_values
        || trace.final_norm.len() != expected_hidden_values
    {
        return Err("Gemma4 trace embedding/final-norm shape mismatch".into());
    }
    if trace.layer_outputs.len() != config.decoder.num_hidden_layers {
        return Err(format!(
            "Gemma4 trace layer count mismatch: expected {} got {}",
            config.decoder.num_hidden_layers,
            trace.layer_outputs.len()
        ));
    }
    let mut names = Vec::with_capacity(trace.layer_outputs.len() + 3);
    append_array(
        arrays,
        step_id,
        "embedding",
        vec![1, tokens, hidden],
        trace.embedding.clone(),
    )?;
    names.push("embedding".into());
    for (layer_index, values) in trace.layer_outputs.iter().enumerate() {
        if values.len() != expected_hidden_values {
            return Err(format!(
                "Gemma4 trace layer {layer_index} shape mismatch: expected {expected_hidden_values}, got {}",
                values.len()
            ));
        }
        let name = format!("layer-{layer_index:04}");
        append_array(
            arrays,
            step_id,
            &name,
            vec![1, tokens, hidden],
            values.clone(),
        )?;
        names.push(name);
    }
    append_array(
        arrays,
        step_id,
        "final-norm",
        vec![1, tokens, hidden],
        trace.final_norm.clone(),
    )?;
    names.push("final-norm".into());
    if trace.logits_last.len() != config.decoder.vocab_size {
        return Err(format!(
            "Gemma trace logits width mismatch: expected {} got {}",
            config.decoder.vocab_size,
            trace.logits_last.len()
        ));
    }
    append_array(
        arrays,
        step_id,
        "logits-last",
        vec![1, config.decoder.vocab_size],
        trace.logits_last.clone(),
    )?;
    names.push("logits-last".into());
    Ok(names)
}

fn append_array(
    arrays: &mut Vec<TraceArray>,
    step_id: &str,
    name: &str,
    shape: Vec<usize>,
    values: Vec<f32>,
) -> Result<(), String> {
    let expected = shape.iter().try_fold(1_usize, |product, dimension| {
        product
            .checked_mul(*dimension)
            .ok_or_else(|| format!("trace {name} shape product overflows"))
    })?;
    if values.len() != expected {
        return Err(format!(
            "trace {name} values length mismatch: expected {expected}, got {}",
            values.len()
        ));
    }
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!(
            "trace {name} contains non-finite value at {index}: {value}"
        ));
    }
    arrays.push(TraceArray {
        key: format!("{step_id}__{name}"),
        shape,
        values,
    });
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
    let object = metadata
        .as_object_mut()
        .ok_or_else(|| "trace metadata is not an object".to_string())?;
    object.insert("tensors_file".to_string(), json!("tensors.npz"));
    object.insert(
        "tensors_sha256".to_string(),
        json!(sha256_file(&tensor_path)?),
    );
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
        let name = format!("{}.npy", array.key);
        let payload = npy_f32(&array.shape, &array.values)?;
        let size = u32::try_from(payload.len())
            .map_err(|_| format!("{} NPY payload exceeds ZIP32 limit", array.key))?;
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
    let padding = (16 - (10 + base_header.len() + 1) % 16) % 16;
    let header = format!("{base_header}{}\n", " ".repeat(padding));
    let header_length = u16::try_from(header.len())
        .map_err(|_| "NPY header exceeds version 1.0 length limit".to_string())?;
    let values_bytes = values
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "NPY tensor byte length overflows".to_string())?;
    let mut bytes = Vec::with_capacity(10 + header.len() + values_bytes);
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
        let header_length = u16::from_le_bytes(bytes[8..10].try_into().unwrap()) as usize;
        assert_eq!((10 + header_length) % 16, 0);
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
