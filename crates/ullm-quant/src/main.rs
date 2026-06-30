use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use serde::{Deserialize, Serialize};

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct KernelVersion {
    major: u32,
    minor: u32,
    patch: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct AqQuantMetrics {
    elements: u64,
    groups: u64,
    sse: f64,
    ref_sse: f64,
    max_abs_error: f32,
    index_counts: [u64; 16],
    scale_index_min: u32,
    scale_index_max: u32,
    scale_window_improved_groups: u64,
}

const ULLM_AQ_DTYPE_BF16: u32 = 1;
const ULLM_AQ_DTYPE_F16: u32 = 2;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct AqQuantizeChunkRequestV1 {
    struct_size: usize,
    dtype: u32,
    reserved0: u32,
    input: *const u8,
    input_bytes: usize,
    group_size: usize,
    scale_values: *const f32,
    scale_count: usize,
    codebook: *const f32,
    codebook_count: usize,
    tensor_scale: f32,
    reserved1: u32,
    scale_window: usize,
    packed_indices: *mut u8,
    packed_indices_bytes: usize,
    scale_indices: *mut u8,
    scale_indices_bytes: usize,
}

unsafe extern "C" {
    fn ullm_aq_get_kernel_version() -> KernelVersion;
    fn ullm_aq_pack_nibbles(low: *const u8, high: *const u8, output: *mut u8, len: usize) -> usize;
    fn ullm_aq_quantize_chunk_v1(
        request: *const AqQuantizeChunkRequestV1,
        metrics: *mut AqQuantMetrics,
        metrics_size: usize,
    ) -> i32;
}

#[derive(Debug)]
struct Options {
    threads: usize,
    io_threads: usize,
    max_working_memory_mib: usize,
    model_dir: Option<PathBuf>,
    plan_output: Option<PathBuf>,
    inspect_tensor: Option<String>,
    skip_inspect: bool,
    inspect_aq_format: Option<String>,
    codebook_json: Option<PathBuf>,
    inspect_codebook_family: Option<String>,
    inspect_codebook_candidate: Option<String>,
    prototype_output_dir: Option<PathBuf>,
    prototype_verify: bool,
    tensor_scale_override: Option<f32>,
    chunk_bytes: usize,
    scale_window: usize,
    aq_policy: String,
    aq_high_families: Vec<String>,
    aq_low_format: String,
    aq_high_format: String,
    dry_run: bool,
}

#[derive(Debug, Deserialize)]
struct SafetensorsIndex {
    weight_map: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Deserialize)]
struct TensorHeader {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

#[derive(Debug, Deserialize)]
struct CodebookExport {
    codebooks: Vec<CodebookEntry>,
}

#[derive(Debug, Deserialize)]
struct CodebookEntry {
    family: String,
    candidate_id: String,
    values_f32: Vec<f32>,
}

#[derive(Debug)]
struct SafetensorsMetadata {
    data_start: u64,
    tensors: BTreeMap<String, TensorHeader>,
}

#[derive(Debug)]
struct TensorLocation {
    source_file: PathBuf,
    data_start: u64,
    header: TensorHeader,
}

#[derive(Debug)]
struct TensorInspectResult {
    name: String,
    source_file: PathBuf,
    dtype: String,
    shape: Vec<usize>,
    payload_bytes: usize,
    chunk_bytes: usize,
    chunks: usize,
    fnv1a64: u64,
    numeric_stats: Option<NumericStats>,
    aq_group_stats: Option<AqGroupStats>,
    quant_dry_run_stats: Option<QuantDryRunStats>,
}

#[derive(Clone, Debug)]
struct NumericStats {
    elements: usize,
    finite_elements: usize,
    nan_elements: usize,
    min: f32,
    max: f32,
    sum_abs: f64,
    max_abs: f32,
}

#[derive(Clone, Debug)]
struct AqGroupStats {
    format: String,
    scale_format: String,
    scale_values: Vec<f32>,
    group_size: usize,
    groups: usize,
    sum_absmax: f64,
    max_absmax: f32,
    zero_absmax_groups: usize,
    scale_index_min: usize,
    scale_index_max: usize,
    scale_clamped_low: usize,
    scale_clamped_high: usize,
    sum_scale_relative_error: f64,
}

#[derive(Clone, Debug)]
struct QuantDryRunStats {
    elements: usize,
    groups: usize,
    sse: f64,
    ref_sse: f64,
    max_abs_error: f32,
    index_counts: Vec<usize>,
    tensor_scale: f32,
    scale_window: usize,
    scale_index_min: usize,
    scale_index_max: usize,
    scale_window_improved_groups: usize,
}

#[derive(Debug, Serialize)]
struct TensorPlan {
    name: String,
    source_file: String,
    dtype: String,
    shape: Vec<usize>,
    family: String,
    n_elements: usize,
    n_bytes: usize,
    supported_input: bool,
    action: String,
    quant_format: Option<String>,
    quant_role: Option<String>,
    estimated_output_bytes: usize,
    estimated_effective_bpp: f64,
}

#[derive(Clone, Debug, Serialize)]
struct AqPolicyPlan {
    policy_id: String,
    low_format: String,
    high_format: String,
    high_families: Vec<String>,
}

#[derive(Debug, Serialize)]
struct ModelPlan {
    schema_version: String,
    model_dir: String,
    aq_policy: AqPolicyPlan,
    tensor_count: usize,
    supported_tensor_count: usize,
    passthrough_tensor_count: usize,
    total_tensor_bytes: usize,
    total_estimated_output_bytes: usize,
    estimated_output_to_input_ratio: f64,
    tensors: Vec<TensorPlan>,
}

#[derive(Debug, Serialize, Deserialize)]
struct PrototypeManifest {
    schema_version: String,
    source_model_dir: String,
    tensors: Vec<PrototypeTensorManifest>,
    codebooks: Vec<PrototypeCodebookManifest>,
}

#[derive(Debug, Serialize, Deserialize)]
struct PrototypeTensorManifest {
    name: String,
    source_file: String,
    dtype: String,
    shape: Vec<usize>,
    family: String,
    candidate_id: String,
    scale_format: String,
    group_size: usize,
    tensor_scale: f32,
    scale_window: usize,
    elements: usize,
    groups: usize,
    index_file: String,
    index_encoding: String,
    scale_file: String,
    scale_encoding: String,
    codebook_file: String,
    metrics: PrototypeTensorMetrics,
}

#[derive(Debug, Serialize, Deserialize)]
struct PrototypeTensorMetrics {
    mse: f64,
    relative_mse: f64,
    max_abs_error: f32,
    scale_index_min: usize,
    scale_index_max: usize,
    scale_window_improved_groups: usize,
    index_counts: Vec<usize>,
}

#[derive(Debug, Serialize, Deserialize)]
struct PrototypeCodebookManifest {
    family: String,
    candidate_id: String,
    file: String,
    encoding: String,
    entries: usize,
}

#[derive(Debug)]
struct PrototypeVerifyResult {
    elements: usize,
    groups: usize,
    mse: f64,
    relative_mse: f64,
    max_abs_error: f32,
    index_file_bytes: usize,
    scale_file_bytes: usize,
    codebook_entries: usize,
}

fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(NonZeroUsize::get)
        .map(|threads| threads.min(64))
        .unwrap_or(1)
}

fn parse_usize(flag: &str, value: Option<String>) -> Result<usize, String> {
    let raw = value.ok_or_else(|| format!("{flag} requires a value"))?;
    let parsed = raw
        .parse::<usize>()
        .map_err(|_| format!("{flag} must be a positive integer"))?;
    if parsed == 0 {
        return Err(format!("{flag} must be >= 1"));
    }
    Ok(parsed)
}

fn parse_usize_zero_allowed(flag: &str, value: Option<String>) -> Result<usize, String> {
    let raw = value.ok_or_else(|| format!("{flag} requires a value"))?;
    raw.parse::<usize>()
        .map_err(|_| format!("{flag} must be a non-negative integer"))
}

fn parse_positive_f32(flag: &str, value: Option<String>) -> Result<f32, String> {
    let raw = value.ok_or_else(|| format!("{flag} requires a value"))?;
    let parsed = raw
        .parse::<f32>()
        .map_err(|_| format!("{flag} must be a positive finite float"))?;
    if !parsed.is_finite() || parsed <= 0.0 {
        return Err(format!("{flag} must be a positive finite float"));
    }
    Ok(parsed)
}

fn parse_options() -> Result<Options, String> {
    let mut args = env::args().skip(1);
    let mut options = Options {
        threads: default_threads(),
        io_threads: 2,
        max_working_memory_mib: 4096,
        model_dir: None,
        plan_output: None,
        inspect_tensor: None,
        skip_inspect: false,
        inspect_aq_format: None,
        codebook_json: None,
        inspect_codebook_family: None,
        inspect_codebook_candidate: None,
        prototype_output_dir: None,
        prototype_verify: true,
        tensor_scale_override: None,
        chunk_bytes: 64 * 1024 * 1024,
        scale_window: 0,
        aq_policy: "all-g16".to_string(),
        aq_high_families: Vec::new(),
        aq_low_format: "aq4_e4m3_g16_ts_flloyd16".to_string(),
        aq_high_format: "aq4_e4m3_g8_ts_flloyd16".to_string(),
        dry_run: false,
    };

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--threads" => options.threads = parse_usize("--threads", args.next())?,
            "--io-threads" => options.io_threads = parse_usize("--io-threads", args.next())?,
            "--model-dir" => {
                options.model_dir = Some(PathBuf::from(
                    args.next()
                        .ok_or_else(|| "--model-dir requires a value".to_string())?,
                ));
            }
            "--plan-output" => {
                options.plan_output =
                    Some(PathBuf::from(args.next().ok_or_else(|| {
                        "--plan-output requires a value".to_string()
                    })?));
            }
            "--inspect-tensor" => {
                options.inspect_tensor = Some(
                    args.next()
                        .ok_or_else(|| "--inspect-tensor requires a value".to_string())?,
                );
            }
            "--skip-inspect" => options.skip_inspect = true,
            "--inspect-aq-format" => {
                options.inspect_aq_format = Some(
                    args.next()
                        .ok_or_else(|| "--inspect-aq-format requires a value".to_string())?,
                );
            }
            "--codebook-json" => {
                options.codebook_json =
                    Some(PathBuf::from(args.next().ok_or_else(|| {
                        "--codebook-json requires a value".to_string()
                    })?));
            }
            "--inspect-codebook-family" => {
                options.inspect_codebook_family = Some(
                    args.next()
                        .ok_or_else(|| "--inspect-codebook-family requires a value".to_string())?,
                );
            }
            "--inspect-codebook-candidate" => {
                options.inspect_codebook_candidate =
                    Some(args.next().ok_or_else(|| {
                        "--inspect-codebook-candidate requires a value".to_string()
                    })?);
            }
            "--prototype-output-dir" => {
                options.prototype_output_dir =
                    Some(PathBuf::from(args.next().ok_or_else(|| {
                        "--prototype-output-dir requires a value".to_string()
                    })?));
            }
            "--prototype-skip-verify" => options.prototype_verify = false,
            "--tensor-scale-override" => {
                options.tensor_scale_override =
                    Some(parse_positive_f32("--tensor-scale-override", args.next())?)
            }
            "--chunk-bytes" => options.chunk_bytes = parse_usize("--chunk-bytes", args.next())?,
            "--scale-window" => {
                options.scale_window = parse_usize_zero_allowed("--scale-window", args.next())?
            }
            "--aq-policy" => {
                options.aq_policy = args
                    .next()
                    .ok_or_else(|| "--aq-policy requires a value".to_string())?;
            }
            "--aq-high-family" => {
                options.aq_high_families.push(
                    args.next()
                        .ok_or_else(|| "--aq-high-family requires a value".to_string())?,
                );
            }
            "--aq-low-format" => {
                options.aq_low_format = args
                    .next()
                    .ok_or_else(|| "--aq-low-format requires a value".to_string())?;
            }
            "--aq-high-format" => {
                options.aq_high_format = args
                    .next()
                    .ok_or_else(|| "--aq-high-format requires a value".to_string())?;
            }
            "--max-working-memory-mib" => {
                options.max_working_memory_mib =
                    parse_usize("--max-working-memory-mib", args.next())?;
            }
            "--dry-run" => options.dry_run = true,
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            unknown => return Err(format!("unknown argument: {unknown}")),
        }
    }

    Ok(options)
}

fn print_help() {
    println!("ullm-quant");
    println!();
    println!("Options:");
    println!("  --model-dir <PATH>            HF safetensors model directory");
    println!("  --plan-output <PATH>          write metadata plan JSON");
    println!(
        "  --inspect-tensor <NAME>       read one tensor payload in chunks and print checksum"
    );
    println!("  --skip-inspect                use --inspect-tensor only as a target selector");
    println!("  --inspect-aq-format <ID>      also compute group absmax stats for an aq format");
    println!("  --codebook-json <PATH>        exported aq family codebook JSON");
    println!("  --inspect-codebook-family <F> inspect one family from --codebook-json");
    println!("  --inspect-codebook-candidate <ID>");
    println!("  --prototype-output-dir <PATH> write one inspected tensor to a .ullm.d prototype");
    println!("  --prototype-skip-verify       skip prototype re-read/dequant verification");
    println!("  --tensor-scale-override <F>   skip tensor-scale estimation for prototype output");
    println!("  --chunk-bytes <N>             payload chunk size for inspection/conversion");
    println!("  --scale-window <N>            try +/- N scale entries during quant dry-run");
    println!("  --aq-policy <ID>              all-g16, all-g8, p4p6, p4p9, or custom");
    println!("  --aq-high-family <FAMILY>     high-format family for custom policy; repeatable");
    println!("  --aq-low-format <ID>          low-budget aq candidate id");
    println!("  --aq-high-format <ID>         high-budget aq candidate id");
    println!("  --threads <N>                 compute worker threads");
    println!("  --io-threads <N>              read/write helper threads");
    println!("  --max-working-memory-mib <N>  working-memory budget");
    println!("  --dry-run                     print the current skeleton plan");
}

fn family_for_tensor(name: &str) -> &'static str {
    if name.contains("self_attn.q_proj") {
        "attn_q"
    } else if name.contains("self_attn.k_proj") {
        "attn_k"
    } else if name.contains("self_attn.v_proj") {
        "attn_v"
    } else if name.contains("self_attn.o_proj") {
        "attn_o"
    } else if name.contains("linear_attn.in_proj_qkv") {
        "linear_attn_qkv"
    } else if name.contains("linear_attn.in_proj_a") {
        "linear_attn_a"
    } else if name.contains("linear_attn.in_proj_b") {
        "linear_attn_b"
    } else if name.contains("linear_attn.in_proj_z") {
        "linear_attn_z"
    } else if name.contains("linear_attn.out_proj") {
        "linear_attn_out"
    } else if name.contains("mlp.gate_proj") {
        "mlp_gate"
    } else if name.contains("mlp.up_proj") {
        "mlp_up"
    } else if name.contains("mlp.down_proj") {
        "mlp_down"
    } else if name.contains("embed_tokens") {
        "embed"
    } else if name.contains("lm_head") {
        "lm_head"
    } else if name.contains("router") {
        "moe_router"
    } else if name.contains("experts") {
        "moe_expert"
    } else {
        "other"
    }
}

fn is_default_quant_family(family: &str) -> bool {
    matches!(
        family,
        "attn_q"
            | "attn_k"
            | "attn_v"
            | "attn_o"
            | "linear_attn_qkv"
            | "linear_attn_a"
            | "linear_attn_b"
            | "linear_attn_z"
            | "linear_attn_out"
            | "mlp_gate"
            | "mlp_up"
            | "mlp_down"
    )
}

fn default_quant_families() -> BTreeSet<String> {
    [
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_o",
        "linear_attn_qkv",
        "linear_attn_a",
        "linear_attn_b",
        "linear_attn_z",
        "linear_attn_out",
        "mlp_gate",
        "mlp_up",
        "mlp_down",
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}

fn resolve_aq_policy(options: &Options) -> Result<AqPolicyPlan, String> {
    let high_families: BTreeSet<String> = match options.aq_policy.as_str() {
        "all-g16" => BTreeSet::new(),
        "all-g8" => default_quant_families(),
        "p4p6" => ["attn_k", "attn_o", "attn_v", "linear_attn_out"]
            .into_iter()
            .map(str::to_string)
            .collect(),
        "p4p9" => [
            "attn_k",
            "attn_o",
            "attn_q",
            "attn_v",
            "linear_attn_out",
            "mlp_gate",
            "mlp_up",
        ]
        .into_iter()
        .map(str::to_string)
        .collect(),
        "custom" => options.aq_high_families.iter().cloned().collect(),
        unknown => {
            return Err(format!(
                "unknown --aq-policy {unknown}; expected all-g16, all-g8, p4p6, p4p9, or custom"
            ));
        }
    };
    if options.aq_policy != "custom" && !options.aq_high_families.is_empty() {
        return Err("--aq-high-family can only be used with --aq-policy custom".to_string());
    }
    let default_families = default_quant_families();
    let unknown_families: Vec<_> = high_families
        .iter()
        .filter(|family| !default_families.contains(*family))
        .cloned()
        .collect();
    if !unknown_families.is_empty() {
        return Err(format!(
            "unknown aq high families: {}",
            unknown_families.join(",")
        ));
    }
    Ok(AqPolicyPlan {
        policy_id: options.aq_policy.clone(),
        low_format: options.aq_low_format.clone(),
        high_format: options.aq_high_format.clone(),
        high_families: high_families.into_iter().collect(),
    })
}

fn is_supported_input(dtype: &str, shape: &[usize], family: &str) -> bool {
    matches!(dtype, "BF16" | "F16" | "F32") && shape.len() >= 2 && is_default_quant_family(family)
}

fn quant_assignment(
    supported_input: bool,
    family: &str,
    policy: &AqPolicyPlan,
) -> (Option<String>, Option<String>) {
    if !supported_input {
        return (None, None);
    }
    if policy.high_families.iter().any(|item| item == family) {
        (Some(policy.high_format.clone()), Some("high".to_string()))
    } else {
        (Some(policy.low_format.clone()), Some("low".to_string()))
    }
}

fn aq_group_size(format: &str) -> Result<usize, String> {
    if format.contains("_g8_") {
        Ok(8)
    } else if format.contains("_g16_") {
        Ok(16)
    } else {
        Err(format!("cannot infer aq group size from format: {format}"))
    }
}

fn aq_scale_format(format: &str) -> Result<&'static str, String> {
    if format.contains("_e8m0_") {
        Ok("e8m0")
    } else if format.contains("_e5m2_") {
        Ok("e5m2")
    } else if format.contains("_e4m3_") {
        Ok("e4m3")
    } else if format.contains("_ue5m3_") {
        Ok("ue5m3")
    } else {
        Err(format!(
            "cannot infer aq scale format from format: {format}"
        ))
    }
}

fn decode_e8m0() -> Vec<f32> {
    (0..255).map(|code| 2.0f32.powi(code - 127)).collect()
}

fn decode_ieee_like_float(exp_bits: u32, mant_bits: u32, bias: i32) -> Vec<f32> {
    let mut values = Vec::new();
    let max_exp = (1u32 << exp_bits) - 1;
    for exp in 0..max_exp {
        for mant in 0..(1u32 << mant_bits) {
            if exp == 0 {
                if mant == 0 {
                    continue;
                }
                values.push((mant as f32 / (1u32 << mant_bits) as f32) * 2.0f32.powi(1 - bias));
            } else {
                values.push(
                    (1.0 + mant as f32 / (1u32 << mant_bits) as f32)
                        * 2.0f32.powi(exp as i32 - bias),
                );
            }
        }
    }
    values.sort_by(|left, right| left.total_cmp(right));
    values.dedup_by(|left, right| left == right);
    values
}

fn scale_values(scale_format: &str) -> Result<Vec<f32>, String> {
    match scale_format {
        "e8m0" => Ok(decode_e8m0()),
        "e5m2" => Ok(decode_ieee_like_float(5, 2, 15)),
        "e4m3" => Ok(decode_ieee_like_float(4, 3, 7)),
        "ue5m3" => Ok(decode_ieee_like_float(5, 3, 15)),
        _ => Err(format!("unknown aq scale format: {scale_format}")),
    }
}

fn nearest_scale_index(target: f32, scales: &[f32]) -> (usize, bool, bool) {
    debug_assert!(!scales.is_empty());
    if target <= scales[0] {
        return (0, target < scales[0], false);
    }
    let last = scales.len() - 1;
    if target >= scales[last] {
        return (last, false, target > scales[last]);
    }
    let idx = scales.partition_point(|scale| *scale < target);
    let prev = idx - 1;
    if (target - scales[prev]).abs() < (target - scales[idx]).abs() {
        (prev, false, false)
    } else {
        (idx, false, false)
    }
}

fn lower_median(values: &mut [f32]) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|left, right| left.total_cmp(right));
    Some(values[(values.len() - 1) / 2])
}

fn aq_uses_tensor_scale(format: &str) -> bool {
    format.contains("_ts_")
}

fn max_codebook_abs(codebook: &[f32]) -> f32 {
    codebook
        .iter()
        .map(|value| value.abs())
        .fold(0.0f32, f32::max)
        .max(1e-12)
}

fn load_codebook_export(path: &Path) -> Result<CodebookExport, String> {
    let text = fs::read_to_string(path)
        .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    serde_json::from_str(&text)
        .map_err(|err| format!("failed to parse codebook JSON {}: {err}", path.display()))
}

fn select_codebook<'a>(
    export: &'a CodebookExport,
    family: &str,
    candidate_id: &str,
) -> Result<&'a [f32], String> {
    let entry = export
        .codebooks
        .iter()
        .find(|entry| entry.family == family && entry.candidate_id == candidate_id)
        .ok_or_else(|| {
            format!("codebook not found for family={family}, candidate={candidate_id}")
        })?;
    if entry.values_f32.len() != 16 {
        return Err(format!(
            "codebook for family={family}, candidate={candidate_id} has {} entries, expected 16",
            entry.values_f32.len()
        ));
    }
    Ok(&entry.values_f32)
}

fn div_ceil(value: usize, divisor: usize) -> usize {
    value.div_ceil(divisor)
}

fn estimate_output_bytes(
    n_elements: usize,
    n_bytes: usize,
    quant_format: Option<&str>,
) -> Result<usize, String> {
    match quant_format {
        Some(format) => {
            let group_size = aq_group_size(format)?;
            let index_bytes = div_ceil(n_elements, 2);
            let scale_bytes = div_ceil(n_elements, group_size);
            Ok(index_bytes + scale_bytes)
        }
        None => Ok(n_bytes),
    }
}

fn effective_bpp(n_elements: usize, bytes: usize) -> f64 {
    if n_elements == 0 {
        0.0
    } else {
        (bytes as f64 * 8.0) / n_elements as f64
    }
}

fn tensor_elements(shape: &[usize]) -> Result<usize, String> {
    shape.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim)
            .ok_or_else(|| format!("tensor element count overflows usize for shape {shape:?}"))
    })
}

fn read_safetensors_metadata(path: &Path) -> Result<SafetensorsMetadata, String> {
    let mut file =
        fs::File::open(path).map_err(|err| format!("failed to open {}: {err}", path.display()))?;
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes).map_err(|err| {
        format!(
            "failed to read safetensors header length from {}: {err}",
            path.display()
        )
    })?;
    let header_len = u64::from_le_bytes(len_bytes) as usize;
    if header_len > 128 * 1024 * 1024 {
        return Err(format!(
            "safetensors header is unexpectedly large: {header_len} bytes"
        ));
    }
    let data_start = 8u64
        .checked_add(header_len as u64)
        .ok_or_else(|| format!("safetensors data start overflows for {}", path.display()))?;
    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes).map_err(|err| {
        format!(
            "failed to read safetensors header from {}: {err}",
            path.display()
        )
    })?;
    let raw: BTreeMap<String, serde_json::Value> =
        serde_json::from_slice(&header_bytes).map_err(|err| {
            format!(
                "failed to parse safetensors header {}: {err}",
                path.display()
            )
        })?;
    let mut tensors = BTreeMap::new();
    for (name, value) in raw {
        if name == "__metadata__" {
            continue;
        }
        let header: TensorHeader = serde_json::from_value(value).map_err(|err| {
            format!(
                "failed to parse tensor header {name} in {}: {err}",
                path.display()
            )
        })?;
        tensors.insert(name, header);
    }
    Ok(SafetensorsMetadata {
        data_start,
        tensors,
    })
}

fn read_safetensors_header(path: &Path) -> Result<BTreeMap<String, TensorHeader>, String> {
    Ok(read_safetensors_metadata(path)?.tensors)
}

fn tensor_payload_bytes(header: &TensorHeader) -> Result<usize, String> {
    header
        .data_offsets
        .get(1)
        .zip(header.data_offsets.first())
        .map(|(end, start)| end.saturating_sub(*start))
        .ok_or_else(|| "invalid safetensors data offsets".to_string())
}

fn read_tensor_payload_chunk(
    path: &Path,
    data_start: u64,
    header: &TensorHeader,
    offset: usize,
    len: usize,
) -> Result<Vec<u8>, String> {
    let payload_len = tensor_payload_bytes(header)?;
    if offset > payload_len {
        return Err(format!(
            "tensor chunk offset {offset} exceeds payload length {payload_len}"
        ));
    }
    let read_len = len.min(payload_len - offset);
    let tensor_start = data_start
        .checked_add(header.data_offsets[0] as u64)
        .ok_or_else(|| format!("tensor data start overflows for {}", path.display()))?;
    let absolute_offset = tensor_start
        .checked_add(offset as u64)
        .ok_or_else(|| format!("tensor chunk offset overflows for {}", path.display()))?;
    let mut file =
        fs::File::open(path).map_err(|err| format!("failed to open {}: {err}", path.display()))?;
    file.seek(SeekFrom::Start(absolute_offset))
        .map_err(|err| format!("failed to seek {}: {err}", path.display()))?;
    let mut bytes = vec![0u8; read_len];
    file.read_exact(&mut bytes)
        .map_err(|err| format!("failed to read tensor chunk from {}: {err}", path.display()))?;
    Ok(bytes)
}

fn find_tensor_location(model_dir: &Path, tensor_name: &str) -> Result<TensorLocation, String> {
    for path in safetensor_files(model_dir)? {
        let metadata = read_safetensors_metadata(&path)?;
        if let Some(header) = metadata.tensors.get(tensor_name) {
            return Ok(TensorLocation {
                source_file: path,
                data_start: metadata.data_start,
                header: header.clone(),
            });
        }
    }
    Err(format!(
        "tensor {tensor_name} not found in {}",
        model_dir.display()
    ))
}

fn fnv1a64_update(mut hash: u64, bytes: &[u8]) -> u64 {
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn numeric_element_size(dtype: &str) -> Option<usize> {
    match dtype {
        "BF16" => Some(2),
        "F16" => Some(2),
        "F32" => Some(4),
        _ => None,
    }
}

fn f16_to_f32(raw: u16) -> f32 {
    let sign = (u32::from(raw & 0x8000)) << 16;
    let exp = (raw >> 10) & 0x1f;
    let frac = raw & 0x03ff;
    let bits = if exp == 0 {
        if frac == 0 {
            sign
        } else {
            let mut frac_norm = frac;
            let mut exp_unbiased = -14i32;
            while (frac_norm & 0x0400) == 0 {
                frac_norm <<= 1;
                exp_unbiased -= 1;
            }
            frac_norm &= 0x03ff;
            sign | (((exp_unbiased + 127) as u32) << 23) | (u32::from(frac_norm) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (u32::from(frac) << 13)
    } else {
        let exp_f32 = i32::from(exp) - 15 + 127;
        sign | ((exp_f32 as u32) << 23) | (u32::from(frac) << 13)
    };
    f32::from_bits(bits)
}

fn decode_numeric_value(dtype: &str, bytes: &[u8]) -> Result<f32, String> {
    match dtype {
        "BF16" => {
            let raw = u16::from_le_bytes([bytes[0], bytes[1]]);
            Ok(f32::from_bits(u32::from(raw) << 16))
        }
        "F16" => {
            let raw = u16::from_le_bytes([bytes[0], bytes[1]]);
            Ok(f16_to_f32(raw))
        }
        "F32" => Ok(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])),
        _ => Err(format!("numeric stats are not supported for dtype {dtype}")),
    }
}

fn new_numeric_stats() -> NumericStats {
    NumericStats {
        elements: 0,
        finite_elements: 0,
        nan_elements: 0,
        min: f32::INFINITY,
        max: f32::NEG_INFINITY,
        sum_abs: 0.0,
        max_abs: 0.0,
    }
}

fn update_numeric_stats(dtype: &str, bytes: &[u8], stats: &mut NumericStats) -> Result<(), String> {
    let element_size = numeric_element_size(dtype)
        .ok_or_else(|| format!("numeric stats are not supported for dtype {dtype}"))?;
    if bytes.len() % element_size != 0 {
        return Err(format!(
            "{dtype} numeric chunk has {} bytes, not divisible by element size {element_size}",
            bytes.len()
        ));
    }
    for chunk in bytes.chunks_exact(element_size) {
        let value = decode_numeric_value(dtype, chunk)?;
        stats.elements += 1;
        if value.is_nan() {
            stats.nan_elements += 1;
            continue;
        }
        stats.finite_elements += 1;
        stats.min = stats.min.min(value);
        stats.max = stats.max.max(value);
        let abs = value.abs();
        stats.sum_abs += f64::from(abs);
        stats.max_abs = stats.max_abs.max(abs);
    }
    Ok(())
}

fn new_aq_group_stats(format: &str, group_size: usize) -> Result<AqGroupStats, String> {
    let scale_format = aq_scale_format(format)?;
    let scale_values = scale_values(scale_format)?;
    Ok(AqGroupStats {
        format: format.to_string(),
        scale_format: scale_format.to_string(),
        scale_values,
        group_size,
        groups: 0,
        sum_absmax: 0.0,
        max_absmax: 0.0,
        zero_absmax_groups: 0,
        scale_index_min: usize::MAX,
        scale_index_max: 0,
        scale_clamped_low: 0,
        scale_clamped_high: 0,
        sum_scale_relative_error: 0.0,
    })
}

fn update_aq_group_stats(
    dtype: &str,
    bytes: &[u8],
    stats: &mut AqGroupStats,
) -> Result<(), String> {
    let element_size = numeric_element_size(dtype)
        .ok_or_else(|| format!("aq group stats are not supported for dtype {dtype}"))?;
    let group_bytes = stats
        .group_size
        .checked_mul(element_size)
        .ok_or_else(|| "aq group byte size overflows".to_string())?;
    if bytes.len() % group_bytes != 0 {
        return Err(format!(
            "{dtype} aq group chunk has {} bytes, not divisible by group byte size {group_bytes}",
            bytes.len()
        ));
    }
    for group in bytes.chunks_exact(group_bytes) {
        let mut absmax = 0.0f32;
        for item in group.chunks_exact(element_size) {
            let value = decode_numeric_value(dtype, item)?;
            if value.is_nan() {
                continue;
            }
            absmax = absmax.max(value.abs());
        }
        stats.groups += 1;
        stats.sum_absmax += f64::from(absmax);
        stats.max_absmax = stats.max_absmax.max(absmax);
        if absmax == 0.0 {
            stats.zero_absmax_groups += 1;
            continue;
        }
        let (scale_index, clamped_low, clamped_high) =
            nearest_scale_index(absmax, &stats.scale_values);
        stats.scale_index_min = stats.scale_index_min.min(scale_index);
        stats.scale_index_max = stats.scale_index_max.max(scale_index);
        if clamped_low {
            stats.scale_clamped_low += 1;
        }
        if clamped_high {
            stats.scale_clamped_high += 1;
        }
        let scale_value = stats.scale_values[scale_index];
        stats.sum_scale_relative_error += f64::from((scale_value - absmax).abs() / absmax);
    }
    Ok(())
}

fn collect_group_target_scales(
    dtype: &str,
    bytes: &[u8],
    group_size: usize,
    max_code: f32,
    target_scales: &mut Vec<f32>,
) -> Result<(), String> {
    let element_size = numeric_element_size(dtype)
        .ok_or_else(|| format!("tensor scale estimation is not supported for dtype {dtype}"))?;
    let group_bytes = group_size
        .checked_mul(element_size)
        .ok_or_else(|| "tensor scale group byte size overflows".to_string())?;
    if bytes.len() % group_bytes != 0 {
        return Err(format!(
            "{dtype} tensor scale chunk has {} bytes, not divisible by group byte size {group_bytes}",
            bytes.len()
        ));
    }
    for group in bytes.chunks_exact(group_bytes) {
        let mut absmax = 0.0f32;
        for item in group.chunks_exact(element_size) {
            let value = decode_numeric_value(dtype, item)?;
            if !value.is_nan() {
                absmax = absmax.max(value.abs());
            }
        }
        if absmax > 0.0 {
            target_scales.push(absmax / max_code);
        }
    }
    Ok(())
}

fn estimate_tensor_scale(
    location: &TensorLocation,
    payload_bytes: usize,
    chunk_bytes: usize,
    group_size: usize,
    scale_values: &[f32],
    codebook: &[f32],
) -> Result<f32, String> {
    if codebook.is_empty() {
        return Err("tensor scale estimation requires a non-empty codebook".to_string());
    }
    if scale_values.is_empty() {
        return Err("tensor scale estimation requires at least one scale value".to_string());
    }
    let group_bytes = group_size
        .checked_mul(numeric_element_size(&location.header.dtype).ok_or_else(|| {
            format!(
                "tensor scale is not supported for dtype {}",
                location.header.dtype
            )
        })?)
        .ok_or_else(|| "tensor scale group byte size overflows".to_string())?;
    let estimated_groups = payload_bytes / group_bytes;
    let mut target_scales = Vec::with_capacity(estimated_groups);
    let max_code = max_codebook_abs(codebook);
    let mut offset = 0usize;
    while offset < payload_bytes {
        let bytes = read_tensor_payload_chunk(
            &location.source_file,
            location.data_start,
            &location.header,
            offset,
            chunk_bytes,
        )?;
        if bytes.is_empty() {
            break;
        }
        collect_group_target_scales(
            &location.header.dtype,
            &bytes,
            group_size,
            max_code,
            &mut target_scales,
        )?;
        offset += bytes.len();
    }
    let Some(target_median) = lower_median(&mut target_scales) else {
        return Ok(1.0);
    };
    let mut scale_values_for_median = scale_values.to_vec();
    let Some(scale_median) = lower_median(&mut scale_values_for_median) else {
        return Ok(1.0);
    };
    if !target_median.is_finite()
        || !scale_median.is_finite()
        || target_median <= 0.0
        || scale_median <= 0.0
    {
        return Ok(1.0);
    }
    let tensor_scale = target_median / scale_median;
    if tensor_scale.is_finite() && tensor_scale > 0.0 {
        Ok(tensor_scale)
    } else {
        Ok(1.0)
    }
}

fn nearest_codebook_index(value: f32, codebook: &[f32]) -> usize {
    let mut best_index = 0usize;
    let mut best_error = f32::INFINITY;
    for (index, entry) in codebook.iter().enumerate() {
        let error = (value - *entry).abs();
        if error < best_error {
            best_error = error;
            best_index = index;
        }
    }
    best_index
}

fn choose_best_scale_index_for_group(
    dtype: &str,
    group: &[u8],
    element_size: usize,
    scale_values: &[f32],
    codebook: &[f32],
    tensor_scale: f32,
    scale_window: usize,
    max_code: f32,
) -> Result<(usize, usize), String> {
    let mut absmax = 0.0f32;
    for item in group.chunks_exact(element_size) {
        let value = decode_numeric_value(dtype, item)?;
        if !value.is_nan() {
            absmax = absmax.max(value.abs());
        }
    }
    let scale_target = absmax / tensor_scale / max_code;
    let (center_scale_index, _, _) = nearest_scale_index(scale_target, scale_values);
    let scale_start = center_scale_index.saturating_sub(scale_window);
    let scale_end = center_scale_index
        .saturating_add(scale_window)
        .min(scale_values.len() - 1);
    let mut best_scale_index = center_scale_index;
    let mut best_group_sse = f64::INFINITY;
    for scale_index in scale_start..=scale_end {
        let combined_scale = scale_values[scale_index] * tensor_scale;
        let mut group_sse = 0.0f64;
        for item in group.chunks_exact(element_size) {
            let value = decode_numeric_value(dtype, item)?;
            if value.is_nan() {
                continue;
            }
            let normalized = value / combined_scale;
            let codebook_index = nearest_codebook_index(normalized, codebook);
            let recon = codebook[codebook_index] * combined_scale;
            let error = value - recon;
            group_sse += f64::from(error * error);
        }
        if group_sse < best_group_sse {
            best_group_sse = group_sse;
            best_scale_index = scale_index;
        }
    }
    Ok((center_scale_index, best_scale_index))
}

fn new_quant_dry_run_stats(
    codebook_len: usize,
    tensor_scale: f32,
    scale_window: usize,
) -> QuantDryRunStats {
    QuantDryRunStats {
        elements: 0,
        groups: 0,
        sse: 0.0,
        ref_sse: 0.0,
        max_abs_error: 0.0,
        index_counts: vec![0; codebook_len],
        tensor_scale,
        scale_window,
        scale_index_min: usize::MAX,
        scale_index_max: 0,
        scale_window_improved_groups: 0,
    }
}

fn update_quant_dry_run_stats(
    dtype: &str,
    bytes: &[u8],
    group_size: usize,
    scale_values: &[f32],
    codebook: &[f32],
    tensor_scale: f32,
    scale_window: usize,
    stats: &mut QuantDryRunStats,
) -> Result<(), String> {
    if codebook.is_empty() {
        return Err("quant dry-run requires a non-empty codebook".to_string());
    }
    if scale_values.is_empty() {
        return Err("quant dry-run requires at least one scale value".to_string());
    }
    if !tensor_scale.is_finite() || tensor_scale <= 0.0 {
        return Err("quant dry-run requires a positive finite tensor scale".to_string());
    }
    let element_size = numeric_element_size(dtype)
        .ok_or_else(|| format!("quant dry-run is not supported for dtype {dtype}"))?;
    let group_bytes = group_size
        .checked_mul(element_size)
        .ok_or_else(|| "quant dry-run group byte size overflows".to_string())?;
    if bytes.len() % group_bytes != 0 {
        return Err(format!(
            "{dtype} quant dry-run chunk has {} bytes, not divisible by group byte size {group_bytes}",
            bytes.len()
        ));
    }
    let max_code = max_codebook_abs(codebook);
    for group in bytes.chunks_exact(group_bytes) {
        let (center_scale_index, best_scale_index) = choose_best_scale_index_for_group(
            dtype,
            group,
            element_size,
            scale_values,
            codebook,
            tensor_scale,
            scale_window,
            max_code,
        )?;
        if best_scale_index != center_scale_index {
            stats.scale_window_improved_groups += 1;
        }
        stats.scale_index_min = stats.scale_index_min.min(best_scale_index);
        stats.scale_index_max = stats.scale_index_max.max(best_scale_index);
        let combined_scale = scale_values[best_scale_index] * tensor_scale;
        for item in group.chunks_exact(element_size) {
            let value = decode_numeric_value(dtype, item)?;
            if value.is_nan() {
                continue;
            }
            let normalized = value / combined_scale;
            let codebook_index = nearest_codebook_index(normalized, codebook);
            let recon = codebook[codebook_index] * combined_scale;
            let error = value - recon;
            stats.elements += 1;
            stats.sse += f64::from(error * error);
            stats.ref_sse += f64::from(value * value);
            stats.max_abs_error = stats.max_abs_error.max(error.abs());
            stats.index_counts[codebook_index] += 1;
        }
        stats.groups += 1;
    }
    Ok(())
}

fn sanitize_file_stem(name: &str) -> String {
    let mut output = String::with_capacity(name.len());
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() {
            output.push(ch);
        } else {
            output.push('_');
        }
    }
    if output.is_empty() {
        "tensor".to_string()
    } else {
        output
    }
}

fn relative_path_string(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

fn write_f32_le_file(path: &Path, values: &[f32]) -> Result<(), String> {
    let file = fs::File::create(path)
        .map_err(|err| format!("failed to create {}: {err}", path.display()))?;
    let mut writer = BufWriter::new(file);
    for value in values {
        writer
            .write_all(&value.to_le_bytes())
            .map_err(|err| format!("failed to write {}: {err}", path.display()))?;
    }
    writer
        .flush()
        .map_err(|err| format!("failed to flush {}: {err}", path.display()))
}

fn read_f32_le_file(path: &Path) -> Result<Vec<f32>, String> {
    let bytes =
        fs::read(path).map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    if bytes.len() % 4 != 0 {
        return Err(format!(
            "{} has {} bytes, not divisible by 4",
            path.display(),
            bytes.len()
        ));
    }
    let mut values = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(values)
}

fn empty_aq_quant_metrics() -> AqQuantMetrics {
    AqQuantMetrics {
        elements: 0,
        groups: 0,
        sse: 0.0,
        ref_sse: 0.0,
        max_abs_error: 0.0,
        index_counts: [0; 16],
        scale_index_min: u32::MAX,
        scale_index_max: 0,
        scale_window_improved_groups: 0,
    }
}

fn merge_cxx_quant_metrics(stats: &mut QuantDryRunStats, metrics: &AqQuantMetrics) {
    stats.elements += metrics.elements as usize;
    stats.groups += metrics.groups as usize;
    stats.sse += metrics.sse;
    stats.ref_sse += metrics.ref_sse;
    stats.max_abs_error = stats.max_abs_error.max(metrics.max_abs_error);
    for (index, count) in metrics.index_counts.iter().enumerate() {
        if index < stats.index_counts.len() {
            stats.index_counts[index] += *count as usize;
        }
    }
    if metrics.groups > 0 {
        stats.scale_index_min = stats.scale_index_min.min(metrics.scale_index_min as usize);
        stats.scale_index_max = stats.scale_index_max.max(metrics.scale_index_max as usize);
    }
    stats.scale_window_improved_groups += metrics.scale_window_improved_groups as usize;
}

fn cxx_error_message(code: i32) -> &'static str {
    match code {
        -1 => "null pointer",
        -2 => "invalid argument",
        -3 => "invalid input byte layout",
        -4 => "output buffer is too small",
        -5 => "unsupported dtype",
        _ => "unknown error",
    }
}

fn cxx_dtype_id(dtype: &str) -> Option<u32> {
    match dtype {
        "BF16" => Some(ULLM_AQ_DTYPE_BF16),
        "F16" => Some(ULLM_AQ_DTYPE_F16),
        _ => None,
    }
}

fn quantize_chunk_to_writers<WIndex: Write, WScale: Write>(
    dtype: &str,
    bytes: &[u8],
    group_size: usize,
    scale_values: &[f32],
    codebook: &[f32],
    tensor_scale: f32,
    scale_window: usize,
    stats: &mut QuantDryRunStats,
    index_writer: &mut WIndex,
    scale_writer: &mut WScale,
) -> Result<(), String> {
    if scale_values.len() > u8::MAX as usize + 1 {
        return Err("prototype scale index writer only supports up to 256 scales".to_string());
    }
    let element_size = numeric_element_size(dtype)
        .ok_or_else(|| format!("prototype quantization is not supported for dtype {dtype}"))?;
    let group_bytes = group_size
        .checked_mul(element_size)
        .ok_or_else(|| "prototype group byte size overflows".to_string())?;
    if bytes.len() % group_bytes != 0 {
        return Err(format!(
            "{dtype} prototype chunk has {} bytes, not divisible by group byte size {group_bytes}",
            bytes.len()
        ));
    }

    let groups = bytes.len() / group_bytes;
    if let Some(dtype_id) = cxx_dtype_id(dtype) {
        if codebook.len() != 16 {
            return Err("C++ prototype kernel requires 16 codebook entries".to_string());
        }
        let elements = bytes.len() / element_size;
        let mut packed_indices = vec![0u8; elements.div_ceil(2)];
        let mut scale_indices = vec![0u8; groups];
        let mut metrics = empty_aq_quant_metrics();
        let request = AqQuantizeChunkRequestV1 {
            struct_size: std::mem::size_of::<AqQuantizeChunkRequestV1>(),
            dtype: dtype_id,
            reserved0: 0,
            input: bytes.as_ptr(),
            input_bytes: bytes.len(),
            group_size,
            scale_values: scale_values.as_ptr(),
            scale_count: scale_values.len(),
            codebook: codebook.as_ptr(),
            codebook_count: codebook.len(),
            tensor_scale,
            reserved1: 0,
            scale_window,
            packed_indices: packed_indices.as_mut_ptr(),
            packed_indices_bytes: packed_indices.len(),
            scale_indices: scale_indices.as_mut_ptr(),
            scale_indices_bytes: scale_indices.len(),
        };
        let status = unsafe {
            ullm_aq_quantize_chunk_v1(
                &request,
                &mut metrics,
                std::mem::size_of::<AqQuantMetrics>(),
            )
        };
        if status != 0 {
            return Err(format!(
                "C++ BF16 prototype kernel failed with {status}: {}",
                cxx_error_message(status)
            ));
        }
        merge_cxx_quant_metrics(stats, &metrics);
        index_writer
            .write_all(&packed_indices)
            .map_err(|err| format!("failed to write prototype index bytes: {err}"))?;
        scale_writer
            .write_all(&scale_indices)
            .map_err(|err| format!("failed to write prototype scale bytes: {err}"))?;
        return Ok(());
    }

    let mut packed_indices = Vec::with_capacity(bytes.len() / (2 * element_size));
    let mut scale_indices = Vec::with_capacity(groups);
    let max_code = max_codebook_abs(codebook);

    for group in bytes.chunks_exact(group_bytes) {
        let (center_scale_index, best_scale_index) = choose_best_scale_index_for_group(
            dtype,
            group,
            element_size,
            scale_values,
            codebook,
            tensor_scale,
            scale_window,
            max_code,
        )?;
        if best_scale_index != center_scale_index {
            stats.scale_window_improved_groups += 1;
        }
        stats.scale_index_min = stats.scale_index_min.min(best_scale_index);
        stats.scale_index_max = stats.scale_index_max.max(best_scale_index);
        scale_indices.push(best_scale_index as u8);

        let combined_scale = scale_values[best_scale_index] * tensor_scale;
        let mut pending_low: Option<u8> = None;
        for item in group.chunks_exact(element_size) {
            let value = decode_numeric_value(dtype, item)?;
            if value.is_nan() {
                let nibble = 0u8;
                if let Some(low) = pending_low.take() {
                    packed_indices.push(low | (nibble << 4));
                } else {
                    pending_low = Some(nibble);
                }
                continue;
            }
            let normalized = value / combined_scale;
            let codebook_index = nearest_codebook_index(normalized, codebook);
            let recon = codebook[codebook_index] * combined_scale;
            let error = value - recon;
            stats.elements += 1;
            stats.sse += f64::from(error * error);
            stats.ref_sse += f64::from(value * value);
            stats.max_abs_error = stats.max_abs_error.max(error.abs());
            stats.index_counts[codebook_index] += 1;

            let nibble = codebook_index as u8 & 0x0f;
            if let Some(low) = pending_low.take() {
                packed_indices.push(low | (nibble << 4));
            } else {
                pending_low = Some(nibble);
            }
        }
        if let Some(low) = pending_low {
            packed_indices.push(low);
        }
        stats.groups += 1;
    }

    index_writer
        .write_all(&packed_indices)
        .map_err(|err| format!("failed to write prototype index bytes: {err}"))?;
    scale_writer
        .write_all(&scale_indices)
        .map_err(|err| format!("failed to write prototype scale bytes: {err}"))?;
    Ok(())
}

fn write_prototype_tensor(
    model_dir: &Path,
    tensor_name: &str,
    aq_format: &str,
    family: &str,
    candidate_id: &str,
    codebook: &[f32],
    chunk_bytes: usize,
    scale_window: usize,
    tensor_scale_override: Option<f32>,
    output_dir: &Path,
) -> Result<PrototypeManifest, String> {
    let location = find_tensor_location(model_dir, tensor_name)?;
    let payload_bytes = tensor_payload_bytes(&location.header)?;
    let element_size = numeric_element_size(&location.header.dtype).ok_or_else(|| {
        format!(
            "prototype output is not supported for dtype {}",
            location.header.dtype
        )
    })?;
    let group_size = aq_group_size(aq_format)?;
    let group_bytes = group_size
        .checked_mul(element_size)
        .ok_or_else(|| "prototype group byte size overflows".to_string())?;
    if chunk_bytes % group_bytes != 0 {
        return Err(format!(
            "--chunk-bytes must be divisible by {group_bytes} for prototype output"
        ));
    }
    if payload_bytes % group_bytes != 0 {
        return Err(format!(
            "tensor payload bytes {payload_bytes} are not divisible by group byte size {group_bytes}"
        ));
    }

    let group_stats = new_aq_group_stats(aq_format, group_size)?;
    let tensor_scale = if let Some(value) = tensor_scale_override {
        value
    } else if aq_uses_tensor_scale(aq_format) {
        estimate_tensor_scale(
            &location,
            payload_bytes,
            chunk_bytes,
            group_size,
            &group_stats.scale_values,
            codebook,
        )?
    } else {
        1.0
    };

    let tensor_stem = sanitize_file_stem(tensor_name);
    let codebook_stem = sanitize_file_stem(&format!("{family}__{candidate_id}"));
    let tensors_dir = output_dir.join("tensors");
    let codebooks_dir = output_dir.join("codebooks");
    fs::create_dir_all(&tensors_dir)
        .map_err(|err| format!("failed to create {}: {err}", tensors_dir.display()))?;
    fs::create_dir_all(&codebooks_dir)
        .map_err(|err| format!("failed to create {}: {err}", codebooks_dir.display()))?;

    let index_rel = PathBuf::from("tensors").join(format!("{tensor_stem}.idx4"));
    let scale_rel = PathBuf::from("tensors").join(format!("{tensor_stem}.scale_u8"));
    let codebook_rel = PathBuf::from("codebooks").join(format!("{codebook_stem}.f32"));
    let index_path = output_dir.join(&index_rel);
    let scale_path = output_dir.join(&scale_rel);
    let codebook_path = output_dir.join(&codebook_rel);
    write_f32_le_file(&codebook_path, codebook)?;

    let index_file = fs::File::create(&index_path)
        .map_err(|err| format!("failed to create {}: {err}", index_path.display()))?;
    let scale_file = fs::File::create(&scale_path)
        .map_err(|err| format!("failed to create {}: {err}", scale_path.display()))?;
    let mut index_writer = BufWriter::new(index_file);
    let mut scale_writer = BufWriter::new(scale_file);
    let mut stats = new_quant_dry_run_stats(codebook.len(), tensor_scale, scale_window);

    let mut offset = 0usize;
    while offset < payload_bytes {
        let bytes = read_tensor_payload_chunk(
            &location.source_file,
            location.data_start,
            &location.header,
            offset,
            chunk_bytes,
        )?;
        if bytes.is_empty() {
            break;
        }
        quantize_chunk_to_writers(
            &location.header.dtype,
            &bytes,
            group_size,
            &group_stats.scale_values,
            codebook,
            tensor_scale,
            scale_window,
            &mut stats,
            &mut index_writer,
            &mut scale_writer,
        )?;
        offset += bytes.len();
    }
    index_writer
        .flush()
        .map_err(|err| format!("failed to flush {}: {err}", index_path.display()))?;
    scale_writer
        .flush()
        .map_err(|err| format!("failed to flush {}: {err}", scale_path.display()))?;

    let elements = payload_bytes / element_size;
    let manifest = PrototypeManifest {
        schema_version: "ullm-prototype-manifest-v0.1".to_string(),
        source_model_dir: model_dir.display().to_string(),
        tensors: vec![PrototypeTensorManifest {
            name: tensor_name.to_string(),
            source_file: location.source_file.display().to_string(),
            dtype: location.header.dtype,
            shape: location.header.shape,
            family: family.to_string(),
            candidate_id: candidate_id.to_string(),
            scale_format: group_stats.scale_format,
            group_size,
            tensor_scale,
            scale_window,
            elements,
            groups: stats.groups,
            index_file: relative_path_string(&index_rel),
            index_encoding: "idx4_low_nibble_first".to_string(),
            scale_file: relative_path_string(&scale_rel),
            scale_encoding: "u8_scale_table_index".to_string(),
            codebook_file: relative_path_string(&codebook_rel),
            metrics: PrototypeTensorMetrics {
                mse: if stats.elements > 0 {
                    stats.sse / stats.elements as f64
                } else {
                    0.0
                },
                relative_mse: if stats.ref_sse > 0.0 {
                    stats.sse / stats.ref_sse
                } else {
                    0.0
                },
                max_abs_error: stats.max_abs_error,
                scale_index_min: stats.scale_index_min,
                scale_index_max: stats.scale_index_max,
                scale_window_improved_groups: stats.scale_window_improved_groups,
                index_counts: stats.index_counts,
            },
        }],
        codebooks: vec![PrototypeCodebookManifest {
            family: family.to_string(),
            candidate_id: candidate_id.to_string(),
            file: relative_path_string(&codebook_rel),
            encoding: "f32_le".to_string(),
            entries: codebook.len(),
        }],
    };
    let manifest_path = output_dir.join("manifest.json");
    let manifest_text = serde_json::to_string_pretty(&manifest)
        .map_err(|err| format!("failed to serialize prototype manifest: {err}"))?;
    fs::write(&manifest_path, manifest_text + "\n")
        .map_err(|err| format!("failed to write {}: {err}", manifest_path.display()))?;
    Ok(manifest)
}

fn verify_prototype_tensor(
    output_dir: &Path,
    tensor_index: usize,
    chunk_bytes: usize,
) -> Result<PrototypeVerifyResult, String> {
    let manifest_path = output_dir.join("manifest.json");
    let manifest_text = fs::read_to_string(&manifest_path)
        .map_err(|err| format!("failed to read {}: {err}", manifest_path.display()))?;
    let manifest: PrototypeManifest = serde_json::from_str(&manifest_text)
        .map_err(|err| format!("failed to parse {}: {err}", manifest_path.display()))?;
    let tensor = manifest
        .tensors
        .get(tensor_index)
        .ok_or_else(|| format!("prototype tensor index {tensor_index} is out of range"))?;
    let source_file = PathBuf::from(&tensor.source_file);
    let metadata = read_safetensors_metadata(&source_file)?;
    let header = metadata.tensors.get(&tensor.name).ok_or_else(|| {
        format!(
            "tensor {} not found in {}",
            tensor.name,
            source_file.display()
        )
    })?;
    let payload_bytes = tensor_payload_bytes(header)?;
    let element_size = numeric_element_size(&header.dtype).ok_or_else(|| {
        format!(
            "prototype verify is not supported for dtype {}",
            header.dtype
        )
    })?;
    let group_bytes = tensor
        .group_size
        .checked_mul(element_size)
        .ok_or_else(|| "prototype verify group byte size overflows".to_string())?;
    if chunk_bytes % group_bytes != 0 {
        return Err(format!(
            "--chunk-bytes must be divisible by {group_bytes} for prototype verify"
        ));
    }

    let index_path = output_dir.join(&tensor.index_file);
    let scale_path = output_dir.join(&tensor.scale_file);
    let codebook_path = output_dir.join(&tensor.codebook_file);
    let index_bytes = fs::read(&index_path)
        .map_err(|err| format!("failed to read {}: {err}", index_path.display()))?;
    let scale_indices = fs::read(&scale_path)
        .map_err(|err| format!("failed to read {}: {err}", scale_path.display()))?;
    let codebook = read_f32_le_file(&codebook_path)?;
    let scale_values = scale_values(&tensor.scale_format)?;

    let expected_elements = payload_bytes / element_size;
    let expected_groups = expected_elements / tensor.group_size;
    let expected_index_bytes = expected_elements.div_ceil(2);
    if index_bytes.len() != expected_index_bytes {
        return Err(format!(
            "{} has {} bytes, expected {expected_index_bytes}",
            index_path.display(),
            index_bytes.len()
        ));
    }
    if scale_indices.len() != expected_groups {
        return Err(format!(
            "{} has {} bytes, expected {expected_groups}",
            scale_path.display(),
            scale_indices.len()
        ));
    }

    let mut offset = 0usize;
    let mut element_cursor = 0usize;
    let mut group_cursor = 0usize;
    let mut elements = 0usize;
    let mut sse = 0.0f64;
    let mut ref_sse = 0.0f64;
    let mut max_abs_error = 0.0f32;
    while offset < payload_bytes {
        let bytes = read_tensor_payload_chunk(
            &source_file,
            metadata.data_start,
            header,
            offset,
            chunk_bytes,
        )?;
        if bytes.is_empty() {
            break;
        }
        for group in bytes.chunks_exact(group_bytes) {
            let scale_index = usize::from(scale_indices[group_cursor]);
            let scale = *scale_values.get(scale_index).ok_or_else(|| {
                format!("scale index {scale_index} at group {group_cursor} is out of range")
            })?;
            let combined_scale = scale * tensor.tensor_scale;
            for item in group.chunks_exact(element_size) {
                let packed = index_bytes[element_cursor / 2];
                let codebook_index = if element_cursor % 2 == 0 {
                    packed & 0x0f
                } else {
                    (packed >> 4) & 0x0f
                } as usize;
                let code = *codebook.get(codebook_index).ok_or_else(|| {
                    format!(
                        "codebook index {codebook_index} at element {element_cursor} is out of range"
                    )
                })?;
                let value = decode_numeric_value(&header.dtype, item)?;
                element_cursor += 1;
                if value.is_nan() {
                    continue;
                }
                let recon = code * combined_scale;
                let error = value - recon;
                elements += 1;
                sse += f64::from(error * error);
                ref_sse += f64::from(value * value);
                max_abs_error = max_abs_error.max(error.abs());
            }
            group_cursor += 1;
        }
        offset += bytes.len();
    }
    Ok(PrototypeVerifyResult {
        elements,
        groups: group_cursor,
        mse: if elements > 0 {
            sse / elements as f64
        } else {
            0.0
        },
        relative_mse: if ref_sse > 0.0 { sse / ref_sse } else { 0.0 },
        max_abs_error,
        index_file_bytes: index_bytes.len(),
        scale_file_bytes: scale_indices.len(),
        codebook_entries: codebook.len(),
    })
}

fn inspect_tensor_chunks(
    model_dir: &Path,
    tensor_name: &str,
    chunk_bytes: usize,
    aq_format: Option<&str>,
    codebook: Option<&[f32]>,
    scale_window: usize,
) -> Result<TensorInspectResult, String> {
    let location = find_tensor_location(model_dir, tensor_name)?;
    let payload_bytes = tensor_payload_bytes(&location.header)?;
    let mut numeric_stats = if let Some(element_size) = numeric_element_size(&location.header.dtype)
    {
        if chunk_bytes % element_size != 0 {
            return Err(format!(
                "--chunk-bytes must be divisible by {element_size} for {} stats",
                location.header.dtype
            ));
        }
        Some(new_numeric_stats())
    } else {
        None
    };
    let mut aq_group_stats = if let Some(format) = aq_format {
        let element_size = numeric_element_size(&location.header.dtype).ok_or_else(|| {
            format!(
                "aq group stats are not supported for dtype {}",
                location.header.dtype
            )
        })?;
        let group_size = aq_group_size(format)?;
        let group_bytes = group_size
            .checked_mul(element_size)
            .ok_or_else(|| "aq group byte size overflows".to_string())?;
        if chunk_bytes % group_bytes != 0 {
            return Err(format!(
                "--chunk-bytes must be divisible by {group_bytes} for {format} group stats"
            ));
        }
        if payload_bytes % group_bytes != 0 {
            return Err(format!(
                "tensor payload bytes {payload_bytes} are not divisible by group byte size {group_bytes}"
            ));
        }
        Some(new_aq_group_stats(format, group_size)?)
    } else {
        None
    };
    if codebook.is_some() && aq_group_stats.is_none() {
        return Err(
            "--inspect-aq-format is required for quant dry-run with a codebook".to_string(),
        );
    }
    let quant_tensor_scale = if let (Some(format), Some(group_stats), Some(codebook_values)) =
        (aq_format, aq_group_stats.as_ref(), codebook)
    {
        if aq_uses_tensor_scale(format) {
            estimate_tensor_scale(
                &location,
                payload_bytes,
                chunk_bytes,
                group_stats.group_size,
                &group_stats.scale_values,
                codebook_values,
            )?
        } else {
            1.0
        }
    } else {
        1.0
    };
    let mut quant_dry_run_stats = codebook
        .map(|values| new_quant_dry_run_stats(values.len(), quant_tensor_scale, scale_window));
    let mut offset = 0usize;
    let mut chunks = 0usize;
    let mut hash = 0xcbf2_9ce4_8422_2325;
    while offset < payload_bytes {
        let bytes = read_tensor_payload_chunk(
            &location.source_file,
            location.data_start,
            &location.header,
            offset,
            chunk_bytes,
        )?;
        if bytes.is_empty() {
            break;
        }
        hash = fnv1a64_update(hash, &bytes);
        if let Some(stats) = numeric_stats.as_mut() {
            update_numeric_stats(&location.header.dtype, &bytes, stats)?;
        }
        if let Some(stats) = aq_group_stats.as_mut() {
            update_aq_group_stats(&location.header.dtype, &bytes, stats)?;
        }
        if let (Some(group_stats), Some(codebook_values), Some(stats)) = (
            aq_group_stats.as_ref(),
            codebook,
            quant_dry_run_stats.as_mut(),
        ) {
            update_quant_dry_run_stats(
                &location.header.dtype,
                &bytes,
                group_stats.group_size,
                &group_stats.scale_values,
                codebook_values,
                quant_tensor_scale,
                scale_window,
                stats,
            )?;
        }
        offset += bytes.len();
        chunks += 1;
    }
    Ok(TensorInspectResult {
        name: tensor_name.to_string(),
        source_file: location.source_file,
        dtype: location.header.dtype,
        shape: location.header.shape,
        payload_bytes,
        chunk_bytes,
        chunks,
        fnv1a64: hash,
        numeric_stats,
        aq_group_stats,
        quant_dry_run_stats,
    })
}

fn safetensor_files(model_dir: &Path) -> Result<Vec<PathBuf>, String> {
    let index_path = model_dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let text = fs::read_to_string(&index_path)
            .map_err(|err| format!("failed to read {}: {err}", index_path.display()))?;
        let index: SafetensorsIndex = serde_json::from_str(&text)
            .map_err(|err| format!("failed to parse {}: {err}", index_path.display()))?;
        let mut files = Vec::new();
        for filename in index.weight_map.values() {
            let path = model_dir.join(filename);
            if !files.contains(&path) {
                files.push(path);
            }
        }
        return Ok(files);
    }

    let mut files = Vec::new();
    for entry in fs::read_dir(model_dir)
        .map_err(|err| format!("failed to read model dir {}: {err}", model_dir.display()))?
    {
        let path = entry
            .map_err(|err| format!("failed to read dir entry in {}: {err}", model_dir.display()))?
            .path();
        if path.extension().is_some_and(|ext| ext == "safetensors") {
            files.push(path);
        }
    }
    files.sort();
    if files.is_empty() {
        return Err(format!(
            "no safetensors files found in {}",
            model_dir.display()
        ));
    }
    Ok(files)
}

fn build_model_plan(model_dir: &Path, aq_policy: &AqPolicyPlan) -> Result<ModelPlan, String> {
    let mut tensors = Vec::new();
    for path in safetensor_files(model_dir)? {
        let headers = read_safetensors_header(&path)?;
        for (name, header) in headers {
            let n_elements = tensor_elements(&header.shape)?;
            let n_bytes =
                tensor_payload_bytes(&header).map_err(|err| format!("{err} for tensor {name}"))?;
            let family = family_for_tensor(&name).to_string();
            let supported_input = is_supported_input(&header.dtype, &header.shape, &family);
            let (quant_format, quant_role) = quant_assignment(supported_input, &family, aq_policy);
            let estimated_output_bytes =
                estimate_output_bytes(n_elements, n_bytes, quant_format.as_deref())?;
            let estimated_effective_bpp = effective_bpp(n_elements, estimated_output_bytes);
            tensors.push(TensorPlan {
                family,
                action: if supported_input {
                    "quantize".to_string()
                } else {
                    "passthrough".to_string()
                },
                quant_format,
                quant_role,
                estimated_output_bytes,
                estimated_effective_bpp,
                name,
                source_file: path.display().to_string(),
                dtype: header.dtype,
                shape: header.shape,
                n_elements,
                n_bytes,
                supported_input,
            });
        }
    }
    tensors.sort_by(|left, right| left.name.cmp(&right.name));
    let supported_tensor_count = tensors
        .iter()
        .filter(|tensor| tensor.supported_input)
        .count();
    let total_tensor_bytes = tensors.iter().map(|tensor| tensor.n_bytes).sum();
    let total_estimated_output_bytes = tensors
        .iter()
        .map(|tensor| tensor.estimated_output_bytes)
        .sum();
    let estimated_output_to_input_ratio = if total_tensor_bytes == 0 {
        0.0
    } else {
        total_estimated_output_bytes as f64 / total_tensor_bytes as f64
    };
    Ok(ModelPlan {
        schema_version: "ullm-quant-plan-v0.3".to_string(),
        model_dir: model_dir.display().to_string(),
        aq_policy: aq_policy.clone(),
        tensor_count: tensors.len(),
        supported_tensor_count,
        passthrough_tensor_count: tensors.len() - supported_tensor_count,
        total_tensor_bytes,
        total_estimated_output_bytes,
        estimated_output_to_input_ratio,
        tensors,
    })
}

fn run_pack_smoke() -> Result<Vec<u8>, String> {
    let low = [0x00, 0x01, 0x0f, 0x08];
    let high = [0x01, 0x02, 0x00, 0x07];
    let mut output = [0u8; 4];
    let written = unsafe {
        ullm_aq_pack_nibbles(
            low.as_ptr(),
            high.as_ptr(),
            output.as_mut_ptr(),
            output.len(),
        )
    };
    if written != output.len() {
        return Err(format!(
            "pack smoke wrote {written}, expected {}",
            output.len()
        ));
    }
    let expected = [0x10, 0x21, 0x0f, 0x78];
    if output != expected {
        return Err(format!(
            "pack smoke output mismatch: {output:?} != {expected:?}"
        ));
    }
    Ok(output.to_vec())
}

fn run() -> Result<(), String> {
    let options = parse_options()?;
    let aq_policy = resolve_aq_policy(&options)?;
    let version = unsafe { ullm_aq_get_kernel_version() };
    let packed = run_pack_smoke()?;
    let plan = match options.model_dir.as_deref() {
        Some(model_dir) => Some(build_model_plan(model_dir, &aq_policy)?),
        None => None,
    };
    let selected_codebook = if let (Some(path), Some(family), Some(candidate_id)) = (
        options.codebook_json.as_deref(),
        options.inspect_codebook_family.as_deref(),
        options.inspect_codebook_candidate.as_deref(),
    ) {
        let export = load_codebook_export(path)?;
        Some(select_codebook(&export, family, candidate_id)?.to_vec())
    } else {
        None
    };

    println!("ullm-quant skeleton");
    println!(
        "kernel_version={}.{}.{}",
        version.major, version.minor, version.patch
    );
    println!("threads={}", options.threads);
    println!("io_threads={}", options.io_threads);
    println!("max_working_memory_mib={}", options.max_working_memory_mib);
    println!("aq_policy={}", aq_policy.policy_id);
    println!("aq_low_format={}", aq_policy.low_format);
    println!("aq_high_format={}", aq_policy.high_format);
    println!("aq_high_families={}", aq_policy.high_families.join(","));
    println!("dry_run={}", options.dry_run);
    println!("pack_smoke=ok {packed:?}");
    if let Some(plan) = &plan {
        println!("plan_model_dir={}", plan.model_dir);
        println!("plan_tensor_count={}", plan.tensor_count);
        println!(
            "plan_supported_tensor_count={}",
            plan.supported_tensor_count
        );
        println!(
            "plan_passthrough_tensor_count={}",
            plan.passthrough_tensor_count
        );
        println!("plan_total_tensor_bytes={}", plan.total_tensor_bytes);
        println!(
            "plan_total_estimated_output_bytes={}",
            plan.total_estimated_output_bytes
        );
        println!(
            "plan_estimated_output_to_input_ratio={:.6}",
            plan.estimated_output_to_input_ratio
        );
    }
    if let Some(tensor_name) = options
        .inspect_tensor
        .as_deref()
        .filter(|_| !options.skip_inspect)
    {
        let model_dir = options
            .model_dir
            .as_deref()
            .ok_or_else(|| "--inspect-tensor requires --model-dir".to_string())?;
        let inspect = inspect_tensor_chunks(
            model_dir,
            tensor_name,
            options.chunk_bytes,
            options.inspect_aq_format.as_deref(),
            selected_codebook.as_deref(),
            options.scale_window,
        )?;
        println!("inspect_tensor={}", inspect.name);
        println!("inspect_source_file={}", inspect.source_file.display());
        println!("inspect_dtype={}", inspect.dtype);
        println!("inspect_shape={:?}", inspect.shape);
        println!("inspect_payload_bytes={}", inspect.payload_bytes);
        println!("inspect_chunk_bytes={}", inspect.chunk_bytes);
        println!("inspect_chunks={}", inspect.chunks);
        println!("inspect_fnv1a64={:016x}", inspect.fnv1a64);
        if let Some(stats) = &inspect.numeric_stats {
            println!("inspect_numeric_elements={}", stats.elements);
            println!("inspect_numeric_finite_elements={}", stats.finite_elements);
            println!("inspect_numeric_nan_elements={}", stats.nan_elements);
            if stats.finite_elements > 0 {
                println!("inspect_numeric_min={:.9}", stats.min);
                println!("inspect_numeric_max={:.9}", stats.max);
                println!(
                    "inspect_numeric_mean_abs={:.9}",
                    stats.sum_abs / stats.finite_elements as f64
                );
                println!("inspect_numeric_max_abs={:.9}", stats.max_abs);
            }
        }
        if let Some(stats) = &inspect.aq_group_stats {
            println!("inspect_aq_format={}", stats.format);
            println!("inspect_aq_scale_format={}", stats.scale_format);
            println!("inspect_aq_scale_count={}", stats.scale_values.len());
            println!("inspect_aq_group_size={}", stats.group_size);
            println!("inspect_aq_groups={}", stats.groups);
            if stats.groups > 0 {
                println!(
                    "inspect_aq_group_absmax_mean={:.9}",
                    stats.sum_absmax / stats.groups as f64
                );
                println!("inspect_aq_group_absmax_max={:.9}", stats.max_absmax);
                println!("inspect_aq_zero_absmax_groups={}", stats.zero_absmax_groups);
                if stats.groups > stats.zero_absmax_groups {
                    println!("inspect_aq_scale_index_min={}", stats.scale_index_min);
                    println!("inspect_aq_scale_index_max={}", stats.scale_index_max);
                    println!("inspect_aq_scale_clamped_low={}", stats.scale_clamped_low);
                    println!("inspect_aq_scale_clamped_high={}", stats.scale_clamped_high);
                    println!(
                        "inspect_aq_scale_relative_error_mean={:.9}",
                        stats.sum_scale_relative_error
                            / (stats.groups - stats.zero_absmax_groups) as f64
                    );
                }
            }
        }
        if let Some(stats) = &inspect.quant_dry_run_stats {
            println!("inspect_quant_dry_run=scale_window_nearest_codebook");
            println!("inspect_quant_tensor_scale={:.12}", stats.tensor_scale);
            println!("inspect_quant_scale_window={}", stats.scale_window);
            println!("inspect_quant_elements={}", stats.elements);
            println!("inspect_quant_groups={}", stats.groups);
            if stats.elements > 0 {
                println!(
                    "inspect_quant_mse={:.12}",
                    stats.sse / stats.elements as f64
                );
            }
            if stats.ref_sse > 0.0 {
                println!(
                    "inspect_quant_relative_mse={:.12}",
                    stats.sse / stats.ref_sse
                );
            }
            println!("inspect_quant_max_abs_error={:.9}", stats.max_abs_error);
            if stats.groups > 0 {
                println!("inspect_quant_scale_index_min={}", stats.scale_index_min);
                println!("inspect_quant_scale_index_max={}", stats.scale_index_max);
                println!(
                    "inspect_quant_scale_window_improved_groups={}",
                    stats.scale_window_improved_groups
                );
            }
            println!(
                "inspect_quant_index_counts={}",
                stats
                    .index_counts
                    .iter()
                    .map(|count| count.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            );
        }
    }
    if options.codebook_json.is_some()
        || options.inspect_codebook_family.is_some()
        || options.inspect_codebook_candidate.is_some()
    {
        let path = options
            .codebook_json
            .as_deref()
            .ok_or_else(|| "--codebook-json is required for codebook inspection".to_string())?;
        let family = options
            .inspect_codebook_family
            .as_deref()
            .ok_or_else(|| "--inspect-codebook-family is required".to_string())?;
        let candidate_id = options
            .inspect_codebook_candidate
            .as_deref()
            .ok_or_else(|| "--inspect-codebook-candidate is required".to_string())?;
        let codebook = selected_codebook
            .as_deref()
            .ok_or_else(|| "selected codebook was not loaded".to_string())?;
        println!("inspect_codebook_json={}", path.display());
        println!("inspect_codebook_family={family}");
        println!("inspect_codebook_candidate={candidate_id}");
        println!("inspect_codebook_entries={}", codebook.len());
        if let (Some(min), Some(max)) = (
            codebook.iter().copied().reduce(f32::min),
            codebook.iter().copied().reduce(f32::max),
        ) {
            println!("inspect_codebook_min={min:.9}");
            println!("inspect_codebook_max={max:.9}");
        }
        let values = codebook
            .iter()
            .map(|value| format!("{value:.9}"))
            .collect::<Vec<_>>()
            .join(",");
        println!("inspect_codebook_values={values}");
    }
    if let Some(output_dir) = options.prototype_output_dir.as_deref() {
        let model_dir = options
            .model_dir
            .as_deref()
            .ok_or_else(|| "--prototype-output-dir requires --model-dir".to_string())?;
        let tensor_name = options
            .inspect_tensor
            .as_deref()
            .ok_or_else(|| "--prototype-output-dir requires --inspect-tensor".to_string())?;
        let aq_format = options
            .inspect_aq_format
            .as_deref()
            .ok_or_else(|| "--prototype-output-dir requires --inspect-aq-format".to_string())?;
        let family = options.inspect_codebook_family.as_deref().ok_or_else(|| {
            "--prototype-output-dir requires --inspect-codebook-family".to_string()
        })?;
        let candidate_id = options
            .inspect_codebook_candidate
            .as_deref()
            .ok_or_else(|| {
                "--prototype-output-dir requires --inspect-codebook-candidate".to_string()
            })?;
        let codebook = selected_codebook
            .as_deref()
            .ok_or_else(|| "--prototype-output-dir requires a selected codebook".to_string())?;
        let manifest = write_prototype_tensor(
            model_dir,
            tensor_name,
            aq_format,
            family,
            candidate_id,
            codebook,
            options.chunk_bytes,
            options.scale_window,
            options.tensor_scale_override,
            output_dir,
        )?;
        let tensor = manifest
            .tensors
            .first()
            .ok_or_else(|| "prototype manifest has no tensors".to_string())?;
        println!("prototype_output_dir={}", output_dir.display());
        println!(
            "prototype_manifest={}",
            output_dir.join("manifest.json").display()
        );
        println!("prototype_tensor={}", tensor.name);
        println!("prototype_index_file={}", tensor.index_file);
        println!("prototype_scale_file={}", tensor.scale_file);
        println!("prototype_codebook_file={}", tensor.codebook_file);
        println!("prototype_relative_mse={:.12}", tensor.metrics.relative_mse);
        println!(
            "prototype_tensor_scale_source={}",
            if options.tensor_scale_override.is_some() {
                "override"
            } else {
                "estimated"
            }
        );
        println!(
            "prototype_max_abs_error={:.9}",
            tensor.metrics.max_abs_error
        );
        if options.prototype_verify {
            let verification = verify_prototype_tensor(output_dir, 0, options.chunk_bytes)?;
            let relative_mse_delta =
                (verification.relative_mse - tensor.metrics.relative_mse).abs();
            if relative_mse_delta > 1e-9 {
                return Err(format!(
                    "prototype verification relative MSE delta {relative_mse_delta:.12} exceeds tolerance"
                ));
            }
            println!("prototype_verify_elements={}", verification.elements);
            println!("prototype_verify_groups={}", verification.groups);
            println!("prototype_verify_mse={:.12}", verification.mse);
            println!(
                "prototype_verify_relative_mse={:.12}",
                verification.relative_mse
            );
            println!(
                "prototype_verify_max_abs_error={:.9}",
                verification.max_abs_error
            );
            println!(
                "prototype_verify_index_file_bytes={}",
                verification.index_file_bytes
            );
            println!(
                "prototype_verify_scale_file_bytes={}",
                verification.scale_file_bytes
            );
            println!(
                "prototype_verify_codebook_entries={}",
                verification.codebook_entries
            );
        } else {
            println!("prototype_verify=skipped");
        }
    }
    if let (Some(plan), Some(output)) = (&plan, options.plan_output.as_deref()) {
        let text = serde_json::to_string_pretty(plan)
            .map_err(|err| format!("failed to serialize plan: {err}"))?;
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent)
                .map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
        }
        fs::write(output, text)
            .map_err(|err| format!("failed to write {}: {err}", output.display()))?;
        println!("plan_output={}", output.display());
    }

    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AqQuantMetrics, AqQuantizeChunkRequestV1, CodebookEntry, CodebookExport, Options,
        ULLM_AQ_DTYPE_BF16, ULLM_AQ_DTYPE_F16, default_threads, effective_bpp,
        empty_aq_quant_metrics, estimate_output_bytes, family_for_tensor, new_aq_group_stats,
        new_numeric_stats, new_quant_dry_run_stats, quant_assignment, read_safetensors_metadata,
        read_tensor_payload_chunk, resolve_aq_policy, select_codebook, ullm_aq_quantize_chunk_v1,
        update_aq_group_stats, update_numeric_stats, update_quant_dry_run_stats,
    };
    use std::fs;
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn default_thread_count_is_nonzero() {
        assert!(default_threads() >= 1);
    }

    #[test]
    fn qwen35_family_names_are_detected() {
        assert_eq!(
            family_for_tensor("model.language_model.layers.0.linear_attn.out_proj.weight"),
            "linear_attn_out"
        );
        assert_eq!(
            family_for_tensor("model.language_model.layers.0.mlp.down_proj.weight"),
            "mlp_down"
        );
    }

    fn test_options(policy: &str) -> Options {
        Options {
            threads: 1,
            io_threads: 1,
            max_working_memory_mib: 1024,
            model_dir: None,
            plan_output: None,
            inspect_tensor: None,
            skip_inspect: false,
            inspect_aq_format: None,
            codebook_json: None,
            inspect_codebook_family: None,
            inspect_codebook_candidate: None,
            prototype_output_dir: None,
            prototype_verify: true,
            tensor_scale_override: None,
            chunk_bytes: 1024,
            scale_window: 0,
            aq_policy: policy.to_string(),
            aq_high_families: Vec::new(),
            aq_low_format: "aq4_e4m3_g16_ts_flloyd16".to_string(),
            aq_high_format: "aq4_e4m3_g8_ts_flloyd16".to_string(),
            dry_run: true,
        }
    }

    #[test]
    fn p4p6_assigns_attention_to_high_and_mlp_up_to_low() {
        let policy = resolve_aq_policy(&test_options("p4p6")).expect("p4p6 policy");
        let (format, role) = quant_assignment(true, "attn_k", &policy);
        assert_eq!(format.as_deref(), Some("aq4_e4m3_g8_ts_flloyd16"));
        assert_eq!(role.as_deref(), Some("high"));

        let (format, role) = quant_assignment(true, "mlp_up", &policy);
        assert_eq!(format.as_deref(), Some("aq4_e4m3_g16_ts_flloyd16"));
        assert_eq!(role.as_deref(), Some("low"));
    }

    #[test]
    fn aq_output_byte_estimate_matches_group_size() {
        let g16_bytes =
            estimate_output_bytes(32, 64, Some("aq4_e4m3_g16_ts_flloyd16")).expect("g16");
        assert_eq!(g16_bytes, 18);
        assert_eq!(effective_bpp(32, g16_bytes), 4.5);

        let g8_bytes = estimate_output_bytes(32, 64, Some("aq4_e4m3_g8_ts_flloyd16")).expect("g8");
        assert_eq!(g8_bytes, 20);
        assert_eq!(effective_bpp(32, g8_bytes), 5.0);
    }

    #[test]
    fn safetensors_payload_chunk_reader_uses_data_offsets() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("ullm-quant-test-{unique}.safetensors"));
        let header = br#"{"x":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}"#;
        let mut file = fs::File::create(&path).expect("create temp safetensors");
        file.write_all(&(header.len() as u64).to_le_bytes())
            .expect("write header len");
        file.write_all(header).expect("write header");
        file.write_all(&[1, 2, 3, 4]).expect("write payload");
        drop(file);

        let metadata = read_safetensors_metadata(&path).expect("metadata");
        let tensor = metadata.tensors.get("x").expect("tensor x");
        let chunk = read_tensor_payload_chunk(&path, metadata.data_start, tensor, 1, 2)
            .expect("payload chunk");
        assert_eq!(chunk, vec![2, 3]);

        fs::remove_file(path).expect("remove temp safetensors");
    }

    #[test]
    fn bf16_numeric_stats_are_decoded_from_little_endian_payload() {
        let mut stats = new_numeric_stats();
        let payload = [
            0x80, 0x3f, // 1.0
            0x00, 0xc0, // -2.0
            0x40, 0x40, // 3.0
        ];
        update_numeric_stats("BF16", &payload, &mut stats).expect("bf16 stats");
        assert_eq!(stats.elements, 3);
        assert_eq!(stats.finite_elements, 3);
        assert_eq!(stats.nan_elements, 0);
        assert_eq!(stats.min, -2.0);
        assert_eq!(stats.max, 3.0);
        assert_eq!(stats.max_abs, 3.0);
        assert_eq!(stats.sum_abs, 6.0);
    }

    #[test]
    fn f16_numeric_stats_are_decoded_from_little_endian_payload() {
        let mut stats = new_numeric_stats();
        let payload = [
            0x00, 0x3c, // 1.0
            0x00, 0xc0, // -2.0
            0x00, 0x42, // 3.0
        ];
        update_numeric_stats("F16", &payload, &mut stats).expect("f16 stats");
        assert_eq!(stats.elements, 3);
        assert_eq!(stats.finite_elements, 3);
        assert_eq!(stats.nan_elements, 0);
        assert_eq!(stats.min, -2.0);
        assert_eq!(stats.max, 3.0);
        assert_eq!(stats.max_abs, 3.0);
        assert_eq!(stats.sum_abs, 6.0);
    }

    #[test]
    fn aq_group_stats_track_absmax_per_group() {
        let mut stats = new_aq_group_stats("aq4_e4m3_g2_test", 2).expect("group stats");
        let payload = [
            0x80, 0x3f, // 1.0
            0x00, 0xc0, // -2.0
            0x40, 0x40, // 3.0
            0x00, 0x3f, // 0.5
        ];
        update_aq_group_stats("BF16", &payload, &mut stats).expect("group stats");
        assert_eq!(stats.groups, 2);
        assert_eq!(stats.max_absmax, 3.0);
        assert_eq!(stats.sum_absmax, 5.0);
        assert_eq!(stats.zero_absmax_groups, 0);
        assert_eq!(stats.scale_format, "e4m3");
        assert_eq!(stats.scale_values.len(), 119);
        assert!(stats.scale_index_min <= stats.scale_index_max);
    }

    #[test]
    fn quant_dry_run_reconstructs_with_nearest_codebook_entry() {
        let payload = [
            0x80, 0x3f, // 1.0
            0x00, 0xbf, // -0.5
            0x00, 0x3f, // 0.5
            0x80, 0xbf, // -1.0
        ];
        let codebook = [-1.0, -0.5, 0.5, 1.0];
        let mut stats = new_quant_dry_run_stats(codebook.len(), 1.0, 0);
        update_quant_dry_run_stats("BF16", &payload, 2, &[1.0], &codebook, 1.0, 0, &mut stats)
            .expect("quant dry run");
        assert_eq!(stats.elements, 4);
        assert_eq!(stats.groups, 2);
        assert_eq!(stats.sse, 0.0);
        assert_eq!(stats.index_counts, vec![1, 1, 1, 1]);
    }

    #[test]
    fn quant_dry_run_scale_window_can_choose_lower_error_scale() {
        let mut payload = Vec::new();
        for value in [1.9f32, 0.75, 0.75, 0.75] {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        let codebook = [0.0, 1.0];
        let mut stats = new_quant_dry_run_stats(codebook.len(), 1.0, 1);
        update_quant_dry_run_stats(
            "F32",
            &payload,
            4,
            &[1.0, 2.0],
            &codebook,
            1.0,
            1,
            &mut stats,
        )
        .expect("quant dry run");
        assert_eq!(stats.groups, 1);
        assert_eq!(stats.scale_index_min, 0);
        assert_eq!(stats.scale_index_max, 0);
        assert_eq!(stats.scale_window_improved_groups, 1);
        assert!((stats.sse - 0.9975).abs() < 1e-6);
    }

    #[test]
    fn cxx_bf16_kernel_quantizes_and_packs_scale_window_result() {
        let payload = [
            0xf3, 0x3f, // 1.8984375
            0x40, 0x3f, // 0.75
            0x40, 0x3f, // 0.75
            0x40, 0x3f, // 0.75
        ];
        let scales = [1.0f32, 2.0];
        let codebook = [
            0.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ];
        let mut packed = [0u8; 2];
        let mut scale_indices = [0u8; 1];
        let mut metrics = empty_aq_quant_metrics();
        let request = AqQuantizeChunkRequestV1 {
            struct_size: std::mem::size_of::<AqQuantizeChunkRequestV1>(),
            dtype: ULLM_AQ_DTYPE_BF16,
            reserved0: 0,
            input: payload.as_ptr(),
            input_bytes: payload.len(),
            group_size: 4,
            scale_values: scales.as_ptr(),
            scale_count: scales.len(),
            codebook: codebook.as_ptr(),
            codebook_count: codebook.len(),
            tensor_scale: 1.0,
            reserved1: 0,
            scale_window: 1,
            packed_indices: packed.as_mut_ptr(),
            packed_indices_bytes: packed.len(),
            scale_indices: scale_indices.as_mut_ptr(),
            scale_indices_bytes: scale_indices.len(),
        };
        let status = unsafe {
            ullm_aq_quantize_chunk_v1(
                &request,
                &mut metrics,
                std::mem::size_of::<AqQuantMetrics>(),
            )
        };
        assert_eq!(status, 0);
        assert_eq!(scale_indices, [0]);
        assert_eq!(packed, [0x11, 0x11]);
        assert_eq!(metrics.groups, 1);
        assert_eq!(metrics.elements, 4);
        assert_eq!(metrics.scale_window_improved_groups, 1);
        assert_eq!(metrics.index_counts[1], 4);
    }

    #[test]
    fn cxx_v1_kernel_rejects_unsupported_dtype_and_short_output() {
        let payload = [0u8; 8];
        let scales = [1.0f32];
        let codebook = [0.0f32; 16];
        let mut packed = [0u8; 1];
        let mut scale_indices = [0u8; 1];
        let mut metrics = empty_aq_quant_metrics();
        let mut request = AqQuantizeChunkRequestV1 {
            struct_size: std::mem::size_of::<AqQuantizeChunkRequestV1>(),
            dtype: 99,
            reserved0: 0,
            input: payload.as_ptr(),
            input_bytes: payload.len(),
            group_size: 4,
            scale_values: scales.as_ptr(),
            scale_count: scales.len(),
            codebook: codebook.as_ptr(),
            codebook_count: codebook.len(),
            tensor_scale: 1.0,
            reserved1: 0,
            scale_window: 0,
            packed_indices: packed.as_mut_ptr(),
            packed_indices_bytes: packed.len(),
            scale_indices: scale_indices.as_mut_ptr(),
            scale_indices_bytes: scale_indices.len(),
        };
        let unsupported = unsafe {
            ullm_aq_quantize_chunk_v1(
                &request,
                &mut metrics,
                std::mem::size_of::<AqQuantMetrics>(),
            )
        };
        assert_eq!(unsupported, -5);

        request.dtype = ULLM_AQ_DTYPE_BF16;
        request.packed_indices_bytes = 1;
        let short_output = unsafe {
            ullm_aq_quantize_chunk_v1(
                &request,
                &mut metrics,
                std::mem::size_of::<AqQuantMetrics>(),
            )
        };
        assert_eq!(short_output, -4);
    }

    #[test]
    fn cxx_v1_kernel_handles_all_zero_bf16_group() {
        let payload = [0u8; 8];
        let scales = [1.0f32];
        let codebook = [
            0.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ];
        let mut packed = [0xffu8; 2];
        let mut scale_indices = [0xffu8; 1];
        let mut metrics = empty_aq_quant_metrics();
        let request = AqQuantizeChunkRequestV1 {
            struct_size: std::mem::size_of::<AqQuantizeChunkRequestV1>(),
            dtype: ULLM_AQ_DTYPE_BF16,
            reserved0: 0,
            input: payload.as_ptr(),
            input_bytes: payload.len(),
            group_size: 4,
            scale_values: scales.as_ptr(),
            scale_count: scales.len(),
            codebook: codebook.as_ptr(),
            codebook_count: codebook.len(),
            tensor_scale: 1.0,
            reserved1: 0,
            scale_window: 0,
            packed_indices: packed.as_mut_ptr(),
            packed_indices_bytes: packed.len(),
            scale_indices: scale_indices.as_mut_ptr(),
            scale_indices_bytes: scale_indices.len(),
        };
        let status = unsafe {
            ullm_aq_quantize_chunk_v1(
                &request,
                &mut metrics,
                std::mem::size_of::<AqQuantMetrics>(),
            )
        };
        assert_eq!(status, 0);
        assert_eq!(packed, [0x00, 0x00]);
        assert_eq!(scale_indices, [0]);
        assert_eq!(metrics.elements, 4);
        assert_eq!(metrics.groups, 1);
        assert_eq!(metrics.sse, 0.0);
        assert_eq!(metrics.ref_sse, 0.0);
        assert_eq!(metrics.index_counts[0], 4);
    }

    #[test]
    fn cxx_v1_kernel_quantizes_f16_chunk() {
        let payload = [
            0x00, 0xbc, // -1.0
            0x00, 0x38, // 0.5
            0x00, 0x3c, // 1.0
            0x00, 0x00, // 0.0
        ];
        let scales = [1.0f32];
        let codebook = [
            -1.0f32, -0.5, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ];
        let mut packed = [0u8; 2];
        let mut scale_indices = [0u8; 1];
        let mut metrics = empty_aq_quant_metrics();
        let request = AqQuantizeChunkRequestV1 {
            struct_size: std::mem::size_of::<AqQuantizeChunkRequestV1>(),
            dtype: ULLM_AQ_DTYPE_F16,
            reserved0: 0,
            input: payload.as_ptr(),
            input_bytes: payload.len(),
            group_size: 4,
            scale_values: scales.as_ptr(),
            scale_count: scales.len(),
            codebook: codebook.as_ptr(),
            codebook_count: codebook.len(),
            tensor_scale: 1.0,
            reserved1: 0,
            scale_window: 0,
            packed_indices: packed.as_mut_ptr(),
            packed_indices_bytes: packed.len(),
            scale_indices: scale_indices.as_mut_ptr(),
            scale_indices_bytes: scale_indices.len(),
        };
        let status = unsafe {
            ullm_aq_quantize_chunk_v1(
                &request,
                &mut metrics,
                std::mem::size_of::<AqQuantMetrics>(),
            )
        };
        assert_eq!(status, 0);
        assert_eq!(packed, [0x30, 0x24]);
        assert_eq!(scale_indices, [0]);
        assert_eq!(metrics.elements, 4);
        assert_eq!(metrics.sse, 0.0);
        assert_eq!(metrics.index_counts[0], 1);
        assert_eq!(metrics.index_counts[2], 1);
        assert_eq!(metrics.index_counts[3], 1);
        assert_eq!(metrics.index_counts[4], 1);
    }

    #[test]
    fn cxx_v1_kernel_writes_nan_as_zero_index_and_excludes_metrics() {
        let payload = [
            0xc0, 0x7f, // NaN
            0x80, 0x3f, // 1.0
            0xc0, 0x7f, // NaN
            0x80, 0xbf, // -1.0
        ];
        let scales = [1.0f32];
        let codebook = [
            -1.0f32, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ];
        let mut packed = [0u8; 2];
        let mut scale_indices = [0u8; 1];
        let mut metrics = empty_aq_quant_metrics();
        let request = AqQuantizeChunkRequestV1 {
            struct_size: std::mem::size_of::<AqQuantizeChunkRequestV1>(),
            dtype: ULLM_AQ_DTYPE_BF16,
            reserved0: 0,
            input: payload.as_ptr(),
            input_bytes: payload.len(),
            group_size: 4,
            scale_values: scales.as_ptr(),
            scale_count: scales.len(),
            codebook: codebook.as_ptr(),
            codebook_count: codebook.len(),
            tensor_scale: 1.0,
            reserved1: 0,
            scale_window: 0,
            packed_indices: packed.as_mut_ptr(),
            packed_indices_bytes: packed.len(),
            scale_indices: scale_indices.as_mut_ptr(),
            scale_indices_bytes: scale_indices.len(),
        };
        let status = unsafe {
            ullm_aq_quantize_chunk_v1(
                &request,
                &mut metrics,
                std::mem::size_of::<AqQuantMetrics>(),
            )
        };
        assert_eq!(status, 0);
        assert_eq!(packed, [0x20, 0x00]);
        assert_eq!(scale_indices, [0]);
        assert_eq!(metrics.elements, 2);
        assert_eq!(metrics.groups, 1);
        assert_eq!(metrics.index_counts[0], 1);
        assert_eq!(metrics.index_counts[2], 1);
    }

    #[test]
    fn cxx_v1_kernel_rejects_invalid_scale_codebook_and_layout() {
        let payload = [0u8; 8];
        let scales = [1.0f32];
        let codebook = [0.0f32; 16];
        let mut packed = [0u8; 2];
        let mut scale_indices = [0u8; 1];
        let mut metrics = empty_aq_quant_metrics();
        let mut request = AqQuantizeChunkRequestV1 {
            struct_size: std::mem::size_of::<AqQuantizeChunkRequestV1>(),
            dtype: ULLM_AQ_DTYPE_BF16,
            reserved0: 0,
            input: payload.as_ptr(),
            input_bytes: payload.len(),
            group_size: 4,
            scale_values: scales.as_ptr(),
            scale_count: 0,
            codebook: codebook.as_ptr(),
            codebook_count: codebook.len(),
            tensor_scale: 1.0,
            reserved1: 0,
            scale_window: 0,
            packed_indices: packed.as_mut_ptr(),
            packed_indices_bytes: packed.len(),
            scale_indices: scale_indices.as_mut_ptr(),
            scale_indices_bytes: scale_indices.len(),
        };
        let scale_count_zero = unsafe {
            ullm_aq_quantize_chunk_v1(
                &request,
                &mut metrics,
                std::mem::size_of::<AqQuantMetrics>(),
            )
        };
        assert_eq!(scale_count_zero, -2);

        request.scale_count = scales.len();
        request.codebook_count = 15;
        let bad_codebook_count = unsafe {
            ullm_aq_quantize_chunk_v1(
                &request,
                &mut metrics,
                std::mem::size_of::<AqQuantMetrics>(),
            )
        };
        assert_eq!(bad_codebook_count, -2);

        request.codebook_count = codebook.len();
        request.input_bytes = 6;
        let bad_layout = unsafe {
            ullm_aq_quantize_chunk_v1(
                &request,
                &mut metrics,
                std::mem::size_of::<AqQuantMetrics>(),
            )
        };
        assert_eq!(bad_layout, -3);
    }

    #[test]
    fn cxx_v1_kernel_matches_rust_scalar_metrics_on_bf16_chunk() {
        fn append_bf16(bytes: &mut Vec<u8>, value: f32) {
            let raw = (value.to_bits() >> 16) as u16;
            bytes.extend_from_slice(&raw.to_le_bytes());
        }

        let values = [-1.0f32, -0.6, 0.2, 1.5, 0.0, 0.1, -0.2, 0.4];
        let mut payload = Vec::new();
        for value in values {
            append_bf16(&mut payload, value);
        }
        let scales = [0.5f32, 1.0, 2.0];
        let codebook = [
            -1.0f32, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0,
        ];
        let mut rust_stats = new_quant_dry_run_stats(codebook.len(), 1.0, 1);
        update_quant_dry_run_stats(
            "BF16",
            &payload,
            4,
            &scales,
            &codebook,
            1.0,
            1,
            &mut rust_stats,
        )
        .expect("rust scalar dry run");

        let mut packed = vec![0u8; payload.len() / 4];
        let mut scale_indices = vec![0u8; payload.len() / (2 * 4)];
        let mut cxx_metrics = empty_aq_quant_metrics();
        let request = AqQuantizeChunkRequestV1 {
            struct_size: std::mem::size_of::<AqQuantizeChunkRequestV1>(),
            dtype: ULLM_AQ_DTYPE_BF16,
            reserved0: 0,
            input: payload.as_ptr(),
            input_bytes: payload.len(),
            group_size: 4,
            scale_values: scales.as_ptr(),
            scale_count: scales.len(),
            codebook: codebook.as_ptr(),
            codebook_count: codebook.len(),
            tensor_scale: 1.0,
            reserved1: 0,
            scale_window: 1,
            packed_indices: packed.as_mut_ptr(),
            packed_indices_bytes: packed.len(),
            scale_indices: scale_indices.as_mut_ptr(),
            scale_indices_bytes: scale_indices.len(),
        };
        let status = unsafe {
            ullm_aq_quantize_chunk_v1(
                &request,
                &mut cxx_metrics,
                std::mem::size_of::<AqQuantMetrics>(),
            )
        };
        assert_eq!(status, 0);
        assert_eq!(rust_stats.elements, cxx_metrics.elements as usize);
        assert_eq!(rust_stats.groups, cxx_metrics.groups as usize);
        assert!((rust_stats.sse - cxx_metrics.sse).abs() < 1e-12);
        assert!((rust_stats.ref_sse - cxx_metrics.ref_sse).abs() < 1e-12);
        assert_eq!(rust_stats.max_abs_error, cxx_metrics.max_abs_error);
        assert_eq!(
            rust_stats.scale_index_min,
            cxx_metrics.scale_index_min as usize
        );
        assert_eq!(
            rust_stats.scale_index_max,
            cxx_metrics.scale_index_max as usize
        );
        assert_eq!(
            rust_stats.scale_window_improved_groups,
            cxx_metrics.scale_window_improved_groups as usize
        );
        for (left, right) in rust_stats.index_counts.iter().zip(cxx_metrics.index_counts) {
            assert_eq!(*left, right as usize);
        }
    }

    #[test]
    fn exported_codebook_selection_requires_16_entries() {
        let export = CodebookExport {
            codebooks: vec![CodebookEntry {
                family: "mlp_up".to_string(),
                candidate_id: "aq4_e4m3_g16_ts_flloyd16".to_string(),
                values_f32: (0..16).map(|value| value as f32).collect(),
            }],
        };
        let values =
            select_codebook(&export, "mlp_up", "aq4_e4m3_g16_ts_flloyd16").expect("codebook");
        assert_eq!(values.len(), 16);
        assert_eq!(values[15], 15.0);
    }
}
