use std::env;
use std::collections::BTreeMap;
use std::fs;
use std::io::Read;
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

unsafe extern "C" {
    fn ullm_aq_get_kernel_version() -> KernelVersion;
    fn ullm_aq_pack_nibbles(
        low: *const u8,
        high: *const u8,
        output: *mut u8,
        len: usize,
    ) -> usize;
}

#[derive(Debug)]
struct Options {
    threads: usize,
    io_threads: usize,
    max_working_memory_mib: usize,
    model_dir: Option<PathBuf>,
    plan_output: Option<PathBuf>,
    dry_run: bool,
}

#[derive(Debug, Deserialize)]
struct SafetensorsIndex {
    weight_map: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct TensorHeader {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
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
}

#[derive(Debug, Serialize)]
struct ModelPlan {
    schema_version: String,
    model_dir: String,
    tensor_count: usize,
    supported_tensor_count: usize,
    passthrough_tensor_count: usize,
    total_tensor_bytes: usize,
    tensors: Vec<TensorPlan>,
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

fn parse_options() -> Result<Options, String> {
    let mut args = env::args().skip(1);
    let mut options = Options {
        threads: default_threads(),
        io_threads: 2,
        max_working_memory_mib: 4096,
        model_dir: None,
        plan_output: None,
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
                options.plan_output = Some(PathBuf::from(
                    args.next()
                        .ok_or_else(|| "--plan-output requires a value".to_string())?,
                ));
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

fn is_supported_input(dtype: &str, shape: &[usize], family: &str) -> bool {
    matches!(dtype, "BF16" | "F16" | "F32") && shape.len() >= 2 && is_default_quant_family(family)
}

fn tensor_elements(shape: &[usize]) -> Result<usize, String> {
    shape.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim)
            .ok_or_else(|| format!("tensor element count overflows usize for shape {shape:?}"))
    })
}

fn read_safetensors_header(path: &Path) -> Result<BTreeMap<String, TensorHeader>, String> {
    let mut file = fs::File::open(path).map_err(|err| format!("failed to open {}: {err}", path.display()))?;
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)
        .map_err(|err| format!("failed to read safetensors header length from {}: {err}", path.display()))?;
    let header_len = u64::from_le_bytes(len_bytes) as usize;
    if header_len > 128 * 1024 * 1024 {
        return Err(format!("safetensors header is unexpectedly large: {header_len} bytes"));
    }
    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes)
        .map_err(|err| format!("failed to read safetensors header from {}: {err}", path.display()))?;
    let raw: BTreeMap<String, serde_json::Value> = serde_json::from_slice(&header_bytes)
        .map_err(|err| format!("failed to parse safetensors header {}: {err}", path.display()))?;
    let mut tensors = BTreeMap::new();
    for (name, value) in raw {
        if name == "__metadata__" {
            continue;
        }
        let header: TensorHeader = serde_json::from_value(value)
            .map_err(|err| format!("failed to parse tensor header {name} in {}: {err}", path.display()))?;
        tensors.insert(name, header);
    }
    Ok(tensors)
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
        return Err(format!("no safetensors files found in {}", model_dir.display()));
    }
    Ok(files)
}

fn build_model_plan(model_dir: &Path) -> Result<ModelPlan, String> {
    let mut tensors = Vec::new();
    for path in safetensor_files(model_dir)? {
        let headers = read_safetensors_header(&path)?;
        for (name, header) in headers {
            let n_elements = tensor_elements(&header.shape)?;
            let n_bytes = header
                .data_offsets
                .get(1)
                .zip(header.data_offsets.first())
                .map(|(end, start)| end.saturating_sub(*start))
                .ok_or_else(|| format!("invalid data offsets for {name}"))?;
            let family = family_for_tensor(&name).to_string();
            let supported_input = is_supported_input(&header.dtype, &header.shape, &family);
            tensors.push(TensorPlan {
                family,
                action: if supported_input { "quantize".to_string() } else { "passthrough".to_string() },
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
    let supported_tensor_count = tensors.iter().filter(|tensor| tensor.supported_input).count();
    let total_tensor_bytes = tensors.iter().map(|tensor| tensor.n_bytes).sum();
    Ok(ModelPlan {
        schema_version: "ullm-quant-plan-v0.1".to_string(),
        model_dir: model_dir.display().to_string(),
        tensor_count: tensors.len(),
        supported_tensor_count,
        passthrough_tensor_count: tensors.len() - supported_tensor_count,
        total_tensor_bytes,
        tensors,
    })
}

fn run_pack_smoke() -> Result<Vec<u8>, String> {
    let low = [0x00, 0x01, 0x0f, 0x08];
    let high = [0x01, 0x02, 0x00, 0x07];
    let mut output = [0u8; 4];
    let written = unsafe {
        ullm_aq_pack_nibbles(low.as_ptr(), high.as_ptr(), output.as_mut_ptr(), output.len())
    };
    if written != output.len() {
        return Err(format!("pack smoke wrote {written}, expected {}", output.len()));
    }
    let expected = [0x10, 0x21, 0x0f, 0x78];
    if output != expected {
        return Err(format!("pack smoke output mismatch: {output:?} != {expected:?}"));
    }
    Ok(output.to_vec())
}

fn run() -> Result<(), String> {
    let options = parse_options()?;
    let version = unsafe { ullm_aq_get_kernel_version() };
    let packed = run_pack_smoke()?;
    let plan = match options.model_dir.as_deref() {
        Some(model_dir) => Some(build_model_plan(model_dir)?),
        None => None,
    };

    println!("ullm-quant skeleton");
    println!(
        "kernel_version={}.{}.{}",
        version.major, version.minor, version.patch
    );
    println!("threads={}", options.threads);
    println!("io_threads={}", options.io_threads);
    println!("max_working_memory_mib={}", options.max_working_memory_mib);
    println!("dry_run={}", options.dry_run);
    println!("pack_smoke=ok {packed:?}");
    if let Some(plan) = &plan {
        println!("plan_model_dir={}", plan.model_dir);
        println!("plan_tensor_count={}", plan.tensor_count);
        println!("plan_supported_tensor_count={}", plan.supported_tensor_count);
        println!("plan_passthrough_tensor_count={}", plan.passthrough_tensor_count);
        println!("plan_total_tensor_bytes={}", plan.total_tensor_bytes);
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
    use super::{default_threads, family_for_tensor};

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
}
