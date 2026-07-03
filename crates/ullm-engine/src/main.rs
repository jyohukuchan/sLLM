// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

use std::env;
use std::fs::File;
use std::io::Read;
use std::process::ExitCode;
use ullm_engine::loader::{LoadOptions, LoadedPayload, WeightRegistry};
use ullm_engine::package::{ReferencedFile, ReferencedFileRole, TensorSelector};

fn main() -> ExitCode {
    match env::args().nth(1).as_deref() {
        Some("inspect-devices") => inspect_devices(),
        Some("runtime-smoke") => runtime_smoke(),
        Some("runtime-memory-smoke") => runtime_memory_smoke(env::args().nth(2)),
        Some("runtime-stream-smoke") => runtime_stream_smoke(env::args().nth(2)),
        Some("runtime-copy-smoke") => runtime_copy_smoke(env::args().nth(2)),
        Some("inspect-package") => inspect_package(env::args().nth(2)),
        Some("package-load-smoke") => package_load_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
        ),
        Some("package-tensor-load-smoke") => package_tensor_load_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
        ),
        Some("package-weight-register-smoke") => package_weight_register_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
        ),
        Some("package-weight-register-many-smoke") => package_weight_register_many_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
        ),
        Some("-h") | Some("--help") | None => {
            print_help();
            ExitCode::SUCCESS
        }
        Some(command) => {
            eprintln!("unknown command: {command}");
            print_help();
            ExitCode::from(2)
        }
    }
}

fn inspect_devices() -> ExitCode {
    println!("uLLM runtime ABI {}", ullm_runtime_sys::abi_version());
    let count = match ullm_runtime_sys::device_count() {
        Ok(count) => count,
        Err(err) => {
            eprintln!("failed to query device count: {err}");
            return ExitCode::from(1);
        }
    };
    println!("devices: {count}");
    for index in 0..count {
        match ullm_runtime_sys::device_info(index) {
            Ok(info) => {
                println!(
                    "[{index}] backend={} id={} name=\"{}\" mem={} compute={}.{} arch=\"{}\" flags={}",
                    info.backend,
                    info.device_id,
                    info.name,
                    info.total_global_mem,
                    info.compute_major,
                    info.compute_minor,
                    info.gcn_arch_name,
                    info.flags
                );
            }
            Err(err) => {
                eprintln!("failed to query device {index}: {err}");
                return ExitCode::from(1);
            }
        }
    }
    ExitCode::SUCCESS
}

fn runtime_smoke() -> ExitCode {
    let lhs = [1.0_f32, 2.0, 3.5, -4.0];
    let rhs = [10.0_f32, -2.0, 0.5, 4.0];
    let out = match ullm_runtime_sys::smoke_add_f32(&lhs, &rhs) {
        Ok(out) => out,
        Err(err) => {
            eprintln!("runtime smoke failed: {err}");
            return ExitCode::from(1);
        }
    };
    println!("runtime-smoke add_f32 output: {out:?}");
    if out == [11.0, 0.0, 4.0, 0.0] {
        ExitCode::SUCCESS
    } else {
        eprintln!("runtime smoke produced unexpected output");
        ExitCode::from(1)
    }
}

fn runtime_memory_smoke(device_index: Option<String>) -> ExitCode {
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let mut context = match ullm_runtime_sys::RuntimeContext::create(device_index) {
        Ok(context) => context,
        Err(err) => {
            eprintln!("failed to create runtime context: {err}");
            return ExitCode::from(1);
        }
    };
    let info = match context.device_info() {
        Ok(info) => info,
        Err(err) => {
            eprintln!("failed to query runtime context device: {err}");
            return ExitCode::from(1);
        }
    };
    let bytes = 4 * 1024 * 1024;
    let buffer = match context.alloc_buffer(bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let actual = match buffer.size() {
        Ok(bytes) => bytes,
        Err(err) => {
            eprintln!("failed to query runtime buffer size: {err}");
            return ExitCode::from(1);
        }
    };
    println!(
        "runtime-memory-smoke backend={} device_index={} name=\"{}\" bytes={}",
        info.backend, device_index, info.name, actual
    );
    if actual == bytes {
        ExitCode::SUCCESS
    } else {
        eprintln!("runtime memory smoke returned unexpected buffer size");
        ExitCode::from(1)
    }
}

fn runtime_stream_smoke(device_index: Option<String>) -> ExitCode {
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let mut context = match ullm_runtime_sys::RuntimeContext::create(device_index) {
        Ok(context) => context,
        Err(err) => {
            eprintln!("failed to create runtime context: {err}");
            return ExitCode::from(1);
        }
    };
    let info = match context.device_info() {
        Ok(info) => info,
        Err(err) => {
            eprintln!("failed to query runtime context device: {err}");
            return ExitCode::from(1);
        }
    };
    let mut stream = match context.create_stream() {
        Ok(stream) => stream,
        Err(err) => {
            eprintln!("failed to create runtime stream: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream: {err}");
        return ExitCode::from(1);
    }
    println!(
        "runtime-stream-smoke backend={} device_index={} name=\"{}\" synchronized=true",
        info.backend, device_index, info.name
    );
    ExitCode::SUCCESS
}

fn runtime_copy_smoke(device_index: Option<String>) -> ExitCode {
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let mut context = match ullm_runtime_sys::RuntimeContext::create(device_index) {
        Ok(context) => context,
        Err(err) => {
            eprintln!("failed to create runtime context: {err}");
            return ExitCode::from(1);
        }
    };
    let info = match context.device_info() {
        Ok(info) => info,
        Err(err) => {
            eprintln!("failed to query runtime context device: {err}");
            return ExitCode::from(1);
        }
    };
    let mut stream = match context.create_stream() {
        Ok(stream) => stream,
        Err(err) => {
            eprintln!("failed to create runtime stream: {err}");
            return ExitCode::from(1);
        }
    };
    let bytes = 4096_usize;
    let mut buffer = match context.alloc_buffer(bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let input: Vec<u8> = (0..bytes)
        .map(|index| (index.wrapping_mul(31).wrapping_add(7) & 0xff) as u8)
        .collect();
    if let Err(err) = buffer.copy_from_host(0, &input, Some(&mut stream)) {
        eprintln!("failed to copy host data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after host copy: {err}");
        return ExitCode::from(1);
    }

    let mut output = vec![0_u8; bytes];
    if let Err(err) = buffer.copy_to_host(0, &mut output, Some(&mut stream)) {
        eprintln!("failed to copy runtime buffer data back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after device copy: {err}");
        return ExitCode::from(1);
    }
    if input != output {
        eprintln!("runtime copy smoke returned mismatched bytes");
        return ExitCode::from(1);
    }
    println!(
        "runtime-copy-smoke backend={} device_index={} name=\"{}\" bytes={} verified=true",
        info.backend, device_index, info.name, bytes
    );
    ExitCode::SUCCESS
}

fn inspect_package(path: Option<String>) -> ExitCode {
    let Some(path) = path else {
        eprintln!("inspect-package requires a .ullm.d path");
        return ExitCode::from(2);
    };
    let summary = match ullm_engine::package::inspect_package(path) {
        Ok(summary) => summary,
        Err(err) => {
            eprintln!("failed to inspect package: {err}");
            return ExitCode::from(1);
        }
    };
    println!("package: {}", summary.package_dir.display());
    println!(
        "schema: {}",
        summary
            .schema_version
            .unwrap_or_else(|| "unknown".to_string())
    );
    if let Some(source) = summary.source_model_dir {
        println!("source_model_dir: {source}");
    }
    println!("quantized_tensors: {}", summary.quantized_tensors);
    println!("passthrough_tensors: {}", summary.passthrough_tensors);
    println!("codebooks: {}", summary.codebooks);
    println!("quantized_elements: {}", summary.quantized_elements);
    println!("passthrough_elements: {}", summary.passthrough_elements);
    println!("referenced_files: {}", summary.referenced_files);
    println!("referenced_file_bytes: {}", summary.referenced_file_bytes);
    println!(
        "missing_referenced_files: {}",
        summary.missing_referenced_files
    );
    println!(
        "declared_passthrough_payload_bytes: {}",
        summary.declared_passthrough_payload_bytes
    );
    ExitCode::SUCCESS
}

fn package_load_smoke(
    path: Option<String>,
    device_index: Option<String>,
    max_bytes: Option<String>,
    payload_role: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-load-smoke requires a .ullm.d path");
        return ExitCode::from(2);
    };
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let max_bytes = match parse_optional_usize(max_bytes, 1024 * 1024, "max bytes") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("max bytes must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let payload_role = match parse_optional_payload_role(payload_role) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let summary = match ullm_engine::package::inspect_package(&path) {
        Ok(summary) => summary,
        Err(err) => {
            eprintln!("failed to inspect package: {err}");
            return ExitCode::from(1);
        }
    };
    let selected = match ullm_engine::package::select_existing_referenced_file(&path, payload_role)
    {
        Ok(selected) => selected,
        Err(err) => {
            eprintln!("failed to select package payload: {err}");
            return ExitCode::from(1);
        }
    };
    let data = match read_bounded_file(&selected.absolute_path, max_bytes) {
        Ok(data) => data,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    if data.is_empty() {
        eprintln!("selected payload produced zero bytes after applying max-bytes");
        return ExitCode::from(1);
    }

    let mut context = match ullm_runtime_sys::RuntimeContext::create(device_index) {
        Ok(context) => context,
        Err(err) => {
            eprintln!("failed to create runtime context: {err}");
            return ExitCode::from(1);
        }
    };
    let info = match context.device_info() {
        Ok(info) => info,
        Err(err) => {
            eprintln!("failed to query runtime context device: {err}");
            return ExitCode::from(1);
        }
    };
    let mut stream = match context.create_stream() {
        Ok(stream) => stream,
        Err(err) => {
            eprintln!("failed to create runtime stream: {err}");
            return ExitCode::from(1);
        }
    };
    let mut buffer = match context.alloc_buffer(data.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = buffer.copy_from_host(0, &data, Some(&mut stream)) {
        eprintln!("failed to copy package payload into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after package payload load: {err}");
        return ExitCode::from(1);
    }
    let mut output = vec![0_u8; data.len()];
    if let Err(err) = buffer.copy_to_host(0, &mut output, Some(&mut stream)) {
        eprintln!("failed to copy package payload back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after package payload readback: {err}");
        return ExitCode::from(1);
    }
    if data != output {
        eprintln!("package payload roundtrip produced mismatched bytes");
        return ExitCode::from(1);
    }
    println!(
        "package-load-smoke package={} schema={} role={} file={} file_bytes={} copied_bytes={} owner_index={} owner_name=\"{}\" backend={} device_index={} name=\"{}\" verified=true",
        summary.package_dir.display(),
        summary
            .schema_version
            .unwrap_or_else(|| "unknown".to_string()),
        selected.role.as_str(),
        selected.relative_path,
        selected.bytes,
        data.len(),
        selected
            .owner_index
            .map(|index| index.to_string())
            .unwrap_or_else(|| "none".to_string()),
        selected.owner_name.unwrap_or_else(|| "none".to_string()),
        info.backend,
        device_index,
        info.name
    );
    ExitCode::SUCCESS
}

fn package_tensor_load_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    tensor_selector: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-tensor-load-smoke requires a .ullm.d path");
        return ExitCode::from(2);
    };
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let chunk_bytes = match parse_optional_usize(chunk_bytes, 1024 * 1024, "chunk bytes") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("chunk bytes must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let selector = TensorSelector::parse(tensor_selector.as_deref());
    let bundle = match ullm_engine::package::select_tensor_payload_bundle(&path, &selector) {
        Ok(bundle) => bundle,
        Err(err) => {
            eprintln!("failed to select package tensor payloads: {err}");
            return ExitCode::from(1);
        }
    };

    let mut context = match ullm_runtime_sys::RuntimeContext::create(device_index) {
        Ok(context) => context,
        Err(err) => {
            eprintln!("failed to create runtime context: {err}");
            return ExitCode::from(1);
        }
    };
    let info = match context.device_info() {
        Ok(info) => info,
        Err(err) => {
            eprintln!("failed to query runtime context device: {err}");
            return ExitCode::from(1);
        }
    };
    let mut stream = match context.create_stream() {
        Ok(stream) => stream,
        Err(err) => {
            eprintln!("failed to create runtime stream: {err}");
            return ExitCode::from(1);
        }
    };

    let index_summary =
        match roundtrip_file_chunks(&mut context, &mut stream, &bundle.index_file, chunk_bytes) {
            Ok(summary) => summary,
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::from(1);
            }
        };
    let scale_summary =
        match roundtrip_file_chunks(&mut context, &mut stream, &bundle.scale_file, chunk_bytes) {
            Ok(summary) => summary,
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::from(1);
            }
        };
    let codebook_summary = match roundtrip_file_chunks(
        &mut context,
        &mut stream,
        &bundle.codebook_file,
        chunk_bytes,
    ) {
        Ok(summary) => summary,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };

    println!(
        "package-tensor-load-smoke package={} tensor_index={} tensor=\"{}\" dtype={} family={} candidate_id={} elements={} groups={} backend={} device_index={} name=\"{}\" chunk_bytes={} verified=true",
        path,
        bundle.tensor_index,
        bundle.tensor_name,
        bundle.dtype.as_deref().unwrap_or("unknown"),
        bundle.family.as_deref().unwrap_or("unknown"),
        bundle.candidate_id.as_deref().unwrap_or("unknown"),
        bundle.elements,
        bundle.groups,
        info.backend,
        device_index,
        info.name,
        chunk_bytes
    );
    print_file_roundtrip_summary("tensor-index", &bundle.index_file, &index_summary);
    print_file_roundtrip_summary("tensor-scale", &bundle.scale_file, &scale_summary);
    print_file_roundtrip_summary("tensor-codebook", &bundle.codebook_file, &codebook_summary);
    ExitCode::SUCCESS
}

fn package_weight_register_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    tensor_selector: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-weight-register-smoke requires a .ullm.d path");
        return ExitCode::from(2);
    };
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let chunk_bytes = match parse_optional_usize(chunk_bytes, 1024 * 1024, "chunk bytes") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("chunk bytes must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let selector = TensorSelector::parse(tensor_selector.as_deref());
    let bundle = match ullm_engine::package::select_tensor_payload_bundle(&path, &selector) {
        Ok(bundle) => bundle,
        Err(err) => {
            eprintln!("failed to select package tensor payloads: {err}");
            return ExitCode::from(1);
        }
    };

    let mut context = match ullm_runtime_sys::RuntimeContext::create(device_index) {
        Ok(context) => context,
        Err(err) => {
            eprintln!("failed to create runtime context: {err}");
            return ExitCode::from(1);
        }
    };
    let info = match context.device_info() {
        Ok(info) => info,
        Err(err) => {
            eprintln!("failed to query runtime context device: {err}");
            return ExitCode::from(1);
        }
    };
    let mut stream = match context.create_stream() {
        Ok(stream) => stream,
        Err(err) => {
            eprintln!("failed to create runtime stream: {err}");
            return ExitCode::from(1);
        }
    };
    let mut registry = WeightRegistry::new();
    let registry_index = match registry.load_and_insert(
        &mut context,
        &mut stream,
        &bundle,
        LoadOptions {
            chunk_bytes,
            verify: true,
        },
    ) {
        Ok(index) => index,
        Err(err) => {
            eprintln!("failed to register package tensor payloads: {err}");
            return ExitCode::from(1);
        }
    };
    let Some(loaded) = registry.get(registry_index) else {
        eprintln!("registered tensor disappeared from weight registry");
        return ExitCode::from(1);
    };

    println!(
        "package-weight-register-smoke package={} registry_index={} registry_tensors={} registry_payload_bytes={} resident_payload_bytes={} codebook_payloads={} tensor_index={} tensor=\"{}\" dtype={} family={} candidate_id={} elements={} groups={} backend={} device_index={} name=\"{}\" chunk_bytes={} verified=true",
        path,
        registry_index,
        registry.len(),
        registry.total_payload_bytes(),
        registry.resident_payload_bytes(),
        registry.codebook_payloads(),
        loaded.tensor_index,
        loaded.tensor_name,
        loaded.dtype.as_deref().unwrap_or("unknown"),
        loaded.family.as_deref().unwrap_or("unknown"),
        loaded.candidate_id.as_deref().unwrap_or("unknown"),
        loaded.elements,
        loaded.groups,
        info.backend,
        device_index,
        info.name,
        chunk_bytes
    );
    print_loaded_payload_summary(&loaded.index);
    print_loaded_payload_summary(&loaded.scale);
    print_loaded_payload_summary(&loaded.codebook);
    ExitCode::SUCCESS
}

fn package_weight_register_many_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    max_tensors: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-weight-register-many-smoke requires a .ullm.d path");
        return ExitCode::from(2);
    };
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let chunk_bytes = match parse_optional_usize(chunk_bytes, 1024 * 1024, "chunk bytes") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("chunk bytes must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let max_tensors = match parse_optional_usize(max_tensors, 2, "max tensors") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("max tensors must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let bundles = match ullm_engine::package::list_tensor_payload_bundles(&path) {
        Ok(bundles) => bundles,
        Err(err) => {
            eprintln!("failed to list package tensor payloads: {err}");
            return ExitCode::from(1);
        }
    };
    if bundles.is_empty() {
        eprintln!("package contains no quantized tensor payload bundles");
        return ExitCode::from(1);
    }
    let selected_count = bundles.len().min(max_tensors);

    let mut context = match ullm_runtime_sys::RuntimeContext::create(device_index) {
        Ok(context) => context,
        Err(err) => {
            eprintln!("failed to create runtime context: {err}");
            return ExitCode::from(1);
        }
    };
    let info = match context.device_info() {
        Ok(info) => info,
        Err(err) => {
            eprintln!("failed to query runtime context device: {err}");
            return ExitCode::from(1);
        }
    };
    let mut stream = match context.create_stream() {
        Ok(stream) => stream,
        Err(err) => {
            eprintln!("failed to create runtime stream: {err}");
            return ExitCode::from(1);
        }
    };
    let mut registry = WeightRegistry::new();
    let registry_indices = match registry.load_and_insert_many(
        &mut context,
        &mut stream,
        &bundles[..selected_count],
        LoadOptions {
            chunk_bytes,
            verify: true,
        },
    ) {
        Ok(indices) => indices,
        Err(err) => {
            eprintln!("failed to register package tensor payloads: {err}");
            return ExitCode::from(1);
        }
    };

    println!(
        "package-weight-register-many-smoke package={} selected_tensors={} package_tensors={} registry_tensors={} registry_payload_bytes={} resident_payload_bytes={} codebook_payloads={} backend={} device_index={} name=\"{}\" chunk_bytes={} verified=true",
        path,
        selected_count,
        bundles.len(),
        registry.len(),
        registry.total_payload_bytes(),
        registry.resident_payload_bytes(),
        registry.codebook_payloads(),
        info.backend,
        device_index,
        info.name,
        chunk_bytes
    );
    for registry_index in registry_indices {
        let Some(loaded) = registry.get(registry_index) else {
            eprintln!("registered tensor disappeared from weight registry");
            return ExitCode::from(1);
        };
        println!(
            "  registered_tensor registry_index={} tensor_index={} tensor=\"{}\" bytes={}",
            registry_index,
            loaded.tensor_index,
            loaded.tensor_name,
            loaded.total_payload_bytes()
        );
    }
    ExitCode::SUCCESS
}

fn print_help() {
    eprintln!(
        "usage: ullm-engine <inspect-devices|runtime-smoke|runtime-memory-smoke [DEVICE_INDEX]|runtime-stream-smoke [DEVICE_INDEX]|runtime-copy-smoke [DEVICE_INDEX]|inspect-package PATH|package-load-smoke PACKAGE_DIR [DEVICE_INDEX] [MAX_BYTES] [PAYLOAD_ROLE]|package-tensor-load-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [TENSOR_SELECTOR]|package-weight-register-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [TENSOR_SELECTOR]|package-weight-register-many-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [MAX_TENSORS]>"
    );
    eprintln!(
        "payload roles: smallest|tensor-index|tensor-scale|tensor-codebook|codebook|passthrough"
    );
    eprintln!("tensor selector: omitted or numeric index, exact tensor name, or unique substring");
}

fn parse_optional_device_index(device_index: Option<String>) -> Result<u32, ExitCode> {
    match device_index {
        Some(value) => match value.parse::<u32>() {
            Ok(value) => Ok(value),
            Err(err) => {
                eprintln!("invalid device index: {err}");
                Err(ExitCode::from(2))
            }
        },
        None => Ok(0),
    }
}

fn parse_optional_usize(
    value: Option<String>,
    default: usize,
    label: &str,
) -> Result<usize, ExitCode> {
    match value {
        Some(value) => match value.parse::<usize>() {
            Ok(value) => Ok(value),
            Err(err) => {
                eprintln!("invalid {label}: {err}");
                Err(ExitCode::from(2))
            }
        },
        None => Ok(default),
    }
}

fn parse_optional_payload_role(value: Option<String>) -> Result<ReferencedFileRole, ExitCode> {
    match value {
        Some(value) => ReferencedFileRole::parse(&value).ok_or_else(|| {
            eprintln!(
                "invalid payload role: {value}; expected smallest, tensor-index, tensor-scale, tensor-codebook, codebook, or passthrough"
            );
            ExitCode::from(2)
        }),
        None => Ok(ReferencedFileRole::Smallest),
    }
}

fn read_bounded_file(path: &std::path::Path, max_bytes: usize) -> Result<Vec<u8>, String> {
    let file =
        File::open(path).map_err(|err| format!("failed to open {}: {err}", path.display()))?;
    let limit = u64::try_from(max_bytes).unwrap_or(u64::MAX);
    let mut reader = file.take(limit);
    let mut data = Vec::with_capacity(max_bytes.min(1024 * 1024));
    reader
        .read_to_end(&mut data)
        .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    Ok(data)
}

#[derive(Debug, Clone, Copy)]
struct FileRoundtripSummary {
    bytes: u64,
    chunks: u64,
}

fn roundtrip_file_chunks(
    context: &mut ullm_runtime_sys::RuntimeContext,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    referenced: &ReferencedFile,
    chunk_bytes: usize,
) -> Result<FileRoundtripSummary, String> {
    if chunk_bytes == 0 {
        return Err("chunk bytes must be greater than zero".to_string());
    }
    let mut file = File::open(&referenced.absolute_path).map_err(|err| {
        format!(
            "failed to open {}: {err}",
            referenced.absolute_path.display()
        )
    })?;
    let capacity = usize::try_from(referenced.bytes)
        .ok()
        .map_or(chunk_bytes, |bytes| bytes.min(chunk_bytes));
    if capacity == 0 {
        return Err(format!(
            "referenced file {} is empty",
            referenced.absolute_path.display()
        ));
    }
    let mut buffer = context.alloc_buffer(capacity)?;
    let mut input = vec![0_u8; capacity];
    let mut output = vec![0_u8; capacity];
    let mut total = 0_u64;
    let mut chunks = 0_u64;

    loop {
        let read = file.read(&mut input).map_err(|err| {
            format!(
                "failed to read {}: {err}",
                referenced.absolute_path.display()
            )
        })?;
        if read == 0 {
            break;
        }
        buffer.copy_from_host(0, &input[..read], Some(stream))?;
        stream.synchronize()?;
        buffer.copy_to_host(0, &mut output[..read], Some(stream))?;
        stream.synchronize()?;
        if input[..read] != output[..read] {
            return Err(format!(
                "runtime roundtrip mismatch for {} at chunk {}",
                referenced.relative_path, chunks
            ));
        }
        total += read as u64;
        chunks += 1;
    }

    if total != referenced.bytes {
        return Err(format!(
            "roundtrip byte count mismatch for {}: expected {} got {}",
            referenced.relative_path, referenced.bytes, total
        ));
    }
    Ok(FileRoundtripSummary {
        bytes: total,
        chunks,
    })
}

fn print_file_roundtrip_summary(
    role: &str,
    referenced: &ReferencedFile,
    summary: &FileRoundtripSummary,
) {
    println!(
        "  file role={} path={} bytes={} chunks={} verified=true",
        role, referenced.relative_path, summary.bytes, summary.chunks
    );
}

fn print_loaded_payload_summary(payload: &LoadedPayload) {
    let buffer_bytes = payload
        .buffer
        .size()
        .map(|bytes| bytes.to_string())
        .unwrap_or_else(|err| format!("error:{err}"));
    println!(
        "  registered role={} path={} bytes={} chunks={} buffer_bytes={} resident=true",
        payload.role.as_str(),
        payload.relative_path,
        payload.bytes,
        payload.chunks,
        buffer_bytes
    );
}
