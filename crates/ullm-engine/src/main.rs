// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

use std::env;
use std::process::ExitCode;

fn main() -> ExitCode {
    match env::args().nth(1).as_deref() {
        Some("inspect-devices") => inspect_devices(),
        Some("runtime-smoke") => runtime_smoke(),
        Some("runtime-memory-smoke") => runtime_memory_smoke(env::args().nth(2)),
        Some("runtime-stream-smoke") => runtime_stream_smoke(env::args().nth(2)),
        Some("inspect-package") => inspect_package(env::args().nth(2)),
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

fn print_help() {
    eprintln!(
        "usage: ullm-engine <inspect-devices|runtime-smoke|runtime-memory-smoke [DEVICE_INDEX]|runtime-stream-smoke [DEVICE_INDEX]|inspect-package PATH>"
    );
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
