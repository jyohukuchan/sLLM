// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

use std::env;
use std::fs::File;
use std::io::Read;
use std::process::ExitCode;
use std::time::Instant;
use ullm_engine::loader::{
    LoadOptions, LoadedPayload, LoadedTensorBundle, WeightRegistry, load_package_tensor_prefix,
};
use ullm_engine::package::{
    PassthroughPayloadBundle, ReferencedFile, ReferencedFileRole, TensorSelector,
};

fn main() -> ExitCode {
    match env::args().nth(1).as_deref() {
        Some("inspect-devices") => inspect_devices(),
        Some("runtime-smoke") => runtime_smoke(),
        Some("runtime-memory-smoke") => runtime_memory_smoke(env::args().nth(2)),
        Some("runtime-stream-smoke") => runtime_stream_smoke(env::args().nth(2)),
        Some("runtime-copy-smoke") => runtime_copy_smoke(env::args().nth(2)),
        Some("runtime-rmsnorm-smoke") => runtime_rmsnorm_smoke(env::args().nth(2)),
        Some("runtime-silu-mul-smoke") => runtime_silu_mul_smoke(env::args().nth(2)),
        Some("runtime-mlp-smoke") => runtime_mlp_smoke(env::args().nth(2)),
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
        Some("package-materialize-smoke") => package_materialize_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
        ),
        Some("package-mlp-smoke") => package_mlp_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
        ),
        Some("package-materialize-matvec-smoke") => package_materialize_matvec_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
        ),
        Some("package-rmsnorm-smoke") => package_rmsnorm_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
        ),
        Some("package-rmsnorm-mlp-smoke") => package_rmsnorm_mlp_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
        ),
        Some("package-linear-attn-proj-smoke") => package_linear_attn_proj_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
        ),
        Some("package-materialize-bench") => package_materialize_bench(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
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

fn runtime_rmsnorm_smoke(device_index: Option<String>) -> ExitCode {
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

    let elements = 4_usize;
    let epsilon = 1e-5_f32;
    let input = [1.0_f32, 2.0, -3.0, 4.0];
    let weight = [0.5_f32, 1.0, 1.5, -2.0];
    let expected = {
        let mean_square = input.iter().map(|value| value * value).sum::<f32>() / elements as f32;
        let inv_rms = 1.0_f32 / (mean_square + epsilon).sqrt();
        input
            .iter()
            .zip(weight.iter())
            .map(|(input_value, weight_value)| input_value * inv_rms * weight_value)
            .collect::<Vec<_>>()
    };

    let mut input_bytes = Vec::with_capacity(elements * std::mem::size_of::<f32>());
    let mut weight_bytes = Vec::with_capacity(elements * std::mem::size_of::<f32>());
    for value in &input {
        input_bytes.extend_from_slice(&value.to_le_bytes());
    }
    for value in &weight {
        weight_bytes.extend_from_slice(&value.to_le_bytes());
    }
    let output_bytes = elements * std::mem::size_of::<f32>();

    let mut input_buffer = match context.alloc_buffer(input_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate input runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = input_buffer.copy_from_host(0, &input_bytes, Some(&mut stream)) {
        eprintln!("failed to copy input data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after input copy: {err}");
        return ExitCode::from(1);
    }

    let mut weight_buffer = match context.alloc_buffer(weight_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate weight runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = weight_buffer.copy_from_host(0, &weight_bytes, Some(&mut stream)) {
        eprintln!("failed to copy weight data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after weight copy: {err}");
        return ExitCode::from(1);
    }

    let mut output_buffer = match context.alloc_buffer(output_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate output runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };

    if let Err(err) = ullm_runtime_sys::rmsnorm_f32(
        &input_buffer,
        &weight_buffer,
        elements,
        epsilon,
        &mut output_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run runtime rmsnorm_f32: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after rmsnorm: {err}");
        return ExitCode::from(1);
    }

    let mut output_raw = vec![0_u8; output_bytes];
    if let Err(err) = output_buffer.copy_to_host(0, &mut output_raw, Some(&mut stream)) {
        eprintln!("failed to copy rmsnorm result back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after output copy: {err}");
        return ExitCode::from(1);
    }
    let output = decode_f32_le_values(&output_raw);

    if output.len() != expected.len()
        || expected
            .iter()
            .zip(output.iter())
            .any(|(lhs, rhs)| (*lhs - *rhs).abs() > 1e-5_f32)
    {
        eprintln!(
            "runtime rmsnorm smoke produced unexpected output: output={:?} expected={:?}",
            output, expected
        );
        return ExitCode::from(1);
    }
    println!(
        "runtime-rmsnorm-smoke backend={} device_index={} name=\"{}\" elements={} epsilon={} output={} verified=true",
        info.backend,
        device_index,
        info.name,
        elements,
        epsilon,
        format_f32_preview(&output)
    );
    ExitCode::SUCCESS
}

fn runtime_silu_mul_smoke(device_index: Option<String>) -> ExitCode {
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

    let elements = 4_usize;
    let epsilon = 1e-5_f32;
    let gate = [-1.0_f32, 0.0_f32, 1.0_f32, 2.0_f32];
    let up = [3.0_f32, -4.0_f32, 5.0_f32, 6.0_f32];
    let expected = gate
        .iter()
        .zip(up.iter())
        .map(|(gate_value, up_value)| {
            let gate_value = *gate_value;
            gate_value * (1.0_f32 / (1.0_f32 + (-gate_value).exp())) * *up_value
        })
        .collect::<Vec<_>>();

    let mut gate_bytes = Vec::with_capacity(elements * std::mem::size_of::<f32>());
    let mut up_bytes = Vec::with_capacity(elements * std::mem::size_of::<f32>());
    for value in &gate {
        gate_bytes.extend_from_slice(&value.to_le_bytes());
    }
    for value in &up {
        up_bytes.extend_from_slice(&value.to_le_bytes());
    }
    let output_bytes = elements * std::mem::size_of::<f32>();

    let mut gate_buffer = match context.alloc_buffer(gate_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate gate runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = gate_buffer.copy_from_host(0, &gate_bytes, Some(&mut stream)) {
        eprintln!("failed to copy gate data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after gate copy: {err}");
        return ExitCode::from(1);
    }

    let mut up_buffer = match context.alloc_buffer(up_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate up runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = up_buffer.copy_from_host(0, &up_bytes, Some(&mut stream)) {
        eprintln!("failed to copy up data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after up copy: {err}");
        return ExitCode::from(1);
    }

    let mut output_buffer = match context.alloc_buffer(output_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate output runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };

    if let Err(err) = ullm_runtime_sys::silu_mul_f32(
        &gate_buffer,
        &up_buffer,
        elements,
        &mut output_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run runtime silu_mul_f32: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after silu_mul: {err}");
        return ExitCode::from(1);
    }

    let mut output_raw = vec![0_u8; output_bytes];
    if let Err(err) = output_buffer.copy_to_host(0, &mut output_raw, Some(&mut stream)) {
        eprintln!("failed to copy silu_mul result back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after output copy: {err}");
        return ExitCode::from(1);
    }
    let output = decode_f32_le_values(&output_raw);

    if output.len() != expected.len()
        || expected
            .iter()
            .zip(output.iter())
            .any(|(lhs, rhs)| (*lhs - *rhs).abs() > epsilon)
    {
        eprintln!(
            "runtime silu mul smoke produced unexpected output: output={:?} expected={:?}",
            output, expected
        );
        return ExitCode::from(1);
    }
    println!(
        "runtime-silu-mul-smoke backend={} device_index={} name=\"{}\" elements={} output={} verified=true",
        info.backend,
        device_index,
        info.name,
        elements,
        format_f32_preview(&output)
    );
    ExitCode::SUCCESS
}

fn runtime_mlp_smoke(device_index: Option<String>) -> ExitCode {
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

    const HIDDEN: usize = 4;
    const INTERMEDIATE: usize = 6;
    let epsilon = 1e-5_f32;

    let input = [0.45_f32, -1.20_f32, 0.95_f32, -0.35_f32];
    let norm_weight = [1.10_f32, -0.75_f32, 0.90_f32, 1.25_f32];
    let gate_matrix = [
        0.25_f32, -0.40_f32, 0.55_f32, 0.33_f32, //
        -0.60_f32, 0.80_f32, 0.45_f32, -0.70_f32, //
        1.10_f32, 0.20_f32, -0.30_f32, 0.45_f32, //
        0.65_f32, -0.55_f32, 0.85_f32, -0.10_f32, //
        -0.20_f32, 0.33_f32, 0.77_f32, -0.91_f32, //
        0.44_f32, -0.88_f32, 0.12_f32, 0.56_f32, //
    ];
    let up_matrix = [
        -0.30_f32, 0.70_f32, 0.90_f32, -0.50_f32, //
        1.05_f32, -0.95_f32, 0.25_f32, 0.60_f32, //
        0.20_f32, -0.15_f32, 0.40_f32, 1.10_f32, //
        -0.80_f32, 0.65_f32, 0.55_f32, -0.45_f32, //
        0.30_f32, 0.30_f32, 0.30_f32, 0.30_f32, //
        -0.25_f32, 1.20_f32, -1.10_f32, 0.45_f32, //
    ];
    let down_matrix = [
        0.50_f32, -0.30_f32, 0.70_f32, -0.60_f32, 0.40_f32, 0.20_f32, //
        0.10_f32, 0.90_f32, -0.40_f32, 0.80_f32, -0.15_f32, 0.60_f32, //
        -0.70_f32, 0.65_f32, 0.20_f32, 0.25_f32, 1.05_f32, -0.80_f32, //
        0.45_f32, -0.10_f32, -0.55_f32, 0.30_f32, 0.50_f32, 0.85_f32, //
    ];

    let expected_normed = runtime_host_rmsnorm_f32(&input, &norm_weight, epsilon);
    let expected_gate =
        runtime_host_matvec_f32(&gate_matrix, &expected_normed, INTERMEDIATE, HIDDEN);
    let expected_up = runtime_host_matvec_f32(&up_matrix, &expected_normed, INTERMEDIATE, HIDDEN);
    let expected_activated = runtime_host_silu_mul_f32(&expected_gate, &expected_up);
    let expected_output =
        runtime_host_matvec_f32(&down_matrix, &expected_activated, HIDDEN, INTERMEDIATE);

    let hidden_bytes = HIDDEN * std::mem::size_of::<f32>();
    let intermediate_bytes = INTERMEDIATE * std::mem::size_of::<f32>();
    let gate_matrix_byte_count = gate_matrix.len().checked_mul(std::mem::size_of::<f32>());
    if gate_matrix_byte_count.is_none() {
        eprintln!("gate matrix byte size overflows");
        return ExitCode::from(1);
    }
    let up_matrix_byte_count = up_matrix.len().checked_mul(std::mem::size_of::<f32>());
    if up_matrix_byte_count.is_none() {
        eprintln!("up matrix byte size overflows");
        return ExitCode::from(1);
    }
    let down_matrix_byte_count = down_matrix.len().checked_mul(std::mem::size_of::<f32>());
    if down_matrix_byte_count.is_none() {
        eprintln!("down matrix byte size overflows");
        return ExitCode::from(1);
    }
    let gate_matrix_bytes = gate_matrix_byte_count.unwrap();
    let up_matrix_bytes = up_matrix_byte_count.unwrap();
    let down_matrix_bytes = down_matrix_byte_count.unwrap();

    let mut input_buffer = match context.alloc_buffer(input.len() * std::mem::size_of::<f32>()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate input buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let input_bytes = encode_f32_to_bytes(&input);
    if let Err(err) = input_buffer.copy_from_host(0, &input_bytes, Some(&mut stream)) {
        eprintln!("failed to copy input data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after input copy: {err}");
        return ExitCode::from(1);
    }

    let mut norm_weight_buffer =
        match context.alloc_buffer(norm_weight.len() * std::mem::size_of::<f32>()) {
            Ok(buffer) => buffer,
            Err(err) => {
                eprintln!("failed to allocate norm weight buffer: {err}");
                return ExitCode::from(1);
            }
        };
    let norm_weight_bytes = encode_f32_to_bytes(&norm_weight);
    if let Err(err) = norm_weight_buffer.copy_from_host(0, &norm_weight_bytes, Some(&mut stream)) {
        eprintln!("failed to copy norm weight data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after norm weight copy: {err}");
        return ExitCode::from(1);
    }

    let mut gate_matrix_buffer = match context.alloc_buffer(gate_matrix_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate gate matrix buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let gate_matrix_bytes = encode_f32_to_bytes(&gate_matrix);
    if let Err(err) = gate_matrix_buffer.copy_from_host(0, &gate_matrix_bytes, Some(&mut stream)) {
        eprintln!("failed to copy gate matrix into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after gate matrix copy: {err}");
        return ExitCode::from(1);
    }

    let mut up_matrix_buffer = match context.alloc_buffer(up_matrix_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate up matrix buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let up_matrix_bytes = encode_f32_to_bytes(&up_matrix);
    if let Err(err) = up_matrix_buffer.copy_from_host(0, &up_matrix_bytes, Some(&mut stream)) {
        eprintln!("failed to copy up matrix into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after up matrix copy: {err}");
        return ExitCode::from(1);
    }

    let mut down_matrix_buffer = match context.alloc_buffer(down_matrix_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate down matrix buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let down_matrix_bytes = encode_f32_to_bytes(&down_matrix);
    if let Err(err) = down_matrix_buffer.copy_from_host(0, &down_matrix_bytes, Some(&mut stream)) {
        eprintln!("failed to copy down matrix into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after down matrix copy: {err}");
        return ExitCode::from(1);
    }

    let mut normed_buffer = match context.alloc_buffer(hidden_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate normed output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = ullm_runtime_sys::rmsnorm_f32(
        &input_buffer,
        &norm_weight_buffer,
        HIDDEN,
        epsilon,
        &mut normed_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run runtime rmsnorm_f32: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after rmsnorm: {err}");
        return ExitCode::from(1);
    }

    let mut gate_buffer = match context.alloc_buffer(intermediate_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate gate output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = ullm_runtime_sys::matvec_f32(
        &gate_matrix_buffer,
        &normed_buffer,
        INTERMEDIATE,
        HIDDEN,
        &mut gate_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run runtime gate matvec: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after gate matvec: {err}");
        return ExitCode::from(1);
    }

    let mut up_buffer = match context.alloc_buffer(intermediate_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate up output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = ullm_runtime_sys::matvec_f32(
        &up_matrix_buffer,
        &normed_buffer,
        INTERMEDIATE,
        HIDDEN,
        &mut up_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run runtime up matvec: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after up matvec: {err}");
        return ExitCode::from(1);
    }

    let mut activated_buffer = match context.alloc_buffer(intermediate_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate activated output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = ullm_runtime_sys::silu_mul_f32(
        &gate_buffer,
        &up_buffer,
        INTERMEDIATE,
        &mut activated_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run runtime silu_mul_f32: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after silu_mul: {err}");
        return ExitCode::from(1);
    }

    let mut output_buffer = match context.alloc_buffer(hidden_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = ullm_runtime_sys::matvec_f32(
        &down_matrix_buffer,
        &activated_buffer,
        HIDDEN,
        INTERMEDIATE,
        &mut output_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run runtime down matvec: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after output matvec: {err}");
        return ExitCode::from(1);
    }

    let mut output_bytes = vec![0_u8; hidden_bytes];
    if let Err(err) = output_buffer.copy_to_host(0, &mut output_bytes, Some(&mut stream)) {
        eprintln!("failed to copy runtime output back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after output copy: {err}");
        return ExitCode::from(1);
    }
    let output = decode_f32_le_values(&output_bytes);

    if output.len() != expected_output.len()
        || expected_output
            .iter()
            .zip(output.iter())
            .any(|(expected, actual)| (*expected - *actual).abs() > 1e-4_f32)
    {
        eprintln!(
            "runtime mlp smoke produced unexpected output: output={:?} expected={:?}",
            output, expected_output
        );
        return ExitCode::from(1);
    }

    println!(
        "runtime-mlp-smoke backend={} device_index={} name=\"{}\" hidden={} intermediate={} output={} verified=true",
        info.backend,
        device_index,
        info.name,
        HIDDEN,
        INTERMEDIATE,
        format_f32_preview(&output)
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
    let loaded_package = match load_package_tensor_prefix(
        &mut context,
        &mut stream,
        &path,
        max_tensors,
        LoadOptions {
            chunk_bytes,
            verify: true,
        },
    ) {
        Ok(loaded) => loaded,
        Err(err) => {
            eprintln!("failed to register package tensor payloads: {err}");
            return ExitCode::from(1);
        }
    };
    let registry = loaded_package.registry();

    println!(
        "package-weight-register-many-smoke package={} selected_tensors={} package_tensors={} registry_tensors={} registry_payload_bytes={} resident_payload_bytes={} codebook_payloads={} backend={} device_index={} name=\"{}\" chunk_bytes={} verified=true",
        path,
        loaded_package.loaded_tensor_count,
        loaded_package.summary.quantized_tensors,
        registry.len(),
        registry.total_payload_bytes(),
        registry.resident_payload_bytes(),
        registry.codebook_payloads(),
        info.backend,
        device_index,
        info.name,
        chunk_bytes
    );
    for &registry_index in &loaded_package.registry_indices {
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

fn package_materialize_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    tensor_selector: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-materialize-smoke requires a .ullm.d path");
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

    let materialize = match materialize_config(loaded) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let mut output = match context.alloc_buffer(materialize.output_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate materialized output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = ullm_runtime_sys::aq4_dequant_f32(
        loaded.index.buffer.as_ref(),
        loaded.scale.buffer.as_ref(),
        loaded.codebook.buffer.as_ref(),
        &materialize.scale_values,
        materialize.group_size,
        materialize.tensor_scale,
        materialize.elements,
        &mut output,
        Some(&mut stream),
    ) {
        eprintln!("failed to materialize AQ4 tensor: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after materialize: {err}");
        return ExitCode::from(1);
    }

    let preview_count = materialize.elements.min(8);
    let mut preview_bytes = vec![0_u8; preview_count * std::mem::size_of::<f32>()];
    if let Err(err) = output.copy_to_host(0, &mut preview_bytes, Some(&mut stream)) {
        eprintln!("failed to copy materialized preview back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after preview copy: {err}");
        return ExitCode::from(1);
    }
    let preview = decode_f32_le_values(&preview_bytes);
    println!(
        "package-materialize-smoke package={} registry_index={} tensor_index={} tensor=\"{}\" elements={} output_bytes={} scale_format={} scale_count={} group_size={} tensor_scale={:.9} backend={} device_index={} name=\"{}\" preview={} verified=true",
        path,
        registry_index,
        loaded.tensor_index,
        loaded.tensor_name,
        materialize.elements,
        materialize.output_bytes,
        materialize.scale_format,
        materialize.scale_values.len(),
        materialize.group_size,
        materialize.tensor_scale,
        info.backend,
        device_index,
        info.name,
        format_f32_preview(&preview)
    );
    ExitCode::SUCCESS
}

fn package_materialize_bench(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    tensor_selector: Option<String>,
    repeats: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-materialize-bench requires a .ullm.d path");
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
    let repeats = match parse_optional_usize(repeats, 20, "repeats") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("repeats must be greater than zero");
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
    let materialize = match materialize_config(loaded) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let mut output = match context.alloc_buffer(materialize.output_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate materialized output buffer: {err}");
            return ExitCode::from(1);
        }
    };

    if let Err(err) = ullm_runtime_sys::aq4_dequant_f32(
        loaded.index.buffer.as_ref(),
        loaded.scale.buffer.as_ref(),
        loaded.codebook.buffer.as_ref(),
        &materialize.scale_values,
        materialize.group_size,
        materialize.tensor_scale,
        materialize.elements,
        &mut output,
        Some(&mut stream),
    ) {
        eprintln!("failed to warm up AQ4 materialize: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize warmup materialize: {err}");
        return ExitCode::from(1);
    }

    let mut elapsed_ms = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let start = Instant::now();
        if let Err(err) = ullm_runtime_sys::aq4_dequant_f32(
            loaded.index.buffer.as_ref(),
            loaded.scale.buffer.as_ref(),
            loaded.codebook.buffer.as_ref(),
            &materialize.scale_values,
            materialize.group_size,
            materialize.tensor_scale,
            materialize.elements,
            &mut output,
            Some(&mut stream),
        ) {
            eprintln!("failed to materialize AQ4 tensor during benchmark: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!("failed to synchronize materialize benchmark: {err}");
            return ExitCode::from(1);
        }
        elapsed_ms.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    elapsed_ms.sort_by(|left, right| left.total_cmp(right));
    let mean_ms = elapsed_ms.iter().sum::<f64>() / elapsed_ms.len() as f64;
    let min_ms = elapsed_ms[0];
    let p50_ms = elapsed_ms[elapsed_ms.len() / 2];
    let p95_index = ((elapsed_ms.len() - 1) * 95) / 100;
    let p95_ms = elapsed_ms[p95_index];
    let output_gib = materialize.output_bytes as f64 / 1024.0 / 1024.0 / 1024.0;
    let output_gib_per_s = if mean_ms > 0.0 {
        output_gib / (mean_ms / 1000.0)
    } else {
        0.0
    };
    println!(
        "package-materialize-bench package={} registry_index={} tensor_index={} tensor=\"{}\" elements={} output_bytes={} scale_format={} scale_count={} group_size={} tensor_scale={:.9} backend={} device_index={} name=\"{}\" repeats={} mean_ms={:.6} min_ms={:.6} p50_ms={:.6} p95_ms={:.6} output_gib_per_s={:.6} verified=true",
        path,
        registry_index,
        loaded.tensor_index,
        loaded.tensor_name,
        materialize.elements,
        materialize.output_bytes,
        materialize.scale_format,
        materialize.scale_values.len(),
        materialize.group_size,
        materialize.tensor_scale,
        info.backend,
        device_index,
        info.name,
        repeats,
        mean_ms,
        min_ms,
        p50_ms,
        p95_ms,
        output_gib_per_s
    );
    ExitCode::SUCCESS
}

fn package_materialize_matvec_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    tensor_selector: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-materialize-matvec-smoke requires a .ullm.d path");
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
    let materialize = match materialize_config(loaded) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let (rows, cols) = match matrix_shape_rows_cols(&loaded.shape, materialize.elements) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let mut matrix = match context.alloc_buffer(materialize.output_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate materialized matrix buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = ullm_runtime_sys::aq4_dequant_f32(
        loaded.index.buffer.as_ref(),
        loaded.scale.buffer.as_ref(),
        loaded.codebook.buffer.as_ref(),
        &materialize.scale_values,
        materialize.group_size,
        materialize.tensor_scale,
        materialize.elements,
        &mut matrix,
        Some(&mut stream),
    ) {
        eprintln!("failed to materialize AQ4 tensor: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after materialize: {err}");
        return ExitCode::from(1);
    }

    let mut input = Vec::with_capacity(cols);
    for i in 0..cols {
        input.push(((i % 17) as f32 - 8.0) / 16.0);
    }
    let input_byte_count = match cols.checked_mul(std::mem::size_of::<f32>()) {
        Some(value) => value,
        None => {
            eprintln!("matvec input byte size overflows");
            return ExitCode::from(1);
        }
    };
    let output_byte_count = match rows.checked_mul(std::mem::size_of::<f32>()) {
        Some(value) => value,
        None => {
            eprintln!("matvec output byte size overflows");
            return ExitCode::from(1);
        }
    };
    let mut input_bytes = Vec::with_capacity(input_byte_count);
    for value in &input {
        input_bytes.extend_from_slice(&value.to_le_bytes());
    }
    let mut input_buffer = match context.alloc_buffer(input_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate input buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = input_buffer.copy_from_host(0, &input_bytes, Some(&mut stream)) {
        eprintln!("failed to copy input vector into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after input copy: {err}");
        return ExitCode::from(1);
    }

    let mut output = match context.alloc_buffer(output_byte_count) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = ullm_runtime_sys::matvec_f32(
        &matrix,
        &input_buffer,
        rows,
        cols,
        &mut output,
        Some(&mut stream),
    ) {
        eprintln!("failed to run matvec f32: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after matvec: {err}");
        return ExitCode::from(1);
    }

    let preview_count = rows.min(8);
    let mut preview_bytes = vec![0_u8; preview_count * std::mem::size_of::<f32>()];
    if let Err(err) = output.copy_to_host(0, &mut preview_bytes, Some(&mut stream)) {
        eprintln!("failed to copy matvec preview back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after preview copy: {err}");
        return ExitCode::from(1);
    }
    let preview = decode_f32_le_values(&preview_bytes);
    println!(
        "package-materialize-matvec-smoke package={} registry_index={} tensor_index={} tensor=\"{}\" elements={} rows={} cols={} output_bytes={} scale_format={} scale_count={} group_size={} tensor_scale={:.9} backend={} device_index={} name=\"{}\" preview={} verified=true",
        path,
        registry_index,
        loaded.tensor_index,
        loaded.tensor_name,
        materialize.elements,
        rows,
        cols,
        output_byte_count,
        materialize.scale_format,
        materialize.scale_values.len(),
        materialize.group_size,
        materialize.tensor_scale,
        info.backend,
        device_index,
        info.name,
        format_f32_preview(&preview)
    );
    ExitCode::SUCCESS
}

fn package_rmsnorm_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    norm_kind: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-rmsnorm-smoke requires a .ullm.d path");
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
    let layer_index = match parse_optional_usize(layer_index, 0, "layer index") {
        Ok(value) => value,
        Err(code) => return code,
    };
    let norm_kind = match normalize_norm_kind(norm_kind.as_deref()) {
        Ok(value) => value,
        Err(code) => return code,
    };

    let tensor_name = match norm_kind {
        NormKind::Input => {
            format!("model.language_model.layers.{layer_index}.input_layernorm.weight")
        }
        NormKind::Post => {
            format!("model.language_model.layers.{layer_index}.post_attention_layernorm.weight")
        }
    };
    let selector = TensorSelector::Name(tensor_name.clone());
    let bundle = match ullm_engine::package::select_passthrough_payload_bundle(&path, &selector) {
        Ok(bundle) => bundle,
        Err(err) => {
            eprintln!("failed to select package passthrough tensor: {err}");
            return ExitCode::from(1);
        }
    };
    let elements = match usize::try_from(bundle.elements) {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("passthrough tensor has zero elements");
            return ExitCode::from(1);
        }
        Err(_) => {
            eprintln!("passthrough tensor element count is too large for this host");
            return ExitCode::from(1);
        }
    };
    let dtype = match resolve_passthrough_dtype(&bundle, &tensor_name) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let payload = match read_passthrough_payload_f32_bytes(&bundle, chunk_bytes, dtype) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to read passthrough payload for {tensor_name}: {err}");
            return ExitCode::from(1);
        }
    };
    if payload.len() != elements {
        eprintln!(
            "passthrough tensor element count mismatch for {tensor_name}: expected {} got {}",
            elements,
            payload.len()
        );
        return ExitCode::from(1);
    }

    let input = deterministic_f32_vector(elements);
    let epsilon = 1e-5_f32;
    let expected = runtime_host_rmsnorm_f32(&input, &payload, epsilon);
    if expected.len() != elements {
        eprintln!("failed to build deterministic RMSNorm reference");
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

    let input_bytes = encode_f32_to_bytes(&input);
    let weight_bytes = encode_f32_to_bytes(&payload);
    let mut input_buffer = match context.alloc_buffer(input_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate RMSNorm input buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = input_buffer.copy_from_host(0, &input_bytes, Some(&mut stream)) {
        eprintln!("failed to copy deterministic input into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after input copy: {err}");
        return ExitCode::from(1);
    }

    let mut weight_buffer = match context.alloc_buffer(weight_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate RMSNorm weight buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = weight_buffer.copy_from_host(0, &weight_bytes, Some(&mut stream)) {
        eprintln!("failed to copy RMSNorm weight into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after weight copy: {err}");
        return ExitCode::from(1);
    }

    let mut output_buffer = match context.alloc_buffer(weight_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate RMSNorm output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = ullm_runtime_sys::rmsnorm_f32(
        &input_buffer,
        &weight_buffer,
        elements,
        epsilon,
        &mut output_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run runtime rmsnorm_f32: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after rmsnorm: {err}");
        return ExitCode::from(1);
    }

    let mut output_raw = vec![0_u8; weight_bytes.len()];
    if let Err(err) = output_buffer.copy_to_host(0, &mut output_raw, Some(&mut stream)) {
        eprintln!("failed to copy runtime RMSNorm output back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after output copy: {err}");
        return ExitCode::from(1);
    }
    let output = decode_f32_le_values(&output_raw);

    if output.len() != expected.len() {
        eprintln!(
            "runtime RMSNorm output size mismatch: expected {} got {}",
            expected.len(),
            output.len()
        );
        return ExitCode::from(1);
    }

    let mut max_abs_diff = 0.0_f32;
    for (lhs, rhs) in output.iter().zip(expected.iter()) {
        let diff = (lhs - rhs).abs();
        if diff > 1e-4_f32 {
            eprintln!(
                "package-rmsnorm-smoke mismatch for tensor={tensor_name}: max_abs_diff={diff}"
            );
            return ExitCode::from(1);
        }
        if diff > max_abs_diff {
            max_abs_diff = diff;
        }
    }

    println!(
        "package-rmsnorm-smoke package={} tensor_index={} tensor=\"{}\" dtype={} elements={} shape_len={} payload_bytes={} payload_path={} device_index={} name=\"{}\" epsilon={} norm_kind={} chunk_bytes={} max_abs_diff={max_abs_diff:.9} preview={} verified=true",
        path,
        bundle.tensor_index,
        bundle.tensor_name,
        dtype,
        elements,
        bundle.shape.len(),
        bundle.payload_bytes,
        bundle.payload_file.relative_path,
        device_index,
        info.name,
        epsilon,
        norm_kind.as_str(),
        chunk_bytes,
        format_f32_preview(&output[..output.len().min(8)])
    );
    ExitCode::SUCCESS
}

fn package_mlp_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-mlp-smoke requires a .ullm.d path");
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
    let layer_index = match parse_optional_usize(layer_index, 0, "layer index") {
        Ok(value) => value,
        Err(code) => return code,
    };

    let gate_tensor = format!("model.language_model.layers.{layer_index}.mlp.gate_proj.weight");
    let up_tensor = format!("model.language_model.layers.{layer_index}.mlp.up_proj.weight");
    let down_tensor = format!("model.language_model.layers.{layer_index}.mlp.down_proj.weight");

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
    let (gate_rows, gate_cols, gate_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &gate_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize gate tensor: {err}");
            return ExitCode::from(1);
        }
    };
    let (up_rows, up_cols, up_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &up_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize up tensor: {err}");
            return ExitCode::from(1);
        }
    };
    let (down_rows, down_cols, down_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &down_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize down tensor: {err}");
            return ExitCode::from(1);
        }
    };

    if gate_rows != up_rows || gate_cols != up_cols {
        eprintln!("gate and up tensor shapes must match");
        return ExitCode::from(1);
    }
    if down_cols != up_rows {
        eprintln!(
            "down tensor shape mismatch: expected cols={} but got {}",
            up_rows, down_cols
        );
        return ExitCode::from(1);
    }
    if down_rows != up_cols {
        eprintln!(
            "down tensor shape mismatch: expected rows={} but got {}",
            up_cols, down_rows
        );
        return ExitCode::from(1);
    }

    let intermediate = gate_rows;
    let hidden = gate_cols;

    let mut input = Vec::with_capacity(hidden);
    for i in 0..hidden {
        input.push((i % 23) as f32 / 16.0 - 11.0 / 16.0);
    }
    let input_bytes = encode_f32_to_bytes(&input);
    let mut input_buffer = match context.alloc_buffer(input_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate input buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = input_buffer.copy_from_host(0, &input_bytes, Some(&mut stream)) {
        eprintln!("failed to copy input vector into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after input copy: {err}");
        return ExitCode::from(1);
    }

    let intermediate_bytes = match intermediate.checked_mul(std::mem::size_of::<f32>()) {
        Some(value) => value,
        None => {
            eprintln!("intermediate byte size overflows");
            return ExitCode::from(1);
        }
    };

    let mut gate_buffer = match context.alloc_buffer(intermediate_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate gate output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = ullm_runtime_sys::matvec_f32(
        &gate_matrix,
        &input_buffer,
        gate_rows,
        gate_cols,
        &mut gate_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run gate matvec: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after gate matvec: {err}");
        return ExitCode::from(1);
    }

    let mut up_buffer = match context.alloc_buffer(intermediate_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate up output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = ullm_runtime_sys::matvec_f32(
        &up_matrix,
        &input_buffer,
        up_rows,
        up_cols,
        &mut up_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run up matvec: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after up matvec: {err}");
        return ExitCode::from(1);
    }

    let mut activated_buffer = match context.alloc_buffer(intermediate_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate activated output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = ullm_runtime_sys::silu_mul_f32(
        &gate_buffer,
        &up_buffer,
        intermediate,
        &mut activated_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run silu_mul: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after silu_mul: {err}");
        return ExitCode::from(1);
    }

    let hidden_bytes = match hidden.checked_mul(std::mem::size_of::<f32>()) {
        Some(value) => value,
        None => {
            eprintln!("output byte size overflows");
            return ExitCode::from(1);
        }
    };
    let mut output_buffer = match context.alloc_buffer(hidden_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = ullm_runtime_sys::matvec_f32(
        &down_matrix,
        &activated_buffer,
        down_rows,
        down_cols,
        &mut output_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run down matvec: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after output matvec: {err}");
        return ExitCode::from(1);
    }

    let preview_count = hidden.min(8);
    let mut output_preview_bytes = vec![0_u8; preview_count * std::mem::size_of::<f32>()];
    if let Err(err) = output_buffer.copy_to_host(0, &mut output_preview_bytes, Some(&mut stream)) {
        eprintln!("failed to copy output preview to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after preview copy: {err}");
        return ExitCode::from(1);
    }
    let preview = decode_f32_le_values(&output_preview_bytes);

    println!(
        "package-mlp-smoke package={} layer={} gate_tensor=\"{}\" up_tensor=\"{}\" down_tensor=\"{}\" hidden={} intermediate={} backend={} device_index={} name=\"{}\" preview={} verified=true",
        path,
        layer_index,
        gate_tensor,
        up_tensor,
        down_tensor,
        hidden,
        intermediate,
        info.backend,
        device_index,
        info.name,
        format_f32_preview(&preview)
    );
    ExitCode::SUCCESS
}

fn package_rmsnorm_mlp_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    norm_kind: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-rmsnorm-mlp-smoke requires a .ullm.d path");
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
    let layer_index = match parse_optional_usize(layer_index, 0, "layer index") {
        Ok(value) => value,
        Err(code) => return code,
    };
    let norm_kind = match normalize_norm_kind(Some(norm_kind.as_deref().unwrap_or("post"))) {
        Ok(value) => value,
        Err(code) => return code,
    };

    let norm_tensor = match norm_kind {
        NormKind::Input => {
            format!("model.language_model.layers.{layer_index}.input_layernorm.weight")
        }
        NormKind::Post => {
            format!("model.language_model.layers.{layer_index}.post_attention_layernorm.weight")
        }
    };
    let gate_tensor = format!("model.language_model.layers.{layer_index}.mlp.gate_proj.weight");
    let up_tensor = format!("model.language_model.layers.{layer_index}.mlp.up_proj.weight");
    let down_tensor = format!("model.language_model.layers.{layer_index}.mlp.down_proj.weight");

    let selector = TensorSelector::Name(norm_tensor.clone());
    let norm_bundle =
        match ullm_engine::package::select_passthrough_payload_bundle(&path, &selector) {
            Ok(bundle) => bundle,
            Err(err) => {
                eprintln!("failed to select package passthrough tensor: {err}");
                return ExitCode::from(1);
            }
        };
    let norm_elements = match usize::try_from(norm_bundle.elements) {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("RMSNorm tensor has zero elements");
            return ExitCode::from(1);
        }
        Err(_) => {
            eprintln!("RMSNorm tensor element count is too large for this host");
            return ExitCode::from(1);
        }
    };
    let dtype = match resolve_passthrough_dtype(&norm_bundle, &norm_tensor) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let norm_weight = match read_passthrough_payload_f32_bytes(&norm_bundle, chunk_bytes, dtype) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to read passthrough payload for {norm_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    if norm_weight.len() != norm_elements {
        eprintln!(
            "passthrough tensor element count mismatch for {norm_tensor}: expected {} got {}",
            norm_elements,
            norm_weight.len()
        );
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

    let mut registry = WeightRegistry::new();
    let (gate_rows, gate_cols, gate_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &gate_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize gate tensor: {err}");
            return ExitCode::from(1);
        }
    };
    let (up_rows, up_cols, up_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &up_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize up tensor: {err}");
            return ExitCode::from(1);
        }
    };
    let (down_rows, down_cols, down_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &down_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize down tensor: {err}");
            return ExitCode::from(1);
        }
    };

    if gate_rows != up_rows || gate_cols != up_cols {
        eprintln!(
            "gate and up tensor shapes must match: gate=({gate_rows}, {gate_cols}), up=({up_rows}, {up_cols})"
        );
        return ExitCode::from(1);
    }
    if down_rows != up_cols || down_cols != gate_rows {
        eprintln!(
            "down tensor shape mismatch: expected shape ({}, {}) from gate/up, got ({}, {})",
            gate_cols, gate_rows, down_rows, down_cols
        );
        return ExitCode::from(1);
    }

    let hidden = gate_cols;
    let intermediate = gate_rows;
    if norm_elements != hidden {
        eprintln!(
            "RMSNorm element count must match MLP hidden dimension: norm={norm_elements}, hidden={hidden}"
        );
        return ExitCode::from(1);
    }

    let epsilon = 1e-5_f32;
    let input = deterministic_f32_vector(hidden);
    let input_bytes = encode_f32_to_bytes(&input);
    let weight_bytes = encode_f32_to_bytes(&norm_weight);

    let mut input_buffer = match context.alloc_buffer(input_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate RMSNorm input buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = input_buffer.copy_from_host(0, &input_bytes, Some(&mut stream)) {
        eprintln!("failed to copy deterministic RMSNorm input into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after RMSNorm input copy: {err}");
        return ExitCode::from(1);
    }

    let mut weight_buffer = match context.alloc_buffer(weight_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate RMSNorm weight buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = weight_buffer.copy_from_host(0, &weight_bytes, Some(&mut stream)) {
        eprintln!("failed to copy RMSNorm weight into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after RMSNorm weight copy: {err}");
        return ExitCode::from(1);
    }

    let mut normed_buffer = match context.alloc_buffer(weight_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate RMSNorm output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = ullm_runtime_sys::rmsnorm_f32(
        &input_buffer,
        &weight_buffer,
        hidden,
        epsilon,
        &mut normed_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run runtime rmsnorm_f32: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after rmsnorm: {err}");
        return ExitCode::from(1);
    }

    let intermediate_bytes = match intermediate.checked_mul(std::mem::size_of::<f32>()) {
        Some(value) => value,
        None => {
            eprintln!("intermediate byte size overflows");
            return ExitCode::from(1);
        }
    };
    let mut gate_buffer = match context.alloc_buffer(intermediate_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate gate output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = ullm_runtime_sys::matvec_f32(
        &gate_matrix,
        &normed_buffer,
        gate_rows,
        gate_cols,
        &mut gate_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run gate matvec: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after gate matvec: {err}");
        return ExitCode::from(1);
    }

    let mut up_buffer = match context.alloc_buffer(intermediate_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate up output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = ullm_runtime_sys::matvec_f32(
        &up_matrix,
        &normed_buffer,
        up_rows,
        up_cols,
        &mut up_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run up matvec: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after up matvec: {err}");
        return ExitCode::from(1);
    }

    let mut activated_buffer = match context.alloc_buffer(intermediate_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate activated output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = ullm_runtime_sys::silu_mul_f32(
        &gate_buffer,
        &up_buffer,
        intermediate,
        &mut activated_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run silu_mul: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after silu_mul: {err}");
        return ExitCode::from(1);
    }

    let hidden_bytes = match hidden.checked_mul(std::mem::size_of::<f32>()) {
        Some(value) => value,
        None => {
            eprintln!("output byte size overflows");
            return ExitCode::from(1);
        }
    };
    let mut output_buffer = match context.alloc_buffer(hidden_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = ullm_runtime_sys::matvec_f32(
        &down_matrix,
        &activated_buffer,
        down_rows,
        down_cols,
        &mut output_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run down matvec: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after output matvec: {err}");
        return ExitCode::from(1);
    }

    let preview_count = hidden.min(8);
    let mut output_preview_bytes = vec![0_u8; preview_count * std::mem::size_of::<f32>()];
    if let Err(err) = output_buffer.copy_to_host(0, &mut output_preview_bytes, Some(&mut stream)) {
        eprintln!("failed to copy output preview to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after output preview copy: {err}");
        return ExitCode::from(1);
    }
    let preview = decode_f32_le_values(&output_preview_bytes);

    println!(
        "package-rmsnorm-mlp-smoke package={} layer={} norm_kind={} norm_tensor=\"{}\" gate_tensor=\"{}\" up_tensor=\"{}\" down_tensor=\"{}\" hidden={} intermediate={} backend={} device_index={} name=\"{}\" preview={} verified=true",
        path,
        layer_index,
        norm_kind.as_str(),
        norm_tensor,
        gate_tensor,
        up_tensor,
        down_tensor,
        hidden,
        intermediate,
        info.backend,
        device_index,
        info.name,
        format_f32_preview(&preview)
    );
    ExitCode::SUCCESS
}

fn package_linear_attn_proj_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    projection: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-linear-attn-proj-smoke requires a .ullm.d path");
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
    let layer_index = match parse_optional_usize(layer_index, 0, "layer index") {
        Ok(value) => value,
        Err(code) => return code,
    };
    let projection = match parse_linear_attn_projection(projection.as_deref()) {
        Ok(value) => value,
        Err(code) => return code,
    };

    let requested_projections = match projection {
        LinearAttnProjection::A => {
            vec![(
                "a",
                format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_a.weight"),
            )]
        }
        LinearAttnProjection::B => {
            vec![(
                "b",
                format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_b.weight"),
            )]
        }
        LinearAttnProjection::Qkv => {
            vec![(
                "qkv",
                format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_qkv.weight"),
            )]
        }
        LinearAttnProjection::Z => {
            vec![(
                "z",
                format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_z.weight"),
            )]
        }
        LinearAttnProjection::Out => {
            vec![(
                "out",
                format!("model.language_model.layers.{layer_index}.linear_attn.out_proj.weight"),
            )]
        }
        LinearAttnProjection::All => vec![
            (
                "a",
                format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_a.weight"),
            ),
            (
                "b",
                format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_b.weight"),
            ),
            (
                "qkv",
                format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_qkv.weight"),
            ),
            (
                "z",
                format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_z.weight"),
            ),
            (
                "out",
                format!("model.language_model.layers.{layer_index}.linear_attn.out_proj.weight"),
            ),
        ],
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
    let mut input_buffer: Option<ullm_runtime_sys::RuntimeBuffer> = None;
    let mut hidden = None;
    for (projection_name, tensor_name) in requested_projections {
        let (rows, cols, matrix) = match materialize_selected_aq4_matrix(
            &mut context,
            &mut stream,
            &mut registry,
            &path,
            &tensor_name,
            chunk_bytes,
        ) {
            Ok(value) => value,
            Err(err) => {
                eprintln!("failed to materialize projection {projection_name}: {err}");
                return ExitCode::from(1);
            }
        };

        match hidden {
            Some(expected) if expected != cols => {
                eprintln!(
                    "projection {projection_name} tensor {tensor_name} has cols={cols}, expected hidden={expected}"
                );
                return ExitCode::from(1);
            }
            Some(_) => {}
            None => {
                hidden = Some(cols);
                let input = deterministic_f32_vector(cols);
                let input_bytes = encode_f32_to_bytes(&input);
                let mut buffer = match context.alloc_buffer(input_bytes.len()) {
                    Ok(buffer) => buffer,
                    Err(err) => {
                        eprintln!("failed to allocate shared input buffer: {err}");
                        return ExitCode::from(1);
                    }
                };
                if let Err(err) = buffer.copy_from_host(0, &input_bytes, Some(&mut stream)) {
                    eprintln!(
                        "failed to copy deterministic input data into shared runtime buffer: {err}"
                    );
                    return ExitCode::from(1);
                }
                if let Err(err) = stream.synchronize() {
                    eprintln!("failed to synchronize runtime stream after input copy: {err}");
                    return ExitCode::from(1);
                }
                input_buffer = Some(buffer);
            }
        }

        let Some(shared_input) = input_buffer.as_mut() else {
            eprintln!("shared runtime input buffer was not initialized");
            return ExitCode::from(1);
        };

        let output_bytes = match rows.checked_mul(std::mem::size_of::<f32>()) {
            Some(value) => value,
            None => {
                eprintln!("output byte size overflows for projection {projection_name}");
                return ExitCode::from(1);
            }
        };
        let mut output = match context.alloc_buffer(output_bytes) {
            Ok(buffer) => buffer,
            Err(err) => {
                eprintln!("failed to allocate output buffer for {projection_name}: {err}");
                return ExitCode::from(1);
            }
        };
        if let Err(err) = ullm_runtime_sys::matvec_f32(
            &matrix,
            shared_input,
            rows,
            cols,
            &mut output,
            Some(&mut stream),
        ) {
            eprintln!("failed to run matvec for projection {projection_name}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!("failed to synchronize runtime stream after matvec: {err}");
            return ExitCode::from(1);
        }

        let preview_count = rows.min(8);
        let mut output_preview_bytes = vec![0_u8; preview_count * std::mem::size_of::<f32>()];
        if let Err(err) = output.copy_to_host(0, &mut output_preview_bytes, Some(&mut stream)) {
            eprintln!("failed to copy matvec preview for {projection_name}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!("failed to synchronize runtime stream after preview copy: {err}");
            return ExitCode::from(1);
        }
        let preview = decode_f32_le_values(&output_preview_bytes);
        println!(
            "package-linear-attn-proj-smoke package={} layer={} projection={} tensor=\"{}\" hidden={} rows={} backend={} device_index={} name=\"{}\" preview={} verified=true",
            path,
            layer_index,
            projection_name,
            tensor_name,
            cols,
            rows,
            info.backend,
            device_index,
            info.name,
            format_f32_preview(&preview)
        );
    }
    ExitCode::SUCCESS
}

#[derive(Debug, Clone, Copy)]
enum NormKind {
    Input,
    Post,
}

#[derive(Debug, Clone, Copy)]
enum LinearAttnProjection {
    A,
    B,
    Qkv,
    Z,
    Out,
    All,
}

impl NormKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Input => "input",
            Self::Post => "post",
        }
    }
}

fn parse_linear_attn_projection(value: Option<&str>) -> Result<LinearAttnProjection, ExitCode> {
    match value.unwrap_or("all") {
        "a" => Ok(LinearAttnProjection::A),
        "b" => Ok(LinearAttnProjection::B),
        "qkv" => Ok(LinearAttnProjection::Qkv),
        "z" => Ok(LinearAttnProjection::Z),
        "out" => Ok(LinearAttnProjection::Out),
        "all" => Ok(LinearAttnProjection::All),
        value => {
            eprintln!("invalid projection: {value}; expected a, b, qkv, z, out, or all");
            Err(ExitCode::from(2))
        }
    }
}

fn normalize_norm_kind(kind: Option<&str>) -> Result<NormKind, ExitCode> {
    match kind.unwrap_or("input") {
        "input" => Ok(NormKind::Input),
        "post" => Ok(NormKind::Post),
        value => {
            eprintln!("invalid norm kind: {value}; expected input or post");
            Err(ExitCode::from(2))
        }
    }
}

fn resolve_passthrough_dtype<'a>(
    bundle: &'a PassthroughPayloadBundle,
    tensor_name: &str,
) -> Result<&'a str, String> {
    if let Some(dtype) = bundle.dtype.as_deref() {
        return match dtype {
            "BF16" | "F32" => Ok(dtype),
            _ => Err(format!(
                "unsupported passthrough dtype \"{dtype}\" for tensor {tensor_name}"
            )),
        };
    }

    let payload_bytes = if bundle.payload_bytes == 0 {
        bundle.payload_file.bytes
    } else {
        bundle.payload_bytes
    };
    let bf16_bytes = bundle.elements.checked_mul(2).ok_or_else(|| {
        format!("passthrough tensor {tensor_name} element count overflow while inferring dtype")
    })?;
    let f32_bytes = bundle.elements.checked_mul(4).ok_or_else(|| {
        format!("passthrough tensor {tensor_name} element count overflow while inferring dtype")
    })?;
    if payload_bytes == bf16_bytes {
        Ok("BF16")
    } else if payload_bytes == f32_bytes {
        Ok("F32")
    } else {
        Err(format!(
            "could not infer passthrough dtype for tensor {tensor_name}; declare dtype in manifest"
        ))
    }
}

fn read_passthrough_payload_f32_bytes(
    bundle: &PassthroughPayloadBundle,
    chunk_bytes: usize,
    dtype: &str,
) -> Result<Vec<f32>, String> {
    let mut file = File::open(&bundle.payload_file.absolute_path).map_err(|err| {
        format!(
            "failed to open {}: {err}",
            bundle.payload_file.absolute_path.display()
        )
    })?;
    let payload_bytes = if bundle.payload_bytes == 0 {
        bundle.payload_file.bytes
    } else {
        bundle.payload_bytes
    };
    if payload_bytes != bundle.payload_file.bytes {
        return Err(format!(
            "passthrough tensor {} payload bytes mismatch: declared {} actual {}",
            bundle.tensor_name, payload_bytes, bundle.payload_file.bytes
        ));
    }
    let element_size = match dtype {
        "BF16" => 2_usize,
        "F32" => 4_usize,
        _ => {
            return Err(format!(
                "unsupported passthrough dtype {dtype} for tensor {}",
                bundle.tensor_name
            ));
        }
    };
    if chunk_bytes == 0 {
        return Err("chunk bytes must be greater than zero".to_string());
    }
    let expected_bytes = usize::try_from(payload_bytes)
        .map_err(|_| "passthrough payload is too large for this host".to_string())?;
    let expected_elements = usize::try_from(bundle.elements)
        .map_err(|_| "payload element count too large".to_string())?;
    if !expected_bytes.is_multiple_of(element_size) {
        return Err(format!(
            "passthrough tensor {} payload is not aligned to {element_size}-byte elements",
            bundle.tensor_name
        ));
    }
    if expected_bytes / element_size != expected_elements {
        return Err(format!(
            "passthrough tensor {} payload has {} elements, expected {}",
            bundle.tensor_name,
            expected_bytes / element_size,
            expected_elements
        ));
    }

    let mut values = Vec::with_capacity(expected_elements);
    let mut scratch = vec![0_u8; chunk_bytes];
    let mut read_bytes = 0_usize;
    let mut carry = Vec::with_capacity(element_size - 1);
    let mut merge = Vec::with_capacity(chunk_bytes + element_size);
    loop {
        let read = file.read(&mut scratch).map_err(|err| {
            format!(
                "failed to read {}: {err}",
                bundle.payload_file.absolute_path.display()
            )
        })?;
        if read == 0 {
            break;
        }
        read_bytes = read_bytes.saturating_add(read);
        if read_bytes > expected_bytes {
            return Err(format!(
                "passthrough tensor {} payload is larger than declared bytes {}",
                bundle.tensor_name, expected_bytes
            ));
        }

        merge.clear();
        if carry.is_empty() {
            merge.extend_from_slice(&scratch[..read]);
        } else {
            merge.extend_from_slice(&carry);
            carry.clear();
            merge.extend_from_slice(&scratch[..read]);
        }

        let decode_end = (merge.len() / element_size) * element_size;
        for bytes in merge[..decode_end].chunks_exact(element_size) {
            let value = match dtype {
                "BF16" => {
                    let raw = u16::from_le_bytes([bytes[0], bytes[1]]);
                    f32::from_bits(u32::from(raw) << 16)
                }
                "F32" => {
                    let raw = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                    raw
                }
                _ => unreachable!(),
            };
            values.push(value);
        }
        if decode_end < merge.len() {
            carry.extend_from_slice(&merge[decode_end..]);
        }
    }
    if read_bytes != expected_bytes {
        return Err(format!(
            "passthrough tensor {} payload bytes mismatch while reading file: expected {} got {}",
            bundle.tensor_name, expected_bytes, read_bytes
        ));
    }
    if values.len() != expected_elements {
        return Err(format!(
            "passthrough tensor {} payload elements mismatch: expected {} got {}",
            bundle.tensor_name,
            expected_elements,
            values.len()
        ));
    }
    Ok(values)
}

fn deterministic_f32_vector(elements: usize) -> Vec<f32> {
    let mut values = Vec::with_capacity(elements);
    for index in 0..elements {
        values.push(((index as f32).sin() + 1.0_f32) / 2.0_f32);
    }
    values
}

fn materialize_selected_aq4_matrix(
    context: &mut ullm_runtime_sys::RuntimeContext,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    registry: &mut WeightRegistry,
    path: &str,
    tensor_name: &str,
    chunk_bytes: usize,
) -> Result<(usize, usize, ullm_runtime_sys::RuntimeBuffer), String> {
    let selector = TensorSelector::Name(tensor_name.to_string());
    let bundle = ullm_engine::package::select_tensor_payload_bundle(path, &selector)
        .map_err(|err| format!("failed to select tensor payloads for {tensor_name}: {err}"))?;
    let registry_index = registry
        .load_and_insert(
            context,
            stream,
            &bundle,
            LoadOptions {
                chunk_bytes,
                verify: true,
            },
        )
        .map_err(|err| format!("failed to register tensor payloads for {tensor_name}: {err}"))?;
    let loaded = registry
        .get(registry_index)
        .ok_or_else(|| "registered tensor disappeared from weight registry".to_string())?;
    let materialize = materialize_config(loaded).map_err(|err| {
        format!(
            "failed to prepare materialize config for {tensor_name} (registry index {registry_index}): {err}"
        )
    })?;
    let (rows, cols) = matrix_shape_rows_cols(&loaded.shape, materialize.elements)
        .map_err(|err| format!("invalid shape for {tensor_name}: {err}"))?;
    let mut output = context
        .alloc_buffer(materialize.output_bytes)
        .map_err(|err| {
            format!("failed to allocate materialized output for {tensor_name}: {err}")
        })?;
    if let Err(err) = ullm_runtime_sys::aq4_dequant_f32(
        loaded.index.buffer.as_ref(),
        loaded.scale.buffer.as_ref(),
        loaded.codebook.buffer.as_ref(),
        &materialize.scale_values,
        materialize.group_size,
        materialize.tensor_scale,
        materialize.elements,
        &mut output,
        Some(stream),
    ) {
        return Err(format!(
            "failed to materialize AQ4 tensor {tensor_name}: {err}"
        ));
    }
    stream.synchronize().map_err(|err| {
        format!("failed to synchronize runtime stream after materializing {tensor_name}: {err}")
    })?;
    Ok((rows, cols, output))
}

fn print_help() {
    eprintln!(
        "usage: ullm-engine <inspect-devices|runtime-smoke|runtime-memory-smoke [DEVICE_INDEX]|runtime-stream-smoke [DEVICE_INDEX]|runtime-copy-smoke [DEVICE_INDEX]|runtime-rmsnorm-smoke [DEVICE_INDEX]|runtime-silu-mul-smoke [DEVICE_INDEX]|runtime-mlp-smoke [DEVICE_INDEX]|inspect-package PATH|package-load-smoke PACKAGE_DIR [DEVICE_INDEX] [MAX_BYTES] [PAYLOAD_ROLE]|package-tensor-load-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [TENSOR_SELECTOR]|package-weight-register-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [TENSOR_SELECTOR]|package-weight-register-many-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [MAX_TENSORS]|package-materialize-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [TENSOR_SELECTOR]|package-mlp-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX]|package-materialize-matvec-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [TENSOR_SELECTOR]|package-rmsnorm-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [input|post]|package-rmsnorm-mlp-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [input|post]|package-linear-attn-proj-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [a|b|qkv|z|out|all]|package-materialize-bench PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [TENSOR_SELECTOR] [REPEATS]>"
    );
    eprintln!("linear attention projection selector: a|b|qkv|z|out|all");
    eprintln!(
        "payload roles: smallest|tensor-index|tensor-scale|tensor-codebook|codebook|passthrough"
    );
    eprintln!("tensor selector: omitted or numeric index, exact tensor name, or unique substring");
}

#[derive(Debug)]
struct MaterializeConfig {
    scale_format: String,
    scale_values: Vec<f32>,
    group_size: usize,
    tensor_scale: f32,
    elements: usize,
    output_bytes: usize,
}

fn materialize_config(loaded: &LoadedTensorBundle) -> Result<MaterializeConfig, String> {
    let scale_format = loaded
        .scale_format
        .as_deref()
        .ok_or_else(|| "selected tensor does not declare scale_format".to_string())?;
    let scale_values = ullm_engine::aq::scale_values(scale_format)?;
    let group_size = match loaded.group_size {
        Some(value) if value > 0 => value,
        Some(_) | None => {
            return Err("selected tensor does not declare a valid group_size".to_string());
        }
    };
    let tensor_scale = match loaded.tensor_scale {
        Some(value) if value.is_finite() && value > 0.0 => value,
        Some(_) | None => {
            return Err("selected tensor does not declare a valid tensor_scale".to_string());
        }
    };
    if loaded.index_encoding.as_deref() != Some("idx4_low_nibble_first") {
        return Err("selected tensor uses unsupported index encoding".to_string());
    }
    if loaded.scale_encoding.as_deref() != Some("u8_scale_table_index") {
        return Err("selected tensor uses unsupported scale encoding".to_string());
    }
    let elements = usize::try_from(loaded.elements)
        .map_err(|_| "selected tensor has too many elements for this host".to_string())?;
    let output_bytes = elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "materialized output byte size overflows".to_string())?;
    Ok(MaterializeConfig {
        scale_format: scale_format.to_string(),
        scale_values,
        group_size,
        tensor_scale,
        elements,
        output_bytes,
    })
}

fn matrix_shape_rows_cols(shape: &[u64], elements: usize) -> Result<(usize, usize), String> {
    let shape = match shape {
        shape if shape.len() == 2 => shape,
        _ => return Err("selected tensor shape is not 2D".to_string()),
    };
    let rows_u64 = shape[0];
    let cols_u64 = shape[1];
    if rows_u64 == 0 || cols_u64 == 0 {
        return Err("selected tensor has zero rows or columns".to_string());
    }
    let expected_elements = rows_u64
        .checked_mul(cols_u64)
        .ok_or_else(|| "selected tensor shape overflows element count".to_string())?;
    if expected_elements
        != u64::try_from(elements)
            .map_err(|_| "selected tensor has too many elements".to_string())?
    {
        return Err(format!(
            "selected tensor shape has {expected_elements} elements but materialize produced {elements}"
        ));
    }
    let rows = usize::try_from(rows_u64)
        .map_err(|_| "selected tensor row count does not fit host usize".to_string())?;
    let cols = usize::try_from(cols_u64)
        .map_err(|_| "selected tensor column count does not fit host usize".to_string())?;
    Ok((rows, cols))
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

fn decode_f32_le_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("f32 chunk")))
        .collect()
}

fn encode_f32_to_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<f32>());
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn runtime_host_rmsnorm_f32(input: &[f32], weight: &[f32], epsilon: f32) -> Vec<f32> {
    if input.len() != weight.len() || input.is_empty() {
        return Vec::new();
    }
    let mean_square = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
    let inv_rms = 1.0_f32 / (mean_square + epsilon).sqrt();
    input
        .iter()
        .zip(weight.iter())
        .map(|(input_value, weight_value)| input_value * inv_rms * weight_value)
        .collect()
}

fn runtime_host_matvec_f32(matrix: &[f32], input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    if rows == 0 || cols == 0 || matrix.len() != rows * cols || input.len() != cols {
        return Vec::new();
    }
    let mut output = Vec::with_capacity(rows);
    for row in 0..rows {
        let mut value = 0.0_f32;
        let row_start = row * cols;
        for col in 0..cols {
            value += matrix[row_start + col] * input[col];
        }
        output.push(value);
    }
    output
}

fn runtime_host_silu_mul_f32(gate: &[f32], up: &[f32]) -> Vec<f32> {
    if gate.len() != up.len() {
        return Vec::new();
    }
    gate.iter()
        .zip(up.iter())
        .map(|(gate_value, up_value)| {
            let gate_value = *gate_value;
            gate_value * (1.0_f32 / (1.0_f32 + (-gate_value).exp())) * *up_value
        })
        .collect()
}

fn format_f32_preview(values: &[f32]) -> String {
    let joined = values
        .iter()
        .map(|value| format!("{value:.7}"))
        .collect::<Vec<_>>()
        .join(",");
    format!("[{joined}]")
}
