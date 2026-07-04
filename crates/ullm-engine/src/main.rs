// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

use std::env;
use std::fs::File;
use std::io::Read;
use std::process::ExitCode;
use std::time::Instant;
use ullm_engine::decoder::{
    PagedDecodeShape, Qwen3DecoderLayerStepState, Qwen3SelfAttnBlockStepState,
    Qwen3SelfAttnDecodeState,
};
use ullm_engine::loader::{
    LoadOptions, LoadedPayload, LoadedTensorBundle, WeightRegistry, load_package_tensor_prefix,
};
use ullm_engine::package::{
    PassthroughPayloadBundle, ReferencedFile, ReferencedFileRole, TensorSelector,
};
use ullm_engine::scheduler::{KvBlockAllocator, KvBlockAllocatorStats, RequestId};

fn main() -> ExitCode {
    match env::args().nth(1).as_deref() {
        Some("inspect-devices") => inspect_devices(),
        Some("runtime-smoke") => runtime_smoke(),
        Some("runtime-memory-smoke") => runtime_memory_smoke(env::args().nth(2)),
        Some("runtime-stream-smoke") => runtime_stream_smoke(env::args().nth(2)),
        Some("runtime-copy-smoke") => runtime_copy_smoke(env::args().nth(2)),
        Some("runtime-rmsnorm-smoke") => runtime_rmsnorm_smoke(env::args().nth(2)),
        Some("runtime-silu-mul-smoke") => runtime_silu_mul_smoke(env::args().nth(2)),
        Some("runtime-sigmoid-mul-smoke") => runtime_sigmoid_mul_smoke(env::args().nth(2)),
        Some("runtime-add-smoke") => runtime_add_smoke(env::args().nth(2)),
        Some("runtime-rope-smoke") => runtime_rope_smoke(env::args().nth(2)),
        Some("runtime-causal-attn-smoke") => runtime_causal_attn_smoke(env::args().nth(2)),
        Some("runtime-decode-attn-smoke") => runtime_decode_attn_smoke(env::args().nth(2)),
        Some("runtime-paged-decode-attn-smoke") => {
            runtime_paged_decode_attn_smoke(env::args().nth(2))
        }
        Some("runtime-paged-kv-write-smoke") => runtime_paged_kv_write_smoke(env::args().nth(2)),
        Some("runtime-kv-paged-decode-smoke") => {
            runtime_kv_paged_decode_attn_smoke(env::args().nth(2))
        }
        Some("runtime-depthwise-conv1d-smoke") => {
            runtime_depthwise_conv1d_smoke(env::args().nth(2))
        }
        Some("runtime-linear-attn-gate-beta-smoke") => {
            runtime_linear_attn_gate_beta_smoke(env::args().nth(2))
        }
        Some("runtime-linear-attn-recurrent-smoke") => {
            runtime_linear_attn_recurrent_smoke(env::args().nth(2))
        }
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
        Some("package-mlp-block-smoke") => package_mlp_block_smoke(
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
        Some("package-self-attn-proj-smoke") => package_self_attn_proj_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
        ),
        Some("package-self-attn-qk-norm-smoke") => package_self_attn_qk_norm_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
        ),
        Some("package-self-attn-rope-smoke") => package_self_attn_rope_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
            env::args().nth(7),
            env::args().nth(8),
            env::args().nth(9),
        ),
        Some("package-self-attn-attention-smoke") => package_self_attn_attention_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
            env::args().nth(7),
            env::args().nth(8),
            env::args().nth(9),
        ),
        Some("package-self-attn-decode-smoke") => package_self_attn_decode_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
            env::args().nth(7),
            env::args().nth(8),
            env::args().nth(9),
        ),
        Some("package-self-attn-block-smoke") => package_self_attn_block_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
            env::args().nth(7),
            env::args().nth(8),
            env::args().nth(9),
        ),
        Some("package-self-attn-mlp-block-smoke") => package_self_attn_mlp_block_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
            env::args().nth(7),
            env::args().nth(8),
            env::args().nth(9),
        ),
        Some("package-linear-attn-qkv-norm-smoke") => package_linear_attn_qkv_norm_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
        ),
        Some("package-linear-attn-conv1d-smoke") => package_linear_attn_conv1d_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
        ),
        Some("package-linear-attn-gate-beta-smoke") => package_linear_attn_gate_beta_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
        ),
        Some("package-linear-attn-recurrent-smoke") => package_linear_attn_recurrent_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
        ),
        Some("package-linear-attn-post-smoke") => package_linear_attn_post_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
        ),
        Some("package-linear-attn-workflow-smoke") => package_linear_attn_workflow_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
        ),
        Some("package-linear-attn-block-smoke") => package_linear_attn_block_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
        ),
        Some("package-linear-attn-mlp-block-smoke") => package_linear_attn_mlp_block_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
        ),
        Some("package-linear-attn-aux-smoke") => package_linear_attn_aux_smoke(
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

fn runtime_sigmoid_mul_smoke(device_index: Option<String>) -> ExitCode {
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
    let input = [3.0_f32, -4.0_f32, 5.0_f32, 6.0_f32];
    let expected = runtime_host_sigmoid_mul_f32(&gate, &input);

    let gate_bytes = encode_f32_to_bytes(&gate);
    let input_bytes = encode_f32_to_bytes(&input);
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

    let mut output_buffer = match context.alloc_buffer(output_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate output runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };

    if let Err(err) = ullm_runtime_sys::sigmoid_mul_f32(
        &gate_buffer,
        &input_buffer,
        elements,
        &mut output_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run runtime sigmoid_mul_f32: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after sigmoid_mul: {err}");
        return ExitCode::from(1);
    }

    let mut output_raw = vec![0_u8; output_bytes];
    if let Err(err) = output_buffer.copy_to_host(0, &mut output_raw, Some(&mut stream)) {
        eprintln!("failed to copy sigmoid_mul result back to host: {err}");
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
            "runtime sigmoid mul smoke produced unexpected output: output={:?} expected={:?}",
            output, expected
        );
        return ExitCode::from(1);
    }
    println!(
        "runtime-sigmoid-mul-smoke backend={} device_index={} name=\"{}\" elements={} output={} verified=true",
        info.backend,
        device_index,
        info.name,
        elements,
        format_f32_preview(&output)
    );
    ExitCode::SUCCESS
}

fn runtime_add_smoke(device_index: Option<String>) -> ExitCode {
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

    let lhs = [-1.0_f32, 0.0, 1.0, 2.0, 8.5, -13.25, 0.125, 64.0];
    let rhs = [3.0_f32, -4.0, 5.0, 6.0, -0.25, 2.0, -0.5, -63.5];
    let expected = runtime_host_add_f32(&lhs, &rhs);
    let lhs_bytes = encode_f32_to_bytes(&lhs);
    let rhs_bytes = encode_f32_to_bytes(&rhs);
    let output_bytes = lhs.len() * std::mem::size_of::<f32>();

    let mut lhs_buffer = match context.alloc_buffer(lhs_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate lhs runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut rhs_buffer = match context.alloc_buffer(rhs_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate rhs runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut output_buffer = match context.alloc_buffer(output_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate output runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };

    if let Err(err) = lhs_buffer.copy_from_host(0, &lhs_bytes, Some(&mut stream)) {
        eprintln!("failed to copy lhs data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = rhs_buffer.copy_from_host(0, &rhs_bytes, Some(&mut stream)) {
        eprintln!("failed to copy rhs data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after input copies: {err}");
        return ExitCode::from(1);
    }

    if let Err(err) = ullm_runtime_sys::add_f32(
        &lhs_buffer,
        &rhs_buffer,
        lhs.len(),
        &mut output_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run runtime add_f32: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after add_f32: {err}");
        return ExitCode::from(1);
    }

    let mut output_raw = vec![0_u8; output_bytes];
    if let Err(err) = output_buffer.copy_to_host(0, &mut output_raw, Some(&mut stream)) {
        eprintln!("failed to copy add_f32 result back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after output copy: {err}");
        return ExitCode::from(1);
    }
    let output = decode_f32_le_values(&output_raw);
    let tolerance = 1e-6_f32;
    if output.len() != expected.len()
        || expected
            .iter()
            .zip(output.iter())
            .any(|(lhs, rhs)| (*lhs - *rhs).abs() > tolerance)
    {
        eprintln!(
            "runtime add smoke produced unexpected output: output={:?} expected={:?}",
            output, expected
        );
        return ExitCode::from(1);
    }

    println!(
        "runtime-add-smoke backend={} device_index={} name=\"{}\" elements={} output={} verified=true",
        info.backend,
        device_index,
        info.name,
        lhs.len(),
        format_f32_preview(&output)
    );
    ExitCode::SUCCESS
}

fn runtime_rope_smoke(device_index: Option<String>) -> ExitCode {
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

    let sequence_len = 2_usize;
    let heads = 2_usize;
    let head_dim = 6_usize;
    let rotary_dim = 4_usize;
    let position_offset = 3_usize;
    let rope_base = 10000.0_f32;
    let elements = sequence_len * heads * head_dim;
    let input = (0..elements)
        .map(|index| (index as f32 - 11.0) / 7.0)
        .collect::<Vec<_>>();
    let expected = runtime_host_rope_f32(
        &input,
        sequence_len,
        heads,
        head_dim,
        rotary_dim,
        position_offset,
        rope_base,
    );
    let input_bytes = encode_f32_to_bytes(&input);
    let output_bytes = input.len() * std::mem::size_of::<f32>();

    let mut input_buffer = match context.alloc_buffer(input_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate input runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut output_buffer = match context.alloc_buffer(output_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate output runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };

    if let Err(err) = input_buffer.copy_from_host(0, &input_bytes, Some(&mut stream)) {
        eprintln!("failed to copy RoPE input into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after input copy: {err}");
        return ExitCode::from(1);
    }

    if let Err(err) = ullm_runtime_sys::rope_f32(
        &input_buffer,
        sequence_len,
        heads,
        head_dim,
        rotary_dim,
        position_offset,
        rope_base,
        &mut output_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run runtime rope_f32: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after rope_f32: {err}");
        return ExitCode::from(1);
    }

    let mut output_raw = vec![0_u8; output_bytes];
    if let Err(err) = output_buffer.copy_to_host(0, &mut output_raw, Some(&mut stream)) {
        eprintln!("failed to copy rope_f32 result back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after output copy: {err}");
        return ExitCode::from(1);
    }
    let output = decode_f32_le_values(&output_raw);
    let tolerance = if info.backend == "cpu" {
        1e-5_f32
    } else {
        1e-4_f32
    };
    if output.len() != expected.len() {
        eprintln!(
            "runtime RoPE smoke produced unexpected output length: output={} expected={}",
            output.len(),
            expected.len()
        );
        return ExitCode::from(1);
    }
    let mut max_abs_diff = 0.0_f32;
    for (actual, expected_value) in output.iter().zip(expected.iter()) {
        let diff = (*actual - *expected_value).abs();
        if diff > tolerance {
            eprintln!(
                "runtime RoPE smoke produced unexpected output: max_abs_diff={diff} tolerance={tolerance} output={:?} expected={:?}",
                output, expected
            );
            return ExitCode::from(1);
        }
        max_abs_diff = max_abs_diff.max(diff);
    }

    println!(
        "runtime-rope-smoke backend={} device_index={} name=\"{}\" sequence_len={} heads={} head_dim={} rotary_dim={} position_offset={} rope_base={} output={} max_abs_diff={max_abs_diff:.9} verified=true",
        info.backend,
        device_index,
        info.name,
        sequence_len,
        heads,
        head_dim,
        rotary_dim,
        position_offset,
        rope_base,
        format_f32_preview(&output)
    );
    ExitCode::SUCCESS
}

fn runtime_causal_attn_smoke(device_index: Option<String>) -> ExitCode {
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

    let sequence_len = 3_usize;
    let q_heads = 4_usize;
    let kv_heads = 2_usize;
    let head_dim = 3_usize;
    let value_dim = 2_usize;
    let softmax_scale = 1.0_f32 / (head_dim as f32).sqrt();
    let q = (0..sequence_len * q_heads * head_dim)
        .map(|index| (index as f32 - 8.0) / 11.0)
        .collect::<Vec<_>>();
    let k = (0..sequence_len * kv_heads * head_dim)
        .map(|index| ((index * 3) as f32 - 7.0) / 13.0)
        .collect::<Vec<_>>();
    let v = (0..sequence_len * kv_heads * value_dim)
        .map(|index| ((index * 5) as f32 - 9.0) / 17.0)
        .collect::<Vec<_>>();
    let expected = runtime_host_causal_attn_f32(
        &q,
        &k,
        &v,
        sequence_len,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
    );
    let q_bytes = encode_f32_to_bytes(&q);
    let k_bytes = encode_f32_to_bytes(&k);
    let v_bytes = encode_f32_to_bytes(&v);
    let output_bytes = expected.len() * std::mem::size_of::<f32>();

    let mut q_buffer = match context.alloc_buffer(q_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate q runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut k_buffer = match context.alloc_buffer(k_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate k runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut v_buffer = match context.alloc_buffer(v_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate v runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut output_buffer = match context.alloc_buffer(output_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate causal attention output buffer: {err}");
            return ExitCode::from(1);
        }
    };

    if let Err(err) = q_buffer.copy_from_host(0, &q_bytes, Some(&mut stream)) {
        eprintln!("failed to copy causal attention q input: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = k_buffer.copy_from_host(0, &k_bytes, Some(&mut stream)) {
        eprintln!("failed to copy causal attention k input: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = v_buffer.copy_from_host(0, &v_bytes, Some(&mut stream)) {
        eprintln!("failed to copy causal attention v input: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!(
            "failed to synchronize runtime stream after causal attention input copies: {err}"
        );
        return ExitCode::from(1);
    }

    if let Err(err) = ullm_runtime_sys::causal_attn_f32(
        &q_buffer,
        &k_buffer,
        &v_buffer,
        sequence_len,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        &mut output_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run runtime causal_attn_f32: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after causal_attn_f32: {err}");
        return ExitCode::from(1);
    }

    let mut output_raw = vec![0_u8; output_bytes];
    if let Err(err) = output_buffer.copy_to_host(0, &mut output_raw, Some(&mut stream)) {
        eprintln!("failed to copy causal_attn_f32 result back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after causal attention output copy: {err}");
        return ExitCode::from(1);
    }
    let output = decode_f32_le_values(&output_raw);
    let tolerance = if info.backend == "cpu" {
        1e-5_f32
    } else {
        1e-4_f32
    };
    let max_abs_diff = match verify_f32_close(
        "runtime causal attention smoke",
        &output,
        &expected,
        tolerance,
        tolerance,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}; output={output:?} expected={expected:?}");
            return ExitCode::from(1);
        }
    };

    println!(
        "runtime-causal-attn-smoke backend={} device_index={} name=\"{}\" sequence_len={} q_heads={} kv_heads={} head_dim={} value_dim={} softmax_scale={softmax_scale:.9} output={} max_abs_diff={max_abs_diff:.9} verified=true",
        info.backend,
        device_index,
        info.name,
        sequence_len,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        format_f32_preview(&output)
    );
    ExitCode::SUCCESS
}

fn runtime_decode_attn_smoke(device_index: Option<String>) -> ExitCode {
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

    let cache_len = 3_usize;
    let q_heads = 4_usize;
    let kv_heads = 2_usize;
    let head_dim = 3_usize;
    let value_dim = 2_usize;
    let softmax_scale = 1.0_f32 / (head_dim as f32).sqrt();
    let q = (0..q_heads * head_dim)
        .map(|index| (index as f32 - 8.0) / 11.0)
        .collect::<Vec<_>>();
    let k = (0..cache_len * kv_heads * head_dim)
        .map(|index| ((index * 3) as f32 - 7.0) / 13.0)
        .collect::<Vec<_>>();
    let v = (0..cache_len * kv_heads * value_dim)
        .map(|index| ((index * 5) as f32 - 9.0) / 17.0)
        .collect::<Vec<_>>();
    let expected = runtime_host_decode_attn_f32(
        &q,
        &k,
        &v,
        cache_len,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
    );
    let q_bytes = encode_f32_to_bytes(&q);
    let k_bytes = encode_f32_to_bytes(&k);
    let v_bytes = encode_f32_to_bytes(&v);
    let output_bytes = expected.len() * std::mem::size_of::<f32>();

    let mut q_buffer = match context.alloc_buffer(q_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate decode attention q runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut k_buffer = match context.alloc_buffer(k_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate decode attention k cache runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut v_buffer = match context.alloc_buffer(v_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate decode attention v cache runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut output_buffer = match context.alloc_buffer(output_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate decode attention output buffer: {err}");
            return ExitCode::from(1);
        }
    };

    if let Err(err) = q_buffer.copy_from_host(0, &q_bytes, Some(&mut stream)) {
        eprintln!("failed to copy decode attention q input: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = k_buffer.copy_from_host(0, &k_bytes, Some(&mut stream)) {
        eprintln!("failed to copy decode attention k cache: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = v_buffer.copy_from_host(0, &v_bytes, Some(&mut stream)) {
        eprintln!("failed to copy decode attention v cache: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!(
            "failed to synchronize runtime stream after decode attention input copies: {err}"
        );
        return ExitCode::from(1);
    }

    if let Err(err) = ullm_runtime_sys::decode_attn_f32(
        &q_buffer,
        &k_buffer,
        &v_buffer,
        cache_len,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        &mut output_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run runtime decode_attn_f32: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after decode_attn_f32: {err}");
        return ExitCode::from(1);
    }

    let mut output_raw = vec![0_u8; output_bytes];
    if let Err(err) = output_buffer.copy_to_host(0, &mut output_raw, Some(&mut stream)) {
        eprintln!("failed to copy decode_attn_f32 result back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after decode attention output copy: {err}");
        return ExitCode::from(1);
    }
    let output = decode_f32_le_values(&output_raw);
    let tolerance = if info.backend == "cpu" {
        1e-5_f32
    } else {
        1e-4_f32
    };
    let max_abs_diff = match verify_f32_close(
        "runtime decode attention smoke",
        &output,
        &expected,
        tolerance,
        tolerance,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}; output={output:?} expected={expected:?}");
            return ExitCode::from(1);
        }
    };

    println!(
        "runtime-decode-attn-smoke backend={} device_index={} name=\"{}\" cache_len={} q_heads={} kv_heads={} head_dim={} value_dim={} softmax_scale={softmax_scale:.9} output={} max_abs_diff={max_abs_diff:.9} verified=true",
        info.backend,
        device_index,
        info.name,
        cache_len,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        format_f32_preview(&output)
    );
    ExitCode::SUCCESS
}

fn runtime_paged_decode_attn_smoke(device_index: Option<String>) -> ExitCode {
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

    let cache_len = 5_usize;
    let block_size = 2_usize;
    let cache_blocks = 4_usize;
    let q_heads = 4_usize;
    let kv_heads = 2_usize;
    let head_dim = 3_usize;
    let value_dim = 2_usize;
    let softmax_scale = 1.0_f32 / (head_dim as f32).sqrt();
    let q = (0..q_heads * head_dim)
        .map(|index| (index as f32 - 8.0) / 11.0)
        .collect::<Vec<_>>();
    let k = (0..cache_blocks * block_size * kv_heads * head_dim)
        .map(|index| ((index * 3) as f32 - 7.0) / 13.0)
        .collect::<Vec<_>>();
    let v = (0..cache_blocks * block_size * kv_heads * value_dim)
        .map(|index| ((index * 5) as f32 - 9.0) / 17.0)
        .collect::<Vec<_>>();
    let block_table = vec![2_u32, 0_u32, 3_u32];
    let expected = runtime_host_paged_decode_attn_f32(
        &q,
        &k,
        &v,
        &block_table,
        cache_len,
        block_size,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
    );
    let q_bytes = encode_f32_to_bytes(&q);
    let k_bytes = encode_f32_to_bytes(&k);
    let v_bytes = encode_f32_to_bytes(&v);
    let block_table_bytes = encode_u32_to_bytes(&block_table);
    let output_bytes = expected.len() * std::mem::size_of::<f32>();

    let mut q_buffer = match context.alloc_buffer(q_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate paged decode attention q runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut k_buffer = match context.alloc_buffer(k_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate paged decode attention k cache runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut v_buffer = match context.alloc_buffer(v_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate paged decode attention v cache runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut block_table_buffer = match context.alloc_buffer(block_table_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!(
                "failed to allocate paged decode attention block table runtime buffer: {err}"
            );
            return ExitCode::from(1);
        }
    };
    let mut output_buffer = match context.alloc_buffer(output_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate paged decode attention output buffer: {err}");
            return ExitCode::from(1);
        }
    };

    if let Err(err) = q_buffer.copy_from_host(0, &q_bytes, Some(&mut stream)) {
        eprintln!("failed to copy paged decode attention q input: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = k_buffer.copy_from_host(0, &k_bytes, Some(&mut stream)) {
        eprintln!("failed to copy paged decode attention k cache: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = v_buffer.copy_from_host(0, &v_bytes, Some(&mut stream)) {
        eprintln!("failed to copy paged decode attention v cache: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = block_table_buffer.copy_from_host(0, &block_table_bytes, Some(&mut stream)) {
        eprintln!("failed to copy paged decode attention block table: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!(
            "failed to synchronize runtime stream after paged decode attention input copies: {err}"
        );
        return ExitCode::from(1);
    }

    if let Err(err) = ullm_runtime_sys::paged_decode_attn_f32(
        &q_buffer,
        &k_buffer,
        &v_buffer,
        &block_table_buffer,
        cache_len,
        block_size,
        cache_blocks,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        &mut output_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run runtime paged_decode_attn_f32: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after paged_decode_attn_f32: {err}");
        return ExitCode::from(1);
    }

    let mut output_raw = vec![0_u8; output_bytes];
    if let Err(err) = output_buffer.copy_to_host(0, &mut output_raw, Some(&mut stream)) {
        eprintln!("failed to copy paged_decode_attn_f32 result back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!(
            "failed to synchronize runtime stream after paged decode attention output copy: {err}"
        );
        return ExitCode::from(1);
    }
    let output = decode_f32_le_values(&output_raw);
    let tolerance = if info.backend == "cpu" {
        1e-5_f32
    } else {
        1e-4_f32
    };
    let max_abs_diff = match verify_f32_close(
        "runtime paged decode attention smoke",
        &output,
        &expected,
        tolerance,
        tolerance,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}; output={output:?} expected={expected:?}");
            return ExitCode::from(1);
        }
    };

    println!(
        "runtime-paged-decode-attn-smoke backend={} device_index={} name=\"{}\" cache_len={} block_size={} cache_blocks={} block_table={:?} q_heads={} kv_heads={} head_dim={} value_dim={} softmax_scale={softmax_scale:.9} output={} max_abs_diff={max_abs_diff:.9} verified=true",
        info.backend,
        device_index,
        info.name,
        cache_len,
        block_size,
        cache_blocks,
        block_table,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        format_f32_preview(&output)
    );
    ExitCode::SUCCESS
}

fn runtime_paged_kv_write_smoke(device_index: Option<String>) -> ExitCode {
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

    let cache_len = 3_usize;
    let block_size = 2_usize;
    let (block_table, cache_blocks, stats) =
        match allocate_fragmented_paged_decode_blocks(cache_len, block_size) {
            Ok(value) => value,
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::from(1);
            }
        };
    let kv_heads = 2_usize;
    let head_dim = 3_usize;
    let value_dim = 2_usize;
    let logical_k = (0..cache_len * kv_heads * head_dim)
        .map(|index| ((index * 3) as f32 - 7.0) / 13.0)
        .collect::<Vec<_>>();
    let logical_v = (0..cache_len * kv_heads * value_dim)
        .map(|index| ((index * 5) as f32 - 9.0) / 17.0)
        .collect::<Vec<_>>();
    let (expected_k_cache, expected_v_cache) = match pack_paged_decode_kv_cache_for_block_table(
        &logical_k,
        &logical_v,
        &block_table,
        cache_len,
        block_size,
        cache_blocks,
        kv_heads,
        head_dim,
        value_dim,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };

    let k_token_bytes = kv_heads * head_dim * std::mem::size_of::<f32>();
    let v_token_bytes = kv_heads * value_dim * std::mem::size_of::<f32>();
    let k_cache_bytes = expected_k_cache.len() * std::mem::size_of::<f32>();
    let v_cache_bytes = expected_v_cache.len() * std::mem::size_of::<f32>();
    let block_table_bytes = encode_u32_to_bytes(&block_table);
    let zero_k_cache = vec![0_u8; k_cache_bytes];
    let zero_v_cache = vec![0_u8; v_cache_bytes];

    let mut k_token_buffer = match context.alloc_buffer(k_token_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate paged KV write k token buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut v_token_buffer = match context.alloc_buffer(v_token_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate paged KV write v token buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut block_table_buffer = match context.alloc_buffer(block_table_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate paged KV write block table buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut k_cache_buffer = match context.alloc_buffer(k_cache_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate paged KV write k cache buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut v_cache_buffer = match context.alloc_buffer(v_cache_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate paged KV write v cache buffer: {err}");
            return ExitCode::from(1);
        }
    };

    if let Err(err) = block_table_buffer.copy_from_host(0, &block_table_bytes, Some(&mut stream)) {
        eprintln!("failed to copy paged KV write block table: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = k_cache_buffer.copy_from_host(0, &zero_k_cache, Some(&mut stream)) {
        eprintln!("failed to initialize paged KV write k cache: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = v_cache_buffer.copy_from_host(0, &zero_v_cache, Some(&mut stream)) {
        eprintln!("failed to initialize paged KV write v cache: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize paged KV write initial copies: {err}");
        return ExitCode::from(1);
    }

    for timestep in 0..cache_len {
        let k_start = timestep * kv_heads * head_dim;
        let k_end = k_start + kv_heads * head_dim;
        let v_start = timestep * kv_heads * value_dim;
        let v_end = v_start + kv_heads * value_dim;
        let k_bytes = encode_f32_to_bytes(&logical_k[k_start..k_end]);
        let v_bytes = encode_f32_to_bytes(&logical_v[v_start..v_end]);
        if let Err(err) = k_token_buffer.copy_from_host(0, &k_bytes, Some(&mut stream)) {
            eprintln!("failed to copy paged KV write timestep {timestep} k: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = v_token_buffer.copy_from_host(0, &v_bytes, Some(&mut stream)) {
            eprintln!("failed to copy paged KV write timestep {timestep} v: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!("failed to synchronize paged KV write timestep {timestep} inputs: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = ullm_runtime_sys::paged_kv_write_f32(
            &k_token_buffer,
            &v_token_buffer,
            &block_table_buffer,
            timestep,
            block_size,
            cache_blocks,
            kv_heads,
            head_dim,
            value_dim,
            &mut k_cache_buffer,
            &mut v_cache_buffer,
            Some(&mut stream),
        ) {
            eprintln!("failed to run paged_kv_write_f32 for timestep {timestep}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!("failed to synchronize paged KV write timestep {timestep}: {err}");
            return ExitCode::from(1);
        }
    }

    let mut k_cache_raw = vec![0_u8; k_cache_bytes];
    let mut v_cache_raw = vec![0_u8; v_cache_bytes];
    if let Err(err) = k_cache_buffer.copy_to_host(0, &mut k_cache_raw, Some(&mut stream)) {
        eprintln!("failed to copy paged KV write k cache back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = v_cache_buffer.copy_to_host(0, &mut v_cache_raw, Some(&mut stream)) {
        eprintln!("failed to copy paged KV write v cache back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize paged KV write readback: {err}");
        return ExitCode::from(1);
    }
    let k_cache = decode_f32_le_values(&k_cache_raw);
    let v_cache = decode_f32_le_values(&v_cache_raw);
    let k_max_abs_diff = match verify_f32_close(
        "runtime paged KV write k cache",
        &k_cache,
        &expected_k_cache,
        1e-5,
        1e-5,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let v_max_abs_diff = match verify_f32_close(
        "runtime paged KV write v cache",
        &v_cache,
        &expected_v_cache,
        1e-5,
        1e-5,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };

    println!(
        "runtime-paged-kv-write-smoke backend={} device_index={} name=\"{}\" cache_len={} block_size={} cache_blocks={} block_table={:?} free_blocks={} allocated_block_count={} free_runs={} largest_free_run={} kv_heads={} head_dim={} value_dim={} k_cache_preview={} v_cache_preview={} k_max_abs_diff={k_max_abs_diff:.9} v_max_abs_diff={v_max_abs_diff:.9} verified=true",
        info.backend,
        device_index,
        info.name,
        cache_len,
        block_size,
        cache_blocks,
        block_table,
        stats.free_blocks,
        stats.allocated_blocks,
        stats.free_runs,
        stats.largest_free_run,
        kv_heads,
        head_dim,
        value_dim,
        format_f32_preview(&k_cache[..8.min(k_cache.len())]),
        format_f32_preview(&v_cache[..8.min(v_cache.len())]),
    );
    ExitCode::SUCCESS
}

fn runtime_kv_paged_decode_attn_smoke(device_index: Option<String>) -> ExitCode {
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

    let mut allocator = KvBlockAllocator::with_block_size(4, 2);
    let fragment = match allocator.allocate(RequestId(10), 3) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to allocate fragmenting KV blocks: {err}");
            return ExitCode::from(1);
        }
    };
    let freed = allocator.free_request(fragment.request_id);
    if freed != fragment.blocks.len() {
        eprintln!(
            "freed KV block count {freed} does not match allocated fragment blocks {}",
            fragment.blocks.len()
        );
        return ExitCode::from(1);
    }
    let cache_len = 3_usize;
    let block_size = allocator.block_size_tokens() as usize;
    let block_count = (cache_len - 1) / block_size + 1;
    let allocation = match allocator.allocate(RequestId(11), block_count) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to allocate decode KV blocks: {err}");
            return ExitCode::from(1);
        }
    };
    let block_table = allocation.blocks;
    let cache_blocks = allocator.total_blocks() as usize;
    let stats = allocator.stats();

    let q_heads = 4_usize;
    let kv_heads = 2_usize;
    let head_dim = 3_usize;
    let value_dim = 2_usize;
    let softmax_scale = 1.0_f32 / (head_dim as f32).sqrt();
    let q = (0..q_heads * head_dim)
        .map(|index| (index as f32 - 8.0) / 11.0)
        .collect::<Vec<_>>();
    let logical_k = (0..cache_len * kv_heads * head_dim)
        .map(|index| ((index * 3) as f32 - 7.0) / 13.0)
        .collect::<Vec<_>>();
    let logical_v = (0..cache_len * kv_heads * value_dim)
        .map(|index| ((index * 5) as f32 - 9.0) / 17.0)
        .collect::<Vec<_>>();
    let (paged_k_cache, paged_v_cache) = match pack_paged_decode_kv_cache_for_block_table(
        &logical_k,
        &logical_v,
        &block_table,
        cache_len,
        block_size,
        cache_blocks,
        kv_heads,
        head_dim,
        value_dim,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let (output, max_abs_diff) = match runtime_paged_decode_attn_verify(
        &mut context,
        &mut stream,
        &q,
        &paged_k_cache,
        &paged_v_cache,
        &block_table,
        cache_len,
        block_size,
        cache_blocks,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        "runtime kv paged decode attention smoke",
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let contiguous_expected = runtime_host_decode_attn_f32(
        &q,
        &logical_k,
        &logical_v,
        cache_len,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
    );
    let logical_paged_max_abs_diff = match verify_f32_close(
        "runtime kv paged decode attention logical-vs-paged",
        &output,
        &contiguous_expected,
        1e-4,
        1e-4,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };

    println!(
        "runtime-kv-paged-decode-smoke backend={} device_index={} name=\"{}\" cache_len={} block_size={} cache_blocks={} allocated_blocks={:?} free_blocks={} allocated_block_count={} free_runs={} largest_free_run={} q_heads={} kv_heads={} head_dim={} value_dim={} softmax_scale={softmax_scale:.9} paged_k_cache_preview={} paged_v_cache_preview={} output={} max_abs_diff={max_abs_diff:.9} logical_paged_max_abs_diff={logical_paged_max_abs_diff:.9} verified=true",
        info.backend,
        device_index,
        info.name,
        cache_len,
        block_size,
        cache_blocks,
        block_table,
        stats.free_blocks,
        stats.allocated_blocks,
        stats.free_runs,
        stats.largest_free_run,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        format_f32_preview(&paged_k_cache[..8.min(paged_k_cache.len())]),
        format_f32_preview(&paged_v_cache[..8.min(paged_v_cache.len())]),
        format_f32_preview(&output)
    );
    ExitCode::SUCCESS
}

fn runtime_depthwise_conv1d_smoke(device_index: Option<String>) -> ExitCode {
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

    let channels = 3_usize;
    let sequence_len = 5_usize;
    let kernel_size = 3_usize;
    let input = [
        1.0_f32, 0.5, -1.0, 2.0, 1.0, 0.5, 3.0, -0.5, 0.5, 4.0, -1.0, 1.5, 5.0, 0.0, -2.0,
    ];
    let weight = [1.0_f32, -1.0, 2.0, 0.5, 1.0, -0.5, -1.0, 1.0, 1.5];
    let expected =
        runtime_host_depthwise_conv1d_f32(&input, &weight, channels, sequence_len, kernel_size);
    if expected.is_empty() {
        eprintln!("failed to build deterministic depthwise conv1d reference");
        return ExitCode::from(1);
    }

    let input_bytes = encode_f32_to_bytes(&input);
    let weight_bytes = encode_f32_to_bytes(&weight);
    let output_bytes = input_bytes.len();

    let mut input_buffer = match context.alloc_buffer(input_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate depthwise conv1d input buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut weight_buffer = match context.alloc_buffer(weight_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate depthwise conv1d weight buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut output_buffer = match context.alloc_buffer(output_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate depthwise conv1d output buffer: {err}");
            return ExitCode::from(1);
        }
    };

    if let Err(err) = input_buffer.copy_from_host(0, &input_bytes, Some(&mut stream)) {
        eprintln!("failed to copy depthwise conv1d input data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = weight_buffer.copy_from_host(0, &weight_bytes, Some(&mut stream)) {
        eprintln!("failed to copy depthwise conv1d weight data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after depthwise conv1d input copy: {err}");
        return ExitCode::from(1);
    }

    if let Err(err) = ullm_runtime_sys::depthwise_conv1d_f32(
        &input_buffer,
        &weight_buffer,
        channels,
        sequence_len,
        kernel_size,
        &mut output_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run runtime depthwise_conv1d_f32: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after depthwise conv1d: {err}");
        return ExitCode::from(1);
    }

    let mut output_raw = vec![0_u8; output_bytes];
    if let Err(err) = output_buffer.copy_to_host(0, &mut output_raw, Some(&mut stream)) {
        eprintln!("failed to copy depthwise conv1d result back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after depthwise conv1d output copy: {err}");
        return ExitCode::from(1);
    }
    let output = decode_f32_le_values(&output_raw);

    let mut max_abs_diff = 0.0_f32;
    for (lhs, rhs) in output.iter().zip(expected.iter()) {
        let diff = (lhs - rhs).abs();
        if diff > 1e-5_f32 {
            eprintln!(
                "runtime depthwise conv1d smoke produced unexpected output: max_abs_diff={diff} output={:?} expected={:?}",
                output, expected
            );
            return ExitCode::from(1);
        }
        if diff > max_abs_diff {
            max_abs_diff = diff;
        }
    }
    println!(
        "runtime-depthwise-conv1d-smoke backend={} device_index={} name=\"{}\" channels={} sequence_len={} kernel_size={} output={} max_abs_diff={max_abs_diff:.9} verified=true",
        info.backend,
        device_index,
        info.name,
        channels,
        sequence_len,
        kernel_size,
        format_f32_preview(&output[..8.min(output.len())])
    );
    ExitCode::SUCCESS
}

fn runtime_linear_attn_gate_beta_smoke(device_index: Option<String>) -> ExitCode {
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

    let heads = 3_usize;
    let sequence_len = 4_usize;
    let a = [
        -2.0_f32, -0.5, 0.25, 0.75, 1.0, -1.25, 2.0, -3.0, 0.5, 21.0, -20.0, 4.0,
    ];
    let b = [
        -4.0_f32, -1.0, 0.0, 0.5, 1.0, 3.0, -2.0, 2.5, -0.25, 4.0, -3.5, 1.5,
    ];
    let a_log = [-0.75_f32, 0.0, 0.5];
    let dt_bias = [0.25_f32, -0.5, 1.25];
    let (expected_gate, expected_beta) =
        runtime_host_linear_attn_gate_beta_f32(&a, &b, &a_log, &dt_bias, heads, sequence_len);
    if expected_gate.is_empty() || expected_beta.is_empty() {
        eprintln!("failed to build deterministic linear attention gate beta reference");
        return ExitCode::from(1);
    }

    let a_bytes = encode_f32_to_bytes(&a);
    let b_bytes = encode_f32_to_bytes(&b);
    let a_log_bytes = encode_f32_to_bytes(&a_log);
    let dt_bias_bytes = encode_f32_to_bytes(&dt_bias);
    let output_bytes = a_bytes.len();

    let mut a_buffer = match context.alloc_buffer(a_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate a runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut b_buffer = match context.alloc_buffer(b_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate b runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut a_log_buffer = match context.alloc_buffer(a_log_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate A_log runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut dt_bias_buffer = match context.alloc_buffer(dt_bias_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate dt_bias runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut gate_output_buffer = match context.alloc_buffer(output_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate gate output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut beta_output_buffer = match context.alloc_buffer(output_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate beta output buffer: {err}");
            return ExitCode::from(1);
        }
    };

    if let Err(err) = a_buffer.copy_from_host(0, &a_bytes, Some(&mut stream)) {
        eprintln!("failed to copy a data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = b_buffer.copy_from_host(0, &b_bytes, Some(&mut stream)) {
        eprintln!("failed to copy b data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = a_log_buffer.copy_from_host(0, &a_log_bytes, Some(&mut stream)) {
        eprintln!("failed to copy A_log data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = dt_bias_buffer.copy_from_host(0, &dt_bias_bytes, Some(&mut stream)) {
        eprintln!("failed to copy dt_bias data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after gate beta input copy: {err}");
        return ExitCode::from(1);
    }

    if let Err(err) = ullm_runtime_sys::linear_attn_gate_beta_f32(
        &a_buffer,
        &b_buffer,
        &a_log_buffer,
        &dt_bias_buffer,
        heads,
        sequence_len,
        &mut gate_output_buffer,
        &mut beta_output_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run runtime linear_attn_gate_beta_f32: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after linear attention gate beta: {err}");
        return ExitCode::from(1);
    }

    let mut gate_output_raw = vec![0_u8; output_bytes];
    let mut beta_output_raw = vec![0_u8; output_bytes];
    if let Err(err) = gate_output_buffer.copy_to_host(0, &mut gate_output_raw, Some(&mut stream)) {
        eprintln!("failed to copy gate output back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = beta_output_buffer.copy_to_host(0, &mut beta_output_raw, Some(&mut stream)) {
        eprintln!("failed to copy beta output back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after gate beta output copy: {err}");
        return ExitCode::from(1);
    }
    let gate_output = decode_f32_le_values(&gate_output_raw);
    let beta_output = decode_f32_le_values(&beta_output_raw);

    let mut max_abs_diff = 0.0_f32;
    for ((gate, expected_gate), (beta, expected_beta)) in gate_output
        .iter()
        .zip(expected_gate.iter())
        .zip(beta_output.iter().zip(expected_beta.iter()))
    {
        let gate_diff = (gate - expected_gate).abs();
        let beta_diff = (beta - expected_beta).abs();
        let diff = gate_diff.max(beta_diff);
        if diff > 1e-5_f32 {
            eprintln!(
                "runtime linear attention gate beta smoke produced unexpected output: max_abs_diff={diff} gate={:?} expected_gate={:?} beta={:?} expected_beta={:?}",
                gate_output, expected_gate, beta_output, expected_beta
            );
            return ExitCode::from(1);
        }
        if diff > max_abs_diff {
            max_abs_diff = diff;
        }
    }
    println!(
        "runtime-linear-attn-gate-beta-smoke backend={} device_index={} name=\"{}\" heads={} sequence_len={} gate={} beta={} max_abs_diff={max_abs_diff:.9} verified=true",
        info.backend,
        device_index,
        info.name,
        heads,
        sequence_len,
        format_f32_preview(&gate_output[..8.min(gate_output.len())]),
        format_f32_preview(&beta_output[..8.min(beta_output.len())]),
    );
    ExitCode::SUCCESS
}

fn runtime_linear_attn_recurrent_smoke(device_index: Option<String>) -> ExitCode {
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

    let key_heads = 1_usize;
    let value_heads = 2_usize;
    let sequence_len = 3_usize;
    let key_dim = 3_usize;
    let value_dim = 2_usize;
    let q = [0.25_f32, -0.5, 0.75, 0.6, -0.2, 0.3, -0.1, 0.8, -0.35];
    let k = [-0.3_f32, 0.4, 0.2, 0.1, 0.2, -0.6, 0.55, -0.1, 0.25];
    let v = [
        0.7_f32, -0.2, -0.5, 0.4, 0.25, 0.3, -0.1, -0.6, 0.9, 0.05, -0.35, 0.8,
    ];
    let gate = [-0.2_f32, -0.5, -0.1, -0.3, -0.7, -0.05];
    let beta = [0.8_f32, 0.6, 0.5, 0.9, 0.7, 0.4];
    let initial_state = [
        0.01_f32, -0.02, 0.03, 0.04, -0.01, 0.02, -0.03, 0.05, 0.02, -0.04, 0.01, 0.03,
    ];
    let mut expected_state = initial_state.to_vec();
    let expected_output = runtime_host_linear_attn_recurrent_f32(
        &q,
        &k,
        &v,
        &gate,
        &beta,
        key_heads,
        value_heads,
        sequence_len,
        key_dim,
        value_dim,
        &mut expected_state,
    );
    if expected_output.is_empty() {
        eprintln!("failed to build deterministic linear attention recurrent reference");
        return ExitCode::from(1);
    }

    let q_bytes = encode_f32_to_bytes(&q);
    let k_bytes = encode_f32_to_bytes(&k);
    let v_bytes = encode_f32_to_bytes(&v);
    let gate_bytes = encode_f32_to_bytes(&gate);
    let beta_bytes = encode_f32_to_bytes(&beta);
    let state_bytes = encode_f32_to_bytes(&initial_state);
    let output_bytes = v_bytes.len();

    let mut q_buffer = match context.alloc_buffer(q_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate q runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut k_buffer = match context.alloc_buffer(k_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate k runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut v_buffer = match context.alloc_buffer(v_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate v runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut gate_buffer = match context.alloc_buffer(gate_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate gate runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut beta_buffer = match context.alloc_buffer(beta_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate beta runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut state_buffer = match context.alloc_buffer(state_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate state runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut output_buffer = match context.alloc_buffer(output_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate output runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };

    if let Err(err) = q_buffer.copy_from_host(0, &q_bytes, Some(&mut stream)) {
        eprintln!("failed to copy q data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = k_buffer.copy_from_host(0, &k_bytes, Some(&mut stream)) {
        eprintln!("failed to copy k data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = v_buffer.copy_from_host(0, &v_bytes, Some(&mut stream)) {
        eprintln!("failed to copy v data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = gate_buffer.copy_from_host(0, &gate_bytes, Some(&mut stream)) {
        eprintln!("failed to copy gate data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = beta_buffer.copy_from_host(0, &beta_bytes, Some(&mut stream)) {
        eprintln!("failed to copy beta data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = state_buffer.copy_from_host(0, &state_bytes, Some(&mut stream)) {
        eprintln!("failed to copy state data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after recurrent input copy: {err}");
        return ExitCode::from(1);
    }

    if let Err(err) = ullm_runtime_sys::linear_attn_recurrent_f32(
        &q_buffer,
        &k_buffer,
        &v_buffer,
        &gate_buffer,
        &beta_buffer,
        key_heads,
        value_heads,
        sequence_len,
        key_dim,
        value_dim,
        &mut state_buffer,
        &mut output_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run runtime linear_attn_recurrent_f32: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after linear attention recurrent: {err}");
        return ExitCode::from(1);
    }

    let mut output_raw = vec![0_u8; output_bytes];
    let mut final_state_raw = vec![0_u8; state_bytes.len()];
    if let Err(err) = output_buffer.copy_to_host(0, &mut output_raw, Some(&mut stream)) {
        eprintln!("failed to copy recurrent output back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = state_buffer.copy_to_host(0, &mut final_state_raw, Some(&mut stream)) {
        eprintln!("failed to copy recurrent state back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after recurrent output copy: {err}");
        return ExitCode::from(1);
    }
    let output = decode_f32_le_values(&output_raw);
    let final_state = decode_f32_le_values(&final_state_raw);

    let mut max_abs_diff = 0.0_f32;
    for (lhs, rhs) in output.iter().zip(expected_output.iter()) {
        let diff = (lhs - rhs).abs();
        if diff > 1e-5_f32 {
            eprintln!(
                "runtime linear attention recurrent smoke produced unexpected output: max_abs_diff={diff} output={:?} expected={:?}",
                output, expected_output
            );
            return ExitCode::from(1);
        }
        if diff > max_abs_diff {
            max_abs_diff = diff;
        }
    }
    for (lhs, rhs) in final_state.iter().zip(expected_state.iter()) {
        let diff = (lhs - rhs).abs();
        if diff > 1e-5_f32 {
            eprintln!(
                "runtime linear attention recurrent smoke produced unexpected state: max_abs_diff={diff} state={:?} expected={:?}",
                final_state, expected_state
            );
            return ExitCode::from(1);
        }
        if diff > max_abs_diff {
            max_abs_diff = diff;
        }
    }
    println!(
        "runtime-linear-attn-recurrent-smoke backend={} device_index={} name=\"{}\" key_heads={} value_heads={} sequence_len={} key_dim={} value_dim={} output={} state={} max_abs_diff={max_abs_diff:.9} verified=true",
        info.backend,
        device_index,
        info.name,
        key_heads,
        value_heads,
        sequence_len,
        key_dim,
        value_dim,
        format_f32_preview(&output[..8.min(output.len())]),
        format_f32_preview(&final_state[..8.min(final_state.len())]),
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
    package_rmsnorm_mlp_smoke_impl(
        "package-rmsnorm-mlp-smoke",
        false,
        path,
        device_index,
        chunk_bytes,
        layer_index,
        norm_kind,
    )
}

fn package_mlp_block_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    norm_kind: Option<String>,
) -> ExitCode {
    package_rmsnorm_mlp_smoke_impl(
        "package-mlp-block-smoke",
        true,
        path,
        device_index,
        chunk_bytes,
        layer_index,
        norm_kind,
    )
}

fn package_rmsnorm_mlp_smoke_impl(
    command_name: &str,
    include_block: bool,
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    norm_kind: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("{command_name} requires a .ullm.d path");
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

    let mut output_bytes = vec![0_u8; hidden_bytes];
    if let Err(err) = output_buffer.copy_to_host(0, &mut output_bytes, Some(&mut stream)) {
        eprintln!("failed to copy MLP output to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after MLP output copy: {err}");
        return ExitCode::from(1);
    }
    let output = decode_f32_le_values(&output_bytes);

    if include_block {
        let mut block_output_buffer = match context.alloc_buffer(hidden_bytes) {
            Ok(buffer) => buffer,
            Err(err) => {
                eprintln!("failed to allocate MLP block output buffer: {err}");
                return ExitCode::from(1);
            }
        };
        if let Err(err) = ullm_runtime_sys::add_f32(
            &input_buffer,
            &output_buffer,
            hidden,
            &mut block_output_buffer,
            Some(&mut stream),
        ) {
            eprintln!("failed to run MLP residual add: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!("failed to synchronize runtime stream after MLP residual add: {err}");
            return ExitCode::from(1);
        }

        let mut block_output_bytes = vec![0_u8; hidden_bytes];
        if let Err(err) =
            block_output_buffer.copy_to_host(0, &mut block_output_bytes, Some(&mut stream))
        {
            eprintln!("failed to copy MLP block output to host: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!("failed to synchronize runtime stream after MLP block output copy: {err}");
            return ExitCode::from(1);
        }
        let block_output = decode_f32_le_values(&block_output_bytes);
        let expected_block_output = runtime_host_add_f32(&input, &output);
        if expected_block_output.len() != block_output.len() {
            eprintln!(
                "{command_name} output size mismatch: expected {} got {}",
                expected_block_output.len(),
                block_output.len()
            );
            return ExitCode::from(1);
        }

        let mut block_max_abs_diff = 0.0_f32;
        for (lhs, rhs) in block_output.iter().zip(expected_block_output.iter()) {
            let diff = (lhs - rhs).abs();
            let tolerance = 1e-6_f32.max(rhs.abs() * 1e-6_f32);
            if diff > tolerance {
                eprintln!(
                    "{command_name} residual output mismatch: max_abs_diff={diff} tolerance={tolerance}"
                );
                return ExitCode::from(1);
            }
            if diff > block_max_abs_diff {
                block_max_abs_diff = diff;
            }
        }

        println!(
            "{command_name} package={} layer={} norm_kind={} norm_tensor=\"{}\" gate_tensor=\"{}\" up_tensor=\"{}\" down_tensor=\"{}\" hidden={} intermediate={} backend={} device_index={} name=\"{}\" residual_preview={} mlp_output_preview={} block_output_preview={} block_max_abs_diff={block_max_abs_diff:.9} verified=true",
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
            format_f32_preview(&input[..8.min(input.len())]),
            format_f32_preview(&output[..8.min(output.len())]),
            format_f32_preview(&block_output[..8.min(block_output.len())])
        );
    } else {
        println!(
            "{command_name} package={} layer={} norm_kind={} norm_tensor=\"{}\" gate_tensor=\"{}\" up_tensor=\"{}\" down_tensor=\"{}\" hidden={} intermediate={} backend={} device_index={} name=\"{}\" preview={} verified=true",
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
            format_f32_preview(&output[..8.min(output.len())])
        );
    }
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

fn package_self_attn_proj_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    projection: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-self-attn-proj-smoke requires a .ullm.d path");
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
    let projection = match parse_self_attn_projection(projection.as_deref()) {
        Ok(value) => value,
        Err(code) => return code,
    };

    let requested_projections = match projection {
        SelfAttnProjection::Q => {
            vec![(
                "q",
                format!("model.language_model.layers.{layer_index}.self_attn.q_proj.weight"),
            )]
        }
        SelfAttnProjection::K => {
            vec![(
                "k",
                format!("model.language_model.layers.{layer_index}.self_attn.k_proj.weight"),
            )]
        }
        SelfAttnProjection::V => {
            vec![(
                "v",
                format!("model.language_model.layers.{layer_index}.self_attn.v_proj.weight"),
            )]
        }
        SelfAttnProjection::O => {
            vec![(
                "o",
                format!("model.language_model.layers.{layer_index}.self_attn.o_proj.weight"),
            )]
        }
        SelfAttnProjection::All => vec![
            (
                "q",
                format!("model.language_model.layers.{layer_index}.self_attn.q_proj.weight"),
            ),
            (
                "k",
                format!("model.language_model.layers.{layer_index}.self_attn.k_proj.weight"),
            ),
            (
                "v",
                format!("model.language_model.layers.{layer_index}.self_attn.v_proj.weight"),
            ),
            (
                "o",
                format!("model.language_model.layers.{layer_index}.self_attn.o_proj.weight"),
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
                eprintln!("failed to materialize self-attn projection {projection_name}: {err}");
                return ExitCode::from(1);
            }
        };

        match hidden {
            Some(expected) if expected != cols => {
                eprintln!(
                    "self-attn projection {projection_name} tensor {tensor_name} has cols={cols}, expected hidden={expected}"
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
                        eprintln!("failed to allocate shared self-attn input buffer: {err}");
                        return ExitCode::from(1);
                    }
                };
                if let Err(err) = buffer.copy_from_host(0, &input_bytes, Some(&mut stream)) {
                    eprintln!(
                        "failed to copy deterministic self-attn input data into shared runtime buffer: {err}"
                    );
                    return ExitCode::from(1);
                }
                if let Err(err) = stream.synchronize() {
                    eprintln!(
                        "failed to synchronize runtime stream after self-attn input copy: {err}"
                    );
                    return ExitCode::from(1);
                }
                input_buffer = Some(buffer);
            }
        }

        let Some(shared_input) = input_buffer.as_mut() else {
            eprintln!("shared self-attn runtime input buffer was not initialized");
            return ExitCode::from(1);
        };

        let output_bytes = match rows.checked_mul(std::mem::size_of::<f32>()) {
            Some(value) => value,
            None => {
                eprintln!("output byte size overflows for self-attn projection {projection_name}");
                return ExitCode::from(1);
            }
        };
        let mut output = match context.alloc_buffer(output_bytes) {
            Ok(buffer) => buffer,
            Err(err) => {
                eprintln!(
                    "failed to allocate output buffer for self-attn projection {projection_name}: {err}"
                );
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
            eprintln!("failed to run matvec for self-attn projection {projection_name}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!("failed to synchronize runtime stream after self-attn matvec: {err}");
            return ExitCode::from(1);
        }

        let preview_count = rows.min(8);
        let mut output_preview_bytes = vec![0_u8; preview_count * std::mem::size_of::<f32>()];
        if let Err(err) = output.copy_to_host(0, &mut output_preview_bytes, Some(&mut stream)) {
            eprintln!("failed to copy self-attn matvec preview for {projection_name}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!("failed to synchronize runtime stream after self-attn preview copy: {err}");
            return ExitCode::from(1);
        }
        let preview = decode_f32_le_values(&output_preview_bytes);
        println!(
            "package-self-attn-proj-smoke package={} layer={} projection={} tensor=\"{}\" hidden={} rows={} backend={} device_index={} name=\"{}\" preview={} verified=true",
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

fn package_self_attn_qk_norm_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-self-attn-qk-norm-smoke requires a .ullm.d path");
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

    let q_tensor = format!("model.language_model.layers.{layer_index}.self_attn.q_proj.weight");
    let k_tensor = format!("model.language_model.layers.{layer_index}.self_attn.k_proj.weight");
    let q_norm_tensor =
        format!("model.language_model.layers.{layer_index}.self_attn.q_norm.weight");
    let k_norm_tensor =
        format!("model.language_model.layers.{layer_index}.self_attn.k_norm.weight");

    let q_norm = match read_named_passthrough_f32(&path, &q_norm_tensor, chunk_bytes) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let k_norm = match read_named_passthrough_f32(&path, &k_norm_tensor, chunk_bytes) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    if q_norm.values.is_empty() || k_norm.values.is_empty() {
        eprintln!("self-attn q/k norm weights must not be empty");
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
    let (q_rows, q_cols, q_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &q_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize tensor {q_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    let (k_rows, k_cols, k_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &k_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize tensor {k_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    if q_cols != k_cols {
        eprintln!("self-attn q/k projection hidden sizes differ: q_cols={q_cols}, k_cols={k_cols}");
        return ExitCode::from(1);
    }
    if q_rows % q_norm.values.len() != 0 {
        eprintln!(
            "q projection rows must be a multiple of q_norm elements: rows={q_rows}, q_norm={}",
            q_norm.values.len()
        );
        return ExitCode::from(1);
    }
    if k_rows % k_norm.values.len() != 0 {
        eprintln!(
            "k projection rows must be a multiple of k_norm elements: rows={k_rows}, k_norm={}",
            k_norm.values.len()
        );
        return ExitCode::from(1);
    }

    let input = deterministic_f32_vector(q_cols);
    let input_bytes = encode_f32_to_bytes(&input);
    let mut input_buffer = match context.alloc_buffer(input_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate self-attn q/k input buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = input_buffer.copy_from_host(0, &input_bytes, Some(&mut stream)) {
        eprintln!("failed to copy deterministic self-attn q/k input data: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after self-attn q/k input copy: {err}");
        return ExitCode::from(1);
    }

    let q_projected = match runtime_matvec_to_host_f32(
        &mut context,
        &mut stream,
        &q_matrix,
        &input_buffer,
        q_rows,
        q_cols,
        "self-attn q projection",
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let k_projected = match runtime_matvec_to_host_f32(
        &mut context,
        &mut stream,
        &k_matrix,
        &input_buffer,
        k_rows,
        k_cols,
        "self-attn k projection",
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };

    let epsilon = 1e-5_f32;
    let (q_normed, q_max_abs_diff) = match runtime_headwise_rmsnorm_verify(
        &mut context,
        &mut stream,
        &q_projected,
        &q_norm.values,
        epsilon,
        "package-self-attn-qk-norm-smoke q_norm",
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let (k_normed, k_max_abs_diff) = match runtime_headwise_rmsnorm_verify(
        &mut context,
        &mut stream,
        &k_projected,
        &k_norm.values,
        epsilon,
        "package-self-attn-qk-norm-smoke k_norm",
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };

    println!(
        "package-self-attn-qk-norm-smoke package={} layer={} q_tensor=\"{}\" k_tensor=\"{}\" q_norm_tensor=\"{}\" k_norm_tensor=\"{}\" hidden={} q_rows={} k_rows={} q_head_dim={} k_head_dim={} q_heads={} k_heads={} q_norm_dtype={} k_norm_dtype={} backend={} device_index={} name=\"{}\" q_preview={} k_preview={} q_norm_preview={} k_norm_preview={} q_norm_max_abs_diff={q_max_abs_diff:.9} k_norm_max_abs_diff={k_max_abs_diff:.9} verified=true",
        path,
        layer_index,
        q_tensor,
        k_tensor,
        q_norm_tensor,
        k_norm_tensor,
        q_cols,
        q_rows,
        k_rows,
        q_norm.values.len(),
        k_norm.values.len(),
        q_rows / q_norm.values.len(),
        k_rows / k_norm.values.len(),
        q_norm.dtype,
        k_norm.dtype,
        info.backend,
        device_index,
        info.name,
        format_f32_preview(&q_projected[..8.min(q_projected.len())]),
        format_f32_preview(&k_projected[..8.min(k_projected.len())]),
        format_f32_preview(&q_normed[..8.min(q_normed.len())]),
        format_f32_preview(&k_normed[..8.min(k_normed.len())]),
    );
    ExitCode::SUCCESS
}

fn package_self_attn_rope_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    sequence_len: Option<String>,
    rotary_dim: Option<String>,
    rope_base: Option<String>,
    position_offset: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-self-attn-rope-smoke requires a .ullm.d path");
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
    let layer_index = match parse_optional_usize(layer_index, 3, "layer index") {
        Ok(value) => value,
        Err(code) => return code,
    };
    let sequence_len = match parse_optional_usize(sequence_len, 2, "sequence length") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("sequence length must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let rope_base = match parse_optional_f32(rope_base, 10_000_000.0, "rope base") {
        Ok(value) if value > 1.0 => value,
        Ok(_) => {
            eprintln!("rope base must be greater than one");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let position_offset = match parse_optional_usize(position_offset, 3, "position offset") {
        Ok(value) => value,
        Err(code) => return code,
    };

    let q_tensor = format!("model.language_model.layers.{layer_index}.self_attn.q_proj.weight");
    let k_tensor = format!("model.language_model.layers.{layer_index}.self_attn.k_proj.weight");
    let q_norm_tensor =
        format!("model.language_model.layers.{layer_index}.self_attn.q_norm.weight");
    let k_norm_tensor =
        format!("model.language_model.layers.{layer_index}.self_attn.k_norm.weight");

    let q_norm = match read_named_passthrough_f32(&path, &q_norm_tensor, chunk_bytes) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let k_norm = match read_named_passthrough_f32(&path, &k_norm_tensor, chunk_bytes) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let q_head_dim = q_norm.values.len();
    let k_head_dim = k_norm.values.len();
    if q_head_dim == 0 || k_head_dim == 0 {
        eprintln!("self-attn q/k norm weights must not be empty");
        return ExitCode::from(1);
    }
    if q_head_dim != k_head_dim {
        eprintln!(
            "self-attn q/k head dims differ: q_head_dim={q_head_dim}, k_head_dim={k_head_dim}"
        );
        return ExitCode::from(1);
    }
    let default_rotary_dim = {
        let candidate = if q_head_dim >= 4 {
            q_head_dim / 4
        } else {
            q_head_dim
        };
        candidate - (candidate % 2)
    };
    if default_rotary_dim == 0 {
        eprintln!("default rotary_dim is zero for q_head_dim={q_head_dim}");
        return ExitCode::from(1);
    }
    let rotary_dim = match parse_optional_usize(rotary_dim, default_rotary_dim, "rotary dim") {
        Ok(value) => value,
        Err(code) => return code,
    };
    if rotary_dim == 0 || rotary_dim > q_head_dim || !rotary_dim.is_multiple_of(2) {
        eprintln!(
            "rotary dim must be even and no greater than head_dim: rotary_dim={rotary_dim}, head_dim={q_head_dim}"
        );
        return ExitCode::from(2);
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
    let (q_rows, q_cols, q_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &q_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize tensor {q_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    let (k_rows, k_cols, k_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &k_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize tensor {k_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    if q_cols != k_cols {
        eprintln!("self-attn q/k projection hidden sizes differ: q_cols={q_cols}, k_cols={k_cols}");
        return ExitCode::from(1);
    }
    if q_rows % q_head_dim != 0 {
        eprintln!(
            "q projection rows must be a multiple of q_head_dim: rows={q_rows}, q_head_dim={q_head_dim}"
        );
        return ExitCode::from(1);
    }
    if k_rows % k_head_dim != 0 {
        eprintln!(
            "k projection rows must be a multiple of k_head_dim: rows={k_rows}, k_head_dim={k_head_dim}"
        );
        return ExitCode::from(1);
    }
    let q_heads = q_rows / q_head_dim;
    let k_heads = k_rows / k_head_dim;

    let hidden_bytes = match q_cols.checked_mul(std::mem::size_of::<f32>()) {
        Some(value) => value,
        None => {
            eprintln!("hidden input byte size overflows");
            return ExitCode::from(1);
        }
    };
    let base_input = deterministic_f32_vector(q_cols);
    let mut input_buffer = match context.alloc_buffer(hidden_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate self-attn rope input buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut q_projected = Vec::with_capacity(sequence_len * q_rows);
    let mut k_projected = Vec::with_capacity(sequence_len * k_rows);
    for timestep in 0..sequence_len {
        let step_input = linear_attn_step_input(&base_input, timestep);
        let step_input_bytes = encode_f32_to_bytes(&step_input);
        if let Err(err) = input_buffer.copy_from_host(0, &step_input_bytes, Some(&mut stream)) {
            eprintln!("failed to copy self-attn rope timestep {timestep} input: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!(
                "failed to synchronize after self-attn rope timestep {timestep} input copy: {err}"
            );
            return ExitCode::from(1);
        }
        let q_step = match runtime_matvec_to_host_f32(
            &mut context,
            &mut stream,
            &q_matrix,
            &input_buffer,
            q_rows,
            q_cols,
            "self-attn rope q projection",
        ) {
            Ok(value) => value,
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::from(1);
            }
        };
        let k_step = match runtime_matvec_to_host_f32(
            &mut context,
            &mut stream,
            &k_matrix,
            &input_buffer,
            k_rows,
            k_cols,
            "self-attn rope k projection",
        ) {
            Ok(value) => value,
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::from(1);
            }
        };
        q_projected.extend(q_step);
        k_projected.extend(k_step);
    }

    let epsilon = 1e-5_f32;
    let (q_normed, q_norm_max_abs_diff) = match runtime_headwise_rmsnorm_verify(
        &mut context,
        &mut stream,
        &q_projected,
        &q_norm.values,
        epsilon,
        "package-self-attn-rope-smoke q_norm",
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let (k_normed, k_norm_max_abs_diff) = match runtime_headwise_rmsnorm_verify(
        &mut context,
        &mut stream,
        &k_projected,
        &k_norm.values,
        epsilon,
        "package-self-attn-rope-smoke k_norm",
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let (q_rope, q_rope_max_abs_diff) = match runtime_rope_verify(
        &mut context,
        &mut stream,
        &q_normed,
        sequence_len,
        q_heads,
        q_head_dim,
        rotary_dim,
        position_offset,
        rope_base,
        "package-self-attn-rope-smoke q_rope",
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let (k_rope, k_rope_max_abs_diff) = match runtime_rope_verify(
        &mut context,
        &mut stream,
        &k_normed,
        sequence_len,
        k_heads,
        k_head_dim,
        rotary_dim,
        position_offset,
        rope_base,
        "package-self-attn-rope-smoke k_rope",
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };

    println!(
        "package-self-attn-rope-smoke package={} layer={} q_tensor=\"{}\" k_tensor=\"{}\" q_norm_tensor=\"{}\" k_norm_tensor=\"{}\" hidden={} sequence_len={} q_rows={} k_rows={} q_heads={} k_heads={} head_dim={} rotary_dim={} position_offset={} rope_base={} q_norm_dtype={} k_norm_dtype={} backend={} device_index={} name=\"{}\" q_norm_preview={} k_norm_preview={} q_rope_preview={} k_rope_preview={} q_norm_max_abs_diff={q_norm_max_abs_diff:.9} k_norm_max_abs_diff={k_norm_max_abs_diff:.9} q_rope_max_abs_diff={q_rope_max_abs_diff:.9} k_rope_max_abs_diff={k_rope_max_abs_diff:.9} verified=true",
        path,
        layer_index,
        q_tensor,
        k_tensor,
        q_norm_tensor,
        k_norm_tensor,
        q_cols,
        sequence_len,
        q_rows,
        k_rows,
        q_heads,
        k_heads,
        q_head_dim,
        rotary_dim,
        position_offset,
        rope_base,
        q_norm.dtype,
        k_norm.dtype,
        info.backend,
        device_index,
        info.name,
        format_f32_preview(&q_normed[..8.min(q_normed.len())]),
        format_f32_preview(&k_normed[..8.min(k_normed.len())]),
        format_f32_preview(&q_rope[..8.min(q_rope.len())]),
        format_f32_preview(&k_rope[..8.min(k_rope.len())]),
    );
    ExitCode::SUCCESS
}

fn package_self_attn_attention_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    sequence_len: Option<String>,
    rotary_dim: Option<String>,
    rope_base: Option<String>,
    position_offset: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-self-attn-attention-smoke requires a .ullm.d path");
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
    let layer_index = match parse_optional_usize(layer_index, 3, "layer index") {
        Ok(value) => value,
        Err(code) => return code,
    };
    let sequence_len = match parse_optional_usize(sequence_len, 2, "sequence length") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("sequence length must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let rope_base = match parse_optional_f32(rope_base, 10_000_000.0, "rope base") {
        Ok(value) if value > 1.0 => value,
        Ok(_) => {
            eprintln!("rope base must be greater than one");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let position_offset = match parse_optional_usize(position_offset, 3, "position offset") {
        Ok(value) => value,
        Err(code) => return code,
    };

    let q_tensor = format!("model.language_model.layers.{layer_index}.self_attn.q_proj.weight");
    let k_tensor = format!("model.language_model.layers.{layer_index}.self_attn.k_proj.weight");
    let v_tensor = format!("model.language_model.layers.{layer_index}.self_attn.v_proj.weight");
    let q_norm_tensor =
        format!("model.language_model.layers.{layer_index}.self_attn.q_norm.weight");
    let k_norm_tensor =
        format!("model.language_model.layers.{layer_index}.self_attn.k_norm.weight");

    let q_norm = match read_named_passthrough_f32(&path, &q_norm_tensor, chunk_bytes) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let k_norm = match read_named_passthrough_f32(&path, &k_norm_tensor, chunk_bytes) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let head_dim = q_norm.values.len();
    if head_dim == 0 || k_norm.values.len() != head_dim {
        eprintln!(
            "self-attn q/k norm head dims must be nonzero and equal: q_head_dim={} k_head_dim={}",
            head_dim,
            k_norm.values.len()
        );
        return ExitCode::from(1);
    }
    let default_rotary_dim = {
        let candidate = if head_dim >= 4 {
            head_dim / 4
        } else {
            head_dim
        };
        candidate - (candidate % 2)
    };
    if default_rotary_dim == 0 {
        eprintln!("default rotary_dim is zero for head_dim={head_dim}");
        return ExitCode::from(1);
    }
    let rotary_dim = match parse_optional_usize(rotary_dim, default_rotary_dim, "rotary dim") {
        Ok(value) => value,
        Err(code) => return code,
    };
    if rotary_dim == 0 || rotary_dim > head_dim || !rotary_dim.is_multiple_of(2) {
        eprintln!(
            "rotary dim must be even and no greater than head_dim: rotary_dim={rotary_dim}, head_dim={head_dim}"
        );
        return ExitCode::from(2);
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
    let (q_rows, q_cols, q_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &q_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize tensor {q_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    let (k_rows, k_cols, k_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &k_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize tensor {k_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    let (v_rows, v_cols, v_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &v_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize tensor {v_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    if q_cols != k_cols || q_cols != v_cols {
        eprintln!(
            "self-attn q/k/v projection hidden sizes differ: q_cols={q_cols}, k_cols={k_cols}, v_cols={v_cols}"
        );
        return ExitCode::from(1);
    }
    if k_rows % head_dim != 0 {
        eprintln!("k rows must be a multiple of head_dim: k_rows={k_rows}, head_dim={head_dim}");
        return ExitCode::from(1);
    }
    let kv_heads = k_rows / head_dim;
    if kv_heads == 0 {
        eprintln!("kv_heads must be greater than zero");
        return ExitCode::from(1);
    }
    if v_rows % kv_heads != 0 {
        eprintln!("v rows must be a multiple of kv_heads: v_rows={v_rows}, kv_heads={kv_heads}");
        return ExitCode::from(1);
    }
    let value_dim = v_rows / kv_heads;

    let hidden_bytes = match q_cols.checked_mul(std::mem::size_of::<f32>()) {
        Some(value) => value,
        None => {
            eprintln!("hidden input byte size overflows");
            return ExitCode::from(1);
        }
    };
    let base_input = deterministic_f32_vector(q_cols);
    let mut input_buffer = match context.alloc_buffer(hidden_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate self-attn attention input buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut q_projected = Vec::with_capacity(sequence_len * q_rows);
    let mut k_projected = Vec::with_capacity(sequence_len * k_rows);
    let mut v_projected = Vec::with_capacity(sequence_len * v_rows);
    for timestep in 0..sequence_len {
        let step_input = linear_attn_step_input(&base_input, timestep);
        let step_input_bytes = encode_f32_to_bytes(&step_input);
        if let Err(err) = input_buffer.copy_from_host(0, &step_input_bytes, Some(&mut stream)) {
            eprintln!("failed to copy self-attn attention timestep {timestep} input: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!(
                "failed to synchronize after self-attn attention timestep {timestep} input copy: {err}"
            );
            return ExitCode::from(1);
        }
        let q_step = match runtime_matvec_to_host_f32(
            &mut context,
            &mut stream,
            &q_matrix,
            &input_buffer,
            q_rows,
            q_cols,
            "self-attn attention q projection",
        ) {
            Ok(value) => value,
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::from(1);
            }
        };
        let k_step = match runtime_matvec_to_host_f32(
            &mut context,
            &mut stream,
            &k_matrix,
            &input_buffer,
            k_rows,
            k_cols,
            "self-attn attention k projection",
        ) {
            Ok(value) => value,
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::from(1);
            }
        };
        let v_step = match runtime_matvec_to_host_f32(
            &mut context,
            &mut stream,
            &v_matrix,
            &input_buffer,
            v_rows,
            v_cols,
            "self-attn attention v projection",
        ) {
            Ok(value) => value,
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::from(1);
            }
        };
        q_projected.extend(q_step);
        k_projected.extend(k_step);
        v_projected.extend(v_step);
    }
    let q_projection_split =
        match split_self_attn_q_projection(&q_projected, sequence_len, q_rows, q_cols, head_dim) {
            Ok(value) => value,
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::from(1);
            }
        };
    let q_heads = q_projection_split.q_heads;
    if !q_heads.is_multiple_of(kv_heads) {
        eprintln!("q_heads must be a multiple of kv_heads: q_heads={q_heads}, kv_heads={kv_heads}");
        return ExitCode::from(1);
    }
    let q_projection_layout = q_projection_split.layout;
    let q_gate_elements = q_projection_split.gate.as_ref().map_or(0, Vec::len);
    let q_projected = q_projection_split.query;

    let epsilon = 1e-5_f32;
    let (q_normed, q_norm_max_abs_diff) = match runtime_headwise_rmsnorm_verify(
        &mut context,
        &mut stream,
        &q_projected,
        &q_norm.values,
        epsilon,
        "package-self-attn-attention-smoke q_norm",
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let (k_normed, k_norm_max_abs_diff) = match runtime_headwise_rmsnorm_verify(
        &mut context,
        &mut stream,
        &k_projected,
        &k_norm.values,
        epsilon,
        "package-self-attn-attention-smoke k_norm",
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let (q_rope, q_rope_max_abs_diff) = match runtime_rope_verify(
        &mut context,
        &mut stream,
        &q_normed,
        sequence_len,
        q_heads,
        head_dim,
        rotary_dim,
        position_offset,
        rope_base,
        "package-self-attn-attention-smoke q_rope",
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let (k_rope, k_rope_max_abs_diff) = match runtime_rope_verify(
        &mut context,
        &mut stream,
        &k_normed,
        sequence_len,
        kv_heads,
        head_dim,
        rotary_dim,
        position_offset,
        rope_base,
        "package-self-attn-attention-smoke k_rope",
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let softmax_scale = 1.0_f32 / (head_dim as f32).sqrt();
    let (attention_output, attention_max_abs_diff) = match runtime_causal_attn_verify(
        &mut context,
        &mut stream,
        &q_rope,
        &k_rope,
        &v_projected,
        sequence_len,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        "package-self-attn-attention-smoke attention",
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };

    println!(
        "package-self-attn-attention-smoke package={} layer={} q_tensor=\"{}\" k_tensor=\"{}\" v_tensor=\"{}\" q_norm_tensor=\"{}\" k_norm_tensor=\"{}\" hidden={} sequence_len={} q_projection_layout={} q_gate_elements={} q_heads={} kv_heads={} head_dim={} value_dim={} rotary_dim={} position_offset={} rope_base={} softmax_scale={softmax_scale:.9} q_norm_dtype={} k_norm_dtype={} backend={} device_index={} name=\"{}\" q_rope_preview={} k_rope_preview={} v_preview={} attention_preview={} q_norm_max_abs_diff={q_norm_max_abs_diff:.9} k_norm_max_abs_diff={k_norm_max_abs_diff:.9} q_rope_max_abs_diff={q_rope_max_abs_diff:.9} k_rope_max_abs_diff={k_rope_max_abs_diff:.9} attention_max_abs_diff={attention_max_abs_diff:.9} verified=true",
        path,
        layer_index,
        q_tensor,
        k_tensor,
        v_tensor,
        q_norm_tensor,
        k_norm_tensor,
        q_cols,
        sequence_len,
        q_projection_layout,
        q_gate_elements,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        rotary_dim,
        position_offset,
        rope_base,
        q_norm.dtype,
        k_norm.dtype,
        info.backend,
        device_index,
        info.name,
        format_f32_preview(&q_rope[..8.min(q_rope.len())]),
        format_f32_preview(&k_rope[..8.min(k_rope.len())]),
        format_f32_preview(&v_projected[..8.min(v_projected.len())]),
        format_f32_preview(&attention_output[..8.min(attention_output.len())]),
    );
    ExitCode::SUCCESS
}

fn package_self_attn_decode_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    sequence_len: Option<String>,
    rotary_dim: Option<String>,
    rope_base: Option<String>,
    position_offset: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-self-attn-decode-smoke requires a .ullm.d path");
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
    let layer_index = match parse_optional_usize(layer_index, 3, "layer index") {
        Ok(value) => value,
        Err(code) => return code,
    };
    let sequence_len = match parse_optional_usize(sequence_len, 3, "sequence length") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("sequence length must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let rope_base = match parse_optional_f32(rope_base, 10_000_000.0, "rope base") {
        Ok(value) if value > 1.0 => value,
        Ok(_) => {
            eprintln!("rope base must be greater than one");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let position_offset = match parse_optional_usize(position_offset, 3, "position offset") {
        Ok(value) => value,
        Err(code) => return code,
    };

    let q_tensor = format!("model.language_model.layers.{layer_index}.self_attn.q_proj.weight");
    let k_tensor = format!("model.language_model.layers.{layer_index}.self_attn.k_proj.weight");
    let v_tensor = format!("model.language_model.layers.{layer_index}.self_attn.v_proj.weight");
    let q_norm_tensor =
        format!("model.language_model.layers.{layer_index}.self_attn.q_norm.weight");
    let k_norm_tensor =
        format!("model.language_model.layers.{layer_index}.self_attn.k_norm.weight");

    let q_norm = match read_named_passthrough_f32(&path, &q_norm_tensor, chunk_bytes) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let k_norm = match read_named_passthrough_f32(&path, &k_norm_tensor, chunk_bytes) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let head_dim = q_norm.values.len();
    if head_dim == 0 || k_norm.values.len() != head_dim {
        eprintln!(
            "self-attn q/k norm head dims must be nonzero and equal: q_head_dim={} k_head_dim={}",
            head_dim,
            k_norm.values.len()
        );
        return ExitCode::from(1);
    }
    let default_rotary_dim = {
        let candidate = if head_dim >= 4 {
            head_dim / 4
        } else {
            head_dim
        };
        candidate - (candidate % 2)
    };
    if default_rotary_dim == 0 {
        eprintln!("default rotary_dim is zero for head_dim={head_dim}");
        return ExitCode::from(1);
    }
    let rotary_dim = match parse_optional_usize(rotary_dim, default_rotary_dim, "rotary dim") {
        Ok(value) => value,
        Err(code) => return code,
    };
    if rotary_dim == 0 || rotary_dim > head_dim || !rotary_dim.is_multiple_of(2) {
        eprintln!(
            "rotary dim must be even and no greater than head_dim: rotary_dim={rotary_dim}, head_dim={head_dim}"
        );
        return ExitCode::from(2);
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
    let (q_rows, q_cols, q_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &q_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize tensor {q_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    let (k_rows, k_cols, k_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &k_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize tensor {k_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    let (v_rows, v_cols, v_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &v_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize tensor {v_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    if q_cols != k_cols || q_cols != v_cols {
        eprintln!(
            "self-attn q/k/v projection hidden sizes differ: q_cols={q_cols}, k_cols={k_cols}, v_cols={v_cols}"
        );
        return ExitCode::from(1);
    }
    if k_rows % head_dim != 0 {
        eprintln!("k rows must be a multiple of head_dim: k_rows={k_rows}, head_dim={head_dim}");
        return ExitCode::from(1);
    }
    let kv_heads = k_rows / head_dim;
    if kv_heads == 0 {
        eprintln!("kv_heads must be greater than zero");
        return ExitCode::from(1);
    }
    if v_rows % kv_heads != 0 {
        eprintln!("v rows must be a multiple of kv_heads: v_rows={v_rows}, kv_heads={kv_heads}");
        return ExitCode::from(1);
    }
    let value_dim = v_rows / kv_heads;

    let hidden_bytes = match q_cols.checked_mul(std::mem::size_of::<f32>()) {
        Some(value) => value,
        None => {
            eprintln!("hidden input byte size overflows");
            return ExitCode::from(1);
        }
    };
    let base_input = deterministic_f32_vector(q_cols);
    let mut input_buffer = match context.alloc_buffer(hidden_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate self-attn decode input buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut q_projected = Vec::with_capacity(sequence_len * q_rows);
    let mut k_projected = Vec::with_capacity(sequence_len * k_rows);
    let mut v_projected = Vec::with_capacity(sequence_len * v_rows);
    for timestep in 0..sequence_len {
        let step_input = linear_attn_step_input(&base_input, timestep);
        let step_input_bytes = encode_f32_to_bytes(&step_input);
        if let Err(err) = input_buffer.copy_from_host(0, &step_input_bytes, Some(&mut stream)) {
            eprintln!("failed to copy self-attn decode timestep {timestep} input: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!(
                "failed to synchronize after self-attn decode timestep {timestep} input copy: {err}"
            );
            return ExitCode::from(1);
        }
        let q_step = match runtime_matvec_to_host_f32(
            &mut context,
            &mut stream,
            &q_matrix,
            &input_buffer,
            q_rows,
            q_cols,
            "self-attn decode q projection",
        ) {
            Ok(value) => value,
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::from(1);
            }
        };
        let k_step = match runtime_matvec_to_host_f32(
            &mut context,
            &mut stream,
            &k_matrix,
            &input_buffer,
            k_rows,
            k_cols,
            "self-attn decode k projection",
        ) {
            Ok(value) => value,
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::from(1);
            }
        };
        let v_step = match runtime_matvec_to_host_f32(
            &mut context,
            &mut stream,
            &v_matrix,
            &input_buffer,
            v_rows,
            v_cols,
            "self-attn decode v projection",
        ) {
            Ok(value) => value,
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::from(1);
            }
        };
        q_projected.extend(q_step);
        k_projected.extend(k_step);
        v_projected.extend(v_step);
    }
    let q_projection_split =
        match split_self_attn_q_projection(&q_projected, sequence_len, q_rows, q_cols, head_dim) {
            Ok(value) => value,
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::from(1);
            }
        };
    let q_heads = q_projection_split.q_heads;
    if !q_heads.is_multiple_of(kv_heads) {
        eprintln!("q_heads must be a multiple of kv_heads: q_heads={q_heads}, kv_heads={kv_heads}");
        return ExitCode::from(1);
    }
    let q_projection_layout = q_projection_split.layout;
    let q_gate_elements = q_projection_split.gate.as_ref().map_or(0, Vec::len);
    let q_projected = q_projection_split.query;

    let epsilon = 1e-5_f32;
    let (q_normed, q_norm_max_abs_diff) = match runtime_headwise_rmsnorm_verify(
        &mut context,
        &mut stream,
        &q_projected,
        &q_norm.values,
        epsilon,
        "package-self-attn-decode-smoke q_norm",
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let (k_normed, k_norm_max_abs_diff) = match runtime_headwise_rmsnorm_verify(
        &mut context,
        &mut stream,
        &k_projected,
        &k_norm.values,
        epsilon,
        "package-self-attn-decode-smoke k_norm",
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let (q_rope, q_rope_max_abs_diff) = match runtime_rope_verify(
        &mut context,
        &mut stream,
        &q_normed,
        sequence_len,
        q_heads,
        head_dim,
        rotary_dim,
        position_offset,
        rope_base,
        "package-self-attn-decode-smoke q_rope",
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let (k_rope, k_rope_max_abs_diff) = match runtime_rope_verify(
        &mut context,
        &mut stream,
        &k_normed,
        sequence_len,
        kv_heads,
        head_dim,
        rotary_dim,
        position_offset,
        rope_base,
        "package-self-attn-decode-smoke k_rope",
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let softmax_scale = 1.0_f32 / (head_dim as f32).sqrt();
    let (attention_output, attention_max_abs_diff) = match runtime_causal_attn_verify(
        &mut context,
        &mut stream,
        &q_rope,
        &k_rope,
        &v_projected,
        sequence_len,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        "package-self-attn-decode-smoke causal reference",
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };

    let decode_q_start = (sequence_len - 1) * q_heads * head_dim;
    let decode_q_end = decode_q_start + q_heads * head_dim;
    let decode_q = &q_rope[decode_q_start..decode_q_end];
    let (decode_output, decode_max_abs_diff) = match runtime_decode_attn_verify(
        &mut context,
        &mut stream,
        decode_q,
        &k_rope,
        &v_projected,
        sequence_len,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        "package-self-attn-decode-smoke decode",
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let paged_block_size = 2_usize;
    let (paged_block_table, paged_cache_blocks, paged_allocator_stats) =
        match allocate_fragmented_paged_decode_blocks(sequence_len, paged_block_size) {
            Ok(value) => value,
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::from(1);
            }
        };
    let paged_decode = match runtime_paged_kv_write_decode_verify(
        &mut context,
        &mut stream,
        &q_rope,
        &k_rope,
        &v_projected,
        &paged_block_table,
        sequence_len,
        paged_block_size,
        paged_cache_blocks,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        "package-self-attn-decode-smoke runtime_paged_kv_write_decode",
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let decode_paged_max_abs_diff = match verify_f32_close(
        "package-self-attn-decode-smoke decode-vs-paged-decode",
        &paged_decode.output,
        &decode_output,
        1e-4,
        1e-4,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let causal_last_start = (sequence_len - 1) * q_heads * value_dim;
    let causal_last_end = causal_last_start + q_heads * value_dim;
    let causal_last = &attention_output[causal_last_start..causal_last_end];
    let causal_decode_max_abs_diff = match verify_f32_close(
        "package-self-attn-decode-smoke causal-vs-decode",
        &decode_output,
        causal_last,
        1e-4,
        1e-4,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let causal_paged_decode_max_abs_diff = match verify_f32_close(
        "package-self-attn-decode-smoke causal-vs-paged-decode",
        &paged_decode.output,
        causal_last,
        1e-4,
        1e-4,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let paged_kv_write_k_max_abs_diff = paged_decode.k_write_max_abs_diff;
    let paged_kv_write_v_max_abs_diff = paged_decode.v_write_max_abs_diff;
    let paged_decode_max_abs_diff = paged_decode.output_max_abs_diff;
    let paged_step_decode_max_abs_diff = paged_decode.step_output_max_abs_diff;
    let causal_paged_step_decode_max_abs_diff = match verify_f32_close(
        "package-self-attn-decode-smoke causal-vs-paged-step-decode",
        &paged_decode.step_outputs,
        &attention_output,
        1e-4,
        1e-4,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };

    println!(
        "package-self-attn-decode-smoke package={} layer={} q_tensor=\"{}\" k_tensor=\"{}\" v_tensor=\"{}\" q_norm_tensor=\"{}\" k_norm_tensor=\"{}\" hidden={} cache_len={} paged_block_size={} paged_cache_blocks={} paged_block_table={:?} paged_allocator_free_blocks={} paged_allocator_allocated_blocks={} paged_allocator_free_runs={} paged_allocator_largest_free_run={} q_projection_layout={} q_gate_elements={} q_heads={} kv_heads={} head_dim={} value_dim={} rotary_dim={} position_offset={} rope_base={} softmax_scale={softmax_scale:.9} q_norm_dtype={} k_norm_dtype={} backend={} device_index={} name=\"{}\" decode_q_preview={} k_cache_preview={} v_cache_preview={} paged_k_cache_preview={} paged_v_cache_preview={} causal_last_preview={} decode_preview={} paged_decode_preview={} q_norm_max_abs_diff={q_norm_max_abs_diff:.9} k_norm_max_abs_diff={k_norm_max_abs_diff:.9} q_rope_max_abs_diff={q_rope_max_abs_diff:.9} k_rope_max_abs_diff={k_rope_max_abs_diff:.9} attention_max_abs_diff={attention_max_abs_diff:.9} decode_max_abs_diff={decode_max_abs_diff:.9} paged_kv_write_k_max_abs_diff={paged_kv_write_k_max_abs_diff:.9} paged_kv_write_v_max_abs_diff={paged_kv_write_v_max_abs_diff:.9} paged_decode_max_abs_diff={paged_decode_max_abs_diff:.9} paged_step_decode_max_abs_diff={paged_step_decode_max_abs_diff:.9} decode_paged_max_abs_diff={decode_paged_max_abs_diff:.9} causal_decode_max_abs_diff={causal_decode_max_abs_diff:.9} causal_paged_decode_max_abs_diff={causal_paged_decode_max_abs_diff:.9} causal_paged_step_decode_max_abs_diff={causal_paged_step_decode_max_abs_diff:.9} verified=true",
        path,
        layer_index,
        q_tensor,
        k_tensor,
        v_tensor,
        q_norm_tensor,
        k_norm_tensor,
        q_cols,
        sequence_len,
        paged_block_size,
        paged_cache_blocks,
        paged_block_table,
        paged_allocator_stats.free_blocks,
        paged_allocator_stats.allocated_blocks,
        paged_allocator_stats.free_runs,
        paged_allocator_stats.largest_free_run,
        q_projection_layout,
        q_gate_elements,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        rotary_dim,
        position_offset,
        rope_base,
        q_norm.dtype,
        k_norm.dtype,
        info.backend,
        device_index,
        info.name,
        format_f32_preview(&decode_q[..8.min(decode_q.len())]),
        format_f32_preview(&k_rope[..8.min(k_rope.len())]),
        format_f32_preview(&v_projected[..8.min(v_projected.len())]),
        format_f32_preview(&paged_decode.k_cache[..8.min(paged_decode.k_cache.len())]),
        format_f32_preview(&paged_decode.v_cache[..8.min(paged_decode.v_cache.len())]),
        format_f32_preview(&causal_last[..8.min(causal_last.len())]),
        format_f32_preview(&decode_output[..8.min(decode_output.len())]),
        format_f32_preview(&paged_decode.output[..8.min(paged_decode.output.len())]),
    );
    ExitCode::SUCCESS
}

fn package_self_attn_block_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    sequence_len: Option<String>,
    rotary_dim: Option<String>,
    rope_base: Option<String>,
    position_offset: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-self-attn-block-smoke requires a .ullm.d path");
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
    let layer_index = match parse_optional_usize(layer_index, 3, "layer index") {
        Ok(value) => value,
        Err(code) => return code,
    };
    let sequence_len = match parse_optional_usize(sequence_len, 2, "sequence length") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("sequence length must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let rope_base = match parse_optional_f32(rope_base, 10_000_000.0, "rope base") {
        Ok(value) if value > 1.0 => value,
        Ok(_) => {
            eprintln!("rope base must be greater than one");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let position_offset = match parse_optional_usize(position_offset, 3, "position offset") {
        Ok(value) => value,
        Err(code) => return code,
    };

    let q_tensor = format!("model.language_model.layers.{layer_index}.self_attn.q_proj.weight");
    let k_tensor = format!("model.language_model.layers.{layer_index}.self_attn.k_proj.weight");
    let v_tensor = format!("model.language_model.layers.{layer_index}.self_attn.v_proj.weight");
    let o_tensor = format!("model.language_model.layers.{layer_index}.self_attn.o_proj.weight");
    let q_norm_tensor =
        format!("model.language_model.layers.{layer_index}.self_attn.q_norm.weight");
    let k_norm_tensor =
        format!("model.language_model.layers.{layer_index}.self_attn.k_norm.weight");

    let q_norm = match read_named_passthrough_f32(&path, &q_norm_tensor, chunk_bytes) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let k_norm = match read_named_passthrough_f32(&path, &k_norm_tensor, chunk_bytes) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let head_dim = q_norm.values.len();
    if head_dim == 0 || k_norm.values.len() != head_dim {
        eprintln!(
            "self-attn q/k norm head dims must be nonzero and equal: q_head_dim={} k_head_dim={}",
            head_dim,
            k_norm.values.len()
        );
        return ExitCode::from(1);
    }
    let default_rotary_dim = {
        let candidate = if head_dim >= 4 {
            head_dim / 4
        } else {
            head_dim
        };
        candidate - (candidate % 2)
    };
    if default_rotary_dim == 0 {
        eprintln!("default rotary_dim is zero for head_dim={head_dim}");
        return ExitCode::from(1);
    }
    let rotary_dim = match parse_optional_usize(rotary_dim, default_rotary_dim, "rotary dim") {
        Ok(value) => value,
        Err(code) => return code,
    };
    if rotary_dim == 0 || rotary_dim > head_dim || !rotary_dim.is_multiple_of(2) {
        eprintln!(
            "rotary dim must be even and no greater than head_dim: rotary_dim={rotary_dim}, head_dim={head_dim}"
        );
        return ExitCode::from(2);
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
    let (q_rows, q_cols, q_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &q_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize tensor {q_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    let (k_rows, k_cols, k_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &k_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize tensor {k_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    let (v_rows, v_cols, v_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &v_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize tensor {v_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    let (o_rows, o_cols, o_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &o_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize tensor {o_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    if q_cols != k_cols || q_cols != v_cols {
        eprintln!(
            "self-attn q/k/v projection hidden sizes differ: q_cols={q_cols}, k_cols={k_cols}, v_cols={v_cols}"
        );
        return ExitCode::from(1);
    }
    if k_rows % head_dim != 0 {
        eprintln!("k rows must be a multiple of head_dim: k_rows={k_rows}, head_dim={head_dim}");
        return ExitCode::from(1);
    }
    let kv_heads = k_rows / head_dim;
    if kv_heads == 0 {
        eprintln!("kv_heads must be greater than zero");
        return ExitCode::from(1);
    }
    if v_rows % kv_heads != 0 {
        eprintln!("v rows must be a multiple of kv_heads: v_rows={v_rows}, kv_heads={kv_heads}");
        return ExitCode::from(1);
    }
    let value_dim = v_rows / kv_heads;

    let hidden_bytes = match q_cols.checked_mul(std::mem::size_of::<f32>()) {
        Some(value) => value,
        None => {
            eprintln!("hidden input byte size overflows");
            return ExitCode::from(1);
        }
    };

    let base_input = deterministic_f32_vector(q_cols);
    let mut residual_sequence = Vec::with_capacity(sequence_len * q_cols);
    let mut input_buffer = match context.alloc_buffer(hidden_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate self-attn block input buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut q_projected = Vec::with_capacity(sequence_len * q_rows);
    let mut k_projected = Vec::with_capacity(sequence_len * k_rows);
    let mut v_projected = Vec::with_capacity(sequence_len * v_rows);
    for timestep in 0..sequence_len {
        let step_input = linear_attn_step_input(&base_input, timestep);
        residual_sequence.extend_from_slice(&step_input);
        let step_input_bytes = encode_f32_to_bytes(&step_input);
        if let Err(err) = input_buffer.copy_from_host(0, &step_input_bytes, Some(&mut stream)) {
            eprintln!("failed to copy self-attn block timestep {timestep} input: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!(
                "failed to synchronize after self-attn block timestep {timestep} input copy: {err}"
            );
            return ExitCode::from(1);
        }
        let q_step = match runtime_matvec_to_host_f32(
            &mut context,
            &mut stream,
            &q_matrix,
            &input_buffer,
            q_rows,
            q_cols,
            "self-attn block q projection",
        ) {
            Ok(value) => value,
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::from(1);
            }
        };
        let k_step = match runtime_matvec_to_host_f32(
            &mut context,
            &mut stream,
            &k_matrix,
            &input_buffer,
            k_rows,
            k_cols,
            "self-attn block k projection",
        ) {
            Ok(value) => value,
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::from(1);
            }
        };
        let v_step = match runtime_matvec_to_host_f32(
            &mut context,
            &mut stream,
            &v_matrix,
            &input_buffer,
            v_rows,
            v_cols,
            "self-attn block v projection",
        ) {
            Ok(value) => value,
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::from(1);
            }
        };
        q_projected.extend(q_step);
        k_projected.extend(k_step);
        v_projected.extend(v_step);
    }
    let q_projection_split =
        match split_self_attn_q_projection(&q_projected, sequence_len, q_rows, q_cols, head_dim) {
            Ok(value) => value,
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::from(1);
            }
        };
    let q_heads = q_projection_split.q_heads;
    if !q_heads.is_multiple_of(kv_heads) {
        eprintln!("q_heads must be a multiple of kv_heads: q_heads={q_heads}, kv_heads={kv_heads}");
        return ExitCode::from(1);
    }
    if o_cols != q_heads * value_dim || o_rows != q_cols {
        eprintln!(
            "o projection shape mismatch: o_rows={o_rows}, o_cols={o_cols}, hidden={q_cols}, attention_width={}",
            q_heads * value_dim
        );
        return ExitCode::from(1);
    }
    let q_projection_layout = q_projection_split.layout;
    let q_gate = q_projection_split.gate;
    let q_gate_elements = q_gate.as_ref().map_or(0, Vec::len);
    let q_projected = q_projection_split.query;

    let epsilon = 1e-5_f32;
    let (q_normed, q_norm_max_abs_diff) = match runtime_headwise_rmsnorm_verify(
        &mut context,
        &mut stream,
        &q_projected,
        &q_norm.values,
        epsilon,
        "package-self-attn-block-smoke q_norm",
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let (k_normed, k_norm_max_abs_diff) = match runtime_headwise_rmsnorm_verify(
        &mut context,
        &mut stream,
        &k_projected,
        &k_norm.values,
        epsilon,
        "package-self-attn-block-smoke k_norm",
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let (q_rope, q_rope_max_abs_diff) = match runtime_rope_verify(
        &mut context,
        &mut stream,
        &q_normed,
        sequence_len,
        q_heads,
        head_dim,
        rotary_dim,
        position_offset,
        rope_base,
        "package-self-attn-block-smoke q_rope",
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let (k_rope, k_rope_max_abs_diff) = match runtime_rope_verify(
        &mut context,
        &mut stream,
        &k_normed,
        sequence_len,
        kv_heads,
        head_dim,
        rotary_dim,
        position_offset,
        rope_base,
        "package-self-attn-block-smoke k_rope",
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let softmax_scale = 1.0_f32 / (head_dim as f32).sqrt();
    let (attention_output, attention_max_abs_diff) = match runtime_causal_attn_verify(
        &mut context,
        &mut stream,
        &q_rope,
        &k_rope,
        &v_projected,
        sequence_len,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        "package-self-attn-block-smoke attention",
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let (attention_projection_input, output_gate_layout, output_gate_max_abs_diff) = match q_gate
        .as_ref()
    {
        Some(gate) => {
            let expected_gated_attention = runtime_host_sigmoid_mul_f32(&gate, &attention_output);
            if expected_gated_attention.is_empty() {
                eprintln!(
                    "failed to apply Qwen3.5 self-attn output gate: gate_elements={} attention_elements={}",
                    gate.len(),
                    attention_output.len()
                );
                return ExitCode::from(1);
            }
            let gate_bytes = encode_f32_to_bytes(&gate);
            let attention_bytes = encode_f32_to_bytes(&attention_output);
            let output_gate_bytes_len = attention_bytes.len();
            let mut gate_buffer = match context.alloc_buffer(gate_bytes.len()) {
                Ok(buffer) => buffer,
                Err(err) => {
                    eprintln!("failed to allocate self-attn output gate buffer: {err}");
                    return ExitCode::from(1);
                }
            };
            let mut attention_buffer = match context.alloc_buffer(output_gate_bytes_len) {
                Ok(buffer) => buffer,
                Err(err) => {
                    eprintln!("failed to allocate self-attn output gate input buffer: {err}");
                    return ExitCode::from(1);
                }
            };
            let mut gated_attention_buffer = match context.alloc_buffer(output_gate_bytes_len) {
                Ok(buffer) => buffer,
                Err(err) => {
                    eprintln!("failed to allocate self-attn gated attention buffer: {err}");
                    return ExitCode::from(1);
                }
            };
            if let Err(err) = gate_buffer.copy_from_host(0, &gate_bytes, Some(&mut stream)) {
                eprintln!("failed to copy self-attn output gate into runtime buffer: {err}");
                return ExitCode::from(1);
            }
            if let Err(err) =
                attention_buffer.copy_from_host(0, &attention_bytes, Some(&mut stream))
            {
                eprintln!("failed to copy self-attn attention into runtime gate input: {err}");
                return ExitCode::from(1);
            }
            if let Err(err) = stream.synchronize() {
                eprintln!("failed to synchronize after self-attn output gate input copy: {err}");
                return ExitCode::from(1);
            }
            if let Err(err) = ullm_runtime_sys::sigmoid_mul_f32(
                &gate_buffer,
                &attention_buffer,
                attention_output.len(),
                &mut gated_attention_buffer,
                Some(&mut stream),
            ) {
                eprintln!("failed to run self-attn output gate sigmoid_mul_f32: {err}");
                return ExitCode::from(1);
            }
            if let Err(err) = stream.synchronize() {
                eprintln!("failed to synchronize after self-attn output gate sigmoid_mul: {err}");
                return ExitCode::from(1);
            }
            let mut gated_attention_raw = vec![0_u8; output_gate_bytes_len];
            if let Err(err) =
                gated_attention_buffer.copy_to_host(0, &mut gated_attention_raw, Some(&mut stream))
            {
                eprintln!("failed to copy self-attn gated attention output: {err}");
                return ExitCode::from(1);
            }
            if let Err(err) = stream.synchronize() {
                eprintln!("failed to synchronize after self-attn gated attention copy: {err}");
                return ExitCode::from(1);
            }
            let gated_attention = decode_f32_le_values(&gated_attention_raw);
            let output_gate_max_abs_diff = match verify_f32_close(
                "package-self-attn-block-smoke output gate",
                &gated_attention,
                &expected_gated_attention,
                1e-5,
                1e-6,
            ) {
                Ok(value) => value,
                Err(err) => {
                    eprintln!("{err}");
                    return ExitCode::from(1);
                }
            };
            (gated_attention, "runtime-sigmoid", output_gate_max_abs_diff)
        }
        None => (attention_output.clone(), "none", 0.0_f32),
    };
    let o_matrix_bytes = match o_rows
        .checked_mul(o_cols)
        .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
    {
        Some(value) => value,
        None => {
            eprintln!("o projection matrix byte size overflows");
            return ExitCode::from(1);
        }
    };
    let mut o_matrix_raw = vec![0_u8; o_matrix_bytes];
    if let Err(err) = o_matrix.copy_to_host(0, &mut o_matrix_raw, Some(&mut stream)) {
        eprintln!("failed to copy materialized o projection to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize after o projection host copy: {err}");
        return ExitCode::from(1);
    }
    let o_matrix_host = decode_f32_le_values(&o_matrix_raw);
    let (attn_projected, o_proj_max_abs_diff) = match runtime_matvec_sequence_to_host_f32(
        &mut context,
        &mut stream,
        &o_matrix,
        &o_matrix_host,
        &attention_projection_input,
        sequence_len,
        o_rows,
        o_cols,
        "package-self-attn-block-smoke o projection",
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };

    let residual_bytes = encode_f32_to_bytes(&residual_sequence);
    let attn_projected_bytes = encode_f32_to_bytes(&attn_projected);
    let block_bytes_len = residual_bytes.len();
    let mut residual_buffer = match context.alloc_buffer(block_bytes_len) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate self-attn residual buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut attn_projected_buffer = match context.alloc_buffer(block_bytes_len) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate self-attn projected buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut block_buffer = match context.alloc_buffer(block_bytes_len) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate self-attn block output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = residual_buffer.copy_from_host(0, &residual_bytes, Some(&mut stream)) {
        eprintln!("failed to copy residual sequence into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) =
        attn_projected_buffer.copy_from_host(0, &attn_projected_bytes, Some(&mut stream))
    {
        eprintln!("failed to copy attention projection into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = ullm_runtime_sys::add_f32(
        &residual_buffer,
        &attn_projected_buffer,
        residual_sequence.len(),
        &mut block_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run self-attn residual add: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize after self-attn residual add: {err}");
        return ExitCode::from(1);
    }
    let mut block_raw = vec![0_u8; block_bytes_len];
    if let Err(err) = block_buffer.copy_to_host(0, &mut block_raw, Some(&mut stream)) {
        eprintln!("failed to copy self-attn block output: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize after self-attn block output copy: {err}");
        return ExitCode::from(1);
    }
    let block_output = decode_f32_le_values(&block_raw);
    let expected_block_output = runtime_host_add_f32(&residual_sequence, &attn_projected);
    let block_max_abs_diff = match verify_f32_close(
        "package-self-attn-block-smoke residual add",
        &block_output,
        &expected_block_output,
        1e-5,
        1e-6,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };

    println!(
        "package-self-attn-block-smoke package={} layer={} q_tensor=\"{}\" k_tensor=\"{}\" v_tensor=\"{}\" o_tensor=\"{}\" q_norm_tensor=\"{}\" k_norm_tensor=\"{}\" hidden={} sequence_len={} q_projection_layout={} q_gate_elements={} output_gate_layout={} q_heads={} kv_heads={} head_dim={} value_dim={} rotary_dim={} position_offset={} rope_base={} softmax_scale={softmax_scale:.9} q_norm_dtype={} k_norm_dtype={} backend={} device_index={} name=\"{}\" attention_preview={} gated_attention_preview={} projected_preview={} block_preview={} q_norm_max_abs_diff={q_norm_max_abs_diff:.9} k_norm_max_abs_diff={k_norm_max_abs_diff:.9} q_rope_max_abs_diff={q_rope_max_abs_diff:.9} k_rope_max_abs_diff={k_rope_max_abs_diff:.9} attention_max_abs_diff={attention_max_abs_diff:.9} output_gate_max_abs_diff={output_gate_max_abs_diff:.9} o_proj_max_abs_diff={o_proj_max_abs_diff:.9} block_max_abs_diff={block_max_abs_diff:.9} verified=true",
        path,
        layer_index,
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        q_norm_tensor,
        k_norm_tensor,
        q_cols,
        sequence_len,
        q_projection_layout,
        q_gate_elements,
        output_gate_layout,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        rotary_dim,
        position_offset,
        rope_base,
        q_norm.dtype,
        k_norm.dtype,
        info.backend,
        device_index,
        info.name,
        format_f32_preview(&attention_output[..8.min(attention_output.len())]),
        format_f32_preview(&attention_projection_input[..8.min(attention_projection_input.len())]),
        format_f32_preview(&attn_projected[..8.min(attn_projected.len())]),
        format_f32_preview(&block_output[..8.min(block_output.len())]),
    );
    ExitCode::SUCCESS
}

struct SelfAttnBlockSmokeRun {
    residual_sequence: Vec<f32>,
    q_rope: Vec<f32>,
    k_rope: Vec<f32>,
    v_projected: Vec<f32>,
    q_gate: Option<Vec<f32>>,
    attention_output: Vec<f32>,
    attention_projection_input: Vec<f32>,
    attn_projected: Vec<f32>,
    block_output: Vec<f32>,
    o_matrix: ullm_runtime_sys::RuntimeBuffer,
    paged_block_table: Vec<u32>,
    paged_block_size: usize,
    paged_cache_blocks: usize,
    hidden: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
    q_projection_layout: &'static str,
    q_gate_elements: usize,
    output_gate_layout: &'static str,
    q_norm_max_abs_diff: f32,
    k_norm_max_abs_diff: f32,
    q_rope_max_abs_diff: f32,
    k_rope_max_abs_diff: f32,
    attention_max_abs_diff: f32,
    paged_kv_write_k_max_abs_diff: f32,
    paged_kv_write_v_max_abs_diff: f32,
    paged_step_attention_max_abs_diff: f32,
    causal_paged_step_attention_max_abs_diff: f32,
    output_gate_max_abs_diff: f32,
    o_proj_max_abs_diff: f32,
    block_max_abs_diff: f32,
    causal_paged_block_max_abs_diff: f32,
}

struct Qwen3MlpRuntimeWeights {
    gate_rows: usize,
    gate_cols: usize,
    gate_matrix: ullm_runtime_sys::RuntimeBuffer,
    up_matrix: ullm_runtime_sys::RuntimeBuffer,
    down_matrix: ullm_runtime_sys::RuntimeBuffer,
}

struct Qwen3DecoderLayerRuntimeWeights {
    hidden: usize,
    intermediate: usize,
    post_norm_weight: ullm_runtime_sys::RuntimeBuffer,
    mlp: Qwen3MlpRuntimeWeights,
}

#[allow(clippy::too_many_arguments)]
fn run_self_attn_block_sequence_smoke(
    context: &mut ullm_runtime_sys::RuntimeContext,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    path: &str,
    chunk_bytes: usize,
    sequence_len: usize,
    rotary_dim: usize,
    rope_base: f32,
    position_offset: usize,
    q_tensor: &str,
    k_tensor: &str,
    v_tensor: &str,
    o_tensor: &str,
    q_norm: &PassthroughF32Data,
    k_norm: &PassthroughF32Data,
    label: &str,
) -> Result<SelfAttnBlockSmokeRun, String> {
    let head_dim = q_norm.values.len();
    if head_dim == 0 || k_norm.values.len() != head_dim {
        return Err(format!(
            "self-attn q/k norm head dims must be nonzero and equal: q_head_dim={} k_head_dim={}",
            head_dim,
            k_norm.values.len()
        ));
    }

    let (q_rows, q_cols, k_rows, v_rows, o_rows, o_cols, q_matrix, k_matrix, v_matrix, o_matrix) = {
        let mut registry = WeightRegistry::new();
        let (q_rows, q_cols, q_matrix) = materialize_selected_aq4_matrix(
            context,
            stream,
            &mut registry,
            path,
            q_tensor,
            chunk_bytes,
        )?;
        let (k_rows, k_cols, k_matrix) = materialize_selected_aq4_matrix(
            context,
            stream,
            &mut registry,
            path,
            k_tensor,
            chunk_bytes,
        )?;
        let (v_rows, v_cols, v_matrix) = materialize_selected_aq4_matrix(
            context,
            stream,
            &mut registry,
            path,
            v_tensor,
            chunk_bytes,
        )?;
        let (o_rows, o_cols, o_matrix) = materialize_selected_aq4_matrix(
            context,
            stream,
            &mut registry,
            path,
            o_tensor,
            chunk_bytes,
        )?;
        if q_cols != k_cols || q_cols != v_cols {
            return Err(format!(
                "self-attn q/k/v projection hidden sizes differ: q_cols={q_cols}, k_cols={k_cols}, v_cols={v_cols}"
            ));
        }
        (
            q_rows, q_cols, k_rows, v_rows, o_rows, o_cols, q_matrix, k_matrix, v_matrix, o_matrix,
        )
    };

    if k_rows % head_dim != 0 {
        return Err(format!(
            "k rows must be a multiple of head_dim: k_rows={k_rows}, head_dim={head_dim}"
        ));
    }
    let kv_heads = k_rows / head_dim;
    if kv_heads == 0 {
        return Err("kv_heads must be greater than zero".to_string());
    }
    if v_rows % kv_heads != 0 {
        return Err(format!(
            "v rows must be a multiple of kv_heads: v_rows={v_rows}, kv_heads={kv_heads}"
        ));
    }
    let value_dim = v_rows / kv_heads;

    let hidden_bytes = q_cols
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "hidden input byte size overflows".to_string())?;

    let base_input = deterministic_f32_vector(q_cols);
    let mut residual_sequence = Vec::with_capacity(sequence_len * q_cols);
    let mut input_buffer = context
        .alloc_buffer(hidden_bytes)
        .map_err(|err| format!("failed to allocate {label} input buffer: {err}"))?;
    let mut q_projected = Vec::with_capacity(sequence_len * q_rows);
    let mut k_projected = Vec::with_capacity(sequence_len * k_rows);
    let mut v_projected = Vec::with_capacity(sequence_len * v_rows);
    for timestep in 0..sequence_len {
        let step_input = linear_attn_step_input(&base_input, timestep);
        residual_sequence.extend_from_slice(&step_input);
        let step_input_bytes = encode_f32_to_bytes(&step_input);
        input_buffer
            .copy_from_host(0, &step_input_bytes, Some(stream))
            .map_err(|err| format!("failed to copy {label} timestep {timestep} input: {err}"))?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize after {label} timestep {timestep} input copy: {err}")
        })?;
        let q_step = runtime_matvec_to_host_f32(
            context,
            stream,
            &q_matrix,
            &input_buffer,
            q_rows,
            q_cols,
            &format!("{label} q projection"),
        )?;
        let k_step = runtime_matvec_to_host_f32(
            context,
            stream,
            &k_matrix,
            &input_buffer,
            k_rows,
            q_cols,
            &format!("{label} k projection"),
        )?;
        let v_step = runtime_matvec_to_host_f32(
            context,
            stream,
            &v_matrix,
            &input_buffer,
            v_rows,
            q_cols,
            &format!("{label} v projection"),
        )?;
        q_projected.extend(q_step);
        k_projected.extend(k_step);
        v_projected.extend(v_step);
    }

    let q_projection_split =
        split_self_attn_q_projection(&q_projected, sequence_len, q_rows, q_cols, head_dim)?;
    let q_heads = q_projection_split.q_heads;
    if !q_heads.is_multiple_of(kv_heads) {
        return Err(format!(
            "q_heads must be a multiple of kv_heads: q_heads={q_heads}, kv_heads={kv_heads}"
        ));
    }
    if o_cols != q_heads * value_dim || o_rows != q_cols {
        return Err(format!(
            "o projection shape mismatch: o_rows={o_rows}, o_cols={o_cols}, hidden={q_cols}, attention_width={}",
            q_heads * value_dim
        ));
    }
    let q_projection_layout = q_projection_split.layout;
    let q_gate = q_projection_split.gate;
    let q_gate_elements = q_gate.as_ref().map_or(0, Vec::len);
    let q_projected = q_projection_split.query;

    let epsilon = 1e-5_f32;
    let (q_normed, q_norm_max_abs_diff) = runtime_headwise_rmsnorm_verify(
        context,
        stream,
        &q_projected,
        &q_norm.values,
        epsilon,
        &format!("{label} q_norm"),
    )?;
    let (k_normed, k_norm_max_abs_diff) = runtime_headwise_rmsnorm_verify(
        context,
        stream,
        &k_projected,
        &k_norm.values,
        epsilon,
        &format!("{label} k_norm"),
    )?;
    let (q_rope, q_rope_max_abs_diff) = runtime_rope_verify(
        context,
        stream,
        &q_normed,
        sequence_len,
        q_heads,
        head_dim,
        rotary_dim,
        position_offset,
        rope_base,
        &format!("{label} q_rope"),
    )?;
    let (k_rope, k_rope_max_abs_diff) = runtime_rope_verify(
        context,
        stream,
        &k_normed,
        sequence_len,
        kv_heads,
        head_dim,
        rotary_dim,
        position_offset,
        rope_base,
        &format!("{label} k_rope"),
    )?;
    let softmax_scale = 1.0_f32 / (head_dim as f32).sqrt();
    let (attention_output, attention_max_abs_diff) = runtime_causal_attn_verify(
        context,
        stream,
        &q_rope,
        &k_rope,
        &v_projected,
        sequence_len,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        &format!("{label} attention"),
    )?;
    let paged_block_size = 2_usize;
    let (paged_block_table, paged_cache_blocks, _) =
        allocate_fragmented_paged_decode_blocks(sequence_len, paged_block_size)?;
    let (expected_paged_k_cache, expected_paged_v_cache) =
        pack_paged_decode_kv_cache_for_block_table(
            &k_rope,
            &v_projected,
            &paged_block_table,
            sequence_len,
            paged_block_size,
            paged_cache_blocks,
            kv_heads,
            head_dim,
            value_dim,
        )?;
    let o_matrix_bytes = o_rows
        .checked_mul(o_cols)
        .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| "o projection matrix byte size overflows".to_string())?;
    let mut o_matrix_raw = vec![0_u8; o_matrix_bytes];
    o_matrix
        .copy_to_host(0, &mut o_matrix_raw, Some(stream))
        .map_err(|err| format!("failed to copy materialized o projection to host: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize after o projection host copy: {err}"))?;
    let o_matrix_host = decode_f32_le_values(&o_matrix_raw);

    let decode_shape = PagedDecodeShape {
        block_size: paged_block_size,
        cache_blocks: paged_cache_blocks,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
    };
    let mut qwen_step_state = Qwen3SelfAttnBlockStepState::new(
        context,
        stream,
        decode_shape,
        paged_block_table.clone(),
        q_cols,
        softmax_scale,
    )?;
    let q_token_elements = q_heads * head_dim;
    let k_token_elements = kv_heads * head_dim;
    let v_token_elements = kv_heads * value_dim;
    let attention_elements = q_heads * value_dim;
    if let Some(gate) = q_gate.as_ref() {
        if gate.len() != sequence_len * attention_elements {
            return Err(format!(
                "Qwen3.5 output gate length {} does not match sequence_len={sequence_len} attention_elements={attention_elements}",
                gate.len()
            ));
        }
    }
    let mut paged_step_attention_output = Vec::with_capacity(sequence_len * attention_elements);
    let mut attention_projection_input = Vec::with_capacity(sequence_len * attention_elements);
    let mut attn_projected = Vec::with_capacity(sequence_len * o_rows);
    let mut block_output = Vec::with_capacity(sequence_len * q_cols);
    let mut paged_step_attention_max_abs_diff = 0.0_f32;
    let mut output_gate_max_abs_diff = 0.0_f32;
    let mut o_proj_max_abs_diff = 0.0_f32;
    let mut block_max_abs_diff = 0.0_f32;

    for timestep in 0..sequence_len {
        let q_start = timestep * q_token_elements;
        let q_end = q_start + q_token_elements;
        let k_start = timestep * k_token_elements;
        let k_end = k_start + k_token_elements;
        let v_start = timestep * v_token_elements;
        let v_end = v_start + v_token_elements;
        let attention_start = timestep * attention_elements;
        let attention_end = attention_start + attention_elements;
        let residual_start = timestep * q_cols;
        let residual_end = residual_start + q_cols;
        let gate_step = q_gate
            .as_ref()
            .map(|gate| &gate[attention_start..attention_end]);
        let step = qwen_step_state
            .step(
                stream,
                &o_matrix,
                &q_rope[q_start..q_end],
                &k_rope[k_start..k_end],
                &v_projected[v_start..v_end],
                gate_step,
                &residual_sequence[residual_start..residual_end],
            )
            .map_err(|err| {
                format!("failed to run {label} Qwen3 self-attn step {timestep}: {err}")
            })?;
        if step.cache_position != timestep {
            return Err(format!(
                "{label} Qwen3 self-attn step wrote position {}, expected {timestep}",
                step.cache_position
            ));
        }
        if step.cache_len != timestep + 1 {
            return Err(format!(
                "{label} Qwen3 self-attn step reported cache_len {}, expected {}",
                step.cache_len,
                timestep + 1
            ));
        }
        let expected_step_output = runtime_host_paged_decode_attn_f32(
            &q_rope[q_start..q_end],
            &expected_paged_k_cache,
            &expected_paged_v_cache,
            &paged_block_table,
            timestep + 1,
            paged_block_size,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
        );
        let step_attention_diff = verify_f32_close(
            &format!("{label} Qwen3 self-attn step {timestep} attention"),
            &step.attention_output,
            &expected_step_output,
            1e-4,
            1e-4,
        )?;
        paged_step_attention_max_abs_diff =
            paged_step_attention_max_abs_diff.max(step_attention_diff);

        let expected_projection_input = if let Some(gate) = gate_step {
            runtime_host_sigmoid_mul_f32(gate, &step.attention_output)
        } else {
            step.attention_output.clone()
        };
        let gate_diff = verify_f32_close(
            &format!("{label} Qwen3 self-attn step {timestep} output gate"),
            &step.attention_projection_input,
            &expected_projection_input,
            1e-5,
            1e-6,
        )?;
        output_gate_max_abs_diff = output_gate_max_abs_diff.max(gate_diff);

        let expected_projected = runtime_host_matvec_f32(
            &o_matrix_host,
            &step.attention_projection_input,
            o_rows,
            o_cols,
        );
        let projected_diff = verify_f32_close(
            &format!("{label} Qwen3 self-attn step {timestep} o projection"),
            &step.projected_output,
            &expected_projected,
            1e-4,
            1e-5,
        )?;
        o_proj_max_abs_diff = o_proj_max_abs_diff.max(projected_diff);

        let expected_block_step = runtime_host_add_f32(
            &residual_sequence[residual_start..residual_end],
            &step.projected_output,
        );
        let block_diff = verify_f32_close(
            &format!("{label} Qwen3 self-attn step {timestep} residual add"),
            &step.block_output,
            &expected_block_step,
            1e-5,
            1e-6,
        )?;
        block_max_abs_diff = block_max_abs_diff.max(block_diff);

        paged_step_attention_output.extend_from_slice(&step.attention_output);
        attention_projection_input.extend_from_slice(&step.attention_projection_input);
        attn_projected.extend_from_slice(&step.projected_output);
        block_output.extend_from_slice(&step.block_output);
    }
    let paged_cache = qwen_step_state
        .read_cache_to_host(stream)
        .map_err(|err| format!("failed to read {label} Qwen3 self-attn paged cache: {err}"))?;
    let paged_kv_write_k_max_abs_diff = verify_f32_close(
        &format!("{label} Qwen3 self-attn paged k cache write"),
        &paged_cache.k,
        &expected_paged_k_cache,
        1e-5,
        1e-5,
    )?;
    let paged_kv_write_v_max_abs_diff = verify_f32_close(
        &format!("{label} Qwen3 self-attn paged v cache write"),
        &paged_cache.v,
        &expected_paged_v_cache,
        1e-5,
        1e-5,
    )?;
    let causal_paged_step_attention_max_abs_diff = verify_f32_close(
        &format!("{label} causal-vs-paged-step-attention"),
        &paged_step_attention_output,
        &attention_output,
        1e-4,
        1e-4,
    )?;
    let output_gate_layout = if q_gate.is_some() {
        "runtime-sigmoid"
    } else {
        "none"
    };
    let causal_attention_projection_input = if let Some(gate) = q_gate.as_ref() {
        runtime_host_sigmoid_mul_f32(gate, &attention_output)
    } else {
        attention_output.clone()
    };
    let mut causal_attn_projected = Vec::with_capacity(sequence_len * o_rows);
    for timestep in 0..sequence_len {
        let input_start = timestep * o_cols;
        let input_end = input_start + o_cols;
        causal_attn_projected.extend(runtime_host_matvec_f32(
            &o_matrix_host,
            &causal_attention_projection_input[input_start..input_end],
            o_rows,
            o_cols,
        ));
    }

    let expected_causal_block_output =
        runtime_host_add_f32(&residual_sequence, &causal_attn_projected);
    let causal_paged_block_max_abs_diff = verify_f32_close(
        &format!("{label} causal-vs-paged-block"),
        &block_output,
        &expected_causal_block_output,
        3e-3,
        2e-5,
    )?;

    Ok(SelfAttnBlockSmokeRun {
        residual_sequence,
        q_rope,
        k_rope,
        v_projected,
        q_gate,
        attention_output,
        attention_projection_input,
        attn_projected,
        block_output,
        o_matrix,
        paged_block_table,
        paged_block_size,
        paged_cache_blocks,
        hidden: q_cols,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        q_projection_layout,
        q_gate_elements,
        output_gate_layout,
        q_norm_max_abs_diff,
        k_norm_max_abs_diff,
        q_rope_max_abs_diff,
        k_rope_max_abs_diff,
        attention_max_abs_diff,
        paged_kv_write_k_max_abs_diff,
        paged_kv_write_v_max_abs_diff,
        paged_step_attention_max_abs_diff,
        causal_paged_step_attention_max_abs_diff,
        output_gate_max_abs_diff,
        o_proj_max_abs_diff,
        block_max_abs_diff,
        causal_paged_block_max_abs_diff,
    })
}

#[allow(clippy::too_many_arguments)]
fn qwen3_decoder_layer_runtime_weights_from_package(
    context: &mut ullm_runtime_sys::RuntimeContext,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    path: &str,
    chunk_bytes: usize,
    hidden: usize,
    post_norm: &PassthroughF32Data,
    gate_tensor: &str,
    up_tensor: &str,
    down_tensor: &str,
) -> Result<Qwen3DecoderLayerRuntimeWeights, String> {
    if post_norm.values.len() != hidden {
        return Err(format!(
            "post RMSNorm length must match hidden={hidden}: len={}",
            post_norm.values.len()
        ));
    }
    let post_norm_weight_bytes = encode_f32_to_bytes(&post_norm.values);
    let mut post_norm_weight_buffer = context
        .alloc_buffer(post_norm_weight_bytes.len())
        .map_err(|err| format!("failed to allocate post RMSNorm weight buffer: {err}"))?;
    post_norm_weight_buffer
        .copy_from_host(0, &post_norm_weight_bytes, Some(stream))
        .map_err(|err| format!("failed to copy post RMSNorm weight into runtime buffer: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize after post RMSNorm weight copy: {err}"))?;

    let mut registry = WeightRegistry::new();
    let (gate_rows, gate_cols, gate_matrix) = materialize_selected_aq4_matrix(
        context,
        stream,
        &mut registry,
        path,
        gate_tensor,
        chunk_bytes,
    )?;
    let (up_rows, up_cols, up_matrix) = materialize_selected_aq4_matrix(
        context,
        stream,
        &mut registry,
        path,
        up_tensor,
        chunk_bytes,
    )?;
    let (down_rows, down_cols, down_matrix) = materialize_selected_aq4_matrix(
        context,
        stream,
        &mut registry,
        path,
        down_tensor,
        chunk_bytes,
    )?;
    if gate_rows != up_rows || gate_cols != up_cols || gate_cols != hidden {
        return Err(format!(
            "MLP gate/up shape mismatch: gate=[{gate_rows},{gate_cols}] up=[{up_rows},{up_cols}] hidden={hidden}"
        ));
    }
    if down_rows != hidden || down_cols != gate_rows {
        return Err(format!(
            "MLP down shape mismatch: expected [{hidden},{gate_rows}], got [{down_rows},{down_cols}]"
        ));
    }
    let intermediate = gate_rows;

    Ok(Qwen3DecoderLayerRuntimeWeights {
        hidden,
        intermediate,
        post_norm_weight: post_norm_weight_buffer,
        mlp: Qwen3MlpRuntimeWeights {
            gate_rows,
            gate_cols,
            gate_matrix,
            up_matrix,
            down_matrix,
        },
    })
}

fn package_self_attn_mlp_block_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    sequence_len: Option<String>,
    rotary_dim: Option<String>,
    rope_base: Option<String>,
    position_offset: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-self-attn-mlp-block-smoke requires a .ullm.d path");
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
    let layer_index = match parse_optional_usize(layer_index, 3, "layer index") {
        Ok(value) => value,
        Err(code) => return code,
    };
    let sequence_len = match parse_optional_usize(sequence_len, 2, "sequence length") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("sequence length must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let rope_base = match parse_optional_f32(rope_base, 10_000_000.0, "rope base") {
        Ok(value) if value > 1.0 => value,
        Ok(_) => {
            eprintln!("rope base must be greater than one");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let position_offset = match parse_optional_usize(position_offset, 3, "position offset") {
        Ok(value) => value,
        Err(code) => return code,
    };

    let q_norm_tensor =
        format!("model.language_model.layers.{layer_index}.self_attn.q_norm.weight");
    let k_norm_tensor =
        format!("model.language_model.layers.{layer_index}.self_attn.k_norm.weight");
    let post_norm_tensor =
        format!("model.language_model.layers.{layer_index}.post_attention_layernorm.weight");

    let q_norm = match read_named_passthrough_f32(&path, &q_norm_tensor, chunk_bytes) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let k_norm = match read_named_passthrough_f32(&path, &k_norm_tensor, chunk_bytes) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let post_norm = match read_named_passthrough_f32(&path, &post_norm_tensor, chunk_bytes) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };

    let head_dim = q_norm.values.len();
    if head_dim == 0 || k_norm.values.len() != head_dim {
        eprintln!(
            "self-attn q/k norm head dims must be nonzero and equal: q_head_dim={} k_head_dim={}",
            head_dim,
            k_norm.values.len()
        );
        return ExitCode::from(1);
    }
    let default_rotary_dim = {
        let candidate = if head_dim >= 4 {
            head_dim / 4
        } else {
            head_dim
        };
        candidate - (candidate % 2)
    };
    if default_rotary_dim == 0 {
        eprintln!("default rotary_dim is zero for head_dim={head_dim}");
        return ExitCode::from(1);
    }
    let rotary_dim = match parse_optional_usize(rotary_dim, default_rotary_dim, "rotary dim") {
        Ok(value) => value,
        Err(code) => return code,
    };
    if rotary_dim == 0 || rotary_dim > head_dim || !rotary_dim.is_multiple_of(2) {
        eprintln!(
            "rotary dim must be even and no greater than head_dim: rotary_dim={rotary_dim}, head_dim={head_dim}"
        );
        return ExitCode::from(2);
    }

    let result = package_self_attn_mlp_block_smoke_impl(
        &path,
        device_index,
        chunk_bytes,
        layer_index,
        sequence_len,
        rotary_dim,
        rope_base,
        position_offset,
        q_norm,
        k_norm,
        post_norm,
    );

    match result {
        Ok(line) => {
            println!("{line}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("{err}");
            ExitCode::from(1)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn package_self_attn_mlp_block_smoke_impl(
    path: &str,
    device_index: u32,
    chunk_bytes: usize,
    layer_index: usize,
    sequence_len: usize,
    rotary_dim: usize,
    rope_base: f32,
    position_offset: usize,
    q_norm: PassthroughF32Data,
    k_norm: PassthroughF32Data,
    post_norm: PassthroughF32Data,
) -> Result<String, String> {
    let q_tensor = format!("model.language_model.layers.{layer_index}.self_attn.q_proj.weight");
    let k_tensor = format!("model.language_model.layers.{layer_index}.self_attn.k_proj.weight");
    let v_tensor = format!("model.language_model.layers.{layer_index}.self_attn.v_proj.weight");
    let o_tensor = format!("model.language_model.layers.{layer_index}.self_attn.o_proj.weight");
    let q_norm_tensor =
        format!("model.language_model.layers.{layer_index}.self_attn.q_norm.weight");
    let k_norm_tensor =
        format!("model.language_model.layers.{layer_index}.self_attn.k_norm.weight");
    let post_norm_tensor =
        format!("model.language_model.layers.{layer_index}.post_attention_layernorm.weight");
    let gate_tensor = format!("model.language_model.layers.{layer_index}.mlp.gate_proj.weight");
    let up_tensor = format!("model.language_model.layers.{layer_index}.mlp.up_proj.weight");
    let down_tensor = format!("model.language_model.layers.{layer_index}.mlp.down_proj.weight");

    let mut context = ullm_runtime_sys::RuntimeContext::create(device_index)
        .map_err(|err| format!("failed to create runtime context: {err}"))?;
    let info = context
        .device_info()
        .map_err(|err| format!("failed to query runtime context device: {err}"))?;
    let mut stream = context
        .create_stream()
        .map_err(|err| format!("failed to create runtime stream: {err}"))?;

    let self_attn = run_self_attn_block_sequence_smoke(
        &mut context,
        &mut stream,
        path,
        chunk_bytes,
        sequence_len,
        rotary_dim,
        rope_base,
        position_offset,
        &q_tensor,
        &k_tensor,
        &v_tensor,
        &o_tensor,
        &q_norm,
        &k_norm,
        "package-self-attn-mlp-block-smoke",
    )?;

    let hidden = self_attn.hidden;
    let mlp_epsilon = 1e-5_f32;
    let layer_weights = qwen3_decoder_layer_runtime_weights_from_package(
        &mut context,
        &mut stream,
        path,
        chunk_bytes,
        hidden,
        &post_norm,
        &gate_tensor,
        &up_tensor,
        &down_tensor,
    )?;
    if layer_weights.hidden != hidden {
        return Err(format!(
            "Qwen3 decoder layer runtime weight hidden mismatch: expected={hidden} got={}",
            layer_weights.hidden
        ));
    }
    if layer_weights.mlp.gate_rows != layer_weights.intermediate
        || layer_weights.mlp.gate_cols != hidden
    {
        return Err("Qwen3 decoder layer runtime MLP gate shape is inconsistent".to_string());
    }

    let mut post_normed_expected = Vec::with_capacity(sequence_len * hidden);
    for timestep in 0..sequence_len {
        let start = timestep * hidden;
        let end = start + hidden;
        let expected = runtime_host_rmsnorm_f32(
            &self_attn.block_output[start..end],
            &post_norm.values,
            mlp_epsilon,
        );
        post_normed_expected.extend_from_slice(&expected);
    }

    let (post_normed, mlp_output, layer_output, post_norm_max_abs_diff, layer_block_max_abs_diff) = {
        let decode_shape = PagedDecodeShape {
            block_size: self_attn.paged_block_size,
            cache_blocks: self_attn.paged_cache_blocks,
            q_heads: self_attn.q_heads,
            kv_heads: self_attn.kv_heads,
            head_dim: self_attn.head_dim,
            value_dim: self_attn.value_dim,
        };
        let mut layer_step_state = Qwen3DecoderLayerStepState::new(
            &mut context,
            &mut stream,
            decode_shape,
            self_attn.paged_block_table.clone(),
            hidden,
            layer_weights.intermediate,
            self_attn.softmax_scale,
            mlp_epsilon,
        )?;
        let q_token_elements = self_attn.q_heads * self_attn.head_dim;
        let k_token_elements = self_attn.kv_heads * self_attn.head_dim;
        let v_token_elements = self_attn.kv_heads * self_attn.value_dim;
        let attention_elements = self_attn.q_heads * self_attn.value_dim;
        let mut layer_step_block_output = Vec::with_capacity(sequence_len * hidden);
        let mut post_normed = Vec::with_capacity(sequence_len * hidden);
        let mut mlp_output = Vec::with_capacity(sequence_len * hidden);
        let mut layer_output = Vec::with_capacity(sequence_len * hidden);
        for timestep in 0..sequence_len {
            let q_start = timestep * q_token_elements;
            let q_end = q_start + q_token_elements;
            let k_start = timestep * k_token_elements;
            let k_end = k_start + k_token_elements;
            let v_start = timestep * v_token_elements;
            let v_end = v_start + v_token_elements;
            let attention_start = timestep * attention_elements;
            let attention_end = attention_start + attention_elements;
            let residual_start = timestep * hidden;
            let residual_end = residual_start + hidden;
            let gate_step = self_attn
                .q_gate
                .as_ref()
                .map(|gate| &gate[attention_start..attention_end]);
            let step = layer_step_state
                .step(
                    &mut stream,
                    &self_attn.o_matrix,
                    &layer_weights.post_norm_weight,
                    &layer_weights.mlp.gate_matrix,
                    &layer_weights.mlp.up_matrix,
                    &layer_weights.mlp.down_matrix,
                    &self_attn.q_rope[q_start..q_end],
                    &self_attn.k_rope[k_start..k_end],
                    &self_attn.v_projected[v_start..v_end],
                    gate_step,
                    &self_attn.residual_sequence[residual_start..residual_end],
                )
                .map_err(|err| {
                    format!("failed to run Qwen3 decoder layer step {timestep}: {err}")
                })?;
            if step.cache_position != timestep {
                return Err(format!(
                    "Qwen3 decoder layer step wrote position {}, expected {timestep}",
                    step.cache_position
                ));
            }
            if step.cache_len != timestep + 1 {
                return Err(format!(
                    "Qwen3 decoder layer step reported cache_len {}, expected {}",
                    step.cache_len,
                    timestep + 1
                ));
            }
            layer_step_block_output.extend_from_slice(&step.block_output);
            post_normed.extend_from_slice(&step.post_normed);
            mlp_output.extend_from_slice(&step.mlp_output);
            layer_output.extend_from_slice(&step.layer_output);
        }

        let (expected_paged_k_cache, expected_paged_v_cache) =
            pack_paged_decode_kv_cache_for_block_table(
                &self_attn.k_rope,
                &self_attn.v_projected,
                &self_attn.paged_block_table,
                sequence_len,
                self_attn.paged_block_size,
                self_attn.paged_cache_blocks,
                self_attn.kv_heads,
                self_attn.head_dim,
                self_attn.value_dim,
            )?;
        let layer_cache = layer_step_state
            .read_cache_to_host(&mut stream)
            .map_err(|err| format!("failed to read Qwen3 decoder layer paged cache: {err}"))?;
        verify_f32_close(
            "package-self-attn-mlp-block-smoke layer step paged k cache write",
            &layer_cache.k,
            &expected_paged_k_cache,
            1e-5,
            1e-5,
        )?;
        verify_f32_close(
            "package-self-attn-mlp-block-smoke layer step paged v cache write",
            &layer_cache.v,
            &expected_paged_v_cache,
            1e-5,
            1e-5,
        )?;
        verify_f32_close(
            "package-self-attn-mlp-block-smoke layer step attention block",
            &layer_step_block_output,
            &self_attn.block_output,
            1e-4,
            1e-5,
        )?;
        let post_norm_max_abs_diff = verify_f32_close(
            "package-self-attn-mlp-block-smoke post RMSNorm",
            &post_normed,
            &post_normed_expected,
            1e-4,
            1e-5,
        )?;
        let expected_layer_output = runtime_host_add_f32(&layer_step_block_output, &mlp_output);
        let layer_block_max_abs_diff = verify_f32_close(
            "package-self-attn-mlp-block-smoke layer residual",
            &layer_output,
            &expected_layer_output,
            1e-5,
            1e-6,
        )?;
        (
            post_normed,
            mlp_output,
            layer_output,
            post_norm_max_abs_diff,
            layer_block_max_abs_diff,
        )
    };

    Ok(format!(
        "package-self-attn-mlp-block-smoke package={} layer={} q_tensor=\"{}\" k_tensor=\"{}\" v_tensor=\"{}\" o_tensor=\"{}\" q_norm_tensor=\"{}\" k_norm_tensor=\"{}\" post_norm_tensor=\"{}\" gate_tensor=\"{}\" up_tensor=\"{}\" down_tensor=\"{}\" hidden={} sequence_len={} paged_block_size={} paged_cache_blocks={} paged_block_table={:?} q_projection_layout={} q_gate_elements={} output_gate_layout={} q_heads={} kv_heads={} head_dim={} value_dim={} rotary_dim={} position_offset={} rope_base={} softmax_scale={:.9} q_norm_dtype={} k_norm_dtype={} post_norm_dtype={} backend={} device_index={} name=\"{}\" residual_preview={} attention_preview={} gated_attention_preview={} projected_preview={} attention_block_preview={} post_norm_preview={} mlp_output_preview={} layer_output_preview={} q_norm_max_abs_diff={:.9} k_norm_max_abs_diff={:.9} q_rope_max_abs_diff={:.9} k_rope_max_abs_diff={:.9} attention_max_abs_diff={:.9} paged_kv_write_k_max_abs_diff={:.9} paged_kv_write_v_max_abs_diff={:.9} paged_step_attention_max_abs_diff={:.9} causal_paged_step_attention_max_abs_diff={:.9} output_gate_max_abs_diff={:.9} o_proj_max_abs_diff={:.9} block_max_abs_diff={:.9} causal_paged_block_max_abs_diff={:.9} post_norm_max_abs_diff={:.9} layer_block_max_abs_diff={:.9} verified=true",
        path,
        layer_index,
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        q_norm_tensor,
        k_norm_tensor,
        post_norm_tensor,
        gate_tensor,
        up_tensor,
        down_tensor,
        hidden,
        sequence_len,
        self_attn.paged_block_size,
        self_attn.paged_cache_blocks,
        self_attn.paged_block_table,
        self_attn.q_projection_layout,
        self_attn.q_gate_elements,
        self_attn.output_gate_layout,
        self_attn.q_heads,
        self_attn.kv_heads,
        self_attn.head_dim,
        self_attn.value_dim,
        rotary_dim,
        position_offset,
        rope_base,
        self_attn.softmax_scale,
        q_norm.dtype,
        k_norm.dtype,
        post_norm.dtype,
        info.backend,
        device_index,
        info.name,
        format_f32_preview(
            &self_attn.residual_sequence[..8.min(self_attn.residual_sequence.len())]
        ),
        format_f32_preview(&self_attn.attention_output[..8.min(self_attn.attention_output.len())]),
        format_f32_preview(
            &self_attn.attention_projection_input
                [..8.min(self_attn.attention_projection_input.len())],
        ),
        format_f32_preview(&self_attn.attn_projected[..8.min(self_attn.attn_projected.len())]),
        format_f32_preview(&self_attn.block_output[..8.min(self_attn.block_output.len())]),
        format_f32_preview(&post_normed[..8.min(post_normed.len())]),
        format_f32_preview(&mlp_output[..8.min(mlp_output.len())]),
        format_f32_preview(&layer_output[..8.min(layer_output.len())]),
        self_attn.q_norm_max_abs_diff,
        self_attn.k_norm_max_abs_diff,
        self_attn.q_rope_max_abs_diff,
        self_attn.k_rope_max_abs_diff,
        self_attn.attention_max_abs_diff,
        self_attn.paged_kv_write_k_max_abs_diff,
        self_attn.paged_kv_write_v_max_abs_diff,
        self_attn.paged_step_attention_max_abs_diff,
        self_attn.causal_paged_step_attention_max_abs_diff,
        self_attn.output_gate_max_abs_diff,
        self_attn.o_proj_max_abs_diff,
        self_attn.block_max_abs_diff,
        self_attn.causal_paged_block_max_abs_diff,
        post_norm_max_abs_diff,
        layer_block_max_abs_diff,
    ))
}

fn package_linear_attn_aux_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    aux: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-linear-attn-aux-smoke requires a .ullm.d path");
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
    let aux = match parse_linear_attn_aux(aux.as_deref()) {
        Ok(value) => value,
        Err(code) => return code,
    };

    let requested_aux = match aux {
        LinearAttnAux::ALog => vec![(
            "a-log",
            format!("model.language_model.layers.{layer_index}.linear_attn.A_log"),
        )],
        LinearAttnAux::DtBias => vec![(
            "dt-bias",
            format!("model.language_model.layers.{layer_index}.linear_attn.dt_bias"),
        )],
        LinearAttnAux::Conv1d => vec![(
            "conv1d",
            format!("model.language_model.layers.{layer_index}.linear_attn.conv1d.weight"),
        )],
        LinearAttnAux::Norm => vec![(
            "norm",
            format!("model.language_model.layers.{layer_index}.linear_attn.norm.weight"),
        )],
        LinearAttnAux::All => vec![
            (
                "a-log",
                format!("model.language_model.layers.{layer_index}.linear_attn.A_log"),
            ),
            (
                "dt-bias",
                format!("model.language_model.layers.{layer_index}.linear_attn.dt_bias"),
            ),
            (
                "conv1d",
                format!("model.language_model.layers.{layer_index}.linear_attn.conv1d.weight"),
            ),
            (
                "norm",
                format!("model.language_model.layers.{layer_index}.linear_attn.norm.weight"),
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

    for (aux_name, tensor_name) in requested_aux {
        let selector = TensorSelector::Name(tensor_name.clone());
        let bundle = match ullm_engine::package::select_passthrough_payload_bundle(&path, &selector)
        {
            Ok(bundle) => bundle,
            Err(err) => {
                eprintln!("failed to select package passthrough tensor {tensor_name}: {err}");
                return ExitCode::from(1);
            }
        };

        if let Err(err) = validate_passthrough_shape_elements(&bundle) {
            eprintln!("invalid passthrough shape for {tensor_name}: {err}");
            return ExitCode::from(1);
        }

        let elements = match usize::try_from(bundle.elements) {
            Ok(value) if value > 0 => value,
            Ok(_) => {
                eprintln!("passthrough tensor {tensor_name} has zero elements");
                return ExitCode::from(1);
            }
            Err(_) => {
                eprintln!(
                    "passthrough tensor {tensor_name} element count is too large for this host"
                );
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
        let payload_bytes = if bundle.payload_bytes == 0 {
            bundle.payload_file.bytes
        } else {
            bundle.payload_bytes
        };
        if payload.len() != elements {
            eprintln!(
                "passthrough tensor element count mismatch for {tensor_name}: expected {elements} got {}",
                payload.len()
            );
            return ExitCode::from(1);
        }

        let payload_f32_bytes = encode_f32_to_bytes(&payload);
        let mut buffer = match context.alloc_buffer(payload_f32_bytes.len()) {
            Ok(buffer) => buffer,
            Err(err) => {
                eprintln!("failed to allocate runtime buffer for {tensor_name}: {err}");
                return ExitCode::from(1);
            }
        };
        if let Err(err) = buffer.copy_from_host(0, &payload_f32_bytes, Some(&mut stream)) {
            eprintln!("failed to copy payload for {tensor_name} into runtime buffer: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!("failed to synchronize runtime stream after payload copy: {err}");
            return ExitCode::from(1);
        }

        let mut output = vec![0_u8; payload_f32_bytes.len()];
        if let Err(err) = buffer.copy_to_host(0, &mut output, Some(&mut stream)) {
            eprintln!("failed to copy payload back for {tensor_name}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!("failed to synchronize runtime stream after payload readback: {err}");
            return ExitCode::from(1);
        }
        if payload_f32_bytes != output {
            eprintln!("runtime roundtrip mismatch for {tensor_name}");
            return ExitCode::from(1);
        }

        let preview = decode_f32_le_values(&output);
        let preview_count = preview.len().min(8);
        println!(
            "package-linear-attn-aux-smoke package={} layer={} aux={} tensor=\"{}\" dtype={} elements={} shape={} payload_bytes={} backend={} device_index={} name=\"{}\" preview={} verified=true",
            path,
            layer_index,
            aux_name,
            tensor_name,
            dtype,
            elements,
            format_u64_shape(&bundle.shape),
            payload_bytes,
            info.backend,
            device_index,
            info.name,
            format_f32_preview(&preview[..preview_count])
        );
    }
    ExitCode::SUCCESS
}

fn package_linear_attn_qkv_norm_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-linear-attn-qkv-norm-smoke requires a .ullm.d path");
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

    let qkv_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_qkv.weight");
    let norm_tensor = format!("model.language_model.layers.{layer_index}.linear_attn.norm.weight");

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
    if norm_elements != 128 {
        eprintln!("RMSNorm tensor must have 128 elements, got {norm_elements}");
        return ExitCode::from(1);
    }
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
            "passthrough tensor element count mismatch for {norm_tensor}: expected {norm_elements} got {}",
            norm_weight.len()
        );
        return ExitCode::from(1);
    }

    let mut registry = WeightRegistry::new();
    let (qkv_rows, qkv_cols, qkv_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &qkv_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize tensor {qkv_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    if qkv_rows < 128 {
        eprintln!(
            "qkv tensor {qkv_tensor} has too few rows for preview validation: rows={qkv_rows}, expected at least 128"
        );
        return ExitCode::from(1);
    }

    let input = deterministic_f32_vector(qkv_cols);
    let input_bytes = encode_f32_to_bytes(&input);
    let mut input_buffer = match context.alloc_buffer(input_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate qkv input buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = input_buffer.copy_from_host(0, &input_bytes, Some(&mut stream)) {
        eprintln!("failed to copy deterministic qkv input into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after input copy: {err}");
        return ExitCode::from(1);
    }

    let qkv_output_bytes = match qkv_rows.checked_mul(std::mem::size_of::<f32>()) {
        Some(value) => value,
        None => {
            eprintln!("qkv output byte size overflows");
            return ExitCode::from(1);
        }
    };
    let mut qkv_output = match context.alloc_buffer(qkv_output_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate qkv output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = ullm_runtime_sys::matvec_f32(
        &qkv_matrix,
        &input_buffer,
        qkv_rows,
        qkv_cols,
        &mut qkv_output,
        Some(&mut stream),
    ) {
        eprintln!("failed to run qkv matvec: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after qkv matvec: {err}");
        return ExitCode::from(1);
    }

    let qkv_preview_count = 128_usize;
    let mut qkv_preview_bytes = vec![0_u8; qkv_preview_count * std::mem::size_of::<f32>()];
    if let Err(err) = qkv_output.copy_to_host(0, &mut qkv_preview_bytes, Some(&mut stream)) {
        eprintln!("failed to copy qkv preview output back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after qkv preview copy: {err}");
        return ExitCode::from(1);
    }
    let qkv_output_preview = decode_f32_le_values(&qkv_preview_bytes);
    let norm_input = qkv_output_preview;

    let epsilon = 1e-5_f32;
    let expected = runtime_host_rmsnorm_f32(&norm_input, &norm_weight, epsilon);
    if expected.len() != norm_elements {
        eprintln!("failed to build deterministic RMSNorm reference");
        return ExitCode::from(1);
    }

    let norm_input_bytes = encode_f32_to_bytes(&norm_input);
    let norm_weight_bytes = encode_f32_to_bytes(&norm_weight);
    let mut norm_input_buffer = match context.alloc_buffer(norm_input_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate qkv-rmsnorm input buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = norm_input_buffer.copy_from_host(0, &norm_input_bytes, Some(&mut stream)) {
        eprintln!("failed to copy qkv output preview to qkv-rmsnorm input: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after qkv-rmsnorm input copy: {err}");
        return ExitCode::from(1);
    }
    let mut norm_weight_buffer = match context.alloc_buffer(norm_weight_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate norm weight buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = norm_weight_buffer.copy_from_host(0, &norm_weight_bytes, Some(&mut stream)) {
        eprintln!("failed to copy norm weight into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after norm weight copy: {err}");
        return ExitCode::from(1);
    }

    let mut norm_output_buffer = match context.alloc_buffer(norm_weight_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate qkv-rmsnorm output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = ullm_runtime_sys::rmsnorm_f32(
        &norm_input_buffer,
        &norm_weight_buffer,
        norm_elements,
        epsilon,
        &mut norm_output_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run runtime rmsnorm_f32: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after rmsnorm: {err}");
        return ExitCode::from(1);
    }

    let mut norm_output_bytes = vec![0_u8; norm_weight_bytes.len()];
    if let Err(err) = norm_output_buffer.copy_to_host(0, &mut norm_output_bytes, Some(&mut stream))
    {
        eprintln!("failed to copy qkv-rmsnorm output back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after qkv-rmsnorm output copy: {err}");
        return ExitCode::from(1);
    }
    let norm_output = decode_f32_le_values(&norm_output_bytes);

    if norm_output.len() != expected.len() {
        eprintln!(
            "runtime RMSNorm output size mismatch: expected {} got {}",
            expected.len(),
            norm_output.len()
        );
        return ExitCode::from(1);
    }

    let mut max_abs_diff = 0.0_f32;
    for (lhs, rhs) in norm_output.iter().zip(expected.iter()) {
        let diff = (lhs - rhs).abs();
        if diff > 1e-4_f32 {
            eprintln!(
                "package-linear-attn-qkv-norm-smoke mismatch for layer={layer_index}: max_abs_diff={diff}"
            );
            return ExitCode::from(1);
        }
        if diff > max_abs_diff {
            max_abs_diff = diff;
        }
    }

    let qkv_preview = &norm_input[..8.min(norm_input.len())];
    let norm_preview = &norm_output[..8.min(norm_output.len())];
    println!(
        "package-linear-attn-qkv-norm-smoke package={} layer={} qkv_tensor=\"{}\" norm_tensor=\"{}\" hidden={} qkv_rows={} norm_elements={} backend={} device_index={} name=\"{}\" qkv_preview={} norm_preview={} max_abs_diff={max_abs_diff:.9} verified=true",
        path,
        layer_index,
        qkv_tensor,
        norm_tensor,
        qkv_cols,
        qkv_rows,
        norm_elements,
        info.backend,
        device_index,
        info.name,
        format_f32_preview(qkv_preview),
        format_f32_preview(norm_preview),
    );
    ExitCode::SUCCESS
}

fn package_linear_attn_conv1d_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    sequence_len: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-linear-attn-conv1d-smoke requires a .ullm.d path");
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
    let sequence_len = match parse_optional_usize(sequence_len, 4, "sequence length") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("sequence length must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };

    let qkv_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_qkv.weight");
    let conv_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.conv1d.weight");

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

    let selector = TensorSelector::Name(conv_tensor.clone());
    let conv_bundle =
        match ullm_engine::package::select_passthrough_payload_bundle(&path, &selector) {
            Ok(bundle) => bundle,
            Err(err) => {
                eprintln!("failed to select package conv1d passthrough tensor: {err}");
                return ExitCode::from(1);
            }
        };
    if conv_bundle.shape.len() != 3 || conv_bundle.shape[1] != 1 {
        eprintln!(
            "conv1d tensor shape must be [channels,1,kernel], got {}",
            format_u64_shape(&conv_bundle.shape)
        );
        return ExitCode::from(1);
    }
    let conv_channels = match usize::try_from(conv_bundle.shape[0]) {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("conv1d tensor has zero channels");
            return ExitCode::from(1);
        }
        Err(_) => {
            eprintln!("conv1d channel count is too large for this host");
            return ExitCode::from(1);
        }
    };
    let kernel_size = match usize::try_from(conv_bundle.shape[2]) {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("conv1d tensor has zero kernel size");
            return ExitCode::from(1);
        }
        Err(_) => {
            eprintln!("conv1d kernel size is too large for this host");
            return ExitCode::from(1);
        }
    };
    let dtype = match resolve_passthrough_dtype(&conv_bundle, &conv_tensor) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let conv_weight = match read_passthrough_payload_f32_bytes(&conv_bundle, chunk_bytes, dtype) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to read passthrough payload for {conv_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    let expected_conv_elements = match conv_channels.checked_mul(kernel_size) {
        Some(value) => value,
        None => {
            eprintln!("conv1d weight element count overflows");
            return ExitCode::from(1);
        }
    };
    if conv_weight.len() != expected_conv_elements {
        eprintln!(
            "conv1d weight element count mismatch: expected {expected_conv_elements} got {}",
            conv_weight.len()
        );
        return ExitCode::from(1);
    }

    let mut registry = WeightRegistry::new();
    let (qkv_rows, qkv_cols, qkv_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &qkv_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize tensor {qkv_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    if qkv_rows != conv_channels {
        eprintln!(
            "qkv rows must match conv1d channels: qkv_rows={qkv_rows}, conv_channels={conv_channels}"
        );
        return ExitCode::from(1);
    }

    let qkv_step_bytes = match qkv_rows.checked_mul(std::mem::size_of::<f32>()) {
        Some(value) => value,
        None => {
            eprintln!("qkv step byte size overflows");
            return ExitCode::from(1);
        }
    };
    let qkv_sequence_bytes_len = match qkv_step_bytes.checked_mul(sequence_len) {
        Some(value) => value,
        None => {
            eprintln!("qkv sequence byte size overflows");
            return ExitCode::from(1);
        }
    };
    let input_bytes_len = match qkv_cols.checked_mul(std::mem::size_of::<f32>()) {
        Some(value) => value,
        None => {
            eprintln!("qkv input byte size overflows");
            return ExitCode::from(1);
        }
    };
    let mut input_buffer = match context.alloc_buffer(input_bytes_len) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate qkv input buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut qkv_step_buffer = match context.alloc_buffer(qkv_step_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate qkv step output buffer: {err}");
            return ExitCode::from(1);
        }
    };

    let base_input = deterministic_f32_vector(qkv_cols);
    let mut qkv_sequence_bytes = vec![0_u8; qkv_sequence_bytes_len];
    for timestep in 0..sequence_len {
        let step_input = base_input
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let phase = (index % 17) as f32 - 8.0_f32;
                *value + (timestep as f32) * phase * 0.00025_f32
            })
            .collect::<Vec<_>>();
        let step_input_bytes = encode_f32_to_bytes(&step_input);
        if let Err(err) = input_buffer.copy_from_host(0, &step_input_bytes, Some(&mut stream)) {
            eprintln!("failed to copy qkv input timestep {timestep} into runtime buffer: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = ullm_runtime_sys::matvec_f32(
            &qkv_matrix,
            &input_buffer,
            qkv_rows,
            qkv_cols,
            &mut qkv_step_buffer,
            Some(&mut stream),
        ) {
            eprintln!("failed to run qkv matvec for timestep {timestep}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!("failed to synchronize runtime stream after qkv timestep {timestep}: {err}");
            return ExitCode::from(1);
        }

        let start = timestep * qkv_step_bytes;
        let end = start + qkv_step_bytes;
        if let Err(err) =
            qkv_step_buffer.copy_to_host(0, &mut qkv_sequence_bytes[start..end], Some(&mut stream))
        {
            eprintln!("failed to copy qkv timestep {timestep} back to host: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!(
                "failed to synchronize runtime stream after qkv timestep {timestep} host copy: {err}"
            );
            return ExitCode::from(1);
        }
    }
    let qkv_sequence = decode_f32_le_values(&qkv_sequence_bytes);
    let expected = runtime_host_depthwise_conv1d_f32(
        &qkv_sequence,
        &conv_weight,
        qkv_rows,
        sequence_len,
        kernel_size,
    );
    if expected.len() != qkv_sequence.len() {
        eprintln!("failed to build package depthwise conv1d reference");
        return ExitCode::from(1);
    }

    let mut conv_input_buffer = match context.alloc_buffer(qkv_sequence_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate conv1d input buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let conv_weight_bytes = encode_f32_to_bytes(&conv_weight);
    let mut conv_weight_buffer = match context.alloc_buffer(conv_weight_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate conv1d weight buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut conv_output_buffer = match context.alloc_buffer(qkv_sequence_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate conv1d output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = conv_input_buffer.copy_from_host(0, &qkv_sequence_bytes, Some(&mut stream)) {
        eprintln!("failed to copy conv1d input sequence into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = conv_weight_buffer.copy_from_host(0, &conv_weight_bytes, Some(&mut stream)) {
        eprintln!("failed to copy conv1d weight into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after conv1d input copy: {err}");
        return ExitCode::from(1);
    }

    if let Err(err) = ullm_runtime_sys::depthwise_conv1d_f32(
        &conv_input_buffer,
        &conv_weight_buffer,
        qkv_rows,
        sequence_len,
        kernel_size,
        &mut conv_output_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run runtime depthwise_conv1d_f32: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after conv1d: {err}");
        return ExitCode::from(1);
    }

    let mut conv_output_bytes = vec![0_u8; qkv_sequence_bytes.len()];
    if let Err(err) = conv_output_buffer.copy_to_host(0, &mut conv_output_bytes, Some(&mut stream))
    {
        eprintln!("failed to copy conv1d output back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after conv1d output copy: {err}");
        return ExitCode::from(1);
    }
    let conv_output = decode_f32_le_values(&conv_output_bytes);

    if conv_output.len() != expected.len() {
        eprintln!(
            "runtime depthwise conv1d output size mismatch: expected {} got {}",
            expected.len(),
            conv_output.len()
        );
        return ExitCode::from(1);
    }

    let mut max_abs_diff = 0.0_f32;
    for (lhs, rhs) in conv_output.iter().zip(expected.iter()) {
        let diff = (lhs - rhs).abs();
        if diff > 1e-4_f32 {
            eprintln!(
                "package-linear-attn-conv1d-smoke mismatch for layer={layer_index}: max_abs_diff={diff}"
            );
            return ExitCode::from(1);
        }
        if diff > max_abs_diff {
            max_abs_diff = diff;
        }
    }

    let qkv_preview = &qkv_sequence[..8.min(qkv_sequence.len())];
    let conv_preview = &conv_output[..8.min(conv_output.len())];
    println!(
        "package-linear-attn-conv1d-smoke package={} layer={} qkv_tensor=\"{}\" conv_tensor=\"{}\" hidden={} channels={} sequence_len={} kernel_size={} dtype={} backend={} device_index={} name=\"{}\" qkv_preview={} conv_preview={} max_abs_diff={max_abs_diff:.9} verified=true",
        path,
        layer_index,
        qkv_tensor,
        conv_tensor,
        qkv_cols,
        qkv_rows,
        sequence_len,
        kernel_size,
        dtype,
        info.backend,
        device_index,
        info.name,
        format_f32_preview(qkv_preview),
        format_f32_preview(conv_preview),
    );
    ExitCode::SUCCESS
}

fn package_linear_attn_gate_beta_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    sequence_len: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-linear-attn-gate-beta-smoke requires a .ullm.d path");
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
    let sequence_len = match parse_optional_usize(sequence_len, 4, "sequence length") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("sequence length must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };

    let a_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_a.weight");
    let b_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_b.weight");
    let a_log_tensor = format!("model.language_model.layers.{layer_index}.linear_attn.A_log");
    let dt_bias_tensor = format!("model.language_model.layers.{layer_index}.linear_attn.dt_bias");

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

    let a_log_selector = TensorSelector::Name(a_log_tensor.clone());
    let a_log_bundle =
        match ullm_engine::package::select_passthrough_payload_bundle(&path, &a_log_selector) {
            Ok(bundle) => bundle,
            Err(err) => {
                eprintln!("failed to select package A_log passthrough tensor: {err}");
                return ExitCode::from(1);
            }
        };
    if let Err(err) = validate_passthrough_shape_elements(&a_log_bundle) {
        eprintln!("invalid A_log shape for {a_log_tensor}: {err}");
        return ExitCode::from(1);
    }
    let a_log_dtype = match resolve_passthrough_dtype(&a_log_bundle, &a_log_tensor) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let a_log = match read_passthrough_payload_f32_bytes(&a_log_bundle, chunk_bytes, a_log_dtype) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to read passthrough payload for {a_log_tensor}: {err}");
            return ExitCode::from(1);
        }
    };

    let dt_bias_selector = TensorSelector::Name(dt_bias_tensor.clone());
    let dt_bias_bundle =
        match ullm_engine::package::select_passthrough_payload_bundle(&path, &dt_bias_selector) {
            Ok(bundle) => bundle,
            Err(err) => {
                eprintln!("failed to select package dt_bias passthrough tensor: {err}");
                return ExitCode::from(1);
            }
        };
    if let Err(err) = validate_passthrough_shape_elements(&dt_bias_bundle) {
        eprintln!("invalid dt_bias shape for {dt_bias_tensor}: {err}");
        return ExitCode::from(1);
    }
    let dt_bias_dtype = match resolve_passthrough_dtype(&dt_bias_bundle, &dt_bias_tensor) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let dt_bias =
        match read_passthrough_payload_f32_bytes(&dt_bias_bundle, chunk_bytes, dt_bias_dtype) {
            Ok(value) => value,
            Err(err) => {
                eprintln!("failed to read passthrough payload for {dt_bias_tensor}: {err}");
                return ExitCode::from(1);
            }
        };

    let mut registry = WeightRegistry::new();
    let (a_rows, a_cols, a_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &a_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize tensor {a_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    let (b_rows, b_cols, b_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &b_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize tensor {b_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    if a_rows == 0 || b_rows == 0 || a_cols == 0 || b_cols == 0 {
        eprintln!("linear attention a/b projection matrix has zero dimension");
        return ExitCode::from(1);
    }
    if a_rows != b_rows || a_cols != b_cols {
        eprintln!(
            "linear attention a/b projection shapes differ: a=[{a_rows},{a_cols}] b=[{b_rows},{b_cols}]"
        );
        return ExitCode::from(1);
    }
    let heads = a_rows;
    let hidden = a_cols;
    if a_log.len() != heads {
        eprintln!(
            "A_log tensor {a_log_tensor} length must match heads: len={} heads={heads}",
            a_log.len()
        );
        return ExitCode::from(1);
    }
    if dt_bias.len() != heads {
        eprintln!(
            "dt_bias tensor {dt_bias_tensor} length must match heads: len={} heads={heads}",
            dt_bias.len()
        );
        return ExitCode::from(1);
    }

    let step_bytes = match heads.checked_mul(std::mem::size_of::<f32>()) {
        Some(value) => value,
        None => {
            eprintln!("linear attention gate beta step byte size overflows");
            return ExitCode::from(1);
        }
    };
    let sequence_bytes_len = match step_bytes.checked_mul(sequence_len) {
        Some(value) => value,
        None => {
            eprintln!("linear attention gate beta sequence byte size overflows");
            return ExitCode::from(1);
        }
    };
    let input_bytes_len = match hidden.checked_mul(std::mem::size_of::<f32>()) {
        Some(value) => value,
        None => {
            eprintln!("linear attention gate beta input byte size overflows");
            return ExitCode::from(1);
        }
    };
    let mut input_buffer = match context.alloc_buffer(input_bytes_len) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate gate beta input buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut a_step_buffer = match context.alloc_buffer(step_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate a step output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut b_step_buffer = match context.alloc_buffer(step_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate b step output buffer: {err}");
            return ExitCode::from(1);
        }
    };

    let base_input = deterministic_f32_vector(hidden);
    let mut a_sequence_bytes = vec![0_u8; sequence_bytes_len];
    let mut b_sequence_bytes = vec![0_u8; sequence_bytes_len];
    for timestep in 0..sequence_len {
        let step_input = base_input
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let phase = (index % 17) as f32 - 8.0_f32;
                *value + (timestep as f32) * phase * 0.00025_f32
            })
            .collect::<Vec<_>>();
        let step_input_bytes = encode_f32_to_bytes(&step_input);
        if let Err(err) = input_buffer.copy_from_host(0, &step_input_bytes, Some(&mut stream)) {
            eprintln!("failed to copy gate beta input timestep {timestep}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = ullm_runtime_sys::matvec_f32(
            &a_matrix,
            &input_buffer,
            heads,
            hidden,
            &mut a_step_buffer,
            Some(&mut stream),
        ) {
            eprintln!("failed to run a matvec for timestep {timestep}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = ullm_runtime_sys::matvec_f32(
            &b_matrix,
            &input_buffer,
            heads,
            hidden,
            &mut b_step_buffer,
            Some(&mut stream),
        ) {
            eprintln!("failed to run b matvec for timestep {timestep}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!(
                "failed to synchronize runtime stream after gate beta timestep {timestep}: {err}"
            );
            return ExitCode::from(1);
        }

        let start = timestep * step_bytes;
        let end = start + step_bytes;
        if let Err(err) =
            a_step_buffer.copy_to_host(0, &mut a_sequence_bytes[start..end], Some(&mut stream))
        {
            eprintln!("failed to copy a timestep {timestep} back to host: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) =
            b_step_buffer.copy_to_host(0, &mut b_sequence_bytes[start..end], Some(&mut stream))
        {
            eprintln!("failed to copy b timestep {timestep} back to host: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!(
                "failed to synchronize runtime stream after gate beta timestep {timestep} host copy: {err}"
            );
            return ExitCode::from(1);
        }
    }

    let a_sequence = decode_f32_le_values(&a_sequence_bytes);
    let b_sequence = decode_f32_le_values(&b_sequence_bytes);
    let (expected_gate, expected_beta) = runtime_host_linear_attn_gate_beta_f32(
        &a_sequence,
        &b_sequence,
        &a_log,
        &dt_bias,
        heads,
        sequence_len,
    );
    if expected_gate.is_empty() || expected_beta.is_empty() {
        eprintln!("failed to build package linear attention gate beta reference");
        return ExitCode::from(1);
    }

    let a_log_bytes = encode_f32_to_bytes(&a_log);
    let dt_bias_bytes = encode_f32_to_bytes(&dt_bias);
    let mut a_sequence_buffer = match context.alloc_buffer(a_sequence_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate a sequence buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut b_sequence_buffer = match context.alloc_buffer(b_sequence_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate b sequence buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut a_log_buffer = match context.alloc_buffer(a_log_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate A_log buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut dt_bias_buffer = match context.alloc_buffer(dt_bias_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate dt_bias buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut gate_output_buffer = match context.alloc_buffer(sequence_bytes_len) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate gate output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut beta_output_buffer = match context.alloc_buffer(sequence_bytes_len) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate beta output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = a_sequence_buffer.copy_from_host(0, &a_sequence_bytes, Some(&mut stream)) {
        eprintln!("failed to copy a sequence into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = b_sequence_buffer.copy_from_host(0, &b_sequence_bytes, Some(&mut stream)) {
        eprintln!("failed to copy b sequence into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = a_log_buffer.copy_from_host(0, &a_log_bytes, Some(&mut stream)) {
        eprintln!("failed to copy A_log into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = dt_bias_buffer.copy_from_host(0, &dt_bias_bytes, Some(&mut stream)) {
        eprintln!("failed to copy dt_bias into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after gate beta sequence copy: {err}");
        return ExitCode::from(1);
    }

    if let Err(err) = ullm_runtime_sys::linear_attn_gate_beta_f32(
        &a_sequence_buffer,
        &b_sequence_buffer,
        &a_log_buffer,
        &dt_bias_buffer,
        heads,
        sequence_len,
        &mut gate_output_buffer,
        &mut beta_output_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run runtime linear_attn_gate_beta_f32: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after linear attention gate beta: {err}");
        return ExitCode::from(1);
    }

    let mut gate_output_bytes = vec![0_u8; sequence_bytes_len];
    let mut beta_output_bytes = vec![0_u8; sequence_bytes_len];
    if let Err(err) = gate_output_buffer.copy_to_host(0, &mut gate_output_bytes, Some(&mut stream))
    {
        eprintln!("failed to copy gate output back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = beta_output_buffer.copy_to_host(0, &mut beta_output_bytes, Some(&mut stream))
    {
        eprintln!("failed to copy beta output back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after gate beta output copy: {err}");
        return ExitCode::from(1);
    }
    let gate_output = decode_f32_le_values(&gate_output_bytes);
    let beta_output = decode_f32_le_values(&beta_output_bytes);

    let mut max_abs_diff = 0.0_f32;
    for ((gate, expected_gate), (beta, expected_beta)) in gate_output
        .iter()
        .zip(expected_gate.iter())
        .zip(beta_output.iter().zip(expected_beta.iter()))
    {
        let gate_diff = (gate - expected_gate).abs();
        let beta_diff = (beta - expected_beta).abs();
        let diff = gate_diff.max(beta_diff);
        if diff > 1e-4_f32 {
            eprintln!(
                "package-linear-attn-gate-beta-smoke mismatch for layer={layer_index}: max_abs_diff={diff}"
            );
            return ExitCode::from(1);
        }
        if diff > max_abs_diff {
            max_abs_diff = diff;
        }
    }

    println!(
        "package-linear-attn-gate-beta-smoke package={} layer={} a_tensor=\"{}\" b_tensor=\"{}\" a_log_tensor=\"{}\" dt_bias_tensor=\"{}\" hidden={} heads={} sequence_len={} a_log_dtype={} dt_bias_dtype={} backend={} device_index={} name=\"{}\" a_preview={} b_preview={} gate_preview={} beta_preview={} max_abs_diff={max_abs_diff:.9} verified=true",
        path,
        layer_index,
        a_tensor,
        b_tensor,
        a_log_tensor,
        dt_bias_tensor,
        hidden,
        heads,
        sequence_len,
        a_log_dtype,
        dt_bias_dtype,
        info.backend,
        device_index,
        info.name,
        format_f32_preview(&a_sequence[..8.min(a_sequence.len())]),
        format_f32_preview(&b_sequence[..8.min(b_sequence.len())]),
        format_f32_preview(&gate_output[..8.min(gate_output.len())]),
        format_f32_preview(&beta_output[..8.min(beta_output.len())]),
    );
    ExitCode::SUCCESS
}

fn package_linear_attn_recurrent_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    sequence_len: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-linear-attn-recurrent-smoke requires a .ullm.d path");
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
    let sequence_len = match parse_optional_usize(sequence_len, 4, "sequence length") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("sequence length must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };

    let key_heads = 16_usize;
    let value_heads = 32_usize;
    let key_dim = 128_usize;
    let value_dim = 128_usize;
    let q_elements_per_step = key_heads * key_dim;
    let k_elements_per_step = key_heads * key_dim;
    let v_elements_per_step = value_heads * value_dim;
    let recurrent_channels = q_elements_per_step + k_elements_per_step + v_elements_per_step;
    let q_scale = 1.0_f32 / (key_dim as f32).sqrt();
    let qk_l2_norm = true;

    let qkv_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_qkv.weight");
    let conv_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.conv1d.weight");
    let a_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_a.weight");
    let b_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_b.weight");
    let a_log_tensor = format!("model.language_model.layers.{layer_index}.linear_attn.A_log");
    let dt_bias_tensor = format!("model.language_model.layers.{layer_index}.linear_attn.dt_bias");

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

    let conv_selector = TensorSelector::Name(conv_tensor.clone());
    let conv_bundle =
        match ullm_engine::package::select_passthrough_payload_bundle(&path, &conv_selector) {
            Ok(bundle) => bundle,
            Err(err) => {
                eprintln!("failed to select package conv1d passthrough tensor: {err}");
                return ExitCode::from(1);
            }
        };
    if conv_bundle.shape.len() != 3 || conv_bundle.shape[1] != 1 {
        eprintln!(
            "conv1d tensor shape must be [channels,1,kernel], got {}",
            format_u64_shape(&conv_bundle.shape)
        );
        return ExitCode::from(1);
    }
    let conv_channels = match usize::try_from(conv_bundle.shape[0]) {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("conv1d tensor has zero channels");
            return ExitCode::from(1);
        }
        Err(_) => {
            eprintln!("conv1d channel count is too large for this host");
            return ExitCode::from(1);
        }
    };
    if conv_channels != recurrent_channels {
        eprintln!(
            "conv1d channels must match Qwen3.5 linear attention q/k/v layout: conv_channels={conv_channels}, expected={recurrent_channels}"
        );
        return ExitCode::from(1);
    }
    let kernel_size = match usize::try_from(conv_bundle.shape[2]) {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("conv1d tensor has zero kernel size");
            return ExitCode::from(1);
        }
        Err(_) => {
            eprintln!("conv1d kernel size is too large for this host");
            return ExitCode::from(1);
        }
    };
    let conv_dtype = match resolve_passthrough_dtype(&conv_bundle, &conv_tensor) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let conv_weight =
        match read_passthrough_payload_f32_bytes(&conv_bundle, chunk_bytes, conv_dtype) {
            Ok(value) => value,
            Err(err) => {
                eprintln!("failed to read passthrough payload for {conv_tensor}: {err}");
                return ExitCode::from(1);
            }
        };
    if conv_weight.len() != conv_channels * kernel_size {
        eprintln!(
            "conv1d weight element count mismatch: expected {} got {}",
            conv_channels * kernel_size,
            conv_weight.len()
        );
        return ExitCode::from(1);
    }

    let a_log_selector = TensorSelector::Name(a_log_tensor.clone());
    let a_log_bundle =
        match ullm_engine::package::select_passthrough_payload_bundle(&path, &a_log_selector) {
            Ok(bundle) => bundle,
            Err(err) => {
                eprintln!("failed to select package A_log passthrough tensor: {err}");
                return ExitCode::from(1);
            }
        };
    if let Err(err) = validate_passthrough_shape_elements(&a_log_bundle) {
        eprintln!("invalid A_log shape for {a_log_tensor}: {err}");
        return ExitCode::from(1);
    }
    let a_log_dtype = match resolve_passthrough_dtype(&a_log_bundle, &a_log_tensor) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let a_log = match read_passthrough_payload_f32_bytes(&a_log_bundle, chunk_bytes, a_log_dtype) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to read passthrough payload for {a_log_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    if a_log.len() != value_heads {
        eprintln!(
            "A_log tensor {a_log_tensor} length must match value heads: len={} value_heads={value_heads}",
            a_log.len()
        );
        return ExitCode::from(1);
    }

    let dt_bias_selector = TensorSelector::Name(dt_bias_tensor.clone());
    let dt_bias_bundle =
        match ullm_engine::package::select_passthrough_payload_bundle(&path, &dt_bias_selector) {
            Ok(bundle) => bundle,
            Err(err) => {
                eprintln!("failed to select package dt_bias passthrough tensor: {err}");
                return ExitCode::from(1);
            }
        };
    if let Err(err) = validate_passthrough_shape_elements(&dt_bias_bundle) {
        eprintln!("invalid dt_bias shape for {dt_bias_tensor}: {err}");
        return ExitCode::from(1);
    }
    let dt_bias_dtype = match resolve_passthrough_dtype(&dt_bias_bundle, &dt_bias_tensor) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let dt_bias =
        match read_passthrough_payload_f32_bytes(&dt_bias_bundle, chunk_bytes, dt_bias_dtype) {
            Ok(value) => value,
            Err(err) => {
                eprintln!("failed to read passthrough payload for {dt_bias_tensor}: {err}");
                return ExitCode::from(1);
            }
        };
    if dt_bias.len() != value_heads {
        eprintln!(
            "dt_bias tensor {dt_bias_tensor} length must match value heads: len={} value_heads={value_heads}",
            dt_bias.len()
        );
        return ExitCode::from(1);
    }

    let mut registry = WeightRegistry::new();
    let (qkv_rows, qkv_cols, qkv_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &qkv_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize tensor {qkv_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    if qkv_rows != conv_channels {
        eprintln!(
            "qkv rows must match conv1d channels: qkv_rows={qkv_rows}, conv_channels={conv_channels}"
        );
        return ExitCode::from(1);
    }

    let (a_rows, a_cols, a_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &a_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize tensor {a_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    let (b_rows, b_cols, b_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &b_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize tensor {b_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    if a_rows != value_heads || b_rows != value_heads {
        eprintln!(
            "linear attention a/b rows must match value_heads={value_heads}: a_rows={a_rows}, b_rows={b_rows}"
        );
        return ExitCode::from(1);
    }
    if a_cols != qkv_cols || b_cols != qkv_cols {
        eprintln!(
            "linear attention a/b hidden sizes must match qkv hidden={qkv_cols}: a_cols={a_cols}, b_cols={b_cols}"
        );
        return ExitCode::from(1);
    }

    let qkv_step_bytes = qkv_rows * std::mem::size_of::<f32>();
    let gate_beta_step_bytes = value_heads * std::mem::size_of::<f32>();
    let qkv_sequence_bytes_len = match qkv_step_bytes.checked_mul(sequence_len) {
        Some(value) => value,
        None => {
            eprintln!("qkv sequence byte size overflows");
            return ExitCode::from(1);
        }
    };
    let gate_beta_sequence_bytes_len = match gate_beta_step_bytes.checked_mul(sequence_len) {
        Some(value) => value,
        None => {
            eprintln!("linear attention gate beta sequence byte size overflows");
            return ExitCode::from(1);
        }
    };
    let input_bytes_len = match qkv_cols.checked_mul(std::mem::size_of::<f32>()) {
        Some(value) => value,
        None => {
            eprintln!("linear attention input byte size overflows");
            return ExitCode::from(1);
        }
    };

    let mut input_buffer = match context.alloc_buffer(input_bytes_len) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate linear attention input buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut qkv_step_buffer = match context.alloc_buffer(qkv_step_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate qkv step output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut a_step_buffer = match context.alloc_buffer(gate_beta_step_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate a step output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut b_step_buffer = match context.alloc_buffer(gate_beta_step_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate b step output buffer: {err}");
            return ExitCode::from(1);
        }
    };

    let base_input = deterministic_f32_vector(qkv_cols);
    let mut qkv_sequence_bytes = vec![0_u8; qkv_sequence_bytes_len];
    let mut a_sequence_bytes = vec![0_u8; gate_beta_sequence_bytes_len];
    let mut b_sequence_bytes = vec![0_u8; gate_beta_sequence_bytes_len];
    for timestep in 0..sequence_len {
        let step_input = linear_attn_step_input(&base_input, timestep);
        let step_input_bytes = encode_f32_to_bytes(&step_input);
        if let Err(err) = input_buffer.copy_from_host(0, &step_input_bytes, Some(&mut stream)) {
            eprintln!("failed to copy linear attention input timestep {timestep}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = ullm_runtime_sys::matvec_f32(
            &qkv_matrix,
            &input_buffer,
            qkv_rows,
            qkv_cols,
            &mut qkv_step_buffer,
            Some(&mut stream),
        ) {
            eprintln!("failed to run qkv matvec for timestep {timestep}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = ullm_runtime_sys::matvec_f32(
            &a_matrix,
            &input_buffer,
            value_heads,
            qkv_cols,
            &mut a_step_buffer,
            Some(&mut stream),
        ) {
            eprintln!("failed to run a matvec for timestep {timestep}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = ullm_runtime_sys::matvec_f32(
            &b_matrix,
            &input_buffer,
            value_heads,
            qkv_cols,
            &mut b_step_buffer,
            Some(&mut stream),
        ) {
            eprintln!("failed to run b matvec for timestep {timestep}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!(
                "failed to synchronize runtime stream after linear attention timestep {timestep}: {err}"
            );
            return ExitCode::from(1);
        }

        let qkv_start = timestep * qkv_step_bytes;
        let qkv_end = qkv_start + qkv_step_bytes;
        if let Err(err) = qkv_step_buffer.copy_to_host(
            0,
            &mut qkv_sequence_bytes[qkv_start..qkv_end],
            Some(&mut stream),
        ) {
            eprintln!("failed to copy qkv timestep {timestep} back to host: {err}");
            return ExitCode::from(1);
        }
        let gate_beta_start = timestep * gate_beta_step_bytes;
        let gate_beta_end = gate_beta_start + gate_beta_step_bytes;
        if let Err(err) = a_step_buffer.copy_to_host(
            0,
            &mut a_sequence_bytes[gate_beta_start..gate_beta_end],
            Some(&mut stream),
        ) {
            eprintln!("failed to copy a timestep {timestep} back to host: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = b_step_buffer.copy_to_host(
            0,
            &mut b_sequence_bytes[gate_beta_start..gate_beta_end],
            Some(&mut stream),
        ) {
            eprintln!("failed to copy b timestep {timestep} back to host: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!(
                "failed to synchronize runtime stream after timestep {timestep} host copy: {err}"
            );
            return ExitCode::from(1);
        }
    }

    let qkv_sequence = decode_f32_le_values(&qkv_sequence_bytes);
    let expected_conv = runtime_host_depthwise_conv1d_f32(
        &qkv_sequence,
        &conv_weight,
        qkv_rows,
        sequence_len,
        kernel_size,
    );
    if expected_conv.is_empty() {
        eprintln!("failed to build package depthwise conv1d reference");
        return ExitCode::from(1);
    }

    let mut conv_input_buffer = match context.alloc_buffer(qkv_sequence_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate conv1d input buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let conv_weight_bytes = encode_f32_to_bytes(&conv_weight);
    let mut conv_weight_buffer = match context.alloc_buffer(conv_weight_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate conv1d weight buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut conv_output_buffer = match context.alloc_buffer(qkv_sequence_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate conv1d output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = conv_input_buffer.copy_from_host(0, &qkv_sequence_bytes, Some(&mut stream)) {
        eprintln!("failed to copy conv1d input sequence into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = conv_weight_buffer.copy_from_host(0, &conv_weight_bytes, Some(&mut stream)) {
        eprintln!("failed to copy conv1d weight into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after conv1d input copy: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = ullm_runtime_sys::depthwise_conv1d_f32(
        &conv_input_buffer,
        &conv_weight_buffer,
        qkv_rows,
        sequence_len,
        kernel_size,
        &mut conv_output_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run runtime depthwise_conv1d_f32: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after conv1d: {err}");
        return ExitCode::from(1);
    }

    let mut conv_output_bytes = vec![0_u8; qkv_sequence_bytes.len()];
    if let Err(err) = conv_output_buffer.copy_to_host(0, &mut conv_output_bytes, Some(&mut stream))
    {
        eprintln!("failed to copy conv1d output back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after conv1d output copy: {err}");
        return ExitCode::from(1);
    }
    let conv_output = decode_f32_le_values(&conv_output_bytes);
    let mut conv_max_abs_diff = 0.0_f32;
    for (lhs, rhs) in conv_output.iter().zip(expected_conv.iter()) {
        let diff = (lhs - rhs).abs();
        if diff > 1e-4_f32 {
            eprintln!(
                "package-linear-attn-recurrent-smoke conv1d mismatch for layer={layer_index}: max_abs_diff={diff}"
            );
            return ExitCode::from(1);
        }
        if diff > conv_max_abs_diff {
            conv_max_abs_diff = diff;
        }
    }

    let qkv_split = match split_linear_attn_qkv_for_recurrent(
        &conv_output,
        sequence_len,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        qk_l2_norm,
        q_scale,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to split linear attention qkv: {err}");
            return ExitCode::from(1);
        }
    };

    let a_sequence = decode_f32_le_values(&a_sequence_bytes);
    let b_sequence = decode_f32_le_values(&b_sequence_bytes);
    let (expected_gate, expected_beta) = runtime_host_linear_attn_gate_beta_f32(
        &a_sequence,
        &b_sequence,
        &a_log,
        &dt_bias,
        value_heads,
        sequence_len,
    );
    if expected_gate.is_empty() || expected_beta.is_empty() {
        eprintln!("failed to build package linear attention gate beta reference");
        return ExitCode::from(1);
    }

    let a_log_bytes = encode_f32_to_bytes(&a_log);
    let dt_bias_bytes = encode_f32_to_bytes(&dt_bias);
    let mut a_sequence_buffer = match context.alloc_buffer(a_sequence_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate a sequence buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut b_sequence_buffer = match context.alloc_buffer(b_sequence_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate b sequence buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut a_log_buffer = match context.alloc_buffer(a_log_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate A_log buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut dt_bias_buffer = match context.alloc_buffer(dt_bias_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate dt_bias buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut gate_output_buffer = match context.alloc_buffer(gate_beta_sequence_bytes_len) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate gate output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut beta_output_buffer = match context.alloc_buffer(gate_beta_sequence_bytes_len) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate beta output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = a_sequence_buffer.copy_from_host(0, &a_sequence_bytes, Some(&mut stream)) {
        eprintln!("failed to copy a sequence into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = b_sequence_buffer.copy_from_host(0, &b_sequence_bytes, Some(&mut stream)) {
        eprintln!("failed to copy b sequence into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = a_log_buffer.copy_from_host(0, &a_log_bytes, Some(&mut stream)) {
        eprintln!("failed to copy A_log into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = dt_bias_buffer.copy_from_host(0, &dt_bias_bytes, Some(&mut stream)) {
        eprintln!("failed to copy dt_bias into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after gate beta sequence copy: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = ullm_runtime_sys::linear_attn_gate_beta_f32(
        &a_sequence_buffer,
        &b_sequence_buffer,
        &a_log_buffer,
        &dt_bias_buffer,
        value_heads,
        sequence_len,
        &mut gate_output_buffer,
        &mut beta_output_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run runtime linear_attn_gate_beta_f32: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after linear attention gate beta: {err}");
        return ExitCode::from(1);
    }

    let mut gate_output_bytes = vec![0_u8; gate_beta_sequence_bytes_len];
    let mut beta_output_bytes = vec![0_u8; gate_beta_sequence_bytes_len];
    if let Err(err) = gate_output_buffer.copy_to_host(0, &mut gate_output_bytes, Some(&mut stream))
    {
        eprintln!("failed to copy gate output back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = beta_output_buffer.copy_to_host(0, &mut beta_output_bytes, Some(&mut stream))
    {
        eprintln!("failed to copy beta output back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after gate beta output copy: {err}");
        return ExitCode::from(1);
    }
    let gate_output = decode_f32_le_values(&gate_output_bytes);
    let beta_output = decode_f32_le_values(&beta_output_bytes);
    let mut gate_beta_max_abs_diff = 0.0_f32;
    for ((gate, expected_gate), (beta, expected_beta)) in gate_output
        .iter()
        .zip(expected_gate.iter())
        .zip(beta_output.iter().zip(expected_beta.iter()))
    {
        let gate_diff = (gate - expected_gate).abs();
        let beta_diff = (beta - expected_beta).abs();
        let diff = gate_diff.max(beta_diff);
        if diff > 1e-4_f32 {
            eprintln!(
                "package-linear-attn-recurrent-smoke gate/beta mismatch for layer={layer_index}: max_abs_diff={diff}"
            );
            return ExitCode::from(1);
        }
        if diff > gate_beta_max_abs_diff {
            gate_beta_max_abs_diff = diff;
        }
    }

    let state_elements = match value_heads
        .checked_mul(key_dim)
        .and_then(|value| value.checked_mul(value_dim))
    {
        Some(value) => value,
        None => {
            eprintln!("linear attention recurrent state element count overflows");
            return ExitCode::from(1);
        }
    };
    let output_elements = match sequence_len.checked_mul(v_elements_per_step) {
        Some(value) => value,
        None => {
            eprintln!("linear attention recurrent output element count overflows");
            return ExitCode::from(1);
        }
    };
    let initial_state = vec![0.0_f32; state_elements];
    let mut expected_state = initial_state.clone();
    let expected_recurrent_output = runtime_host_linear_attn_recurrent_f32(
        &qkv_split.q,
        &qkv_split.k,
        &qkv_split.v,
        &expected_gate,
        &expected_beta,
        key_heads,
        value_heads,
        sequence_len,
        key_dim,
        value_dim,
        &mut expected_state,
    );
    if expected_recurrent_output.len() != output_elements {
        eprintln!("failed to build package linear attention recurrent reference");
        return ExitCode::from(1);
    }

    let q_bytes = encode_f32_to_bytes(&qkv_split.q);
    let k_bytes = encode_f32_to_bytes(&qkv_split.k);
    let v_bytes = encode_f32_to_bytes(&qkv_split.v);
    let state_bytes = encode_f32_to_bytes(&initial_state);
    let output_bytes_len = output_elements * std::mem::size_of::<f32>();
    let mut q_buffer = match context.alloc_buffer(q_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate q runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut k_buffer = match context.alloc_buffer(k_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate k runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut v_buffer = match context.alloc_buffer(v_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate v runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut state_buffer = match context.alloc_buffer(state_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate state runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut recurrent_output_buffer = match context.alloc_buffer(output_bytes_len) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate recurrent output runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = q_buffer.copy_from_host(0, &q_bytes, Some(&mut stream)) {
        eprintln!("failed to copy q data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = k_buffer.copy_from_host(0, &k_bytes, Some(&mut stream)) {
        eprintln!("failed to copy k data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = v_buffer.copy_from_host(0, &v_bytes, Some(&mut stream)) {
        eprintln!("failed to copy v data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = state_buffer.copy_from_host(0, &state_bytes, Some(&mut stream)) {
        eprintln!("failed to copy state data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after recurrent input copy: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = ullm_runtime_sys::linear_attn_recurrent_f32(
        &q_buffer,
        &k_buffer,
        &v_buffer,
        &gate_output_buffer,
        &beta_output_buffer,
        key_heads,
        value_heads,
        sequence_len,
        key_dim,
        value_dim,
        &mut state_buffer,
        &mut recurrent_output_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run runtime linear_attn_recurrent_f32: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after linear attention recurrent: {err}");
        return ExitCode::from(1);
    }

    let mut recurrent_output_bytes = vec![0_u8; output_bytes_len];
    let mut final_state_bytes = vec![0_u8; state_bytes.len()];
    if let Err(err) =
        recurrent_output_buffer.copy_to_host(0, &mut recurrent_output_bytes, Some(&mut stream))
    {
        eprintln!("failed to copy recurrent output back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = state_buffer.copy_to_host(0, &mut final_state_bytes, Some(&mut stream)) {
        eprintln!("failed to copy recurrent state back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after recurrent output copy: {err}");
        return ExitCode::from(1);
    }
    let recurrent_output = decode_f32_le_values(&recurrent_output_bytes);
    let final_state = decode_f32_le_values(&final_state_bytes);
    let mut recurrent_max_abs_diff = 0.0_f32;
    for (lhs, rhs) in recurrent_output
        .iter()
        .zip(expected_recurrent_output.iter())
    {
        let diff = (lhs - rhs).abs();
        let tolerance = 1e-3_f32.max(rhs.abs() * 1e-5_f32);
        if diff > tolerance {
            eprintln!(
                "package-linear-attn-recurrent-smoke output mismatch for layer={layer_index}: max_abs_diff={diff} tolerance={tolerance}"
            );
            return ExitCode::from(1);
        }
        if diff > recurrent_max_abs_diff {
            recurrent_max_abs_diff = diff;
        }
    }
    for (lhs, rhs) in final_state.iter().zip(expected_state.iter()) {
        let diff = (lhs - rhs).abs();
        let tolerance = 1e-3_f32.max(rhs.abs() * 1e-5_f32);
        if diff > tolerance {
            eprintln!(
                "package-linear-attn-recurrent-smoke state mismatch for layer={layer_index}: max_abs_diff={diff} tolerance={tolerance}"
            );
            return ExitCode::from(1);
        }
        if diff > recurrent_max_abs_diff {
            recurrent_max_abs_diff = diff;
        }
    }

    println!(
        "package-linear-attn-recurrent-smoke package={} layer={} qkv_tensor=\"{}\" conv_tensor=\"{}\" a_tensor=\"{}\" b_tensor=\"{}\" a_log_tensor=\"{}\" dt_bias_tensor=\"{}\" hidden={} key_heads={} value_heads={} key_dim={} value_dim={} sequence_len={} kernel_size={} qk_l2_norm={} q_scale={q_scale:.9} conv_dtype={} a_log_dtype={} dt_bias_dtype={} backend={} device_index={} name=\"{}\" q_preview={} k_preview={} v_preview={} gate_preview={} output_preview={} state_preview={} conv_max_abs_diff={conv_max_abs_diff:.9} gate_beta_max_abs_diff={gate_beta_max_abs_diff:.9} recurrent_max_abs_diff={recurrent_max_abs_diff:.9} verified=true",
        path,
        layer_index,
        qkv_tensor,
        conv_tensor,
        a_tensor,
        b_tensor,
        a_log_tensor,
        dt_bias_tensor,
        qkv_cols,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        sequence_len,
        kernel_size,
        qk_l2_norm,
        conv_dtype,
        a_log_dtype,
        dt_bias_dtype,
        info.backend,
        device_index,
        info.name,
        format_f32_preview(&qkv_split.q[..8.min(qkv_split.q.len())]),
        format_f32_preview(&qkv_split.k[..8.min(qkv_split.k.len())]),
        format_f32_preview(&qkv_split.v[..8.min(qkv_split.v.len())]),
        format_f32_preview(&gate_output[..8.min(gate_output.len())]),
        format_f32_preview(&recurrent_output[..8.min(recurrent_output.len())]),
        format_f32_preview(&final_state[..8.min(final_state.len())]),
    );
    ExitCode::SUCCESS
}

fn package_linear_attn_post_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    sequence_len: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-linear-attn-post-smoke requires a .ullm.d path");
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
    let sequence_len = match parse_optional_usize(sequence_len, 4, "sequence length") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("sequence length must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };

    let value_heads = 32_usize;
    let value_dim = 128_usize;
    let hidden = value_heads * value_dim;
    let epsilon = 1e-6_f32;

    let z_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_z.weight");
    let norm_tensor = format!("model.language_model.layers.{layer_index}.linear_attn.norm.weight");
    let out_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.out_proj.weight");

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

    let norm_selector = TensorSelector::Name(norm_tensor.clone());
    let norm_bundle =
        match ullm_engine::package::select_passthrough_payload_bundle(&path, &norm_selector) {
            Ok(bundle) => bundle,
            Err(err) => {
                eprintln!("failed to select package linear attention norm tensor: {err}");
                return ExitCode::from(1);
            }
        };
    if let Err(err) = validate_passthrough_shape_elements(&norm_bundle) {
        eprintln!("invalid linear attention norm shape for {norm_tensor}: {err}");
        return ExitCode::from(1);
    }
    let norm_dtype = match resolve_passthrough_dtype(&norm_bundle, &norm_tensor) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let norm_weight =
        match read_passthrough_payload_f32_bytes(&norm_bundle, chunk_bytes, norm_dtype) {
            Ok(value) => value,
            Err(err) => {
                eprintln!("failed to read passthrough payload for {norm_tensor}: {err}");
                return ExitCode::from(1);
            }
        };
    if norm_weight.len() != value_dim {
        eprintln!(
            "linear attention norm length must match value_dim={value_dim}: len={}",
            norm_weight.len()
        );
        return ExitCode::from(1);
    }

    let mut registry = WeightRegistry::new();
    let (z_rows, z_cols, z_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &z_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize tensor {z_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    if z_rows != hidden {
        eprintln!("z projection rows must match hidden={hidden}: z_rows={z_rows}");
        return ExitCode::from(1);
    }

    let (out_rows, out_cols, out_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &out_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize tensor {out_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    if out_rows != z_cols || out_cols != hidden {
        eprintln!("out projection shape must be [{z_cols},{hidden}], got [{out_rows},{out_cols}]");
        return ExitCode::from(1);
    }

    let hidden_bytes = hidden * std::mem::size_of::<f32>();
    let sequence_elements = match sequence_len.checked_mul(hidden) {
        Some(value) => value,
        None => {
            eprintln!("linear attention post sequence element count overflows");
            return ExitCode::from(1);
        }
    };
    let sequence_bytes_len = match sequence_elements.checked_mul(std::mem::size_of::<f32>()) {
        Some(value) => value,
        None => {
            eprintln!("linear attention post sequence byte size overflows");
            return ExitCode::from(1);
        }
    };
    let input_bytes_len = match z_cols.checked_mul(std::mem::size_of::<f32>()) {
        Some(value) => value,
        None => {
            eprintln!("linear attention post input byte size overflows");
            return ExitCode::from(1);
        }
    };

    let mut hidden_input_buffer = match context.alloc_buffer(input_bytes_len) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate hidden input buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut z_step_buffer = match context.alloc_buffer(hidden_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate z step output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let base_input = deterministic_f32_vector(z_cols);
    let mut z_sequence_bytes = vec![0_u8; sequence_bytes_len];
    for timestep in 0..sequence_len {
        let step_input = linear_attn_step_input(&base_input, timestep);
        let step_input_bytes = encode_f32_to_bytes(&step_input);
        if let Err(err) =
            hidden_input_buffer.copy_from_host(0, &step_input_bytes, Some(&mut stream))
        {
            eprintln!("failed to copy z input timestep {timestep}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = ullm_runtime_sys::matvec_f32(
            &z_matrix,
            &hidden_input_buffer,
            z_rows,
            z_cols,
            &mut z_step_buffer,
            Some(&mut stream),
        ) {
            eprintln!("failed to run z matvec for timestep {timestep}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!("failed to synchronize runtime stream after z timestep {timestep}: {err}");
            return ExitCode::from(1);
        }
        let start = timestep * hidden_bytes;
        let end = start + hidden_bytes;
        if let Err(err) =
            z_step_buffer.copy_to_host(0, &mut z_sequence_bytes[start..end], Some(&mut stream))
        {
            eprintln!("failed to copy z timestep {timestep} back to host: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!(
                "failed to synchronize runtime stream after z timestep {timestep} host copy: {err}"
            );
            return ExitCode::from(1);
        }
    }
    let z_sequence = decode_f32_le_values(&z_sequence_bytes);

    let core_output = deterministic_linear_attn_core_output(sequence_len, value_heads, value_dim);
    let mut expected_normed = vec![0.0_f32; sequence_elements];
    for timestep in 0..sequence_len {
        for value_head in 0..value_heads {
            let start = (timestep * value_heads + value_head) * value_dim;
            let end = start + value_dim;
            let normed = runtime_host_rmsnorm_f32(&core_output[start..end], &norm_weight, epsilon);
            if normed.len() != value_dim {
                eprintln!("failed to build linear attention post RMSNorm reference");
                return ExitCode::from(1);
            }
            expected_normed[start..end].copy_from_slice(&normed);
        }
    }
    let expected_activated = runtime_host_silu_mul_f32(&z_sequence, &expected_normed);
    if expected_activated.len() != sequence_elements {
        eprintln!("failed to build linear attention gated RMSNorm reference");
        return ExitCode::from(1);
    }

    let norm_weight_bytes = encode_f32_to_bytes(&norm_weight);
    let mut norm_weight_buffer = match context.alloc_buffer(norm_weight_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate linear attention norm weight buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = norm_weight_buffer.copy_from_host(0, &norm_weight_bytes, Some(&mut stream)) {
        eprintln!("failed to copy linear attention norm weight into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after norm weight copy: {err}");
        return ExitCode::from(1);
    }

    let mut norm_input_buffer = match context.alloc_buffer(norm_weight_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate linear attention norm input buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut norm_output_buffer = match context.alloc_buffer(norm_weight_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate linear attention norm output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut normed_sequence_bytes = vec![0_u8; sequence_bytes_len];
    for row in 0..(sequence_len * value_heads) {
        let start = row * value_dim;
        let end = start + value_dim;
        let input_bytes = encode_f32_to_bytes(&core_output[start..end]);
        if let Err(err) = norm_input_buffer.copy_from_host(0, &input_bytes, Some(&mut stream)) {
            eprintln!("failed to copy linear attention norm row {row}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = ullm_runtime_sys::rmsnorm_f32(
            &norm_input_buffer,
            &norm_weight_buffer,
            value_dim,
            epsilon,
            &mut norm_output_buffer,
            Some(&mut stream),
        ) {
            eprintln!("failed to run linear attention post rmsnorm row {row}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!("failed to synchronize runtime stream after norm row {row}: {err}");
            return ExitCode::from(1);
        }
        let byte_start = start * std::mem::size_of::<f32>();
        let byte_end = end * std::mem::size_of::<f32>();
        if let Err(err) = norm_output_buffer.copy_to_host(
            0,
            &mut normed_sequence_bytes[byte_start..byte_end],
            Some(&mut stream),
        ) {
            eprintln!("failed to copy linear attention norm row {row} back to host: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!("failed to synchronize runtime stream after norm row {row} host copy: {err}");
            return ExitCode::from(1);
        }
    }
    let normed_sequence = decode_f32_le_values(&normed_sequence_bytes);
    let mut norm_max_abs_diff = 0.0_f32;
    for (lhs, rhs) in normed_sequence.iter().zip(expected_normed.iter()) {
        let diff = (lhs - rhs).abs();
        if diff > 1e-4_f32 {
            eprintln!("package-linear-attn-post-smoke RMSNorm mismatch: max_abs_diff={diff}");
            return ExitCode::from(1);
        }
        if diff > norm_max_abs_diff {
            norm_max_abs_diff = diff;
        }
    }

    let mut z_sequence_buffer = match context.alloc_buffer(sequence_bytes_len) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate z sequence buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut normed_sequence_buffer = match context.alloc_buffer(sequence_bytes_len) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate normed sequence buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut activated_sequence_buffer = match context.alloc_buffer(sequence_bytes_len) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate activated sequence buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = z_sequence_buffer.copy_from_host(0, &z_sequence_bytes, Some(&mut stream)) {
        eprintln!("failed to copy z sequence into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) =
        normed_sequence_buffer.copy_from_host(0, &normed_sequence_bytes, Some(&mut stream))
    {
        eprintln!("failed to copy normed sequence into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after gated RMSNorm input copy: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = ullm_runtime_sys::silu_mul_f32(
        &z_sequence_buffer,
        &normed_sequence_buffer,
        sequence_elements,
        &mut activated_sequence_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run linear attention post silu_mul: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after gated RMSNorm silu_mul: {err}");
        return ExitCode::from(1);
    }
    let mut activated_sequence_bytes = vec![0_u8; sequence_bytes_len];
    if let Err(err) =
        activated_sequence_buffer.copy_to_host(0, &mut activated_sequence_bytes, Some(&mut stream))
    {
        eprintln!("failed to copy activated sequence back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after activated sequence copy: {err}");
        return ExitCode::from(1);
    }
    let activated_sequence = decode_f32_le_values(&activated_sequence_bytes);
    let mut activation_max_abs_diff = 0.0_f32;
    for (lhs, rhs) in activated_sequence.iter().zip(expected_activated.iter()) {
        let diff = (lhs - rhs).abs();
        if diff > 1e-4_f32 {
            eprintln!("package-linear-attn-post-smoke gated RMSNorm mismatch: max_abs_diff={diff}");
            return ExitCode::from(1);
        }
        if diff > activation_max_abs_diff {
            activation_max_abs_diff = diff;
        }
    }

    let out_matrix_bytes_len = match out_rows
        .checked_mul(out_cols)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
    {
        Some(value) => value,
        None => {
            eprintln!("out projection matrix byte size overflows");
            return ExitCode::from(1);
        }
    };
    let mut out_matrix_bytes = vec![0_u8; out_matrix_bytes_len];
    if let Err(err) = out_matrix.copy_to_host(0, &mut out_matrix_bytes, Some(&mut stream)) {
        eprintln!("failed to copy out projection matrix back to host for reference: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after out matrix host copy: {err}");
        return ExitCode::from(1);
    }
    let out_matrix_host = decode_f32_le_values(&out_matrix_bytes);
    let mut expected_output = Vec::with_capacity(sequence_len * out_rows);
    for timestep in 0..sequence_len {
        let start = timestep * hidden;
        let end = start + hidden;
        let output = runtime_host_matvec_f32(
            &out_matrix_host,
            &expected_activated[start..end],
            out_rows,
            out_cols,
        );
        if output.len() != out_rows {
            eprintln!("failed to build linear attention post out projection reference");
            return ExitCode::from(1);
        }
        expected_output.extend_from_slice(&output);
    }

    let output_sequence_bytes_len = match sequence_len
        .checked_mul(out_rows)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
    {
        Some(value) => value,
        None => {
            eprintln!("linear attention post output byte size overflows");
            return ExitCode::from(1);
        }
    };
    let mut out_input_buffer = match context.alloc_buffer(hidden_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate out projection input buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut out_step_buffer = match context.alloc_buffer(out_rows * std::mem::size_of::<f32>()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate out projection step buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut output_sequence_bytes = vec![0_u8; output_sequence_bytes_len];
    for timestep in 0..sequence_len {
        let start = timestep * hidden_bytes;
        let end = start + hidden_bytes;
        if let Err(err) = out_input_buffer.copy_from_host(
            0,
            &activated_sequence_bytes[start..end],
            Some(&mut stream),
        ) {
            eprintln!("failed to copy out projection input timestep {timestep}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = ullm_runtime_sys::matvec_f32(
            &out_matrix,
            &out_input_buffer,
            out_rows,
            out_cols,
            &mut out_step_buffer,
            Some(&mut stream),
        ) {
            eprintln!("failed to run out projection matvec timestep {timestep}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!(
                "failed to synchronize runtime stream after out projection timestep {timestep}: {err}"
            );
            return ExitCode::from(1);
        }
        let output_start = timestep * out_rows * std::mem::size_of::<f32>();
        let output_end = output_start + out_rows * std::mem::size_of::<f32>();
        if let Err(err) = out_step_buffer.copy_to_host(
            0,
            &mut output_sequence_bytes[output_start..output_end],
            Some(&mut stream),
        ) {
            eprintln!("failed to copy out projection timestep {timestep} back to host: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!(
                "failed to synchronize runtime stream after out projection timestep {timestep} host copy: {err}"
            );
            return ExitCode::from(1);
        }
    }
    let output_sequence = decode_f32_le_values(&output_sequence_bytes);
    let mut output_max_abs_diff = 0.0_f32;
    for (lhs, rhs) in output_sequence.iter().zip(expected_output.iter()) {
        let diff = (lhs - rhs).abs();
        let tolerance = 2e-3_f32.max(rhs.abs() * 1e-5_f32);
        if diff > tolerance {
            eprintln!(
                "package-linear-attn-post-smoke out projection mismatch: max_abs_diff={diff} tolerance={tolerance}"
            );
            return ExitCode::from(1);
        }
        if diff > output_max_abs_diff {
            output_max_abs_diff = diff;
        }
    }

    println!(
        "package-linear-attn-post-smoke package={} layer={} z_tensor=\"{}\" norm_tensor=\"{}\" out_tensor=\"{}\" hidden={} value_heads={} value_dim={} sequence_len={} norm_dtype={} backend={} device_index={} name=\"{}\" core_preview={} z_preview={} normed_preview={} activated_preview={} output_preview={} norm_max_abs_diff={norm_max_abs_diff:.9} activation_max_abs_diff={activation_max_abs_diff:.9} output_max_abs_diff={output_max_abs_diff:.9} verified=true",
        path,
        layer_index,
        z_tensor,
        norm_tensor,
        out_tensor,
        hidden,
        value_heads,
        value_dim,
        sequence_len,
        norm_dtype,
        info.backend,
        device_index,
        info.name,
        format_f32_preview(&core_output[..8.min(core_output.len())]),
        format_f32_preview(&z_sequence[..8.min(z_sequence.len())]),
        format_f32_preview(&normed_sequence[..8.min(normed_sequence.len())]),
        format_f32_preview(&activated_sequence[..8.min(activated_sequence.len())]),
        format_f32_preview(&output_sequence[..8.min(output_sequence.len())]),
    );
    ExitCode::SUCCESS
}

fn package_linear_attn_workflow_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    sequence_len: Option<String>,
) -> ExitCode {
    package_linear_attn_workflow_smoke_impl(
        "package-linear-attn-workflow-smoke",
        false,
        path,
        device_index,
        chunk_bytes,
        layer_index,
        sequence_len,
    )
}

fn package_linear_attn_block_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    sequence_len: Option<String>,
) -> ExitCode {
    package_linear_attn_workflow_smoke_impl(
        "package-linear-attn-block-smoke",
        true,
        path,
        device_index,
        chunk_bytes,
        layer_index,
        sequence_len,
    )
}

fn package_linear_attn_workflow_smoke_impl(
    command_name: &str,
    include_block: bool,
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    sequence_len: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("{command_name} requires a .ullm.d path");
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
    let sequence_len = match parse_optional_usize(sequence_len, 4, "sequence length") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("sequence length must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };

    let key_heads = 16_usize;
    let value_heads = 32_usize;
    let key_dim = 128_usize;
    let value_dim = 128_usize;
    let hidden_size = value_heads * value_dim;
    let q_elements_per_step = key_heads * key_dim;
    let k_elements_per_step = key_heads * key_dim;
    let v_elements_per_step = hidden_size;
    let recurrent_channels = q_elements_per_step + k_elements_per_step + v_elements_per_step;
    let q_scale = 1.0_f32 / (key_dim as f32).sqrt();
    let qk_l2_norm = true;
    let epsilon = 1e-6_f32;

    let input_norm_tensor =
        format!("model.language_model.layers.{layer_index}.input_layernorm.weight");
    let qkv_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_qkv.weight");
    let conv_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.conv1d.weight");
    let a_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_a.weight");
    let b_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_b.weight");
    let a_log_tensor = format!("model.language_model.layers.{layer_index}.linear_attn.A_log");
    let dt_bias_tensor = format!("model.language_model.layers.{layer_index}.linear_attn.dt_bias");
    let z_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_z.weight");
    let norm_tensor = format!("model.language_model.layers.{layer_index}.linear_attn.norm.weight");
    let out_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.out_proj.weight");

    let mut input_norm_dtype = String::new();
    let mut input_norm_weight = Vec::new();
    if include_block {
        let input_norm_selector = TensorSelector::Name(input_norm_tensor.clone());
        let input_norm_bundle = match ullm_engine::package::select_passthrough_payload_bundle(
            &path,
            &input_norm_selector,
        ) {
            Ok(bundle) => bundle,
            Err(err) => {
                eprintln!("failed to select package input RMSNorm tensor: {err}");
                return ExitCode::from(1);
            }
        };
        if let Err(err) = validate_passthrough_shape_elements(&input_norm_bundle) {
            eprintln!("invalid input RMSNorm shape for {input_norm_tensor}: {err}");
            return ExitCode::from(1);
        }
        input_norm_dtype = match resolve_passthrough_dtype(&input_norm_bundle, &input_norm_tensor) {
            Ok(value) => value.to_string(),
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::from(1);
            }
        };
        input_norm_weight = match read_passthrough_payload_f32_bytes(
            &input_norm_bundle,
            chunk_bytes,
            &input_norm_dtype,
        ) {
            Ok(value) => value,
            Err(err) => {
                eprintln!("failed to read passthrough payload for {input_norm_tensor}: {err}");
                return ExitCode::from(1);
            }
        };
        if input_norm_weight.len() != hidden_size {
            eprintln!(
                "input RMSNorm length must match hidden_size={hidden_size}: len={}",
                input_norm_weight.len()
            );
            return ExitCode::from(1);
        }
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

    let conv_selector = TensorSelector::Name(conv_tensor.clone());
    let conv_bundle =
        match ullm_engine::package::select_passthrough_payload_bundle(&path, &conv_selector) {
            Ok(bundle) => bundle,
            Err(err) => {
                eprintln!("failed to select package conv1d passthrough tensor: {err}");
                return ExitCode::from(1);
            }
        };
    if conv_bundle.shape.len() != 3 || conv_bundle.shape[1] != 1 {
        eprintln!(
            "conv1d tensor shape must be [channels,1,kernel], got {}",
            format_u64_shape(&conv_bundle.shape)
        );
        return ExitCode::from(1);
    }
    let conv_channels = match usize::try_from(conv_bundle.shape[0]) {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("conv1d tensor has zero channels");
            return ExitCode::from(1);
        }
        Err(_) => {
            eprintln!("conv1d channel count is too large for this host");
            return ExitCode::from(1);
        }
    };
    if conv_channels != recurrent_channels {
        eprintln!(
            "conv1d channels must match Qwen3.5 linear attention q/k/v layout: conv_channels={conv_channels}, expected={recurrent_channels}"
        );
        return ExitCode::from(1);
    }
    let kernel_size = match usize::try_from(conv_bundle.shape[2]) {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("conv1d tensor has zero kernel size");
            return ExitCode::from(1);
        }
        Err(_) => {
            eprintln!("conv1d kernel size is too large for this host");
            return ExitCode::from(1);
        }
    };
    let conv_dtype = match resolve_passthrough_dtype(&conv_bundle, &conv_tensor) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let conv_weight =
        match read_passthrough_payload_f32_bytes(&conv_bundle, chunk_bytes, conv_dtype) {
            Ok(value) => value,
            Err(err) => {
                eprintln!("failed to read passthrough payload for {conv_tensor}: {err}");
                return ExitCode::from(1);
            }
        };
    if conv_weight.len() != conv_channels * kernel_size {
        eprintln!(
            "conv1d weight element count mismatch: expected {} got {}",
            conv_channels * kernel_size,
            conv_weight.len()
        );
        return ExitCode::from(1);
    }

    let a_log_selector = TensorSelector::Name(a_log_tensor.clone());
    let a_log_bundle =
        match ullm_engine::package::select_passthrough_payload_bundle(&path, &a_log_selector) {
            Ok(bundle) => bundle,
            Err(err) => {
                eprintln!("failed to select package A_log passthrough tensor: {err}");
                return ExitCode::from(1);
            }
        };
    if let Err(err) = validate_passthrough_shape_elements(&a_log_bundle) {
        eprintln!("invalid A_log shape for {a_log_tensor}: {err}");
        return ExitCode::from(1);
    }
    let a_log_dtype = match resolve_passthrough_dtype(&a_log_bundle, &a_log_tensor) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let a_log = match read_passthrough_payload_f32_bytes(&a_log_bundle, chunk_bytes, a_log_dtype) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to read passthrough payload for {a_log_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    if a_log.len() != value_heads {
        eprintln!(
            "A_log tensor {a_log_tensor} length must match value_heads={value_heads}: len={}",
            a_log.len()
        );
        return ExitCode::from(1);
    }

    let dt_bias_selector = TensorSelector::Name(dt_bias_tensor.clone());
    let dt_bias_bundle =
        match ullm_engine::package::select_passthrough_payload_bundle(&path, &dt_bias_selector) {
            Ok(bundle) => bundle,
            Err(err) => {
                eprintln!("failed to select package dt_bias passthrough tensor: {err}");
                return ExitCode::from(1);
            }
        };
    if let Err(err) = validate_passthrough_shape_elements(&dt_bias_bundle) {
        eprintln!("invalid dt_bias shape for {dt_bias_tensor}: {err}");
        return ExitCode::from(1);
    }
    let dt_bias_dtype = match resolve_passthrough_dtype(&dt_bias_bundle, &dt_bias_tensor) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let dt_bias =
        match read_passthrough_payload_f32_bytes(&dt_bias_bundle, chunk_bytes, dt_bias_dtype) {
            Ok(value) => value,
            Err(err) => {
                eprintln!("failed to read passthrough payload for {dt_bias_tensor}: {err}");
                return ExitCode::from(1);
            }
        };
    if dt_bias.len() != value_heads {
        eprintln!(
            "dt_bias tensor {dt_bias_tensor} length must match value_heads={value_heads}: len={}",
            dt_bias.len()
        );
        return ExitCode::from(1);
    }

    let norm_selector = TensorSelector::Name(norm_tensor.clone());
    let norm_bundle =
        match ullm_engine::package::select_passthrough_payload_bundle(&path, &norm_selector) {
            Ok(bundle) => bundle,
            Err(err) => {
                eprintln!("failed to select package linear attention norm tensor: {err}");
                return ExitCode::from(1);
            }
        };
    if let Err(err) = validate_passthrough_shape_elements(&norm_bundle) {
        eprintln!("invalid linear attention norm shape for {norm_tensor}: {err}");
        return ExitCode::from(1);
    }
    let norm_dtype = match resolve_passthrough_dtype(&norm_bundle, &norm_tensor) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let norm_weight =
        match read_passthrough_payload_f32_bytes(&norm_bundle, chunk_bytes, norm_dtype) {
            Ok(value) => value,
            Err(err) => {
                eprintln!("failed to read passthrough payload for {norm_tensor}: {err}");
                return ExitCode::from(1);
            }
        };
    if norm_weight.len() != value_dim {
        eprintln!(
            "linear attention norm length must match value_dim={value_dim}: len={}",
            norm_weight.len()
        );
        return ExitCode::from(1);
    }

    let mut registry = WeightRegistry::new();
    let (qkv_rows, qkv_cols, qkv_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &qkv_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize tensor {qkv_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    if qkv_rows != conv_channels {
        eprintln!(
            "qkv rows must match conv1d channels: qkv_rows={qkv_rows}, conv_channels={conv_channels}"
        );
        return ExitCode::from(1);
    }
    if qkv_cols != hidden_size {
        eprintln!("qkv input cols must match hidden_size={hidden_size}: qkv_cols={qkv_cols}");
        return ExitCode::from(1);
    }

    let (a_rows, a_cols, a_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &a_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize tensor {a_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    let (b_rows, b_cols, b_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &b_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize tensor {b_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    let (z_rows, z_cols, z_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &z_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize tensor {z_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    let (out_rows, out_cols, out_matrix) = match materialize_selected_aq4_matrix(
        &mut context,
        &mut stream,
        &mut registry,
        &path,
        &out_tensor,
        chunk_bytes,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to materialize tensor {out_tensor}: {err}");
            return ExitCode::from(1);
        }
    };
    if a_rows != value_heads || b_rows != value_heads {
        eprintln!(
            "linear attention a/b rows must match value_heads={value_heads}: a_rows={a_rows}, b_rows={b_rows}"
        );
        return ExitCode::from(1);
    }
    if a_cols != hidden_size || b_cols != hidden_size {
        eprintln!(
            "linear attention a/b hidden sizes must match hidden_size={hidden_size}: a_cols={a_cols}, b_cols={b_cols}"
        );
        return ExitCode::from(1);
    }
    if z_rows != hidden_size || z_cols != hidden_size {
        eprintln!(
            "z projection shape must be [{hidden_size},{hidden_size}], got [{z_rows},{z_cols}]"
        );
        return ExitCode::from(1);
    }
    if out_rows != hidden_size || out_cols != hidden_size {
        eprintln!(
            "out projection shape must be [{hidden_size},{hidden_size}], got [{out_rows},{out_cols}]"
        );
        return ExitCode::from(1);
    }

    let qkv_step_bytes = qkv_rows * std::mem::size_of::<f32>();
    let gate_beta_step_bytes = value_heads * std::mem::size_of::<f32>();
    let hidden_bytes = hidden_size * std::mem::size_of::<f32>();
    let qkv_sequence_bytes_len = match qkv_step_bytes.checked_mul(sequence_len) {
        Some(value) => value,
        None => {
            eprintln!("qkv sequence byte size overflows");
            return ExitCode::from(1);
        }
    };
    let gate_beta_sequence_bytes_len = match gate_beta_step_bytes.checked_mul(sequence_len) {
        Some(value) => value,
        None => {
            eprintln!("linear attention gate beta sequence byte size overflows");
            return ExitCode::from(1);
        }
    };
    let hidden_sequence_bytes_len = match hidden_bytes.checked_mul(sequence_len) {
        Some(value) => value,
        None => {
            eprintln!("linear attention hidden sequence byte size overflows");
            return ExitCode::from(1);
        }
    };

    let mut input_buffer = match context.alloc_buffer(hidden_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate linear attention input buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let input_norm_weight_bytes = if include_block {
        encode_f32_to_bytes(&input_norm_weight)
    } else {
        Vec::new()
    };
    let mut input_norm_weight_buffer = if include_block {
        Some(match context.alloc_buffer(input_norm_weight_bytes.len()) {
            Ok(buffer) => buffer,
            Err(err) => {
                eprintln!("failed to allocate input RMSNorm weight buffer: {err}");
                return ExitCode::from(1);
            }
        })
    } else {
        None
    };
    if let Some(buffer) = input_norm_weight_buffer.as_mut() {
        if let Err(err) = buffer.copy_from_host(0, &input_norm_weight_bytes, Some(&mut stream)) {
            eprintln!("failed to copy input RMSNorm weight into runtime buffer: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!(
                "failed to synchronize runtime stream after input RMSNorm weight copy: {err}"
            );
            return ExitCode::from(1);
        }
    }
    let mut input_norm_output_buffer = if include_block {
        Some(match context.alloc_buffer(hidden_bytes) {
            Ok(buffer) => buffer,
            Err(err) => {
                eprintln!("failed to allocate input RMSNorm output buffer: {err}");
                return ExitCode::from(1);
            }
        })
    } else {
        None
    };
    let mut qkv_step_buffer = match context.alloc_buffer(qkv_step_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate qkv step output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut a_step_buffer = match context.alloc_buffer(gate_beta_step_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate a step output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut b_step_buffer = match context.alloc_buffer(gate_beta_step_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate b step output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut z_step_buffer = match context.alloc_buffer(hidden_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate z step output buffer: {err}");
            return ExitCode::from(1);
        }
    };

    let base_input = deterministic_f32_vector(hidden_size);
    let mut residual_sequence_bytes = if include_block {
        vec![0_u8; hidden_sequence_bytes_len]
    } else {
        Vec::new()
    };
    let mut input_norm_sequence_bytes = if include_block {
        vec![0_u8; hidden_sequence_bytes_len]
    } else {
        Vec::new()
    };
    let mut expected_input_norm = if include_block {
        Vec::with_capacity(sequence_len * hidden_size)
    } else {
        Vec::new()
    };
    let mut qkv_sequence_bytes = vec![0_u8; qkv_sequence_bytes_len];
    let mut a_sequence_bytes = vec![0_u8; gate_beta_sequence_bytes_len];
    let mut b_sequence_bytes = vec![0_u8; gate_beta_sequence_bytes_len];
    let mut z_sequence_bytes = vec![0_u8; hidden_sequence_bytes_len];
    for timestep in 0..sequence_len {
        let step_input = linear_attn_step_input(&base_input, timestep);
        let step_input_bytes = encode_f32_to_bytes(&step_input);
        if include_block {
            let residual_start = timestep * hidden_bytes;
            let residual_end = residual_start + hidden_bytes;
            residual_sequence_bytes[residual_start..residual_end]
                .copy_from_slice(&step_input_bytes);
            let expected_normed =
                runtime_host_rmsnorm_f32(&step_input, &input_norm_weight, epsilon);
            if expected_normed.len() != hidden_size {
                eprintln!("failed to build input RMSNorm reference for timestep {timestep}");
                return ExitCode::from(1);
            }
            expected_input_norm.extend_from_slice(&expected_normed);
        }
        if let Err(err) = input_buffer.copy_from_host(0, &step_input_bytes, Some(&mut stream)) {
            eprintln!("failed to copy linear attention input timestep {timestep}: {err}");
            return ExitCode::from(1);
        }
        if include_block {
            let input_norm_weight_buffer = input_norm_weight_buffer
                .as_ref()
                .expect("input RMSNorm weight buffer exists in block mode");
            let input_norm_output_buffer = input_norm_output_buffer
                .as_mut()
                .expect("input RMSNorm output buffer exists in block mode");
            if let Err(err) = ullm_runtime_sys::rmsnorm_f32(
                &input_buffer,
                input_norm_weight_buffer,
                hidden_size,
                epsilon,
                input_norm_output_buffer,
                Some(&mut stream),
            ) {
                eprintln!("failed to run input RMSNorm timestep {timestep}: {err}");
                return ExitCode::from(1);
            }
            if let Err(err) = stream.synchronize() {
                eprintln!(
                    "failed to synchronize runtime stream after input RMSNorm timestep {timestep}: {err}"
                );
                return ExitCode::from(1);
            }
            let norm_start = timestep * hidden_bytes;
            let norm_end = norm_start + hidden_bytes;
            if let Err(err) = input_norm_output_buffer.copy_to_host(
                0,
                &mut input_norm_sequence_bytes[norm_start..norm_end],
                Some(&mut stream),
            ) {
                eprintln!("failed to copy input RMSNorm timestep {timestep} back to host: {err}");
                return ExitCode::from(1);
            }
            if let Err(err) = stream.synchronize() {
                eprintln!(
                    "failed to synchronize runtime stream after input RMSNorm timestep {timestep} host copy: {err}"
                );
                return ExitCode::from(1);
            }
        }
        let projection_input_buffer = if include_block {
            input_norm_output_buffer
                .as_ref()
                .expect("input RMSNorm output buffer exists in block mode")
        } else {
            &input_buffer
        };
        if let Err(err) = ullm_runtime_sys::matvec_f32(
            &qkv_matrix,
            projection_input_buffer,
            qkv_rows,
            qkv_cols,
            &mut qkv_step_buffer,
            Some(&mut stream),
        ) {
            eprintln!("failed to run qkv matvec for timestep {timestep}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = ullm_runtime_sys::matvec_f32(
            &a_matrix,
            projection_input_buffer,
            value_heads,
            hidden_size,
            &mut a_step_buffer,
            Some(&mut stream),
        ) {
            eprintln!("failed to run a matvec for timestep {timestep}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = ullm_runtime_sys::matvec_f32(
            &b_matrix,
            projection_input_buffer,
            value_heads,
            hidden_size,
            &mut b_step_buffer,
            Some(&mut stream),
        ) {
            eprintln!("failed to run b matvec for timestep {timestep}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = ullm_runtime_sys::matvec_f32(
            &z_matrix,
            projection_input_buffer,
            hidden_size,
            hidden_size,
            &mut z_step_buffer,
            Some(&mut stream),
        ) {
            eprintln!("failed to run z matvec for timestep {timestep}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!(
                "failed to synchronize runtime stream after linear attention timestep {timestep}: {err}"
            );
            return ExitCode::from(1);
        }

        let qkv_start = timestep * qkv_step_bytes;
        let qkv_end = qkv_start + qkv_step_bytes;
        if let Err(err) = qkv_step_buffer.copy_to_host(
            0,
            &mut qkv_sequence_bytes[qkv_start..qkv_end],
            Some(&mut stream),
        ) {
            eprintln!("failed to copy qkv timestep {timestep} back to host: {err}");
            return ExitCode::from(1);
        }
        let gate_beta_start = timestep * gate_beta_step_bytes;
        let gate_beta_end = gate_beta_start + gate_beta_step_bytes;
        if let Err(err) = a_step_buffer.copy_to_host(
            0,
            &mut a_sequence_bytes[gate_beta_start..gate_beta_end],
            Some(&mut stream),
        ) {
            eprintln!("failed to copy a timestep {timestep} back to host: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = b_step_buffer.copy_to_host(
            0,
            &mut b_sequence_bytes[gate_beta_start..gate_beta_end],
            Some(&mut stream),
        ) {
            eprintln!("failed to copy b timestep {timestep} back to host: {err}");
            return ExitCode::from(1);
        }
        let z_start = timestep * hidden_bytes;
        let z_end = z_start + hidden_bytes;
        if let Err(err) =
            z_step_buffer.copy_to_host(0, &mut z_sequence_bytes[z_start..z_end], Some(&mut stream))
        {
            eprintln!("failed to copy z timestep {timestep} back to host: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!(
                "failed to synchronize runtime stream after timestep {timestep} host copy: {err}"
            );
            return ExitCode::from(1);
        }
    }

    let input_norm_sequence = if include_block {
        decode_f32_le_values(&input_norm_sequence_bytes)
    } else {
        Vec::new()
    };
    let mut input_norm_max_abs_diff = 0.0_f32;
    if include_block {
        if input_norm_sequence.len() != expected_input_norm.len() {
            eprintln!(
                "{command_name} input RMSNorm output size mismatch: expected {} got {}",
                expected_input_norm.len(),
                input_norm_sequence.len()
            );
            return ExitCode::from(1);
        }
        for (lhs, rhs) in input_norm_sequence.iter().zip(expected_input_norm.iter()) {
            let diff = (lhs - rhs).abs();
            let tolerance = 1e-4_f32.max(rhs.abs() * 1e-5_f32);
            if diff > tolerance {
                eprintln!(
                    "{command_name} input RMSNorm mismatch for layer={layer_index}: max_abs_diff={diff} tolerance={tolerance}"
                );
                return ExitCode::from(1);
            }
            if diff > input_norm_max_abs_diff {
                input_norm_max_abs_diff = diff;
            }
        }
    }

    let qkv_sequence = decode_f32_le_values(&qkv_sequence_bytes);
    let expected_conv = runtime_host_depthwise_conv1d_f32(
        &qkv_sequence,
        &conv_weight,
        qkv_rows,
        sequence_len,
        kernel_size,
    );
    if expected_conv.is_empty() {
        eprintln!("failed to build package depthwise conv1d reference");
        return ExitCode::from(1);
    }

    let mut conv_input_buffer = match context.alloc_buffer(qkv_sequence_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate conv1d input buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let conv_weight_bytes = encode_f32_to_bytes(&conv_weight);
    let mut conv_weight_buffer = match context.alloc_buffer(conv_weight_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate conv1d weight buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut conv_output_buffer = match context.alloc_buffer(qkv_sequence_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate conv1d output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = conv_input_buffer.copy_from_host(0, &qkv_sequence_bytes, Some(&mut stream)) {
        eprintln!("failed to copy conv1d input sequence into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = conv_weight_buffer.copy_from_host(0, &conv_weight_bytes, Some(&mut stream)) {
        eprintln!("failed to copy conv1d weight into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after conv1d input copy: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = ullm_runtime_sys::depthwise_conv1d_f32(
        &conv_input_buffer,
        &conv_weight_buffer,
        qkv_rows,
        sequence_len,
        kernel_size,
        &mut conv_output_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run runtime depthwise_conv1d_f32: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after conv1d: {err}");
        return ExitCode::from(1);
    }
    let mut conv_output_bytes = vec![0_u8; qkv_sequence_bytes.len()];
    if let Err(err) = conv_output_buffer.copy_to_host(0, &mut conv_output_bytes, Some(&mut stream))
    {
        eprintln!("failed to copy conv1d output back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after conv1d output copy: {err}");
        return ExitCode::from(1);
    }
    let conv_output = decode_f32_le_values(&conv_output_bytes);
    let mut conv_max_abs_diff = 0.0_f32;
    for (lhs, rhs) in conv_output.iter().zip(expected_conv.iter()) {
        let diff = (lhs - rhs).abs();
        if diff > 1e-4_f32 {
            eprintln!(
                "package-linear-attn-workflow-smoke conv1d mismatch for layer={layer_index}: max_abs_diff={diff}"
            );
            return ExitCode::from(1);
        }
        if diff > conv_max_abs_diff {
            conv_max_abs_diff = diff;
        }
    }

    let qkv_split = match split_linear_attn_qkv_for_recurrent(
        &conv_output,
        sequence_len,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        qk_l2_norm,
        q_scale,
    ) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("failed to split linear attention qkv: {err}");
            return ExitCode::from(1);
        }
    };

    let a_sequence = decode_f32_le_values(&a_sequence_bytes);
    let b_sequence = decode_f32_le_values(&b_sequence_bytes);
    let z_sequence = decode_f32_le_values(&z_sequence_bytes);
    let (expected_gate, expected_beta) = runtime_host_linear_attn_gate_beta_f32(
        &a_sequence,
        &b_sequence,
        &a_log,
        &dt_bias,
        value_heads,
        sequence_len,
    );
    if expected_gate.is_empty() || expected_beta.is_empty() {
        eprintln!("failed to build package linear attention gate beta reference");
        return ExitCode::from(1);
    }

    let a_log_bytes = encode_f32_to_bytes(&a_log);
    let dt_bias_bytes = encode_f32_to_bytes(&dt_bias);
    let mut a_sequence_buffer = match context.alloc_buffer(a_sequence_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate a sequence buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut b_sequence_buffer = match context.alloc_buffer(b_sequence_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate b sequence buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut a_log_buffer = match context.alloc_buffer(a_log_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate A_log buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut dt_bias_buffer = match context.alloc_buffer(dt_bias_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate dt_bias buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut gate_output_buffer = match context.alloc_buffer(gate_beta_sequence_bytes_len) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate gate output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut beta_output_buffer = match context.alloc_buffer(gate_beta_sequence_bytes_len) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate beta output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = a_sequence_buffer.copy_from_host(0, &a_sequence_bytes, Some(&mut stream)) {
        eprintln!("failed to copy a sequence into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = b_sequence_buffer.copy_from_host(0, &b_sequence_bytes, Some(&mut stream)) {
        eprintln!("failed to copy b sequence into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = a_log_buffer.copy_from_host(0, &a_log_bytes, Some(&mut stream)) {
        eprintln!("failed to copy A_log into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = dt_bias_buffer.copy_from_host(0, &dt_bias_bytes, Some(&mut stream)) {
        eprintln!("failed to copy dt_bias into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after gate beta sequence copy: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = ullm_runtime_sys::linear_attn_gate_beta_f32(
        &a_sequence_buffer,
        &b_sequence_buffer,
        &a_log_buffer,
        &dt_bias_buffer,
        value_heads,
        sequence_len,
        &mut gate_output_buffer,
        &mut beta_output_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run runtime linear_attn_gate_beta_f32: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after linear attention gate beta: {err}");
        return ExitCode::from(1);
    }
    let mut gate_output_bytes = vec![0_u8; gate_beta_sequence_bytes_len];
    let mut beta_output_bytes = vec![0_u8; gate_beta_sequence_bytes_len];
    if let Err(err) = gate_output_buffer.copy_to_host(0, &mut gate_output_bytes, Some(&mut stream))
    {
        eprintln!("failed to copy gate output back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = beta_output_buffer.copy_to_host(0, &mut beta_output_bytes, Some(&mut stream))
    {
        eprintln!("failed to copy beta output back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after gate beta output copy: {err}");
        return ExitCode::from(1);
    }
    let gate_output = decode_f32_le_values(&gate_output_bytes);
    let beta_output = decode_f32_le_values(&beta_output_bytes);
    let mut gate_beta_max_abs_diff = 0.0_f32;
    for ((gate, expected_gate), (beta, expected_beta)) in gate_output
        .iter()
        .zip(expected_gate.iter())
        .zip(beta_output.iter().zip(expected_beta.iter()))
    {
        let gate_diff = (gate - expected_gate).abs();
        let beta_diff = (beta - expected_beta).abs();
        let diff = gate_diff.max(beta_diff);
        if diff > 1e-4_f32 {
            eprintln!(
                "package-linear-attn-workflow-smoke gate/beta mismatch for layer={layer_index}: max_abs_diff={diff}"
            );
            return ExitCode::from(1);
        }
        if diff > gate_beta_max_abs_diff {
            gate_beta_max_abs_diff = diff;
        }
    }

    let state_elements = match value_heads
        .checked_mul(key_dim)
        .and_then(|value| value.checked_mul(value_dim))
    {
        Some(value) => value,
        None => {
            eprintln!("linear attention recurrent state element count overflows");
            return ExitCode::from(1);
        }
    };
    let recurrent_output_elements = match sequence_len.checked_mul(hidden_size) {
        Some(value) => value,
        None => {
            eprintln!("linear attention recurrent output element count overflows");
            return ExitCode::from(1);
        }
    };
    let initial_state = vec![0.0_f32; state_elements];
    let mut expected_state = initial_state.clone();
    let expected_recurrent_output = runtime_host_linear_attn_recurrent_f32(
        &qkv_split.q,
        &qkv_split.k,
        &qkv_split.v,
        &expected_gate,
        &expected_beta,
        key_heads,
        value_heads,
        sequence_len,
        key_dim,
        value_dim,
        &mut expected_state,
    );
    if expected_recurrent_output.len() != recurrent_output_elements {
        eprintln!("failed to build package linear attention recurrent reference");
        return ExitCode::from(1);
    }

    let q_bytes = encode_f32_to_bytes(&qkv_split.q);
    let k_bytes = encode_f32_to_bytes(&qkv_split.k);
    let v_bytes = encode_f32_to_bytes(&qkv_split.v);
    let state_bytes = encode_f32_to_bytes(&initial_state);
    let recurrent_output_bytes_len = recurrent_output_elements * std::mem::size_of::<f32>();
    let mut q_buffer = match context.alloc_buffer(q_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate q runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut k_buffer = match context.alloc_buffer(k_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate k runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut v_buffer = match context.alloc_buffer(v_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate v runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut state_buffer = match context.alloc_buffer(state_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate state runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut recurrent_output_buffer = match context.alloc_buffer(recurrent_output_bytes_len) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate recurrent output runtime buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = q_buffer.copy_from_host(0, &q_bytes, Some(&mut stream)) {
        eprintln!("failed to copy q data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = k_buffer.copy_from_host(0, &k_bytes, Some(&mut stream)) {
        eprintln!("failed to copy k data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = v_buffer.copy_from_host(0, &v_bytes, Some(&mut stream)) {
        eprintln!("failed to copy v data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = state_buffer.copy_from_host(0, &state_bytes, Some(&mut stream)) {
        eprintln!("failed to copy state data into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after recurrent input copy: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = ullm_runtime_sys::linear_attn_recurrent_f32(
        &q_buffer,
        &k_buffer,
        &v_buffer,
        &gate_output_buffer,
        &beta_output_buffer,
        key_heads,
        value_heads,
        sequence_len,
        key_dim,
        value_dim,
        &mut state_buffer,
        &mut recurrent_output_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run runtime linear_attn_recurrent_f32: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after linear attention recurrent: {err}");
        return ExitCode::from(1);
    }
    let mut recurrent_output_bytes = vec![0_u8; recurrent_output_bytes_len];
    let mut final_state_bytes = vec![0_u8; state_bytes.len()];
    if let Err(err) =
        recurrent_output_buffer.copy_to_host(0, &mut recurrent_output_bytes, Some(&mut stream))
    {
        eprintln!("failed to copy recurrent output back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = state_buffer.copy_to_host(0, &mut final_state_bytes, Some(&mut stream)) {
        eprintln!("failed to copy recurrent state back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after recurrent output copy: {err}");
        return ExitCode::from(1);
    }
    let recurrent_output = decode_f32_le_values(&recurrent_output_bytes);
    let final_state = decode_f32_le_values(&final_state_bytes);
    let mut recurrent_max_abs_diff = 0.0_f32;
    for (lhs, rhs) in recurrent_output
        .iter()
        .zip(expected_recurrent_output.iter())
    {
        let diff = (lhs - rhs).abs();
        let tolerance = 1e-3_f32.max(rhs.abs() * 1e-5_f32);
        if diff > tolerance {
            eprintln!(
                "package-linear-attn-workflow-smoke recurrent output mismatch for layer={layer_index}: max_abs_diff={diff} tolerance={tolerance}"
            );
            return ExitCode::from(1);
        }
        if diff > recurrent_max_abs_diff {
            recurrent_max_abs_diff = diff;
        }
    }
    for (lhs, rhs) in final_state.iter().zip(expected_state.iter()) {
        let diff = (lhs - rhs).abs();
        let tolerance = 1e-3_f32.max(rhs.abs() * 1e-5_f32);
        if diff > tolerance {
            eprintln!(
                "package-linear-attn-workflow-smoke recurrent state mismatch for layer={layer_index}: max_abs_diff={diff} tolerance={tolerance}"
            );
            return ExitCode::from(1);
        }
        if diff > recurrent_max_abs_diff {
            recurrent_max_abs_diff = diff;
        }
    }

    let mut expected_normed = vec![0.0_f32; recurrent_output_elements];
    for timestep in 0..sequence_len {
        for value_head in 0..value_heads {
            let start = (timestep * value_heads + value_head) * value_dim;
            let end = start + value_dim;
            let normed = runtime_host_rmsnorm_f32(
                &expected_recurrent_output[start..end],
                &norm_weight,
                epsilon,
            );
            if normed.len() != value_dim {
                eprintln!("failed to build linear attention workflow RMSNorm reference");
                return ExitCode::from(1);
            }
            expected_normed[start..end].copy_from_slice(&normed);
        }
    }
    let expected_activated = runtime_host_silu_mul_f32(&z_sequence, &expected_normed);
    if expected_activated.len() != recurrent_output_elements {
        eprintln!("failed to build linear attention workflow gated RMSNorm reference");
        return ExitCode::from(1);
    }

    let norm_weight_bytes = encode_f32_to_bytes(&norm_weight);
    let mut norm_weight_buffer = match context.alloc_buffer(norm_weight_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate linear attention norm weight buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = norm_weight_buffer.copy_from_host(0, &norm_weight_bytes, Some(&mut stream)) {
        eprintln!("failed to copy linear attention norm weight into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after norm weight copy: {err}");
        return ExitCode::from(1);
    }
    let mut norm_input_buffer = match context.alloc_buffer(norm_weight_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate linear attention norm input buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut norm_output_buffer = match context.alloc_buffer(norm_weight_bytes.len()) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate linear attention norm output buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut normed_sequence_bytes = vec![0_u8; recurrent_output_bytes_len];
    for row in 0..(sequence_len * value_heads) {
        let start = row * value_dim;
        let end = start + value_dim;
        let byte_start = start * std::mem::size_of::<f32>();
        let byte_end = end * std::mem::size_of::<f32>();
        if let Err(err) = norm_input_buffer.copy_from_host(
            0,
            &recurrent_output_bytes[byte_start..byte_end],
            Some(&mut stream),
        ) {
            eprintln!("failed to copy linear attention workflow norm row {row}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = ullm_runtime_sys::rmsnorm_f32(
            &norm_input_buffer,
            &norm_weight_buffer,
            value_dim,
            epsilon,
            &mut norm_output_buffer,
            Some(&mut stream),
        ) {
            eprintln!("failed to run linear attention workflow rmsnorm row {row}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!("failed to synchronize runtime stream after workflow norm row {row}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = norm_output_buffer.copy_to_host(
            0,
            &mut normed_sequence_bytes[byte_start..byte_end],
            Some(&mut stream),
        ) {
            eprintln!(
                "failed to copy linear attention workflow norm row {row} back to host: {err}"
            );
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!(
                "failed to synchronize runtime stream after workflow norm row {row} host copy: {err}"
            );
            return ExitCode::from(1);
        }
    }
    let normed_sequence = decode_f32_le_values(&normed_sequence_bytes);
    let mut norm_max_abs_diff = 0.0_f32;
    for (lhs, rhs) in normed_sequence.iter().zip(expected_normed.iter()) {
        let diff = (lhs - rhs).abs();
        let tolerance = 1e-3_f32.max(rhs.abs() * 1e-5_f32);
        if diff > tolerance {
            eprintln!(
                "package-linear-attn-workflow-smoke RMSNorm mismatch: max_abs_diff={diff} tolerance={tolerance}"
            );
            return ExitCode::from(1);
        }
        if diff > norm_max_abs_diff {
            norm_max_abs_diff = diff;
        }
    }

    let mut z_sequence_buffer = match context.alloc_buffer(hidden_sequence_bytes_len) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate z sequence buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut normed_sequence_buffer = match context.alloc_buffer(recurrent_output_bytes_len) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate normed sequence buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut activated_sequence_buffer = match context.alloc_buffer(recurrent_output_bytes_len) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate activated sequence buffer: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = z_sequence_buffer.copy_from_host(0, &z_sequence_bytes, Some(&mut stream)) {
        eprintln!("failed to copy z sequence into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) =
        normed_sequence_buffer.copy_from_host(0, &normed_sequence_bytes, Some(&mut stream))
    {
        eprintln!("failed to copy normed sequence into runtime buffer: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after gated RMSNorm input copy: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = ullm_runtime_sys::silu_mul_f32(
        &z_sequence_buffer,
        &normed_sequence_buffer,
        recurrent_output_elements,
        &mut activated_sequence_buffer,
        Some(&mut stream),
    ) {
        eprintln!("failed to run linear attention workflow silu_mul: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after workflow silu_mul: {err}");
        return ExitCode::from(1);
    }
    let mut activated_sequence_bytes = vec![0_u8; recurrent_output_bytes_len];
    if let Err(err) =
        activated_sequence_buffer.copy_to_host(0, &mut activated_sequence_bytes, Some(&mut stream))
    {
        eprintln!("failed to copy activated sequence back to host: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after activated sequence copy: {err}");
        return ExitCode::from(1);
    }
    let activated_sequence = decode_f32_le_values(&activated_sequence_bytes);
    let mut activation_max_abs_diff = 0.0_f32;
    for (lhs, rhs) in activated_sequence.iter().zip(expected_activated.iter()) {
        let diff = (lhs - rhs).abs();
        let tolerance = 1e-3_f32.max(rhs.abs() * 1e-5_f32);
        if diff > tolerance {
            eprintln!(
                "package-linear-attn-workflow-smoke gated RMSNorm mismatch: max_abs_diff={diff} tolerance={tolerance}"
            );
            return ExitCode::from(1);
        }
        if diff > activation_max_abs_diff {
            activation_max_abs_diff = diff;
        }
    }

    let out_matrix_bytes_len = match out_rows
        .checked_mul(out_cols)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
    {
        Some(value) => value,
        None => {
            eprintln!("out projection matrix byte size overflows");
            return ExitCode::from(1);
        }
    };
    let mut out_matrix_bytes = vec![0_u8; out_matrix_bytes_len];
    if let Err(err) = out_matrix.copy_to_host(0, &mut out_matrix_bytes, Some(&mut stream)) {
        eprintln!("failed to copy out projection matrix back to host for reference: {err}");
        return ExitCode::from(1);
    }
    if let Err(err) = stream.synchronize() {
        eprintln!("failed to synchronize runtime stream after out matrix host copy: {err}");
        return ExitCode::from(1);
    }
    let out_matrix_host = decode_f32_le_values(&out_matrix_bytes);
    let mut expected_output = Vec::with_capacity(sequence_len * hidden_size);
    for timestep in 0..sequence_len {
        let start = timestep * hidden_size;
        let end = start + hidden_size;
        let output = runtime_host_matvec_f32(
            &out_matrix_host,
            &expected_activated[start..end],
            out_rows,
            out_cols,
        );
        if output.len() != out_rows {
            eprintln!("failed to build linear attention workflow out projection reference");
            return ExitCode::from(1);
        }
        expected_output.extend_from_slice(&output);
    }
    let residual_sequence = if include_block {
        decode_f32_le_values(&residual_sequence_bytes)
    } else {
        Vec::new()
    };
    let expected_block_output = if include_block {
        let output = runtime_host_add_f32(&residual_sequence, &expected_output);
        if output.len() != expected_output.len() {
            eprintln!("failed to build {command_name} residual add reference");
            return ExitCode::from(1);
        }
        output
    } else {
        Vec::new()
    };

    let mut out_input_buffer = match context.alloc_buffer(hidden_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate out projection input buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut out_step_buffer = match context.alloc_buffer(hidden_bytes) {
        Ok(buffer) => buffer,
        Err(err) => {
            eprintln!("failed to allocate out projection step buffer: {err}");
            return ExitCode::from(1);
        }
    };
    let mut residual_step_buffer = if include_block {
        Some(match context.alloc_buffer(hidden_bytes) {
            Ok(buffer) => buffer,
            Err(err) => {
                eprintln!("failed to allocate residual step buffer: {err}");
                return ExitCode::from(1);
            }
        })
    } else {
        None
    };
    let mut block_step_buffer = if include_block {
        Some(match context.alloc_buffer(hidden_bytes) {
            Ok(buffer) => buffer,
            Err(err) => {
                eprintln!("failed to allocate block step output buffer: {err}");
                return ExitCode::from(1);
            }
        })
    } else {
        None
    };
    let mut output_sequence_bytes = vec![0_u8; hidden_sequence_bytes_len];
    let mut block_sequence_bytes = if include_block {
        vec![0_u8; hidden_sequence_bytes_len]
    } else {
        Vec::new()
    };
    for timestep in 0..sequence_len {
        let start = timestep * hidden_bytes;
        let end = start + hidden_bytes;
        if let Err(err) = out_input_buffer.copy_from_host(
            0,
            &activated_sequence_bytes[start..end],
            Some(&mut stream),
        ) {
            eprintln!("failed to copy out projection input timestep {timestep}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = ullm_runtime_sys::matvec_f32(
            &out_matrix,
            &out_input_buffer,
            out_rows,
            out_cols,
            &mut out_step_buffer,
            Some(&mut stream),
        ) {
            eprintln!("failed to run out projection matvec timestep {timestep}: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!(
                "failed to synchronize runtime stream after out projection timestep {timestep}: {err}"
            );
            return ExitCode::from(1);
        }
        if let Err(err) = out_step_buffer.copy_to_host(
            0,
            &mut output_sequence_bytes[start..end],
            Some(&mut stream),
        ) {
            eprintln!("failed to copy out projection timestep {timestep} back to host: {err}");
            return ExitCode::from(1);
        }
        if let Err(err) = stream.synchronize() {
            eprintln!(
                "failed to synchronize runtime stream after out projection timestep {timestep} host copy: {err}"
            );
            return ExitCode::from(1);
        }
        if include_block {
            let residual_step_buffer = residual_step_buffer
                .as_mut()
                .expect("residual step buffer exists in block mode");
            let block_step_buffer = block_step_buffer
                .as_mut()
                .expect("block step output buffer exists in block mode");
            if let Err(err) = residual_step_buffer.copy_from_host(
                0,
                &residual_sequence_bytes[start..end],
                Some(&mut stream),
            ) {
                eprintln!("failed to copy residual timestep {timestep} into runtime buffer: {err}");
                return ExitCode::from(1);
            }
            if let Err(err) = ullm_runtime_sys::add_f32(
                residual_step_buffer,
                &out_step_buffer,
                hidden_size,
                block_step_buffer,
                Some(&mut stream),
            ) {
                eprintln!("failed to run runtime residual add timestep {timestep}: {err}");
                return ExitCode::from(1);
            }
            if let Err(err) = stream.synchronize() {
                eprintln!(
                    "failed to synchronize runtime stream after residual add timestep {timestep}: {err}"
                );
                return ExitCode::from(1);
            }
            if let Err(err) = block_step_buffer.copy_to_host(
                0,
                &mut block_sequence_bytes[start..end],
                Some(&mut stream),
            ) {
                eprintln!("failed to copy block output timestep {timestep} back to host: {err}");
                return ExitCode::from(1);
            }
            if let Err(err) = stream.synchronize() {
                eprintln!(
                    "failed to synchronize runtime stream after block output timestep {timestep} host copy: {err}"
                );
                return ExitCode::from(1);
            }
        }
    }
    let output_sequence = decode_f32_le_values(&output_sequence_bytes);
    let mut output_max_abs_diff = 0.0_f32;
    for (lhs, rhs) in output_sequence.iter().zip(expected_output.iter()) {
        let diff = (lhs - rhs).abs();
        let tolerance = 3e-3_f32.max(rhs.abs() * 2e-5_f32);
        if diff > tolerance {
            eprintln!(
                "package-linear-attn-workflow-smoke out projection mismatch: max_abs_diff={diff} tolerance={tolerance}"
            );
            return ExitCode::from(1);
        }
        if diff > output_max_abs_diff {
            output_max_abs_diff = diff;
        }
    }

    let block_output = if include_block {
        decode_f32_le_values(&block_sequence_bytes)
    } else {
        Vec::new()
    };
    let mut block_max_abs_diff = 0.0_f32;
    if include_block {
        if block_output.len() != expected_block_output.len() {
            eprintln!(
                "{command_name} output size mismatch: expected {} got {}",
                expected_block_output.len(),
                block_output.len()
            );
            return ExitCode::from(1);
        }
        for (lhs, rhs) in block_output.iter().zip(expected_block_output.iter()) {
            let diff = (lhs - rhs).abs();
            let tolerance = 3e-3_f32.max(rhs.abs() * 2e-5_f32);
            if diff > tolerance {
                eprintln!(
                    "{command_name} residual output mismatch: max_abs_diff={diff} tolerance={tolerance}"
                );
                return ExitCode::from(1);
            }
            if diff > block_max_abs_diff {
                block_max_abs_diff = diff;
            }
        }
    }

    if include_block {
        println!(
            "package-linear-attn-block-smoke package={} layer={} input_norm_tensor=\"{}\" qkv_tensor=\"{}\" conv_tensor=\"{}\" a_tensor=\"{}\" b_tensor=\"{}\" a_log_tensor=\"{}\" dt_bias_tensor=\"{}\" z_tensor=\"{}\" norm_tensor=\"{}\" out_tensor=\"{}\" hidden={} key_heads={} value_heads={} key_dim={} value_dim={} sequence_len={} kernel_size={} qk_l2_norm={} q_scale={q_scale:.9} input_norm_dtype={} conv_dtype={} a_log_dtype={} dt_bias_dtype={} norm_dtype={} backend={} device_index={} name=\"{}\" residual_preview={} input_norm_preview={} workflow_output_preview={} block_output_preview={} input_norm_max_abs_diff={input_norm_max_abs_diff:.9} conv_max_abs_diff={conv_max_abs_diff:.9} gate_beta_max_abs_diff={gate_beta_max_abs_diff:.9} recurrent_max_abs_diff={recurrent_max_abs_diff:.9} norm_max_abs_diff={norm_max_abs_diff:.9} activation_max_abs_diff={activation_max_abs_diff:.9} output_max_abs_diff={output_max_abs_diff:.9} block_max_abs_diff={block_max_abs_diff:.9} verified=true",
            path,
            layer_index,
            input_norm_tensor,
            qkv_tensor,
            conv_tensor,
            a_tensor,
            b_tensor,
            a_log_tensor,
            dt_bias_tensor,
            z_tensor,
            norm_tensor,
            out_tensor,
            hidden_size,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            sequence_len,
            kernel_size,
            qk_l2_norm,
            input_norm_dtype,
            conv_dtype,
            a_log_dtype,
            dt_bias_dtype,
            norm_dtype,
            info.backend,
            device_index,
            info.name,
            format_f32_preview(&residual_sequence[..8.min(residual_sequence.len())]),
            format_f32_preview(&input_norm_sequence[..8.min(input_norm_sequence.len())]),
            format_f32_preview(&output_sequence[..8.min(output_sequence.len())]),
            format_f32_preview(&block_output[..8.min(block_output.len())]),
        );
        return ExitCode::SUCCESS;
    }

    println!(
        "package-linear-attn-workflow-smoke package={} layer={} qkv_tensor=\"{}\" conv_tensor=\"{}\" a_tensor=\"{}\" b_tensor=\"{}\" a_log_tensor=\"{}\" dt_bias_tensor=\"{}\" z_tensor=\"{}\" norm_tensor=\"{}\" out_tensor=\"{}\" hidden={} key_heads={} value_heads={} key_dim={} value_dim={} sequence_len={} kernel_size={} qk_l2_norm={} q_scale={q_scale:.9} conv_dtype={} a_log_dtype={} dt_bias_dtype={} norm_dtype={} backend={} device_index={} name=\"{}\" recurrent_preview={} z_preview={} activated_preview={} output_preview={} conv_max_abs_diff={conv_max_abs_diff:.9} gate_beta_max_abs_diff={gate_beta_max_abs_diff:.9} recurrent_max_abs_diff={recurrent_max_abs_diff:.9} norm_max_abs_diff={norm_max_abs_diff:.9} activation_max_abs_diff={activation_max_abs_diff:.9} output_max_abs_diff={output_max_abs_diff:.9} verified=true",
        path,
        layer_index,
        qkv_tensor,
        conv_tensor,
        a_tensor,
        b_tensor,
        a_log_tensor,
        dt_bias_tensor,
        z_tensor,
        norm_tensor,
        out_tensor,
        hidden_size,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        sequence_len,
        kernel_size,
        qk_l2_norm,
        conv_dtype,
        a_log_dtype,
        dt_bias_dtype,
        norm_dtype,
        info.backend,
        device_index,
        info.name,
        format_f32_preview(&recurrent_output[..8.min(recurrent_output.len())]),
        format_f32_preview(&z_sequence[..8.min(z_sequence.len())]),
        format_f32_preview(&activated_sequence[..8.min(activated_sequence.len())]),
        format_f32_preview(&output_sequence[..8.min(output_sequence.len())]),
    );
    ExitCode::SUCCESS
}

fn package_linear_attn_mlp_block_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    sequence_len: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-linear-attn-mlp-block-smoke requires a .ullm.d path");
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
    let sequence_len = match parse_optional_usize(sequence_len, 1, "sequence length") {
        Ok(value) if value > 0 => value,
        Ok(_) => {
            eprintln!("sequence length must be greater than zero");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };

    let result = if sequence_len == 1 {
        package_linear_attn_mlp_block_smoke_impl(&path, device_index, chunk_bytes, layer_index)
    } else {
        package_linear_attn_mlp_block_sequence_smoke_impl(
            &path,
            device_index,
            chunk_bytes,
            layer_index,
            sequence_len,
        )
    };

    match result {
        Ok(line) => {
            println!("{line}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("{err}");
            ExitCode::from(1)
        }
    }
}

fn package_linear_attn_mlp_block_smoke_impl(
    path: &str,
    device_index: u32,
    chunk_bytes: usize,
    layer_index: usize,
) -> Result<String, String> {
    let key_heads = 16_usize;
    let value_heads = 32_usize;
    let key_dim = 128_usize;
    let value_dim = 128_usize;
    let hidden = value_heads * value_dim;
    let q_elements_per_step = key_heads * key_dim;
    let k_elements_per_step = key_heads * key_dim;
    let v_elements_per_step = hidden;
    let qkv_rows_expected = q_elements_per_step + k_elements_per_step + v_elements_per_step;
    let q_scale = 1.0_f32 / (key_dim as f32).sqrt();
    let qk_l2_norm = true;
    let sequence_len = 1_usize;
    let input_epsilon = 1e-6_f32;
    let mlp_epsilon = 1e-5_f32;

    let input_norm_tensor =
        format!("model.language_model.layers.{layer_index}.input_layernorm.weight");
    let qkv_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_qkv.weight");
    let conv_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.conv1d.weight");
    let a_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_a.weight");
    let b_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_b.weight");
    let a_log_tensor = format!("model.language_model.layers.{layer_index}.linear_attn.A_log");
    let dt_bias_tensor = format!("model.language_model.layers.{layer_index}.linear_attn.dt_bias");
    let z_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_z.weight");
    let norm_tensor = format!("model.language_model.layers.{layer_index}.linear_attn.norm.weight");
    let out_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.out_proj.weight");
    let post_norm_tensor =
        format!("model.language_model.layers.{layer_index}.post_attention_layernorm.weight");
    let gate_tensor = format!("model.language_model.layers.{layer_index}.mlp.gate_proj.weight");
    let up_tensor = format!("model.language_model.layers.{layer_index}.mlp.up_proj.weight");
    let down_tensor = format!("model.language_model.layers.{layer_index}.mlp.down_proj.weight");

    let input_norm = read_named_passthrough_f32(path, &input_norm_tensor, chunk_bytes)?;
    if input_norm.values.len() != hidden {
        return Err(format!(
            "input RMSNorm length must match hidden={hidden}: len={}",
            input_norm.values.len()
        ));
    }
    let conv = read_named_passthrough_f32(path, &conv_tensor, chunk_bytes)?;
    if conv.shape.len() != 3 || conv.shape[1] != 1 {
        return Err(format!(
            "conv1d tensor shape must be [channels,1,kernel], got {}",
            format_u64_shape(&conv.shape)
        ));
    }
    let conv_channels = usize::try_from(conv.shape[0])
        .map_err(|_| "conv1d channel count is too large for this host".to_string())?;
    let kernel_size = usize::try_from(conv.shape[2])
        .map_err(|_| "conv1d kernel size is too large for this host".to_string())?;
    if conv_channels != qkv_rows_expected {
        return Err(format!(
            "conv1d channels must match q/k/v layout: conv_channels={conv_channels}, expected={qkv_rows_expected}"
        ));
    }
    if conv.values.len() != conv_channels * kernel_size {
        return Err(format!(
            "conv1d weight element count mismatch: expected {} got {}",
            conv_channels * kernel_size,
            conv.values.len()
        ));
    }
    let a_log = read_named_passthrough_f32(path, &a_log_tensor, chunk_bytes)?;
    if a_log.values.len() != value_heads {
        return Err(format!(
            "A_log length must match value_heads={value_heads}: len={}",
            a_log.values.len()
        ));
    }
    let dt_bias = read_named_passthrough_f32(path, &dt_bias_tensor, chunk_bytes)?;
    if dt_bias.values.len() != value_heads {
        return Err(format!(
            "dt_bias length must match value_heads={value_heads}: len={}",
            dt_bias.values.len()
        ));
    }
    let attn_norm = read_named_passthrough_f32(path, &norm_tensor, chunk_bytes)?;
    if attn_norm.values.len() != value_dim {
        return Err(format!(
            "linear attention norm length must match value_dim={value_dim}: len={}",
            attn_norm.values.len()
        ));
    }
    let post_norm = read_named_passthrough_f32(path, &post_norm_tensor, chunk_bytes)?;
    if post_norm.values.len() != hidden {
        return Err(format!(
            "post RMSNorm length must match hidden={hidden}: len={}",
            post_norm.values.len()
        ));
    }

    let mut context = ullm_runtime_sys::RuntimeContext::create(device_index)
        .map_err(|err| format!("failed to create runtime context: {err}"))?;
    let info = context
        .device_info()
        .map_err(|err| format!("failed to query runtime context device: {err}"))?;
    let mut stream = context
        .create_stream()
        .map_err(|err| format!("failed to create runtime stream: {err}"))?;

    let hidden_bytes = hidden
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "hidden byte size overflows".to_string())?;
    let qkv_bytes = qkv_rows_expected
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "qkv byte size overflows".to_string())?;
    let gate_beta_bytes = value_heads
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "gate/beta byte size overflows".to_string())?;

    let residual = deterministic_f32_vector(hidden);
    let residual_bytes = encode_f32_to_bytes(&residual);
    let input_norm_weight_bytes = encode_f32_to_bytes(&input_norm.values);
    let conv_weight_bytes = encode_f32_to_bytes(&conv.values);
    let a_log_bytes = encode_f32_to_bytes(&a_log.values);
    let dt_bias_bytes = encode_f32_to_bytes(&dt_bias.values);
    let attn_norm_weight_bytes = encode_f32_to_bytes(&attn_norm.values);
    let post_norm_weight_bytes = encode_f32_to_bytes(&post_norm.values);

    let mut input_buffer = context
        .alloc_buffer(hidden_bytes)
        .map_err(|err| format!("failed to allocate input buffer: {err}"))?;
    let mut input_norm_weight_buffer = context
        .alloc_buffer(input_norm_weight_bytes.len())
        .map_err(|err| format!("failed to allocate input RMSNorm weight buffer: {err}"))?;
    let mut input_normed_buffer = context
        .alloc_buffer(hidden_bytes)
        .map_err(|err| format!("failed to allocate input RMSNorm output buffer: {err}"))?;
    input_buffer
        .copy_from_host(0, &residual_bytes, Some(&mut stream))
        .map_err(|err| format!("failed to copy residual input into runtime buffer: {err}"))?;
    input_norm_weight_buffer
        .copy_from_host(0, &input_norm_weight_bytes, Some(&mut stream))
        .map_err(|err| format!("failed to copy input RMSNorm weight into runtime buffer: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize after input copy: {err}"))?;
    ullm_runtime_sys::rmsnorm_f32(
        &input_buffer,
        &input_norm_weight_buffer,
        hidden,
        input_epsilon,
        &mut input_normed_buffer,
        Some(&mut stream),
    )
    .map_err(|err| format!("failed to run input RMSNorm: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize after input RMSNorm: {err}"))?;
    let mut input_normed_bytes = vec![0_u8; hidden_bytes];
    input_normed_buffer
        .copy_to_host(0, &mut input_normed_bytes, Some(&mut stream))
        .map_err(|err| format!("failed to copy input RMSNorm output to host: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize after input RMSNorm copy: {err}"))?;
    let input_normed = decode_f32_le_values(&input_normed_bytes);
    let expected_input_normed =
        runtime_host_rmsnorm_f32(&residual, &input_norm.values, input_epsilon);
    let input_norm_max_abs_diff = verify_f32_close(
        "package-linear-attn-mlp-block-smoke input RMSNorm",
        &input_normed,
        &expected_input_normed,
        1e-4,
        1e-5,
    )?;

    let (
        attention_block_output,
        attn_output,
        attn_block_max_abs_diff,
        conv_max_abs_diff,
        gate_beta_max_abs_diff,
        recurrent_max_abs_diff,
        attn_norm_max_abs_diff,
        attn_activation_max_abs_diff,
        attn_output_max_abs_diff,
    ) = {
        let mut registry = WeightRegistry::new();
        let (qkv_rows, qkv_cols, qkv_matrix) = materialize_selected_aq4_matrix(
            &mut context,
            &mut stream,
            &mut registry,
            path,
            &qkv_tensor,
            chunk_bytes,
        )?;
        let (a_rows, a_cols, a_matrix) = materialize_selected_aq4_matrix(
            &mut context,
            &mut stream,
            &mut registry,
            path,
            &a_tensor,
            chunk_bytes,
        )?;
        let (b_rows, b_cols, b_matrix) = materialize_selected_aq4_matrix(
            &mut context,
            &mut stream,
            &mut registry,
            path,
            &b_tensor,
            chunk_bytes,
        )?;
        let (z_rows, z_cols, z_matrix) = materialize_selected_aq4_matrix(
            &mut context,
            &mut stream,
            &mut registry,
            path,
            &z_tensor,
            chunk_bytes,
        )?;
        let (out_rows, out_cols, out_matrix) = materialize_selected_aq4_matrix(
            &mut context,
            &mut stream,
            &mut registry,
            path,
            &out_tensor,
            chunk_bytes,
        )?;
        if qkv_rows != qkv_rows_expected || qkv_cols != hidden {
            return Err(format!(
                "qkv shape must be [{qkv_rows_expected},{hidden}], got [{qkv_rows},{qkv_cols}]"
            ));
        }
        if a_rows != value_heads || b_rows != value_heads || a_cols != hidden || b_cols != hidden {
            return Err(format!(
                "a/b shape must be [{value_heads},{hidden}], got a=[{a_rows},{a_cols}] b=[{b_rows},{b_cols}]"
            ));
        }
        if z_rows != hidden || z_cols != hidden || out_rows != hidden || out_cols != hidden {
            return Err(format!(
                "z/out shape must be [{hidden},{hidden}], got z=[{z_rows},{z_cols}] out=[{out_rows},{out_cols}]"
            ));
        }

        let mut qkv_buffer = context
            .alloc_buffer(qkv_bytes)
            .map_err(|err| format!("failed to allocate qkv output buffer: {err}"))?;
        let mut a_buffer = context
            .alloc_buffer(gate_beta_bytes)
            .map_err(|err| format!("failed to allocate a output buffer: {err}"))?;
        let mut b_buffer = context
            .alloc_buffer(gate_beta_bytes)
            .map_err(|err| format!("failed to allocate b output buffer: {err}"))?;
        let mut z_buffer = context
            .alloc_buffer(hidden_bytes)
            .map_err(|err| format!("failed to allocate z output buffer: {err}"))?;
        ullm_runtime_sys::matvec_f32(
            &qkv_matrix,
            &input_normed_buffer,
            qkv_rows,
            qkv_cols,
            &mut qkv_buffer,
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to run qkv matvec: {err}"))?;
        ullm_runtime_sys::matvec_f32(
            &a_matrix,
            &input_normed_buffer,
            a_rows,
            a_cols,
            &mut a_buffer,
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to run a matvec: {err}"))?;
        ullm_runtime_sys::matvec_f32(
            &b_matrix,
            &input_normed_buffer,
            b_rows,
            b_cols,
            &mut b_buffer,
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to run b matvec: {err}"))?;
        ullm_runtime_sys::matvec_f32(
            &z_matrix,
            &input_normed_buffer,
            z_rows,
            z_cols,
            &mut z_buffer,
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to run z matvec: {err}"))?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize after linear attention projections: {err}")
        })?;

        let mut qkv_bytes_host = vec![0_u8; qkv_bytes];
        let mut a_bytes_host = vec![0_u8; gate_beta_bytes];
        let mut b_bytes_host = vec![0_u8; gate_beta_bytes];
        let mut z_bytes_host = vec![0_u8; hidden_bytes];
        qkv_buffer
            .copy_to_host(0, &mut qkv_bytes_host, Some(&mut stream))
            .map_err(|err| format!("failed to copy qkv output to host: {err}"))?;
        a_buffer
            .copy_to_host(0, &mut a_bytes_host, Some(&mut stream))
            .map_err(|err| format!("failed to copy a output to host: {err}"))?;
        b_buffer
            .copy_to_host(0, &mut b_bytes_host, Some(&mut stream))
            .map_err(|err| format!("failed to copy b output to host: {err}"))?;
        z_buffer
            .copy_to_host(0, &mut z_bytes_host, Some(&mut stream))
            .map_err(|err| format!("failed to copy z output to host: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after projection host copies: {err}"))?;
        let qkv_output = decode_f32_le_values(&qkv_bytes_host);
        let a_output = decode_f32_le_values(&a_bytes_host);
        let b_output = decode_f32_le_values(&b_bytes_host);
        let z_output = decode_f32_le_values(&z_bytes_host);

        let mut conv_weight_buffer = context
            .alloc_buffer(conv_weight_bytes.len())
            .map_err(|err| format!("failed to allocate conv1d weight buffer: {err}"))?;
        let mut conv_output_buffer = context
            .alloc_buffer(qkv_bytes)
            .map_err(|err| format!("failed to allocate conv1d output buffer: {err}"))?;
        conv_weight_buffer
            .copy_from_host(0, &conv_weight_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy conv1d weight into runtime buffer: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after conv1d weight copy: {err}"))?;
        ullm_runtime_sys::depthwise_conv1d_f32(
            &qkv_buffer,
            &conv_weight_buffer,
            qkv_rows,
            sequence_len,
            kernel_size,
            &mut conv_output_buffer,
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to run depthwise conv1d: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after conv1d: {err}"))?;
        let mut conv_output_bytes = vec![0_u8; qkv_bytes];
        conv_output_buffer
            .copy_to_host(0, &mut conv_output_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy conv1d output to host: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after conv1d host copy: {err}"))?;
        let conv_output = decode_f32_le_values(&conv_output_bytes);
        let expected_conv = runtime_host_depthwise_conv1d_f32(
            &qkv_output,
            &conv.values,
            qkv_rows,
            sequence_len,
            kernel_size,
        );
        let conv_max_abs_diff = verify_f32_close(
            "package-linear-attn-mlp-block-smoke conv1d",
            &conv_output,
            &expected_conv,
            1e-4,
            1e-5,
        )?;

        let mut a_log_buffer = context
            .alloc_buffer(a_log_bytes.len())
            .map_err(|err| format!("failed to allocate A_log buffer: {err}"))?;
        let mut dt_bias_buffer = context
            .alloc_buffer(dt_bias_bytes.len())
            .map_err(|err| format!("failed to allocate dt_bias buffer: {err}"))?;
        let mut gate_buffer = context
            .alloc_buffer(gate_beta_bytes)
            .map_err(|err| format!("failed to allocate gate output buffer: {err}"))?;
        let mut beta_buffer = context
            .alloc_buffer(gate_beta_bytes)
            .map_err(|err| format!("failed to allocate beta output buffer: {err}"))?;
        a_log_buffer
            .copy_from_host(0, &a_log_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy A_log into runtime buffer: {err}"))?;
        dt_bias_buffer
            .copy_from_host(0, &dt_bias_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy dt_bias into runtime buffer: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after gate/beta aux copy: {err}"))?;
        ullm_runtime_sys::linear_attn_gate_beta_f32(
            &a_buffer,
            &b_buffer,
            &a_log_buffer,
            &dt_bias_buffer,
            value_heads,
            sequence_len,
            &mut gate_buffer,
            &mut beta_buffer,
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to run linear attention gate/beta: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after gate/beta: {err}"))?;
        let mut gate_bytes = vec![0_u8; gate_beta_bytes];
        let mut beta_bytes = vec![0_u8; gate_beta_bytes];
        gate_buffer
            .copy_to_host(0, &mut gate_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy gate output to host: {err}"))?;
        beta_buffer
            .copy_to_host(0, &mut beta_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy beta output to host: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after gate/beta host copy: {err}"))?;
        let gate_output = decode_f32_le_values(&gate_bytes);
        let beta_output = decode_f32_le_values(&beta_bytes);
        let (expected_gate, expected_beta) = runtime_host_linear_attn_gate_beta_f32(
            &a_output,
            &b_output,
            &a_log.values,
            &dt_bias.values,
            value_heads,
            sequence_len,
        );
        let gate_max_abs_diff = verify_f32_close(
            "package-linear-attn-mlp-block-smoke gate",
            &gate_output,
            &expected_gate,
            1e-4,
            1e-5,
        )?;
        let beta_max_abs_diff = verify_f32_close(
            "package-linear-attn-mlp-block-smoke beta",
            &beta_output,
            &expected_beta,
            1e-4,
            1e-5,
        )?;
        let gate_beta_max_abs_diff = gate_max_abs_diff.max(beta_max_abs_diff);

        let qkv_split = split_linear_attn_qkv_for_recurrent(
            &conv_output,
            sequence_len,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            qk_l2_norm,
            q_scale,
        )
        .map_err(|err| format!("failed to split qkv for recurrent: {err}"))?;
        let state_elements = value_heads
            .checked_mul(key_dim)
            .and_then(|value| value.checked_mul(value_dim))
            .ok_or_else(|| "linear attention state element count overflows".to_string())?;
        let mut expected_state = vec![0.0_f32; state_elements];
        let expected_recurrent = runtime_host_linear_attn_recurrent_f32(
            &qkv_split.q,
            &qkv_split.k,
            &qkv_split.v,
            &expected_gate,
            &expected_beta,
            key_heads,
            value_heads,
            sequence_len,
            key_dim,
            value_dim,
            &mut expected_state,
        );
        let q_bytes = encode_f32_to_bytes(&qkv_split.q);
        let k_bytes = encode_f32_to_bytes(&qkv_split.k);
        let v_bytes = encode_f32_to_bytes(&qkv_split.v);
        let state_bytes = encode_f32_to_bytes(&vec![0.0_f32; state_elements]);
        let mut q_buffer = context
            .alloc_buffer(q_bytes.len())
            .map_err(|err| format!("failed to allocate q buffer: {err}"))?;
        let mut k_buffer = context
            .alloc_buffer(k_bytes.len())
            .map_err(|err| format!("failed to allocate k buffer: {err}"))?;
        let mut v_buffer = context
            .alloc_buffer(v_bytes.len())
            .map_err(|err| format!("failed to allocate v buffer: {err}"))?;
        let mut state_buffer = context
            .alloc_buffer(state_bytes.len())
            .map_err(|err| format!("failed to allocate recurrent state buffer: {err}"))?;
        let mut recurrent_buffer = context
            .alloc_buffer(hidden_bytes)
            .map_err(|err| format!("failed to allocate recurrent output buffer: {err}"))?;
        q_buffer
            .copy_from_host(0, &q_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy q into runtime buffer: {err}"))?;
        k_buffer
            .copy_from_host(0, &k_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy k into runtime buffer: {err}"))?;
        v_buffer
            .copy_from_host(0, &v_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy v into runtime buffer: {err}"))?;
        state_buffer
            .copy_from_host(0, &state_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy recurrent state into runtime buffer: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after recurrent input copy: {err}"))?;
        ullm_runtime_sys::linear_attn_recurrent_f32(
            &q_buffer,
            &k_buffer,
            &v_buffer,
            &gate_buffer,
            &beta_buffer,
            key_heads,
            value_heads,
            sequence_len,
            key_dim,
            value_dim,
            &mut state_buffer,
            &mut recurrent_buffer,
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to run linear attention recurrent: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after recurrent: {err}"))?;
        let mut recurrent_bytes = vec![0_u8; hidden_bytes];
        recurrent_buffer
            .copy_to_host(0, &mut recurrent_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy recurrent output to host: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after recurrent host copy: {err}"))?;
        let recurrent_output = decode_f32_le_values(&recurrent_bytes);
        let recurrent_max_abs_diff = verify_f32_close(
            "package-linear-attn-mlp-block-smoke recurrent",
            &recurrent_output,
            &expected_recurrent,
            1e-3,
            1e-5,
        )?;

        let mut expected_attn_normed = vec![0.0_f32; hidden];
        for value_head in 0..value_heads {
            let start = value_head * value_dim;
            let end = start + value_dim;
            let normed = runtime_host_rmsnorm_f32(
                &expected_recurrent[start..end],
                &attn_norm.values,
                input_epsilon,
            );
            expected_attn_normed[start..end].copy_from_slice(&normed);
        }
        let mut attn_norm_weight_buffer = context
            .alloc_buffer(attn_norm_weight_bytes.len())
            .map_err(|err| {
                format!("failed to allocate linear attention norm weight buffer: {err}")
            })?;
        let mut attn_norm_input_buffer = context
            .alloc_buffer(attn_norm_weight_bytes.len())
            .map_err(|err| {
                format!("failed to allocate linear attention norm input buffer: {err}")
            })?;
        let mut attn_norm_output_buffer = context
            .alloc_buffer(attn_norm_weight_bytes.len())
            .map_err(|err| {
                format!("failed to allocate linear attention norm output buffer: {err}")
            })?;
        attn_norm_weight_buffer
            .copy_from_host(0, &attn_norm_weight_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy linear attention norm weight: {err}"))?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize after linear attention norm weight copy: {err}")
        })?;
        let mut attn_normed_bytes = vec![0_u8; hidden_bytes];
        for value_head in 0..value_heads {
            let start = value_head * value_dim;
            let byte_start = start * std::mem::size_of::<f32>();
            let byte_end = byte_start + attn_norm_weight_bytes.len();
            attn_norm_input_buffer
                .copy_from_host(0, &recurrent_bytes[byte_start..byte_end], Some(&mut stream))
                .map_err(|err| {
                    format!("failed to copy linear attention norm row {value_head}: {err}")
                })?;
            ullm_runtime_sys::rmsnorm_f32(
                &attn_norm_input_buffer,
                &attn_norm_weight_buffer,
                value_dim,
                input_epsilon,
                &mut attn_norm_output_buffer,
                Some(&mut stream),
            )
            .map_err(|err| {
                format!("failed to run linear attention norm row {value_head}: {err}")
            })?;
            stream.synchronize().map_err(|err| {
                format!("failed to synchronize after linear attention norm row {value_head}: {err}")
            })?;
            attn_norm_output_buffer
                .copy_to_host(
                    0,
                    &mut attn_normed_bytes[byte_start..byte_end],
                    Some(&mut stream),
                )
                .map_err(|err| {
                    format!("failed to copy linear attention norm row {value_head}: {err}")
                })?;
            stream.synchronize().map_err(|err| {
                format!(
                    "failed to synchronize after linear attention norm row copy {value_head}: {err}"
                )
            })?;
        }
        let attn_normed = decode_f32_le_values(&attn_normed_bytes);
        let attn_norm_max_abs_diff = verify_f32_close(
            "package-linear-attn-mlp-block-smoke linear attention norm",
            &attn_normed,
            &expected_attn_normed,
            1e-3,
            1e-5,
        )?;
        let expected_attn_activated = runtime_host_silu_mul_f32(&z_output, &expected_attn_normed);
        let mut attn_normed_buffer = context
            .alloc_buffer(hidden_bytes)
            .map_err(|err| format!("failed to allocate linear attention normed buffer: {err}"))?;
        let mut attn_activated_buffer = context.alloc_buffer(hidden_bytes).map_err(|err| {
            format!("failed to allocate linear attention activated buffer: {err}")
        })?;
        attn_normed_buffer
            .copy_from_host(0, &attn_normed_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy linear attention normed values: {err}"))?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize after linear attention normed copy: {err}")
        })?;
        ullm_runtime_sys::silu_mul_f32(
            &z_buffer,
            &attn_normed_buffer,
            hidden,
            &mut attn_activated_buffer,
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to run linear attention SiLU-mul: {err}"))?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize after linear attention SiLU-mul: {err}")
        })?;
        let mut attn_activated_bytes = vec![0_u8; hidden_bytes];
        attn_activated_buffer
            .copy_to_host(0, &mut attn_activated_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy linear attention activated values: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after activated host copy: {err}"))?;
        let attn_activated = decode_f32_le_values(&attn_activated_bytes);
        let attn_activation_max_abs_diff = verify_f32_close(
            "package-linear-attn-mlp-block-smoke linear attention activation",
            &attn_activated,
            &expected_attn_activated,
            1e-3,
            1e-5,
        )?;

        let out_matrix_bytes_len = out_rows
            .checked_mul(out_cols)
            .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| "out projection matrix byte size overflows".to_string())?;
        let mut out_matrix_bytes = vec![0_u8; out_matrix_bytes_len];
        out_matrix
            .copy_to_host(0, &mut out_matrix_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy out projection matrix to host: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after out matrix copy: {err}"))?;
        let out_matrix_host = decode_f32_le_values(&out_matrix_bytes);
        let expected_attn_output = runtime_host_matvec_f32(
            &out_matrix_host,
            &expected_attn_activated,
            out_rows,
            out_cols,
        );
        let mut attn_output_buffer = context
            .alloc_buffer(hidden_bytes)
            .map_err(|err| format!("failed to allocate linear attention output buffer: {err}"))?;
        ullm_runtime_sys::matvec_f32(
            &out_matrix,
            &attn_activated_buffer,
            out_rows,
            out_cols,
            &mut attn_output_buffer,
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to run out projection matvec: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after out projection: {err}"))?;
        let mut attn_output_bytes = vec![0_u8; hidden_bytes];
        attn_output_buffer
            .copy_to_host(0, &mut attn_output_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy linear attention output to host: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after attention output copy: {err}"))?;
        let attn_output = decode_f32_le_values(&attn_output_bytes);
        let attn_output_max_abs_diff = verify_f32_close(
            "package-linear-attn-mlp-block-smoke out projection",
            &attn_output,
            &expected_attn_output,
            3e-3,
            2e-5,
        )?;

        let mut attn_block_buffer = context
            .alloc_buffer(hidden_bytes)
            .map_err(|err| format!("failed to allocate attention block output buffer: {err}"))?;
        ullm_runtime_sys::add_f32(
            &input_buffer,
            &attn_output_buffer,
            hidden,
            &mut attn_block_buffer,
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to run attention residual add: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after attention residual add: {err}"))?;
        let mut attn_block_bytes = vec![0_u8; hidden_bytes];
        attn_block_buffer
            .copy_to_host(0, &mut attn_block_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy attention block output to host: {err}"))?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize after attention block host copy: {err}")
        })?;
        let attention_block_output = decode_f32_le_values(&attn_block_bytes);
        let expected_attention_block = runtime_host_add_f32(&residual, &attn_output);
        let attn_block_max_abs_diff = verify_f32_close(
            "package-linear-attn-mlp-block-smoke attention residual",
            &attention_block_output,
            &expected_attention_block,
            1e-5,
            1e-6,
        )?;
        (
            attention_block_output,
            attn_output,
            attn_block_max_abs_diff,
            conv_max_abs_diff,
            gate_beta_max_abs_diff,
            recurrent_max_abs_diff,
            attn_norm_max_abs_diff,
            attn_activation_max_abs_diff,
            attn_output_max_abs_diff,
        )
    };

    let post_normed_expected =
        runtime_host_rmsnorm_f32(&attention_block_output, &post_norm.values, mlp_epsilon);
    let mut attn_block_buffer = context
        .alloc_buffer(hidden_bytes)
        .map_err(|err| format!("failed to allocate retained attention block buffer: {err}"))?;
    let mut post_norm_weight_buffer = context
        .alloc_buffer(post_norm_weight_bytes.len())
        .map_err(|err| format!("failed to allocate post RMSNorm weight buffer: {err}"))?;
    let mut post_normed_buffer = context
        .alloc_buffer(hidden_bytes)
        .map_err(|err| format!("failed to allocate post RMSNorm output buffer: {err}"))?;
    let attention_block_bytes = encode_f32_to_bytes(&attention_block_output);
    attn_block_buffer
        .copy_from_host(0, &attention_block_bytes, Some(&mut stream))
        .map_err(|err| {
            format!("failed to copy attention block output into runtime buffer: {err}")
        })?;
    post_norm_weight_buffer
        .copy_from_host(0, &post_norm_weight_bytes, Some(&mut stream))
        .map_err(|err| format!("failed to copy post RMSNorm weight into runtime buffer: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize after post RMSNorm input copy: {err}"))?;
    ullm_runtime_sys::rmsnorm_f32(
        &attn_block_buffer,
        &post_norm_weight_buffer,
        hidden,
        mlp_epsilon,
        &mut post_normed_buffer,
        Some(&mut stream),
    )
    .map_err(|err| format!("failed to run post RMSNorm: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize after post RMSNorm: {err}"))?;
    let mut post_normed_bytes = vec![0_u8; hidden_bytes];
    post_normed_buffer
        .copy_to_host(0, &mut post_normed_bytes, Some(&mut stream))
        .map_err(|err| format!("failed to copy post RMSNorm output to host: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize after post RMSNorm host copy: {err}"))?;
    let post_normed = decode_f32_le_values(&post_normed_bytes);
    let post_norm_max_abs_diff = verify_f32_close(
        "package-linear-attn-mlp-block-smoke post RMSNorm",
        &post_normed,
        &post_normed_expected,
        1e-4,
        1e-5,
    )?;

    let (mlp_output, layer_output, layer_block_max_abs_diff) = {
        let mut registry = WeightRegistry::new();
        let (gate_rows, gate_cols, gate_matrix) = materialize_selected_aq4_matrix(
            &mut context,
            &mut stream,
            &mut registry,
            path,
            &gate_tensor,
            chunk_bytes,
        )?;
        let (up_rows, up_cols, up_matrix) = materialize_selected_aq4_matrix(
            &mut context,
            &mut stream,
            &mut registry,
            path,
            &up_tensor,
            chunk_bytes,
        )?;
        let (down_rows, down_cols, down_matrix) = materialize_selected_aq4_matrix(
            &mut context,
            &mut stream,
            &mut registry,
            path,
            &down_tensor,
            chunk_bytes,
        )?;
        if gate_rows != up_rows || gate_cols != up_cols || gate_cols != hidden {
            return Err(format!(
                "MLP gate/up shape mismatch: gate=[{gate_rows},{gate_cols}] up=[{up_rows},{up_cols}] hidden={hidden}"
            ));
        }
        if down_rows != hidden || down_cols != gate_rows {
            return Err(format!(
                "MLP down shape mismatch: expected [{hidden},{gate_rows}], got [{down_rows},{down_cols}]"
            ));
        }
        let intermediate = gate_rows;
        let intermediate_bytes = intermediate
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "MLP intermediate byte size overflows".to_string())?;
        let mut gate_buffer = context
            .alloc_buffer(intermediate_bytes)
            .map_err(|err| format!("failed to allocate MLP gate buffer: {err}"))?;
        let mut up_buffer = context
            .alloc_buffer(intermediate_bytes)
            .map_err(|err| format!("failed to allocate MLP up buffer: {err}"))?;
        let mut mlp_activated_buffer = context
            .alloc_buffer(intermediate_bytes)
            .map_err(|err| format!("failed to allocate MLP activated buffer: {err}"))?;
        let mut mlp_output_buffer = context
            .alloc_buffer(hidden_bytes)
            .map_err(|err| format!("failed to allocate MLP output buffer: {err}"))?;
        ullm_runtime_sys::matvec_f32(
            &gate_matrix,
            &post_normed_buffer,
            gate_rows,
            gate_cols,
            &mut gate_buffer,
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to run MLP gate matvec: {err}"))?;
        ullm_runtime_sys::matvec_f32(
            &up_matrix,
            &post_normed_buffer,
            up_rows,
            up_cols,
            &mut up_buffer,
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to run MLP up matvec: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after MLP gate/up: {err}"))?;
        ullm_runtime_sys::silu_mul_f32(
            &gate_buffer,
            &up_buffer,
            intermediate,
            &mut mlp_activated_buffer,
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to run MLP SiLU-mul: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after MLP SiLU-mul: {err}"))?;
        ullm_runtime_sys::matvec_f32(
            &down_matrix,
            &mlp_activated_buffer,
            down_rows,
            down_cols,
            &mut mlp_output_buffer,
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to run MLP down matvec: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after MLP down: {err}"))?;
        let mut mlp_output_bytes = vec![0_u8; hidden_bytes];
        mlp_output_buffer
            .copy_to_host(0, &mut mlp_output_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy MLP output to host: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after MLP output copy: {err}"))?;
        let mlp_output = decode_f32_le_values(&mlp_output_bytes);

        let mut layer_output_buffer = context
            .alloc_buffer(hidden_bytes)
            .map_err(|err| format!("failed to allocate layer output buffer: {err}"))?;
        ullm_runtime_sys::add_f32(
            &attn_block_buffer,
            &mlp_output_buffer,
            hidden,
            &mut layer_output_buffer,
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to run MLP residual add: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after MLP residual add: {err}"))?;
        let mut layer_output_bytes = vec![0_u8; hidden_bytes];
        layer_output_buffer
            .copy_to_host(0, &mut layer_output_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy layer output to host: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after layer output copy: {err}"))?;
        let layer_output = decode_f32_le_values(&layer_output_bytes);
        let expected_layer_output = runtime_host_add_f32(&attention_block_output, &mlp_output);
        let layer_block_max_abs_diff = verify_f32_close(
            "package-linear-attn-mlp-block-smoke layer residual",
            &layer_output,
            &expected_layer_output,
            1e-5,
            1e-6,
        )?;
        (mlp_output, layer_output, layer_block_max_abs_diff)
    };

    Ok(format!(
        "package-linear-attn-mlp-block-smoke package={} layer={} input_norm_tensor=\"{}\" qkv_tensor=\"{}\" conv_tensor=\"{}\" a_tensor=\"{}\" b_tensor=\"{}\" a_log_tensor=\"{}\" dt_bias_tensor=\"{}\" z_tensor=\"{}\" norm_tensor=\"{}\" out_tensor=\"{}\" post_norm_tensor=\"{}\" gate_tensor=\"{}\" up_tensor=\"{}\" down_tensor=\"{}\" hidden={} key_heads={} value_heads={} key_dim={} value_dim={} sequence_len={} kernel_size={} qk_l2_norm={} q_scale={q_scale:.9} input_norm_dtype={} conv_dtype={} a_log_dtype={} dt_bias_dtype={} norm_dtype={} post_norm_dtype={} backend={} device_index={} name=\"{}\" residual_preview={} attention_output_preview={} attention_block_preview={} post_norm_preview={} mlp_output_preview={} layer_output_preview={} input_norm_max_abs_diff={input_norm_max_abs_diff:.9} conv_max_abs_diff={conv_max_abs_diff:.9} gate_beta_max_abs_diff={gate_beta_max_abs_diff:.9} recurrent_max_abs_diff={recurrent_max_abs_diff:.9} attn_norm_max_abs_diff={attn_norm_max_abs_diff:.9} attn_activation_max_abs_diff={attn_activation_max_abs_diff:.9} attn_output_max_abs_diff={attn_output_max_abs_diff:.9} attn_block_max_abs_diff={attn_block_max_abs_diff:.9} post_norm_max_abs_diff={post_norm_max_abs_diff:.9} layer_block_max_abs_diff={layer_block_max_abs_diff:.9} verified=true",
        path,
        layer_index,
        input_norm_tensor,
        qkv_tensor,
        conv_tensor,
        a_tensor,
        b_tensor,
        a_log_tensor,
        dt_bias_tensor,
        z_tensor,
        norm_tensor,
        out_tensor,
        post_norm_tensor,
        gate_tensor,
        up_tensor,
        down_tensor,
        hidden,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        sequence_len,
        kernel_size,
        qk_l2_norm,
        input_norm.dtype,
        conv.dtype,
        a_log.dtype,
        dt_bias.dtype,
        attn_norm.dtype,
        post_norm.dtype,
        info.backend,
        device_index,
        info.name,
        format_f32_preview(&residual[..8.min(residual.len())]),
        format_f32_preview(&attn_output[..8.min(attn_output.len())]),
        format_f32_preview(&attention_block_output[..8.min(attention_block_output.len())]),
        format_f32_preview(&post_normed[..8.min(post_normed.len())]),
        format_f32_preview(&mlp_output[..8.min(mlp_output.len())]),
        format_f32_preview(&layer_output[..8.min(layer_output.len())]),
    ))
}

fn package_linear_attn_mlp_block_sequence_smoke_impl(
    path: &str,
    device_index: u32,
    chunk_bytes: usize,
    layer_index: usize,
    sequence_len: usize,
) -> Result<String, String> {
    let key_heads = 16_usize;
    let value_heads = 32_usize;
    let key_dim = 128_usize;
    let value_dim = 128_usize;
    let hidden = value_heads * value_dim;
    let q_elements_per_step = key_heads * key_dim;
    let k_elements_per_step = key_heads * key_dim;
    let v_elements_per_step = hidden;
    let qkv_rows_expected = q_elements_per_step + k_elements_per_step + v_elements_per_step;
    let q_scale = 1.0_f32 / (key_dim as f32).sqrt();
    let qk_l2_norm = true;
    let input_epsilon = 1e-6_f32;
    let mlp_epsilon = 1e-5_f32;

    let input_norm_tensor =
        format!("model.language_model.layers.{layer_index}.input_layernorm.weight");
    let qkv_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_qkv.weight");
    let conv_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.conv1d.weight");
    let a_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_a.weight");
    let b_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_b.weight");
    let a_log_tensor = format!("model.language_model.layers.{layer_index}.linear_attn.A_log");
    let dt_bias_tensor = format!("model.language_model.layers.{layer_index}.linear_attn.dt_bias");
    let z_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_z.weight");
    let norm_tensor = format!("model.language_model.layers.{layer_index}.linear_attn.norm.weight");
    let out_tensor =
        format!("model.language_model.layers.{layer_index}.linear_attn.out_proj.weight");
    let post_norm_tensor =
        format!("model.language_model.layers.{layer_index}.post_attention_layernorm.weight");
    let gate_tensor = format!("model.language_model.layers.{layer_index}.mlp.gate_proj.weight");
    let up_tensor = format!("model.language_model.layers.{layer_index}.mlp.up_proj.weight");
    let down_tensor = format!("model.language_model.layers.{layer_index}.mlp.down_proj.weight");

    let input_norm = read_named_passthrough_f32(path, &input_norm_tensor, chunk_bytes)?;
    if input_norm.values.len() != hidden {
        return Err(format!(
            "input RMSNorm length must match hidden={hidden}: len={}",
            input_norm.values.len()
        ));
    }
    let conv = read_named_passthrough_f32(path, &conv_tensor, chunk_bytes)?;
    if conv.shape.len() != 3 || conv.shape[1] != 1 {
        return Err(format!(
            "conv1d tensor shape must be [channels,1,kernel], got {}",
            format_u64_shape(&conv.shape)
        ));
    }
    let conv_channels = usize::try_from(conv.shape[0])
        .map_err(|_| "conv1d channel count is too large for this host".to_string())?;
    let kernel_size = usize::try_from(conv.shape[2])
        .map_err(|_| "conv1d kernel size is too large for this host".to_string())?;
    if conv_channels != qkv_rows_expected {
        return Err(format!(
            "conv1d channels must match q/k/v layout: conv_channels={conv_channels}, expected={qkv_rows_expected}"
        ));
    }
    if conv.values.len() != conv_channels * kernel_size {
        return Err(format!(
            "conv1d weight element count mismatch: expected {} got {}",
            conv_channels * kernel_size,
            conv.values.len()
        ));
    }
    let a_log = read_named_passthrough_f32(path, &a_log_tensor, chunk_bytes)?;
    if a_log.values.len() != value_heads {
        return Err(format!(
            "A_log length must match value_heads={value_heads}: len={}",
            a_log.values.len()
        ));
    }
    let dt_bias = read_named_passthrough_f32(path, &dt_bias_tensor, chunk_bytes)?;
    if dt_bias.values.len() != value_heads {
        return Err(format!(
            "dt_bias length must match value_heads={value_heads}: len={}",
            dt_bias.values.len()
        ));
    }
    let attn_norm = read_named_passthrough_f32(path, &norm_tensor, chunk_bytes)?;
    if attn_norm.values.len() != value_dim {
        return Err(format!(
            "linear attention norm length must match value_dim={value_dim}: len={}",
            attn_norm.values.len()
        ));
    }
    let post_norm = read_named_passthrough_f32(path, &post_norm_tensor, chunk_bytes)?;
    if post_norm.values.len() != hidden {
        return Err(format!(
            "post RMSNorm length must match hidden={hidden}: len={}",
            post_norm.values.len()
        ));
    }

    let mut context = ullm_runtime_sys::RuntimeContext::create(device_index)
        .map_err(|err| format!("failed to create runtime context: {err}"))?;
    let info = context
        .device_info()
        .map_err(|err| format!("failed to query runtime context device: {err}"))?;
    let mut stream = context
        .create_stream()
        .map_err(|err| format!("failed to create runtime stream: {err}"))?;

    let hidden_bytes = hidden
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "hidden byte size overflows".to_string())?;
    let hidden_sequence_bytes = hidden_bytes
        .checked_mul(sequence_len)
        .ok_or_else(|| "hidden sequence byte size overflows".to_string())?;
    let qkv_step_bytes = qkv_rows_expected
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "qkv step byte size overflows".to_string())?;
    let qkv_sequence_bytes = qkv_step_bytes
        .checked_mul(sequence_len)
        .ok_or_else(|| "qkv sequence byte size overflows".to_string())?;
    let gate_beta_step_bytes = value_heads
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "gate/beta step byte size overflows".to_string())?;
    let gate_beta_sequence_bytes = gate_beta_step_bytes
        .checked_mul(sequence_len)
        .ok_or_else(|| "gate/beta sequence byte size overflows".to_string())?;

    let base_residual = deterministic_f32_vector(hidden);
    let input_norm_weight_bytes = encode_f32_to_bytes(&input_norm.values);
    let conv_weight_bytes = encode_f32_to_bytes(&conv.values);
    let a_log_bytes = encode_f32_to_bytes(&a_log.values);
    let dt_bias_bytes = encode_f32_to_bytes(&dt_bias.values);
    let attn_norm_weight_bytes = encode_f32_to_bytes(&attn_norm.values);
    let post_norm_weight_bytes = encode_f32_to_bytes(&post_norm.values);

    let mut input_buffer = context
        .alloc_buffer(hidden_bytes)
        .map_err(|err| format!("failed to allocate input buffer: {err}"))?;
    let mut input_norm_weight_buffer = context
        .alloc_buffer(input_norm_weight_bytes.len())
        .map_err(|err| format!("failed to allocate input RMSNorm weight buffer: {err}"))?;
    let mut input_normed_buffer = context
        .alloc_buffer(hidden_bytes)
        .map_err(|err| format!("failed to allocate input RMSNorm output buffer: {err}"))?;
    input_norm_weight_buffer
        .copy_from_host(0, &input_norm_weight_bytes, Some(&mut stream))
        .map_err(|err| format!("failed to copy input RMSNorm weight into runtime buffer: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize after input norm weight copy: {err}"))?;

    let mut residual_sequence = Vec::with_capacity(sequence_len * hidden);
    let mut expected_input_normed = Vec::with_capacity(sequence_len * hidden);
    let mut input_normed_sequence_bytes = vec![0_u8; hidden_sequence_bytes];
    for timestep in 0..sequence_len {
        let residual = linear_attn_step_input(&base_residual, timestep);
        let residual_bytes = encode_f32_to_bytes(&residual);
        input_buffer
            .copy_from_host(0, &residual_bytes, Some(&mut stream))
            .map_err(|err| {
                format!("failed to copy residual timestep {timestep} into runtime buffer: {err}")
            })?;
        ullm_runtime_sys::rmsnorm_f32(
            &input_buffer,
            &input_norm_weight_buffer,
            hidden,
            input_epsilon,
            &mut input_normed_buffer,
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to run input RMSNorm timestep {timestep}: {err}"))?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize after input RMSNorm timestep {timestep}: {err}")
        })?;
        let byte_start = timestep * hidden_bytes;
        let byte_end = byte_start + hidden_bytes;
        input_normed_buffer
            .copy_to_host(
                0,
                &mut input_normed_sequence_bytes[byte_start..byte_end],
                Some(&mut stream),
            )
            .map_err(|err| {
                format!("failed to copy input RMSNorm timestep {timestep} to host: {err}")
            })?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize after input RMSNorm host copy {timestep}: {err}")
        })?;
        let expected = runtime_host_rmsnorm_f32(&residual, &input_norm.values, input_epsilon);
        residual_sequence.extend_from_slice(&residual);
        expected_input_normed.extend_from_slice(&expected);
    }
    let input_normed = decode_f32_le_values(&input_normed_sequence_bytes);
    let input_norm_max_abs_diff = verify_f32_close(
        "package-linear-attn-mlp-block-smoke input RMSNorm",
        &input_normed,
        &expected_input_normed,
        1e-4,
        1e-5,
    )?;

    let (
        attention_block_output,
        attn_output,
        attn_block_max_abs_diff,
        conv_max_abs_diff,
        gate_beta_max_abs_diff,
        recurrent_max_abs_diff,
        attn_norm_max_abs_diff,
        attn_activation_max_abs_diff,
        attn_output_max_abs_diff,
    ) = {
        let mut registry = WeightRegistry::new();
        let (qkv_rows, qkv_cols, qkv_matrix) = materialize_selected_aq4_matrix(
            &mut context,
            &mut stream,
            &mut registry,
            path,
            &qkv_tensor,
            chunk_bytes,
        )?;
        let (a_rows, a_cols, a_matrix) = materialize_selected_aq4_matrix(
            &mut context,
            &mut stream,
            &mut registry,
            path,
            &a_tensor,
            chunk_bytes,
        )?;
        let (b_rows, b_cols, b_matrix) = materialize_selected_aq4_matrix(
            &mut context,
            &mut stream,
            &mut registry,
            path,
            &b_tensor,
            chunk_bytes,
        )?;
        let (z_rows, z_cols, z_matrix) = materialize_selected_aq4_matrix(
            &mut context,
            &mut stream,
            &mut registry,
            path,
            &z_tensor,
            chunk_bytes,
        )?;
        let (out_rows, out_cols, out_matrix) = materialize_selected_aq4_matrix(
            &mut context,
            &mut stream,
            &mut registry,
            path,
            &out_tensor,
            chunk_bytes,
        )?;
        if qkv_rows != qkv_rows_expected || qkv_cols != hidden {
            return Err(format!(
                "qkv shape must be [{qkv_rows_expected},{hidden}], got [{qkv_rows},{qkv_cols}]"
            ));
        }
        if a_rows != value_heads || b_rows != value_heads || a_cols != hidden || b_cols != hidden {
            return Err(format!(
                "a/b shape must be [{value_heads},{hidden}], got a=[{a_rows},{a_cols}] b=[{b_rows},{b_cols}]"
            ));
        }
        if z_rows != hidden || z_cols != hidden || out_rows != hidden || out_cols != hidden {
            return Err(format!(
                "z/out shape must be [{hidden},{hidden}], got z=[{z_rows},{z_cols}] out=[{out_rows},{out_cols}]"
            ));
        }

        let mut qkv_step_buffer = context
            .alloc_buffer(qkv_step_bytes)
            .map_err(|err| format!("failed to allocate qkv step buffer: {err}"))?;
        let mut a_step_buffer = context
            .alloc_buffer(gate_beta_step_bytes)
            .map_err(|err| format!("failed to allocate a step buffer: {err}"))?;
        let mut b_step_buffer = context
            .alloc_buffer(gate_beta_step_bytes)
            .map_err(|err| format!("failed to allocate b step buffer: {err}"))?;
        let mut z_step_buffer = context
            .alloc_buffer(hidden_bytes)
            .map_err(|err| format!("failed to allocate z step buffer: {err}"))?;
        let mut qkv_sequence_bytes_host = vec![0_u8; qkv_sequence_bytes];
        let mut a_sequence_bytes = vec![0_u8; gate_beta_sequence_bytes];
        let mut b_sequence_bytes = vec![0_u8; gate_beta_sequence_bytes];
        let mut z_sequence_bytes = vec![0_u8; hidden_sequence_bytes];
        for timestep in 0..sequence_len {
            let hidden_start = timestep * hidden_bytes;
            let hidden_end = hidden_start + hidden_bytes;
            input_normed_buffer
                .copy_from_host(
                    0,
                    &input_normed_sequence_bytes[hidden_start..hidden_end],
                    Some(&mut stream),
                )
                .map_err(|err| format!("failed to copy input normed timestep {timestep}: {err}"))?;
            ullm_runtime_sys::matvec_f32(
                &qkv_matrix,
                &input_normed_buffer,
                qkv_rows,
                qkv_cols,
                &mut qkv_step_buffer,
                Some(&mut stream),
            )
            .map_err(|err| format!("failed to run qkv matvec timestep {timestep}: {err}"))?;
            ullm_runtime_sys::matvec_f32(
                &a_matrix,
                &input_normed_buffer,
                a_rows,
                a_cols,
                &mut a_step_buffer,
                Some(&mut stream),
            )
            .map_err(|err| format!("failed to run a matvec timestep {timestep}: {err}"))?;
            ullm_runtime_sys::matvec_f32(
                &b_matrix,
                &input_normed_buffer,
                b_rows,
                b_cols,
                &mut b_step_buffer,
                Some(&mut stream),
            )
            .map_err(|err| format!("failed to run b matvec timestep {timestep}: {err}"))?;
            ullm_runtime_sys::matvec_f32(
                &z_matrix,
                &input_normed_buffer,
                z_rows,
                z_cols,
                &mut z_step_buffer,
                Some(&mut stream),
            )
            .map_err(|err| format!("failed to run z matvec timestep {timestep}: {err}"))?;
            stream.synchronize().map_err(|err| {
                format!("failed to synchronize after projections timestep {timestep}: {err}")
            })?;
            let qkv_start = timestep * qkv_step_bytes;
            let qkv_end = qkv_start + qkv_step_bytes;
            qkv_step_buffer
                .copy_to_host(
                    0,
                    &mut qkv_sequence_bytes_host[qkv_start..qkv_end],
                    Some(&mut stream),
                )
                .map_err(|err| format!("failed to copy qkv timestep {timestep}: {err}"))?;
            let gate_start = timestep * gate_beta_step_bytes;
            let gate_end = gate_start + gate_beta_step_bytes;
            a_step_buffer
                .copy_to_host(
                    0,
                    &mut a_sequence_bytes[gate_start..gate_end],
                    Some(&mut stream),
                )
                .map_err(|err| format!("failed to copy a timestep {timestep}: {err}"))?;
            b_step_buffer
                .copy_to_host(
                    0,
                    &mut b_sequence_bytes[gate_start..gate_end],
                    Some(&mut stream),
                )
                .map_err(|err| format!("failed to copy b timestep {timestep}: {err}"))?;
            z_step_buffer
                .copy_to_host(
                    0,
                    &mut z_sequence_bytes[hidden_start..hidden_end],
                    Some(&mut stream),
                )
                .map_err(|err| format!("failed to copy z timestep {timestep}: {err}"))?;
            stream.synchronize().map_err(|err| {
                format!("failed to synchronize after projection copies timestep {timestep}: {err}")
            })?;
        }

        let qkv_output = decode_f32_le_values(&qkv_sequence_bytes_host);
        let a_output = decode_f32_le_values(&a_sequence_bytes);
        let b_output = decode_f32_le_values(&b_sequence_bytes);
        let z_output = decode_f32_le_values(&z_sequence_bytes);
        let mut qkv_sequence_buffer = context
            .alloc_buffer(qkv_sequence_bytes)
            .map_err(|err| format!("failed to allocate qkv sequence buffer: {err}"))?;
        let mut conv_weight_buffer = context
            .alloc_buffer(conv_weight_bytes.len())
            .map_err(|err| format!("failed to allocate conv1d weight buffer: {err}"))?;
        let mut conv_output_buffer = context
            .alloc_buffer(qkv_sequence_bytes)
            .map_err(|err| format!("failed to allocate conv1d output buffer: {err}"))?;
        qkv_sequence_buffer
            .copy_from_host(0, &qkv_sequence_bytes_host, Some(&mut stream))
            .map_err(|err| format!("failed to copy qkv sequence into runtime buffer: {err}"))?;
        conv_weight_buffer
            .copy_from_host(0, &conv_weight_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy conv1d weight into runtime buffer: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after conv1d input copy: {err}"))?;
        ullm_runtime_sys::depthwise_conv1d_f32(
            &qkv_sequence_buffer,
            &conv_weight_buffer,
            qkv_rows,
            sequence_len,
            kernel_size,
            &mut conv_output_buffer,
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to run depthwise conv1d: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after conv1d: {err}"))?;
        let mut conv_output_bytes = vec![0_u8; qkv_sequence_bytes];
        conv_output_buffer
            .copy_to_host(0, &mut conv_output_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy conv1d output to host: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after conv1d host copy: {err}"))?;
        let conv_output = decode_f32_le_values(&conv_output_bytes);
        let expected_conv = runtime_host_depthwise_conv1d_f32(
            &qkv_output,
            &conv.values,
            qkv_rows,
            sequence_len,
            kernel_size,
        );
        let conv_max_abs_diff = verify_f32_close(
            "package-linear-attn-mlp-block-smoke conv1d",
            &conv_output,
            &expected_conv,
            1e-4,
            1e-5,
        )?;

        let mut a_sequence_buffer = context
            .alloc_buffer(gate_beta_sequence_bytes)
            .map_err(|err| format!("failed to allocate a sequence buffer: {err}"))?;
        let mut b_sequence_buffer = context
            .alloc_buffer(gate_beta_sequence_bytes)
            .map_err(|err| format!("failed to allocate b sequence buffer: {err}"))?;
        let mut a_log_buffer = context
            .alloc_buffer(a_log_bytes.len())
            .map_err(|err| format!("failed to allocate A_log buffer: {err}"))?;
        let mut dt_bias_buffer = context
            .alloc_buffer(dt_bias_bytes.len())
            .map_err(|err| format!("failed to allocate dt_bias buffer: {err}"))?;
        let mut gate_buffer = context
            .alloc_buffer(gate_beta_sequence_bytes)
            .map_err(|err| format!("failed to allocate gate output buffer: {err}"))?;
        let mut beta_buffer = context
            .alloc_buffer(gate_beta_sequence_bytes)
            .map_err(|err| format!("failed to allocate beta output buffer: {err}"))?;
        a_sequence_buffer
            .copy_from_host(0, &a_sequence_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy a sequence into runtime buffer: {err}"))?;
        b_sequence_buffer
            .copy_from_host(0, &b_sequence_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy b sequence into runtime buffer: {err}"))?;
        a_log_buffer
            .copy_from_host(0, &a_log_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy A_log into runtime buffer: {err}"))?;
        dt_bias_buffer
            .copy_from_host(0, &dt_bias_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy dt_bias into runtime buffer: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after gate/beta aux copy: {err}"))?;
        ullm_runtime_sys::linear_attn_gate_beta_f32(
            &a_sequence_buffer,
            &b_sequence_buffer,
            &a_log_buffer,
            &dt_bias_buffer,
            value_heads,
            sequence_len,
            &mut gate_buffer,
            &mut beta_buffer,
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to run linear attention gate/beta: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after gate/beta: {err}"))?;
        let mut gate_bytes = vec![0_u8; gate_beta_sequence_bytes];
        let mut beta_bytes = vec![0_u8; gate_beta_sequence_bytes];
        gate_buffer
            .copy_to_host(0, &mut gate_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy gate output to host: {err}"))?;
        beta_buffer
            .copy_to_host(0, &mut beta_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy beta output to host: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after gate/beta host copy: {err}"))?;
        let gate_output = decode_f32_le_values(&gate_bytes);
        let beta_output = decode_f32_le_values(&beta_bytes);
        let (expected_gate, expected_beta) = runtime_host_linear_attn_gate_beta_f32(
            &a_output,
            &b_output,
            &a_log.values,
            &dt_bias.values,
            value_heads,
            sequence_len,
        );
        let gate_max_abs_diff = verify_f32_close(
            "package-linear-attn-mlp-block-smoke gate",
            &gate_output,
            &expected_gate,
            1e-4,
            1e-5,
        )?;
        let beta_max_abs_diff = verify_f32_close(
            "package-linear-attn-mlp-block-smoke beta",
            &beta_output,
            &expected_beta,
            1e-4,
            1e-5,
        )?;
        let gate_beta_max_abs_diff = gate_max_abs_diff.max(beta_max_abs_diff);

        let qkv_split = split_linear_attn_qkv_for_recurrent(
            &conv_output,
            sequence_len,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            qk_l2_norm,
            q_scale,
        )
        .map_err(|err| format!("failed to split qkv for recurrent: {err}"))?;
        let state_elements = value_heads
            .checked_mul(key_dim)
            .and_then(|value| value.checked_mul(value_dim))
            .ok_or_else(|| "linear attention state element count overflows".to_string())?;
        let mut expected_state = vec![0.0_f32; state_elements];
        let expected_recurrent = runtime_host_linear_attn_recurrent_f32(
            &qkv_split.q,
            &qkv_split.k,
            &qkv_split.v,
            &expected_gate,
            &expected_beta,
            key_heads,
            value_heads,
            sequence_len,
            key_dim,
            value_dim,
            &mut expected_state,
        );
        let q_bytes = encode_f32_to_bytes(&qkv_split.q);
        let k_bytes = encode_f32_to_bytes(&qkv_split.k);
        let v_bytes = encode_f32_to_bytes(&qkv_split.v);
        let state_bytes = encode_f32_to_bytes(&vec![0.0_f32; state_elements]);
        let mut q_buffer = context
            .alloc_buffer(q_bytes.len())
            .map_err(|err| format!("failed to allocate q buffer: {err}"))?;
        let mut k_buffer = context
            .alloc_buffer(k_bytes.len())
            .map_err(|err| format!("failed to allocate k buffer: {err}"))?;
        let mut v_buffer = context
            .alloc_buffer(v_bytes.len())
            .map_err(|err| format!("failed to allocate v buffer: {err}"))?;
        let mut state_buffer = context
            .alloc_buffer(state_bytes.len())
            .map_err(|err| format!("failed to allocate recurrent state buffer: {err}"))?;
        let mut recurrent_buffer = context
            .alloc_buffer(hidden_sequence_bytes)
            .map_err(|err| format!("failed to allocate recurrent output buffer: {err}"))?;
        q_buffer
            .copy_from_host(0, &q_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy q into runtime buffer: {err}"))?;
        k_buffer
            .copy_from_host(0, &k_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy k into runtime buffer: {err}"))?;
        v_buffer
            .copy_from_host(0, &v_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy v into runtime buffer: {err}"))?;
        state_buffer
            .copy_from_host(0, &state_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy recurrent state into runtime buffer: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after recurrent input copy: {err}"))?;
        ullm_runtime_sys::linear_attn_recurrent_f32(
            &q_buffer,
            &k_buffer,
            &v_buffer,
            &gate_buffer,
            &beta_buffer,
            key_heads,
            value_heads,
            sequence_len,
            key_dim,
            value_dim,
            &mut state_buffer,
            &mut recurrent_buffer,
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to run linear attention recurrent: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after recurrent: {err}"))?;
        let mut recurrent_bytes = vec![0_u8; hidden_sequence_bytes];
        recurrent_buffer
            .copy_to_host(0, &mut recurrent_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy recurrent output to host: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after recurrent host copy: {err}"))?;
        let recurrent_output = decode_f32_le_values(&recurrent_bytes);
        let recurrent_max_abs_diff = verify_f32_close(
            "package-linear-attn-mlp-block-smoke recurrent",
            &recurrent_output,
            &expected_recurrent,
            1e-3,
            1e-5,
        )?;

        let mut expected_attn_normed = vec![0.0_f32; sequence_len * hidden];
        for row in 0..(sequence_len * value_heads) {
            let start = row * value_dim;
            let end = start + value_dim;
            let normed = runtime_host_rmsnorm_f32(
                &expected_recurrent[start..end],
                &attn_norm.values,
                input_epsilon,
            );
            expected_attn_normed[start..end].copy_from_slice(&normed);
        }
        let mut attn_norm_weight_buffer = context
            .alloc_buffer(attn_norm_weight_bytes.len())
            .map_err(|err| {
                format!("failed to allocate linear attention norm weight buffer: {err}")
            })?;
        let mut attn_norm_input_buffer = context
            .alloc_buffer(attn_norm_weight_bytes.len())
            .map_err(|err| {
                format!("failed to allocate linear attention norm input buffer: {err}")
            })?;
        let mut attn_norm_output_buffer = context
            .alloc_buffer(attn_norm_weight_bytes.len())
            .map_err(|err| {
                format!("failed to allocate linear attention norm output buffer: {err}")
            })?;
        attn_norm_weight_buffer
            .copy_from_host(0, &attn_norm_weight_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy linear attention norm weight: {err}"))?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize after linear attention norm weight copy: {err}")
        })?;
        let mut attn_normed_bytes = vec![0_u8; hidden_sequence_bytes];
        for row in 0..(sequence_len * value_heads) {
            let start = row * value_dim;
            let byte_start = start * std::mem::size_of::<f32>();
            let byte_end = byte_start + attn_norm_weight_bytes.len();
            attn_norm_input_buffer
                .copy_from_host(0, &recurrent_bytes[byte_start..byte_end], Some(&mut stream))
                .map_err(|err| format!("failed to copy linear attention norm row {row}: {err}"))?;
            ullm_runtime_sys::rmsnorm_f32(
                &attn_norm_input_buffer,
                &attn_norm_weight_buffer,
                value_dim,
                input_epsilon,
                &mut attn_norm_output_buffer,
                Some(&mut stream),
            )
            .map_err(|err| format!("failed to run linear attention norm row {row}: {err}"))?;
            stream.synchronize().map_err(|err| {
                format!("failed to synchronize after linear attention norm row {row}: {err}")
            })?;
            attn_norm_output_buffer
                .copy_to_host(
                    0,
                    &mut attn_normed_bytes[byte_start..byte_end],
                    Some(&mut stream),
                )
                .map_err(|err| format!("failed to copy linear attention norm row {row}: {err}"))?;
            stream.synchronize().map_err(|err| {
                format!("failed to synchronize after linear attention norm copy row {row}: {err}")
            })?;
        }
        let attn_normed = decode_f32_le_values(&attn_normed_bytes);
        let attn_norm_max_abs_diff = verify_f32_close(
            "package-linear-attn-mlp-block-smoke linear attention norm",
            &attn_normed,
            &expected_attn_normed,
            1e-3,
            1e-5,
        )?;
        let expected_attn_activated = runtime_host_silu_mul_f32(&z_output, &expected_attn_normed);
        let mut z_sequence_buffer = context
            .alloc_buffer(hidden_sequence_bytes)
            .map_err(|err| format!("failed to allocate z sequence buffer: {err}"))?;
        let mut attn_normed_buffer = context
            .alloc_buffer(hidden_sequence_bytes)
            .map_err(|err| format!("failed to allocate linear attention normed buffer: {err}"))?;
        let mut attn_activated_buffer =
            context.alloc_buffer(hidden_sequence_bytes).map_err(|err| {
                format!("failed to allocate linear attention activated buffer: {err}")
            })?;
        z_sequence_buffer
            .copy_from_host(0, &z_sequence_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy z sequence into runtime buffer: {err}"))?;
        attn_normed_buffer
            .copy_from_host(0, &attn_normed_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy linear attention normed values: {err}"))?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize after linear attention normed copy: {err}")
        })?;
        ullm_runtime_sys::silu_mul_f32(
            &z_sequence_buffer,
            &attn_normed_buffer,
            sequence_len * hidden,
            &mut attn_activated_buffer,
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to run linear attention SiLU-mul: {err}"))?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize after linear attention SiLU-mul: {err}")
        })?;
        let mut attn_activated_bytes = vec![0_u8; hidden_sequence_bytes];
        attn_activated_buffer
            .copy_to_host(0, &mut attn_activated_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy linear attention activated values: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after activated host copy: {err}"))?;
        let attn_activated = decode_f32_le_values(&attn_activated_bytes);
        let attn_activation_max_abs_diff = verify_f32_close(
            "package-linear-attn-mlp-block-smoke linear attention activation",
            &attn_activated,
            &expected_attn_activated,
            1e-3,
            1e-5,
        )?;

        let out_matrix_bytes_len = out_rows
            .checked_mul(out_cols)
            .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| "out projection matrix byte size overflows".to_string())?;
        let mut out_matrix_bytes = vec![0_u8; out_matrix_bytes_len];
        out_matrix
            .copy_to_host(0, &mut out_matrix_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy out projection matrix to host: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after out matrix copy: {err}"))?;
        let out_matrix_host = decode_f32_le_values(&out_matrix_bytes);
        let mut expected_attn_output = Vec::with_capacity(sequence_len * hidden);
        let mut attn_activated_step_buffer = context
            .alloc_buffer(hidden_bytes)
            .map_err(|err| format!("failed to allocate attention activated step buffer: {err}"))?;
        let mut attn_output_step_buffer = context
            .alloc_buffer(hidden_bytes)
            .map_err(|err| format!("failed to allocate linear attention output buffer: {err}"))?;
        let mut residual_step_buffer = context
            .alloc_buffer(hidden_bytes)
            .map_err(|err| format!("failed to allocate residual step buffer: {err}"))?;
        let mut attn_block_step_buffer = context
            .alloc_buffer(hidden_bytes)
            .map_err(|err| format!("failed to allocate attention block step buffer: {err}"))?;
        let mut attn_output_bytes = vec![0_u8; hidden_sequence_bytes];
        let mut attn_block_bytes = vec![0_u8; hidden_sequence_bytes];
        let residual_sequence_bytes = encode_f32_to_bytes(&residual_sequence);
        for timestep in 0..sequence_len {
            let element_start = timestep * hidden;
            let element_end = element_start + hidden;
            let byte_start = timestep * hidden_bytes;
            let byte_end = byte_start + hidden_bytes;
            let expected_step = runtime_host_matvec_f32(
                &out_matrix_host,
                &expected_attn_activated[element_start..element_end],
                out_rows,
                out_cols,
            );
            expected_attn_output.extend_from_slice(&expected_step);
            attn_activated_step_buffer
                .copy_from_host(
                    0,
                    &attn_activated_bytes[byte_start..byte_end],
                    Some(&mut stream),
                )
                .map_err(|err| {
                    format!("failed to copy attention activated timestep {timestep}: {err}")
                })?;
            ullm_runtime_sys::matvec_f32(
                &out_matrix,
                &attn_activated_step_buffer,
                out_rows,
                out_cols,
                &mut attn_output_step_buffer,
                Some(&mut stream),
            )
            .map_err(|err| format!("failed to run out projection timestep {timestep}: {err}"))?;
            stream.synchronize().map_err(|err| {
                format!("failed to synchronize after out projection timestep {timestep}: {err}")
            })?;
            attn_output_step_buffer
                .copy_to_host(
                    0,
                    &mut attn_output_bytes[byte_start..byte_end],
                    Some(&mut stream),
                )
                .map_err(|err| {
                    format!("failed to copy linear attention output timestep {timestep}: {err}")
                })?;
            residual_step_buffer
                .copy_from_host(
                    0,
                    &residual_sequence_bytes[byte_start..byte_end],
                    Some(&mut stream),
                )
                .map_err(|err| format!("failed to copy residual timestep {timestep}: {err}"))?;
            ullm_runtime_sys::add_f32(
                &residual_step_buffer,
                &attn_output_step_buffer,
                hidden,
                &mut attn_block_step_buffer,
                Some(&mut stream),
            )
            .map_err(|err| {
                format!("failed to run attention residual add timestep {timestep}: {err}")
            })?;
            stream.synchronize().map_err(|err| {
                format!("failed to synchronize after attention residual timestep {timestep}: {err}")
            })?;
            attn_block_step_buffer
                .copy_to_host(
                    0,
                    &mut attn_block_bytes[byte_start..byte_end],
                    Some(&mut stream),
                )
                .map_err(|err| {
                    format!("failed to copy attention block timestep {timestep}: {err}")
                })?;
            stream.synchronize().map_err(|err| {
                format!(
                    "failed to synchronize after attention block host copy timestep {timestep}: {err}"
                )
            })?;
        }
        let attn_output = decode_f32_le_values(&attn_output_bytes);
        let attn_output_max_abs_diff = verify_f32_close(
            "package-linear-attn-mlp-block-smoke out projection",
            &attn_output,
            &expected_attn_output,
            3e-3,
            2e-5,
        )?;
        let attention_block_output = decode_f32_le_values(&attn_block_bytes);
        let expected_attention_block = runtime_host_add_f32(&residual_sequence, &attn_output);
        let attn_block_max_abs_diff = verify_f32_close(
            "package-linear-attn-mlp-block-smoke attention residual",
            &attention_block_output,
            &expected_attention_block,
            1e-5,
            1e-6,
        )?;
        (
            attention_block_output,
            attn_output,
            attn_block_max_abs_diff,
            conv_max_abs_diff,
            gate_beta_max_abs_diff,
            recurrent_max_abs_diff,
            attn_norm_max_abs_diff,
            attn_activation_max_abs_diff,
            attn_output_max_abs_diff,
        )
    };

    let mut post_normed_expected = Vec::with_capacity(sequence_len * hidden);
    for timestep in 0..sequence_len {
        let start = timestep * hidden;
        let end = start + hidden;
        let expected = runtime_host_rmsnorm_f32(
            &attention_block_output[start..end],
            &post_norm.values,
            mlp_epsilon,
        );
        post_normed_expected.extend_from_slice(&expected);
    }
    let mut attn_block_step_buffer = context
        .alloc_buffer(hidden_bytes)
        .map_err(|err| format!("failed to allocate retained attention block buffer: {err}"))?;
    let mut post_norm_weight_buffer = context
        .alloc_buffer(post_norm_weight_bytes.len())
        .map_err(|err| format!("failed to allocate post RMSNorm weight buffer: {err}"))?;
    let mut post_normed_buffer = context
        .alloc_buffer(hidden_bytes)
        .map_err(|err| format!("failed to allocate post RMSNorm output buffer: {err}"))?;
    let attention_block_bytes = encode_f32_to_bytes(&attention_block_output);
    post_norm_weight_buffer
        .copy_from_host(0, &post_norm_weight_bytes, Some(&mut stream))
        .map_err(|err| format!("failed to copy post RMSNorm weight into runtime buffer: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize after post RMSNorm weight copy: {err}"))?;
    let mut post_normed_bytes = vec![0_u8; hidden_sequence_bytes];
    for timestep in 0..sequence_len {
        let byte_start = timestep * hidden_bytes;
        let byte_end = byte_start + hidden_bytes;
        attn_block_step_buffer
            .copy_from_host(
                0,
                &attention_block_bytes[byte_start..byte_end],
                Some(&mut stream),
            )
            .map_err(|err| {
                format!("failed to copy attention block timestep {timestep} for post norm: {err}")
            })?;
        ullm_runtime_sys::rmsnorm_f32(
            &attn_block_step_buffer,
            &post_norm_weight_buffer,
            hidden,
            mlp_epsilon,
            &mut post_normed_buffer,
            Some(&mut stream),
        )
        .map_err(|err| format!("failed to run post RMSNorm timestep {timestep}: {err}"))?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize after post RMSNorm timestep {timestep}: {err}")
        })?;
        post_normed_buffer
            .copy_to_host(
                0,
                &mut post_normed_bytes[byte_start..byte_end],
                Some(&mut stream),
            )
            .map_err(|err| {
                format!("failed to copy post RMSNorm timestep {timestep} to host: {err}")
            })?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize after post RMSNorm host copy {timestep}: {err}")
        })?;
    }
    let post_normed = decode_f32_le_values(&post_normed_bytes);
    let post_norm_max_abs_diff = verify_f32_close(
        "package-linear-attn-mlp-block-smoke post RMSNorm",
        &post_normed,
        &post_normed_expected,
        1e-4,
        1e-5,
    )?;

    let (mlp_output, layer_output, layer_block_max_abs_diff) = {
        let mut registry = WeightRegistry::new();
        let (gate_rows, gate_cols, gate_matrix) = materialize_selected_aq4_matrix(
            &mut context,
            &mut stream,
            &mut registry,
            path,
            &gate_tensor,
            chunk_bytes,
        )?;
        let (up_rows, up_cols, up_matrix) = materialize_selected_aq4_matrix(
            &mut context,
            &mut stream,
            &mut registry,
            path,
            &up_tensor,
            chunk_bytes,
        )?;
        let (down_rows, down_cols, down_matrix) = materialize_selected_aq4_matrix(
            &mut context,
            &mut stream,
            &mut registry,
            path,
            &down_tensor,
            chunk_bytes,
        )?;
        if gate_rows != up_rows || gate_cols != up_cols || gate_cols != hidden {
            return Err(format!(
                "MLP gate/up shape mismatch: gate=[{gate_rows},{gate_cols}] up=[{up_rows},{up_cols}] hidden={hidden}"
            ));
        }
        if down_rows != hidden || down_cols != gate_rows {
            return Err(format!(
                "MLP down shape mismatch: expected [{hidden},{gate_rows}], got [{down_rows},{down_cols}]"
            ));
        }
        let intermediate = gate_rows;
        let intermediate_bytes = intermediate
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "MLP intermediate byte size overflows".to_string())?;
        let mut post_normed_step_buffer = context
            .alloc_buffer(hidden_bytes)
            .map_err(|err| format!("failed to allocate post normed step buffer: {err}"))?;
        let mut gate_buffer = context
            .alloc_buffer(intermediate_bytes)
            .map_err(|err| format!("failed to allocate MLP gate buffer: {err}"))?;
        let mut up_buffer = context
            .alloc_buffer(intermediate_bytes)
            .map_err(|err| format!("failed to allocate MLP up buffer: {err}"))?;
        let mut mlp_activated_buffer = context
            .alloc_buffer(intermediate_bytes)
            .map_err(|err| format!("failed to allocate MLP activated buffer: {err}"))?;
        let mut mlp_output_buffer = context
            .alloc_buffer(hidden_bytes)
            .map_err(|err| format!("failed to allocate MLP output buffer: {err}"))?;
        let mut layer_output_buffer = context
            .alloc_buffer(hidden_bytes)
            .map_err(|err| format!("failed to allocate layer output buffer: {err}"))?;
        let mut mlp_output_bytes = vec![0_u8; hidden_sequence_bytes];
        let mut layer_output_bytes = vec![0_u8; hidden_sequence_bytes];
        for timestep in 0..sequence_len {
            let byte_start = timestep * hidden_bytes;
            let byte_end = byte_start + hidden_bytes;
            post_normed_step_buffer
                .copy_from_host(
                    0,
                    &post_normed_bytes[byte_start..byte_end],
                    Some(&mut stream),
                )
                .map_err(|err| format!("failed to copy post normed timestep {timestep}: {err}"))?;
            ullm_runtime_sys::matvec_f32(
                &gate_matrix,
                &post_normed_step_buffer,
                gate_rows,
                gate_cols,
                &mut gate_buffer,
                Some(&mut stream),
            )
            .map_err(|err| format!("failed to run MLP gate matvec timestep {timestep}: {err}"))?;
            ullm_runtime_sys::matvec_f32(
                &up_matrix,
                &post_normed_step_buffer,
                up_rows,
                up_cols,
                &mut up_buffer,
                Some(&mut stream),
            )
            .map_err(|err| format!("failed to run MLP up matvec timestep {timestep}: {err}"))?;
            stream.synchronize().map_err(|err| {
                format!("failed to synchronize after MLP gate/up timestep {timestep}: {err}")
            })?;
            ullm_runtime_sys::silu_mul_f32(
                &gate_buffer,
                &up_buffer,
                intermediate,
                &mut mlp_activated_buffer,
                Some(&mut stream),
            )
            .map_err(|err| format!("failed to run MLP SiLU-mul timestep {timestep}: {err}"))?;
            stream.synchronize().map_err(|err| {
                format!("failed to synchronize after MLP SiLU-mul timestep {timestep}: {err}")
            })?;
            ullm_runtime_sys::matvec_f32(
                &down_matrix,
                &mlp_activated_buffer,
                down_rows,
                down_cols,
                &mut mlp_output_buffer,
                Some(&mut stream),
            )
            .map_err(|err| format!("failed to run MLP down matvec timestep {timestep}: {err}"))?;
            stream.synchronize().map_err(|err| {
                format!("failed to synchronize after MLP down timestep {timestep}: {err}")
            })?;
            mlp_output_buffer
                .copy_to_host(
                    0,
                    &mut mlp_output_bytes[byte_start..byte_end],
                    Some(&mut stream),
                )
                .map_err(|err| format!("failed to copy MLP output timestep {timestep}: {err}"))?;
            attn_block_step_buffer
                .copy_from_host(
                    0,
                    &attention_block_bytes[byte_start..byte_end],
                    Some(&mut stream),
                )
                .map_err(|err| {
                    format!(
                        "failed to copy attention block timestep {timestep} for MLP residual: {err}"
                    )
                })?;
            ullm_runtime_sys::add_f32(
                &attn_block_step_buffer,
                &mlp_output_buffer,
                hidden,
                &mut layer_output_buffer,
                Some(&mut stream),
            )
            .map_err(|err| format!("failed to run MLP residual add timestep {timestep}: {err}"))?;
            stream.synchronize().map_err(|err| {
                format!("failed to synchronize after MLP residual timestep {timestep}: {err}")
            })?;
            layer_output_buffer
                .copy_to_host(
                    0,
                    &mut layer_output_bytes[byte_start..byte_end],
                    Some(&mut stream),
                )
                .map_err(|err| format!("failed to copy layer output timestep {timestep}: {err}"))?;
            stream.synchronize().map_err(|err| {
                format!("failed to synchronize after layer output copy timestep {timestep}: {err}")
            })?;
        }
        let mlp_output = decode_f32_le_values(&mlp_output_bytes);
        let layer_output = decode_f32_le_values(&layer_output_bytes);
        let expected_layer_output = runtime_host_add_f32(&attention_block_output, &mlp_output);
        let layer_block_max_abs_diff = verify_f32_close(
            "package-linear-attn-mlp-block-smoke layer residual",
            &layer_output,
            &expected_layer_output,
            1e-5,
            1e-6,
        )?;
        (mlp_output, layer_output, layer_block_max_abs_diff)
    };

    Ok(format!(
        "package-linear-attn-mlp-block-smoke package={} layer={} input_norm_tensor=\"{}\" qkv_tensor=\"{}\" conv_tensor=\"{}\" a_tensor=\"{}\" b_tensor=\"{}\" a_log_tensor=\"{}\" dt_bias_tensor=\"{}\" z_tensor=\"{}\" norm_tensor=\"{}\" out_tensor=\"{}\" post_norm_tensor=\"{}\" gate_tensor=\"{}\" up_tensor=\"{}\" down_tensor=\"{}\" hidden={} key_heads={} value_heads={} key_dim={} value_dim={} sequence_len={} kernel_size={} qk_l2_norm={} q_scale={q_scale:.9} input_norm_dtype={} conv_dtype={} a_log_dtype={} dt_bias_dtype={} norm_dtype={} post_norm_dtype={} backend={} device_index={} name=\"{}\" residual_preview={} attention_output_preview={} attention_block_preview={} post_norm_preview={} mlp_output_preview={} layer_output_preview={} input_norm_max_abs_diff={input_norm_max_abs_diff:.9} conv_max_abs_diff={conv_max_abs_diff:.9} gate_beta_max_abs_diff={gate_beta_max_abs_diff:.9} recurrent_max_abs_diff={recurrent_max_abs_diff:.9} attn_norm_max_abs_diff={attn_norm_max_abs_diff:.9} attn_activation_max_abs_diff={attn_activation_max_abs_diff:.9} attn_output_max_abs_diff={attn_output_max_abs_diff:.9} attn_block_max_abs_diff={attn_block_max_abs_diff:.9} post_norm_max_abs_diff={post_norm_max_abs_diff:.9} layer_block_max_abs_diff={layer_block_max_abs_diff:.9} verified=true",
        path,
        layer_index,
        input_norm_tensor,
        qkv_tensor,
        conv_tensor,
        a_tensor,
        b_tensor,
        a_log_tensor,
        dt_bias_tensor,
        z_tensor,
        norm_tensor,
        out_tensor,
        post_norm_tensor,
        gate_tensor,
        up_tensor,
        down_tensor,
        hidden,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        sequence_len,
        kernel_size,
        qk_l2_norm,
        input_norm.dtype,
        conv.dtype,
        a_log.dtype,
        dt_bias.dtype,
        attn_norm.dtype,
        post_norm.dtype,
        info.backend,
        device_index,
        info.name,
        format_f32_preview(&residual_sequence[..8.min(residual_sequence.len())]),
        format_f32_preview(&attn_output[..8.min(attn_output.len())]),
        format_f32_preview(&attention_block_output[..8.min(attention_block_output.len())]),
        format_f32_preview(&post_normed[..8.min(post_normed.len())]),
        format_f32_preview(&mlp_output[..8.min(mlp_output.len())]),
        format_f32_preview(&layer_output[..8.min(layer_output.len())]),
    ))
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

#[derive(Debug, Clone, Copy)]
enum SelfAttnProjection {
    Q,
    K,
    V,
    O,
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
    let raw = value.unwrap_or("all");
    match raw {
        "a" => Ok(LinearAttnProjection::A),
        "b" => Ok(LinearAttnProjection::B),
        "qkv" => Ok(LinearAttnProjection::Qkv),
        "z" => Ok(LinearAttnProjection::Z),
        "out" => Ok(LinearAttnProjection::Out),
        "all" => Ok(LinearAttnProjection::All),
        _raw => {
            eprintln!("invalid projection: {raw}; expected a, b, qkv, z, out, or all");
            Err(ExitCode::from(2))
        }
    }
}

fn parse_self_attn_projection(value: Option<&str>) -> Result<SelfAttnProjection, ExitCode> {
    let raw = value.unwrap_or("all");
    match raw {
        "q" => Ok(SelfAttnProjection::Q),
        "k" => Ok(SelfAttnProjection::K),
        "v" => Ok(SelfAttnProjection::V),
        "o" | "out" => Ok(SelfAttnProjection::O),
        "all" => Ok(SelfAttnProjection::All),
        _raw => {
            eprintln!("invalid self-attn projection: {raw}; expected q, k, v, o, or all");
            Err(ExitCode::from(2))
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum LinearAttnAux {
    ALog,
    DtBias,
    Conv1d,
    Norm,
    All,
}

fn parse_linear_attn_aux(value: Option<&str>) -> Result<LinearAttnAux, ExitCode> {
    let raw = value.unwrap_or("all");
    let normalized = raw.replace(['-', '_'], "");
    match normalized.as_str() {
        "alog" => Ok(LinearAttnAux::ALog),
        "dtbias" => Ok(LinearAttnAux::DtBias),
        "conv1d" => Ok(LinearAttnAux::Conv1d),
        "norm" => Ok(LinearAttnAux::Norm),
        "all" => Ok(LinearAttnAux::All),
        _value => {
            eprintln!(
                "invalid aux: {raw}; expected a-log, dt-bias, conv1d, norm, or all (aliases: a_log, alog, dt_bias)"
            );
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

fn validate_passthrough_shape_elements(bundle: &PassthroughPayloadBundle) -> Result<(), String> {
    if bundle.shape.is_empty() {
        return Ok(());
    }
    let mut product = 1_u64;
    for dimension in &bundle.shape {
        if *dimension == 0 {
            return Err("shape contains zero".to_string());
        }
        product = product
            .checked_mul(*dimension)
            .ok_or_else(|| "shape element count overflows u64".to_string())?;
    }
    if product != bundle.elements {
        return Err(format!(
            "shape product {} does not match element count {}",
            product, bundle.elements
        ));
    }
    Ok(())
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

struct PassthroughF32Data {
    values: Vec<f32>,
    dtype: String,
    shape: Vec<u64>,
}

fn read_named_passthrough_f32(
    package_path: &str,
    tensor_name: &str,
    chunk_bytes: usize,
) -> Result<PassthroughF32Data, String> {
    let selector = TensorSelector::Name(tensor_name.to_string());
    let bundle = ullm_engine::package::select_passthrough_payload_bundle(package_path, &selector)
        .map_err(|err| {
        format!("failed to select package passthrough tensor {tensor_name}: {err}")
    })?;
    validate_passthrough_shape_elements(&bundle)
        .map_err(|err| format!("invalid passthrough shape for {tensor_name}: {err}"))?;
    let dtype = resolve_passthrough_dtype(&bundle, tensor_name)?.to_string();
    let values = read_passthrough_payload_f32_bytes(&bundle, chunk_bytes, &dtype)
        .map_err(|err| format!("failed to read passthrough payload for {tensor_name}: {err}"))?;
    let expected_elements = usize::try_from(bundle.elements)
        .map_err(|_| format!("passthrough tensor {tensor_name} is too large for this host"))?;
    if values.len() != expected_elements {
        return Err(format!(
            "passthrough tensor element count mismatch for {tensor_name}: expected {} got {}",
            expected_elements,
            values.len()
        ));
    }
    Ok(PassthroughF32Data {
        values,
        dtype,
        shape: bundle.shape,
    })
}

fn runtime_matvec_to_host_f32(
    context: &mut ullm_runtime_sys::RuntimeContext,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    matrix: &ullm_runtime_sys::RuntimeBuffer,
    input: &ullm_runtime_sys::RuntimeBuffer,
    rows: usize,
    cols: usize,
    label: &str,
) -> Result<Vec<f32>, String> {
    let output_bytes = rows
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| format!("{label} output byte size overflows"))?;
    let mut output = context
        .alloc_buffer(output_bytes)
        .map_err(|err| format!("failed to allocate {label} output buffer: {err}"))?;
    ullm_runtime_sys::matvec_f32(matrix, input, rows, cols, &mut output, Some(stream))
        .map_err(|err| format!("failed to run {label} matvec: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize after {label} matvec: {err}"))?;
    let mut output_host = vec![0_u8; output_bytes];
    output
        .copy_to_host(0, &mut output_host, Some(stream))
        .map_err(|err| format!("failed to copy {label} output to host: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize after {label} host copy: {err}"))?;
    Ok(decode_f32_le_values(&output_host))
}

fn runtime_headwise_rmsnorm_verify(
    context: &mut ullm_runtime_sys::RuntimeContext,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    input: &[f32],
    weight: &[f32],
    epsilon: f32,
    label: &str,
) -> Result<(Vec<f32>, f32), String> {
    let head_dim = weight.len();
    if head_dim == 0 {
        return Err(format!("{label} weight must not be empty"));
    }
    if !input.len().is_multiple_of(head_dim) {
        return Err(format!(
            "{label} input length {} is not a multiple of head_dim {head_dim}",
            input.len()
        ));
    }

    let head_bytes = head_dim
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| format!("{label} head byte size overflows"))?;
    let weight_bytes = encode_f32_to_bytes(weight);
    let mut weight_buffer = context
        .alloc_buffer(weight_bytes.len())
        .map_err(|err| format!("failed to allocate {label} weight buffer: {err}"))?;
    weight_buffer
        .copy_from_host(0, &weight_bytes, Some(stream))
        .map_err(|err| format!("failed to copy {label} weight: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize after {label} weight copy: {err}"))?;

    let mut input_buffer = context
        .alloc_buffer(head_bytes)
        .map_err(|err| format!("failed to allocate {label} input buffer: {err}"))?;
    let mut output_buffer = context
        .alloc_buffer(head_bytes)
        .map_err(|err| format!("failed to allocate {label} output buffer: {err}"))?;
    let mut output = Vec::with_capacity(input.len());
    let mut max_abs_diff = 0.0_f32;
    let mut output_head_bytes = vec![0_u8; head_bytes];
    for (head_index, head_input) in input.chunks_exact(head_dim).enumerate() {
        let head_input_bytes = encode_f32_to_bytes(head_input);
        input_buffer
            .copy_from_host(0, &head_input_bytes, Some(stream))
            .map_err(|err| format!("failed to copy {label} head {head_index} input: {err}"))?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize after {label} head {head_index} input copy: {err}")
        })?;
        ullm_runtime_sys::rmsnorm_f32(
            &input_buffer,
            &weight_buffer,
            head_dim,
            epsilon,
            &mut output_buffer,
            Some(stream),
        )
        .map_err(|err| format!("failed to run {label} head {head_index} RMSNorm: {err}"))?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize after {label} head {head_index} RMSNorm: {err}")
        })?;
        output_buffer
            .copy_to_host(0, &mut output_head_bytes, Some(stream))
            .map_err(|err| format!("failed to copy {label} head {head_index} output: {err}"))?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize after {label} head {head_index} output copy: {err}")
        })?;
        let actual = decode_f32_le_values(&output_head_bytes);
        let expected = runtime_host_rmsnorm_f32(head_input, weight, epsilon);
        let head_max_abs_diff = verify_f32_close(
            &format!("{label} head {head_index}"),
            &actual,
            &expected,
            1e-4_f32,
            1e-4_f32,
        )?;
        max_abs_diff = max_abs_diff.max(head_max_abs_diff);
        output.extend(actual);
    }
    Ok((output, max_abs_diff))
}

#[allow(clippy::too_many_arguments)]
fn runtime_rope_verify(
    context: &mut ullm_runtime_sys::RuntimeContext,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    input: &[f32],
    sequence_len: usize,
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    position_offset: usize,
    rope_base: f32,
    label: &str,
) -> Result<(Vec<f32>, f32), String> {
    if input.len() != sequence_len * heads * head_dim {
        return Err(format!(
            "{label} input length {} does not match sequence_len={sequence_len} heads={heads} head_dim={head_dim}",
            input.len()
        ));
    }
    let bytes = input
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| format!("{label} byte size overflows"))?;
    let input_bytes = encode_f32_to_bytes(input);
    let mut input_buffer = context
        .alloc_buffer(bytes)
        .map_err(|err| format!("failed to allocate {label} input buffer: {err}"))?;
    let mut output_buffer = context
        .alloc_buffer(bytes)
        .map_err(|err| format!("failed to allocate {label} output buffer: {err}"))?;
    input_buffer
        .copy_from_host(0, &input_bytes, Some(stream))
        .map_err(|err| format!("failed to copy {label} input: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize after {label} input copy: {err}"))?;
    ullm_runtime_sys::rope_f32(
        &input_buffer,
        sequence_len,
        heads,
        head_dim,
        rotary_dim,
        position_offset,
        rope_base,
        &mut output_buffer,
        Some(stream),
    )
    .map_err(|err| format!("failed to run {label} RoPE: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize after {label} RoPE: {err}"))?;

    let mut output_bytes = vec![0_u8; bytes];
    output_buffer
        .copy_to_host(0, &mut output_bytes, Some(stream))
        .map_err(|err| format!("failed to copy {label} output: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize after {label} output copy: {err}"))?;
    let output = decode_f32_le_values(&output_bytes);
    let expected = runtime_host_rope_f32(
        input,
        sequence_len,
        heads,
        head_dim,
        rotary_dim,
        position_offset,
        rope_base,
    );
    let max_abs_diff = verify_f32_close(label, &output, &expected, 1e-4_f32, 1e-4_f32)?;
    Ok((output, max_abs_diff))
}

#[allow(clippy::too_many_arguments)]
fn runtime_causal_attn_verify(
    context: &mut ullm_runtime_sys::RuntimeContext,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    sequence_len: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
    label: &str,
) -> Result<(Vec<f32>, f32), String> {
    if q.len() != sequence_len * q_heads * head_dim {
        return Err(format!(
            "{label} q length {} does not match sequence_len={sequence_len} q_heads={q_heads} head_dim={head_dim}",
            q.len()
        ));
    }
    if k.len() != sequence_len * kv_heads * head_dim {
        return Err(format!(
            "{label} k length {} does not match sequence_len={sequence_len} kv_heads={kv_heads} head_dim={head_dim}",
            k.len()
        ));
    }
    if v.len() != sequence_len * kv_heads * value_dim {
        return Err(format!(
            "{label} v length {} does not match sequence_len={sequence_len} kv_heads={kv_heads} value_dim={value_dim}",
            v.len()
        ));
    }
    let q_bytes = encode_f32_to_bytes(q);
    let k_bytes = encode_f32_to_bytes(k);
    let v_bytes = encode_f32_to_bytes(v);
    let output_elements = sequence_len * q_heads * value_dim;
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| format!("{label} output byte size overflows"))?;
    let mut q_buffer = context
        .alloc_buffer(q_bytes.len())
        .map_err(|err| format!("failed to allocate {label} q buffer: {err}"))?;
    let mut k_buffer = context
        .alloc_buffer(k_bytes.len())
        .map_err(|err| format!("failed to allocate {label} k buffer: {err}"))?;
    let mut v_buffer = context
        .alloc_buffer(v_bytes.len())
        .map_err(|err| format!("failed to allocate {label} v buffer: {err}"))?;
    let mut output_buffer = context
        .alloc_buffer(output_bytes)
        .map_err(|err| format!("failed to allocate {label} output buffer: {err}"))?;
    q_buffer
        .copy_from_host(0, &q_bytes, Some(stream))
        .map_err(|err| format!("failed to copy {label} q input: {err}"))?;
    k_buffer
        .copy_from_host(0, &k_bytes, Some(stream))
        .map_err(|err| format!("failed to copy {label} k input: {err}"))?;
    v_buffer
        .copy_from_host(0, &v_bytes, Some(stream))
        .map_err(|err| format!("failed to copy {label} v input: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize after {label} input copies: {err}"))?;
    ullm_runtime_sys::causal_attn_f32(
        &q_buffer,
        &k_buffer,
        &v_buffer,
        sequence_len,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        &mut output_buffer,
        Some(stream),
    )
    .map_err(|err| format!("failed to run {label} causal attention: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize after {label} causal attention: {err}"))?;
    let mut output_bytes_host = vec![0_u8; output_bytes];
    output_buffer
        .copy_to_host(0, &mut output_bytes_host, Some(stream))
        .map_err(|err| format!("failed to copy {label} output: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize after {label} output copy: {err}"))?;
    let output = decode_f32_le_values(&output_bytes_host);
    let expected = runtime_host_causal_attn_f32(
        q,
        k,
        v,
        sequence_len,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
    );
    let max_abs_diff = verify_f32_close(label, &output, &expected, 1e-4_f32, 1e-4_f32)?;
    Ok((output, max_abs_diff))
}

#[allow(clippy::too_many_arguments)]
fn runtime_decode_attn_verify(
    context: &mut ullm_runtime_sys::RuntimeContext,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    cache_len: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
    label: &str,
) -> Result<(Vec<f32>, f32), String> {
    if q.len() != q_heads * head_dim {
        return Err(format!(
            "{label} q length {} does not match q_heads={q_heads} head_dim={head_dim}",
            q.len()
        ));
    }
    if k_cache.len() != cache_len * kv_heads * head_dim {
        return Err(format!(
            "{label} k cache length {} does not match cache_len={cache_len} kv_heads={kv_heads} head_dim={head_dim}",
            k_cache.len()
        ));
    }
    if v_cache.len() != cache_len * kv_heads * value_dim {
        return Err(format!(
            "{label} v cache length {} does not match cache_len={cache_len} kv_heads={kv_heads} value_dim={value_dim}",
            v_cache.len()
        ));
    }
    let q_bytes = encode_f32_to_bytes(q);
    let k_bytes = encode_f32_to_bytes(k_cache);
    let v_bytes = encode_f32_to_bytes(v_cache);
    let output_elements = q_heads * value_dim;
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| format!("{label} output byte size overflows"))?;
    let mut q_buffer = context
        .alloc_buffer(q_bytes.len())
        .map_err(|err| format!("failed to allocate {label} q buffer: {err}"))?;
    let mut k_buffer = context
        .alloc_buffer(k_bytes.len())
        .map_err(|err| format!("failed to allocate {label} k cache buffer: {err}"))?;
    let mut v_buffer = context
        .alloc_buffer(v_bytes.len())
        .map_err(|err| format!("failed to allocate {label} v cache buffer: {err}"))?;
    let mut output_buffer = context
        .alloc_buffer(output_bytes)
        .map_err(|err| format!("failed to allocate {label} output buffer: {err}"))?;
    q_buffer
        .copy_from_host(0, &q_bytes, Some(stream))
        .map_err(|err| format!("failed to copy {label} q input: {err}"))?;
    k_buffer
        .copy_from_host(0, &k_bytes, Some(stream))
        .map_err(|err| format!("failed to copy {label} k cache input: {err}"))?;
    v_buffer
        .copy_from_host(0, &v_bytes, Some(stream))
        .map_err(|err| format!("failed to copy {label} v cache input: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize after {label} input copies: {err}"))?;
    ullm_runtime_sys::decode_attn_f32(
        &q_buffer,
        &k_buffer,
        &v_buffer,
        cache_len,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        &mut output_buffer,
        Some(stream),
    )
    .map_err(|err| format!("failed to run {label} decode attention: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize after {label} decode attention: {err}"))?;
    let mut output_bytes_host = vec![0_u8; output_bytes];
    output_buffer
        .copy_to_host(0, &mut output_bytes_host, Some(stream))
        .map_err(|err| format!("failed to copy {label} output: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize after {label} output copy: {err}"))?;
    let output = decode_f32_le_values(&output_bytes_host);
    let expected = runtime_host_decode_attn_f32(
        q,
        k_cache,
        v_cache,
        cache_len,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
    );
    let max_abs_diff = verify_f32_close(label, &output, &expected, 1e-4_f32, 1e-4_f32)?;
    Ok((output, max_abs_diff))
}

#[allow(clippy::too_many_arguments)]
fn runtime_paged_decode_attn_verify(
    context: &mut ullm_runtime_sys::RuntimeContext,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    block_table: &[u32],
    cache_len: usize,
    block_size: usize,
    cache_blocks: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
    label: &str,
) -> Result<(Vec<f32>, f32), String> {
    if cache_len == 0 {
        return Err(format!("{label} cache_len must be greater than zero"));
    }
    if block_size == 0 {
        return Err(format!("{label} block_size must be greater than zero"));
    }
    if cache_blocks == 0 {
        return Err(format!("{label} cache_blocks must be greater than zero"));
    }
    if q.len() != q_heads * head_dim {
        return Err(format!(
            "{label} q length {} does not match q_heads={q_heads} head_dim={head_dim}",
            q.len()
        ));
    }
    let block_table_entries = (cache_len - 1) / block_size + 1;
    if block_table.len() != block_table_entries {
        return Err(format!(
            "{label} block table length {} does not match expected entries {block_table_entries}",
            block_table.len()
        ));
    }
    let physical_tokens = cache_blocks
        .checked_mul(block_size)
        .ok_or_else(|| format!("{label} physical cache token count overflows"))?;
    if k_cache.len() != physical_tokens * kv_heads * head_dim {
        return Err(format!(
            "{label} k cache length {} does not match cache_blocks={cache_blocks} block_size={block_size} kv_heads={kv_heads} head_dim={head_dim}",
            k_cache.len()
        ));
    }
    if v_cache.len() != physical_tokens * kv_heads * value_dim {
        return Err(format!(
            "{label} v cache length {} does not match cache_blocks={cache_blocks} block_size={block_size} kv_heads={kv_heads} value_dim={value_dim}",
            v_cache.len()
        ));
    }
    let q_bytes = encode_f32_to_bytes(q);
    let k_bytes = encode_f32_to_bytes(k_cache);
    let v_bytes = encode_f32_to_bytes(v_cache);
    let block_table_bytes = encode_u32_to_bytes(block_table);
    let output_elements = q_heads * value_dim;
    let output_bytes = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| format!("{label} output byte size overflows"))?;
    let mut q_buffer = context
        .alloc_buffer(q_bytes.len())
        .map_err(|err| format!("failed to allocate {label} q buffer: {err}"))?;
    let mut k_buffer = context
        .alloc_buffer(k_bytes.len())
        .map_err(|err| format!("failed to allocate {label} paged k cache buffer: {err}"))?;
    let mut v_buffer = context
        .alloc_buffer(v_bytes.len())
        .map_err(|err| format!("failed to allocate {label} paged v cache buffer: {err}"))?;
    let mut block_table_buffer = context
        .alloc_buffer(block_table_bytes.len())
        .map_err(|err| format!("failed to allocate {label} block table buffer: {err}"))?;
    let mut output_buffer = context
        .alloc_buffer(output_bytes)
        .map_err(|err| format!("failed to allocate {label} output buffer: {err}"))?;
    q_buffer
        .copy_from_host(0, &q_bytes, Some(stream))
        .map_err(|err| format!("failed to copy {label} q input: {err}"))?;
    k_buffer
        .copy_from_host(0, &k_bytes, Some(stream))
        .map_err(|err| format!("failed to copy {label} paged k cache input: {err}"))?;
    v_buffer
        .copy_from_host(0, &v_bytes, Some(stream))
        .map_err(|err| format!("failed to copy {label} paged v cache input: {err}"))?;
    block_table_buffer
        .copy_from_host(0, &block_table_bytes, Some(stream))
        .map_err(|err| format!("failed to copy {label} block table input: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize after {label} input copies: {err}"))?;
    ullm_runtime_sys::paged_decode_attn_f32(
        &q_buffer,
        &k_buffer,
        &v_buffer,
        &block_table_buffer,
        cache_len,
        block_size,
        cache_blocks,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        &mut output_buffer,
        Some(stream),
    )
    .map_err(|err| format!("failed to run {label} paged decode attention: {err}"))?;
    stream.synchronize().map_err(|err| {
        format!("failed to synchronize after {label} paged decode attention: {err}")
    })?;
    let mut output_bytes_host = vec![0_u8; output_bytes];
    output_buffer
        .copy_to_host(0, &mut output_bytes_host, Some(stream))
        .map_err(|err| format!("failed to copy {label} output: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize after {label} output copy: {err}"))?;
    let output = decode_f32_le_values(&output_bytes_host);
    let expected = runtime_host_paged_decode_attn_f32(
        q,
        k_cache,
        v_cache,
        block_table,
        cache_len,
        block_size,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
    );
    let max_abs_diff = verify_f32_close(label, &output, &expected, 1e-4_f32, 1e-4_f32)?;
    Ok((output, max_abs_diff))
}

struct RuntimePagedKvWriteDecodeResult {
    output: Vec<f32>,
    step_outputs: Vec<f32>,
    output_max_abs_diff: f32,
    step_output_max_abs_diff: f32,
    k_cache: Vec<f32>,
    v_cache: Vec<f32>,
    k_write_max_abs_diff: f32,
    v_write_max_abs_diff: f32,
}

#[allow(clippy::too_many_arguments)]
fn runtime_paged_kv_write_decode_verify(
    context: &mut ullm_runtime_sys::RuntimeContext,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    q_sequence: &[f32],
    logical_k_cache: &[f32],
    logical_v_cache: &[f32],
    block_table: &[u32],
    cache_len: usize,
    block_size: usize,
    cache_blocks: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
    label: &str,
) -> Result<RuntimePagedKvWriteDecodeResult, String> {
    if q_sequence.len() != cache_len * q_heads * head_dim {
        return Err(format!(
            "{label} q sequence length {} does not match cache_len={cache_len} q_heads={q_heads} head_dim={head_dim}",
            q_sequence.len()
        ));
    }
    if logical_k_cache.len() != cache_len * kv_heads * head_dim {
        return Err(format!(
            "{label} logical k cache length {} does not match cache_len={cache_len} kv_heads={kv_heads} head_dim={head_dim}",
            logical_k_cache.len()
        ));
    }
    if logical_v_cache.len() != cache_len * kv_heads * value_dim {
        return Err(format!(
            "{label} logical v cache length {} does not match cache_len={cache_len} kv_heads={kv_heads} value_dim={value_dim}",
            logical_v_cache.len()
        ));
    }
    let (expected_k_cache, expected_v_cache) = pack_paged_decode_kv_cache_for_block_table(
        logical_k_cache,
        logical_v_cache,
        block_table,
        cache_len,
        block_size,
        cache_blocks,
        kv_heads,
        head_dim,
        value_dim,
    )?;
    let shape = PagedDecodeShape {
        block_size,
        cache_blocks,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
    };
    let mut state =
        Qwen3SelfAttnDecodeState::new(context, stream, shape, block_table.to_vec(), softmax_scale)
            .map_err(|err| {
                format!("failed to create {label} Qwen3 self-attn decode state: {err}")
            })?;
    let q_token_elements = q_heads * head_dim;
    let k_token_elements = kv_heads * head_dim;
    let v_token_elements = kv_heads * value_dim;
    let output_elements = q_heads * value_dim;
    let mut step_outputs = Vec::with_capacity(cache_len * output_elements);
    let mut step_output_max_abs_diff = 0.0_f32;

    for timestep in 0..cache_len {
        let q_start = timestep * q_token_elements;
        let q_end = q_start + q_token_elements;
        let k_start = timestep * k_token_elements;
        let k_end = k_start + k_token_elements;
        let v_start = timestep * v_token_elements;
        let v_end = v_start + v_token_elements;
        let step = state
            .step(
                stream,
                &q_sequence[q_start..q_end],
                &logical_k_cache[k_start..k_end],
                &logical_v_cache[v_start..v_end],
            )
            .map_err(|err| format!("failed to run {label} timestep {timestep}: {err}"))?;
        if step.cache_position != timestep {
            return Err(format!(
                "{label} paged decode state wrote position {}, expected {timestep}",
                step.cache_position
            ));
        }
        if step.cache_len != timestep + 1 {
            return Err(format!(
                "{label} paged decode state reported cache_len {}, expected {}",
                step.cache_len,
                timestep + 1
            ));
        }
        let expected_step_output = runtime_host_paged_decode_attn_f32(
            &q_sequence[q_start..q_end],
            &expected_k_cache,
            &expected_v_cache,
            block_table,
            timestep + 1,
            block_size,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
        );
        let step_max_abs_diff = verify_f32_close(
            &format!("{label} timestep {timestep} paged decode step"),
            &step.attention_output,
            &expected_step_output,
            1e-4_f32,
            1e-4_f32,
        )?;
        step_output_max_abs_diff = step_output_max_abs_diff.max(step_max_abs_diff);
        step_outputs.extend_from_slice(&step.attention_output);
    }

    let readback = state
        .read_cache_to_host(stream)
        .map_err(|err| format!("failed to read {label} paged cache: {err}"))?;
    let k_write_max_abs_diff = verify_f32_close(
        &format!("{label} paged k cache write"),
        &readback.k,
        &expected_k_cache,
        1e-5_f32,
        1e-5_f32,
    )?;
    let v_write_max_abs_diff = verify_f32_close(
        &format!("{label} paged v cache write"),
        &readback.v,
        &expected_v_cache,
        1e-5_f32,
        1e-5_f32,
    )?;

    let output_start = (cache_len - 1) * output_elements;
    let output_end = output_start + output_elements;
    let output = step_outputs[output_start..output_end].to_vec();
    let expected_output = runtime_host_paged_decode_attn_f32(
        &q_sequence[(cache_len - 1) * q_token_elements..cache_len * q_token_elements],
        &expected_k_cache,
        &expected_v_cache,
        block_table,
        cache_len,
        block_size,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
    );
    let output_max_abs_diff =
        verify_f32_close(label, &output, &expected_output, 1e-4_f32, 1e-4_f32)?;

    Ok(RuntimePagedKvWriteDecodeResult {
        output,
        step_outputs,
        output_max_abs_diff,
        step_output_max_abs_diff,
        k_cache: readback.k,
        v_cache: readback.v,
        k_write_max_abs_diff,
        v_write_max_abs_diff,
    })
}

fn runtime_matvec_sequence_to_host_f32(
    context: &mut ullm_runtime_sys::RuntimeContext,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    matrix: &ullm_runtime_sys::RuntimeBuffer,
    matrix_host: &[f32],
    input_sequence: &[f32],
    sequence_len: usize,
    rows: usize,
    cols: usize,
    label: &str,
) -> Result<(Vec<f32>, f32), String> {
    if input_sequence.len() != sequence_len * cols {
        return Err(format!(
            "{label} input sequence length {} does not match sequence_len={sequence_len} cols={cols}",
            input_sequence.len()
        ));
    }
    if matrix_host.len() != rows * cols {
        return Err(format!(
            "{label} host matrix length {} does not match rows={rows} cols={cols}",
            matrix_host.len()
        ));
    }
    let input_bytes = cols
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| format!("{label} input byte size overflows"))?;
    let output_bytes = rows
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| format!("{label} output byte size overflows"))?;
    let mut input_buffer = context
        .alloc_buffer(input_bytes)
        .map_err(|err| format!("failed to allocate {label} input buffer: {err}"))?;
    let mut output_buffer = context
        .alloc_buffer(output_bytes)
        .map_err(|err| format!("failed to allocate {label} output buffer: {err}"))?;
    let mut output = Vec::with_capacity(sequence_len * rows);
    let mut expected = Vec::with_capacity(sequence_len * rows);
    let mut output_step_bytes = vec![0_u8; output_bytes];
    for timestep in 0..sequence_len {
        let element_start = timestep * cols;
        let element_end = element_start + cols;
        let step_input = &input_sequence[element_start..element_end];
        let step_input_bytes = encode_f32_to_bytes(step_input);
        input_buffer
            .copy_from_host(0, &step_input_bytes, Some(stream))
            .map_err(|err| format!("failed to copy {label} timestep {timestep} input: {err}"))?;
        ullm_runtime_sys::matvec_f32(
            matrix,
            &input_buffer,
            rows,
            cols,
            &mut output_buffer,
            Some(stream),
        )
        .map_err(|err| format!("failed to run {label} timestep {timestep} matvec: {err}"))?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize after {label} timestep {timestep}: {err}")
        })?;
        output_buffer
            .copy_to_host(0, &mut output_step_bytes, Some(stream))
            .map_err(|err| format!("failed to copy {label} timestep {timestep} output: {err}"))?;
        stream.synchronize().map_err(|err| {
            format!("failed to synchronize after {label} timestep {timestep} output copy: {err}")
        })?;
        output.extend(decode_f32_le_values(&output_step_bytes));
        expected.extend(runtime_host_matvec_f32(matrix_host, step_input, rows, cols));
    }
    let max_abs_diff = verify_f32_close(label, &output, &expected, 3e-3_f32, 2e-5_f32)?;
    Ok((output, max_abs_diff))
}

fn verify_f32_close(
    label: &str,
    actual: &[f32],
    expected: &[f32],
    abs_floor: f32,
    rel_scale: f32,
) -> Result<f32, String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "{label} size mismatch: expected {} got {}",
            expected.len(),
            actual.len()
        ));
    }
    let mut max_abs_diff = 0.0_f32;
    for (actual_value, expected_value) in actual.iter().zip(expected.iter()) {
        let diff = (actual_value - expected_value).abs();
        let tolerance = abs_floor.max(expected_value.abs() * rel_scale);
        if diff > tolerance {
            return Err(format!(
                "{label} mismatch: max_abs_diff={diff} tolerance={tolerance}"
            ));
        }
        if diff > max_abs_diff {
            max_abs_diff = diff;
        }
    }
    Ok(max_abs_diff)
}

fn deterministic_f32_vector(elements: usize) -> Vec<f32> {
    let mut values = Vec::with_capacity(elements);
    for index in 0..elements {
        values.push(((index as f32).sin() + 1.0_f32) / 2.0_f32);
    }
    values
}

fn linear_attn_step_input(base_input: &[f32], timestep: usize) -> Vec<f32> {
    base_input
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let phase = (index % 17) as f32 - 8.0_f32;
            *value + (timestep as f32) * phase * 0.00025_f32
        })
        .collect()
}

fn deterministic_linear_attn_core_output(
    sequence_len: usize,
    value_heads: usize,
    value_dim: usize,
) -> Vec<f32> {
    let elements = sequence_len * value_heads * value_dim;
    let mut values = Vec::with_capacity(elements);
    for index in 0..elements {
        let head_phase = ((index / value_dim) % value_heads) as f32 * 0.0007_f32;
        let dim_phase = (index % value_dim) as f32 * 0.00011_f32;
        values.push(((index as f32 * 0.013_f32).sin() * 0.05_f32) + head_phase - dim_phase);
    }
    values
}

struct LinearAttnQkvSplit {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
fn split_linear_attn_qkv_for_recurrent(
    conv_output: &[f32],
    sequence_len: usize,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    qk_l2_norm: bool,
    q_scale: f32,
) -> Result<LinearAttnQkvSplit, String> {
    if sequence_len == 0 || key_heads == 0 || value_heads == 0 || key_dim == 0 || value_dim == 0 {
        return Err("linear attention q/k/v layout contains a zero dimension".to_string());
    }
    let q_elements_per_step = key_heads
        .checked_mul(key_dim)
        .ok_or_else(|| "q element count overflows".to_string())?;
    let k_elements_per_step = q_elements_per_step;
    let v_elements_per_step = value_heads
        .checked_mul(value_dim)
        .ok_or_else(|| "v element count overflows".to_string())?;
    let step_elements = q_elements_per_step
        .checked_add(k_elements_per_step)
        .and_then(|value| value.checked_add(v_elements_per_step))
        .ok_or_else(|| "linear attention q/k/v step element count overflows".to_string())?;
    let expected_elements = step_elements
        .checked_mul(sequence_len)
        .ok_or_else(|| "linear attention q/k/v sequence element count overflows".to_string())?;
    if conv_output.len() != expected_elements {
        return Err(format!(
            "conv output element count mismatch: expected {expected_elements} got {}",
            conv_output.len()
        ));
    }

    let mut q = vec![0.0_f32; sequence_len * q_elements_per_step];
    let mut k = vec![0.0_f32; sequence_len * k_elements_per_step];
    let mut v = vec![0.0_f32; sequence_len * v_elements_per_step];
    for timestep in 0..sequence_len {
        let step_base = timestep * step_elements;
        let q_base = step_base;
        let k_base = q_base + q_elements_per_step;
        let v_base = k_base + k_elements_per_step;

        for head in 0..key_heads {
            let source_start = q_base + head * key_dim;
            let target_start = (timestep * key_heads + head) * key_dim;
            q[target_start..target_start + key_dim]
                .copy_from_slice(&conv_output[source_start..source_start + key_dim]);
            if qk_l2_norm {
                let norm = (q[target_start..target_start + key_dim]
                    .iter()
                    .map(|value| value * value)
                    .sum::<f32>()
                    + 1e-6_f32)
                    .sqrt();
                for value in &mut q[target_start..target_start + key_dim] {
                    *value = (*value / norm) * q_scale;
                }
            } else {
                for value in &mut q[target_start..target_start + key_dim] {
                    *value *= q_scale;
                }
            }

            let source_start = k_base + head * key_dim;
            let target_start = (timestep * key_heads + head) * key_dim;
            k[target_start..target_start + key_dim]
                .copy_from_slice(&conv_output[source_start..source_start + key_dim]);
            if qk_l2_norm {
                let norm = (k[target_start..target_start + key_dim]
                    .iter()
                    .map(|value| value * value)
                    .sum::<f32>()
                    + 1e-6_f32)
                    .sqrt();
                for value in &mut k[target_start..target_start + key_dim] {
                    *value /= norm;
                }
            }
        }

        let target_v_base = timestep * v_elements_per_step;
        v[target_v_base..target_v_base + v_elements_per_step]
            .copy_from_slice(&conv_output[v_base..v_base + v_elements_per_step]);
    }
    Ok(LinearAttnQkvSplit { q, k, v })
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
        "usage: ullm-engine <inspect-devices|runtime-smoke|runtime-memory-smoke [DEVICE_INDEX]|runtime-stream-smoke [DEVICE_INDEX]|runtime-copy-smoke [DEVICE_INDEX]|runtime-rmsnorm-smoke [DEVICE_INDEX]|runtime-silu-mul-smoke [DEVICE_INDEX]|runtime-sigmoid-mul-smoke [DEVICE_INDEX]|runtime-add-smoke [DEVICE_INDEX]|runtime-rope-smoke [DEVICE_INDEX]|runtime-causal-attn-smoke [DEVICE_INDEX]|runtime-decode-attn-smoke [DEVICE_INDEX]|runtime-paged-decode-attn-smoke [DEVICE_INDEX]|runtime-paged-kv-write-smoke [DEVICE_INDEX]|runtime-kv-paged-decode-smoke [DEVICE_INDEX]|runtime-depthwise-conv1d-smoke [DEVICE_INDEX]|runtime-linear-attn-gate-beta-smoke [DEVICE_INDEX]|runtime-linear-attn-recurrent-smoke [DEVICE_INDEX]|runtime-mlp-smoke [DEVICE_INDEX]|inspect-package PATH|package-load-smoke PACKAGE_DIR [DEVICE_INDEX] [MAX_BYTES] [PAYLOAD_ROLE]|package-tensor-load-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [TENSOR_SELECTOR]|package-weight-register-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [TENSOR_SELECTOR]|package-weight-register-many-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [MAX_TENSORS]|package-materialize-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [TENSOR_SELECTOR]|package-mlp-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX]|package-materialize-matvec-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [TENSOR_SELECTOR]|package-rmsnorm-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [input|post]|package-rmsnorm-mlp-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [input|post]|package-mlp-block-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [input|post]|package-linear-attn-proj-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [a|b|qkv|z|out|all]|package-self-attn-proj-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [q|k|v|o|all]|package-self-attn-qk-norm-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX]|package-self-attn-rope-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [SEQUENCE_LEN] [ROTARY_DIM] [ROPE_BASE] [POSITION_OFFSET]|package-self-attn-attention-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [SEQUENCE_LEN] [ROTARY_DIM] [ROPE_BASE] [POSITION_OFFSET]|package-self-attn-decode-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [SEQUENCE_LEN] [ROTARY_DIM] [ROPE_BASE] [POSITION_OFFSET]|package-self-attn-block-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [SEQUENCE_LEN] [ROTARY_DIM] [ROPE_BASE] [POSITION_OFFSET]|package-self-attn-mlp-block-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [SEQUENCE_LEN] [ROTARY_DIM] [ROPE_BASE] [POSITION_OFFSET]|package-linear-attn-qkv-norm-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX]|package-linear-attn-conv1d-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [SEQUENCE_LEN]|package-linear-attn-gate-beta-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [SEQUENCE_LEN]|package-linear-attn-recurrent-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [SEQUENCE_LEN]|package-linear-attn-post-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [SEQUENCE_LEN]|package-linear-attn-workflow-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [SEQUENCE_LEN]|package-linear-attn-block-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [SEQUENCE_LEN]|package-linear-attn-mlp-block-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [SEQUENCE_LEN]|package-linear-attn-aux-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [a-log|dt-bias|conv1d|norm|all]|package-materialize-bench PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [TENSOR_SELECTOR] [REPEATS]>"
    );
    eprintln!("linear attention projection selector: a|b|qkv|z|out|all");
    eprintln!("self attention projection selector: q|k|v|o|all (alias: out for o)");
    eprintln!(
        "linear attention aux selector: a-log|dt-bias|conv1d|norm|all (aliases: a_log|alog|dt_bias)"
    );
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

fn format_u64_shape(shape: &[u64]) -> String {
    let rendered = shape
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(",");
    format!("[{rendered}]")
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

fn parse_optional_f32(value: Option<String>, default: f32, label: &str) -> Result<f32, ExitCode> {
    match value {
        Some(value) => match value.parse::<f32>() {
            Ok(value) if value.is_finite() => Ok(value),
            Ok(_) => {
                eprintln!("invalid {label}: value must be finite");
                Err(ExitCode::from(2))
            }
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

fn encode_u32_to_bytes(values: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
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

fn runtime_host_sigmoid_mul_f32(gate: &[f32], input: &[f32]) -> Vec<f32> {
    if gate.len() != input.len() {
        return Vec::new();
    }
    gate.iter()
        .zip(input.iter())
        .map(|(gate_value, input_value)| {
            let sigmoid = 1.0_f32 / (1.0_f32 + (-*gate_value).exp());
            sigmoid * *input_value
        })
        .collect()
}

struct SelfAttnQProjectionSplit {
    query: Vec<f32>,
    gate: Option<Vec<f32>>,
    q_heads: usize,
    layout: &'static str,
}

fn split_self_attn_q_projection(
    projected: &[f32],
    sequence_len: usize,
    q_rows: usize,
    hidden: usize,
    head_dim: usize,
) -> Result<SelfAttnQProjectionSplit, String> {
    if sequence_len == 0 || q_rows == 0 || hidden == 0 || head_dim == 0 {
        return Err("self-attn q projection split received a zero dimension".to_string());
    }
    let expected_len = sequence_len
        .checked_mul(q_rows)
        .ok_or_else(|| "self-attn q projection split length overflows".to_string())?;
    if projected.len() != expected_len {
        return Err(format!(
            "self-attn q projection length mismatch: got {}, expected {expected_len}",
            projected.len()
        ));
    }

    let qwen35_gated = hidden
        .checked_mul(2)
        .is_some_and(|gated_rows| gated_rows == q_rows)
        && q_rows.is_multiple_of(2 * head_dim);
    if qwen35_gated {
        let q_heads = q_rows / (2 * head_dim);
        if q_heads == 0 {
            return Err("self-attn gated q projection has zero heads".to_string());
        }
        let query_len = sequence_len
            .checked_mul(q_heads)
            .and_then(|value| value.checked_mul(head_dim))
            .ok_or_else(|| "self-attn gated q projection output length overflows".to_string())?;
        let mut query = Vec::with_capacity(query_len);
        let mut gate = Vec::with_capacity(query_len);
        for timestep in 0..sequence_len {
            let timestep_start = timestep * q_rows;
            for head in 0..q_heads {
                let head_start = timestep_start + head * 2 * head_dim;
                let query_start = head_start;
                let gate_start = head_start + head_dim;
                query.extend_from_slice(&projected[query_start..query_start + head_dim]);
                gate.extend_from_slice(&projected[gate_start..gate_start + head_dim]);
            }
        }
        return Ok(SelfAttnQProjectionSplit {
            query,
            gate: Some(gate),
            q_heads,
            layout: "qwen3.5-gated",
        });
    }

    if !q_rows.is_multiple_of(head_dim) {
        return Err(format!(
            "self-attn q rows must be a multiple of head_dim: q_rows={q_rows}, head_dim={head_dim}"
        ));
    }
    let q_heads = q_rows / head_dim;
    if q_heads == 0 {
        return Err("self-attn q projection has zero heads".to_string());
    }
    Ok(SelfAttnQProjectionSplit {
        query: projected.to_vec(),
        gate: None,
        q_heads,
        layout: "plain",
    })
}

fn runtime_host_add_f32(lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
    if lhs.len() != rhs.len() {
        return Vec::new();
    }
    lhs.iter()
        .zip(rhs.iter())
        .map(|(lhs_value, rhs_value)| lhs_value + rhs_value)
        .collect()
}

fn runtime_host_rope_f32(
    input: &[f32],
    sequence_len: usize,
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    position_offset: usize,
    rope_base: f32,
) -> Vec<f32> {
    if sequence_len == 0
        || heads == 0
        || head_dim == 0
        || rotary_dim == 0
        || rotary_dim > head_dim
        || !rotary_dim.is_multiple_of(2)
        || input.len() != sequence_len * heads * head_dim
    {
        return Vec::new();
    }
    let mut output = vec![0.0_f32; input.len()];
    let half = rotary_dim / 2;
    for timestep in 0..sequence_len {
        let position = (position_offset + timestep) as f32;
        for head in 0..heads {
            let base = (timestep * heads + head) * head_dim;
            for pair_dim in 0..half {
                let exponent = (2.0 * pair_dim as f32) / rotary_dim as f32;
                let theta = position / rope_base.powf(exponent);
                let c = theta.cos();
                let s = theta.sin();
                let first = input[base + pair_dim];
                let second = input[base + half + pair_dim];
                output[base + pair_dim] = first * c - second * s;
                output[base + half + pair_dim] = second * c + first * s;
            }
            output[base + rotary_dim..base + head_dim]
                .copy_from_slice(&input[base + rotary_dim..base + head_dim]);
        }
    }
    output
}

#[allow(clippy::too_many_arguments)]
fn runtime_host_causal_attn_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    sequence_len: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
) -> Vec<f32> {
    if sequence_len == 0
        || q_heads == 0
        || kv_heads == 0
        || head_dim == 0
        || value_dim == 0
        || !q_heads.is_multiple_of(kv_heads)
        || q.len() != sequence_len * q_heads * head_dim
        || k.len() != sequence_len * kv_heads * head_dim
        || v.len() != sequence_len * kv_heads * value_dim
    {
        return Vec::new();
    }
    let mut output = vec![0.0_f32; sequence_len * q_heads * value_dim];
    let q_per_kv = q_heads / kv_heads;
    for timestep in 0..sequence_len {
        for q_head in 0..q_heads {
            let kv_head = q_head / q_per_kv;
            let q_base = (timestep * q_heads + q_head) * head_dim;
            let mut scores = Vec::with_capacity(timestep + 1);
            for source_timestep in 0..=timestep {
                let k_base = (source_timestep * kv_heads + kv_head) * head_dim;
                let score = (0..head_dim)
                    .map(|dim| q[q_base + dim] * k[k_base + dim])
                    .sum::<f32>()
                    * softmax_scale;
                scores.push(score);
            }
            let max_score = scores
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, |max, score| max.max(score));
            let weights = scores
                .iter()
                .map(|score| (*score - max_score).exp())
                .collect::<Vec<_>>();
            let denominator = weights.iter().sum::<f32>();
            let output_base = (timestep * q_heads + q_head) * value_dim;
            for value in 0..value_dim {
                let mut weighted = 0.0_f32;
                for (source_timestep, weight) in weights.iter().enumerate() {
                    let v_index = (source_timestep * kv_heads + kv_head) * value_dim + value;
                    weighted += *weight * v[v_index];
                }
                output[output_base + value] = weighted / denominator;
            }
        }
    }
    output
}

#[allow(clippy::too_many_arguments)]
fn runtime_host_decode_attn_f32(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    cache_len: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
) -> Vec<f32> {
    if cache_len == 0
        || q_heads == 0
        || kv_heads == 0
        || head_dim == 0
        || value_dim == 0
        || !q_heads.is_multiple_of(kv_heads)
        || q.len() != q_heads * head_dim
        || k_cache.len() != cache_len * kv_heads * head_dim
        || v_cache.len() != cache_len * kv_heads * value_dim
    {
        return Vec::new();
    }
    let mut output = vec![0.0_f32; q_heads * value_dim];
    let q_per_kv = q_heads / kv_heads;
    for q_head in 0..q_heads {
        let kv_head = q_head / q_per_kv;
        let q_base = q_head * head_dim;
        let mut scores = Vec::with_capacity(cache_len);
        for source_timestep in 0..cache_len {
            let k_base = (source_timestep * kv_heads + kv_head) * head_dim;
            let score = (0..head_dim)
                .map(|dim| q[q_base + dim] * k_cache[k_base + dim])
                .sum::<f32>()
                * softmax_scale;
            scores.push(score);
        }
        let max_score = scores
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, |max, score| max.max(score));
        let weights = scores
            .iter()
            .map(|score| (*score - max_score).exp())
            .collect::<Vec<_>>();
        let denominator = weights.iter().sum::<f32>();
        let output_base = q_head * value_dim;
        for value in 0..value_dim {
            let mut weighted = 0.0_f32;
            for (source_timestep, weight) in weights.iter().enumerate() {
                let v_index = (source_timestep * kv_heads + kv_head) * value_dim + value;
                weighted += *weight * v_cache[v_index];
            }
            output[output_base + value] = weighted / denominator;
        }
    }
    output
}

#[allow(clippy::too_many_arguments)]
fn runtime_host_paged_decode_attn_f32(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    block_table: &[u32],
    cache_len: usize,
    block_size: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
) -> Vec<f32> {
    if cache_len == 0
        || block_size == 0
        || q_heads == 0
        || kv_heads == 0
        || head_dim == 0
        || value_dim == 0
        || !q_heads.is_multiple_of(kv_heads)
        || q.len() != q_heads * head_dim
        || block_table.len() < (cache_len - 1) / block_size + 1
    {
        return Vec::new();
    }
    let physical_tokens = k_cache.len() / (kv_heads * head_dim);
    if physical_tokens * kv_heads * head_dim != k_cache.len()
        || physical_tokens * kv_heads * value_dim != v_cache.len()
    {
        return Vec::new();
    }
    let mut output = vec![0.0_f32; q_heads * value_dim];
    let q_per_kv = q_heads / kv_heads;
    for q_head in 0..q_heads {
        let kv_head = q_head / q_per_kv;
        let q_base = q_head * head_dim;
        let mut scores = Vec::with_capacity(cache_len);
        for source_timestep in 0..cache_len {
            let block_index = source_timestep / block_size;
            let block_offset = source_timestep - block_index * block_size;
            let physical_timestep = block_table[block_index] as usize * block_size + block_offset;
            if physical_timestep >= physical_tokens {
                return Vec::new();
            }
            let k_base = (physical_timestep * kv_heads + kv_head) * head_dim;
            let score = (0..head_dim)
                .map(|dim| q[q_base + dim] * k_cache[k_base + dim])
                .sum::<f32>()
                * softmax_scale;
            scores.push(score);
        }
        let max_score = scores
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, |max, score| max.max(score));
        let weights = scores
            .iter()
            .map(|score| (*score - max_score).exp())
            .collect::<Vec<_>>();
        let denominator = weights.iter().sum::<f32>();
        let output_base = q_head * value_dim;
        for value in 0..value_dim {
            let mut weighted = 0.0_f32;
            for (source_timestep, weight) in weights.iter().enumerate() {
                let block_index = source_timestep / block_size;
                let block_offset = source_timestep - block_index * block_size;
                let physical_timestep =
                    block_table[block_index] as usize * block_size + block_offset;
                let v_index = (physical_timestep * kv_heads + kv_head) * value_dim + value;
                weighted += *weight * v_cache[v_index];
            }
            output[output_base + value] = weighted / denominator;
        }
    }
    output
}

#[allow(clippy::too_many_arguments)]
fn pack_paged_decode_kv_cache_for_block_table(
    logical_k_cache: &[f32],
    logical_v_cache: &[f32],
    block_table: &[u32],
    cache_len: usize,
    block_size: usize,
    cache_blocks: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    if cache_len == 0 {
        return Err("paged decode cache_len must be greater than zero".to_string());
    }
    if block_size == 0 {
        return Err("paged decode block_size must be greater than zero".to_string());
    }
    if kv_heads == 0 || head_dim == 0 || value_dim == 0 {
        return Err(format!(
            "paged decode dimensions must be nonzero: kv_heads={kv_heads} head_dim={head_dim} value_dim={value_dim}"
        ));
    }
    if logical_k_cache.len() != cache_len * kv_heads * head_dim {
        return Err(format!(
            "logical k cache length {} does not match cache_len={cache_len} kv_heads={kv_heads} head_dim={head_dim}",
            logical_k_cache.len()
        ));
    }
    if logical_v_cache.len() != cache_len * kv_heads * value_dim {
        return Err(format!(
            "logical v cache length {} does not match cache_len={cache_len} kv_heads={kv_heads} value_dim={value_dim}",
            logical_v_cache.len()
        ));
    }
    let block_table_entries = (cache_len - 1) / block_size + 1;
    if block_table.len() != block_table_entries {
        return Err(format!(
            "paged decode block table length {} does not match expected entries {block_table_entries}",
            block_table.len()
        ));
    }
    let physical_tokens = cache_blocks
        .checked_mul(block_size)
        .ok_or_else(|| "paged decode physical token count overflows".to_string())?;
    let mut physical_k_cache = vec![0.0_f32; physical_tokens * kv_heads * head_dim];
    let mut physical_v_cache = vec![0.0_f32; physical_tokens * kv_heads * value_dim];
    for (index, block_id) in block_table.iter().copied().enumerate() {
        if block_id as usize >= cache_blocks {
            return Err(format!(
                "paged decode block_table[{index}]={block_id} exceeds cache_blocks={cache_blocks}"
            ));
        }
    }
    for timestep in 0..cache_len {
        let logical_block = timestep / block_size;
        let block_offset = timestep - logical_block * block_size;
        let physical_block = block_table[logical_block] as usize;
        let physical_timestep = physical_block * block_size + block_offset;
        let logical_k_start = timestep * kv_heads * head_dim;
        let logical_k_end = logical_k_start + kv_heads * head_dim;
        let physical_k_start = physical_timestep * kv_heads * head_dim;
        let physical_k_end = physical_k_start + kv_heads * head_dim;
        physical_k_cache[physical_k_start..physical_k_end]
            .copy_from_slice(&logical_k_cache[logical_k_start..logical_k_end]);

        let logical_v_start = timestep * kv_heads * value_dim;
        let logical_v_end = logical_v_start + kv_heads * value_dim;
        let physical_v_start = physical_timestep * kv_heads * value_dim;
        let physical_v_end = physical_v_start + kv_heads * value_dim;
        physical_v_cache[physical_v_start..physical_v_end]
            .copy_from_slice(&logical_v_cache[logical_v_start..logical_v_end]);
    }
    Ok((physical_k_cache, physical_v_cache))
}

fn allocate_fragmented_paged_decode_blocks(
    cache_len: usize,
    block_size: usize,
) -> Result<(Vec<u32>, usize, KvBlockAllocatorStats), String> {
    if cache_len == 0 {
        return Err("paged decode cache_len must be greater than zero".to_string());
    }
    if block_size == 0 {
        return Err("paged decode block_size must be greater than zero".to_string());
    }
    if block_size > u32::MAX as usize {
        return Err(format!(
            "paged decode block_size={block_size} exceeds u32 block size range"
        ));
    }
    let block_count = (cache_len - 1) / block_size + 1;
    if block_count > u32::MAX as usize - 2 {
        return Err(format!(
            "paged decode block_count={block_count} is too large for allocator smoke"
        ));
    }
    let cache_blocks = block_count
        .checked_add(2)
        .ok_or_else(|| "paged decode cache_blocks overflows".to_string())?;
    let mut allocator = KvBlockAllocator::with_block_size(cache_blocks as u32, block_size as u32);
    let fragment_blocks = cache_blocks - 1;
    let fragment = allocator
        .allocate(RequestId(100), fragment_blocks)
        .map_err(|err| format!("failed to allocate fragmenting KV blocks: {err}"))?;
    let freed = allocator.free_request(fragment.request_id);
    if freed != fragment.blocks.len() {
        return Err(format!(
            "freed KV block count {freed} does not match allocated fragment blocks {}",
            fragment.blocks.len()
        ));
    }
    let allocation = allocator
        .allocate(RequestId(101), block_count)
        .map_err(|err| format!("failed to allocate decode KV blocks: {err}"))?;
    let stats = allocator.stats();
    Ok((allocation.blocks, cache_blocks, stats))
}

fn runtime_host_depthwise_conv1d_f32(
    input: &[f32],
    weight: &[f32],
    channels: usize,
    sequence_len: usize,
    kernel_size: usize,
) -> Vec<f32> {
    if channels == 0
        || sequence_len == 0
        || kernel_size == 0
        || input.len() != channels * sequence_len
        || weight.len() != channels * kernel_size
    {
        return Vec::new();
    }
    let mut output = vec![0.0_f32; channels * sequence_len];
    for timestep in 0..sequence_len {
        for channel in 0..channels {
            let mut value = 0.0_f32;
            for kernel in 0..kernel_size {
                if timestep < kernel {
                    break;
                }
                value += input[(timestep - kernel) * channels + channel]
                    * weight[channel * kernel_size + kernel];
            }
            output[timestep * channels + channel] = value;
        }
    }
    output
}

fn runtime_host_linear_attn_gate_beta_f32(
    a: &[f32],
    b: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    heads: usize,
    sequence_len: usize,
) -> (Vec<f32>, Vec<f32>) {
    let elements = match heads.checked_mul(sequence_len) {
        Some(value) if value > 0 => value,
        _ => return (Vec::new(), Vec::new()),
    };
    if a.len() != elements || b.len() != elements || a_log.len() != heads || dt_bias.len() != heads
    {
        return (Vec::new(), Vec::new());
    }

    let mut gate = vec![0.0_f32; elements];
    let mut beta = vec![0.0_f32; elements];
    for index in 0..elements {
        let head = index % heads;
        let x = a[index] + dt_bias[head];
        let softplus = if x <= 20.0_f32 {
            (1.0_f32 + x.exp()).ln()
        } else {
            x
        };
        gate[index] = -a_log[head].exp() * softplus;
        beta[index] = 1.0_f32 / (1.0_f32 + (-b[index]).exp());
    }
    (gate, beta)
}

#[allow(clippy::too_many_arguments)]
fn runtime_host_linear_attn_recurrent_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    beta: &[f32],
    key_heads: usize,
    value_heads: usize,
    sequence_len: usize,
    key_dim: usize,
    value_dim: usize,
    state: &mut [f32],
) -> Vec<f32> {
    let key_head_sequence_elements = match key_heads.checked_mul(sequence_len) {
        Some(value) if value > 0 => value,
        _ => return Vec::new(),
    };
    let value_head_sequence_elements = match value_heads.checked_mul(sequence_len) {
        Some(value) if value > 0 => value,
        _ => return Vec::new(),
    };
    let qk_elements = match key_head_sequence_elements.checked_mul(key_dim) {
        Some(value) => value,
        None => return Vec::new(),
    };
    let v_elements = match value_head_sequence_elements.checked_mul(value_dim) {
        Some(value) => value,
        None => return Vec::new(),
    };
    let state_elements = match value_heads
        .checked_mul(key_dim)
        .and_then(|value| value.checked_mul(value_dim))
    {
        Some(value) => value,
        None => return Vec::new(),
    };
    if key_heads == 0
        || value_heads == 0
        || !value_heads.is_multiple_of(key_heads)
        || key_dim == 0
        || value_dim == 0
        || q.len() != qk_elements
        || k.len() != qk_elements
        || v.len() != v_elements
        || gate.len() != value_head_sequence_elements
        || beta.len() != value_head_sequence_elements
        || state.len() != state_elements
    {
        return Vec::new();
    }

    let mut output = vec![0.0_f32; v_elements];
    let key_head_group = value_heads / key_heads;
    for timestep in 0..sequence_len {
        for value_head in 0..value_heads {
            let key_head = value_head / key_head_group;
            let value_head_index = timestep * value_heads + value_head;
            let key_head_index = timestep * key_heads + key_head;
            let qk_base = key_head_index * key_dim;
            let v_base = value_head_index * value_dim;
            let state_head_offset = value_head * key_dim * value_dim;
            let decay = gate[value_head_index].exp();
            let beta_value = beta[value_head_index];

            for key in 0..key_dim {
                let state_key_offset = state_head_offset + key * value_dim;
                for value in 0..value_dim {
                    state[state_key_offset + value] *= decay;
                }
            }

            for value in 0..value_dim {
                let mut current = 0.0_f32;
                for key in 0..key_dim {
                    current +=
                        state[state_head_offset + key * value_dim + value] * k[qk_base + key];
                }
                let v_prime = (v[v_base + value] - current) * beta_value;
                for key in 0..key_dim {
                    state[state_head_offset + key * value_dim + value] +=
                        k[qk_base + key] * v_prime;
                }
            }

            for value in 0..value_dim {
                let mut sum = 0.0_f32;
                for key in 0..key_dim {
                    sum += state[state_head_offset + key * value_dim + value] * q[qk_base + key];
                }
                output[v_base + value] = sum;
            }
        }
    }
    output
}

fn format_f32_preview(values: &[f32]) -> String {
    let joined = values
        .iter()
        .map(|value| format!("{value:.7}"))
        .collect::<Vec<_>>()
        .join(",");
    format!("[{joined}]")
}
