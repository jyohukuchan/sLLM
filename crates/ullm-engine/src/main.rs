// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

use std::env;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::process::ExitCode;
use std::time::Instant;
use ullm_engine::decode_runner::{
    Qwen3DecoderLayerDecodeBatchInput, Qwen3DecoderLayerDecodeInputLayout,
    Qwen3DecoderLayerDecodeSequenceView, Qwen3DecoderLayerRequestDecodeRunner,
    Qwen3DecoderLayerStackRequestDecodeRunner, Qwen3SelfAttnDecodeBatchInput,
    Qwen3SelfAttnRequestDecodeRunner, qwen3_decoder_layer_decode_batch_inputs_from_sequences,
    qwen3_decoder_layer_prefill_input_from_sequence,
};
use ullm_engine::decoder::{
    PagedDecodeShape, PagedKvCacheReadback, Qwen3DecoderLayerRuntimeWeights,
    Qwen3DecoderLayerSequenceOutput, Qwen3MlpRuntimeWeights, Qwen3PostAttentionRuntimeWeights,
    Qwen3SelfAttnRuntimePreparedSequence, Qwen3SelfAttnRuntimePreparedSequenceForPagedDecode,
    Qwen3SelfAttnRuntimeShape, Qwen3SelfAttnRuntimeWeights, pack_paged_kv_cache_for_block_table,
    qwen3_causal_attn_to_host_f32, qwen3_decoder_layer_sequence_to_host_f32,
    qwen3_headwise_rmsnorm_to_host_f32, qwen3_rope_to_host_f32,
    qwen3_self_attn_block_sequence_to_host_f32,
    qwen3_self_attn_prepare_sequence_for_paged_decode_f32, qwen3_self_attn_runtime_shape,
    split_qwen3_self_attn_q_projection,
};
use ullm_engine::golden::{GoldenTensorFixture, compare_f32_slices};
use ullm_engine::host_bytes::{decode_f32_le_values, encode_f32_to_bytes, encode_u32_to_bytes};
use ullm_engine::loader::{
    LoadOptions, LoadedPayload, PassthroughF32Data, WeightRegistry,
    effective_rmsnorm_weight_values, load_package_tensor_prefix, materialize_config,
    materialize_selected_aq4_matrix, matrix_shape_rows_cols, read_named_passthrough_f32,
    read_passthrough_payload_f32_bytes, resolve_passthrough_dtype,
    validate_passthrough_shape_elements,
};
use ullm_engine::package::{
    ReferencedFile, ReferencedFileRole, TensorSelector, list_tensor_payload_bundles,
    select_tensor_payload_bundle,
};
use ullm_engine::qwen3_loader::{
    Qwen3PackageModelDecodePlan, Qwen3PackageModelRuntime, Qwen3PackageModelStackRequest,
    qwen3_decoder_layer_runtime_weights_from_package,
    qwen3_package_decoder_layer_runtime_from_package,
    qwen3_package_model_run_prefill_step_from_sequence,
    qwen3_package_model_run_ready_batch_from_sequences, qwen3_package_model_stack_runner,
    qwen3_self_attn_runtime_weights_from_package,
};
use ullm_engine::scheduler::{
    KvBlockAllocator, KvBlockAllocatorStats, Request, RequestId, SchedulerDecodeRequest,
    SchedulerState,
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
        Some("runtime-sigmoid-mul-smoke") => runtime_sigmoid_mul_smoke(env::args().nth(2)),
        Some("runtime-add-smoke") => runtime_add_smoke(env::args().nth(2)),
        Some("runtime-rope-smoke") => runtime_rope_smoke(env::args().nth(2)),
        Some("runtime-causal-attn-smoke") => runtime_causal_attn_smoke(env::args().nth(2)),
        Some("runtime-decode-attn-smoke") => runtime_decode_attn_smoke(env::args().nth(2)),
        Some("runtime-paged-decode-attn-smoke") => {
            runtime_paged_decode_attn_smoke(env::args().nth(2))
        }
        Some("runtime-paged-kv-write-smoke") => runtime_paged_kv_write_smoke(env::args().nth(2)),
        Some("runtime-scheduler-paged-decode-smoke") => {
            runtime_scheduler_paged_decode_smoke(env::args().nth(2))
        }
        Some("runtime-scheduler-layer-decode-smoke") => {
            runtime_scheduler_layer_decode_smoke(env::args().nth(2))
        }
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
        Some("package-self-attn-mlp-block-scheduler-smoke") => {
            package_self_attn_mlp_block_scheduler_smoke(
                env::args().nth(2),
                env::args().nth(3),
                env::args().nth(4),
                env::args().nth(5),
                env::args().nth(6),
                env::args().nth(7),
                env::args().nth(8),
                env::args().nth(9),
            )
        }
        Some("package-self-attn-mlp-block-model-loop-smoke") => {
            package_self_attn_mlp_block_model_loop_smoke(
                env::args().nth(2),
                env::args().nth(3),
                env::args().nth(4),
                env::args().nth(5),
                env::args().nth(6),
                env::args().nth(7),
                env::args().nth(8),
                env::args().nth(9),
                env::args().nth(10),
            )
        }
        Some("package-layer-golden-smoke") => package_layer_golden_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
            env::args().nth(7),
            env::args().nth(8),
            env::args().nth(9),
        ),
        Some("package-golden-prefix-smoke") => package_golden_prefix_smoke(
            env::args().nth(2),
            env::args().nth(3),
            env::args().nth(4),
            env::args().nth(5),
            env::args().nth(6),
            env::args().nth(7),
            env::args().nth(8),
            env::args().nth(9),
            env::args().nth(10),
            env::args().nth(11),
            env::args().nth(12),
            env::args().nth(13),
            env::args().nth(14),
            env::args().nth(15),
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
    let scheduled = match allocate_fragmented_paged_decode_blocks(cache_len, block_size) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let ScheduledPagedDecodeBlocks {
        block_table,
        cache_blocks,
        allocator_stats: stats,
        request_id: scheduler_request_id,
        prefill_tokens: scheduler_prefill_tokens,
        max_new_tokens: scheduler_max_new_tokens,
        cached_tokens: scheduler_cached_tokens,
        generated_tokens: scheduler_generated_tokens,
        active_len: scheduler_active_len,
    } = scheduled;
    let kv_heads = 2_usize;
    let head_dim = 3_usize;
    let value_dim = 2_usize;
    let logical_k = (0..cache_len * kv_heads * head_dim)
        .map(|index| ((index * 3) as f32 - 7.0) / 13.0)
        .collect::<Vec<_>>();
    let logical_v = (0..cache_len * kv_heads * value_dim)
        .map(|index| ((index * 5) as f32 - 9.0) / 17.0)
        .collect::<Vec<_>>();
    let shape = PagedDecodeShape {
        block_size,
        cache_blocks,
        q_heads: kv_heads,
        kv_heads,
        head_dim,
        value_dim,
    };
    let PagedKvCacheReadback {
        k: expected_k_cache,
        v: expected_v_cache,
    } = match pack_paged_kv_cache_for_block_table(
        &logical_k,
        &logical_v,
        &block_table,
        cache_len,
        shape,
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
        "runtime-paged-kv-write-smoke backend={} device_index={} name=\"{}\" cache_len={} block_size={} cache_blocks={} block_table={:?} scheduler_request_id={} scheduler_prefill_tokens={} scheduler_max_new_tokens={} scheduler_cached_tokens={} scheduler_generated_tokens={} scheduler_active_len={} free_blocks={} allocated_block_count={} free_runs={} largest_free_run={} kv_heads={} head_dim={} value_dim={} k_cache_preview={} v_cache_preview={} k_max_abs_diff={k_max_abs_diff:.9} v_max_abs_diff={v_max_abs_diff:.9} verified=true",
        info.backend,
        device_index,
        info.name,
        cache_len,
        block_size,
        cache_blocks,
        block_table,
        scheduler_request_id.0,
        scheduler_prefill_tokens,
        scheduler_max_new_tokens,
        scheduler_cached_tokens,
        scheduler_generated_tokens,
        scheduler_active_len,
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

struct SyntheticSchedulerPagedDecodeRun {
    request_id: RequestId,
    prompt_tokens: usize,
    max_new_tokens: usize,
    total_tokens: usize,
    block_table: Vec<u32>,
    q_sequence: Vec<f32>,
    k_sequence: Vec<f32>,
    v_sequence: Vec<f32>,
    expected_k_cache: Vec<f32>,
    expected_v_cache: Vec<f32>,
    decode_steps: usize,
    attention_max_abs_diff: f32,
    k_cache_max_abs_diff: f32,
    v_cache_max_abs_diff: f32,
}

struct SchedulerLayerDecodeState {
    request_id: RequestId,
    prompt_tokens: usize,
    max_new_tokens: usize,
    total_tokens: usize,
    block_table: Vec<u32>,
    q_sequence: Vec<f32>,
    k_sequence: Vec<f32>,
    v_sequence: Vec<f32>,
    output_gate_sequence: Option<Vec<f32>>,
    residual_sequence: Vec<f32>,
    decode_steps: usize,
}

struct SchedulerLayerDecodeRun {
    state: SchedulerLayerDecodeState,
    checks: SchedulerLayerDecodeSmokeChecks,
}

impl std::ops::Deref for SchedulerLayerDecodeRun {
    type Target = SchedulerLayerDecodeState;

    fn deref(&self) -> &Self::Target {
        &self.state
    }
}

impl std::ops::DerefMut for SchedulerLayerDecodeRun {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.state
    }
}

struct SchedulerLayerDecodeSmokeChecks {
    expected: Qwen3DecoderLayerSequenceOutput,
    attention_max_abs_diff: f32,
    projection_input_max_abs_diff: f32,
    projected_max_abs_diff: f32,
    block_max_abs_diff: f32,
    post_norm_max_abs_diff: f32,
    mlp_max_abs_diff: f32,
    layer_max_abs_diff: f32,
    k_cache_max_abs_diff: f32,
    v_cache_max_abs_diff: f32,
}

impl SchedulerLayerDecodeSmokeChecks {
    fn new(expected: Qwen3DecoderLayerSequenceOutput) -> Self {
        Self {
            expected,
            attention_max_abs_diff: 0.0,
            projection_input_max_abs_diff: 0.0,
            projected_max_abs_diff: 0.0,
            block_max_abs_diff: 0.0,
            post_norm_max_abs_diff: 0.0,
            mlp_max_abs_diff: 0.0,
            layer_max_abs_diff: 0.0,
            k_cache_max_abs_diff: 0.0,
            v_cache_max_abs_diff: 0.0,
        }
    }
}

fn synthetic_scheduler_decode_values(
    request_index: usize,
    total_tokens: usize,
    token_elements: usize,
    salt: usize,
) -> Vec<f32> {
    let mut values = Vec::with_capacity(total_tokens * token_elements);
    for token in 0..total_tokens {
        for element in 0..token_elements {
            let angle = (request_index as f32 + 1.0_f32) * 0.173_f32
                + token as f32 * 0.119_f32
                + element as f32 * 0.037_f32
                + salt as f32 * 0.071_f32;
            values.push(angle.sin() * 0.25_f32 + angle.cos() * 0.05_f32);
        }
    }
    values
}

fn scheduler_layer_decode_run(
    runs: &[SchedulerLayerDecodeRun],
    request_id: RequestId,
) -> Option<&SchedulerLayerDecodeRun> {
    runs.iter().find(|run| run.request_id == request_id)
}

fn scheduler_layer_decode_run_mut(
    runs: &mut [SchedulerLayerDecodeRun],
    request_id: RequestId,
) -> Option<&mut SchedulerLayerDecodeRun> {
    runs.iter_mut().find(|run| run.request_id == request_id)
}

fn scheduler_layer_decode_sequence_view(
    run: &SchedulerLayerDecodeRun,
) -> Qwen3DecoderLayerDecodeSequenceView<'_> {
    Qwen3DecoderLayerDecodeSequenceView {
        request_id: run.request_id,
        q_sequence: &run.q_sequence,
        k_sequence: &run.k_sequence,
        v_sequence: &run.v_sequence,
        output_gate_sequence: run.output_gate_sequence.as_deref(),
        residual_sequence: &run.residual_sequence,
    }
}

fn runtime_f32_buffer_from_values(
    context: &mut ullm_runtime_sys::RuntimeContext,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    label: &str,
    values: &[f32],
) -> Result<ullm_runtime_sys::RuntimeBuffer, String> {
    let mut buffer = context
        .alloc_buffer(
            values
                .len()
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| format!("{label} byte size overflows"))?,
        )
        .map_err(|err| format!("failed to allocate {label}: {err}"))?;
    buffer
        .copy_from_host(0, &encode_f32_to_bytes(values), Some(stream))
        .map_err(|err| format!("failed to copy {label}: {err}"))?;
    Ok(buffer)
}

fn synthetic_scheduler_decoder_layer_weights(
    context: &mut ullm_runtime_sys::RuntimeContext,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    shape: PagedDecodeShape,
    hidden: usize,
    intermediate: usize,
) -> Result<Qwen3DecoderLayerRuntimeWeights, String> {
    if hidden == 0 || intermediate == 0 {
        return Err("synthetic decoder layer hidden/intermediate must be nonzero".to_string());
    }
    let q_rows = shape.q_elements()?;
    let k_rows = shape.k_token_elements()?;
    let v_rows = shape.v_token_elements()?;
    let attention_elements = shape.output_elements()?;
    let q_matrix = synthetic_scheduler_decode_values(0, q_rows, hidden, 11);
    let k_matrix = synthetic_scheduler_decode_values(0, k_rows, hidden, 12);
    let v_matrix = synthetic_scheduler_decode_values(0, v_rows, hidden, 13);
    let o_matrix = synthetic_scheduler_decode_values(0, hidden, attention_elements, 14);
    let post_norm_weight = (0..hidden)
        .map(|index| 0.75_f32 + index as f32 * 0.03125_f32)
        .collect::<Vec<_>>();
    let mlp_gate_matrix = synthetic_scheduler_decode_values(0, intermediate, hidden, 15);
    let mlp_up_matrix = synthetic_scheduler_decode_values(0, intermediate, hidden, 16);
    let mlp_down_matrix = synthetic_scheduler_decode_values(0, hidden, intermediate, 17);

    let weights = Qwen3DecoderLayerRuntimeWeights {
        self_attn: Qwen3SelfAttnRuntimeWeights {
            q_rows,
            q_cols: hidden,
            k_rows,
            v_rows,
            o_rows: hidden,
            o_cols: attention_elements,
            head_dim: shape.head_dim,
            kv_heads: shape.kv_heads,
            value_dim: shape.value_dim,
            q_matrix: runtime_f32_buffer_from_values(
                context,
                stream,
                "synthetic Qwen3 decoder layer q matrix",
                &q_matrix,
            )?,
            k_matrix: runtime_f32_buffer_from_values(
                context,
                stream,
                "synthetic Qwen3 decoder layer k matrix",
                &k_matrix,
            )?,
            v_matrix: runtime_f32_buffer_from_values(
                context,
                stream,
                "synthetic Qwen3 decoder layer v matrix",
                &v_matrix,
            )?,
            o_matrix: runtime_f32_buffer_from_values(
                context,
                stream,
                "synthetic Qwen3 decoder layer o matrix",
                &o_matrix,
            )?,
        },
        post_attention: Qwen3PostAttentionRuntimeWeights {
            hidden,
            intermediate,
            post_norm_weight: runtime_f32_buffer_from_values(
                context,
                stream,
                "synthetic Qwen3 decoder layer post norm weight",
                &post_norm_weight,
            )?,
            mlp: Qwen3MlpRuntimeWeights {
                gate_rows: intermediate,
                gate_cols: hidden,
                gate_matrix: runtime_f32_buffer_from_values(
                    context,
                    stream,
                    "synthetic Qwen3 decoder layer MLP gate matrix",
                    &mlp_gate_matrix,
                )?,
                up_matrix: runtime_f32_buffer_from_values(
                    context,
                    stream,
                    "synthetic Qwen3 decoder layer MLP up matrix",
                    &mlp_up_matrix,
                )?,
                down_matrix: runtime_f32_buffer_from_values(
                    context,
                    stream,
                    "synthetic Qwen3 decoder layer MLP down matrix",
                    &mlp_down_matrix,
                )?,
            },
        },
    };
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize synthetic decoder layer weights: {err}"))?;
    Ok(weights)
}

fn synthetic_scheduler_decode_run(
    runs: &[SyntheticSchedulerPagedDecodeRun],
    request_id: RequestId,
) -> Option<&SyntheticSchedulerPagedDecodeRun> {
    runs.iter().find(|run| run.request_id == request_id)
}

fn synthetic_scheduler_decode_run_mut(
    runs: &mut [SyntheticSchedulerPagedDecodeRun],
    request_id: RequestId,
) -> Option<&mut SyntheticSchedulerPagedDecodeRun> {
    runs.iter_mut().find(|run| run.request_id == request_id)
}

#[allow(clippy::too_many_arguments)]
fn run_synthetic_scheduler_decode_step(
    runner: &mut Qwen3SelfAttnRequestDecodeRunner,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    run: &mut SyntheticSchedulerPagedDecodeRun,
    timestep: usize,
    block_size: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
    label: &str,
) -> Result<(), String> {
    let q_token_elements = q_heads
        .checked_mul(head_dim)
        .ok_or_else(|| format!("{label} q token element count overflows"))?;
    let k_token_elements = kv_heads
        .checked_mul(head_dim)
        .ok_or_else(|| format!("{label} k token element count overflows"))?;
    let v_token_elements = kv_heads
        .checked_mul(value_dim)
        .ok_or_else(|| format!("{label} v token element count overflows"))?;
    let q_start = timestep
        .checked_mul(q_token_elements)
        .ok_or_else(|| format!("{label} q slice start overflows"))?;
    let q_end = q_start
        .checked_add(q_token_elements)
        .ok_or_else(|| format!("{label} q slice end overflows"))?;
    let k_start = timestep
        .checked_mul(k_token_elements)
        .ok_or_else(|| format!("{label} k slice start overflows"))?;
    let k_end = k_start
        .checked_add(k_token_elements)
        .ok_or_else(|| format!("{label} k slice end overflows"))?;
    let v_start = timestep
        .checked_mul(v_token_elements)
        .ok_or_else(|| format!("{label} v slice start overflows"))?;
    let v_end = v_start
        .checked_add(v_token_elements)
        .ok_or_else(|| format!("{label} v slice end overflows"))?;

    let step = runner
        .run_prefill_step(
            stream,
            Qwen3SelfAttnDecodeBatchInput {
                request_id: run.request_id,
                q: &run.q_sequence[q_start..q_end],
                k: &run.k_sequence[k_start..k_end],
                v: &run.v_sequence[v_start..v_end],
            },
        )
        .map_err(|err| {
            format!(
                "{label} failed to run request {:?} timestep {timestep}: {err}",
                run.request_id
            )
        })?;
    if step.cache_position != timestep {
        return Err(format!(
            "{label} request {:?} wrote cache position {}, expected {timestep}",
            run.request_id, step.cache_position
        ));
    }
    if step.cache_len != timestep + 1 {
        return Err(format!(
            "{label} request {:?} reported cache_len {}, expected {}",
            run.request_id,
            step.cache_len,
            timestep + 1
        ));
    }

    let expected = runtime_host_paged_decode_attn_f32(
        &run.q_sequence[q_start..q_end],
        &run.expected_k_cache,
        &run.expected_v_cache,
        &run.block_table,
        timestep + 1,
        block_size,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
    );
    let max_abs_diff = verify_f32_close(
        &format!("{label} request {:?} timestep {timestep}", run.request_id),
        &step.attention_output,
        &expected,
        1e-4,
        1e-4,
    )?;
    run.attention_max_abs_diff = run.attention_max_abs_diff.max(max_abs_diff);
    Ok(())
}

fn synthetic_layer_expected_slice<'a>(
    values: &'a [f32],
    timestep: usize,
    elements: usize,
    label: &str,
) -> Result<&'a [f32], String> {
    let start = timestep
        .checked_mul(elements)
        .ok_or_else(|| format!("{label} slice start overflows"))?;
    let end = start
        .checked_add(elements)
        .ok_or_else(|| format!("{label} slice end overflows"))?;
    values
        .get(start..end)
        .ok_or_else(|| format!("{label} slice {start}..{end} is out of bounds"))
}

fn verify_scheduler_layer_step_output(
    label: &str,
    run: &mut SchedulerLayerDecodeRun,
    step: &ullm_engine::decode_runner::Qwen3DecoderLayerDecodeBatchOutput,
    hidden: usize,
    attention_elements: usize,
) -> Result<(), String> {
    let timestep = step.cache_position;
    if step.cache_len != timestep + 1 {
        return Err(format!(
            "{label} request {:?} cache_len {} did not match timestep + 1 ({})",
            run.request_id,
            step.cache_len,
            timestep + 1
        ));
    }
    let attention_expected = synthetic_layer_expected_slice(
        &run.checks.expected.attention_output,
        timestep,
        attention_elements,
        label,
    )?;
    let projection_input_expected = synthetic_layer_expected_slice(
        &run.checks.expected.attention_projection_input,
        timestep,
        attention_elements,
        label,
    )?;
    let projected_expected = synthetic_layer_expected_slice(
        &run.checks.expected.projected_output,
        timestep,
        hidden,
        label,
    )?;
    let block_expected =
        synthetic_layer_expected_slice(&run.checks.expected.block_output, timestep, hidden, label)?;
    let post_norm_expected =
        synthetic_layer_expected_slice(&run.checks.expected.post_normed, timestep, hidden, label)?;
    let mlp_expected =
        synthetic_layer_expected_slice(&run.checks.expected.mlp_output, timestep, hidden, label)?;
    let layer_expected =
        synthetic_layer_expected_slice(&run.checks.expected.layer_output, timestep, hidden, label)?;

    run.checks.attention_max_abs_diff = run.checks.attention_max_abs_diff.max(verify_f32_close(
        &format!(
            "{label} request {:?} attention timestep {timestep}",
            run.request_id
        ),
        &step.attention_output,
        attention_expected,
        1e-4,
        1e-4,
    )?);
    run.checks.projection_input_max_abs_diff =
        run.checks
            .projection_input_max_abs_diff
            .max(verify_f32_close(
                &format!(
                    "{label} request {:?} projection input timestep {timestep}",
                    run.request_id
                ),
                &step.attention_projection_input,
                projection_input_expected,
                1e-4,
                1e-4,
            )?);
    run.checks.projected_max_abs_diff = run.checks.projected_max_abs_diff.max(verify_f32_close(
        &format!(
            "{label} request {:?} projected timestep {timestep}",
            run.request_id
        ),
        &step.projected_output,
        projected_expected,
        1e-4,
        1e-4,
    )?);
    run.checks.block_max_abs_diff = run.checks.block_max_abs_diff.max(verify_f32_close(
        &format!(
            "{label} request {:?} block timestep {timestep}",
            run.request_id
        ),
        &step.block_output,
        block_expected,
        1e-4,
        1e-4,
    )?);
    run.checks.post_norm_max_abs_diff = run.checks.post_norm_max_abs_diff.max(verify_f32_close(
        &format!(
            "{label} request {:?} post norm timestep {timestep}",
            run.request_id
        ),
        &step.post_normed,
        post_norm_expected,
        1e-4,
        1e-4,
    )?);
    run.checks.mlp_max_abs_diff = run.checks.mlp_max_abs_diff.max(verify_f32_close(
        &format!(
            "{label} request {:?} MLP timestep {timestep}",
            run.request_id
        ),
        &step.mlp_output,
        mlp_expected,
        1e-4,
        1e-4,
    )?);
    run.checks.layer_max_abs_diff = run.checks.layer_max_abs_diff.max(verify_f32_close(
        &format!(
            "{label} request {:?} layer timestep {timestep}",
            run.request_id
        ),
        &step.layer_output,
        layer_expected,
        1e-4,
        1e-4,
    )?);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_scheduler_layer_prefill_step(
    runner: &mut Qwen3DecoderLayerRequestDecodeRunner<'_>,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    run: &mut SchedulerLayerDecodeRun,
    timestep: usize,
    q_token_elements: usize,
    k_token_elements: usize,
    v_token_elements: usize,
    attention_elements: usize,
    hidden: usize,
    label: &str,
) -> Result<(), String> {
    let input_layout = Qwen3DecoderLayerDecodeInputLayout {
        q_token_elements,
        k_token_elements,
        v_token_elements,
        attention_elements,
        hidden,
    };
    let input = qwen3_decoder_layer_prefill_input_from_sequence(
        scheduler_layer_decode_sequence_view(run),
        timestep,
        input_layout,
        label,
    )?;
    let step = runner.run_prefill_step(stream, input)?;
    if step.cache_position != timestep {
        return Err(format!(
            "{label} request {:?} wrote cache_position {}, expected {timestep}",
            run.request_id, step.cache_position
        ));
    }
    verify_scheduler_layer_step_output(label, run, &step, hidden, attention_elements)
}

#[allow(clippy::too_many_arguments)]
fn run_scheduler_layer_stack_prefill_step(
    runner: &mut Qwen3DecoderLayerStackRequestDecodeRunner<'_>,
    layer_index: usize,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    run: &mut SchedulerLayerDecodeRun,
    timestep: usize,
    decode: Qwen3PackageModelDecodePlan,
    label: &str,
) -> Result<(), String> {
    let step = qwen3_package_model_run_prefill_step_from_sequence(
        runner,
        layer_index,
        stream,
        scheduler_layer_decode_sequence_view(run),
        timestep,
        decode,
        label,
    )?;
    if step.cache_position != timestep {
        return Err(format!(
            "{label} request {:?} wrote cache_position {}, expected {timestep}",
            run.request_id, step.cache_position
        ));
    }
    verify_scheduler_layer_step_output(label, run, &step, decode.hidden, decode.attention_elements)
}

#[allow(clippy::too_many_arguments)]
fn run_synthetic_scheduler_ready_batch(
    runner: &mut Qwen3SelfAttnRequestDecodeRunner,
    scheduler: &mut SchedulerState,
    runs: &mut [SyntheticSchedulerPagedDecodeRun],
    stream: &mut ullm_runtime_sys::RuntimeStream,
    expected_ids: &[RequestId],
    max_requests: usize,
    block_size: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
    label: &str,
) -> Result<usize, String> {
    let ready = scheduler
        .ready_decode_batch(max_requests)
        .map_err(|err| format!("{label} failed to prepare ready decode batch: {err}"))?;
    let ready_ids = ready
        .iter()
        .map(|request| request.request.id)
        .collect::<Vec<_>>();
    if ready_ids != expected_ids {
        return Err(format!(
            "{label} ready request ids {:?} did not match expected {:?}",
            ready_ids, expected_ids
        ));
    }

    for request in &ready {
        let run = synthetic_scheduler_decode_run(runs, request.request.id)
            .ok_or_else(|| format!("{label} request {:?} has no decode run", request.request.id))?;
        let expected_cache_position =
            run.prompt_tokens
                .checked_add(run.decode_steps)
                .ok_or_else(|| {
                    format!(
                        "{label} request {:?} cache position overflows",
                        run.request_id
                    )
                })?;
        if request.cache_position != expected_cache_position {
            return Err(format!(
                "{label} request {:?} cache_position {} did not match expected {}",
                request.request.id, request.cache_position, expected_cache_position
            ));
        }
        if request.next_cache_len != request.cache_position + 1 {
            return Err(format!(
                "{label} request {:?} next_cache_len {} did not match cache_position + 1",
                request.request.id, request.next_cache_len
            ));
        }
        let expected_remaining = run
            .max_new_tokens
            .checked_sub(run.decode_steps)
            .ok_or_else(|| format!("{label} request {:?} decode step overflow", run.request_id))?;
        if request.remaining_new_tokens != expected_remaining {
            return Err(format!(
                "{label} request {:?} remaining_new_tokens {} did not match expected {}",
                request.request.id, request.remaining_new_tokens, expected_remaining
            ));
        }
        if request.allocation.blocks != run.block_table {
            return Err(format!(
                "{label} request {:?} block table {:?} did not match run block table {:?}",
                request.request.id, request.allocation.blocks, run.block_table
            ));
        }
    }

    let outputs = {
        let q_token_elements = q_heads
            .checked_mul(head_dim)
            .ok_or_else(|| format!("{label} q token element count overflows"))?;
        let k_token_elements = kv_heads
            .checked_mul(head_dim)
            .ok_or_else(|| format!("{label} k token element count overflows"))?;
        let v_token_elements = kv_heads
            .checked_mul(value_dim)
            .ok_or_else(|| format!("{label} v token element count overflows"))?;
        let mut inputs = Vec::with_capacity(ready.len());
        for request in &ready {
            let run =
                synthetic_scheduler_decode_run(runs, request.request.id).ok_or_else(|| {
                    format!(
                        "{label} request {:?} disappeared while preparing decode input",
                        request.request.id
                    )
                })?;
            let q_start = request
                .cache_position
                .checked_mul(q_token_elements)
                .ok_or_else(|| format!("{label} q slice start overflows"))?;
            let q_end = q_start
                .checked_add(q_token_elements)
                .ok_or_else(|| format!("{label} q slice end overflows"))?;
            let k_start = request
                .cache_position
                .checked_mul(k_token_elements)
                .ok_or_else(|| format!("{label} k slice start overflows"))?;
            let k_end = k_start
                .checked_add(k_token_elements)
                .ok_or_else(|| format!("{label} k slice end overflows"))?;
            let v_start = request
                .cache_position
                .checked_mul(v_token_elements)
                .ok_or_else(|| format!("{label} v slice start overflows"))?;
            let v_end = v_start
                .checked_add(v_token_elements)
                .ok_or_else(|| format!("{label} v slice end overflows"))?;
            inputs.push(Qwen3SelfAttnDecodeBatchInput {
                request_id: request.request.id,
                q: &run.q_sequence[q_start..q_end],
                k: &run.k_sequence[k_start..k_end],
                v: &run.v_sequence[v_start..v_end],
            });
        }
        runner.run_ready_batch(stream, scheduler, &ready, &inputs)?
    };

    for output in outputs {
        let run = synthetic_scheduler_decode_run_mut(runs, output.request_id).ok_or_else(|| {
            format!(
                "{label} advanced request {:?} disappeared",
                output.request_id
            )
        })?;
        run.decode_steps = run.decode_steps.checked_add(1).ok_or_else(|| {
            format!(
                "{label} request {:?} decode step count overflows",
                run.request_id
            )
        })?;
        let q_token_elements = q_heads
            .checked_mul(head_dim)
            .ok_or_else(|| format!("{label} q token element count overflows"))?;
        let q_start = output
            .cache_position
            .checked_mul(q_token_elements)
            .ok_or_else(|| format!("{label} q expected slice start overflows"))?;
        let q_end = q_start
            .checked_add(q_token_elements)
            .ok_or_else(|| format!("{label} q expected slice end overflows"))?;
        let expected = runtime_host_paged_decode_attn_f32(
            &run.q_sequence[q_start..q_end],
            &run.expected_k_cache,
            &run.expected_v_cache,
            &run.block_table,
            output.cache_len,
            block_size,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
        );
        let max_abs_diff = verify_f32_close(
            &format!(
                "{label} request {:?} timestep {}",
                run.request_id, output.cache_position
            ),
            &output.attention_output,
            &expected,
            1e-4,
            1e-4,
        )?;
        run.attention_max_abs_diff = run.attention_max_abs_diff.max(max_abs_diff);
    }
    Ok(expected_ids.len())
}

#[allow(clippy::too_many_arguments)]
fn run_scheduler_layer_ready_batch(
    runner: &mut Qwen3DecoderLayerRequestDecodeRunner<'_>,
    scheduler: &mut SchedulerState,
    runs: &mut [SchedulerLayerDecodeRun],
    stream: &mut ullm_runtime_sys::RuntimeStream,
    expected_ids: &[RequestId],
    max_requests: usize,
    q_token_elements: usize,
    k_token_elements: usize,
    v_token_elements: usize,
    attention_elements: usize,
    hidden: usize,
    advance_scheduler: bool,
    label: &str,
) -> Result<usize, String> {
    let ready = scheduler
        .ready_decode_batch(max_requests)
        .map_err(|err| format!("{label} failed to prepare ready decode batch: {err}"))?;
    let ready_ids = ready
        .iter()
        .map(|request| request.request.id)
        .collect::<Vec<_>>();
    if ready_ids != expected_ids {
        return Err(format!(
            "{label} ready request ids {:?} did not match expected {:?}",
            ready_ids, expected_ids
        ));
    }

    for request in &ready {
        let run = scheduler_layer_decode_run(runs, request.request.id)
            .ok_or_else(|| format!("{label} request {:?} has no layer run", request.request.id))?;
        let expected_cache_position =
            run.prompt_tokens
                .checked_add(run.decode_steps)
                .ok_or_else(|| {
                    format!(
                        "{label} request {:?} cache position overflows",
                        run.request_id
                    )
                })?;
        if request.cache_position != expected_cache_position {
            return Err(format!(
                "{label} request {:?} cache_position {} did not match expected {}",
                request.request.id, request.cache_position, expected_cache_position
            ));
        }
        if request.next_cache_len != request.cache_position + 1 {
            return Err(format!(
                "{label} request {:?} next_cache_len {} did not match cache_position + 1",
                request.request.id, request.next_cache_len
            ));
        }
        let expected_remaining = run
            .max_new_tokens
            .checked_sub(run.decode_steps)
            .ok_or_else(|| format!("{label} request {:?} decode step overflow", run.request_id))?;
        if request.remaining_new_tokens != expected_remaining {
            return Err(format!(
                "{label} request {:?} remaining_new_tokens {} did not match expected {}",
                request.request.id, request.remaining_new_tokens, expected_remaining
            ));
        }
        if request.allocation.blocks != run.block_table {
            return Err(format!(
                "{label} request {:?} block table {:?} did not match run block table {:?}",
                request.request.id, request.allocation.blocks, run.block_table
            ));
        }
    }

    let outputs = {
        let input_layout = Qwen3DecoderLayerDecodeInputLayout {
            q_token_elements,
            k_token_elements,
            v_token_elements,
            attention_elements,
            hidden,
        };
        let sequences = runs
            .iter()
            .map(scheduler_layer_decode_sequence_view)
            .collect::<Vec<_>>();
        let inputs = qwen3_decoder_layer_decode_batch_inputs_from_sequences(
            &ready,
            &sequences,
            input_layout,
            label,
        )?;
        if advance_scheduler {
            runner.run_ready_batch(stream, scheduler, &ready, &inputs)?
        } else {
            runner.run_ready_batch_without_advance(stream, scheduler, &ready, &inputs)?
        }
    };

    for output in outputs {
        let run = scheduler_layer_decode_run_mut(runs, output.request_id).ok_or_else(|| {
            format!(
                "{label} advanced request {:?} disappeared",
                output.request_id
            )
        })?;
        run.decode_steps = run.decode_steps.checked_add(1).ok_or_else(|| {
            format!(
                "{label} request {:?} decode step count overflows",
                run.request_id
            )
        })?;
        verify_scheduler_layer_step_output(label, run, &output, hidden, attention_elements)?;
    }
    Ok(expected_ids.len())
}

#[allow(clippy::too_many_arguments)]
fn run_scheduler_layer_stack_ready_batch(
    runner: &mut Qwen3DecoderLayerStackRequestDecodeRunner<'_>,
    scheduler: &mut SchedulerState,
    runs_by_layer: &mut [Vec<SchedulerLayerDecodeRun>],
    stream: &mut ullm_runtime_sys::RuntimeStream,
    ready: &[SchedulerDecodeRequest],
    decode: Qwen3PackageModelDecodePlan,
    label: &str,
) -> Result<usize, String> {
    if ready.is_empty() {
        return Ok(0);
    }

    for (layer_position, runs) in runs_by_layer.iter().enumerate() {
        for request in ready {
            let run = scheduler_layer_decode_run(runs, request.request.id).ok_or_else(|| {
                format!(
                    "{label} layer {layer_position} request {:?} has no layer run",
                    request.request.id
                )
            })?;
            let expected_cache_position = run
                .prompt_tokens
                .checked_add(run.decode_steps)
                .ok_or_else(|| {
                    format!(
                        "{label} layer {layer_position} request {:?} cache position overflows",
                        run.request_id
                    )
                })?;
            if request.cache_position != expected_cache_position {
                return Err(format!(
                    "{label} layer {layer_position} request {:?} cache_position {} did not match expected {}",
                    request.request.id, request.cache_position, expected_cache_position
                ));
            }
            if request.next_cache_len != request.cache_position + 1 {
                return Err(format!(
                    "{label} layer {layer_position} request {:?} next_cache_len {} did not match cache_position + 1",
                    request.request.id, request.next_cache_len
                ));
            }
            let expected_remaining = run
                .max_new_tokens
                .checked_sub(run.decode_steps)
                .ok_or_else(|| {
                    format!(
                        "{label} layer {layer_position} request {:?} decode step overflow",
                        run.request_id
                    )
                })?;
            if request.remaining_new_tokens != expected_remaining {
                return Err(format!(
                    "{label} layer {layer_position} request {:?} remaining_new_tokens {} did not match expected {}",
                    request.request.id, request.remaining_new_tokens, expected_remaining
                ));
            }
            if request.allocation.blocks != run.block_table {
                return Err(format!(
                    "{label} layer {layer_position} request {:?} block table {:?} did not match run block table {:?}",
                    request.request.id, request.allocation.blocks, run.block_table
                ));
            }
        }
    }

    let layer_sequences = runs_by_layer
        .iter()
        .map(|runs| {
            runs.iter()
                .map(scheduler_layer_decode_sequence_view)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let layer_sequence_refs = layer_sequences
        .iter()
        .map(Vec::as_slice)
        .collect::<Vec<_>>();
    let outputs_by_layer = qwen3_package_model_run_ready_batch_from_sequences(
        runner,
        stream,
        scheduler,
        ready,
        decode,
        &layer_sequence_refs,
        label,
    )?;
    drop(layer_sequence_refs);
    drop(layer_sequences);

    for (layer_position, outputs) in outputs_by_layer.into_iter().enumerate() {
        let runs = runs_by_layer
            .get_mut(layer_position)
            .ok_or_else(|| format!("{label} layer {layer_position} output has no run list"))?;
        for output in outputs {
            let run = scheduler_layer_decode_run_mut(runs, output.request_id).ok_or_else(|| {
                format!(
                    "{label} layer {layer_position} advanced request {:?} disappeared",
                    output.request_id
                )
            })?;
            run.decode_steps = run.decode_steps.checked_add(1).ok_or_else(|| {
                format!(
                    "{label} layer {layer_position} request {:?} decode step count overflows",
                    run.request_id
                )
            })?;
            verify_scheduler_layer_step_output(
                label,
                run,
                &output,
                decode.hidden,
                decode.attention_elements,
            )?;
        }
    }
    Ok(ready.len())
}

fn runtime_scheduler_paged_decode_smoke(device_index: Option<String>) -> ExitCode {
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    match runtime_scheduler_paged_decode_smoke_impl(device_index) {
        Ok(message) => {
            println!("{message}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("{err}");
            ExitCode::from(1)
        }
    }
}

fn runtime_scheduler_paged_decode_smoke_impl(device_index: u32) -> Result<String, String> {
    let mut context = ullm_runtime_sys::RuntimeContext::create(device_index)
        .map_err(|err| format!("failed to create runtime context: {err}"))?;
    let info = context
        .device_info()
        .map_err(|err| format!("failed to query runtime context device: {err}"))?;
    let mut stream = context
        .create_stream()
        .map_err(|err| format!("failed to create runtime stream: {err}"))?;

    let block_size = 2_usize;
    let cache_blocks = 8_usize;
    let q_heads = 4_usize;
    let kv_heads = 2_usize;
    let head_dim = 3_usize;
    let value_dim = 2_usize;
    let softmax_scale = 1.0_f32 / (head_dim as f32).sqrt();
    let requests = vec![
        Request::new(201, 2, 2),
        Request::new(202, 3, 1),
        Request::new(203, 1, 0),
    ];

    let mut scheduler = SchedulerState::with_block_size(cache_blocks as u32, block_size as u32);
    for request in &requests {
        scheduler.enqueue(request.clone());
    }
    let mut decode_runner = Qwen3SelfAttnRequestDecodeRunner::new();
    let mut allocated = scheduler
        .pop_prefill_batch_with_allocation(usize::MAX)
        .map_err(|err| format!("failed to allocate synthetic scheduler decode batch: {err}"))?;
    if allocated.len() != requests.len() {
        return Err(format!(
            "synthetic scheduler decode selected {} requests, expected {}",
            allocated.len(),
            requests.len()
        ));
    }

    let mut runs = Vec::with_capacity(allocated.len());
    for (request_index, scheduled) in allocated.drain(..).enumerate() {
        let request = scheduled.request;
        let block_table = scheduled.allocation.blocks;
        let total_tokens = request
            .prompt_tokens
            .checked_add(request.max_new_tokens)
            .ok_or_else(|| format!("request {:?} total token count overflows", request.id))?;
        if total_tokens == 0 {
            return Err(format!("request {:?} has no tokens to decode", request.id));
        }

        let q_sequence =
            synthetic_scheduler_decode_values(request_index, total_tokens, q_heads * head_dim, 1);
        let k_sequence =
            synthetic_scheduler_decode_values(request_index, total_tokens, kv_heads * head_dim, 2);
        let v_sequence =
            synthetic_scheduler_decode_values(request_index, total_tokens, kv_heads * value_dim, 3);
        let shape = PagedDecodeShape {
            block_size,
            cache_blocks,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
        };
        let PagedKvCacheReadback {
            k: expected_k_cache,
            v: expected_v_cache,
        } = pack_paged_kv_cache_for_block_table(
            &k_sequence,
            &v_sequence,
            &block_table,
            total_tokens,
            shape,
        )?;
        decode_runner.insert_request(
            &mut context,
            &mut stream,
            request.id,
            shape,
            block_table.clone(),
            softmax_scale,
        )?;
        let mut run = SyntheticSchedulerPagedDecodeRun {
            request_id: request.id,
            prompt_tokens: request.prompt_tokens,
            max_new_tokens: request.max_new_tokens,
            total_tokens,
            block_table,
            q_sequence,
            k_sequence,
            v_sequence,
            expected_k_cache,
            expected_v_cache,
            decode_steps: 0,
            attention_max_abs_diff: 0.0,
            k_cache_max_abs_diff: 0.0,
            v_cache_max_abs_diff: 0.0,
        };
        for timestep in 0..run.prompt_tokens {
            run_synthetic_scheduler_decode_step(
                &mut decode_runner,
                &mut stream,
                &mut run,
                timestep,
                block_size,
                q_heads,
                kv_heads,
                head_dim,
                value_dim,
                softmax_scale,
                "runtime scheduler paged decode prefill",
            )?;
        }
        scheduler.complete_prefill(run.request_id).map_err(|err| {
            format!(
                "failed to complete synthetic prefill {:?}: {err}",
                run.request_id
            )
        })?;
        runs.push(run);
    }

    let first_batch_ready = run_synthetic_scheduler_ready_batch(
        &mut decode_runner,
        &mut scheduler,
        &mut runs,
        &mut stream,
        &[RequestId(201), RequestId(202)],
        8,
        block_size,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        "runtime scheduler paged decode first batch",
    )?;
    let second_batch_ready = run_synthetic_scheduler_ready_batch(
        &mut decode_runner,
        &mut scheduler,
        &mut runs,
        &mut stream,
        &[RequestId(201)],
        8,
        block_size,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        "runtime scheduler paged decode second batch",
    )?;
    let final_ready = scheduler
        .ready_decode_batch(8)
        .map_err(|err| format!("failed to query final ready decode batch: {err}"))?
        .len();
    if final_ready != 0 {
        return Err(format!(
            "runtime scheduler paged decode final ready count {final_ready}, expected 0"
        ));
    }

    for run in &mut runs {
        let active = scheduler.active_request(run.request_id).ok_or_else(|| {
            format!(
                "request {:?} missing from active scheduler state",
                run.request_id
            )
        })?;
        if active.cached_tokens != run.total_tokens {
            return Err(format!(
                "request {:?} cached_tokens {} did not match total_tokens {}",
                run.request_id, active.cached_tokens, run.total_tokens
            ));
        }
        if active.generated_tokens != run.max_new_tokens {
            return Err(format!(
                "request {:?} generated_tokens {} did not match max_new_tokens {}",
                run.request_id, active.generated_tokens, run.max_new_tokens
            ));
        }
        let cache = decode_runner
            .read_cache_to_host(run.request_id, &mut stream)
            .map_err(|err| {
                format!(
                    "failed to read synthetic cache for {:?}: {err}",
                    run.request_id
                )
            })?;
        run.k_cache_max_abs_diff = verify_f32_close(
            &format!(
                "runtime scheduler paged decode request {:?} k cache",
                run.request_id
            ),
            &cache.k,
            &run.expected_k_cache,
            1e-5,
            1e-5,
        )?;
        run.v_cache_max_abs_diff = verify_f32_close(
            &format!(
                "runtime scheduler paged decode request {:?} v cache",
                run.request_id
            ),
            &cache.v,
            &run.expected_v_cache,
            1e-5,
            1e-5,
        )?;
    }

    let stats = scheduler.allocator_stats();
    let request_ids = runs.iter().map(|run| run.request_id.0).collect::<Vec<_>>();
    let block_tables = runs
        .iter()
        .map(|run| run.block_table.clone())
        .collect::<Vec<_>>();
    let cached_tokens = runs
        .iter()
        .map(|run| {
            scheduler
                .active_request(run.request_id)
                .map(|active| active.cached_tokens)
                .unwrap_or(0)
        })
        .collect::<Vec<_>>();
    let generated_tokens = runs
        .iter()
        .map(|run| {
            scheduler
                .active_request(run.request_id)
                .map(|active| active.generated_tokens)
                .unwrap_or(0)
        })
        .collect::<Vec<_>>();
    let prompt_tokens = runs.iter().map(|run| run.prompt_tokens).collect::<Vec<_>>();
    let max_new_tokens = runs
        .iter()
        .map(|run| run.max_new_tokens)
        .collect::<Vec<_>>();
    let decode_steps = runs.iter().map(|run| run.decode_steps).collect::<Vec<_>>();
    let attention_max_abs_diff = runs
        .iter()
        .map(|run| run.attention_max_abs_diff)
        .fold(0.0_f32, f32::max);
    let k_cache_max_abs_diff = runs
        .iter()
        .map(|run| run.k_cache_max_abs_diff)
        .fold(0.0_f32, f32::max);
    let v_cache_max_abs_diff = runs
        .iter()
        .map(|run| run.v_cache_max_abs_diff)
        .fold(0.0_f32, f32::max);

    Ok(format!(
        "runtime-scheduler-paged-decode-smoke backend={} device_index={} name=\"{}\" request_count={} request_ids={:?} prompt_tokens={:?} max_new_tokens={:?} block_size={} cache_blocks={} block_tables={:?} first_batch_ready={} second_batch_ready={} final_ready={} decode_steps={:?} cached_tokens={:?} generated_tokens={:?} active_len={} free_blocks={} allocated_block_count={} free_runs={} largest_free_run={} q_heads={} kv_heads={} head_dim={} value_dim={} softmax_scale={softmax_scale:.9} attention_max_abs_diff={attention_max_abs_diff:.9} k_cache_max_abs_diff={k_cache_max_abs_diff:.9} v_cache_max_abs_diff={v_cache_max_abs_diff:.9} verified=true",
        info.backend,
        device_index,
        info.name,
        runs.len(),
        request_ids,
        prompt_tokens,
        max_new_tokens,
        block_size,
        cache_blocks,
        block_tables,
        first_batch_ready,
        second_batch_ready,
        final_ready,
        decode_steps,
        cached_tokens,
        generated_tokens,
        scheduler.active_len(),
        stats.free_blocks,
        stats.allocated_blocks,
        stats.free_runs,
        stats.largest_free_run,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
    ))
}

fn runtime_scheduler_layer_decode_smoke(device_index: Option<String>) -> ExitCode {
    let device_index = match parse_optional_device_index(device_index) {
        Ok(value) => value,
        Err(code) => return code,
    };
    match runtime_scheduler_layer_decode_smoke_impl(device_index) {
        Ok(message) => {
            println!("{message}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("{err}");
            ExitCode::from(1)
        }
    }
}

fn runtime_scheduler_layer_decode_smoke_impl(device_index: u32) -> Result<String, String> {
    let mut context = ullm_runtime_sys::RuntimeContext::create(device_index)
        .map_err(|err| format!("failed to create runtime context: {err}"))?;
    let info = context
        .device_info()
        .map_err(|err| format!("failed to query runtime context device: {err}"))?;
    let mut stream = context
        .create_stream()
        .map_err(|err| format!("failed to create runtime stream: {err}"))?;

    let block_size = 2_usize;
    let cache_blocks = 8_usize;
    let shape = PagedDecodeShape {
        block_size,
        cache_blocks,
        q_heads: 2,
        kv_heads: 1,
        head_dim: 2,
        value_dim: 2,
    };
    let hidden = 4_usize;
    let intermediate = 5_usize;
    let softmax_scale = 1.0_f32 / (shape.head_dim as f32).sqrt();
    let mlp_epsilon = 1e-5_f32;
    let q_token_elements = shape.q_elements()?;
    let k_token_elements = shape.k_token_elements()?;
    let v_token_elements = shape.v_token_elements()?;
    let attention_elements = shape.output_elements()?;
    let weights = synthetic_scheduler_decoder_layer_weights(
        &mut context,
        &mut stream,
        shape,
        hidden,
        intermediate,
    )?;
    let requests = vec![
        Request::new(201, 2, 2),
        Request::new(202, 3, 1),
        Request::new(203, 1, 0),
    ];

    let mut scheduler = SchedulerState::with_block_size(cache_blocks as u32, block_size as u32);
    for request in &requests {
        scheduler.enqueue(request.clone());
    }
    let mut runner = Qwen3DecoderLayerRequestDecodeRunner::new();
    let mut allocated = scheduler
        .pop_prefill_batch_with_allocation(usize::MAX)
        .map_err(|err| format!("failed to allocate synthetic scheduler layer batch: {err}"))?;
    if allocated.len() != requests.len() {
        return Err(format!(
            "synthetic scheduler layer selected {} requests, expected {}",
            allocated.len(),
            requests.len()
        ));
    }

    let mut runs = Vec::with_capacity(allocated.len());
    for (request_index, scheduled) in allocated.drain(..).enumerate() {
        let request = scheduled.request;
        let block_table = scheduled.allocation.blocks;
        let total_tokens = request
            .prompt_tokens
            .checked_add(request.max_new_tokens)
            .ok_or_else(|| format!("request {:?} total token count overflows", request.id))?;
        if total_tokens == 0 {
            return Err(format!("request {:?} has no tokens to decode", request.id));
        }

        let q_sequence =
            synthetic_scheduler_decode_values(request_index, total_tokens, q_token_elements, 21);
        let k_sequence =
            synthetic_scheduler_decode_values(request_index, total_tokens, k_token_elements, 22);
        let v_sequence =
            synthetic_scheduler_decode_values(request_index, total_tokens, v_token_elements, 23);
        let gate_sequence =
            synthetic_scheduler_decode_values(request_index, total_tokens, attention_elements, 24);
        let residual_sequence =
            synthetic_scheduler_decode_values(request_index, total_tokens, hidden, 25);
        let expected = qwen3_decoder_layer_sequence_to_host_f32(
            &weights,
            &mut context,
            &mut stream,
            shape,
            &block_table,
            softmax_scale,
            mlp_epsilon,
            &q_sequence,
            &k_sequence,
            &v_sequence,
            Some(&gate_sequence),
            &residual_sequence,
            total_tokens,
        )?;

        runner.insert_request(
            &mut context,
            &mut stream,
            request.id,
            &weights,
            shape,
            block_table.clone(),
            softmax_scale,
            mlp_epsilon,
        )?;
        let mut run = SchedulerLayerDecodeRun {
            state: SchedulerLayerDecodeState {
                request_id: request.id,
                prompt_tokens: request.prompt_tokens,
                max_new_tokens: request.max_new_tokens,
                total_tokens,
                block_table,
                q_sequence,
                k_sequence,
                v_sequence,
                output_gate_sequence: Some(gate_sequence),
                residual_sequence,
                decode_steps: 0,
            },
            checks: SchedulerLayerDecodeSmokeChecks::new(expected),
        };
        for timestep in 0..run.prompt_tokens {
            run_scheduler_layer_prefill_step(
                &mut runner,
                &mut stream,
                &mut run,
                timestep,
                q_token_elements,
                k_token_elements,
                v_token_elements,
                attention_elements,
                hidden,
                "runtime scheduler layer decode prefill",
            )?;
        }
        scheduler.complete_prefill(run.request_id).map_err(|err| {
            format!(
                "failed to complete synthetic layer prefill {:?}: {err}",
                run.request_id
            )
        })?;
        runs.push(run);
    }

    let first_batch_ready = run_scheduler_layer_ready_batch(
        &mut runner,
        &mut scheduler,
        &mut runs,
        &mut stream,
        &[RequestId(201), RequestId(202)],
        8,
        q_token_elements,
        k_token_elements,
        v_token_elements,
        attention_elements,
        hidden,
        true,
        "runtime scheduler layer decode first batch",
    )?;
    let second_batch_ready = run_scheduler_layer_ready_batch(
        &mut runner,
        &mut scheduler,
        &mut runs,
        &mut stream,
        &[RequestId(201)],
        8,
        q_token_elements,
        k_token_elements,
        v_token_elements,
        attention_elements,
        hidden,
        true,
        "runtime scheduler layer decode second batch",
    )?;
    let final_ready = scheduler
        .ready_decode_batch(8)
        .map_err(|err| format!("failed to query final layer ready decode batch: {err}"))?
        .len();
    if final_ready != 0 {
        return Err(format!(
            "runtime scheduler layer decode final ready count {final_ready}, expected 0"
        ));
    }

    for run in &mut runs {
        let active = scheduler.active_request(run.request_id).ok_or_else(|| {
            format!(
                "layer request {:?} missing from active scheduler state",
                run.request_id
            )
        })?;
        if active.cached_tokens != run.total_tokens {
            return Err(format!(
                "layer request {:?} cached_tokens {} did not match total_tokens {}",
                run.request_id, active.cached_tokens, run.total_tokens
            ));
        }
        if active.generated_tokens != run.max_new_tokens {
            return Err(format!(
                "layer request {:?} generated_tokens {} did not match max_new_tokens {}",
                run.request_id, active.generated_tokens, run.max_new_tokens
            ));
        }
        let cache = runner
            .read_cache_to_host(run.request_id, &mut stream)
            .map_err(|err| {
                format!(
                    "failed to read synthetic layer cache for {:?}: {err}",
                    run.request_id
                )
            })?;
        run.checks.k_cache_max_abs_diff = verify_f32_close(
            &format!(
                "runtime scheduler layer decode request {:?} k cache",
                run.request_id
            ),
            &cache.k,
            &run.checks.expected.paged_cache.k,
            1e-5,
            1e-5,
        )?;
        run.checks.v_cache_max_abs_diff = verify_f32_close(
            &format!(
                "runtime scheduler layer decode request {:?} v cache",
                run.request_id
            ),
            &cache.v,
            &run.checks.expected.paged_cache.v,
            1e-5,
            1e-5,
        )?;
    }

    let stats = scheduler.allocator_stats();
    let request_ids = runs.iter().map(|run| run.request_id.0).collect::<Vec<_>>();
    let block_tables = runs
        .iter()
        .map(|run| run.block_table.clone())
        .collect::<Vec<_>>();
    let cached_tokens = runs
        .iter()
        .map(|run| {
            scheduler
                .active_request(run.request_id)
                .map(|active| active.cached_tokens)
                .unwrap_or(0)
        })
        .collect::<Vec<_>>();
    let generated_tokens = runs
        .iter()
        .map(|run| {
            scheduler
                .active_request(run.request_id)
                .map(|active| active.generated_tokens)
                .unwrap_or(0)
        })
        .collect::<Vec<_>>();
    let prompt_tokens = runs.iter().map(|run| run.prompt_tokens).collect::<Vec<_>>();
    let max_new_tokens = runs
        .iter()
        .map(|run| run.max_new_tokens)
        .collect::<Vec<_>>();
    let decode_steps = runs.iter().map(|run| run.decode_steps).collect::<Vec<_>>();
    let attention_max_abs_diff = runs
        .iter()
        .map(|run| run.checks.attention_max_abs_diff)
        .fold(0.0_f32, f32::max);
    let projection_input_max_abs_diff = runs
        .iter()
        .map(|run| run.checks.projection_input_max_abs_diff)
        .fold(0.0_f32, f32::max);
    let projected_max_abs_diff = runs
        .iter()
        .map(|run| run.checks.projected_max_abs_diff)
        .fold(0.0_f32, f32::max);
    let block_max_abs_diff = runs
        .iter()
        .map(|run| run.checks.block_max_abs_diff)
        .fold(0.0_f32, f32::max);
    let post_norm_max_abs_diff = runs
        .iter()
        .map(|run| run.checks.post_norm_max_abs_diff)
        .fold(0.0_f32, f32::max);
    let mlp_max_abs_diff = runs
        .iter()
        .map(|run| run.checks.mlp_max_abs_diff)
        .fold(0.0_f32, f32::max);
    let layer_max_abs_diff = runs
        .iter()
        .map(|run| run.checks.layer_max_abs_diff)
        .fold(0.0_f32, f32::max);
    let k_cache_max_abs_diff = runs
        .iter()
        .map(|run| run.checks.k_cache_max_abs_diff)
        .fold(0.0_f32, f32::max);
    let v_cache_max_abs_diff = runs
        .iter()
        .map(|run| run.checks.v_cache_max_abs_diff)
        .fold(0.0_f32, f32::max);

    Ok(format!(
        "runtime-scheduler-layer-decode-smoke backend={} device_index={} name=\"{}\" request_count={} request_ids={:?} prompt_tokens={:?} max_new_tokens={:?} block_size={} cache_blocks={} block_tables={:?} first_batch_ready={} second_batch_ready={} final_ready={} decode_steps={:?} cached_tokens={:?} generated_tokens={:?} active_len={} free_blocks={} allocated_block_count={} free_runs={} largest_free_run={} q_heads={} kv_heads={} head_dim={} value_dim={} hidden={} intermediate={} softmax_scale={softmax_scale:.9} mlp_epsilon={mlp_epsilon:.9} attention_max_abs_diff={attention_max_abs_diff:.9} projection_input_max_abs_diff={projection_input_max_abs_diff:.9} projected_max_abs_diff={projected_max_abs_diff:.9} block_max_abs_diff={block_max_abs_diff:.9} post_norm_max_abs_diff={post_norm_max_abs_diff:.9} mlp_max_abs_diff={mlp_max_abs_diff:.9} layer_max_abs_diff={layer_max_abs_diff:.9} k_cache_max_abs_diff={k_cache_max_abs_diff:.9} v_cache_max_abs_diff={v_cache_max_abs_diff:.9} verified=true",
        info.backend,
        device_index,
        info.name,
        runs.len(),
        request_ids,
        prompt_tokens,
        max_new_tokens,
        block_size,
        cache_blocks,
        block_tables,
        first_batch_ready,
        second_batch_ready,
        final_ready,
        decode_steps,
        cached_tokens,
        generated_tokens,
        scheduler.active_len(),
        stats.free_blocks,
        stats.allocated_blocks,
        stats.free_runs,
        stats.largest_free_run,
        shape.q_heads,
        shape.kv_heads,
        shape.head_dim,
        shape.value_dim,
        hidden,
        intermediate,
    ))
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
    let decode_shape = PagedDecodeShape {
        block_size,
        cache_blocks,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
    };
    let PagedKvCacheReadback {
        k: paged_k_cache,
        v: paged_v_cache,
    } = match pack_paged_kv_cache_for_block_table(
        &logical_k,
        &logical_v,
        &block_table,
        cache_len,
        decode_shape,
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
    let q_projection_split = match split_qwen3_self_attn_q_projection(
        &q_projected,
        sequence_len,
        q_rows,
        q_cols,
        head_dim,
    ) {
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
    let q_projection_split = match split_qwen3_self_attn_q_projection(
        &q_projected,
        sequence_len,
        q_rows,
        q_cols,
        head_dim,
    ) {
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
    let paged_decode = match runtime_paged_kv_write_decode_verify(
        &mut context,
        &mut stream,
        &q_rope,
        &k_rope,
        &v_projected,
        sequence_len,
        paged_block_size,
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
        "package-self-attn-decode-smoke package={} layer={} q_tensor=\"{}\" k_tensor=\"{}\" v_tensor=\"{}\" q_norm_tensor=\"{}\" k_norm_tensor=\"{}\" hidden={} cache_len={} paged_block_size={} paged_cache_blocks={} paged_block_table={:?} scheduler_decode_batches={} scheduler_request_id={} scheduler_prefill_tokens={} scheduler_max_new_tokens={} scheduler_cached_tokens={} scheduler_generated_tokens={} scheduler_active_len={} paged_allocator_free_blocks={} paged_allocator_allocated_blocks={} paged_allocator_free_runs={} paged_allocator_largest_free_run={} q_projection_layout={} q_gate_elements={} q_heads={} kv_heads={} head_dim={} value_dim={} rotary_dim={} position_offset={} rope_base={} softmax_scale={softmax_scale:.9} q_norm_dtype={} k_norm_dtype={} backend={} device_index={} name=\"{}\" decode_q_preview={} k_cache_preview={} v_cache_preview={} paged_k_cache_preview={} paged_v_cache_preview={} causal_last_preview={} decode_preview={} paged_decode_preview={} q_norm_max_abs_diff={q_norm_max_abs_diff:.9} k_norm_max_abs_diff={k_norm_max_abs_diff:.9} q_rope_max_abs_diff={q_rope_max_abs_diff:.9} k_rope_max_abs_diff={k_rope_max_abs_diff:.9} attention_max_abs_diff={attention_max_abs_diff:.9} decode_max_abs_diff={decode_max_abs_diff:.9} paged_kv_write_k_max_abs_diff={paged_kv_write_k_max_abs_diff:.9} paged_kv_write_v_max_abs_diff={paged_kv_write_v_max_abs_diff:.9} paged_decode_max_abs_diff={paged_decode_max_abs_diff:.9} paged_step_decode_max_abs_diff={paged_step_decode_max_abs_diff:.9} decode_paged_max_abs_diff={decode_paged_max_abs_diff:.9} causal_decode_max_abs_diff={causal_decode_max_abs_diff:.9} causal_paged_decode_max_abs_diff={causal_paged_decode_max_abs_diff:.9} causal_paged_step_decode_max_abs_diff={causal_paged_step_decode_max_abs_diff:.9} verified=true",
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
        paged_decode.cache_blocks,
        paged_decode.block_table,
        paged_decode.scheduler_decode_batches,
        paged_decode.scheduler_request_id.0,
        paged_decode.scheduler_prefill_tokens,
        paged_decode.scheduler_max_new_tokens,
        paged_decode.scheduler_cached_tokens,
        paged_decode.scheduler_generated_tokens,
        paged_decode.scheduler_active_len,
        paged_decode.allocator_stats.free_blocks,
        paged_decode.allocator_stats.allocated_blocks,
        paged_decode.allocator_stats.free_runs,
        paged_decode.allocator_stats.largest_free_run,
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

    let result = package_self_attn_block_smoke_impl(
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
fn package_self_attn_block_smoke_impl(
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
) -> Result<String, String> {
    let q_tensor = format!("model.language_model.layers.{layer_index}.self_attn.q_proj.weight");
    let k_tensor = format!("model.language_model.layers.{layer_index}.self_attn.k_proj.weight");
    let v_tensor = format!("model.language_model.layers.{layer_index}.self_attn.v_proj.weight");
    let o_tensor = format!("model.language_model.layers.{layer_index}.self_attn.o_proj.weight");
    let q_norm_tensor =
        format!("model.language_model.layers.{layer_index}.self_attn.q_norm.weight");
    let k_norm_tensor =
        format!("model.language_model.layers.{layer_index}.self_attn.k_norm.weight");

    let mut context = ullm_runtime_sys::RuntimeContext::create(device_index)
        .map_err(|err| format!("failed to create runtime context: {err}"))?;
    let info = context
        .device_info()
        .map_err(|err| format!("failed to query runtime context device: {err}"))?;
    let mut stream = context
        .create_stream()
        .map_err(|err| format!("failed to create runtime stream: {err}"))?;

    let self_attn_weights = qwen3_self_attn_runtime_weights_from_package(
        &mut context,
        &mut stream,
        path,
        chunk_bytes,
        &q_tensor,
        &k_tensor,
        &v_tensor,
        &o_tensor,
        &q_norm,
        &k_norm,
    )?;

    let self_attn = run_self_attn_block_sequence_smoke(
        &mut context,
        &mut stream,
        &self_attn_weights,
        sequence_len,
        rotary_dim,
        rope_base,
        position_offset,
        &q_norm,
        &k_norm,
        "package-self-attn-block-smoke",
    )?;

    Ok(format!(
        "package-self-attn-block-smoke package={} layer={} q_tensor=\"{}\" k_tensor=\"{}\" v_tensor=\"{}\" o_tensor=\"{}\" q_norm_tensor=\"{}\" k_norm_tensor=\"{}\" hidden={} sequence_len={} paged_block_size={} paged_cache_blocks={} paged_block_table={:?} scheduler_request_id={} scheduler_prefill_tokens={} scheduler_max_new_tokens={} scheduler_cached_tokens={} scheduler_generated_tokens={} scheduler_active_len={} q_projection_layout={} q_gate_elements={} output_gate_layout={} q_heads={} kv_heads={} head_dim={} value_dim={} rotary_dim={} position_offset={} rope_base={} softmax_scale={:.9} q_norm_dtype={} k_norm_dtype={} backend={} device_index={} name=\"{}\" attention_preview={} gated_attention_preview={} projected_preview={} block_preview={} q_norm_max_abs_diff={:.9} k_norm_max_abs_diff={:.9} q_rope_max_abs_diff={:.9} k_rope_max_abs_diff={:.9} attention_max_abs_diff={:.9} paged_kv_write_k_max_abs_diff={:.9} paged_kv_write_v_max_abs_diff={:.9} paged_step_attention_max_abs_diff={:.9} causal_paged_step_attention_max_abs_diff={:.9} output_gate_max_abs_diff={:.9} o_proj_max_abs_diff={:.9} block_max_abs_diff={:.9} causal_paged_block_max_abs_diff={:.9} verified=true",
        path,
        layer_index,
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        q_norm_tensor,
        k_norm_tensor,
        self_attn.hidden,
        sequence_len,
        self_attn.paged_block_size,
        self_attn.paged_cache_blocks,
        self_attn.paged_block_table,
        self_attn.scheduler_request_id.0,
        self_attn.scheduler_prefill_tokens,
        self_attn.scheduler_max_new_tokens,
        self_attn.scheduler_cached_tokens,
        self_attn.scheduler_generated_tokens,
        self_attn.scheduler_active_len,
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
        info.backend,
        device_index,
        info.name,
        format_f32_preview(&self_attn.attention_output[..8.min(self_attn.attention_output.len())]),
        format_f32_preview(
            &self_attn.attention_projection_input
                [..8.min(self_attn.attention_projection_input.len())],
        ),
        format_f32_preview(&self_attn.attn_projected[..8.min(self_attn.attn_projected.len())]),
        format_f32_preview(&self_attn.block_output[..8.min(self_attn.block_output.len())]),
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
    ))
}

struct Qwen3SelfAttnPreparedSequence {
    residual_sequence: Vec<f32>,
    q_rope: Vec<f32>,
    k_rope: Vec<f32>,
    v_projected: Vec<f32>,
    q_gate: Option<Vec<f32>>,
    attention_output: Vec<f32>,
    expected_paged_k_cache: Vec<f32>,
    expected_paged_v_cache: Vec<f32>,
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
    paged_block_table: Vec<u32>,
    paged_block_size: usize,
    paged_cache_blocks: usize,
    scheduler_request_id: RequestId,
    scheduler_prefill_tokens: usize,
    scheduler_max_new_tokens: usize,
    scheduler_cached_tokens: usize,
    scheduler_generated_tokens: usize,
    scheduler_active_len: usize,
}

struct Qwen3SelfAttnModelLoopPreparedSequence {
    residual_sequence: Vec<f32>,
    q_rope: Vec<f32>,
    k_rope: Vec<f32>,
    v_projected: Vec<f32>,
    q_gate: Option<Vec<f32>>,
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
}

#[allow(dead_code)]
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
    paged_block_table: Vec<u32>,
    paged_block_size: usize,
    paged_cache_blocks: usize,
    scheduler_request_id: RequestId,
    scheduler_prefill_tokens: usize,
    scheduler_max_new_tokens: usize,
    scheduler_cached_tokens: usize,
    scheduler_generated_tokens: usize,
    scheduler_active_len: usize,
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

#[allow(clippy::too_many_arguments)]
fn qwen3_self_attn_prepare_model_loop_sequence_smoke(
    context: &mut ullm_runtime_sys::RuntimeContext,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    self_attn_weights: &Qwen3SelfAttnRuntimeWeights,
    residual_sequence: Vec<f32>,
    sequence_len: usize,
    rotary_dim: usize,
    rope_base: f32,
    position_offset: usize,
    input_norm: &PassthroughF32Data,
    q_norm: &PassthroughF32Data,
    k_norm: &PassthroughF32Data,
    paged_block_table: &[u32],
    paged_block_size: usize,
    paged_cache_blocks: usize,
    label: &str,
) -> Result<Qwen3SelfAttnModelLoopPreparedSequence, String> {
    let Qwen3SelfAttnRuntimeShape {
        hidden,
        q_heads: shape_q_heads,
        kv_heads: _,
        head_dim: _,
        value_dim: _,
        attention_width: _,
        q_projection_layout,
    } = qwen3_self_attn_runtime_shape(self_attn_weights)?;
    let expected_residual_len = sequence_len
        .checked_mul(hidden)
        .ok_or_else(|| format!("{label} residual length overflows"))?;
    if residual_sequence.len() != expected_residual_len {
        return Err(format!(
            "{label} residual length {} does not match expected {}",
            residual_sequence.len(),
            expected_residual_len
        ));
    }
    if input_norm.values.len() != hidden {
        return Err(format!(
            "{label} input RMSNorm length {} does not match hidden={hidden}",
            input_norm.values.len()
        ));
    }

    let original_residual_sequence = residual_sequence;
    let mut attention_input_normed = Vec::with_capacity(original_residual_sequence.len());
    for residual in original_residual_sequence.chunks_exact(hidden) {
        attention_input_normed.extend(runtime_host_rmsnorm_f32(
            residual,
            &input_norm.values,
            1e-6_f32,
        ));
    }

    let prepared = qwen3_self_attn_prepare_sequence_for_paged_decode_f32(
        context,
        stream,
        self_attn_weights,
        attention_input_normed,
        sequence_len,
        &q_norm.values,
        &k_norm.values,
        rotary_dim,
        position_offset,
        rope_base,
        paged_block_table,
        paged_block_size,
        paged_cache_blocks,
    )?;
    let Qwen3SelfAttnRuntimePreparedSequenceForPagedDecode {
        residual_sequence: _,
        prepared:
            Qwen3SelfAttnRuntimePreparedSequence {
                q_query,
                k_projected,
                q_normed,
                k_normed,
                q_rope,
                k_rope,
                v_projected,
                q_gate,
                attention_output,
                shape,
                softmax_scale,
                q_projection_layout: prepared_q_projection_layout,
                q_gate_elements,
                output_gate_layout,
            },
        paged_k_cache: _,
        paged_v_cache: _,
        paged_block_table: _,
        paged_block_size: _,
        paged_cache_blocks: _,
    } = prepared;

    if q_projection_layout != prepared_q_projection_layout {
        return Err(format!(
            "{label} q projection layout changed between shape and prepare: {q_projection_layout} vs {prepared_q_projection_layout}"
        ));
    }
    if shape.q_heads != shape_q_heads {
        return Err(format!(
            "{label} q head count changed between shape and prepare: {} vs {shape_q_heads}",
            shape.q_heads
        ));
    }

    let epsilon = 1e-5_f32;
    let mut expected_q_normed = Vec::with_capacity(q_query.len());
    for head_input in q_query.chunks_exact(shape.head_dim) {
        expected_q_normed.extend(runtime_host_rmsnorm_f32(
            head_input,
            &q_norm.values,
            epsilon,
        ));
    }
    let q_norm_max_abs_diff = verify_f32_close(
        &format!("{label} q_norm"),
        &q_normed,
        &expected_q_normed,
        1e-4_f32,
        1e-4_f32,
    )?;

    let mut expected_k_normed = Vec::with_capacity(k_normed.len());
    for head_input in k_projected.chunks_exact(shape.head_dim) {
        expected_k_normed.extend(runtime_host_rmsnorm_f32(
            head_input,
            &k_norm.values,
            epsilon,
        ));
    }
    let k_norm_max_abs_diff = verify_f32_close(
        &format!("{label} k_norm"),
        &k_normed,
        &expected_k_normed,
        1e-4_f32,
        1e-4_f32,
    )?;

    let expected_q_rope = runtime_host_rope_f32(
        &q_normed,
        sequence_len,
        shape.q_heads,
        shape.head_dim,
        rotary_dim,
        position_offset,
        rope_base,
    );
    let q_rope_max_abs_diff = verify_f32_close(
        &format!("{label} q_rope"),
        &q_rope,
        &expected_q_rope,
        1e-4_f32,
        1e-4_f32,
    )?;
    let expected_k_rope = runtime_host_rope_f32(
        &k_normed,
        sequence_len,
        shape.kv_heads,
        shape.head_dim,
        rotary_dim,
        position_offset,
        rope_base,
    );
    let k_rope_max_abs_diff = verify_f32_close(
        &format!("{label} k_rope"),
        &k_rope,
        &expected_k_rope,
        1e-4_f32,
        1e-4_f32,
    )?;

    let expected_attention = runtime_host_causal_attn_f32(
        &q_rope,
        &k_rope,
        &v_projected,
        sequence_len,
        shape.q_heads,
        shape.kv_heads,
        shape.head_dim,
        shape.value_dim,
        softmax_scale,
    );
    let attention_max_abs_diff = verify_f32_close(
        &format!("{label} attention"),
        &attention_output,
        &expected_attention,
        1e-4_f32,
        1e-4_f32,
    )?;

    Ok(Qwen3SelfAttnModelLoopPreparedSequence {
        residual_sequence: original_residual_sequence,
        q_rope,
        k_rope,
        v_projected,
        q_gate,
        hidden,
        q_heads: shape.q_heads,
        kv_heads: shape.kv_heads,
        head_dim: shape.head_dim,
        value_dim: shape.value_dim,
        softmax_scale,
        q_projection_layout: prepared_q_projection_layout,
        q_gate_elements,
        output_gate_layout,
        q_norm_max_abs_diff,
        k_norm_max_abs_diff,
        q_rope_max_abs_diff,
        k_rope_max_abs_diff,
        attention_max_abs_diff,
    })
}

#[allow(clippy::too_many_arguments)]
fn qwen3_self_attn_prepare_sequence_smoke(
    context: &mut ullm_runtime_sys::RuntimeContext,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    self_attn_weights: &Qwen3SelfAttnRuntimeWeights,
    sequence_len: usize,
    rotary_dim: usize,
    rope_base: f32,
    position_offset: usize,
    q_norm: &PassthroughF32Data,
    k_norm: &PassthroughF32Data,
    label: &str,
) -> Result<Qwen3SelfAttnPreparedSequence, String> {
    let Qwen3SelfAttnRuntimeShape {
        hidden,
        q_heads: shape_q_heads,
        kv_heads,
        head_dim,
        value_dim,
        attention_width: _,
        q_projection_layout,
    } = qwen3_self_attn_runtime_shape(self_attn_weights)?;
    let q_cols = hidden;

    let base_input = deterministic_f32_vector(q_cols);
    let mut residual_sequence = Vec::with_capacity(sequence_len * q_cols);
    for timestep in 0..sequence_len {
        let step_input = linear_attn_step_input(&base_input, timestep);
        residual_sequence.extend_from_slice(&step_input);
    }
    let paged_block_size = 2_usize;
    let scheduled_paged_decode =
        allocate_fragmented_paged_decode_blocks(sequence_len, paged_block_size)?;
    let ScheduledPagedDecodeBlocks {
        block_table: paged_block_table,
        cache_blocks: paged_cache_blocks,
        allocator_stats: _,
        request_id: scheduler_request_id,
        prefill_tokens: scheduler_prefill_tokens,
        max_new_tokens: scheduler_max_new_tokens,
        cached_tokens: scheduler_cached_tokens,
        generated_tokens: scheduler_generated_tokens,
        active_len: scheduler_active_len,
    } = scheduled_paged_decode;

    let prepared = qwen3_self_attn_prepare_sequence_for_paged_decode_f32(
        context,
        stream,
        self_attn_weights,
        residual_sequence,
        sequence_len,
        &q_norm.values,
        &k_norm.values,
        rotary_dim,
        position_offset,
        rope_base,
        &paged_block_table,
        paged_block_size,
        paged_cache_blocks,
    )?;

    if q_projection_layout != prepared.prepared.q_projection_layout {
        return Err(
            "self-attn q projection layout changed between helper and runtime prepare".to_string(),
        );
    }

    let Qwen3SelfAttnRuntimePreparedSequenceForPagedDecode {
        residual_sequence,
        prepared:
            Qwen3SelfAttnRuntimePreparedSequence {
                q_query,
                k_projected,
                q_normed,
                k_normed,
                q_rope,
                k_rope,
                v_projected: prepared_v_projected,
                q_gate,
                attention_output,
                shape,
                softmax_scale,
                q_projection_layout,
                q_gate_elements,
                output_gate_layout,
            },
        paged_k_cache: expected_paged_k_cache,
        paged_v_cache: expected_paged_v_cache,
        paged_block_table,
        paged_block_size,
        paged_cache_blocks,
    } = prepared;

    if shape.q_heads != shape_q_heads {
        return Err(
            "self-attn q projection head count changed between helper and runtime prepare"
                .to_string(),
        );
    }
    let epsilon = 1e-5_f32;
    let mut expected_q_normed = Vec::with_capacity(q_query.len());
    for head_input in q_query.chunks_exact(shape.head_dim) {
        expected_q_normed.extend(runtime_host_rmsnorm_f32(
            head_input,
            &q_norm.values,
            epsilon,
        ));
    }
    let q_norm_max_abs_diff = verify_f32_close(
        &format!("{label} q_norm"),
        &q_normed,
        &expected_q_normed,
        1e-4_f32,
        1e-4_f32,
    )?;
    let mut expected_k_normed = Vec::with_capacity(k_normed.len());
    for head_input in k_projected.chunks_exact(shape.head_dim) {
        expected_k_normed.extend(runtime_host_rmsnorm_f32(
            head_input,
            &k_norm.values,
            epsilon,
        ));
    }
    let k_norm_max_abs_diff = verify_f32_close(
        &format!("{label} k_norm"),
        &k_normed,
        &expected_k_normed,
        1e-4_f32,
        1e-4_f32,
    )?;
    let expected_q_rope = runtime_host_rope_f32(
        &q_normed,
        sequence_len,
        shape.q_heads,
        shape.head_dim,
        rotary_dim,
        position_offset,
        rope_base,
    );
    let q_rope_max_abs_diff = verify_f32_close(
        &format!("{label} q_rope"),
        &q_rope,
        &expected_q_rope,
        1e-4_f32,
        1e-4_f32,
    )?;
    let expected_k_rope = runtime_host_rope_f32(
        &k_normed,
        sequence_len,
        shape.kv_heads,
        shape.head_dim,
        rotary_dim,
        position_offset,
        rope_base,
    );
    let k_rope_max_abs_diff = verify_f32_close(
        &format!("{label} k_rope"),
        &k_rope,
        &expected_k_rope,
        1e-4_f32,
        1e-4_f32,
    )?;

    let expected_attention = runtime_host_causal_attn_f32(
        &q_rope,
        &k_rope,
        &prepared_v_projected,
        sequence_len,
        shape.q_heads,
        shape.kv_heads,
        shape.head_dim,
        shape.value_dim,
        softmax_scale,
    );
    let attention_max_abs_diff = verify_f32_close(
        &format!("{label} attention"),
        &attention_output,
        &expected_attention,
        1e-4_f32,
        1e-4_f32,
    )?;

    Ok(Qwen3SelfAttnPreparedSequence {
        residual_sequence,
        q_rope,
        k_rope,
        v_projected: prepared_v_projected,
        q_gate,
        attention_output,
        expected_paged_k_cache,
        expected_paged_v_cache,
        hidden: q_cols,
        q_heads: shape.q_heads,
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
        paged_block_table,
        paged_block_size,
        paged_cache_blocks,
        scheduler_request_id,
        scheduler_prefill_tokens,
        scheduler_max_new_tokens,
        scheduler_cached_tokens,
        scheduler_generated_tokens,
        scheduler_active_len,
    })
}

struct Qwen3DecoderLayerRequestSequenceRun {
    output: Qwen3DecoderLayerSequenceOutput,
    scheduler_request_id: RequestId,
    scheduler_prefill_tokens: usize,
    scheduler_max_new_tokens: usize,
    scheduler_cached_tokens: usize,
    scheduler_generated_tokens: usize,
    scheduler_active_len: usize,
}

fn push_decoder_layer_step_output(
    output: &mut Qwen3DecoderLayerSequenceOutput,
    step: ullm_engine::decode_runner::Qwen3DecoderLayerDecodeBatchOutput,
) {
    output
        .attention_output
        .extend_from_slice(&step.attention_output);
    output
        .attention_projection_input
        .extend_from_slice(&step.attention_projection_input);
    output
        .projected_output
        .extend_from_slice(&step.projected_output);
    output.block_output.extend_from_slice(&step.block_output);
    output.post_normed.extend_from_slice(&step.post_normed);
    output.mlp_output.extend_from_slice(&step.mlp_output);
    output.layer_output.extend_from_slice(&step.layer_output);
}

#[allow(clippy::too_many_arguments)]
fn qwen3_decoder_layer_request_sequence_to_host_f32(
    layer_weights: &Qwen3DecoderLayerRuntimeWeights,
    context: &mut ullm_runtime_sys::RuntimeContext,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    shape: PagedDecodeShape,
    expected_block_table: &[u32],
    softmax_scale: f32,
    mlp_epsilon: f32,
    q_sequence: &[f32],
    k_sequence: &[f32],
    v_sequence: &[f32],
    output_gate_sequence: Option<&[f32]>,
    residual_sequence: &[f32],
    sequence_len: usize,
) -> Result<Qwen3DecoderLayerRequestSequenceRun, String> {
    if sequence_len == 0 {
        return Err("Qwen3 decoder layer request sequence length must be greater than zero".into());
    }
    let prepared_scheduler = prepare_fragmented_paged_decode_state(sequence_len, shape.block_size)?;
    if prepared_scheduler.block_table != expected_block_table {
        return Err(format!(
            "Qwen3 decoder layer request runner block table {:?} does not match prepared self-attn block table {:?}",
            prepared_scheduler.block_table, expected_block_table
        ));
    }
    if prepared_scheduler.cache_blocks != shape.cache_blocks {
        return Err(format!(
            "Qwen3 decoder layer request runner cache_blocks {} does not match shape cache_blocks {}",
            prepared_scheduler.cache_blocks, shape.cache_blocks
        ));
    }

    let q_token_elements = shape.q_elements()?;
    let k_token_elements = shape.k_token_elements()?;
    let v_token_elements = shape.v_token_elements()?;
    let attention_elements = shape.output_elements()?;
    let hidden = layer_weights.post_attention.hidden;
    let expected_q_len = sequence_len
        .checked_mul(q_token_elements)
        .ok_or_else(|| "Qwen3 decoder layer request q length overflows".to_string())?;
    let expected_k_len = sequence_len
        .checked_mul(k_token_elements)
        .ok_or_else(|| "Qwen3 decoder layer request k length overflows".to_string())?;
    let expected_v_len = sequence_len
        .checked_mul(v_token_elements)
        .ok_or_else(|| "Qwen3 decoder layer request v length overflows".to_string())?;
    let expected_residual_len = sequence_len
        .checked_mul(hidden)
        .ok_or_else(|| "Qwen3 decoder layer request residual length overflows".to_string())?;
    if q_sequence.len() != expected_q_len
        || k_sequence.len() != expected_k_len
        || v_sequence.len() != expected_v_len
        || residual_sequence.len() != expected_residual_len
    {
        return Err(format!(
            "Qwen3 decoder layer request sequence length mismatch: q={} expected_q={} k={} expected_k={} v={} expected_v={} residual={} expected_residual={}",
            q_sequence.len(),
            expected_q_len,
            k_sequence.len(),
            expected_k_len,
            v_sequence.len(),
            expected_v_len,
            residual_sequence.len(),
            expected_residual_len
        ));
    }
    if let Some(gate) = output_gate_sequence {
        let expected_gate_len = sequence_len
            .checked_mul(attention_elements)
            .ok_or_else(|| {
                "Qwen3 decoder layer request output gate length overflows".to_string()
            })?;
        if gate.len() != expected_gate_len {
            return Err(format!(
                "Qwen3 decoder layer request output gate length {} does not match expected {}",
                gate.len(),
                expected_gate_len
            ));
        }
    }

    let mut scheduler = prepared_scheduler.scheduler;
    let request_id = prepared_scheduler.request_id;
    let mut runner = Qwen3DecoderLayerRequestDecodeRunner::new();
    runner.insert_request(
        context,
        stream,
        request_id,
        layer_weights,
        shape,
        prepared_scheduler.block_table.clone(),
        softmax_scale,
        mlp_epsilon,
    )?;

    let mut output = Qwen3DecoderLayerSequenceOutput {
        attention_output: Vec::with_capacity(sequence_len * attention_elements),
        attention_projection_input: Vec::with_capacity(sequence_len * attention_elements),
        projected_output: Vec::with_capacity(sequence_len * hidden),
        block_output: Vec::with_capacity(sequence_len * hidden),
        post_normed: Vec::with_capacity(sequence_len * hidden),
        mlp_output: Vec::with_capacity(sequence_len * hidden),
        layer_output: Vec::with_capacity(sequence_len * hidden),
        paged_cache: PagedKvCacheReadback {
            k: Vec::new(),
            v: Vec::new(),
        },
    };

    for timestep in 0..prepared_scheduler.prefill_tokens {
        let q_start = timestep * q_token_elements;
        let k_start = timestep * k_token_elements;
        let v_start = timestep * v_token_elements;
        let gate_start = timestep * attention_elements;
        let residual_start = timestep * hidden;
        let step = runner.run_prefill_step(
            stream,
            Qwen3DecoderLayerDecodeBatchInput {
                request_id,
                q: &q_sequence[q_start..q_start + q_token_elements],
                k: &k_sequence[k_start..k_start + k_token_elements],
                v: &v_sequence[v_start..v_start + v_token_elements],
                output_gate: output_gate_sequence
                    .map(|gate| &gate[gate_start..gate_start + attention_elements]),
                residual: &residual_sequence[residual_start..residual_start + hidden],
            },
        )?;
        if step.cache_position != timestep || step.cache_len != timestep + 1 {
            return Err(format!(
                "Qwen3 decoder layer prefill step returned cache_position={} cache_len={} for timestep {}",
                step.cache_position, step.cache_len, timestep
            ));
        }
        push_decoder_layer_step_output(&mut output, step);
    }

    scheduler
        .complete_prefill(request_id)
        .map_err(|err| format!("failed to complete Qwen3 decoder layer request prefill: {err}"))?;

    for timestep in prepared_scheduler.prefill_tokens..sequence_len {
        let ready = scheduler
            .ready_decode_batch(1)
            .map_err(|err| format!("failed to ready Qwen3 decoder layer request batch: {err}"))?;
        let request = ready.first().ok_or_else(|| {
            format!("expected one ready Qwen3 decoder layer request at timestep {timestep}")
        })?;
        if request.request.id != request_id || request.cache_position != timestep {
            return Err(format!(
                "Qwen3 decoder layer ready request {:?} cache_position {} does not match request {:?} timestep {}",
                request.request.id, request.cache_position, request_id, timestep
            ));
        }
        let q_start = timestep * q_token_elements;
        let k_start = timestep * k_token_elements;
        let v_start = timestep * v_token_elements;
        let gate_start = timestep * attention_elements;
        let residual_start = timestep * hidden;
        let mut steps = runner.run_ready_batch(
            stream,
            &mut scheduler,
            &ready,
            &[Qwen3DecoderLayerDecodeBatchInput {
                request_id,
                q: &q_sequence[q_start..q_start + q_token_elements],
                k: &k_sequence[k_start..k_start + k_token_elements],
                v: &v_sequence[v_start..v_start + v_token_elements],
                output_gate: output_gate_sequence
                    .map(|gate| &gate[gate_start..gate_start + attention_elements]),
                residual: &residual_sequence[residual_start..residual_start + hidden],
            }],
        )?;
        let step = steps.pop().ok_or_else(|| {
            format!("Qwen3 decoder layer request runner produced no output at timestep {timestep}")
        })?;
        if step.request_id != request_id {
            return Err(format!(
                "Qwen3 decoder layer request runner output request {:?} does not match {:?}",
                step.request_id, request_id
            ));
        }
        push_decoder_layer_step_output(&mut output, step);
    }

    output.paged_cache = runner.read_cache_to_host(request_id, stream)?;
    let active = scheduler
        .active_request(request_id)
        .ok_or_else(|| "Qwen3 decoder layer request is not active after run".to_string())?;
    Ok(Qwen3DecoderLayerRequestSequenceRun {
        output,
        scheduler_request_id: request_id,
        scheduler_prefill_tokens: prepared_scheduler.prefill_tokens,
        scheduler_max_new_tokens: prepared_scheduler.max_new_tokens,
        scheduler_cached_tokens: active.cached_tokens,
        scheduler_generated_tokens: active.generated_tokens,
        scheduler_active_len: scheduler.active_len(),
    })
}

#[allow(clippy::too_many_arguments, dead_code)]
fn run_self_attn_block_sequence_smoke(
    context: &mut ullm_runtime_sys::RuntimeContext,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    self_attn_weights: &Qwen3SelfAttnRuntimeWeights,
    sequence_len: usize,
    rotary_dim: usize,
    rope_base: f32,
    position_offset: usize,
    q_norm: &PassthroughF32Data,
    k_norm: &PassthroughF32Data,
    label: &str,
) -> Result<SelfAttnBlockSmokeRun, String> {
    let prepared = qwen3_self_attn_prepare_sequence_smoke(
        context,
        stream,
        self_attn_weights,
        sequence_len,
        rotary_dim,
        rope_base,
        position_offset,
        q_norm,
        k_norm,
        label,
    )?;
    let Qwen3SelfAttnPreparedSequence {
        residual_sequence,
        q_rope,
        k_rope,
        v_projected,
        q_gate,
        attention_output: prepared_attention_output,
        expected_paged_k_cache,
        expected_paged_v_cache,
        hidden,
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
        paged_block_table,
        paged_block_size,
        paged_cache_blocks,
        scheduler_request_id,
        scheduler_prefill_tokens,
        scheduler_max_new_tokens,
        scheduler_cached_tokens,
        scheduler_generated_tokens,
        scheduler_active_len,
    } = prepared;

    let o_rows = self_attn_weights.o_rows;
    let o_cols = self_attn_weights.o_cols;
    let o_matrix_bytes = o_rows
        .checked_mul(o_cols)
        .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| "o projection matrix byte size overflows".to_string())?;
    let mut o_matrix_raw = vec![0_u8; o_matrix_bytes];
    self_attn_weights
        .o_matrix
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
    let q_token_elements = q_heads * head_dim;
    let attention_elements = q_heads * value_dim;
    let block_sequence_output = qwen3_self_attn_block_sequence_to_host_f32(
        context,
        stream,
        decode_shape,
        &paged_block_table,
        hidden,
        softmax_scale,
        &self_attn_weights.o_matrix,
        &q_rope,
        &k_rope,
        &v_projected,
        q_gate.as_deref(),
        &residual_sequence,
        sequence_len,
    )
    .map_err(|err| format!("failed to run {label} Qwen3 self-attn block sequence: {err}"))?;
    let attention_output = block_sequence_output.attention_output;
    let attention_projection_input = block_sequence_output.attention_projection_input;
    let attn_projected = block_sequence_output.projected_output;
    let block_output = block_sequence_output.block_output;
    let paged_cache = block_sequence_output.paged_cache;

    let mut expected_paged_step_attention_output =
        Vec::with_capacity(sequence_len * attention_elements);
    for timestep in 0..sequence_len {
        let q_start = timestep * q_token_elements;
        let q_end = q_start + q_token_elements;
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
        expected_paged_step_attention_output.extend_from_slice(&expected_step_output);
    }
    let expected_paged_projection_input = if let Some(gate) = q_gate.as_ref() {
        runtime_host_sigmoid_mul_f32(gate, &attention_output)
    } else {
        attention_output.clone()
    };
    let mut expected_paged_attn_projected = Vec::with_capacity(sequence_len * o_rows);
    for timestep in 0..sequence_len {
        let input_start = timestep * attention_elements;
        let input_end = input_start + attention_elements;
        expected_paged_attn_projected.extend(runtime_host_matvec_f32(
            &o_matrix_host,
            &expected_paged_projection_input[input_start..input_end],
            o_rows,
            o_cols,
        ));
    }
    let expected_runtime_block_output = runtime_host_add_f32(&residual_sequence, &attn_projected);

    let paged_step_attention_max_abs_diff = verify_f32_close(
        &format!("{label} Qwen3 self-attn step"),
        &attention_output,
        &expected_paged_step_attention_output,
        1e-4,
        1e-4,
    )?;
    let output_gate_max_abs_diff = verify_f32_close(
        &format!("{label} Qwen3 self-attn output gate"),
        &attention_projection_input,
        &expected_paged_projection_input,
        1e-5,
        1e-6,
    )?;
    let o_proj_max_abs_diff = verify_f32_close(
        &format!("{label} Qwen3 self-attn o projection"),
        &attn_projected,
        &expected_paged_attn_projected,
        1e-4,
        1e-5,
    )?;
    let block_max_abs_diff = verify_f32_close(
        &format!("{label} Qwen3 self-attn residual add"),
        &block_output,
        &expected_runtime_block_output,
        1e-5,
        1e-6,
    )?;

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
        &attention_output,
        &prepared_attention_output,
        1e-4,
        1e-4,
    )?;
    let causal_attention_projection_input = if let Some(gate) = q_gate.as_ref() {
        runtime_host_sigmoid_mul_f32(gate, &prepared_attention_output)
    } else {
        prepared_attention_output.clone()
    };
    let mut causal_attn_projected = Vec::with_capacity(sequence_len * o_rows);
    for timestep in 0..sequence_len {
        let input_start = timestep * attention_elements;
        let input_end = input_start + attention_elements;
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
        paged_block_table,
        paged_block_size,
        paged_cache_blocks,
        scheduler_request_id,
        scheduler_prefill_tokens,
        scheduler_max_new_tokens,
        scheduler_cached_tokens,
        scheduler_generated_tokens,
        scheduler_active_len,
        hidden,
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

    let layer_weights = qwen3_decoder_layer_runtime_weights_from_package(
        &mut context,
        &mut stream,
        path,
        chunk_bytes,
        &q_tensor,
        &k_tensor,
        &v_tensor,
        &o_tensor,
        &q_norm,
        &k_norm,
        &post_norm,
        &gate_tensor,
        &up_tensor,
        &down_tensor,
    )?;
    let self_attn = qwen3_self_attn_prepare_sequence_smoke(
        &mut context,
        &mut stream,
        &layer_weights.self_attn,
        sequence_len,
        rotary_dim,
        rope_base,
        position_offset,
        &q_norm,
        &k_norm,
        "package-self-attn-mlp-block-smoke",
    )?;

    let hidden = self_attn.hidden;
    let mlp_epsilon = 1e-5_f32;
    if layer_weights.post_attention.hidden != hidden {
        return Err(format!(
            "Qwen3 decoder layer runtime weight hidden mismatch: expected={hidden} got={}",
            layer_weights.post_attention.hidden
        ));
    }
    if layer_weights.post_attention.mlp.gate_rows != layer_weights.post_attention.intermediate
        || layer_weights.post_attention.mlp.gate_cols != hidden
    {
        return Err("Qwen3 decoder layer runtime MLP gate shape is inconsistent".to_string());
    }

    let (
        post_normed,
        mlp_output,
        layer_output,
        attention_output,
        attention_projection_input,
        attn_projected,
        layer_step_block_output,
        paged_kv_write_k_max_abs_diff,
        paged_kv_write_v_max_abs_diff,
        paged_step_attention_max_abs_diff,
        causal_paged_step_attention_max_abs_diff,
        output_gate_max_abs_diff,
        o_proj_max_abs_diff,
        block_max_abs_diff,
        causal_paged_block_max_abs_diff,
        post_norm_max_abs_diff,
        layer_block_max_abs_diff,
    ) = {
        let o_rows = layer_weights.self_attn.o_rows;
        let o_cols = layer_weights.self_attn.o_cols;
        let o_matrix_bytes = o_rows
            .checked_mul(o_cols)
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| "o projection matrix byte size overflows".to_string())?;
        let mut o_matrix_raw = vec![0_u8; o_matrix_bytes];
        layer_weights
            .self_attn
            .o_matrix
            .copy_to_host(0, &mut o_matrix_raw, Some(&mut stream))
            .map_err(|err| format!("failed to copy materialized o projection to host: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize after o projection host copy: {err}"))?;
        let o_matrix_host = decode_f32_le_values(&o_matrix_raw);

        let decode_shape = PagedDecodeShape {
            block_size: self_attn.paged_block_size,
            cache_blocks: self_attn.paged_cache_blocks,
            q_heads: self_attn.q_heads,
            kv_heads: self_attn.kv_heads,
            head_dim: self_attn.head_dim,
            value_dim: self_attn.value_dim,
        };
        let layer_sequence_run = qwen3_decoder_layer_request_sequence_to_host_f32(
            &layer_weights,
            &mut context,
            &mut stream,
            decode_shape,
            &self_attn.paged_block_table,
            self_attn.softmax_scale,
            mlp_epsilon,
            &self_attn.q_rope,
            &self_attn.k_rope,
            &self_attn.v_projected,
            self_attn.q_gate.as_deref(),
            &self_attn.residual_sequence,
            sequence_len,
        )?;
        if layer_sequence_run.scheduler_request_id != self_attn.scheduler_request_id
            || layer_sequence_run.scheduler_prefill_tokens != self_attn.scheduler_prefill_tokens
            || layer_sequence_run.scheduler_max_new_tokens != self_attn.scheduler_max_new_tokens
            || layer_sequence_run.scheduler_cached_tokens != self_attn.scheduler_cached_tokens
            || layer_sequence_run.scheduler_generated_tokens != self_attn.scheduler_generated_tokens
            || layer_sequence_run.scheduler_active_len != self_attn.scheduler_active_len
        {
            return Err(format!(
                "package-self-attn-mlp-block-smoke layer request runner scheduler progress mismatch: runner request={} prefill={} max_new={} cached={} generated={} active={} self_attn request={} prefill={} max_new={} cached={} generated={} active={}",
                layer_sequence_run.scheduler_request_id.0,
                layer_sequence_run.scheduler_prefill_tokens,
                layer_sequence_run.scheduler_max_new_tokens,
                layer_sequence_run.scheduler_cached_tokens,
                layer_sequence_run.scheduler_generated_tokens,
                layer_sequence_run.scheduler_active_len,
                self_attn.scheduler_request_id.0,
                self_attn.scheduler_prefill_tokens,
                self_attn.scheduler_max_new_tokens,
                self_attn.scheduler_cached_tokens,
                self_attn.scheduler_generated_tokens,
                self_attn.scheduler_active_len
            ));
        }
        let layer_sequence_output = layer_sequence_run.output;
        let attention_output = layer_sequence_output.attention_output;
        let attention_projection_input = layer_sequence_output.attention_projection_input;
        let attn_projected = layer_sequence_output.projected_output;
        let layer_step_block_output = layer_sequence_output.block_output;
        let post_normed = layer_sequence_output.post_normed;
        let mlp_output = layer_sequence_output.mlp_output;
        let layer_output = layer_sequence_output.layer_output;
        let layer_cache = layer_sequence_output.paged_cache;
        let paged_kv_write_k_max_abs_diff = verify_f32_close(
            "package-self-attn-mlp-block-smoke layer step paged k cache write",
            &layer_cache.k,
            &self_attn.expected_paged_k_cache,
            1e-5,
            1e-5,
        )?;
        let paged_kv_write_v_max_abs_diff = verify_f32_close(
            "package-self-attn-mlp-block-smoke layer step paged v cache write",
            &layer_cache.v,
            &self_attn.expected_paged_v_cache,
            1e-5,
            1e-5,
        )?;
        let q_token_elements = self_attn.q_heads * self_attn.head_dim;
        let attention_elements = self_attn.q_heads * self_attn.value_dim;

        let mut expected_paged_step_attention_output =
            Vec::with_capacity(sequence_len * attention_elements);
        for timestep in 0..sequence_len {
            let q_start = timestep
                .checked_mul(q_token_elements)
                .ok_or_else(|| "self-attn q slice start overflows".to_string())?;
            let q_end = q_start
                .checked_add(q_token_elements)
                .ok_or_else(|| "self-attn q slice end overflows".to_string())?;
            let expected_step_output = runtime_host_paged_decode_attn_f32(
                &self_attn.q_rope[q_start..q_end],
                &self_attn.expected_paged_k_cache,
                &self_attn.expected_paged_v_cache,
                &self_attn.paged_block_table,
                timestep + 1,
                self_attn.paged_block_size,
                self_attn.q_heads,
                self_attn.kv_heads,
                self_attn.head_dim,
                self_attn.value_dim,
                self_attn.softmax_scale,
            );
            expected_paged_step_attention_output.extend_from_slice(&expected_step_output);
        }
        let expected_paged_projection_input = if let Some(gate) = self_attn.q_gate.as_ref() {
            runtime_host_sigmoid_mul_f32(gate, &attention_output)
        } else {
            attention_output.clone()
        };
        let mut expected_paged_attn_projected = Vec::with_capacity(sequence_len * o_rows);
        for timestep in 0..sequence_len {
            let input_start = timestep
                .checked_mul(attention_elements)
                .ok_or_else(|| "attention start overflow".to_string())?;
            let input_end = input_start
                .checked_add(attention_elements)
                .ok_or_else(|| "attention end overflow".to_string())?;
            expected_paged_attn_projected.extend(runtime_host_matvec_f32(
                &o_matrix_host,
                &expected_paged_projection_input[input_start..input_end],
                o_rows,
                o_cols,
            ));
        }

        let expected_runtime_block_output =
            runtime_host_add_f32(&self_attn.residual_sequence, &attn_projected);
        verify_f32_close(
            "package-self-attn-mlp-block-smoke layer step attention block",
            &layer_step_block_output,
            &expected_runtime_block_output,
            1e-4,
            1e-5,
        )?;

        let paged_step_attention_max_abs_diff = verify_f32_close(
            "package-self-attn-mlp-block-smoke Qwen3 self-attn step",
            &attention_output,
            &expected_paged_step_attention_output,
            1e-4,
            1e-4,
        )?;
        let output_gate_max_abs_diff = verify_f32_close(
            "package-self-attn-mlp-block-smoke Qwen3 self-attn output gate",
            &attention_projection_input,
            &expected_paged_projection_input,
            1e-5,
            1e-6,
        )?;
        let o_proj_max_abs_diff = verify_f32_close(
            "package-self-attn-mlp-block-smoke Qwen3 self-attn o projection",
            &attn_projected,
            &expected_paged_attn_projected,
            1e-4,
            1e-5,
        )?;
        let block_max_abs_diff = verify_f32_close(
            "package-self-attn-mlp-block-smoke Qwen3 self-attn residual add",
            &layer_step_block_output,
            &expected_runtime_block_output,
            1e-5,
            1e-6,
        )?;

        let causal_paged_step_attention_max_abs_diff = verify_f32_close(
            "package-self-attn-mlp-block-smoke causal-vs-paged-step-attention",
            &attention_output,
            &self_attn.attention_output,
            1e-4,
            1e-4,
        )?;
        let causal_attention_projection_input = if let Some(gate) = self_attn.q_gate.as_ref() {
            runtime_host_sigmoid_mul_f32(gate, &self_attn.attention_output)
        } else {
            self_attn.attention_output.clone()
        };
        let mut causal_attn_projected = Vec::with_capacity(sequence_len * o_rows);
        for timestep in 0..sequence_len {
            let input_start = timestep
                .checked_mul(attention_elements)
                .ok_or_else(|| "causal attention start overflow".to_string())?;
            let input_end = input_start
                .checked_add(attention_elements)
                .ok_or_else(|| "causal attention end overflow".to_string())?;
            causal_attn_projected.extend(runtime_host_matvec_f32(
                &o_matrix_host,
                &causal_attention_projection_input[input_start..input_end],
                o_rows,
                o_cols,
            ));
        }
        let expected_causal_block_output =
            runtime_host_add_f32(&self_attn.residual_sequence, &causal_attn_projected);
        let causal_paged_block_max_abs_diff = verify_f32_close(
            "package-self-attn-mlp-block-smoke causal-vs-paged-block",
            &layer_step_block_output,
            &expected_causal_block_output,
            3e-3,
            2e-5,
        )?;

        let mut post_normed_expected = Vec::with_capacity(sequence_len * hidden);
        for timestep in 0..sequence_len {
            let start = timestep * hidden;
            let end = start + hidden;
            let expected = runtime_host_rmsnorm_f32(
                &layer_step_block_output[start..end],
                &post_norm.values,
                mlp_epsilon,
            );
            post_normed_expected.extend_from_slice(&expected);
        }

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
            attention_output,
            attention_projection_input,
            attn_projected,
            layer_step_block_output,
            paged_kv_write_k_max_abs_diff,
            paged_kv_write_v_max_abs_diff,
            paged_step_attention_max_abs_diff,
            causal_paged_step_attention_max_abs_diff,
            output_gate_max_abs_diff,
            o_proj_max_abs_diff,
            block_max_abs_diff,
            causal_paged_block_max_abs_diff,
            post_norm_max_abs_diff,
            layer_block_max_abs_diff,
        )
    };

    Ok(format!(
        "package-self-attn-mlp-block-smoke package={} layer={} q_tensor=\"{}\" k_tensor=\"{}\" v_tensor=\"{}\" o_tensor=\"{}\" q_norm_tensor=\"{}\" k_norm_tensor=\"{}\" post_norm_tensor=\"{}\" gate_tensor=\"{}\" up_tensor=\"{}\" down_tensor=\"{}\" hidden={} sequence_len={} paged_block_size={} paged_cache_blocks={} paged_block_table={:?} scheduler_request_id={} scheduler_prefill_tokens={} scheduler_max_new_tokens={} scheduler_cached_tokens={} scheduler_generated_tokens={} scheduler_active_len={} q_projection_layout={} q_gate_elements={} output_gate_layout={} q_heads={} kv_heads={} head_dim={} value_dim={} rotary_dim={} position_offset={} rope_base={} softmax_scale={:.9} q_norm_dtype={} k_norm_dtype={} post_norm_dtype={} backend={} device_index={} name=\"{}\" residual_preview={} attention_preview={} gated_attention_preview={} projected_preview={} attention_block_preview={} post_norm_preview={} mlp_output_preview={} layer_output_preview={} q_norm_max_abs_diff={:.9} k_norm_max_abs_diff={:.9} q_rope_max_abs_diff={:.9} k_rope_max_abs_diff={:.9} attention_max_abs_diff={:.9} paged_kv_write_k_max_abs_diff={:.9} paged_kv_write_v_max_abs_diff={:.9} paged_step_attention_max_abs_diff={:.9} causal_paged_step_attention_max_abs_diff={:.9} output_gate_max_abs_diff={:.9} o_proj_max_abs_diff={:.9} block_max_abs_diff={:.9} causal_paged_block_max_abs_diff={:.9} post_norm_max_abs_diff={:.9} layer_block_max_abs_diff={:.9} verified=true",
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
        self_attn.scheduler_request_id.0,
        self_attn.scheduler_prefill_tokens,
        self_attn.scheduler_max_new_tokens,
        self_attn.scheduler_cached_tokens,
        self_attn.scheduler_generated_tokens,
        self_attn.scheduler_active_len,
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
        format_f32_preview(&attention_output[..8.min(attention_output.len())]),
        format_f32_preview(&attention_projection_input[..8.min(attention_projection_input.len())]),
        format_f32_preview(&attn_projected[..8.min(attn_projected.len())]),
        format_f32_preview(&layer_step_block_output[..8.min(layer_step_block_output.len())]),
        format_f32_preview(&post_normed[..8.min(post_normed.len())]),
        format_f32_preview(&mlp_output[..8.min(mlp_output.len())]),
        format_f32_preview(&layer_output[..8.min(layer_output.len())]),
        self_attn.q_norm_max_abs_diff,
        self_attn.k_norm_max_abs_diff,
        self_attn.q_rope_max_abs_diff,
        self_attn.k_rope_max_abs_diff,
        self_attn.attention_max_abs_diff,
        paged_kv_write_k_max_abs_diff,
        paged_kv_write_v_max_abs_diff,
        paged_step_attention_max_abs_diff,
        causal_paged_step_attention_max_abs_diff,
        output_gate_max_abs_diff,
        o_proj_max_abs_diff,
        block_max_abs_diff,
        causal_paged_block_max_abs_diff,
        post_norm_max_abs_diff,
        layer_block_max_abs_diff,
    ))
}

fn package_self_attn_mlp_block_scheduler_smoke(
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
        eprintln!("package-self-attn-mlp-block-scheduler-smoke requires a .ullm.d path");
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
        Ok(value) if value >= 3 => value,
        Ok(_) => {
            eprintln!("sequence length must be at least three for scheduler layer smoke");
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

    match package_self_attn_mlp_block_scheduler_smoke_impl(
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
    ) {
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
fn package_self_attn_mlp_block_scheduler_smoke_impl(
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

    let layer_weights = qwen3_decoder_layer_runtime_weights_from_package(
        &mut context,
        &mut stream,
        path,
        chunk_bytes,
        &q_tensor,
        &k_tensor,
        &v_tensor,
        &o_tensor,
        &q_norm,
        &k_norm,
        &post_norm,
        &gate_tensor,
        &up_tensor,
        &down_tensor,
    )?;
    let runtime_shape = qwen3_self_attn_runtime_shape(&layer_weights.self_attn)?;
    if layer_weights.post_attention.hidden != runtime_shape.hidden {
        return Err(format!(
            "Qwen3 decoder layer runtime weight hidden mismatch: self_attn={} post_attention={}",
            runtime_shape.hidden, layer_weights.post_attention.hidden
        ));
    }
    if layer_weights.post_attention.mlp.gate_rows != layer_weights.post_attention.intermediate
        || layer_weights.post_attention.mlp.gate_cols != runtime_shape.hidden
    {
        return Err("Qwen3 decoder layer runtime MLP gate shape is inconsistent".to_string());
    }

    let block_size = 2_usize;
    let requests = vec![
        Request::new(201, sequence_len - 2, 2),
        Request::new(202, sequence_len - 1, 1),
        Request::new(203, 1, 0),
    ];
    let mut required_blocks = 0_usize;
    for request in &requests {
        let total_tokens = request
            .prompt_tokens
            .checked_add(request.max_new_tokens)
            .ok_or_else(|| format!("request {:?} total token count overflows", request.id))?;
        if total_tokens == 0 {
            return Err(format!("request {:?} has no tokens to decode", request.id));
        }
        let request_blocks = (total_tokens - 1) / block_size + 1;
        required_blocks = required_blocks
            .checked_add(request_blocks)
            .ok_or_else(|| "package scheduler layer required block count overflows".to_string())?;
    }
    let cache_blocks = required_blocks
        .checked_add(2)
        .ok_or_else(|| "package scheduler layer cache block count overflows".to_string())?;
    if cache_blocks > u32::MAX as usize || block_size > u32::MAX as usize {
        return Err(format!(
            "package scheduler layer block layout exceeds u32 range: cache_blocks={cache_blocks} block_size={block_size}"
        ));
    }
    let decode_shape = PagedDecodeShape {
        block_size,
        cache_blocks,
        q_heads: runtime_shape.q_heads,
        kv_heads: runtime_shape.kv_heads,
        head_dim: runtime_shape.head_dim,
        value_dim: runtime_shape.value_dim,
    };
    let q_token_elements = decode_shape.q_elements()?;
    let k_token_elements = decode_shape.k_token_elements()?;
    let v_token_elements = decode_shape.v_token_elements()?;
    let attention_elements = decode_shape.output_elements()?;
    let hidden = runtime_shape.hidden;
    let intermediate = layer_weights.post_attention.intermediate;
    let softmax_scale = 1.0_f32 / (runtime_shape.head_dim as f32).sqrt();
    let mlp_epsilon = 1e-5_f32;

    let mut scheduler = SchedulerState::with_block_size(cache_blocks as u32, block_size as u32);
    for request in &requests {
        scheduler.enqueue(request.clone());
    }
    let mut runner = Qwen3DecoderLayerRequestDecodeRunner::new();
    let mut allocated = scheduler
        .pop_prefill_batch_with_allocation(usize::MAX)
        .map_err(|err| format!("failed to allocate package scheduler layer batch: {err}"))?;
    if allocated.len() != requests.len() {
        return Err(format!(
            "package scheduler layer selected {} requests, expected {}",
            allocated.len(),
            requests.len()
        ));
    }

    let mut runs = Vec::with_capacity(allocated.len());
    let mut request_position_offsets = Vec::with_capacity(allocated.len());
    let mut q_gate_elements = Vec::with_capacity(allocated.len());
    let mut q_norm_max_abs_diff = 0.0_f32;
    let mut k_norm_max_abs_diff = 0.0_f32;
    let mut q_rope_max_abs_diff = 0.0_f32;
    let mut k_rope_max_abs_diff = 0.0_f32;
    let mut causal_attention_max_abs_diff = 0.0_f32;
    let mut q_projection_layout = None;
    let mut output_gate_layout = None;

    for (request_index, scheduled) in allocated.drain(..).enumerate() {
        let request = scheduled.request;
        let block_table = scheduled.allocation.blocks;
        let total_tokens = request
            .prompt_tokens
            .checked_add(request.max_new_tokens)
            .ok_or_else(|| format!("request {:?} total token count overflows", request.id))?;
        let request_position_offset = position_offset
            .checked_add(request_index.checked_mul(sequence_len).ok_or_else(|| {
                "package scheduler layer request position offset multiplier overflows".to_string()
            })?)
            .ok_or_else(|| {
                "package scheduler layer request position offset overflows".to_string()
            })?;
        request_position_offsets.push(request_position_offset);

        let prepared = qwen3_self_attn_prepare_sequence_smoke(
            &mut context,
            &mut stream,
            &layer_weights.self_attn,
            total_tokens,
            rotary_dim,
            rope_base,
            request_position_offset,
            &q_norm,
            &k_norm,
            &format!(
                "package-self-attn-mlp-block-scheduler-smoke request {:?}",
                request.id
            ),
        )?;
        if prepared.hidden != hidden
            || prepared.q_heads != runtime_shape.q_heads
            || prepared.kv_heads != runtime_shape.kv_heads
            || prepared.head_dim != runtime_shape.head_dim
            || prepared.value_dim != runtime_shape.value_dim
        {
            return Err(format!(
                "package scheduler layer prepared shape mismatch for {:?}: hidden={} q_heads={} kv_heads={} head_dim={} value_dim={}",
                request.id,
                prepared.hidden,
                prepared.q_heads,
                prepared.kv_heads,
                prepared.head_dim,
                prepared.value_dim
            ));
        }
        if let Some(layout) = q_projection_layout {
            if layout != prepared.q_projection_layout {
                return Err(format!(
                    "package scheduler layer q projection layout changed: {layout} vs {}",
                    prepared.q_projection_layout
                ));
            }
        } else {
            q_projection_layout = Some(prepared.q_projection_layout);
        }
        if let Some(layout) = output_gate_layout {
            if layout != prepared.output_gate_layout {
                return Err(format!(
                    "package scheduler layer output gate layout changed: {layout} vs {}",
                    prepared.output_gate_layout
                ));
            }
        } else {
            output_gate_layout = Some(prepared.output_gate_layout);
        }
        q_gate_elements.push(prepared.q_gate_elements);
        q_norm_max_abs_diff = q_norm_max_abs_diff.max(prepared.q_norm_max_abs_diff);
        k_norm_max_abs_diff = k_norm_max_abs_diff.max(prepared.k_norm_max_abs_diff);
        q_rope_max_abs_diff = q_rope_max_abs_diff.max(prepared.q_rope_max_abs_diff);
        k_rope_max_abs_diff = k_rope_max_abs_diff.max(prepared.k_rope_max_abs_diff);
        causal_attention_max_abs_diff =
            causal_attention_max_abs_diff.max(prepared.attention_max_abs_diff);

        let expected = qwen3_decoder_layer_sequence_to_host_f32(
            &layer_weights,
            &mut context,
            &mut stream,
            decode_shape,
            &block_table,
            softmax_scale,
            mlp_epsilon,
            &prepared.q_rope,
            &prepared.k_rope,
            &prepared.v_projected,
            prepared.q_gate.as_deref(),
            &prepared.residual_sequence,
            total_tokens,
        )?;
        runner.insert_request(
            &mut context,
            &mut stream,
            request.id,
            &layer_weights,
            decode_shape,
            block_table.clone(),
            softmax_scale,
            mlp_epsilon,
        )?;
        let mut run = SchedulerLayerDecodeRun {
            state: SchedulerLayerDecodeState {
                request_id: request.id,
                prompt_tokens: request.prompt_tokens,
                max_new_tokens: request.max_new_tokens,
                total_tokens,
                block_table,
                q_sequence: prepared.q_rope,
                k_sequence: prepared.k_rope,
                v_sequence: prepared.v_projected,
                output_gate_sequence: prepared.q_gate,
                residual_sequence: prepared.residual_sequence,
                decode_steps: 0,
            },
            checks: SchedulerLayerDecodeSmokeChecks::new(expected),
        };
        for timestep in 0..run.prompt_tokens {
            run_scheduler_layer_prefill_step(
                &mut runner,
                &mut stream,
                &mut run,
                timestep,
                q_token_elements,
                k_token_elements,
                v_token_elements,
                attention_elements,
                hidden,
                "package scheduler layer prefill",
            )?;
        }
        scheduler.complete_prefill(run.request_id).map_err(|err| {
            format!(
                "failed to complete package scheduler layer prefill {:?}: {err}",
                run.request_id
            )
        })?;
        runs.push(run);
    }

    let first_batch_ready = run_scheduler_layer_ready_batch(
        &mut runner,
        &mut scheduler,
        &mut runs,
        &mut stream,
        &[RequestId(201), RequestId(202)],
        8,
        q_token_elements,
        k_token_elements,
        v_token_elements,
        attention_elements,
        hidden,
        true,
        "package scheduler layer first batch",
    )?;
    let second_batch_ready = run_scheduler_layer_ready_batch(
        &mut runner,
        &mut scheduler,
        &mut runs,
        &mut stream,
        &[RequestId(201)],
        8,
        q_token_elements,
        k_token_elements,
        v_token_elements,
        attention_elements,
        hidden,
        true,
        "package scheduler layer second batch",
    )?;
    let final_ready = scheduler
        .ready_decode_batch(8)
        .map_err(|err| format!("failed to query package scheduler layer final ready batch: {err}"))?
        .len();
    if final_ready != 0 {
        return Err(format!(
            "package scheduler layer final ready count {final_ready}, expected 0"
        ));
    }

    for run in &mut runs {
        let active = scheduler.active_request(run.request_id).ok_or_else(|| {
            format!(
                "package scheduler layer request {:?} missing from active scheduler state",
                run.request_id
            )
        })?;
        if active.cached_tokens != run.total_tokens {
            return Err(format!(
                "package scheduler layer request {:?} cached_tokens {} did not match total_tokens {}",
                run.request_id, active.cached_tokens, run.total_tokens
            ));
        }
        if active.generated_tokens != run.max_new_tokens {
            return Err(format!(
                "package scheduler layer request {:?} generated_tokens {} did not match max_new_tokens {}",
                run.request_id, active.generated_tokens, run.max_new_tokens
            ));
        }
        let cache = runner
            .read_cache_to_host(run.request_id, &mut stream)
            .map_err(|err| {
                format!(
                    "failed to read package scheduler layer cache for {:?}: {err}",
                    run.request_id
                )
            })?;
        run.checks.k_cache_max_abs_diff = verify_f32_close(
            &format!(
                "package scheduler layer request {:?} k cache",
                run.request_id
            ),
            &cache.k,
            &run.checks.expected.paged_cache.k,
            1e-5,
            1e-5,
        )?;
        run.checks.v_cache_max_abs_diff = verify_f32_close(
            &format!(
                "package scheduler layer request {:?} v cache",
                run.request_id
            ),
            &cache.v,
            &run.checks.expected.paged_cache.v,
            1e-5,
            1e-5,
        )?;
    }

    let stats = scheduler.allocator_stats();
    let request_ids = runs.iter().map(|run| run.request_id.0).collect::<Vec<_>>();
    let block_tables = runs
        .iter()
        .map(|run| run.block_table.clone())
        .collect::<Vec<_>>();
    let cached_tokens = runs
        .iter()
        .map(|run| {
            scheduler
                .active_request(run.request_id)
                .map(|active| active.cached_tokens)
                .unwrap_or(0)
        })
        .collect::<Vec<_>>();
    let generated_tokens = runs
        .iter()
        .map(|run| {
            scheduler
                .active_request(run.request_id)
                .map(|active| active.generated_tokens)
                .unwrap_or(0)
        })
        .collect::<Vec<_>>();
    let prompt_tokens = runs.iter().map(|run| run.prompt_tokens).collect::<Vec<_>>();
    let max_new_tokens = runs
        .iter()
        .map(|run| run.max_new_tokens)
        .collect::<Vec<_>>();
    let total_tokens = runs.iter().map(|run| run.total_tokens).collect::<Vec<_>>();
    let decode_steps = runs.iter().map(|run| run.decode_steps).collect::<Vec<_>>();
    let attention_max_abs_diff = runs
        .iter()
        .map(|run| run.checks.attention_max_abs_diff)
        .fold(0.0_f32, f32::max);
    let projection_input_max_abs_diff = runs
        .iter()
        .map(|run| run.checks.projection_input_max_abs_diff)
        .fold(0.0_f32, f32::max);
    let projected_max_abs_diff = runs
        .iter()
        .map(|run| run.checks.projected_max_abs_diff)
        .fold(0.0_f32, f32::max);
    let block_max_abs_diff = runs
        .iter()
        .map(|run| run.checks.block_max_abs_diff)
        .fold(0.0_f32, f32::max);
    let post_norm_max_abs_diff = runs
        .iter()
        .map(|run| run.checks.post_norm_max_abs_diff)
        .fold(0.0_f32, f32::max);
    let mlp_max_abs_diff = runs
        .iter()
        .map(|run| run.checks.mlp_max_abs_diff)
        .fold(0.0_f32, f32::max);
    let layer_max_abs_diff = runs
        .iter()
        .map(|run| run.checks.layer_max_abs_diff)
        .fold(0.0_f32, f32::max);
    let k_cache_max_abs_diff = runs
        .iter()
        .map(|run| run.checks.k_cache_max_abs_diff)
        .fold(0.0_f32, f32::max);
    let v_cache_max_abs_diff = runs
        .iter()
        .map(|run| run.checks.v_cache_max_abs_diff)
        .fold(0.0_f32, f32::max);

    Ok(format!(
        "package-self-attn-mlp-block-scheduler-smoke package={} layer={} q_tensor=\"{}\" k_tensor=\"{}\" v_tensor=\"{}\" o_tensor=\"{}\" q_norm_tensor=\"{}\" k_norm_tensor=\"{}\" post_norm_tensor=\"{}\" gate_tensor=\"{}\" up_tensor=\"{}\" down_tensor=\"{}\" sequence_len={} request_count={} request_ids={:?} prompt_tokens={:?} max_new_tokens={:?} total_tokens={:?} request_position_offsets={:?} paged_block_size={} paged_cache_blocks={} block_tables={:?} first_batch_ready={} second_batch_ready={} final_ready={} decode_steps={:?} cached_tokens={:?} generated_tokens={:?} active_len={} free_blocks={} allocated_block_count={} free_runs={} largest_free_run={} hidden={} intermediate={} q_projection_layout={} q_gate_elements={:?} output_gate_layout={} q_heads={} kv_heads={} head_dim={} value_dim={} rotary_dim={} position_offset={} rope_base={} softmax_scale={softmax_scale:.9} mlp_epsilon={mlp_epsilon:.9} q_norm_dtype={} k_norm_dtype={} post_norm_dtype={} backend={} device_index={} name=\"{}\" q_norm_max_abs_diff={q_norm_max_abs_diff:.9} k_norm_max_abs_diff={k_norm_max_abs_diff:.9} q_rope_max_abs_diff={q_rope_max_abs_diff:.9} k_rope_max_abs_diff={k_rope_max_abs_diff:.9} causal_attention_max_abs_diff={causal_attention_max_abs_diff:.9} attention_max_abs_diff={attention_max_abs_diff:.9} projection_input_max_abs_diff={projection_input_max_abs_diff:.9} projected_max_abs_diff={projected_max_abs_diff:.9} block_max_abs_diff={block_max_abs_diff:.9} post_norm_max_abs_diff={post_norm_max_abs_diff:.9} mlp_max_abs_diff={mlp_max_abs_diff:.9} layer_max_abs_diff={layer_max_abs_diff:.9} k_cache_max_abs_diff={k_cache_max_abs_diff:.9} v_cache_max_abs_diff={v_cache_max_abs_diff:.9} verified=true",
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
        sequence_len,
        runs.len(),
        request_ids,
        prompt_tokens,
        max_new_tokens,
        total_tokens,
        request_position_offsets,
        block_size,
        cache_blocks,
        block_tables,
        first_batch_ready,
        second_batch_ready,
        final_ready,
        decode_steps,
        cached_tokens,
        generated_tokens,
        scheduler.active_len(),
        stats.free_blocks,
        stats.allocated_blocks,
        stats.free_runs,
        stats.largest_free_run,
        hidden,
        intermediate,
        q_projection_layout.unwrap_or("unknown"),
        q_gate_elements,
        output_gate_layout.unwrap_or("unknown"),
        runtime_shape.q_heads,
        runtime_shape.kv_heads,
        runtime_shape.head_dim,
        runtime_shape.value_dim,
        rotary_dim,
        position_offset,
        rope_base,
        q_norm.dtype,
        k_norm.dtype,
        post_norm.dtype,
        info.backend,
        device_index,
        info.name,
    ))
}

struct PackageModelLoopRequestPlan {
    scheduler: SchedulerState,
    requests: Vec<Request>,
    request_ids: Vec<u64>,
    prompt_tokens: Vec<usize>,
    max_new_tokens: Vec<usize>,
    total_tokens: Vec<usize>,
    block_tables: Vec<Vec<u32>>,
    initial_residuals: Vec<Vec<f32>>,
    block_size: usize,
    cache_blocks: usize,
}

#[derive(Default)]
struct PackageModelLoopPreparedDiffs {
    q_norm_max_abs_diff: f32,
    k_norm_max_abs_diff: f32,
    q_rope_max_abs_diff: f32,
    k_rope_max_abs_diff: f32,
    causal_attention_max_abs_diff: f32,
}

struct PackageModelLoopRuntimeDiffs {
    attention_max_abs_diff: f32,
    projection_input_max_abs_diff: f32,
    projected_max_abs_diff: f32,
    block_max_abs_diff: f32,
    post_norm_max_abs_diff: f32,
    mlp_max_abs_diff: f32,
    layer_max_abs_diff: f32,
    k_cache_max_abs_diff: f32,
    v_cache_max_abs_diff: f32,
}

struct PackageModelLoopLayerRunPlan {
    runs_by_layer: Vec<Vec<SchedulerLayerDecodeRun>>,
    q_projection_layouts: Vec<&'static str>,
    output_gate_layouts: Vec<&'static str>,
    q_gate_elements_by_layer: Vec<Vec<usize>>,
    prepared_diffs: PackageModelLoopPreparedDiffs,
}

struct PackageModelLoopExecutionPlan {
    decode: Qwen3PackageModelDecodePlan,
    max_decode_batch_requests: usize,
}

struct PackageModelLoopExecutionSummary {
    first_batch_ready: usize,
    second_batch_ready: usize,
    decode_batch_ready_counts: Vec<usize>,
    final_ready: usize,
}

struct PackageModelLoopSmokeRun {
    model: Qwen3PackageModelRuntime,
    request_plan: PackageModelLoopRequestPlan,
    layer_run_plan: PackageModelLoopLayerRunPlan,
    execution_plan: PackageModelLoopExecutionPlan,
    execution_summary: Option<PackageModelLoopExecutionSummary>,
    sequence_len: usize,
    rotary_dim: usize,
    rope_base: f32,
    position_offset: usize,
}

fn parse_package_model_loop_rotary_dim(
    model: &Qwen3PackageModelRuntime,
    rotary_dim: Option<String>,
) -> Result<usize, String> {
    let rotary_dim = match rotary_dim {
        Some(raw) => raw
            .parse::<usize>()
            .map_err(|err| format!("invalid rotary dim {raw:?}: {err}"))?,
        None => model.default_rotary_dim()?,
    };
    if rotary_dim == 0 || rotary_dim > model.head_dim || !rotary_dim.is_multiple_of(2) {
        return Err(format!(
            "rotary dim must be even and no greater than head_dim: rotary_dim={rotary_dim}, head_dim={}",
            model.head_dim
        ));
    }
    Ok(rotary_dim)
}

impl PackageModelLoopRequestPlan {
    fn new(sequence_len: usize, hidden: usize, block_size: usize) -> Result<Self, String> {
        if block_size == 0 {
            return Err("model-loop block size must be greater than zero".to_string());
        }
        let requests = vec![
            Request::new(201, sequence_len - 2, 2),
            Request::new(202, sequence_len - 1, 1),
            Request::new(203, 1, 0),
        ];

        let mut required_blocks = 0_usize;
        for request in &requests {
            let total_tokens = request
                .prompt_tokens
                .checked_add(request.max_new_tokens)
                .ok_or_else(|| format!("request {:?} total token count overflows", request.id))?;
            if total_tokens == 0 {
                return Err(format!("request {:?} has zero total tokens", request.id));
            }
            let request_blocks = (total_tokens - 1) / block_size + 1;
            required_blocks = required_blocks
                .checked_add(request_blocks)
                .ok_or_else(|| "model-loop required block count overflows".to_string())?;
        }
        let cache_blocks = required_blocks
            .checked_add(2)
            .ok_or_else(|| "model-loop cache block count overflows".to_string())?;
        if cache_blocks > u32::MAX as usize || block_size > u32::MAX as usize {
            return Err(format!(
                "model-loop block layout exceeds u32 range: cache_blocks={cache_blocks} block_size={block_size}"
            ));
        }

        let mut scheduler = SchedulerState::with_block_size(cache_blocks as u32, block_size as u32);
        for request in &requests {
            scheduler.enqueue(request.clone());
        }
        let mut allocated = scheduler
            .pop_prefill_batch_with_allocation(usize::MAX)
            .map_err(|err| format!("failed to allocate model-loop package batch: {err}"))?;
        if allocated.len() != requests.len() {
            return Err(format!(
                "model-loop selected {} requests, expected {}",
                allocated.len(),
                requests.len()
            ));
        }

        let mut request_ids = Vec::with_capacity(allocated.len());
        let mut prompt_tokens = Vec::with_capacity(allocated.len());
        let mut max_new_tokens = Vec::with_capacity(allocated.len());
        let mut total_tokens = Vec::with_capacity(allocated.len());
        let mut block_tables = Vec::with_capacity(allocated.len());
        let mut initial_residuals = Vec::with_capacity(allocated.len());
        for (request_index, scheduled) in allocated.drain(..).enumerate() {
            let request = scheduled.request;
            let request_total_tokens = request
                .prompt_tokens
                .checked_add(request.max_new_tokens)
                .ok_or_else(|| format!("request {:?} total token count overflows", request.id))?;
            let base_input = deterministic_f32_vector(hidden);
            let residual_elements = request_total_tokens.checked_mul(hidden).ok_or_else(|| {
                format!("request {:?} residual element count overflows", request.id)
            })?;
            let mut residual = Vec::with_capacity(residual_elements);
            for timestep in 0..request_total_tokens {
                let shifted_timestep = timestep
                    .checked_add(request_index.checked_mul(sequence_len).ok_or_else(|| {
                        "model-loop residual timestep multiplier overflows".to_string()
                    })?)
                    .ok_or_else(|| "model-loop residual timestep overflows".to_string())?;
                residual.extend(linear_attn_step_input(&base_input, shifted_timestep));
            }
            request_ids.push(request.id.0);
            prompt_tokens.push(request.prompt_tokens);
            max_new_tokens.push(request.max_new_tokens);
            total_tokens.push(request_total_tokens);
            block_tables.push(scheduled.allocation.blocks);
            initial_residuals.push(residual);
        }

        Ok(Self {
            scheduler,
            requests,
            request_ids,
            prompt_tokens,
            max_new_tokens,
            total_tokens,
            block_tables,
            initial_residuals,
            block_size,
            cache_blocks,
        })
    }

    fn request_count(&self) -> usize {
        self.requests.len()
    }

    fn complete_prefill_all(&mut self) -> Result<(), String> {
        for request in &self.requests {
            self.scheduler.complete_prefill(request.id).map_err(|err| {
                format!(
                    "failed to complete package model-loop prefill {:?}: {err}",
                    request.id
                )
            })?;
        }
        Ok(())
    }

    fn cached_tokens(&self) -> Result<Vec<usize>, String> {
        self.requests
            .iter()
            .map(|request| {
                self.scheduler
                    .active_request(request.id)
                    .map(|active| active.cached_tokens)
                    .ok_or_else(|| format!("request {:?} is not active", request.id))
            })
            .collect()
    }

    fn generated_tokens(&self) -> Result<Vec<usize>, String> {
        self.requests
            .iter()
            .map(|request| {
                self.scheduler
                    .active_request(request.id)
                    .map(|active| active.generated_tokens)
                    .ok_or_else(|| format!("request {:?} is not active", request.id))
            })
            .collect()
    }
}

impl PackageModelLoopPreparedDiffs {
    fn observe(&mut self, prepared: &Qwen3SelfAttnModelLoopPreparedSequence) {
        self.q_norm_max_abs_diff = self.q_norm_max_abs_diff.max(prepared.q_norm_max_abs_diff);
        self.k_norm_max_abs_diff = self.k_norm_max_abs_diff.max(prepared.k_norm_max_abs_diff);
        self.q_rope_max_abs_diff = self.q_rope_max_abs_diff.max(prepared.q_rope_max_abs_diff);
        self.k_rope_max_abs_diff = self.k_rope_max_abs_diff.max(prepared.k_rope_max_abs_diff);
        self.causal_attention_max_abs_diff = self
            .causal_attention_max_abs_diff
            .max(prepared.attention_max_abs_diff);
    }
}

impl PackageModelLoopLayerRunPlan {
    #[allow(clippy::too_many_arguments)]
    fn prepare(
        context: &mut ullm_runtime_sys::RuntimeContext,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        model: &Qwen3PackageModelRuntime,
        request_plan: &PackageModelLoopRequestPlan,
        decode_shape: PagedDecodeShape,
        sequence_len: usize,
        rotary_dim: usize,
        rope_base: f32,
        position_offset: usize,
    ) -> Result<Self, String> {
        let mut runs_by_layer: Vec<Vec<SchedulerLayerDecodeRun>> =
            Vec::with_capacity(model.layer_count());
        let mut q_projection_layouts = Vec::with_capacity(model.layer_count());
        let mut output_gate_layouts = Vec::with_capacity(model.layer_count());
        let mut q_gate_elements_by_layer = Vec::with_capacity(model.layer_count());
        let mut prepared_diffs = PackageModelLoopPreparedDiffs::default();

        for (layer_position, layer) in model.layers.iter().enumerate() {
            let mut runs = Vec::with_capacity(request_plan.request_count());
            let mut q_gate_elements = Vec::with_capacity(request_plan.request_count());
            for (request_index, request) in request_plan.requests.iter().enumerate() {
                let residual_sequence = if layer_position == 0 {
                    request_plan.initial_residuals[request_index].clone()
                } else {
                    runs_by_layer[layer_position - 1][request_index]
                        .checks
                        .expected
                        .layer_output
                        .clone()
                };
                let request_position_stride =
                    request_index.checked_mul(sequence_len).ok_or_else(|| {
                        "model-loop request position offset multiplier overflows".to_string()
                    })?;
                let request_position_offset = position_offset
                    .checked_add(request_position_stride)
                    .ok_or_else(|| "model-loop request position offset overflows".to_string())?;
                let prepared = qwen3_self_attn_prepare_model_loop_sequence_smoke(
                    context,
                    stream,
                    &layer.weights.self_attn,
                    residual_sequence,
                    request_plan.total_tokens[request_index],
                    rotary_dim,
                    rope_base,
                    request_position_offset,
                    &layer.input_norm,
                    &layer.q_norm,
                    &layer.k_norm,
                    &request_plan.block_tables[request_index],
                    request_plan.block_size,
                    request_plan.cache_blocks,
                    &format!(
                        "package-self-attn-mlp-block-model-loop-smoke layer {} request {:?}",
                        layer.layer_index, request.id
                    ),
                )?;
                if prepared.hidden != model.hidden
                    || prepared.q_heads != model.q_heads
                    || prepared.kv_heads != model.kv_heads
                    || prepared.head_dim != model.head_dim
                    || prepared.value_dim != model.value_dim
                {
                    return Err(format!(
                        "model-loop prepared shape mismatch for layer {} request {:?}",
                        layer.layer_index, request.id
                    ));
                }
                q_gate_elements.push(prepared.q_gate_elements);
                prepared_diffs.observe(&prepared);
                if request_index == 0 {
                    q_projection_layouts.push(prepared.q_projection_layout);
                    output_gate_layouts.push(prepared.output_gate_layout);
                }

                let expected = qwen3_decoder_layer_sequence_to_host_f32(
                    &layer.weights,
                    context,
                    stream,
                    decode_shape,
                    &request_plan.block_tables[request_index],
                    prepared.softmax_scale,
                    model.mlp_epsilon,
                    &prepared.q_rope,
                    &prepared.k_rope,
                    &prepared.v_projected,
                    prepared.q_gate.as_deref(),
                    &prepared.residual_sequence,
                    request_plan.total_tokens[request_index],
                )?;
                runs.push(SchedulerLayerDecodeRun {
                    state: SchedulerLayerDecodeState {
                        request_id: request.id,
                        prompt_tokens: request.prompt_tokens,
                        max_new_tokens: request.max_new_tokens,
                        total_tokens: request_plan.total_tokens[request_index],
                        block_table: request_plan.block_tables[request_index].clone(),
                        q_sequence: prepared.q_rope,
                        k_sequence: prepared.k_rope,
                        v_sequence: prepared.v_projected,
                        output_gate_sequence: prepared.q_gate,
                        residual_sequence: prepared.residual_sequence,
                        decode_steps: 0,
                    },
                    checks: SchedulerLayerDecodeSmokeChecks::new(expected),
                });
            }
            q_gate_elements_by_layer.push(q_gate_elements);
            runs_by_layer.push(runs);
        }

        Ok(Self {
            runs_by_layer,
            q_projection_layouts,
            output_gate_layouts,
            q_gate_elements_by_layer,
            prepared_diffs,
        })
    }

    fn decode_steps_by_layer(&self) -> Vec<Vec<usize>> {
        self.runs_by_layer
            .iter()
            .map(|runs| runs.iter().map(|run| run.decode_steps).collect::<Vec<_>>())
            .collect::<Vec<_>>()
    }

    fn runtime_diffs(&self) -> PackageModelLoopRuntimeDiffs {
        PackageModelLoopRuntimeDiffs {
            attention_max_abs_diff: self.max_run_diff(|run| run.checks.attention_max_abs_diff),
            projection_input_max_abs_diff: self
                .max_run_diff(|run| run.checks.projection_input_max_abs_diff),
            projected_max_abs_diff: self.max_run_diff(|run| run.checks.projected_max_abs_diff),
            block_max_abs_diff: self.max_run_diff(|run| run.checks.block_max_abs_diff),
            post_norm_max_abs_diff: self.max_run_diff(|run| run.checks.post_norm_max_abs_diff),
            mlp_max_abs_diff: self.max_run_diff(|run| run.checks.mlp_max_abs_diff),
            layer_max_abs_diff: self.max_run_diff(|run| run.checks.layer_max_abs_diff),
            k_cache_max_abs_diff: self.max_run_diff(|run| run.checks.k_cache_max_abs_diff),
            v_cache_max_abs_diff: self.max_run_diff(|run| run.checks.v_cache_max_abs_diff),
        }
    }

    fn stack_requests(&self) -> Vec<Vec<Qwen3PackageModelStackRequest<'_>>> {
        self.runs_by_layer
            .iter()
            .map(|runs| {
                runs.iter()
                    .map(|run| Qwen3PackageModelStackRequest {
                        request_id: run.request_id,
                        block_table: &run.block_table,
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    }

    fn max_run_diff<F>(&self, select: F) -> f32
    where
        F: Fn(&SchedulerLayerDecodeRun) -> f32,
    {
        self.runs_by_layer
            .iter()
            .flatten()
            .map(select)
            .fold(0.0_f32, f32::max)
    }
}

impl PackageModelLoopExecutionPlan {
    fn new(
        model: &Qwen3PackageModelRuntime,
        request_plan: &PackageModelLoopRequestPlan,
    ) -> Result<Self, String> {
        Ok(Self {
            decode: Qwen3PackageModelDecodePlan::from_model(
                model,
                request_plan.block_size,
                request_plan.cache_blocks,
            )?,
            max_decode_batch_requests: 8,
        })
    }

    fn execute(
        &self,
        context: &mut ullm_runtime_sys::RuntimeContext,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        model: &Qwen3PackageModelRuntime,
        request_plan: &mut PackageModelLoopRequestPlan,
        layer_run_plan: &mut PackageModelLoopLayerRunPlan,
    ) -> Result<PackageModelLoopExecutionSummary, String> {
        let mut layer_runner = self.build_layer_runner(context, stream, model, layer_run_plan)?;
        self.run_prefill_layers(&mut layer_runner, stream, model, layer_run_plan)?;
        request_plan.complete_prefill_all()?;

        let decode_batch_ready_counts =
            self.run_decode_batches(&mut layer_runner, stream, request_plan, layer_run_plan)?;
        let final_ready = self.final_ready(request_plan)?;
        self.verify_layer_caches(&layer_runner, stream, model, layer_run_plan)?;

        Ok(PackageModelLoopExecutionSummary {
            first_batch_ready: decode_batch_ready_counts.first().copied().unwrap_or(0),
            second_batch_ready: decode_batch_ready_counts.get(1).copied().unwrap_or(0),
            decode_batch_ready_counts,
            final_ready,
        })
    }

    fn build_layer_runner<'weights>(
        &self,
        context: &mut ullm_runtime_sys::RuntimeContext,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        model: &'weights Qwen3PackageModelRuntime,
        layer_run_plan: &PackageModelLoopLayerRunPlan,
    ) -> Result<Qwen3DecoderLayerStackRequestDecodeRunner<'weights>, String> {
        let layer_requests = layer_run_plan.stack_requests();
        qwen3_package_model_stack_runner(model, context, stream, self.decode, &layer_requests)
    }

    fn run_prefill_layers(
        &self,
        layer_runner: &mut Qwen3DecoderLayerStackRequestDecodeRunner<'_>,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        model: &Qwen3PackageModelRuntime,
        layer_run_plan: &mut PackageModelLoopLayerRunPlan,
    ) -> Result<(), String> {
        for layer_position in 0..layer_runner.layer_count() {
            for run in &mut layer_run_plan.runs_by_layer[layer_position] {
                for timestep in 0..run.prompt_tokens {
                    run_scheduler_layer_stack_prefill_step(
                        layer_runner,
                        layer_position,
                        stream,
                        run,
                        timestep,
                        self.decode,
                        &format!(
                            "package model-loop layer {} prefill",
                            model.layers[layer_position].layer_index
                        ),
                    )?;
                }
            }
        }
        Ok(())
    }

    fn run_decode_batches(
        &self,
        layer_runner: &mut Qwen3DecoderLayerStackRequestDecodeRunner<'_>,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        request_plan: &mut PackageModelLoopRequestPlan,
        layer_run_plan: &mut PackageModelLoopLayerRunPlan,
    ) -> Result<Vec<usize>, String> {
        let mut decode_batch_ready_counts = Vec::new();
        loop {
            let ready = request_plan
                .scheduler
                .ready_decode_batch(self.max_decode_batch_requests)
                .map_err(|err| format!("failed to query model-loop ready batch: {err}"))?;
            if ready.is_empty() {
                break;
            }
            let batch_index = decode_batch_ready_counts.len();
            let label = format!("package model-loop decode batch {batch_index}");
            let ready_count = run_scheduler_layer_stack_ready_batch(
                layer_runner,
                &mut request_plan.scheduler,
                &mut layer_run_plan.runs_by_layer,
                stream,
                &ready,
                self.decode,
                &label,
            )?;
            decode_batch_ready_counts.push(ready_count);
        }
        Ok(decode_batch_ready_counts)
    }

    fn final_ready(&self, request_plan: &PackageModelLoopRequestPlan) -> Result<usize, String> {
        let final_ready = request_plan
            .scheduler
            .ready_decode_batch(self.max_decode_batch_requests)
            .map_err(|err| format!("failed to query model-loop final ready batch: {err}"))?
            .len();
        if final_ready != 0 {
            return Err(format!(
                "package model-loop final ready count {final_ready}, expected 0"
            ));
        }
        Ok(final_ready)
    }

    fn verify_layer_caches(
        &self,
        layer_runner: &Qwen3DecoderLayerStackRequestDecodeRunner<'_>,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        model: &Qwen3PackageModelRuntime,
        layer_run_plan: &mut PackageModelLoopLayerRunPlan,
    ) -> Result<(), String> {
        for layer_position in 0..layer_runner.layer_count() {
            for run in &mut layer_run_plan.runs_by_layer[layer_position] {
                let cache = layer_runner
                    .read_layer_cache_to_host(layer_position, run.request_id, stream)
                    .map_err(|err| {
                        format!(
                            "failed to read package model-loop layer {} cache for {:?}: {err}",
                            model.layers[layer_position].layer_index, run.request_id
                        )
                    })?;
                run.checks.k_cache_max_abs_diff = verify_f32_close(
                    &format!(
                        "package model-loop layer {} request {:?} k cache",
                        model.layers[layer_position].layer_index, run.request_id
                    ),
                    &cache.k,
                    &run.checks.expected.paged_cache.k,
                    1e-5,
                    1e-5,
                )?;
                run.checks.v_cache_max_abs_diff = verify_f32_close(
                    &format!(
                        "package model-loop layer {} request {:?} v cache",
                        model.layers[layer_position].layer_index, run.request_id
                    ),
                    &cache.v,
                    &run.checks.expected.paged_cache.v,
                    1e-5,
                    1e-5,
                )?;
            }
        }
        Ok(())
    }
}

impl PackageModelLoopSmokeRun {
    #[allow(clippy::too_many_arguments)]
    fn new(
        context: &mut ullm_runtime_sys::RuntimeContext,
        stream: &mut ullm_runtime_sys::RuntimeStream,
        path: &str,
        chunk_bytes: usize,
        layer_indices: &[usize],
        sequence_len: usize,
        rotary_dim: Option<String>,
        rope_base: f32,
        position_offset: usize,
    ) -> Result<Self, String> {
        let model =
            Qwen3PackageModelRuntime::load(context, stream, path, chunk_bytes, layer_indices)?;
        let rotary_dim = parse_package_model_loop_rotary_dim(&model, rotary_dim)?;
        let block_size = 2_usize;
        let request_plan =
            PackageModelLoopRequestPlan::new(sequence_len, model.hidden, block_size)?;
        let execution_plan = PackageModelLoopExecutionPlan::new(&model, &request_plan)?;

        let layer_run_plan = PackageModelLoopLayerRunPlan::prepare(
            context,
            stream,
            &model,
            &request_plan,
            execution_plan.decode.decode_shape,
            sequence_len,
            rotary_dim,
            rope_base,
            position_offset,
        )?;

        Ok(Self {
            model,
            request_plan,
            layer_run_plan,
            execution_plan,
            execution_summary: None,
            sequence_len,
            rotary_dim,
            rope_base,
            position_offset,
        })
    }

    fn execute(
        &mut self,
        context: &mut ullm_runtime_sys::RuntimeContext,
        stream: &mut ullm_runtime_sys::RuntimeStream,
    ) -> Result<(), String> {
        if self.execution_summary.is_some() {
            return Err("package model-loop smoke run has already executed".to_string());
        }
        let summary = self.execution_plan.execute(
            context,
            stream,
            &self.model,
            &mut self.request_plan,
            &mut self.layer_run_plan,
        )?;
        self.execution_summary = Some(summary);
        Ok(())
    }

    fn format_output(
        &self,
        path: &str,
        device_index: u32,
        info: &ullm_runtime_sys::DeviceInfo,
    ) -> Result<String, String> {
        let execution_summary = self
            .execution_summary
            .as_ref()
            .ok_or_else(|| "package model-loop smoke run has not executed".to_string())?;
        let stats = self.request_plan.scheduler.allocator_stats();
        let cached_tokens = self.request_plan.cached_tokens()?;
        let generated_tokens = self.request_plan.generated_tokens()?;
        let decode_steps_by_layer = self.layer_run_plan.decode_steps_by_layer();
        let layer_indices = self.model.layer_indices();
        let input_norm_tensors = self
            .model
            .tensor_names_by_layer(|layer| &layer.input_norm_tensor);
        let q_tensors = self.model.tensor_names_by_layer(|layer| &layer.q_tensor);
        let k_tensors = self.model.tensor_names_by_layer(|layer| &layer.k_tensor);
        let v_tensors = self.model.tensor_names_by_layer(|layer| &layer.v_tensor);
        let o_tensors = self.model.tensor_names_by_layer(|layer| &layer.o_tensor);
        let q_norm_tensors = self
            .model
            .tensor_names_by_layer(|layer| &layer.q_norm_tensor);
        let k_norm_tensors = self
            .model
            .tensor_names_by_layer(|layer| &layer.k_norm_tensor);
        let post_norm_tensors = self
            .model
            .tensor_names_by_layer(|layer| &layer.post_norm_tensor);
        let gate_tensors = self.model.tensor_names_by_layer(|layer| &layer.gate_tensor);
        let up_tensors = self.model.tensor_names_by_layer(|layer| &layer.up_tensor);
        let down_tensors = self.model.tensor_names_by_layer(|layer| &layer.down_tensor);
        let input_norm_dtypes = self.model.input_norm_dtypes();
        let q_norm_dtypes = self.model.q_norm_dtypes();
        let k_norm_dtypes = self.model.k_norm_dtypes();
        let post_norm_dtypes = self.model.post_norm_dtypes();
        let prepared_diffs = &self.layer_run_plan.prepared_diffs;
        let runtime_diffs = self.layer_run_plan.runtime_diffs();
        let q_norm_max_abs_diff = prepared_diffs.q_norm_max_abs_diff;
        let k_norm_max_abs_diff = prepared_diffs.k_norm_max_abs_diff;
        let q_rope_max_abs_diff = prepared_diffs.q_rope_max_abs_diff;
        let k_rope_max_abs_diff = prepared_diffs.k_rope_max_abs_diff;
        let causal_attention_max_abs_diff = prepared_diffs.causal_attention_max_abs_diff;
        let attention_max_abs_diff = runtime_diffs.attention_max_abs_diff;
        let projection_input_max_abs_diff = runtime_diffs.projection_input_max_abs_diff;
        let projected_max_abs_diff = runtime_diffs.projected_max_abs_diff;
        let block_max_abs_diff = runtime_diffs.block_max_abs_diff;
        let post_norm_max_abs_diff = runtime_diffs.post_norm_max_abs_diff;
        let mlp_max_abs_diff = runtime_diffs.mlp_max_abs_diff;
        let layer_max_abs_diff = runtime_diffs.layer_max_abs_diff;
        let k_cache_max_abs_diff = runtime_diffs.k_cache_max_abs_diff;
        let v_cache_max_abs_diff = runtime_diffs.v_cache_max_abs_diff;

        Ok(format!(
            "package-self-attn-mlp-block-model-loop-smoke package={} layers={:?} input_norm_tensors={:?} q_tensors={:?} k_tensors={:?} v_tensors={:?} o_tensors={:?} q_norm_tensors={:?} k_norm_tensors={:?} post_norm_tensors={:?} gate_tensors={:?} up_tensors={:?} down_tensors={:?} sequence_len={} request_count={} request_ids={:?} prompt_tokens={:?} max_new_tokens={:?} total_tokens={:?} paged_block_size={} paged_cache_blocks={} block_tables={:?} first_batch_ready={} second_batch_ready={} decode_batch_ready_counts={:?} final_ready={} decode_steps_by_layer={:?} cached_tokens={:?} generated_tokens={:?} active_len={} free_blocks={} allocated_block_count={} free_runs={} largest_free_run={} hidden={} intermediate_by_layer={:?} q_projection_layouts={:?} q_gate_elements_by_layer={:?} output_gate_layouts={:?} q_heads={} kv_heads={} head_dim={} value_dim={} rotary_dim={} position_offset={} rope_base={} softmax_scale={:.9} mlp_epsilon={:.9} input_norm_dtypes={:?} q_norm_dtypes={:?} k_norm_dtypes={:?} post_norm_dtypes={:?} backend={} device_index={} name=\"{}\" q_norm_max_abs_diff={q_norm_max_abs_diff:.9} k_norm_max_abs_diff={k_norm_max_abs_diff:.9} q_rope_max_abs_diff={q_rope_max_abs_diff:.9} k_rope_max_abs_diff={k_rope_max_abs_diff:.9} causal_attention_max_abs_diff={causal_attention_max_abs_diff:.9} attention_max_abs_diff={attention_max_abs_diff:.9} projection_input_max_abs_diff={projection_input_max_abs_diff:.9} projected_max_abs_diff={projected_max_abs_diff:.9} block_max_abs_diff={block_max_abs_diff:.9} post_norm_max_abs_diff={post_norm_max_abs_diff:.9} mlp_max_abs_diff={mlp_max_abs_diff:.9} layer_max_abs_diff={layer_max_abs_diff:.9} k_cache_max_abs_diff={k_cache_max_abs_diff:.9} v_cache_max_abs_diff={v_cache_max_abs_diff:.9} verified=true",
            path,
            layer_indices,
            input_norm_tensors,
            q_tensors,
            k_tensors,
            v_tensors,
            o_tensors,
            q_norm_tensors,
            k_norm_tensors,
            post_norm_tensors,
            gate_tensors,
            up_tensors,
            down_tensors,
            self.sequence_len,
            self.request_plan.request_count(),
            self.request_plan.request_ids,
            self.request_plan.prompt_tokens,
            self.request_plan.max_new_tokens,
            self.request_plan.total_tokens,
            self.request_plan.block_size,
            self.request_plan.cache_blocks,
            self.request_plan.block_tables,
            execution_summary.first_batch_ready,
            execution_summary.second_batch_ready,
            execution_summary.decode_batch_ready_counts,
            execution_summary.final_ready,
            decode_steps_by_layer,
            cached_tokens,
            generated_tokens,
            self.request_plan.scheduler.active_len(),
            stats.free_blocks,
            stats.allocated_blocks,
            stats.free_runs,
            stats.largest_free_run,
            self.model.hidden,
            self.model.intermediates(),
            self.layer_run_plan.q_projection_layouts,
            self.layer_run_plan.q_gate_elements_by_layer,
            self.layer_run_plan.output_gate_layouts,
            self.model.q_heads,
            self.model.kv_heads,
            self.model.head_dim,
            self.model.value_dim,
            self.rotary_dim,
            self.position_offset,
            self.rope_base,
            self.model.softmax_scale,
            self.model.mlp_epsilon,
            input_norm_dtypes,
            q_norm_dtypes,
            k_norm_dtypes,
            post_norm_dtypes,
            info.backend,
            device_index,
            info.name,
        ))
    }
}

fn package_layer_golden_smoke(
    path: Option<String>,
    fixture_path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_index: Option<String>,
    rotary_dim: Option<String>,
    rope_base: Option<String>,
    position_offset: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-layer-golden-smoke requires a .ullm.d path");
        return ExitCode::from(2);
    };
    let Some(fixture_path) = fixture_path else {
        eprintln!("package-layer-golden-smoke requires a golden fixture directory");
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
    let fixture = match GoldenTensorFixture::load(&fixture_path) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let layer_index = match layer_index {
        Some(raw) => match parse_usize_value(&raw, "layer index") {
            Ok(value) => value,
            Err(code) => return code,
        },
        None => {
            if fixture.layers().len() == 1 {
                fixture.layers()[0].layer_index
            } else {
                eprintln!(
                    "package-layer-golden-smoke requires LAYER_INDEX when fixture has {} layers",
                    fixture.layers().len()
                );
                return ExitCode::from(2);
            }
        }
    };
    let rope_base = match parse_optional_f32(rope_base, 10_000_000.0, "rope base") {
        Ok(value) if value > 1.0 => value,
        Ok(_) => {
            eprintln!("rope base must be greater than one");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let default_position_offset = match fixture.metadata().position_ids.first() {
        Some(value) => match usize::try_from(*value) {
            Ok(value) => value,
            Err(_) => {
                eprintln!("golden fixture first position id does not fit usize");
                return ExitCode::from(1);
            }
        },
        None => 0,
    };
    let position_offset =
        match parse_optional_usize(position_offset, default_position_offset, "position offset") {
            Ok(value) => value,
            Err(code) => return code,
        };

    match package_layer_golden_smoke_impl(
        &path,
        &fixture_path,
        fixture,
        device_index,
        chunk_bytes,
        layer_index,
        rotary_dim,
        rope_base,
        position_offset,
    ) {
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
fn package_golden_prefix_smoke(
    path: Option<String>,
    fixture_path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_start: Option<String>,
    layer_end_exclusive: Option<String>,
    rotary_dim: Option<String>,
    rope_base: Option<String>,
    position_offset: Option<String>,
    report_path: Option<String>,
    run_mode: Option<String>,
    row_scale_overrides_path: Option<String>,
    input_dump_dir: Option<String>,
    sampled_token_indices: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-golden-prefix-smoke requires a .ullm.d path");
        return ExitCode::from(2);
    };
    let Some(fixture_path) = fixture_path else {
        eprintln!("package-golden-prefix-smoke requires a golden prefix fixture directory");
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
    let fixture = match GoldenTensorFixture::load(&fixture_path) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let (default_start, default_end_exclusive) = match golden_fixture_default_layer_range(&fixture)
    {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(1);
        }
    };
    let layer_start = match parse_optional_usize(layer_start, default_start, "layer start") {
        Ok(value) => value,
        Err(code) => return code,
    };
    let layer_end_exclusive =
        match parse_optional_usize(layer_end_exclusive, default_end_exclusive, "layer end") {
            Ok(value) => value,
            Err(code) => return code,
        };
    if layer_end_exclusive <= layer_start {
        eprintln!(
            "package-golden-prefix-smoke requires layer end greater than layer start: start={layer_start} end={layer_end_exclusive}"
        );
        return ExitCode::from(2);
    }
    let rope_base = match parse_optional_f32(rope_base, 10_000_000.0, "rope base") {
        Ok(value) if value > 1.0 => value,
        Ok(_) => {
            eprintln!("rope base must be greater than one");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let default_position_offset = match fixture.metadata().position_ids.first() {
        Some(value) => match usize::try_from(*value) {
            Ok(value) => value,
            Err(_) => {
                eprintln!("golden fixture first position id does not fit usize");
                return ExitCode::from(1);
            }
        },
        None => 0,
    };
    let position_offset =
        match parse_optional_usize(position_offset, default_position_offset, "position offset") {
            Ok(value) => value,
            Err(code) => return code,
        };
    let run_mode = match parse_package_golden_prefix_run_mode(run_mode.as_deref()) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let row_scale_overrides =
        match load_package_row_scale_overrides(row_scale_overrides_path.as_deref()) {
            Ok(value) => value,
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::from(2);
            }
        };
    let sampled_token_indices = match sampled_token_indices.as_deref() {
        Some("none") | None => Vec::new(),
        Some(raw) => match parse_usize_csv(raw, "sampled token indices") {
            Ok(value) => value,
            Err(code) => return code,
        },
    };
    let input_dump_dir = input_dump_dir.as_deref().filter(|raw| *raw != "none");

    match package_golden_prefix_smoke_impl(
        &path,
        &fixture_path,
        fixture,
        device_index,
        chunk_bytes,
        layer_start,
        layer_end_exclusive,
        rotary_dim,
        rope_base,
        position_offset,
        report_path.as_deref(),
        run_mode,
        row_scale_overrides.as_ref(),
        input_dump_dir,
        &sampled_token_indices,
    ) {
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
fn package_golden_prefix_smoke_impl(
    path: &str,
    fixture_path: &str,
    fixture: GoldenTensorFixture,
    device_index: u32,
    chunk_bytes: usize,
    layer_start: usize,
    layer_end_exclusive: usize,
    rotary_dim: Option<String>,
    rope_base: f32,
    position_offset: usize,
    report_path: Option<&str>,
    run_mode: PackageGoldenPrefixRunMode,
    row_scale_overrides: Option<&PackageRowScaleOverrides>,
    input_dump_dir: Option<&str>,
    sampled_token_indices: &[usize],
) -> Result<String, String> {
    let manifest_row_scale_override_count = list_tensor_payload_bundles(path)?
        .iter()
        .map(|bundle| bundle.row_scale_overrides.len())
        .sum::<usize>();
    let golden_layers = fixture.select_contiguous_layers(layer_start, layer_end_exclusive)?;
    let sequence_len = fixture.metadata().sequence_len;
    let hidden = fixture.metadata().hidden_size;
    if sequence_len == 0 || hidden == 0 {
        return Err(format!(
            "golden prefix fixture has invalid sequence_len={sequence_len} hidden_size={hidden}"
        ));
    }
    validate_golden_position_ids(
        &fixture.metadata().position_ids,
        sequence_len,
        position_offset,
    )?;
    for golden_layer in &golden_layers {
        validate_golden_hidden_shape(
            &golden_layer.before_shape,
            sequence_len,
            hidden,
            "golden prefix before hidden",
        )?;
        validate_golden_hidden_shape(
            &golden_layer.after_shape,
            sequence_len,
            hidden,
            "golden prefix after hidden",
        )?;
    }

    let expected_hidden_elements = sequence_len
        .checked_mul(hidden)
        .ok_or_else(|| "golden prefix hidden element count overflows".to_string())?;
    let mut current_hidden = fixture.read_initial_before_f32(layer_start)?;
    if current_hidden.len() != expected_hidden_elements {
        return Err(format!(
            "golden prefix initial payload element mismatch: got {} expected {expected_hidden_elements}",
            current_hidden.len()
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
    let block_size = sequence_len;
    let cache_blocks = 1_usize;
    let block_table = vec![0_u32];

    let mut report_entries = Vec::with_capacity(golden_layers.len());
    let mut max_mse = 0.0_f64;
    let mut max_mean_abs_diff = 0.0_f64;
    let mut max_abs_diff = 0.0_f64;
    let mut min_cosine_similarity = 1.0_f64;
    let mut self_attn_rotary_dim = None::<usize>;

    for (layer_position, golden_layer) in golden_layers.iter().enumerate() {
        let layer_index = golden_layer.layer_index;
        if current_hidden.len() != expected_hidden_elements {
            return Err(format!(
                "package golden prefix input length mismatch before layer {}: got {} expected {expected_hidden_elements}",
                layer_index,
                current_hidden.len()
            ));
        }
        let expected_after = fixture.read_layer_after_f32(layer_index)?;
        if expected_after.len() != expected_hidden_elements {
            return Err(format!(
                "golden prefix layer {} after payload element mismatch: got {} expected {expected_hidden_elements}",
                layer_index,
                expected_after.len()
            ));
        }
        let expected_before = fixture.read_layer_before_f32(layer_index)?;
        if expected_before.len() != expected_hidden_elements {
            return Err(format!(
                "golden prefix layer {} before payload element mismatch: got {} expected {expected_hidden_elements}",
                layer_index,
                expected_before.len()
            ));
        }

        let input_metrics = compare_f32_slices(&current_hidden, &expected_before)?;
        let input_preview_len = 8.min(current_hidden.len()).min(expected_before.len());
        let input_expected_preview = expected_before[..input_preview_len].to_vec();
        let input_actual_preview = current_hidden[..input_preview_len].to_vec();
        let input_diff_preview = current_hidden
            .iter()
            .zip(expected_before.iter())
            .take(input_preview_len)
            .map(|(actual, expected)| actual - expected)
            .collect::<Vec<_>>();
        let input_failure_class = package_golden_prefix_failure_class(&input_metrics);
        let input_distribution =
            package_hidden_distribution(&current_hidden, &expected_before, sequence_len, hidden)?;

        let layer_input = match run_mode {
            PackageGoldenPrefixRunMode::ActualPrefix => current_hidden.clone(),
            PackageGoldenPrefixRunMode::GoldenBeforeEachLayer => expected_before.clone(),
        };
        let layer_input_for_delta = layer_input.clone();
        let input_dump_file = match input_dump_dir {
            Some(dump_dir) => Some(write_package_prefix_input_dump(
                dump_dir,
                layer_index,
                run_mode,
                sequence_len,
                hidden,
                &layer_input_for_delta,
            )?),
            None => None,
        };

        let layer_kind = package_decoder_layer_kind(path, layer_index)?;
        let (actual, details) = match layer_kind {
            PackageDecoderLayerKind::SelfAttention => {
                let mut layer = qwen3_package_decoder_layer_runtime_from_package(
                    &mut context,
                    &mut stream,
                    path,
                    chunk_bytes,
                    layer_index,
                )?;
                if layer.runtime_shape.hidden != hidden {
                    return Err(format!(
                        "golden hidden_size {hidden} does not match package self-attn layer {} hidden {}",
                        layer_index, layer.runtime_shape.hidden
                    ));
                }
                let mut applied_row_scale_overrides = Vec::new();
                let self_attn_o_row_scale_overrides = matching_package_row_scale_overrides(
                    row_scale_overrides,
                    layer_index,
                    "self_attn.o_proj.weight",
                );
                applied_row_scale_overrides.extend(apply_package_row_scale_overrides_to_matrix(
                    &mut stream,
                    &mut layer.weights.self_attn.o_matrix,
                    layer.weights.self_attn.o_rows,
                    layer.weights.self_attn.o_cols,
                    &layer.o_tensor,
                    &self_attn_o_row_scale_overrides,
                )?);
                let down_row_scale_overrides = matching_package_row_scale_overrides(
                    row_scale_overrides,
                    layer_index,
                    "mlp.down_proj.weight",
                );
                applied_row_scale_overrides.extend(apply_package_row_scale_overrides_to_matrix(
                    &mut stream,
                    &mut layer.weights.post_attention.mlp.down_matrix,
                    layer.weights.post_attention.hidden,
                    layer.weights.post_attention.intermediate,
                    &layer.down_tensor,
                    &down_row_scale_overrides,
                )?);
                let input_norm_tensor =
                    format!("model.language_model.layers.{layer_index}.input_layernorm.weight");
                let mut input_norm =
                    read_named_passthrough_f32(path, &input_norm_tensor, chunk_bytes)?;
                input_norm.values =
                    effective_rmsnorm_weight_values(&input_norm_tensor, &input_norm.values);
                if input_norm.values.len() != hidden {
                    return Err(format!(
                        "self-attn input RMSNorm length {} does not match hidden={hidden}",
                        input_norm.values.len()
                    ));
                }
                let mut attention_input_normed = Vec::with_capacity(layer_input_for_delta.len());
                for residual in layer_input_for_delta.chunks_exact(hidden) {
                    attention_input_normed.extend(runtime_host_rmsnorm_f32(
                        residual,
                        &input_norm.values,
                        1e-6_f32,
                    ));
                }
                let rotary_dim = match rotary_dim.as_ref() {
                    Some(raw) => parse_package_layer_golden_rotary_dim(
                        layer.runtime_shape.head_dim,
                        Some(raw.clone()),
                    )?,
                    None => {
                        parse_package_layer_golden_rotary_dim(layer.runtime_shape.head_dim, None)?
                    }
                };
                self_attn_rotary_dim = Some(rotary_dim);
                let prepared = qwen3_self_attn_prepare_sequence_for_paged_decode_f32(
                    &mut context,
                    &mut stream,
                    &layer.weights.self_attn,
                    attention_input_normed.clone(),
                    sequence_len,
                    &layer.q_norm.values,
                    &layer.k_norm.values,
                    rotary_dim,
                    position_offset,
                    rope_base,
                    &block_table,
                    block_size,
                    cache_blocks,
                )?;
                let Qwen3SelfAttnRuntimePreparedSequenceForPagedDecode {
                    residual_sequence: _,
                    prepared:
                        Qwen3SelfAttnRuntimePreparedSequence {
                            q_query,
                            k_projected,
                            q_normed,
                            k_normed,
                            q_rope,
                            k_rope,
                            v_projected,
                            q_gate,
                            attention_output,
                            shape,
                            softmax_scale,
                            q_projection_layout,
                            q_gate_elements,
                            output_gate_layout,
                        },
                    paged_k_cache: _,
                    paged_v_cache: _,
                    paged_block_table,
                    paged_block_size,
                    paged_cache_blocks,
                } = prepared;
                let decode_shape = PagedDecodeShape {
                    block_size: paged_block_size,
                    cache_blocks: paged_cache_blocks,
                    q_heads: shape.q_heads,
                    kv_heads: shape.kv_heads,
                    head_dim: shape.head_dim,
                    value_dim: shape.value_dim,
                };
                let layer_output = qwen3_decoder_layer_sequence_to_host_f32(
                    &layer.weights,
                    &mut context,
                    &mut stream,
                    decode_shape,
                    &paged_block_table,
                    softmax_scale,
                    1e-5_f32,
                    &q_rope,
                    &k_rope,
                    &v_projected,
                    q_gate.as_deref(),
                    &layer_input_for_delta,
                    sequence_len,
                )?;
                let causal_attention_runtime_diagnostic =
                    package_self_attn_causal_attention_runtime_diagnostic(
                        &attention_output,
                        &layer_output.attention_output,
                        &layer_output.attention_projection_input,
                        &q_rope,
                        &k_rope,
                        &v_projected,
                        q_gate.as_deref(),
                        sequence_len,
                        shape.q_heads,
                        shape.kv_heads,
                        shape.head_dim,
                        shape.value_dim,
                        softmax_scale,
                    )?;
                let candidate_ids = package_layer_candidate_ids(path, &layer);
                let mut details = serde_json::Map::new();
                insert_json_detail(&mut details, "candidate_ids", candidate_ids);
                insert_json_detail(&mut details, "input_norm_tensor", &input_norm_tensor);
                insert_json_detail(&mut details, "q_tensor", &layer.q_tensor);
                insert_json_detail(&mut details, "k_tensor", &layer.k_tensor);
                insert_json_detail(&mut details, "v_tensor", &layer.v_tensor);
                insert_json_detail(&mut details, "o_tensor", &layer.o_tensor);
                insert_json_detail(&mut details, "gate_tensor", &layer.gate_tensor);
                insert_json_detail(&mut details, "up_tensor", &layer.up_tensor);
                insert_json_detail(&mut details, "down_tensor", &layer.down_tensor);
                insert_json_detail(&mut details, "q_heads", shape.q_heads);
                insert_json_detail(&mut details, "kv_heads", shape.kv_heads);
                insert_json_detail(&mut details, "head_dim", shape.head_dim);
                insert_json_detail(&mut details, "value_dim", shape.value_dim);
                insert_json_detail(&mut details, "rotary_dim", rotary_dim);
                insert_json_detail(&mut details, "position_offset", position_offset);
                insert_json_detail(&mut details, "rope_base", rope_base);
                insert_json_detail(&mut details, "block_size", paged_block_size);
                insert_json_detail(&mut details, "cache_blocks", paged_cache_blocks);
                insert_json_detail(&mut details, "block_table", paged_block_table);
                insert_json_detail(&mut details, "softmax_scale", softmax_scale);
                insert_json_detail(&mut details, "mlp_epsilon", 1e-5_f32);
                insert_json_detail(
                    &mut details,
                    "q_projection_layout",
                    q_projection_layout.to_string(),
                );
                insert_json_detail(&mut details, "q_gate_elements", q_gate_elements);
                insert_json_detail(
                    &mut details,
                    "output_gate_layout",
                    output_gate_layout.to_string(),
                );
                insert_json_detail(&mut details, "input_norm_dtype", &input_norm.dtype);
                insert_json_detail(&mut details, "q_norm_dtype", &layer.q_norm.dtype);
                insert_json_detail(&mut details, "k_norm_dtype", &layer.k_norm.dtype);
                insert_json_detail(&mut details, "post_norm_dtype", &layer.post_norm.dtype);
                if let Some(overrides) = row_scale_overrides {
                    insert_json_detail(
                        &mut details,
                        "row_scale_override_source",
                        &overrides.source_path,
                    );
                }
                if !applied_row_scale_overrides.is_empty() {
                    insert_json_detail(
                        &mut details,
                        "applied_row_scale_overrides",
                        &applied_row_scale_overrides,
                    );
                }
                insert_json_detail(
                    &mut details,
                    "causal_attention_runtime_diagnostic",
                    causal_attention_runtime_diagnostic,
                );
                let extra_hot_input_vectors = [
                    (
                        "attention_input_normed",
                        attention_input_normed.as_slice(),
                        hidden,
                    ),
                    ("attention_q_query", q_query.as_slice(), hidden),
                    (
                        "attention_k_projected",
                        k_projected.as_slice(),
                        shape.kv_heads * shape.head_dim,
                    ),
                    (
                        "attention_v_projected",
                        v_projected.as_slice(),
                        shape.kv_heads * shape.value_dim,
                    ),
                    ("attention_q_normed", q_normed.as_slice(), hidden),
                    (
                        "attention_k_normed",
                        k_normed.as_slice(),
                        shape.kv_heads * shape.head_dim,
                    ),
                    ("attention_q_rope", q_rope.as_slice(), hidden),
                    (
                        "attention_k_rope",
                        k_rope.as_slice(),
                        shape.kv_heads * shape.head_dim,
                    ),
                    ("attention_output", attention_output.as_slice(), hidden),
                ];
                let mut extra_hot_input_vectors = extra_hot_input_vectors.to_vec();
                if let Some(q_gate) = q_gate.as_ref() {
                    extra_hot_input_vectors.push(("attention_q_gate", q_gate.as_slice(), hidden));
                }
                insert_json_detail(
                    &mut details,
                    "module_contribution",
                    package_module_contribution_summary(
                        &layer_input_for_delta,
                        &expected_before,
                        &expected_after,
                        Some(&layer_output.attention_projection_input),
                        &layer_output.projected_output,
                        &layer_output.block_output,
                        &layer_output.post_normed,
                        None,
                        &extra_hot_input_vectors,
                        &layer_output.mlp_output,
                        &layer_output.layer_output,
                        sequence_len,
                        hidden,
                        sampled_token_indices,
                    )?,
                );
                (layer_output.layer_output, details)
            }
            PackageDecoderLayerKind::LinearAttention => {
                let run = package_linear_attn_mlp_block_sequence_run(
                    path,
                    device_index,
                    chunk_bytes,
                    layer_index,
                    sequence_len,
                    layer_input,
                    row_scale_overrides,
                )?;
                let mut details = serde_json::Map::new();
                insert_json_detail(&mut details, "runtime_line", &run.line);
                let runtime_metrics = package_runtime_line_metrics(&run.line);
                if !runtime_metrics.is_empty() {
                    insert_json_detail(&mut details, "runtime_metrics", runtime_metrics);
                }
                if let Some(overrides) = row_scale_overrides {
                    insert_json_detail(
                        &mut details,
                        "row_scale_override_source",
                        &overrides.source_path,
                    );
                }
                if !run.applied_row_scale_overrides.is_empty() {
                    insert_json_detail(
                        &mut details,
                        "applied_row_scale_overrides",
                        &run.applied_row_scale_overrides,
                    );
                }
                let extra_hot_input_vectors = [
                    (
                        "attention_input_normed",
                        run.attention_input_normed.as_slice(),
                        hidden,
                    ),
                    (
                        "attention_qkv_projection",
                        run.attention_qkv_projection.as_slice(),
                        run.attention_qkv_projection_dim,
                    ),
                    (
                        "attention_z_projection",
                        run.attention_z_projection.as_slice(),
                        hidden,
                    ),
                    (
                        "attention_gate_silu",
                        run.attention_gate_silu.as_slice(),
                        hidden,
                    ),
                    (
                        "attention_a_projection",
                        run.attention_a_projection.as_slice(),
                        run.attention_gate_dim,
                    ),
                    (
                        "attention_b_projection",
                        run.attention_b_projection.as_slice(),
                        run.attention_gate_dim,
                    ),
                    (
                        "attention_conv_pre_silu",
                        run.attention_conv_pre_silu.as_slice(),
                        run.attention_qkv_projection_dim,
                    ),
                    (
                        "attention_conv",
                        run.attention_conv.as_slice(),
                        run.attention_qkv_projection_dim,
                    ),
                    (
                        "attention_recurrent_q",
                        run.attention_recurrent_q.as_slice(),
                        run.attention_recurrent_qk_dim,
                    ),
                    (
                        "attention_recurrent_k",
                        run.attention_recurrent_k.as_slice(),
                        run.attention_recurrent_qk_dim,
                    ),
                    (
                        "attention_recurrent_v",
                        run.attention_recurrent_v.as_slice(),
                        hidden,
                    ),
                    (
                        "attention_gate",
                        run.attention_gate.as_slice(),
                        run.attention_gate_dim,
                    ),
                    (
                        "attention_beta",
                        run.attention_beta.as_slice(),
                        run.attention_gate_dim,
                    ),
                    (
                        "attention_recurrent",
                        run.attention_recurrent.as_slice(),
                        hidden,
                    ),
                    (
                        "attention_pre_gate_normed",
                        run.attention_normed.as_slice(),
                        hidden,
                    ),
                    ("attention_normed", run.attention_normed.as_slice(), hidden),
                ];
                insert_json_detail(
                    &mut details,
                    "candidate_ids",
                    package_linear_attn_candidate_ids(path, layer_index),
                );
                insert_json_detail(
                    &mut details,
                    "qkv_tensor",
                    format!(
                        "model.language_model.layers.{layer_index}.linear_attn.in_proj_qkv.weight"
                    ),
                );
                insert_json_detail(
                    &mut details,
                    "out_tensor",
                    format!(
                        "model.language_model.layers.{layer_index}.linear_attn.out_proj.weight"
                    ),
                );
                insert_json_detail(
                    &mut details,
                    "gate_tensor",
                    format!("model.language_model.layers.{layer_index}.mlp.gate_proj.weight"),
                );
                insert_json_detail(
                    &mut details,
                    "up_tensor",
                    format!("model.language_model.layers.{layer_index}.mlp.up_proj.weight"),
                );
                insert_json_detail(
                    &mut details,
                    "down_tensor",
                    format!("model.language_model.layers.{layer_index}.mlp.down_proj.weight"),
                );
                insert_json_detail(
                    &mut details,
                    "module_contribution",
                    package_module_contribution_summary(
                        &layer_input_for_delta,
                        &expected_before,
                        &expected_after,
                        Some(&run.attention_projection_input),
                        &run.attention_output,
                        &run.attention_block_output,
                        &run.post_normed,
                        Some((&run.mlp_activation, run.mlp_intermediate)),
                        &extra_hot_input_vectors,
                        &run.mlp_output,
                        &run.layer_output,
                        sequence_len,
                        hidden,
                        sampled_token_indices,
                    )?,
                );
                (run.layer_output, details)
            }
        };
        let metrics = compare_f32_slices(&actual, &expected_after)?;
        max_mse = max_mse.max(metrics.mse);
        max_mean_abs_diff = max_mean_abs_diff.max(metrics.mean_abs_diff);
        max_abs_diff = max_abs_diff.max(metrics.max_abs_diff);
        min_cosine_similarity = min_cosine_similarity.min(metrics.cosine_similarity);

        let preview_len = 8.min(expected_after.len()).min(actual.len());
        let expected_preview = expected_after[..preview_len].to_vec();
        let actual_preview = actual[..preview_len].to_vec();
        let diff_preview = actual
            .iter()
            .zip(expected_after.iter())
            .take(preview_len)
            .map(|(actual, expected)| actual - expected)
            .collect::<Vec<_>>();
        let failure_class = package_golden_prefix_failure_class(&metrics);
        let mut details = details;
        if let Some(input_dump_file) = input_dump_file.as_ref() {
            insert_json_detail(&mut details, "input_dump_file", input_dump_file);
        }
        insert_json_detail(
            &mut details,
            "manifest_row_scale_override_count",
            manifest_row_scale_override_count,
        );
        let output_distribution =
            package_hidden_distribution(&actual, &expected_after, sequence_len, hidden)?;
        insert_json_detail(&mut details, "input_distribution", input_distribution);
        insert_json_detail(&mut details, "output_distribution", output_distribution);
        append_package_golden_prefix_report_entry(
            &mut report_entries,
            path,
            fixture_path,
            fixture.metadata().fixture_kind.as_deref(),
            device_index,
            &info.backend.to_string(),
            &info.name,
            layer_position,
            layer_index,
            layer_kind.as_str(),
            layer_start,
            layer_end_exclusive,
            sequence_len,
            hidden,
            run_mode,
            &input_metrics,
            input_failure_class,
            input_expected_preview,
            input_actual_preview,
            input_diff_preview,
            &metrics,
            failure_class,
            expected_preview,
            actual_preview,
            diff_preview,
            details,
        );

        current_hidden = actual;
    }

    if let Some(report_path) = report_path {
        write_jsonl_report(report_path, &report_entries)?;
    }

    Ok(format!(
        "package-golden-prefix-smoke package={} fixture={} layers={}..{} layer_count={} sequence_len={} hidden={} run_mode={} block_size={} cache_blocks={} block_table={:?} rotary_dim={} position_offset={} rope_base={} row_scale_overrides={} manifest_row_scale_overrides={} input_dump_dir={} sampled_tokens={} backend={} device_index={} name=\"{}\" max_mse={:.12} max_mean_abs_diff={:.9} max_abs_diff={:.9} min_cosine_similarity={:.9} report={} verified=true",
        path,
        fixture_path,
        layer_start,
        layer_end_exclusive,
        golden_layers.len(),
        sequence_len,
        hidden,
        run_mode.as_str(),
        block_size,
        cache_blocks,
        block_table,
        self_attn_rotary_dim
            .map(|value| value.to_string())
            .unwrap_or_else(|| "none".to_string()),
        position_offset,
        rope_base,
        row_scale_overrides
            .map(|overrides| overrides.source_path.as_str())
            .unwrap_or("none"),
        manifest_row_scale_override_count,
        input_dump_dir.unwrap_or("none"),
        if sampled_token_indices.is_empty() {
            "none".to_string()
        } else {
            sampled_token_indices
                .iter()
                .map(|value| value.to_string())
                .collect::<Vec<_>>()
                .join(",")
        },
        info.backend,
        device_index,
        info.name,
        max_mse,
        max_mean_abs_diff,
        max_abs_diff,
        min_cosine_similarity,
        report_path.unwrap_or("none"),
    ))
}

fn golden_fixture_default_layer_range(
    fixture: &GoldenTensorFixture,
) -> Result<(usize, usize), String> {
    if let (Some(start), Some(end_exclusive)) = (
        fixture.metadata().layer_start,
        fixture.metadata().layer_end_exclusive,
    ) {
        if end_exclusive <= start {
            return Err(format!(
                "golden fixture metadata has invalid layer range: start={start}, end_exclusive={end_exclusive}"
            ));
        }
        return Ok((start, end_exclusive));
    }

    let min_layer = fixture
        .layers()
        .iter()
        .map(|layer| layer.layer_index)
        .min()
        .ok_or_else(|| "golden fixture has no layer entries".to_string())?;
    let max_layer = fixture
        .layers()
        .iter()
        .map(|layer| layer.layer_index)
        .max()
        .ok_or_else(|| "golden fixture has no layer entries".to_string())?;
    let end_exclusive = max_layer
        .checked_add(1)
        .ok_or_else(|| "golden fixture max layer index overflows".to_string())?;
    Ok((min_layer, end_exclusive))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PackageDecoderLayerKind {
    SelfAttention,
    LinearAttention,
}

impl PackageDecoderLayerKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::SelfAttention => "self_attention",
            Self::LinearAttention => "linear_attention",
        }
    }
}

fn package_decoder_layer_kind(
    path: &str,
    layer_index: usize,
) -> Result<PackageDecoderLayerKind, String> {
    let self_attn_q = format!("model.language_model.layers.{layer_index}.self_attn.q_proj.weight");
    if select_tensor_payload_bundle(path, &TensorSelector::Name(self_attn_q)).is_ok() {
        return Ok(PackageDecoderLayerKind::SelfAttention);
    }

    let linear_qkv =
        format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_qkv.weight");
    if select_tensor_payload_bundle(path, &TensorSelector::Name(linear_qkv)).is_ok() {
        return Ok(PackageDecoderLayerKind::LinearAttention);
    }

    Err(format!(
        "package layer {layer_index} has neither supported self_attn nor linear_attn package tensors"
    ))
}

fn package_linear_attn_candidate_ids(path: &str, layer_index: usize) -> Vec<String> {
    [
        format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_qkv.weight"),
        format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_a.weight"),
        format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_b.weight"),
        format!("model.language_model.layers.{layer_index}.linear_attn.in_proj_z.weight"),
        format!("model.language_model.layers.{layer_index}.linear_attn.out_proj.weight"),
        format!("model.language_model.layers.{layer_index}.mlp.gate_proj.weight"),
        format!("model.language_model.layers.{layer_index}.mlp.up_proj.weight"),
        format!("model.language_model.layers.{layer_index}.mlp.down_proj.weight"),
    ]
    .iter()
    .map(|tensor_name| {
        select_tensor_payload_bundle(path, &TensorSelector::Name(tensor_name.clone()))
            .ok()
            .and_then(|bundle| bundle.candidate_id)
            .unwrap_or_else(|| "unknown".to_string())
    })
    .collect()
}

fn insert_json_detail<T: serde::Serialize>(
    details: &mut serde_json::Map<String, serde_json::Value>,
    key: &str,
    value: T,
) {
    details.insert(key.to_string(), serde_json::json!(value));
}

fn package_runtime_line_metrics(line: &str) -> serde_json::Map<String, serde_json::Value> {
    let mut metrics = serde_json::Map::new();
    for token in line.split_whitespace() {
        let Some((key, raw_value)) = token.split_once('=') else {
            continue;
        };
        if !package_runtime_line_metric_key(key) {
            continue;
        }
        let value = raw_value.trim_matches('"').trim_end_matches(',');
        if value == "true" || value == "false" {
            insert_json_detail(&mut metrics, key, value == "true");
        } else if let Ok(value) = value.parse::<i64>() {
            insert_json_detail(&mut metrics, key, value);
        } else if let Ok(value) = value.parse::<f64>() {
            insert_json_detail(&mut metrics, key, value);
        }
    }
    metrics
}

fn package_runtime_line_metric_key(key: &str) -> bool {
    matches!(
        key,
        "hidden"
            | "key_heads"
            | "value_heads"
            | "key_dim"
            | "value_dim"
            | "sequence_len"
            | "kernel_size"
            | "q_scale"
            | "device_index"
            | "verified"
    ) || key.ends_with("_max_abs_diff")
        || key.ends_with("_mse")
        || key.ends_with("_mean_abs_diff")
        || key.ends_with("_cosine_similarity")
}

#[allow(clippy::too_many_arguments)]
fn package_self_attn_causal_attention_runtime_diagnostic(
    prepared_attention_output: &[f32],
    layer_attention_output: &[f32],
    layer_attention_projection_input: &[f32],
    q_rope: &[f32],
    k_rope: &[f32],
    v_projected: &[f32],
    q_gate: Option<&[f32]>,
    sequence_len: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
) -> Result<serde_json::Map<String, serde_json::Value>, String> {
    let attention_width = q_heads
        .checked_mul(value_dim)
        .ok_or_else(|| "self-attn diagnostic attention width overflows".to_string())?;
    let expected_attention_elements = sequence_len
        .checked_mul(attention_width)
        .ok_or_else(|| "self-attn diagnostic attention element count overflows".to_string())?;
    let expected_q_elements = sequence_len
        .checked_mul(q_heads)
        .and_then(|value| value.checked_mul(head_dim))
        .ok_or_else(|| "self-attn diagnostic q element count overflows".to_string())?;
    let expected_k_elements = sequence_len
        .checked_mul(kv_heads)
        .and_then(|value| value.checked_mul(head_dim))
        .ok_or_else(|| "self-attn diagnostic k element count overflows".to_string())?;
    let expected_v_elements = sequence_len
        .checked_mul(kv_heads)
        .and_then(|value| value.checked_mul(value_dim))
        .ok_or_else(|| "self-attn diagnostic v element count overflows".to_string())?;
    for (label, values, expected) in [
        (
            "prepared_attention_output",
            prepared_attention_output,
            expected_attention_elements,
        ),
        (
            "layer_attention_output",
            layer_attention_output,
            expected_attention_elements,
        ),
        (
            "layer_attention_projection_input",
            layer_attention_projection_input,
            expected_attention_elements,
        ),
        ("q_rope", q_rope, expected_q_elements),
        ("k_rope", k_rope, expected_k_elements),
        ("v_projected", v_projected, expected_v_elements),
    ] {
        if values.len() != expected {
            return Err(format!(
                "self-attn diagnostic {label} length mismatch: got {} expected {expected}",
                values.len()
            ));
        }
    }
    if let Some(gate) = q_gate {
        if gate.len() != expected_attention_elements {
            return Err(format!(
                "self-attn diagnostic q_gate length mismatch: got {} expected {expected_attention_elements}",
                gate.len()
            ));
        }
    }

    let host_attention_output = runtime_host_causal_attn_f32(
        q_rope,
        k_rope,
        v_projected,
        sequence_len,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
    );
    if host_attention_output.len() != expected_attention_elements {
        return Err(format!(
            "self-attn diagnostic host attention length mismatch: got {} expected {expected_attention_elements}",
            host_attention_output.len()
        ));
    }

    let prepared_projection_input = package_attention_projection_input_from_gate(
        q_gate,
        prepared_attention_output,
        expected_attention_elements,
        "prepared",
    )?;
    let layer_projection_input_from_attention = package_attention_projection_input_from_gate(
        q_gate,
        layer_attention_output,
        expected_attention_elements,
        "layer",
    )?;
    let host_projection_input = package_attention_projection_input_from_gate(
        q_gate,
        &host_attention_output,
        expected_attention_elements,
        "host",
    )?;

    let mut diagnostic = serde_json::Map::new();
    insert_json_detail(&mut diagnostic, "sequence_len", sequence_len);
    insert_json_detail(&mut diagnostic, "attention_width", attention_width);
    insert_json_detail(&mut diagnostic, "q_heads", q_heads);
    insert_json_detail(&mut diagnostic, "kv_heads", kv_heads);
    insert_json_detail(&mut diagnostic, "head_dim", head_dim);
    insert_json_detail(&mut diagnostic, "value_dim", value_dim);
    insert_json_detail(&mut diagnostic, "softmax_scale", softmax_scale);
    insert_json_detail(&mut diagnostic, "has_q_gate", q_gate.is_some());
    insert_json_detail(
        &mut diagnostic,
        "prepared_attention_vs_host_causal",
        package_hidden_distribution(
            prepared_attention_output,
            &host_attention_output,
            sequence_len,
            attention_width,
        )?,
    );
    insert_json_detail(
        &mut diagnostic,
        "layer_attention_vs_host_causal",
        package_hidden_distribution(
            layer_attention_output,
            &host_attention_output,
            sequence_len,
            attention_width,
        )?,
    );
    insert_json_detail(
        &mut diagnostic,
        "layer_attention_vs_prepared_attention",
        package_hidden_distribution(
            layer_attention_output,
            prepared_attention_output,
            sequence_len,
            attention_width,
        )?,
    );
    insert_json_detail(
        &mut diagnostic,
        "layer_projection_input_vs_host_projection_input",
        package_hidden_distribution(
            layer_attention_projection_input,
            &host_projection_input,
            sequence_len,
            attention_width,
        )?,
    );
    insert_json_detail(
        &mut diagnostic,
        "layer_projection_input_vs_prepared_projection_input",
        package_hidden_distribution(
            layer_attention_projection_input,
            &prepared_projection_input,
            sequence_len,
            attention_width,
        )?,
    );
    insert_json_detail(
        &mut diagnostic,
        "layer_projection_input_vs_layer_attention_gate_replay",
        package_hidden_distribution(
            layer_attention_projection_input,
            &layer_projection_input_from_attention,
            sequence_len,
            attention_width,
        )?,
    );
    insert_json_detail(
        &mut diagnostic,
        "sample_locations",
        package_self_attn_causal_attention_sample_locations(
            prepared_attention_output,
            layer_attention_output,
            &host_attention_output,
            &prepared_projection_input,
            layer_attention_projection_input,
            &host_projection_input,
            q_rope,
            k_rope,
            v_projected,
            q_gate,
            sequence_len,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            softmax_scale,
            attention_width,
        ),
    );
    Ok(diagnostic)
}

fn package_attention_projection_input_from_gate(
    q_gate: Option<&[f32]>,
    attention_output: &[f32],
    expected_attention_elements: usize,
    label: &str,
) -> Result<Vec<f32>, String> {
    if attention_output.len() != expected_attention_elements {
        return Err(format!(
            "self-attn diagnostic {label} attention length mismatch: got {} expected {expected_attention_elements}",
            attention_output.len()
        ));
    }
    match q_gate {
        Some(gate) => {
            if gate.len() != expected_attention_elements {
                return Err(format!(
                    "self-attn diagnostic {label} gate length mismatch: got {} expected {expected_attention_elements}",
                    gate.len()
                ));
            }
            let gated = runtime_host_sigmoid_mul_f32(gate, attention_output);
            if gated.len() != expected_attention_elements {
                return Err(format!(
                    "self-attn diagnostic {label} gated output length mismatch: got {} expected {expected_attention_elements}",
                    gated.len()
                ));
            }
            Ok(gated)
        }
        None => Ok(attention_output.to_vec()),
    }
}

#[allow(clippy::too_many_arguments)]
fn package_self_attn_causal_attention_sample_locations(
    prepared_attention_output: &[f32],
    layer_attention_output: &[f32],
    host_attention_output: &[f32],
    prepared_projection_input: &[f32],
    layer_projection_input: &[f32],
    host_projection_input: &[f32],
    q_rope: &[f32],
    k_rope: &[f32],
    v_projected: &[f32],
    q_gate: Option<&[f32]>,
    sequence_len: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
    attention_width: usize,
) -> Vec<serde_json::Value> {
    let sample_targets = [(8_usize, 503_usize)];
    sample_targets
        .into_iter()
        .filter_map(|(token_index, feature_index)| {
            if feature_index >= attention_width {
                return None;
            }
            let flat_index = token_index.checked_mul(attention_width)?.checked_add(feature_index)?;
            let prepared_attention = *prepared_attention_output.get(flat_index)?;
            let layer_attention = *layer_attention_output.get(flat_index)?;
            let host_attention = *host_attention_output.get(flat_index)?;
            let prepared_projection = *prepared_projection_input.get(flat_index)?;
            let layer_projection = *layer_projection_input.get(flat_index)?;
            let host_projection = *host_projection_input.get(flat_index)?;
            let q_gate_value = q_gate.and_then(|gate| gate.get(flat_index)).copied();
            let q_gate_sigmoid =
                q_gate_value.map(|gate| 1.0_f32 / (1.0_f32 + (-gate).exp()));
            let attention_breakdown = package_self_attn_causal_attention_breakdown(
                q_rope,
                k_rope,
                v_projected,
                token_index,
                feature_index,
                sequence_len,
                q_heads,
                kv_heads,
                head_dim,
                value_dim,
                softmax_scale,
            );
            Some(serde_json::json!({
                "token_index": token_index,
                "feature_index": feature_index,
                "flat_index": flat_index,
                "prepared_attention_output": prepared_attention,
                "layer_attention_output": layer_attention,
                "host_attention_output": host_attention,
                "layer_attention_minus_host_attention": layer_attention - host_attention,
                "prepared_attention_minus_host_attention": prepared_attention - host_attention,
                "layer_attention_minus_prepared_attention": layer_attention - prepared_attention,
                "q_gate": q_gate_value,
                "q_gate_sigmoid": q_gate_sigmoid,
                "prepared_projection_input": prepared_projection,
                "layer_projection_input": layer_projection,
                "host_projection_input": host_projection,
                "layer_projection_minus_host_projection": layer_projection - host_projection,
                "layer_projection_minus_prepared_projection": layer_projection - prepared_projection,
                "attention_breakdown": attention_breakdown,
            }))
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn package_self_attn_causal_attention_breakdown(
    q_rope: &[f32],
    k_rope: &[f32],
    v_projected: &[f32],
    token_index: usize,
    feature_index: usize,
    sequence_len: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
) -> Option<serde_json::Value> {
    if sequence_len == 0
        || q_heads == 0
        || kv_heads == 0
        || head_dim == 0
        || value_dim == 0
        || token_index >= sequence_len
        || !q_heads.is_multiple_of(kv_heads)
    {
        return None;
    }
    let q_head = feature_index / value_dim;
    let value_offset = feature_index % value_dim;
    if q_head >= q_heads {
        return None;
    }
    let q_per_kv = q_heads / kv_heads;
    let kv_head = q_head / q_per_kv;
    if kv_head >= kv_heads {
        return None;
    }
    let expected_q_elements = sequence_len.checked_mul(q_heads)?.checked_mul(head_dim)?;
    let expected_k_elements = sequence_len.checked_mul(kv_heads)?.checked_mul(head_dim)?;
    let expected_v_elements = sequence_len.checked_mul(kv_heads)?.checked_mul(value_dim)?;
    if q_rope.len() != expected_q_elements
        || k_rope.len() != expected_k_elements
        || v_projected.len() != expected_v_elements
    {
        return None;
    }

    let q_base = token_index
        .checked_mul(q_heads)?
        .checked_add(q_head)?
        .checked_mul(head_dim)?;
    let mut scores = Vec::with_capacity(token_index + 1);
    let mut dots = Vec::with_capacity(token_index + 1);
    for source_token in 0..=token_index {
        let k_base = source_token
            .checked_mul(kv_heads)?
            .checked_add(kv_head)?
            .checked_mul(head_dim)?;
        let dot = (0..head_dim)
            .map(|dim| q_rope[q_base + dim] * k_rope[k_base + dim])
            .sum::<f32>();
        dots.push(dot);
        scores.push(dot * softmax_scale);
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
    if denominator == 0.0 || !denominator.is_finite() {
        return None;
    }

    let mut computed_attention_output = 0.0_f32;
    let mut source_tokens = Vec::with_capacity(token_index + 1);
    for source_token in 0..=token_index {
        let weight = weights[source_token] / denominator;
        let v_index = source_token
            .checked_mul(kv_heads)?
            .checked_add(kv_head)?
            .checked_mul(value_dim)?
            .checked_add(value_offset)?;
        let v_value = v_projected[v_index];
        let contribution = weight * v_value;
        computed_attention_output += contribution;
        source_tokens.push(serde_json::json!({
            "source_token_index": source_token,
            "dot": dots[source_token],
            "score": scores[source_token],
            "softmax_weight": weight,
            "v_value": v_value,
            "weighted_v_contribution": contribution,
        }));
    }

    Some(serde_json::json!({
        "q_head": q_head,
        "kv_head": kv_head,
        "q_per_kv": q_per_kv,
        "value_offset": value_offset,
        "softmax_max_score": max_score,
        "softmax_denominator": denominator,
        "computed_attention_output": computed_attention_output,
        "source_tokens": source_tokens,
    }))
}

#[allow(clippy::too_many_arguments)]
fn package_module_contribution_summary(
    actual_before: &[f32],
    expected_before: &[f32],
    expected_after: &[f32],
    attention_projection_input: Option<&[f32]>,
    attention_output: &[f32],
    attention_block_output: &[f32],
    post_normed: &[f32],
    mlp_activation: Option<(&[f32], usize)>,
    extra_hot_input_vectors: &[(&str, &[f32], usize)],
    mlp_output: &[f32],
    actual_after: &[f32],
    sequence_len: usize,
    hidden: usize,
    sampled_token_indices: &[usize],
) -> Result<serde_json::Map<String, serde_json::Value>, String> {
    let expected_elements = sequence_len
        .checked_mul(hidden)
        .ok_or_else(|| "module contribution hidden element count overflows".to_string())?;
    for (label, values) in [
        ("actual_before", actual_before),
        ("expected_before", expected_before),
        ("expected_after", expected_after),
        ("attention_output", attention_output),
        ("attention_block_output", attention_block_output),
        ("post_normed", post_normed),
        ("mlp_output", mlp_output),
        ("actual_after", actual_after),
    ] {
        if values.len() != expected_elements {
            return Err(format!(
                "module contribution {label} length mismatch: got {} expected {expected_elements}",
                values.len()
            ));
        }
    }
    if let Some(values) = attention_projection_input {
        if values.len() != expected_elements {
            return Err(format!(
                "module contribution attention_projection_input length mismatch: got {} expected {expected_elements}",
                values.len()
            ));
        }
    }
    if let Some((values, feature_dim)) = mlp_activation {
        if feature_dim == 0 {
            return Err(
                "module contribution MLP activation feature dimension must be positive".to_string(),
            );
        }
        let expected_mlp_elements = sequence_len.checked_mul(feature_dim).ok_or_else(|| {
            "module contribution MLP activation element count overflows".to_string()
        })?;
        if values.len() != expected_mlp_elements {
            return Err(format!(
                "module contribution MLP activation length mismatch: got {} expected {expected_mlp_elements}",
                values.len()
            ));
        }
    }
    for (name, values, feature_dim) in extra_hot_input_vectors {
        if *feature_dim == 0 {
            return Err(format!(
                "module contribution {name} feature dimension must be positive"
            ));
        }
        let expected_extra_elements = sequence_len
            .checked_mul(*feature_dim)
            .ok_or_else(|| format!("module contribution {name} element count overflows"))?;
        if values.len() != expected_extra_elements {
            return Err(format!(
                "module contribution {name} length mismatch: got {} expected {expected_extra_elements}",
                values.len()
            ));
        }
    }
    for token_index in sampled_token_indices {
        if *token_index >= sequence_len {
            return Err(format!(
                "module contribution sampled token index {} is outside sequence_len={sequence_len}",
                token_index
            ));
        }
    }

    let actual_delta = actual_after
        .iter()
        .zip(actual_before.iter())
        .map(|(after, before)| after - before)
        .collect::<Vec<_>>();
    let expected_delta = expected_after
        .iter()
        .zip(expected_before.iter())
        .map(|(after, before)| after - before)
        .collect::<Vec<_>>();
    let delta_diff = actual_delta
        .iter()
        .zip(expected_delta.iter())
        .map(|(actual, expected)| actual - expected)
        .collect::<Vec<_>>();
    let residual_identity_error = actual_delta
        .iter()
        .zip(attention_output.iter())
        .zip(mlp_output.iter())
        .map(|((delta, attention), mlp)| delta - attention - mlp)
        .collect::<Vec<_>>();

    let max_output_diff_index = actual_after
        .iter()
        .zip(expected_after.iter())
        .enumerate()
        .max_by(
            |(_, (actual_left, expected_left)), (_, (actual_right, expected_right))| {
                (*actual_left - *expected_left)
                    .abs()
                    .partial_cmp(&(*actual_right - *expected_right).abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            },
        )
        .map(|(index, _)| index)
        .unwrap_or(0);
    let hot_hidden_index = if hidden == 0 {
        0
    } else {
        max_output_diff_index % hidden
    };
    let hot_token_index = if hidden == 0 {
        0
    } else {
        max_output_diff_index / hidden
    };

    let point = |flat_index: usize| {
        let output_diff = actual_after[flat_index] - expected_after[flat_index];
        let delta_diff = delta_diff[flat_index];
        serde_json::json!({
            "flat_index": flat_index,
            "token_index": flat_index / hidden,
            "hidden_index": flat_index % hidden,
            "actual_input": actual_before[flat_index],
            "expected_input": expected_before[flat_index],
            "input_diff": actual_before[flat_index] - expected_before[flat_index],
            "attention_output": attention_output[flat_index],
            "attention_block_output": attention_block_output[flat_index],
            "post_normed": post_normed[flat_index],
            "mlp_output": mlp_output[flat_index],
            "actual_delta": actual_delta[flat_index],
            "expected_delta": expected_delta[flat_index],
            "delta_diff": delta_diff,
            "residual_identity_error": residual_identity_error[flat_index],
            "actual_output": actual_after[flat_index],
            "expected_output": expected_after[flat_index],
            "output_diff": output_diff,
            "abs_output_diff": output_diff.abs(),
        })
    };
    let per_token_hot_hidden = (0..sequence_len)
        .map(|token_index| point(token_index * hidden + hot_hidden_index))
        .collect::<Vec<_>>();

    let mut summary = serde_json::Map::new();
    insert_json_detail(
        &mut summary,
        "delta_distribution",
        package_hidden_distribution(&actual_delta, &expected_delta, sequence_len, hidden)?,
    );
    insert_json_detail(
        &mut summary,
        "actual_delta_stats",
        package_slice_distribution_stats(&actual_delta),
    );
    insert_json_detail(
        &mut summary,
        "expected_delta_stats",
        package_slice_distribution_stats(&expected_delta),
    );
    insert_json_detail(
        &mut summary,
        "attention_output_stats",
        package_slice_distribution_stats(attention_output),
    );
    insert_json_detail(
        &mut summary,
        "mlp_output_stats",
        package_slice_distribution_stats(mlp_output),
    );
    insert_json_detail(
        &mut summary,
        "residual_identity_error_stats",
        package_slice_distribution_stats(&residual_identity_error),
    );
    insert_json_detail(&mut summary, "hot_hidden_index", hot_hidden_index);
    insert_json_detail(
        &mut summary,
        "hot_input_vectors",
        package_hot_input_vectors(
            hot_token_index,
            hidden,
            attention_projection_input,
            mlp_activation,
            extra_hot_input_vectors,
        )?,
    );
    if !sampled_token_indices.is_empty() {
        let mut sampled = Vec::new();
        let mut deduped = sampled_token_indices.to_vec();
        deduped.sort_unstable();
        deduped.dedup();
        for token_index in deduped {
            let mut item = package_hot_input_vectors(
                token_index,
                hidden,
                attention_projection_input,
                mlp_activation,
                extra_hot_input_vectors,
            )?;
            insert_json_detail(&mut item, "token_index", token_index);
            sampled.push(serde_json::Value::Object(item));
        }
        insert_json_detail(&mut summary, "sampled_hot_input_vectors", sampled);
    }
    insert_json_detail(
        &mut summary,
        "max_output_diff_trace",
        point(max_output_diff_index),
    );
    insert_json_detail(
        &mut summary,
        "per_token_hot_hidden_trace",
        per_token_hot_hidden,
    );
    Ok(summary)
}

fn package_hot_input_vectors(
    token_index: usize,
    hidden: usize,
    attention_projection_input: Option<&[f32]>,
    mlp_activation: Option<(&[f32], usize)>,
    extra_hot_input_vectors: &[(&str, &[f32], usize)],
) -> Result<serde_json::Map<String, serde_json::Value>, String> {
    let mut vectors = serde_json::Map::new();
    let mut attention_hot_feature_indices = Vec::new();
    let hidden_group_width = if hidden % 128 == 0 { Some(128) } else { None };
    if let Some(values) = attention_projection_input {
        let start = token_index
            .checked_mul(hidden)
            .ok_or_else(|| "attention projection input token offset overflows".to_string())?;
        let end = start
            .checked_add(hidden)
            .ok_or_else(|| "attention projection input token end overflows".to_string())?;
        let slice = values.get(start..end).ok_or_else(|| {
            format!(
                "attention projection input token slice {start}..{end} is outside len {}",
                values.len()
            )
        })?;
        attention_hot_feature_indices = package_top_abs_feature_indices(slice, 8);
        insert_json_detail(
            &mut vectors,
            "attention_projection_input",
            package_vector_summary(
                token_index,
                slice,
                &attention_hot_feature_indices,
                hidden_group_width,
            ),
        );
    }
    if let Some((values, feature_dim)) = mlp_activation {
        let start = token_index
            .checked_mul(feature_dim)
            .ok_or_else(|| "MLP activation token offset overflows".to_string())?;
        let end = start
            .checked_add(feature_dim)
            .ok_or_else(|| "MLP activation token end overflows".to_string())?;
        let slice = values.get(start..end).ok_or_else(|| {
            format!(
                "MLP activation token slice {start}..{end} is outside len {}",
                values.len()
            )
        })?;
        let sampled_feature_indices = package_top_abs_feature_indices(slice, 8);
        insert_json_detail(
            &mut vectors,
            "mlp_activation",
            package_vector_summary(token_index, slice, &sampled_feature_indices, None),
        );
    }
    for (name, values, feature_dim) in extra_hot_input_vectors {
        let start = token_index
            .checked_mul(*feature_dim)
            .ok_or_else(|| format!("{name} token offset overflows"))?;
        let end = start
            .checked_add(*feature_dim)
            .ok_or_else(|| format!("{name} token end overflows"))?;
        let slice = values.get(start..end).ok_or_else(|| {
            format!(
                "{name} token slice {start}..{end} is outside len {}",
                values.len()
            )
        })?;
        let sampled_feature_indices_storage = package_mapped_hot_feature_indices(
            slice,
            *feature_dim,
            hidden,
            &attention_hot_feature_indices,
        );
        let sampled_feature_indices = sampled_feature_indices_storage.as_slice();
        insert_json_detail(
            &mut vectors,
            *name,
            package_vector_summary(
                token_index,
                slice,
                sampled_feature_indices,
                if *feature_dim % 128 == 0 {
                    Some(128)
                } else {
                    None
                },
            ),
        );
    }
    Ok(vectors)
}

fn package_mapped_hot_feature_indices(
    values: &[f32],
    feature_dim: usize,
    hidden: usize,
    attention_hot_feature_indices: &[usize],
) -> Vec<usize> {
    if feature_dim == hidden {
        return attention_hot_feature_indices.to_vec();
    }
    let head_width = 128_usize;
    if hidden % head_width == 0 {
        let value_heads = hidden / head_width;
        if feature_dim == value_heads {
            let mut indices = attention_hot_feature_indices
                .iter()
                .map(|feature_index| feature_index / head_width)
                .filter(|head_index| *head_index < feature_dim)
                .collect::<Vec<_>>();
            indices.sort_unstable();
            indices.dedup();
            if !indices.is_empty() {
                return indices;
            }
        } else if feature_dim % head_width == 0 {
            let feature_heads = feature_dim / head_width;
            if feature_heads > 0 && feature_heads <= value_heads && value_heads % feature_heads == 0
            {
                let value_heads_per_feature_head = value_heads / feature_heads;
                let mut indices = attention_hot_feature_indices
                    .iter()
                    .map(|feature_index| {
                        let value_head = feature_index / head_width;
                        let head_offset = feature_index % head_width;
                        let feature_head = value_head / value_heads_per_feature_head;
                        feature_head * head_width + head_offset
                    })
                    .filter(|feature_index| *feature_index < feature_dim)
                    .collect::<Vec<_>>();
                indices.sort_unstable();
                indices.dedup();
                if !indices.is_empty() {
                    return indices;
                }
            }
        }
        if feature_dim > hidden {
            let v_base = feature_dim - hidden;
            let mut indices = attention_hot_feature_indices
                .iter()
                .map(|feature_index| v_base + feature_index)
                .filter(|feature_index| *feature_index < feature_dim)
                .collect::<Vec<_>>();
            indices.sort_unstable();
            indices.dedup();
            if !indices.is_empty() {
                return indices;
            }
        }
    }
    package_top_abs_feature_indices(values, 8)
}

fn package_vector_summary(
    token_index: usize,
    values: &[f32],
    sampled_feature_indices: &[usize],
    sampled_group_width: Option<usize>,
) -> serde_json::Map<String, serde_json::Value> {
    let mut summary = serde_json::Map::new();
    insert_json_detail(&mut summary, "token_index", token_index);
    insert_json_detail(&mut summary, "feature_count", values.len());
    insert_json_detail(
        &mut summary,
        "stats",
        package_slice_distribution_stats(values),
    );
    insert_json_detail(
        &mut summary,
        "top_abs_features",
        package_top_abs_value_locations(values, 8),
    );
    if !sampled_feature_indices.is_empty() {
        insert_json_detail(
            &mut summary,
            "sampled_features",
            package_sampled_value_locations(values, sampled_feature_indices, sampled_group_width),
        );
    }
    summary
}

fn package_top_abs_feature_indices(values: &[f32], limit: usize) -> Vec<usize> {
    let mut indexed = values.iter().enumerate().collect::<Vec<_>>();
    indexed.sort_by(|(_, left), (_, right)| {
        right
            .abs()
            .partial_cmp(&left.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    indexed
        .into_iter()
        .take(limit)
        .map(|(feature_index, _)| feature_index)
        .collect()
}

fn package_top_abs_value_locations(values: &[f32], limit: usize) -> Vec<serde_json::Value> {
    package_top_abs_feature_indices(values, limit)
        .into_iter()
        .filter_map(|feature_index| {
            values
                .get(feature_index)
                .map(|value| (feature_index, value))
        })
        .map(|(feature_index, value)| {
            serde_json::json!({
                "feature_index": feature_index,
                "value": *value,
                "abs_value": value.abs(),
            })
        })
        .collect()
}

fn package_sampled_value_locations(
    values: &[f32],
    sampled_feature_indices: &[usize],
    sampled_group_width: Option<usize>,
) -> Vec<serde_json::Value> {
    let mut indices = sampled_feature_indices
        .iter()
        .copied()
        .filter(|index| *index < values.len())
        .collect::<Vec<_>>();
    indices.sort_unstable();
    indices.dedup();
    indices
        .into_iter()
        .map(|feature_index| {
            let value = values[feature_index];
            let mut location = serde_json::json!({
                "feature_index": feature_index,
                "value": value,
                "abs_value": value.abs(),
            });
            if let Some(group_width) = sampled_group_width {
                if group_width > 0 {
                    let group_index = feature_index / group_width;
                    let group_start = group_index * group_width;
                    let group_end = (group_start + group_width).min(values.len());
                    if group_start < group_end {
                        if let Some(object) = location.as_object_mut() {
                            insert_json_detail(object, "group_index", group_index);
                            insert_json_detail(object, "group_offset", feature_index - group_start);
                            insert_json_detail(object, "group_width", group_end - group_start);
                            insert_json_detail(
                                object,
                                "group_stats",
                                package_slice_distribution_stats(&values[group_start..group_end]),
                            );
                        }
                    }
                }
            }
            location
        })
        .collect()
}

fn package_hidden_distribution(
    actual: &[f32],
    expected: &[f32],
    sequence_len: usize,
    hidden: usize,
) -> Result<serde_json::Map<String, serde_json::Value>, String> {
    let expected_elements = sequence_len
        .checked_mul(hidden)
        .ok_or_else(|| "hidden distribution element count overflows".to_string())?;
    if actual.len() != expected_elements || expected.len() != expected_elements {
        return Err(format!(
            "hidden distribution length mismatch: actual={} expected={} expected_elements={expected_elements}",
            actual.len(),
            expected.len()
        ));
    }

    let diff = actual
        .iter()
        .zip(expected.iter())
        .map(|(actual, expected)| actual - expected)
        .collect::<Vec<_>>();
    let max_abs_diff = diff
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| {
            left.abs()
                .partial_cmp(&right.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(index, value)| {
            serde_json::json!({
                "flat_index": index,
                "token_index": index / hidden,
                "hidden_index": index % hidden,
                "actual": actual[index],
                "expected": expected[index],
                "diff": *value,
                "abs_diff": value.abs(),
            })
        })
        .unwrap_or_else(|| serde_json::json!(null));

    let per_token = (0..sequence_len)
        .map(|token_index| {
            let start = token_index * hidden;
            let end = start + hidden;
            let token_actual = &actual[start..end];
            let token_expected = &expected[start..end];
            let token_diff = &diff[start..end];
            let metrics = compare_f32_slices(token_actual, token_expected)?;
            Ok(serde_json::json!({
                "token_index": token_index,
                "mse": metrics.mse,
                "mean_abs_diff": metrics.mean_abs_diff,
                "max_abs_diff": metrics.max_abs_diff,
                "cosine_similarity": metrics.cosine_similarity,
                "actual_rms": package_slice_rms(token_actual),
                "expected_rms": package_slice_rms(token_expected),
                "diff_rms": package_slice_rms(token_diff),
                "diff_max_abs_location": package_slice_max_abs_location(
                    token_diff,
                    Some(token_actual),
                    Some(token_expected),
                    token_index,
                ),
            }))
        })
        .collect::<Result<Vec<_>, String>>()?;

    let mut distribution = serde_json::Map::new();
    insert_json_detail(
        &mut distribution,
        "actual_stats",
        package_slice_distribution_stats(actual),
    );
    insert_json_detail(
        &mut distribution,
        "expected_stats",
        package_slice_distribution_stats(expected),
    );
    insert_json_detail(
        &mut distribution,
        "diff_stats",
        package_slice_distribution_stats(&diff),
    );
    insert_json_detail(&mut distribution, "max_abs_diff_location", max_abs_diff);
    insert_json_detail(
        &mut distribution,
        "top_abs_diff_locations",
        package_top_abs_diff_locations(&diff, actual, expected, hidden, 8),
    );
    insert_json_detail(&mut distribution, "per_token", per_token);
    Ok(distribution)
}

fn package_slice_rms(values: &[f32]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let square_sum = values
        .iter()
        .map(|value| {
            let value = f64::from(*value);
            value * value
        })
        .sum::<f64>();
    (square_sum / values.len() as f64).sqrt()
}

fn package_slice_distribution_stats(values: &[f32]) -> serde_json::Map<String, serde_json::Value> {
    let mut finite_count = 0_usize;
    let mut nonfinite_count = 0_usize;
    let mut sum = 0.0_f64;
    let mut square_sum = 0.0_f64;
    let mut abs_sum = 0.0_f64;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut max_abs = 0.0_f64;
    let mut max_abs_index = 0_usize;

    for (index, value) in values.iter().enumerate() {
        let value = f64::from(*value);
        if !value.is_finite() {
            nonfinite_count += 1;
            continue;
        }
        finite_count += 1;
        sum += value;
        square_sum += value * value;
        abs_sum += value.abs();
        min = min.min(value);
        max = max.max(value);
        if value.abs() > max_abs {
            max_abs = value.abs();
            max_abs_index = index;
        }
    }

    let mean = if finite_count == 0 {
        0.0
    } else {
        sum / finite_count as f64
    };
    let mean_square = if finite_count == 0 {
        0.0
    } else {
        square_sum / finite_count as f64
    };
    let variance = (mean_square - mean * mean).max(0.0);
    let mut stats = serde_json::Map::new();
    insert_json_detail(&mut stats, "count", values.len());
    insert_json_detail(&mut stats, "finite_count", finite_count);
    insert_json_detail(&mut stats, "nonfinite_count", nonfinite_count);
    insert_json_detail(&mut stats, "mean", mean);
    insert_json_detail(
        &mut stats,
        "abs_mean",
        if finite_count == 0 {
            0.0
        } else {
            abs_sum / finite_count as f64
        },
    );
    insert_json_detail(&mut stats, "variance", variance);
    insert_json_detail(&mut stats, "stddev", variance.sqrt());
    insert_json_detail(&mut stats, "rms", mean_square.sqrt());
    insert_json_detail(&mut stats, "l2_norm", square_sum.sqrt());
    insert_json_detail(&mut stats, "min", if finite_count == 0 { 0.0 } else { min });
    insert_json_detail(&mut stats, "max", if finite_count == 0 { 0.0 } else { max });
    insert_json_detail(&mut stats, "max_abs", max_abs);
    insert_json_detail(&mut stats, "max_abs_index", max_abs_index);
    stats
}

fn package_slice_max_abs_location(
    diff: &[f32],
    actual: Option<&[f32]>,
    expected: Option<&[f32]>,
    token_index: usize,
) -> serde_json::Value {
    diff.iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| {
            left.abs()
                .partial_cmp(&right.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(hidden_index, value)| {
            serde_json::json!({
                "token_index": token_index,
                "hidden_index": hidden_index,
                "actual": actual.and_then(|values| values.get(hidden_index)).copied(),
                "expected": expected.and_then(|values| values.get(hidden_index)).copied(),
                "diff": *value,
                "abs_diff": value.abs(),
            })
        })
        .unwrap_or_else(|| serde_json::json!(null))
}

fn package_top_abs_diff_locations(
    diff: &[f32],
    actual: &[f32],
    expected: &[f32],
    hidden: usize,
    limit: usize,
) -> Vec<serde_json::Value> {
    let mut indexed = diff.iter().enumerate().collect::<Vec<_>>();
    indexed.sort_by(|(_, left), (_, right)| {
        right
            .abs()
            .partial_cmp(&left.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    indexed
        .into_iter()
        .take(limit)
        .map(|(index, value)| {
            serde_json::json!({
                "flat_index": index,
                "token_index": index / hidden,
                "hidden_index": index % hidden,
                "actual": actual[index],
                "expected": expected[index],
                "diff": *value,
                "abs_diff": value.abs(),
            })
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn append_package_golden_prefix_report_entry(
    report_entries: &mut Vec<serde_json::Value>,
    path: &str,
    fixture_path: &str,
    fixture_kind: Option<&str>,
    device_index: u32,
    backend: &str,
    device_name: &str,
    layer_position: usize,
    layer_index: usize,
    layer_kind: &str,
    layer_start: usize,
    layer_end_exclusive: usize,
    sequence_len: usize,
    hidden: usize,
    run_mode: PackageGoldenPrefixRunMode,
    input_metrics: &ullm_engine::golden::GoldenComparisonMetrics,
    input_failure_class: &str,
    input_expected_preview: Vec<f32>,
    input_actual_preview: Vec<f32>,
    input_diff_preview: Vec<f32>,
    metrics: &ullm_engine::golden::GoldenComparisonMetrics,
    failure_class: &str,
    expected_preview: Vec<f32>,
    actual_preview: Vec<f32>,
    diff_preview: Vec<f32>,
    details: serde_json::Map<String, serde_json::Value>,
) {
    let mut entry = serde_json::Map::new();
    insert_json_detail(&mut entry, "command", "package-golden-prefix-smoke");
    insert_json_detail(&mut entry, "package", path);
    insert_json_detail(&mut entry, "fixture", fixture_path);
    insert_json_detail(&mut entry, "fixture_kind", fixture_kind);
    insert_json_detail(&mut entry, "device_index", device_index);
    insert_json_detail(&mut entry, "backend", backend);
    insert_json_detail(&mut entry, "device_name", device_name);
    insert_json_detail(&mut entry, "layer_position", layer_position);
    insert_json_detail(&mut entry, "layer_index", layer_index);
    insert_json_detail(&mut entry, "layer_kind", layer_kind);
    insert_json_detail(&mut entry, "layer_start", layer_start);
    insert_json_detail(&mut entry, "layer_end_exclusive", layer_end_exclusive);
    insert_json_detail(&mut entry, "sequence_len", sequence_len);
    insert_json_detail(&mut entry, "hidden_size", hidden);
    insert_json_detail(&mut entry, "run_mode", run_mode.as_str());
    insert_json_detail(&mut entry, "input_mse", input_metrics.mse);
    insert_json_detail(
        &mut entry,
        "input_mean_abs_diff",
        input_metrics.mean_abs_diff,
    );
    insert_json_detail(&mut entry, "input_max_abs_diff", input_metrics.max_abs_diff);
    insert_json_detail(
        &mut entry,
        "input_cosine_similarity",
        input_metrics.cosine_similarity,
    );
    insert_json_detail(&mut entry, "input_failure_class", input_failure_class);
    insert_json_detail(&mut entry, "input_expected_preview", input_expected_preview);
    insert_json_detail(&mut entry, "input_actual_preview", input_actual_preview);
    insert_json_detail(&mut entry, "input_diff_preview", input_diff_preview);
    insert_json_detail(&mut entry, "mse", metrics.mse);
    insert_json_detail(&mut entry, "mean_abs_diff", metrics.mean_abs_diff);
    insert_json_detail(&mut entry, "max_abs_diff", metrics.max_abs_diff);
    insert_json_detail(&mut entry, "cosine_similarity", metrics.cosine_similarity);
    insert_json_detail(&mut entry, "failure_class", failure_class);
    insert_json_detail(&mut entry, "expected_preview", expected_preview);
    insert_json_detail(&mut entry, "actual_preview", actual_preview);
    insert_json_detail(&mut entry, "diff_preview", diff_preview);
    insert_json_detail(&mut entry, "verified", true);
    entry.extend(details);
    report_entries.push(serde_json::Value::Object(entry));
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PackageGoldenPrefixRunMode {
    ActualPrefix,
    GoldenBeforeEachLayer,
}

impl PackageGoldenPrefixRunMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::ActualPrefix => "actual_prefix",
            Self::GoldenBeforeEachLayer => "golden_before_each_layer",
        }
    }
}

fn parse_package_golden_prefix_run_mode(
    run_mode: Option<&str>,
) -> Result<PackageGoldenPrefixRunMode, ExitCode> {
    match run_mode {
        Some(raw) => match raw {
            "actual_prefix" => Ok(PackageGoldenPrefixRunMode::ActualPrefix),
            "golden_before_each_layer" => Ok(PackageGoldenPrefixRunMode::GoldenBeforeEachLayer),
            _ => {
                eprintln!(
                    "invalid run_mode: {raw}; expected actual_prefix or golden_before_each_layer"
                );
                Err(ExitCode::from(2))
            }
        },
        None => Ok(PackageGoldenPrefixRunMode::ActualPrefix),
    }
}

fn package_golden_prefix_failure_class(
    metrics: &ullm_engine::golden::GoldenComparisonMetrics,
) -> &'static str {
    if !metrics.mse.is_finite()
        || !metrics.mean_abs_diff.is_finite()
        || !metrics.max_abs_diff.is_finite()
        || !metrics.cosine_similarity.is_finite()
    {
        "numeric_drift"
    } else if metrics.cosine_similarity < 0.5 || metrics.mse > 0.1 {
        "numeric_drift"
    } else if metrics.max_abs_diff > 0.0 {
        "possible_quantization_error"
    } else {
        "ok"
    }
}

fn write_jsonl_report(path: &str, entries: &[serde_json::Value]) -> Result<(), String> {
    let path = std::path::Path::new(path);
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|err| {
                format!(
                    "failed to create report directory {}: {err}",
                    parent.display()
                )
            })?;
        }
    }
    let mut file = File::create(path)
        .map_err(|err| format!("failed to create report {}: {err}", path.display()))?;
    for entry in entries {
        serde_json::to_writer(&mut file, entry)
            .map_err(|err| format!("failed to write report {}: {err}", path.display()))?;
        file.write_all(b"\n")
            .map_err(|err| format!("failed to write report {}: {err}", path.display()))?;
    }
    Ok(())
}

fn write_package_prefix_input_dump(
    dump_dir: &str,
    layer_index: usize,
    run_mode: PackageGoldenPrefixRunMode,
    sequence_len: usize,
    hidden: usize,
    values: &[f32],
) -> Result<String, String> {
    let expected_elements = sequence_len
        .checked_mul(hidden)
        .ok_or_else(|| "prefix input dump element count overflows".to_string())?;
    if values.len() != expected_elements {
        return Err(format!(
            "prefix input dump layer {layer_index} length mismatch: got {} expected {expected_elements}",
            values.len()
        ));
    }

    let dir = std::path::Path::new(dump_dir);
    fs::create_dir_all(dir).map_err(|err| {
        format!(
            "failed to create input dump directory {}: {err}",
            dir.display()
        )
    })?;
    let file_name = format!("layer-{layer_index:04}-input.f32");
    let path = dir.join(&file_name);
    let mut file = File::create(&path)
        .map_err(|err| format!("failed to create input dump {}: {err}", path.display()))?;
    for chunk in values.chunks(4096) {
        let mut bytes = Vec::with_capacity(chunk.len() * std::mem::size_of::<f32>());
        for value in chunk {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        file.write_all(&bytes)
            .map_err(|err| format!("failed to write input dump {}: {err}", path.display()))?;
    }

    let metadata_name = format!("layer-{layer_index:04}-input.json");
    let metadata_path = dir.join(metadata_name);
    let metadata = serde_json::json!({
        "schema_version": "package-golden-prefix-input-dump-v0.1",
        "layer_index": layer_index,
        "run_mode": run_mode.as_str(),
        "dtype": "float32",
        "shape": [1, sequence_len, hidden],
        "file": file_name,
    });
    let mut metadata_file = File::create(&metadata_path).map_err(|err| {
        format!(
            "failed to create input dump metadata {}: {err}",
            metadata_path.display()
        )
    })?;
    serde_json::to_writer_pretty(&mut metadata_file, &metadata).map_err(|err| {
        format!(
            "failed to write input dump metadata {}: {err}",
            metadata_path.display()
        )
    })?;
    metadata_file.write_all(b"\n").map_err(|err| {
        format!(
            "failed to finish input dump metadata {}: {err}",
            metadata_path.display()
        )
    })?;

    Ok(path.to_string_lossy().into_owned())
}

#[allow(clippy::too_many_arguments)]
fn package_layer_golden_smoke_impl(
    path: &str,
    fixture_path: &str,
    fixture: GoldenTensorFixture,
    device_index: u32,
    chunk_bytes: usize,
    layer_index: usize,
    rotary_dim: Option<String>,
    rope_base: f32,
    position_offset: usize,
) -> Result<String, String> {
    let golden_layer = fixture.select_layer(layer_index)?;
    let sequence_len = fixture.metadata().sequence_len;
    let hidden = fixture.metadata().hidden_size;
    if sequence_len == 0 || hidden == 0 {
        return Err(format!(
            "golden fixture has invalid sequence_len={sequence_len} hidden_size={hidden}"
        ));
    }
    validate_golden_hidden_shape(
        &golden_layer.before_shape,
        sequence_len,
        hidden,
        "golden before hidden",
    )?;
    validate_golden_hidden_shape(
        &golden_layer.after_shape,
        sequence_len,
        hidden,
        "golden after hidden",
    )?;
    validate_golden_position_ids(
        &fixture.metadata().position_ids,
        sequence_len,
        position_offset,
    )?;

    let before = fixture.read_layer_before_f32(layer_index)?;
    let after = fixture.read_layer_after_f32(layer_index)?;
    let expected_hidden_elements = sequence_len
        .checked_mul(hidden)
        .ok_or_else(|| "golden hidden element count overflows".to_string())?;
    if before.len() != expected_hidden_elements || after.len() != expected_hidden_elements {
        return Err(format!(
            "golden fixture payload element mismatch: before={} after={} expected={expected_hidden_elements}",
            before.len(),
            after.len()
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
    let layer = qwen3_package_decoder_layer_runtime_from_package(
        &mut context,
        &mut stream,
        path,
        chunk_bytes,
        layer_index,
    )?;
    if layer.runtime_shape.hidden != hidden {
        return Err(format!(
            "golden hidden_size {hidden} does not match package layer hidden {}",
            layer.runtime_shape.hidden
        ));
    }
    let rotary_dim =
        parse_package_layer_golden_rotary_dim(layer.runtime_shape.head_dim, rotary_dim)?;
    let block_size = sequence_len;
    let cache_blocks = 1_usize;
    let block_table = vec![0_u32];
    let prepared = qwen3_self_attn_prepare_sequence_for_paged_decode_f32(
        &mut context,
        &mut stream,
        &layer.weights.self_attn,
        before,
        sequence_len,
        &layer.q_norm.values,
        &layer.k_norm.values,
        rotary_dim,
        position_offset,
        rope_base,
        &block_table,
        block_size,
        cache_blocks,
    )?;
    let Qwen3SelfAttnRuntimePreparedSequenceForPagedDecode {
        residual_sequence,
        prepared:
            Qwen3SelfAttnRuntimePreparedSequence {
                q_query: _,
                k_projected: _,
                q_normed: _,
                k_normed: _,
                q_rope,
                k_rope,
                v_projected,
                q_gate,
                attention_output: _,
                shape,
                softmax_scale,
                q_projection_layout,
                q_gate_elements,
                output_gate_layout,
            },
        paged_k_cache: _,
        paged_v_cache: _,
        paged_block_table,
        paged_block_size,
        paged_cache_blocks,
    } = prepared;
    let decode_shape = PagedDecodeShape {
        block_size: paged_block_size,
        cache_blocks: paged_cache_blocks,
        q_heads: shape.q_heads,
        kv_heads: shape.kv_heads,
        head_dim: shape.head_dim,
        value_dim: shape.value_dim,
    };
    let mlp_epsilon = 1e-5_f32;
    let layer_output = qwen3_decoder_layer_sequence_to_host_f32(
        &layer.weights,
        &mut context,
        &mut stream,
        decode_shape,
        &paged_block_table,
        softmax_scale,
        mlp_epsilon,
        &q_rope,
        &k_rope,
        &v_projected,
        q_gate.as_deref(),
        &residual_sequence,
        sequence_len,
    )?;
    let metrics = compare_f32_slices(&layer_output.layer_output, &after)?;
    let preview_len = 8.min(after.len()).min(layer_output.layer_output.len());
    let diff_preview = layer_output
        .layer_output
        .iter()
        .zip(after.iter())
        .take(preview_len)
        .map(|(actual, expected)| actual - expected)
        .collect::<Vec<_>>();
    let candidate_ids = package_layer_candidate_ids(path, &layer);

    Ok(format!(
        "package-layer-golden-smoke package={} fixture={} layer={} q_tensor=\"{}\" k_tensor=\"{}\" v_tensor=\"{}\" o_tensor=\"{}\" gate_tensor=\"{}\" up_tensor=\"{}\" down_tensor=\"{}\" candidate_ids={:?} sequence_len={} hidden={} before_shape={:?} after_shape={:?} block_size={} cache_blocks={} block_table={:?} q_heads={} kv_heads={} head_dim={} value_dim={} rotary_dim={} position_offset={} rope_base={} softmax_scale={:.9} mlp_epsilon={:.9} q_projection_layout={} q_gate_elements={} output_gate_layout={} q_norm_dtype={} k_norm_dtype={} post_norm_dtype={} backend={} device_index={} name=\"{}\" mse={:.12} mean_abs_diff={:.9} max_abs_diff={:.9} cosine_similarity={:.9} expected_preview={} actual_preview={} diff_preview={} verified=true",
        path,
        fixture_path,
        layer_index,
        layer.q_tensor,
        layer.k_tensor,
        layer.v_tensor,
        layer.o_tensor,
        layer.gate_tensor,
        layer.up_tensor,
        layer.down_tensor,
        candidate_ids,
        sequence_len,
        hidden,
        &golden_layer.before_shape,
        &golden_layer.after_shape,
        paged_block_size,
        paged_cache_blocks,
        paged_block_table,
        shape.q_heads,
        shape.kv_heads,
        shape.head_dim,
        shape.value_dim,
        rotary_dim,
        position_offset,
        rope_base,
        softmax_scale,
        mlp_epsilon,
        q_projection_layout,
        q_gate_elements,
        output_gate_layout,
        layer.q_norm.dtype,
        layer.k_norm.dtype,
        layer.post_norm.dtype,
        info.backend,
        device_index,
        info.name,
        metrics.mse,
        metrics.mean_abs_diff,
        metrics.max_abs_diff,
        metrics.cosine_similarity,
        format_f32_preview(&after[..preview_len]),
        format_f32_preview(&layer_output.layer_output[..preview_len]),
        format_f32_preview(&diff_preview),
    ))
}

fn parse_package_layer_golden_rotary_dim(
    head_dim: usize,
    rotary_dim: Option<String>,
) -> Result<usize, String> {
    let default_rotary_dim = {
        let candidate = if head_dim >= 4 {
            head_dim / 4
        } else {
            head_dim
        };
        candidate - (candidate % 2)
    };
    if default_rotary_dim == 0 {
        return Err(format!(
            "default rotary_dim is zero for head_dim={head_dim}"
        ));
    }
    let rotary_dim = match rotary_dim {
        Some(raw) => raw
            .parse::<usize>()
            .map_err(|err| format!("invalid rotary dim {raw:?}: {err}"))?,
        None => default_rotary_dim,
    };
    if rotary_dim == 0 || rotary_dim > head_dim || !rotary_dim.is_multiple_of(2) {
        return Err(format!(
            "rotary dim must be even and no greater than head_dim: rotary_dim={rotary_dim}, head_dim={head_dim}"
        ));
    }
    Ok(rotary_dim)
}

fn validate_golden_hidden_shape(
    shape: &[usize],
    sequence_len: usize,
    hidden: usize,
    label: &str,
) -> Result<(), String> {
    let mut elements = 1_usize;
    for dim in shape {
        if *dim == 0 {
            return Err(format!("{label} shape contains zero: {shape:?}"));
        }
        elements = elements
            .checked_mul(*dim)
            .ok_or_else(|| format!("{label} shape element count overflows: {shape:?}"))?;
    }
    let expected_elements = sequence_len
        .checked_mul(hidden)
        .ok_or_else(|| format!("{label} expected element count overflows"))?;
    if elements != expected_elements {
        return Err(format!(
            "{label} shape {shape:?} has {elements} elements, expected {expected_elements}"
        ));
    }
    match shape {
        [seq, width] if *seq == sequence_len && *width == hidden => Ok(()),
        [batch, seq, width] if *batch == 1 && *seq == sequence_len && *width == hidden => Ok(()),
        _ => Err(format!(
            "{label} shape {shape:?} must be [sequence_len, hidden] or [1, sequence_len, hidden] with sequence_len={sequence_len} hidden={hidden}"
        )),
    }
}

fn validate_golden_position_ids(
    position_ids: &[u64],
    sequence_len: usize,
    position_offset: usize,
) -> Result<(), String> {
    if position_ids.len() != sequence_len {
        return Err(format!(
            "golden position_ids length {} does not match sequence_len={sequence_len}",
            position_ids.len()
        ));
    }
    for (index, position_id) in position_ids.iter().enumerate() {
        let expected = position_offset
            .checked_add(index)
            .ok_or_else(|| "golden position id expectation overflows".to_string())?;
        let expected = u64::try_from(expected)
            .map_err(|_| "golden expected position id does not fit u64".to_string())?;
        if *position_id != expected {
            return Err(format!(
                "golden position_ids are not contiguous from position_offset={position_offset}: index={index} expected={expected} got={position_id}"
            ));
        }
    }
    Ok(())
}

fn package_layer_candidate_ids(
    path: &str,
    layer: &ullm_engine::qwen3_loader::Qwen3PackageDecoderLayerRuntime,
) -> Vec<String> {
    [
        &layer.q_tensor,
        &layer.k_tensor,
        &layer.v_tensor,
        &layer.o_tensor,
        &layer.gate_tensor,
        &layer.up_tensor,
        &layer.down_tensor,
    ]
    .iter()
    .map(|tensor_name| {
        select_tensor_payload_bundle(path, &TensorSelector::Name((*tensor_name).clone()))
            .ok()
            .and_then(|bundle| bundle.candidate_id)
            .unwrap_or_else(|| "unknown".to_string())
    })
    .collect()
}

fn package_self_attn_mlp_block_model_loop_smoke(
    path: Option<String>,
    device_index: Option<String>,
    chunk_bytes: Option<String>,
    layer_indices: Option<String>,
    second_layer_or_sequence_len: Option<String>,
    sequence_len_or_rotary_dim: Option<String>,
    rotary_dim_or_rope_base: Option<String>,
    rope_base_or_position_offset: Option<String>,
    position_offset_or_extra: Option<String>,
) -> ExitCode {
    let Some(path) = path else {
        eprintln!("package-self-attn-mlp-block-model-loop-smoke requires a .ullm.d path");
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
    let cli_tail = match parse_package_model_loop_cli_tail(
        layer_indices,
        second_layer_or_sequence_len,
        sequence_len_or_rotary_dim,
        rotary_dim_or_rope_base,
        rope_base_or_position_offset,
        position_offset_or_extra,
    ) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let sequence_len = match parse_optional_usize(cli_tail.sequence_len, 3, "sequence length") {
        Ok(value) if value >= 3 => value,
        Ok(_) => {
            eprintln!("sequence length must be at least three for model-loop smoke");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let rope_base = match parse_optional_f32(cli_tail.rope_base, 10_000_000.0, "rope base") {
        Ok(value) if value > 1.0 => value,
        Ok(_) => {
            eprintln!("rope base must be greater than one");
            return ExitCode::from(2);
        }
        Err(code) => return code,
    };
    let position_offset = match parse_optional_usize(cli_tail.position_offset, 3, "position offset")
    {
        Ok(value) => value,
        Err(code) => return code,
    };

    match package_self_attn_mlp_block_model_loop_smoke_impl(
        &path,
        device_index,
        chunk_bytes,
        cli_tail.layer_indices,
        sequence_len,
        cli_tail.rotary_dim,
        rope_base,
        position_offset,
    ) {
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

struct PackageModelLoopCliTail {
    layer_indices: Vec<usize>,
    sequence_len: Option<String>,
    rotary_dim: Option<String>,
    rope_base: Option<String>,
    position_offset: Option<String>,
}

fn parse_package_model_loop_cli_tail(
    layer_indices: Option<String>,
    second_layer_or_sequence_len: Option<String>,
    sequence_len_or_rotary_dim: Option<String>,
    rotary_dim_or_rope_base: Option<String>,
    rope_base_or_position_offset: Option<String>,
    position_offset_or_extra: Option<String>,
) -> Result<PackageModelLoopCliTail, ExitCode> {
    let Some(first) = layer_indices else {
        return Ok(PackageModelLoopCliTail {
            layer_indices: vec![3, 7],
            sequence_len: None,
            rotary_dim: None,
            rope_base: None,
            position_offset: None,
        });
    };

    if first.contains(',') {
        if position_offset_or_extra.is_some() {
            eprintln!(
                "too many model-loop arguments for comma-separated layer list; expected LAYERS_CSV [SEQUENCE_LEN] [ROTARY_DIM] [ROPE_BASE] [POSITION_OFFSET]"
            );
            return Err(ExitCode::from(2));
        }
        return Ok(PackageModelLoopCliTail {
            layer_indices: parse_usize_csv(&first, "layer list")?,
            sequence_len: second_layer_or_sequence_len,
            rotary_dim: sequence_len_or_rotary_dim,
            rope_base: rotary_dim_or_rope_base,
            position_offset: rope_base_or_position_offset,
        });
    }

    let first_layer_index = parse_usize_value(&first, "first layer index")?;
    if let Some(raw) = second_layer_or_sequence_len
        .as_deref()
        .filter(|raw| raw.contains(','))
    {
        let mut layer_indices = Vec::new();
        layer_indices.push(first_layer_index);
        layer_indices.extend(parse_usize_csv(raw, "second layer list")?);
        return Ok(PackageModelLoopCliTail {
            layer_indices,
            sequence_len: sequence_len_or_rotary_dim,
            rotary_dim: rotary_dim_or_rope_base,
            rope_base: rope_base_or_position_offset,
            position_offset: position_offset_or_extra,
        });
    }

    let second_layer_index = match second_layer_or_sequence_len {
        Some(raw) => parse_usize_value(&raw, "second layer index")?,
        None => 7,
    };
    Ok(PackageModelLoopCliTail {
        layer_indices: vec![first_layer_index, second_layer_index],
        sequence_len: sequence_len_or_rotary_dim,
        rotary_dim: rotary_dim_or_rope_base,
        rope_base: rope_base_or_position_offset,
        position_offset: position_offset_or_extra,
    })
}

fn parse_usize_csv(value: &str, label: &str) -> Result<Vec<usize>, ExitCode> {
    let mut parsed = Vec::new();
    for raw in value.split(',') {
        let entry = raw.trim();
        if entry.is_empty() {
            eprintln!("invalid {label}: empty entry in {value:?}");
            return Err(ExitCode::from(2));
        }
        parsed.push(parse_usize_value(entry, label)?);
    }
    if parsed.is_empty() {
        eprintln!("invalid {label}: expected at least one entry");
        return Err(ExitCode::from(2));
    }
    Ok(parsed)
}

fn parse_usize_value(value: &str, label: &str) -> Result<usize, ExitCode> {
    value.parse::<usize>().map_err(|err| {
        eprintln!("invalid {label}: {err}");
        ExitCode::from(2)
    })
}

#[cfg(test)]
mod package_model_loop_cli_tail_tests {
    use super::*;

    fn parse_tail(
        args: [Option<&str>; 6],
    ) -> Result<PackageModelLoopCliTail, std::process::ExitCode> {
        parse_package_model_loop_cli_tail(
            args[0].map(str::to_string),
            args[1].map(str::to_string),
            args[2].map(str::to_string),
            args[3].map(str::to_string),
            args[4].map(str::to_string),
            args[5].map(str::to_string),
        )
    }

    #[test]
    fn package_model_loop_cli_tail_defaults_to_two_layers() {
        let tail = parse_tail([None, None, None, None, None, None]).unwrap();
        assert_eq!(tail.layer_indices, vec![3, 7]);
        assert_eq!(tail.sequence_len, None);
        assert_eq!(tail.rotary_dim, None);
        assert_eq!(tail.rope_base, None);
        assert_eq!(tail.position_offset, None);
    }

    #[test]
    fn package_model_loop_cli_tail_keeps_legacy_two_layer_layout() {
        let tail = parse_tail([
            Some("3"),
            Some("7"),
            Some("5"),
            Some("16"),
            Some("10000000"),
            Some("4"),
        ])
        .unwrap();
        assert_eq!(tail.layer_indices, vec![3, 7]);
        assert_eq!(tail.sequence_len.as_deref(), Some("5"));
        assert_eq!(tail.rotary_dim.as_deref(), Some("16"));
        assert_eq!(tail.rope_base.as_deref(), Some("10000000"));
        assert_eq!(tail.position_offset.as_deref(), Some("4"));
    }

    #[test]
    fn package_model_loop_cli_tail_accepts_first_argument_layer_csv() {
        let tail = parse_tail([
            Some("3,7,11"),
            Some("5"),
            Some("16"),
            Some("10000000"),
            Some("4"),
            None,
        ])
        .unwrap();
        assert_eq!(tail.layer_indices, vec![3, 7, 11]);
        assert_eq!(tail.sequence_len.as_deref(), Some("5"));
        assert_eq!(tail.rotary_dim.as_deref(), Some("16"));
        assert_eq!(tail.rope_base.as_deref(), Some("10000000"));
        assert_eq!(tail.position_offset.as_deref(), Some("4"));
    }

    #[test]
    fn package_model_loop_cli_tail_accepts_second_argument_layer_csv() {
        let tail = parse_tail([
            Some("3"),
            Some("7,11"),
            Some("5"),
            Some("16"),
            Some("10000000"),
            Some("4"),
        ])
        .unwrap();
        assert_eq!(tail.layer_indices, vec![3, 7, 11]);
        assert_eq!(tail.sequence_len.as_deref(), Some("5"));
        assert_eq!(tail.rotary_dim.as_deref(), Some("16"));
        assert_eq!(tail.rope_base.as_deref(), Some("10000000"));
        assert_eq!(tail.position_offset.as_deref(), Some("4"));
    }

    #[test]
    fn package_model_loop_cli_tail_rejects_empty_layer_csv_entry() {
        assert!(parse_tail([Some("3,,7"), None, None, None, None, None]).is_err());
    }
}

#[allow(clippy::too_many_arguments)]
fn package_self_attn_mlp_block_model_loop_smoke_impl(
    path: &str,
    device_index: u32,
    chunk_bytes: usize,
    layer_indices: Vec<usize>,
    sequence_len: usize,
    rotary_dim: Option<String>,
    rope_base: f32,
    position_offset: usize,
) -> Result<String, String> {
    let mut context = ullm_runtime_sys::RuntimeContext::create(device_index)
        .map_err(|err| format!("failed to create runtime context: {err}"))?;
    let info = context
        .device_info()
        .map_err(|err| format!("failed to query runtime context device: {err}"))?;
    let mut stream = context
        .create_stream()
        .map_err(|err| format!("failed to create runtime stream: {err}"))?;

    let mut smoke_run = PackageModelLoopSmokeRun::new(
        &mut context,
        &mut stream,
        path,
        chunk_bytes,
        &layer_indices,
        sequence_len,
        rotary_dim,
        rope_base,
        position_offset,
    )?;
    smoke_run.execute(&mut context, &mut stream)?;
    smoke_run.format_output(path, device_index, &info)
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

    let conv_activated = runtime_host_silu_f32(&conv_output);
    let qkv_split = match split_linear_attn_qkv_for_recurrent(
        &conv_activated,
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

    let conv_activated = runtime_host_silu_f32(&conv_output);
    let qkv_split = match split_linear_attn_qkv_for_recurrent(
        &conv_activated,
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
    let input_norm_weight_values =
        effective_rmsnorm_weight_values(&input_norm_tensor, &input_norm.values);
    let post_norm_weight_values =
        effective_rmsnorm_weight_values(&post_norm_tensor, &post_norm.values);
    let input_norm_weight_bytes = encode_f32_to_bytes(&input_norm_weight_values);
    let conv_weight_bytes = encode_f32_to_bytes(&conv.values);
    let a_log_bytes = encode_f32_to_bytes(&a_log.values);
    let dt_bias_bytes = encode_f32_to_bytes(&dt_bias.values);
    let attn_norm_weight_bytes = encode_f32_to_bytes(&attn_norm.values);
    let post_norm_weight_bytes = encode_f32_to_bytes(&post_norm_weight_values);

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
        runtime_host_rmsnorm_f32(&residual, &input_norm_weight_values, input_epsilon);
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

        let conv_activated = runtime_host_silu_f32(&conv_output);
        let qkv_split = split_linear_attn_qkv_for_recurrent(
            &conv_activated,
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

    let post_normed_expected = runtime_host_rmsnorm_f32(
        &attention_block_output,
        &post_norm_weight_values,
        mlp_epsilon,
    );
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
    let hidden = 32_usize * 128_usize;
    let base_residual = deterministic_f32_vector(hidden);
    let mut residual_sequence = Vec::with_capacity(sequence_len * hidden);
    for timestep in 0..sequence_len {
        residual_sequence.extend(linear_attn_step_input(&base_residual, timestep));
    }
    let run = package_linear_attn_mlp_block_sequence_run(
        path,
        device_index,
        chunk_bytes,
        layer_index,
        sequence_len,
        residual_sequence,
        None,
    )?;
    Ok(run.line)
}

const PACKAGE_ROW_SCALE_OVERRIDES_SCHEMA_VERSION: &str = "package-row-scale-overrides-v0.1";

#[derive(Debug, Clone)]
struct PackageRowScaleOverrides {
    source_path: String,
    overrides: Vec<PackageRowScaleOverride>,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
struct PackageRowScaleOverridesFile {
    schema_version: String,
    overrides: Vec<PackageRowScaleOverride>,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
struct PackageRowScaleOverride {
    layer_index: usize,
    tensor_suffix: String,
    row_index: usize,
    scale: f32,
}

#[derive(Debug, Clone, serde::Serialize)]
struct AppliedPackageRowScaleOverride {
    layer_index: usize,
    tensor_name: String,
    tensor_suffix: String,
    row_index: usize,
    scale: f32,
    rows: usize,
    cols: usize,
}

fn load_package_row_scale_overrides(
    path: Option<&str>,
) -> Result<Option<PackageRowScaleOverrides>, String> {
    let Some(path) = path else {
        return Ok(None);
    };
    if path.is_empty() || path == "none" {
        return Ok(None);
    }
    let raw = fs::read_to_string(path)
        .map_err(|err| format!("failed to read row scale overrides JSON {path}: {err}"))?;
    let parsed: PackageRowScaleOverridesFile = serde_json::from_str(&raw)
        .map_err(|err| format!("failed to parse row scale overrides JSON {path}: {err}"))?;
    if parsed.schema_version != PACKAGE_ROW_SCALE_OVERRIDES_SCHEMA_VERSION {
        return Err(format!(
            "row scale overrides schema_version must be {}, got {}",
            PACKAGE_ROW_SCALE_OVERRIDES_SCHEMA_VERSION, parsed.schema_version
        ));
    }

    let mut seen = std::collections::BTreeSet::<(usize, String, usize)>::new();
    for override_entry in &parsed.overrides {
        validate_package_row_scale_override(override_entry)?;
        let key = (
            override_entry.layer_index,
            override_entry.tensor_suffix.clone(),
            override_entry.row_index,
        );
        if !seen.insert(key) {
            return Err(format!(
                "duplicate row scale override: layer={} tensor_suffix={} row={}",
                override_entry.layer_index, override_entry.tensor_suffix, override_entry.row_index
            ));
        }
    }

    Ok(Some(PackageRowScaleOverrides {
        source_path: path.to_string(),
        overrides: parsed.overrides,
    }))
}

fn validate_package_row_scale_override(
    override_entry: &PackageRowScaleOverride,
) -> Result<(), String> {
    if !matches!(
        override_entry.tensor_suffix.as_str(),
        "linear_attn.out_proj.weight" | "self_attn.o_proj.weight" | "mlp.down_proj.weight"
    ) {
        return Err(format!(
            "unsupported row scale override tensor_suffix={}; expected linear_attn.out_proj.weight, self_attn.o_proj.weight, or mlp.down_proj.weight",
            override_entry.tensor_suffix
        ));
    }
    if !override_entry.scale.is_finite() || override_entry.scale <= 0.0 {
        return Err(format!(
            "row scale override must be finite and positive: layer={} tensor_suffix={} row={} scale={}",
            override_entry.layer_index,
            override_entry.tensor_suffix,
            override_entry.row_index,
            override_entry.scale
        ));
    }
    Ok(())
}

fn matching_package_row_scale_overrides(
    row_scale_overrides: Option<&PackageRowScaleOverrides>,
    layer_index: usize,
    tensor_suffix: &str,
) -> Vec<PackageRowScaleOverride> {
    row_scale_overrides
        .map(|overrides| {
            overrides
                .overrides
                .iter()
                .filter(|override_entry| {
                    override_entry.layer_index == layer_index
                        && override_entry.tensor_suffix == tensor_suffix
                })
                .cloned()
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

fn apply_package_row_scale_overrides_to_matrix(
    stream: &mut ullm_runtime_sys::RuntimeStream,
    matrix: &mut ullm_runtime_sys::RuntimeBuffer,
    rows: usize,
    cols: usize,
    tensor_name: &str,
    overrides: &[PackageRowScaleOverride],
) -> Result<Vec<AppliedPackageRowScaleOverride>, String> {
    if overrides.is_empty() {
        return Ok(Vec::new());
    }
    if rows == 0 || cols == 0 {
        return Err(format!(
            "cannot apply row scale overrides to empty matrix {tensor_name}: rows={rows} cols={cols}"
        ));
    }
    let matrix_bytes_len = rows
        .checked_mul(cols)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| {
            format!("row scale override matrix byte size overflows for {tensor_name}")
        })?;
    let mut matrix_bytes = vec![0_u8; matrix_bytes_len];
    matrix
        .copy_to_host(0, &mut matrix_bytes, Some(stream))
        .map_err(|err| format!("failed to copy {tensor_name} for row scale overrides: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize after {tensor_name} override copy: {err}"))?;

    let mut applied = Vec::with_capacity(overrides.len());
    for override_entry in overrides {
        if override_entry.row_index >= rows {
            return Err(format!(
                "row scale override row out of range for {tensor_name}: row={} rows={rows}",
                override_entry.row_index
            ));
        }
        let row_start = override_entry
            .row_index
            .checked_mul(cols)
            .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| format!("row scale override row offset overflows for {tensor_name}"))?;
        let row_end = row_start
            .checked_add(
                cols.checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| {
                        format!("row scale override row byte size overflows for {tensor_name}")
                    })?,
            )
            .ok_or_else(|| format!("row scale override row end overflows for {tensor_name}"))?;
        for offset in (row_start..row_end).step_by(std::mem::size_of::<f32>()) {
            let mut raw = [0_u8; 4];
            raw.copy_from_slice(&matrix_bytes[offset..offset + 4]);
            let scaled = f32::from_le_bytes(raw) * override_entry.scale;
            matrix_bytes[offset..offset + 4].copy_from_slice(&scaled.to_le_bytes());
        }
        applied.push(AppliedPackageRowScaleOverride {
            layer_index: override_entry.layer_index,
            tensor_name: tensor_name.to_string(),
            tensor_suffix: override_entry.tensor_suffix.clone(),
            row_index: override_entry.row_index,
            scale: override_entry.scale,
            rows,
            cols,
        });
    }

    matrix
        .copy_from_host(0, &matrix_bytes, Some(stream))
        .map_err(|err| format!("failed to copy row-scaled {tensor_name} back to runtime: {err}"))?;
    stream.synchronize().map_err(|err| {
        format!("failed to synchronize after row-scaled {tensor_name} copy back: {err}")
    })?;
    Ok(applied)
}

struct PackageLinearAttnMlpBlockSequenceRun {
    line: String,
    applied_row_scale_overrides: Vec<AppliedPackageRowScaleOverride>,
    attention_input_normed: Vec<f32>,
    attention_qkv_projection: Vec<f32>,
    attention_qkv_projection_dim: usize,
    attention_z_projection: Vec<f32>,
    attention_gate_silu: Vec<f32>,
    attention_a_projection: Vec<f32>,
    attention_b_projection: Vec<f32>,
    attention_gate_dim: usize,
    attention_conv_pre_silu: Vec<f32>,
    attention_conv: Vec<f32>,
    attention_recurrent_q: Vec<f32>,
    attention_recurrent_k: Vec<f32>,
    attention_recurrent_v: Vec<f32>,
    attention_recurrent_qk_dim: usize,
    attention_gate: Vec<f32>,
    attention_beta: Vec<f32>,
    attention_recurrent: Vec<f32>,
    attention_normed: Vec<f32>,
    attention_projection_input: Vec<f32>,
    attention_output: Vec<f32>,
    attention_block_output: Vec<f32>,
    post_normed: Vec<f32>,
    mlp_activation: Vec<f32>,
    mlp_intermediate: usize,
    mlp_output: Vec<f32>,
    layer_output: Vec<f32>,
}

fn package_linear_attn_mlp_block_sequence_run(
    path: &str,
    device_index: u32,
    chunk_bytes: usize,
    layer_index: usize,
    sequence_len: usize,
    residual_sequence: Vec<f32>,
    row_scale_overrides: Option<&PackageRowScaleOverrides>,
) -> Result<PackageLinearAttnMlpBlockSequenceRun, String> {
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

    if residual_sequence.len() != sequence_len * hidden {
        return Err(format!(
            "linear attention residual sequence length mismatch for layer {layer_index}: got {} expected {}",
            residual_sequence.len(),
            sequence_len * hidden
        ));
    }
    let input_norm_weight_values =
        effective_rmsnorm_weight_values(&input_norm_tensor, &input_norm.values);
    let post_norm_weight_values =
        effective_rmsnorm_weight_values(&post_norm_tensor, &post_norm.values);
    let input_norm_weight_bytes = encode_f32_to_bytes(&input_norm_weight_values);
    let conv_weight_bytes = encode_f32_to_bytes(&conv.values);
    let a_log_bytes = encode_f32_to_bytes(&a_log.values);
    let dt_bias_bytes = encode_f32_to_bytes(&dt_bias.values);
    let attn_norm_weight_bytes = encode_f32_to_bytes(&attn_norm.values);
    let post_norm_weight_bytes = encode_f32_to_bytes(&post_norm_weight_values);
    let mut applied_row_scale_overrides = Vec::new();

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

    let mut expected_input_normed = Vec::with_capacity(sequence_len * hidden);
    let mut input_normed_sequence_bytes = vec![0_u8; hidden_sequence_bytes];
    for timestep in 0..sequence_len {
        let residual_start = timestep * hidden;
        let residual_end = residual_start + hidden;
        let residual = &residual_sequence[residual_start..residual_end];
        let residual_bytes = encode_f32_to_bytes(residual);
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
        let expected = runtime_host_rmsnorm_f32(residual, &input_norm_weight_values, input_epsilon);
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
        qkv_output,
        z_output,
        a_output,
        b_output,
        conv_output,
        conv_activated,
        recurrent_q,
        recurrent_k,
        recurrent_v,
        gate_output,
        beta_output,
        recurrent_output,
        attn_normed,
        attn_activated,
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
        let (out_rows, out_cols, mut out_matrix) = materialize_selected_aq4_matrix(
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
        let out_row_scale_overrides = matching_package_row_scale_overrides(
            row_scale_overrides,
            layer_index,
            "linear_attn.out_proj.weight",
        );
        applied_row_scale_overrides.extend(apply_package_row_scale_overrides_to_matrix(
            &mut stream,
            &mut out_matrix,
            out_rows,
            out_cols,
            &out_tensor,
            &out_row_scale_overrides,
        )?);

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

        let conv_activated = runtime_host_silu_f32(&conv_output);
        let qkv_split = split_linear_attn_qkv_for_recurrent(
            &conv_activated,
            sequence_len,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            qk_l2_norm,
            q_scale,
        )
        .map_err(|err| format!("failed to split qkv for recurrent: {err}"))?;
        let recurrent_q = qkv_split.q.clone();
        let recurrent_k = qkv_split.k.clone();
        let recurrent_v = qkv_split.v.clone();
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
            qkv_output,
            z_output,
            a_output,
            b_output,
            conv_output,
            conv_activated,
            recurrent_q,
            recurrent_k,
            recurrent_v,
            gate_output,
            beta_output,
            recurrent_output,
            attn_normed,
            attn_activated,
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

    let z_silu_output = runtime_host_silu_f32(&z_output);

    let mut post_normed_expected = Vec::with_capacity(sequence_len * hidden);
    for timestep in 0..sequence_len {
        let start = timestep * hidden;
        let end = start + hidden;
        let expected = runtime_host_rmsnorm_f32(
            &attention_block_output[start..end],
            &post_norm_weight_values,
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

    let (mlp_activation, mlp_intermediate, mlp_output, layer_output, layer_block_max_abs_diff) = {
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
        let (down_rows, down_cols, mut down_matrix) = materialize_selected_aq4_matrix(
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
        let down_row_scale_overrides = matching_package_row_scale_overrides(
            row_scale_overrides,
            layer_index,
            "mlp.down_proj.weight",
        );
        applied_row_scale_overrides.extend(apply_package_row_scale_overrides_to_matrix(
            &mut stream,
            &mut down_matrix,
            down_rows,
            down_cols,
            &down_tensor,
            &down_row_scale_overrides,
        )?);
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
        let mut mlp_activated_bytes = vec![
            0_u8;
            intermediate_bytes.checked_mul(sequence_len).ok_or_else(
                || "MLP activated sequence byte size overflows".to_string()
            )?
        ];
        for timestep in 0..sequence_len {
            let byte_start = timestep * hidden_bytes;
            let byte_end = byte_start + hidden_bytes;
            let intermediate_byte_start = timestep * intermediate_bytes;
            let intermediate_byte_end = intermediate_byte_start + intermediate_bytes;
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
            mlp_activated_buffer
                .copy_to_host(
                    0,
                    &mut mlp_activated_bytes[intermediate_byte_start..intermediate_byte_end],
                    Some(&mut stream),
                )
                .map_err(|err| {
                    format!("failed to copy MLP activated timestep {timestep}: {err}")
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
        let mlp_activation = decode_f32_le_values(&mlp_activated_bytes);
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
        (
            mlp_activation,
            intermediate,
            mlp_output,
            layer_output,
            layer_block_max_abs_diff,
        )
    };

    let line = format!(
        "package-linear-attn-mlp-block-smoke package={} layer={} input_norm_tensor=\"{}\" qkv_tensor=\"{}\" conv_tensor=\"{}\" a_tensor=\"{}\" b_tensor=\"{}\" a_log_tensor=\"{}\" dt_bias_tensor=\"{}\" z_tensor=\"{}\" norm_tensor=\"{}\" out_tensor=\"{}\" post_norm_tensor=\"{}\" gate_tensor=\"{}\" up_tensor=\"{}\" down_tensor=\"{}\" hidden={} key_heads={} value_heads={} key_dim={} value_dim={} sequence_len={} kernel_size={} qk_l2_norm={} q_scale={q_scale:.9} input_norm_dtype={} conv_dtype={} a_log_dtype={} dt_bias_dtype={} norm_dtype={} post_norm_dtype={} row_scale_overrides={} backend={} device_index={} name=\"{}\" residual_preview={} attention_output_preview={} attention_block_preview={} post_norm_preview={} mlp_output_preview={} layer_output_preview={} input_norm_max_abs_diff={input_norm_max_abs_diff:.9} conv_max_abs_diff={conv_max_abs_diff:.9} gate_beta_max_abs_diff={gate_beta_max_abs_diff:.9} recurrent_max_abs_diff={recurrent_max_abs_diff:.9} attn_norm_max_abs_diff={attn_norm_max_abs_diff:.9} attn_activation_max_abs_diff={attn_activation_max_abs_diff:.9} attn_output_max_abs_diff={attn_output_max_abs_diff:.9} attn_block_max_abs_diff={attn_block_max_abs_diff:.9} post_norm_max_abs_diff={post_norm_max_abs_diff:.9} layer_block_max_abs_diff={layer_block_max_abs_diff:.9} verified=true",
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
        applied_row_scale_overrides.len(),
        info.backend,
        device_index,
        info.name,
        format_f32_preview(&residual_sequence[..8.min(residual_sequence.len())]),
        format_f32_preview(&attn_output[..8.min(attn_output.len())]),
        format_f32_preview(&attention_block_output[..8.min(attention_block_output.len())]),
        format_f32_preview(&post_normed[..8.min(post_normed.len())]),
        format_f32_preview(&mlp_output[..8.min(mlp_output.len())]),
        format_f32_preview(&layer_output[..8.min(layer_output.len())]),
    );
    Ok(PackageLinearAttnMlpBlockSequenceRun {
        line,
        applied_row_scale_overrides,
        attention_input_normed: input_normed,
        attention_qkv_projection: qkv_output,
        attention_qkv_projection_dim: qkv_rows_expected,
        attention_z_projection: z_output,
        attention_gate_silu: z_silu_output,
        attention_a_projection: a_output,
        attention_b_projection: b_output,
        attention_gate_dim: value_heads,
        attention_conv_pre_silu: conv_output,
        attention_conv: conv_activated,
        attention_recurrent_q: recurrent_q,
        attention_recurrent_k: recurrent_k,
        attention_recurrent_v: recurrent_v,
        attention_recurrent_qk_dim: key_heads * key_dim,
        attention_gate: gate_output,
        attention_beta: beta_output,
        attention_recurrent: recurrent_output,
        attention_normed: attn_normed,
        attention_projection_input: attn_activated,
        attention_output: attn_output,
        attention_block_output,
        post_normed,
        mlp_activation,
        mlp_intermediate,
        mlp_output,
        layer_output,
    })
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

    let output = qwen3_headwise_rmsnorm_to_host_f32(context, stream, input, weight, epsilon)
        .map_err(|err| format!("failed to run {label} RMSNorm: {err}"))?;
    if output.len() != input.len() {
        return Err(format!(
            "{label} runtime output size mismatch: expected {} got {}",
            input.len(),
            output.len()
        ));
    }
    let mut max_abs_diff = 0.0_f32;

    for (head_index, head_input) in input.chunks_exact(head_dim).enumerate() {
        let actual_head_start = head_index
            .checked_mul(head_dim)
            .ok_or_else(|| format!("{label} head index multiplication overflow"))?;
        let actual_head_end = actual_head_start
            .checked_add(head_dim)
            .ok_or_else(|| format!("{label} head length multiplication overflow"))?;
        let actual = &output[actual_head_start..actual_head_end];
        let expected = runtime_host_rmsnorm_f32(head_input, weight, epsilon);
        let head_max_abs_diff = verify_f32_close(
            &format!("{label} head {head_index}"),
            &actual,
            &expected,
            1e-4_f32,
            1e-4_f32,
        )?;
        max_abs_diff = max_abs_diff.max(head_max_abs_diff);
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
    let output = qwen3_rope_to_host_f32(
        context,
        stream,
        input,
        sequence_len,
        heads,
        head_dim,
        rotary_dim,
        position_offset,
        rope_base,
    )
    .map_err(|err| format!("failed to run {label} RoPE: {err}"))?;
    if output.len() != input.len() {
        return Err(format!(
            "{label} runtime output size mismatch: expected {} got {}",
            input.len(),
            output.len()
        ));
    }
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
    let output = qwen3_causal_attn_to_host_f32(
        context,
        stream,
        q,
        k,
        v,
        sequence_len,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
    )
    .map_err(|err| format!("failed to run {label} causal attention: {err}"))?;
    let output_elements = sequence_len
        .checked_mul(q_heads)
        .ok_or_else(|| format!("{label} output element count overflows"))?
        .checked_mul(value_dim)
        .ok_or_else(|| format!("{label} output element count overflows"))?;
    if output.len() != output_elements {
        return Err(format!(
            "{label} runtime output size mismatch: expected {} got {}",
            output_elements,
            output.len()
        ));
    }
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
    cache_blocks: usize,
    block_table: Vec<u32>,
    allocator_stats: KvBlockAllocatorStats,
    scheduler_request_id: RequestId,
    scheduler_prefill_tokens: usize,
    scheduler_max_new_tokens: usize,
    scheduler_cached_tokens: usize,
    scheduler_generated_tokens: usize,
    scheduler_active_len: usize,
    scheduler_decode_batches: usize,
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
    cache_len: usize,
    block_size: usize,
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
    let prepared = prepare_fragmented_paged_decode_state(cache_len, block_size)?;
    let mut scheduler = prepared.scheduler;
    let prefill_prompt_tokens = prepared.prefill_tokens;
    let max_new_tokens = prepared.max_new_tokens;
    let block_table = prepared.block_table;
    let cache_blocks = prepared.cache_blocks;
    let scheduler_request_id = prepared.request_id;

    let readback_shape = PagedDecodeShape {
        block_size,
        cache_blocks,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
    };
    let PagedKvCacheReadback {
        k: expected_k_cache,
        v: expected_v_cache,
    } = pack_paged_kv_cache_for_block_table(
        logical_k_cache,
        logical_v_cache,
        &block_table,
        cache_len,
        readback_shape,
    )?;
    let shape = PagedDecodeShape {
        block_size,
        cache_blocks,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
    };
    let mut decode_runner = Qwen3SelfAttnRequestDecodeRunner::new();
    decode_runner.insert_request(
        context,
        stream,
        scheduler_request_id,
        shape,
        block_table.to_vec(),
        softmax_scale,
    )?;
    let q_token_elements = q_heads * head_dim;
    let k_token_elements = kv_heads * head_dim;
    let v_token_elements = kv_heads * value_dim;
    let output_elements = q_heads * value_dim;
    let mut step_outputs = Vec::with_capacity(cache_len * output_elements);
    let mut step_output_max_abs_diff = 0.0_f32;
    let mut scheduler_decode_batches = 0_usize;

    for timestep in 0..prefill_prompt_tokens {
        let q_start = timestep * q_token_elements;
        let q_end = q_start + q_token_elements;
        let k_start = timestep * k_token_elements;
        let k_end = k_start + k_token_elements;
        let v_start = timestep * v_token_elements;
        let v_end = v_start + v_token_elements;

        let step = decode_runner
            .run_prefill_step(
                stream,
                Qwen3SelfAttnDecodeBatchInput {
                    request_id: scheduler_request_id,
                    q: &q_sequence[q_start..q_end],
                    k: &logical_k_cache[k_start..k_end],
                    v: &logical_v_cache[v_start..v_end],
                },
            )
            .map_err(|err| {
                format!("{label} failed to run prefix/prefill decode timestep {timestep}: {err}")
            })?;
        if step.cache_position != timestep {
            return Err(format!(
                "{label} prefix/prefill decode request wrote position {}, expected {timestep}",
                step.cache_position
            ));
        }
        if step.cache_len != timestep + 1 {
            return Err(format!(
                "{label} prefix/prefill decode request reported cache_len {}, expected {}",
                step.cache_len,
                timestep + 1
            ));
        }
        if step.attention_output.len() != output_elements {
            return Err(format!(
                "{label} prefix/prefill timestep {timestep} produced {} outputs, expected {output_elements}",
                step.attention_output.len()
            ));
        }

        let expected_step_output = runtime_host_paged_decode_attn_f32(
            &q_sequence[q_start..q_end],
            &expected_k_cache,
            &expected_v_cache,
            &block_table,
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

    scheduler
        .complete_prefill(scheduler_request_id)
        .map_err(|err| format!("failed to complete decode prefill in {label}: {err}"))?;

    for timestep in prefill_prompt_tokens..cache_len {
        let decode_requests = scheduler
            .ready_decode_batch(1)
            .map_err(|err| format!("failed to ready decode batch in {label}: {err}"))?;
        let request = decode_requests.first().ok_or_else(|| {
            format!("{label} expected one ready decode request for timestep {timestep}, got none")
        })?;

        if request.cache_position != timestep {
            return Err(format!(
                "{label} ready decode request cache position {} does not match timestep {timestep}",
                request.cache_position
            ));
        }
        if request.next_cache_len != timestep + 1 {
            return Err(format!(
                "{label} ready decode request next cache len {} does not match {}",
                request.next_cache_len,
                timestep + 1
            ));
        }
        if request.request.id != scheduler_request_id {
            return Err(format!(
                "{label} ready decode request id {:?} does not match scheduler request {:?}",
                request.request.id, scheduler_request_id
            ));
        }

        let q_start = timestep * q_token_elements;
        let q_end = q_start + q_token_elements;
        let k_start = timestep * k_token_elements;
        let k_end = k_start + k_token_elements;
        let v_start = timestep * v_token_elements;
        let v_end = v_start + v_token_elements;

        let inputs = [Qwen3SelfAttnDecodeBatchInput {
            request_id: request.request.id,
            q: &q_sequence[q_start..q_end],
            k: &logical_k_cache[k_start..k_end],
            v: &logical_v_cache[v_start..v_end],
        }];
        let mut outputs = decode_runner
            .run_ready_batch(stream, &mut scheduler, &decode_requests, &inputs)
            .map_err(|err| format!("failed to run {label} timestep {timestep}: {err}"))?;
        let step = outputs.pop().ok_or_else(|| {
            format!("{label} ready decode batch produced no output for timestep {timestep}")
        })?;
        if step.request_id != scheduler_request_id {
            return Err(format!(
                "{label} output request id {:?} does not match scheduler request {:?}",
                step.request_id, scheduler_request_id
            ));
        }

        if step.cache_position != request.cache_position {
            return Err(format!(
                "{label} paged decode state wrote position {}, expected {}",
                step.cache_position, request.cache_position
            ));
        }
        if step.cache_len != request.next_cache_len {
            return Err(format!(
                "{label} paged decode state reported cache_len {}, expected {}",
                step.cache_len, request.next_cache_len
            ));
        }
        let expected_step_output = runtime_host_paged_decode_attn_f32(
            &q_sequence[q_start..q_end],
            &expected_k_cache,
            &expected_v_cache,
            &block_table,
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

        scheduler_decode_batches += 1;
    }

    let scheduler_active = scheduler
        .active_request(scheduler_request_id)
        .ok_or_else(|| {
            format!(
                "{label} decode request {:?} missing after scheduler progress",
                scheduler_request_id
            )
        })?;

    let readback = decode_runner
        .read_cache_to_host(scheduler_request_id, stream)
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
        &block_table,
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
        cache_blocks,
        block_table,
        allocator_stats: scheduler.allocator_stats(),
        scheduler_request_id,
        scheduler_prefill_tokens: prefill_prompt_tokens,
        scheduler_max_new_tokens: max_new_tokens,
        scheduler_cached_tokens: scheduler_active.cached_tokens,
        scheduler_generated_tokens: scheduler_active.generated_tokens,
        scheduler_active_len: scheduler.active_len(),
        scheduler_decode_batches,
        output_max_abs_diff,
        step_output_max_abs_diff,
        k_cache: readback.k,
        v_cache: readback.v,
        k_write_max_abs_diff,
        v_write_max_abs_diff,
    })
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

fn print_help() {
    eprintln!(
        "usage: ullm-engine <inspect-devices|runtime-smoke|runtime-memory-smoke [DEVICE_INDEX]|runtime-stream-smoke [DEVICE_INDEX]|runtime-copy-smoke [DEVICE_INDEX]|runtime-rmsnorm-smoke [DEVICE_INDEX]|runtime-silu-mul-smoke [DEVICE_INDEX]|runtime-sigmoid-mul-smoke [DEVICE_INDEX]|runtime-add-smoke [DEVICE_INDEX]|runtime-rope-smoke [DEVICE_INDEX]|runtime-causal-attn-smoke [DEVICE_INDEX]|runtime-decode-attn-smoke [DEVICE_INDEX]|runtime-paged-decode-attn-smoke [DEVICE_INDEX]|runtime-paged-kv-write-smoke [DEVICE_INDEX]|runtime-scheduler-paged-decode-smoke [DEVICE_INDEX]|runtime-scheduler-layer-decode-smoke [DEVICE_INDEX]|runtime-kv-paged-decode-smoke [DEVICE_INDEX]|runtime-depthwise-conv1d-smoke [DEVICE_INDEX]|runtime-linear-attn-gate-beta-smoke [DEVICE_INDEX]|runtime-linear-attn-recurrent-smoke [DEVICE_INDEX]|runtime-mlp-smoke [DEVICE_INDEX]|inspect-package PATH|package-load-smoke PACKAGE_DIR [DEVICE_INDEX] [MAX_BYTES] [PAYLOAD_ROLE]|package-tensor-load-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [TENSOR_SELECTOR]|package-weight-register-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [TENSOR_SELECTOR]|package-weight-register-many-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [MAX_TENSORS]|package-materialize-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [TENSOR_SELECTOR]|package-mlp-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX]|package-materialize-matvec-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [TENSOR_SELECTOR]|package-rmsnorm-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [input|post]|package-rmsnorm-mlp-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [input|post]|package-mlp-block-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [input|post]|package-linear-attn-proj-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [a|b|qkv|z|out|all]|package-self-attn-proj-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [q|k|v|o|all]|package-self-attn-qk-norm-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX]|package-self-attn-rope-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [SEQUENCE_LEN] [ROTARY_DIM] [ROPE_BASE] [POSITION_OFFSET]|package-self-attn-attention-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [SEQUENCE_LEN] [ROTARY_DIM] [ROPE_BASE] [POSITION_OFFSET]|package-self-attn-decode-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [SEQUENCE_LEN] [ROTARY_DIM] [ROPE_BASE] [POSITION_OFFSET]|package-self-attn-block-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [SEQUENCE_LEN] [ROTARY_DIM] [ROPE_BASE] [POSITION_OFFSET]|package-self-attn-mlp-block-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [SEQUENCE_LEN] [ROTARY_DIM] [ROPE_BASE] [POSITION_OFFSET]|package-self-attn-mlp-block-scheduler-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [SEQUENCE_LEN] [ROTARY_DIM] [ROPE_BASE] [POSITION_OFFSET]|package-self-attn-mlp-block-model-loop-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX,...|FIRST_LAYER_INDEX SECOND_LAYER_INDEX[,...]] [SEQUENCE_LEN] [ROTARY_DIM] [ROPE_BASE] [POSITION_OFFSET]|package-layer-golden-smoke PACKAGE_DIR GOLDEN_FIXTURE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [ROTARY_DIM] [ROPE_BASE] [POSITION_OFFSET]|package-golden-prefix-smoke PACKAGE_DIR GOLDEN_FIXTURE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_START] [LAYER_END_EXCLUSIVE] [ROTARY_DIM] [ROPE_BASE] [POSITION_OFFSET] [REPORT_PATH] [RUN_MODE] [ROW_SCALE_OVERRIDES_JSON] [INPUT_DUMP_DIR] [SAMPLED_TOKEN_INDICES]|package-linear-attn-qkv-norm-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX]|package-linear-attn-conv1d-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [SEQUENCE_LEN]|package-linear-attn-gate-beta-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [SEQUENCE_LEN]|package-linear-attn-recurrent-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [SEQUENCE_LEN]|package-linear-attn-post-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [SEQUENCE_LEN]|package-linear-attn-workflow-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [SEQUENCE_LEN]|package-linear-attn-block-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [SEQUENCE_LEN]|package-linear-attn-mlp-block-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [SEQUENCE_LEN]|package-linear-attn-aux-smoke PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYER_INDEX] [a-log|dt-bias|conv1d|norm|all]|package-materialize-bench PACKAGE_DIR [DEVICE_INDEX] [CHUNK_BYTES] [TENSOR_SELECTOR] [REPEATS]>"
    );
    eprintln!("linear attention projection selector: a|b|qkv|z|out|all");
    eprintln!("self attention projection selector: q|k|v|o|all (alias: out for o)");
    eprintln!(
        "model-loop layer list: use LAYER_INDEX,... or FIRST_LAYER_INDEX SECOND_LAYER_INDEX[,...]"
    );
    eprintln!(
        "linear attention aux selector: a-log|dt-bias|conv1d|norm|all (aliases: a_log|alog|dt_bias)"
    );
    eprintln!(
        "payload roles: smallest|tensor-index|tensor-scale|tensor-codebook|codebook|passthrough"
    );
    eprintln!("tensor selector: omitted or numeric index, exact tensor name, or unique substring");
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

fn runtime_host_silu_f32(values: &[f32]) -> Vec<f32> {
    values
        .iter()
        .map(|value| {
            let value = *value;
            value * (1.0_f32 / (1.0_f32 + (-value).exp()))
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

struct ScheduledPagedDecodeBlocks {
    block_table: Vec<u32>,
    cache_blocks: usize,
    allocator_stats: KvBlockAllocatorStats,
    request_id: RequestId,
    prefill_tokens: usize,
    max_new_tokens: usize,
    cached_tokens: usize,
    generated_tokens: usize,
    active_len: usize,
}

struct PreparedFragmentedPagedDecodeState {
    scheduler: SchedulerState,
    block_table: Vec<u32>,
    cache_blocks: usize,
    request_id: RequestId,
    prefill_tokens: usize,
    max_new_tokens: usize,
}

fn prepare_fragmented_paged_decode_state(
    cache_len: usize,
    block_size: usize,
) -> Result<PreparedFragmentedPagedDecodeState, String> {
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

    let mut scheduler = SchedulerState::with_block_size(cache_blocks as u32, block_size as u32);
    let fragment_blocks = cache_blocks - 1;
    let fragment_tokens = fragment_blocks
        .checked_mul(block_size)
        .ok_or_else(|| "paged decode fragment token count overflows".to_string())?;
    scheduler.enqueue(Request {
        id: RequestId(100),
        prompt_tokens: fragment_tokens,
        max_new_tokens: 0,
    });
    let fragment_batch = scheduler
        .pop_prefill_batch_with_allocation(fragment_tokens)
        .map_err(|err| format!("failed to allocate fragmenting KV blocks: {err}"))?;
    let fragment = fragment_batch
        .first()
        .ok_or_else(|| "fragmenting KV allocation returned an empty batch".to_string())?;
    let freed = scheduler.release_request(fragment.allocation.request_id);
    if freed != fragment.allocation.blocks.len() {
        return Err(format!(
            "freed KV block count {freed} does not match allocated fragment blocks {}",
            fragment.allocation.blocks.len()
        ));
    }

    let request_id = RequestId(101);
    let (prefill_prompt_tokens, max_new_tokens) = if cache_len > 1 {
        (cache_len - 1, 1)
    } else {
        (cache_len, 0)
    };
    scheduler.enqueue(Request {
        id: request_id,
        prompt_tokens: prefill_prompt_tokens,
        max_new_tokens,
    });
    let mut decode_batch = scheduler
        .pop_prefill_batch_with_allocation(prefill_prompt_tokens)
        .map_err(|err| format!("failed to allocate decode KV blocks: {err}"))?;
    if decode_batch.len() != 1 {
        return Err(format!(
            "decode KV allocation selected {} requests, expected 1",
            decode_batch.len()
        ));
    }
    let allocation = decode_batch.remove(0).allocation;
    if allocation.blocks.len() != block_count {
        return Err(format!(
            "decode KV allocation block count {} does not match expected {block_count}",
            allocation.blocks.len()
        ));
    }

    Ok(PreparedFragmentedPagedDecodeState {
        scheduler,
        block_table: allocation.blocks,
        cache_blocks,
        request_id,
        prefill_tokens: prefill_prompt_tokens,
        max_new_tokens,
    })
}

fn allocate_fragmented_paged_decode_blocks(
    cache_len: usize,
    block_size: usize,
) -> Result<ScheduledPagedDecodeBlocks, String> {
    let prepared = prepare_fragmented_paged_decode_state(cache_len, block_size)?;
    let mut scheduler = prepared.scheduler;
    let cache_blocks = prepared.cache_blocks;
    let request_id = prepared.request_id;

    scheduler
        .complete_prefill(request_id)
        .map_err(|err| format!("failed to complete decode prefill: {err}"))?;

    if prepared.max_new_tokens > 0 {
        let mut ready = scheduler
            .ready_decode_batch(1)
            .map_err(|err| format!("failed to prepare ready decode batch: {err}"))?;
        let request = ready
            .pop()
            .ok_or_else(|| "expected one ready decode request after prefill".to_string())?;
        if request.request.id != request_id {
            return Err(format!(
                "ready decode request {:?} does not match expected {:?}",
                request.request.id, request_id
            ));
        }
        if request.cache_position != prepared.prefill_tokens {
            return Err(format!(
                "ready decode cache_position {} does not match prefill tokens {}",
                request.cache_position, prepared.prefill_tokens
            ));
        }
        scheduler
            .advance_decode(request_id)
            .map_err(|err| format!("failed to advance decode by one token: {err}"))?;
    }

    let active = scheduler
        .active_request(request_id)
        .ok_or_else(|| "decode request is not active after scheduler progress".to_string())?;
    let cached_tokens = active.cached_tokens;
    let generated_tokens = active.generated_tokens;
    let active_len = scheduler.active_len();
    let stats = scheduler.allocator_stats();
    Ok(ScheduledPagedDecodeBlocks {
        block_table: prepared.block_table,
        cache_blocks,
        allocator_stats: stats,
        request_id,
        prefill_tokens: prepared.prefill_tokens,
        max_new_tokens: prepared.max_new_tokens,
        cached_tokens,
        generated_tokens,
        active_len,
    })
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
                let left_padding = kernel_size - 1 - kernel;
                if timestep < left_padding {
                    continue;
                }
                value += input[(timestep - left_padding) * channels + channel]
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
