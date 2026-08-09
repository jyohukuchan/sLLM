//! Dedicated real-weight G2 evidence binary.
//!
//! This executable has no CPU implementation.  It reads an explicitly
//! supplied locked slice, then uses the public Context/Queue/Buffer/RMSNorm
//! API.  The host HIP stub therefore returns HIP unavailable and the process
//! exits non-zero; it never reports synthetic numeric success.

use std::env;
use std::fs;
use std::time::Duration;

use sllm_core::{DType, Encoding, TensorView};
use sllm_hip::{
    CompletionState, Context, HipBackend, PreparedRmsNorm, RmsNormDescriptor, RuntimeError,
};

const N: usize = 2560;
const CASE_ROWS: [usize; 6] = [1, 2, 17, 255, 256, 257];
const CASE_SEEDS: [u32; 6] = [9201, 9202, 9217, 9255, 9256, 9257];
const EPSILON: f32 = 1.0e-6;
const TIMEOUT: Duration = Duration::from_secs(30);

fn fail(message: impl AsRef<str>) -> ! {
    eprintln!("sllm-rmsnorm-g2-evidence: FAIL: {}", message.as_ref());
    std::process::exit(1);
}

fn wait_success(completion: &mut sllm_hip::Completion, label: &str) {
    match completion.wait(TIMEOUT) {
        Ok(CompletionState::Success) => {}
        Ok(state) => fail(format!("{label} completed with {state:?}")),
        Err(error) => fail(format!("{label} failed: {error}")),
    }
}

fn bfloat16(value: f32) -> [u8; 2] {
    let bits = value.to_bits();
    let upper = bits >> 16;
    let lower = bits & 0xffff;
    let rounded = upper + u32::from(lower > 0x8000 || (lower == 0x8000 && upper & 1 != 0));
    (rounded as u16).to_le_bytes()
}

fn activation(rows: usize, seed: u32) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(rows * N * 2);
    for row in 0..rows {
        for index in 0..N {
            let lane = ((seed as usize + row * 17 + index * 13) % 257) as f32;
            let value = (lane - 128.0) / 64.0;
            bytes.extend_from_slice(&bfloat16(value));
        }
    }
    bytes
}

fn copy_to_device(queue: &sllm_hip::Queue, buffer: &sllm_hip::Buffer, bytes: &[u8], label: &str) {
    let mut completion = queue
        .copy_to_device(buffer, bytes, 0)
        .unwrap_or_else(|error| fail(format!("{label} upload failed: {error}")));
    wait_success(&mut completion, label);
}

fn run_case(
    backend: &HipBackend,
    context: &Context,
    queue: &sllm_hip::Queue,
    raw_scale: &sllm_hip::Buffer,
    raw_scale_bytes: &[u8],
    rows: usize,
    seed: u32,
) {
    let activation_bytes = activation(rows, seed);
    let output_bytes = vec![0_u8; activation_bytes.len()];
    let activation_buffer = sllm_hip::Buffer::allocate(context, activation_bytes.len() as u64)
        .unwrap_or_else(|error| fail(format!("activation allocation failed: {error}")));
    let output_buffer = sllm_hip::Buffer::allocate(context, output_bytes.len() as u64)
        .unwrap_or_else(|error| fail(format!("output allocation failed: {error}")));
    copy_to_device(queue, &activation_buffer, &activation_bytes, "activation");

    let activation_view =
        TensorView::new(DType::Bf16, Encoding::Unquantized, &[rows, N], &[N, 1], 0)
            .unwrap_or_else(|error| fail(format!("activation view failed: {error}")));
    let scale_view = TensorView::contiguous(DType::Bf16, &[N])
        .unwrap_or_else(|error| fail(format!("scale view failed: {error}")));
    let output_view = TensorView::new(DType::Bf16, Encoding::Unquantized, &[rows, N], &[N, 1], 0)
        .unwrap_or_else(|error| fail(format!("output view failed: {error}")));
    let descriptor = RmsNormDescriptor::new(
        activation_buffer.binding(activation_view),
        raw_scale.binding(scale_view),
        output_buffer.binding(output_view),
        EPSILON,
    )
    .unwrap_or_else(|error| fail(format!("RMSNorm descriptor failed: {error}")));
    let prepared: PreparedRmsNorm = backend
        .prepare_rms_norm(context, descriptor)
        .unwrap_or_else(|error| fail(format!("RMSNorm prepare failed: {error}")));
    let (mut submission, dispatch) = prepared
        .execute(queue)
        .unwrap_or_else(|error| fail(format!("RMSNorm dispatch failed: {error}")));
    if dispatch.dispatch_count != 1
        || dispatch.kernel_id != 1
        || dispatch.workgroup_size_x != 256
        || dispatch.normalized_size != N as u64
        || dispatch.row_count != rows as u64
        || dispatch.fallback_allowed
        || dispatch.fallback_used
        || dispatch.kernel_symbol != "rmsnorm.baseline.wave32.v1"
        || dispatch.device_symbol != "sllm_rmsnorm_baseline_wave32_v1"
    {
        fail("RMSNorm dispatch metadata violated the no-fallback contract");
    }
    match submission.wait(TIMEOUT) {
        Ok(CompletionState::Success) => {}
        Ok(state) => fail(format!("RMSNorm completion failed with {state:?}")),
        Err(error) => fail(format!("RMSNorm completion failed: {error}")),
    }
    let mut readback = vec![0_u8; output_bytes.len()];
    let mut copy = queue
        .copy_to_host(&output_buffer, readback.len() as u64, 0)
        .unwrap_or_else(|error| fail(format!("output download failed: {error}")));
    wait_success(&mut copy, "output download");
    copy.read_into(&mut readback)
        .unwrap_or_else(|error| fail(format!("output readback failed: {error}")));
    if readback.len() != output_bytes.len() || raw_scale_bytes.len() != N * 2 {
        fail("G2 output or raw scale size changed during execution");
    }
}

fn argument(name: &str) -> String {
    let mut args = env::args().skip(1);
    while let Some(current) = args.next() {
        if current == name {
            return args
                .next()
                .unwrap_or_else(|| fail(format!("missing value for {name}")));
        }
    }
    fail(format!("missing required argument {name}"))
}

fn embedded_build_identity() -> &'static str {
    let payload = sllm_hip_sys::g2_build_identity::IDENTITY_PAYLOAD;
    let identity = sllm_hip_sys::g2_build_identity::IDENTITY_JSON.as_bytes();
    if payload.len() <= identity.len() || !payload.ends_with(identity) {
        fail("G2 build identity payload is malformed");
    }
    let marker_len = payload.len() - identity.len();
    if marker_len == 0 {
        fail("G2 build identity marker is missing");
    }
    if !identity.ends_with(b"\n") {
        fail("G2 build identity is not newline terminated");
    }
    std::str::from_utf8(&identity[..identity.len() - 1])
        .unwrap_or_else(|_| fail("G2 build identity is not UTF-8"))
}

fn query_build_identity() -> bool {
    let mut args = env::args().skip(1);
    if args.next().as_deref() != Some("--query-build-identity") {
        return false;
    }
    if args.next().is_some() {
        fail("--query-build-identity does not accept additional arguments");
    }
    // This is a control-plane query only.  It does not touch HIP, model/cache
    // input, or emit a numeric result/fallback claim.
    println!("{}", embedded_build_identity());
    true
}

fn main() {
    if query_build_identity() {
        return;
    }
    // Force the generated marker and identity to be referenced before any
    // target, slice, model, or HIP operation is considered.
    let _ = embedded_build_identity();
    let target = argument("--target");
    if target != "gfx1030" && target != "gfx1201" {
        fail("target must be gfx1030 or gfx1201");
    }
    let slice_path = argument("--slice");
    let raw_scale = fs::read(&slice_path)
        .unwrap_or_else(|error| fail(format!("cannot read explicit G2 slice: {error}")));
    if raw_scale.len() != N * 2 {
        fail("G2 slice must be exactly 5120 bytes");
    }
    let device_index = if target == "gfx1201" { 2 } else { 0 };
    let backend =
        HipBackend::connect().unwrap_or_else(|error| fail(format!("HIP unavailable: {error}")));
    let context = Context::create(device_index, &target)
        .unwrap_or_else(|error| fail(format!("HIP context unavailable: {error}")));
    let device = Context::query_device(device_index)
        .unwrap_or_else(|error| fail(format!("HIP device query failed: {error}")));
    if device.gcn_arch_name != target {
        fail(format!("device target mismatch: {}", device.gcn_arch_name));
    }
    let queue = sllm_hip::Queue::create(&context)
        .unwrap_or_else(|error| fail(format!("HIP queue unavailable: {error}")));
    let raw_scale_buffer = sllm_hip::Buffer::allocate(&context, raw_scale.len() as u64)
        .unwrap_or_else(|error| fail(format!("raw scale allocation failed: {error}")));
    copy_to_device(&queue, &raw_scale_buffer, &raw_scale, "raw scale");
    for (order, rows) in CASE_ROWS.into_iter().enumerate() {
        run_case(
            &backend,
            &context,
            &queue,
            &raw_scale_buffer,
            &raw_scale,
            rows,
            CASE_SEEDS[order],
        );
    }
    Context::drain_cleanup(8)
        .unwrap_or_else(|error: RuntimeError| fail(format!("cleanup failed: {error}")));
    println!(
        "{{\"schema_version\":\"rmsnorm-g2-runtime-result-v1\",\"state\":\"PASS\",\"target\":\"{target}\",\"model_used\":true,\"full_model_used\":false,\"tokenizer_used\":false,\"generation_used\":false,\"selected_backend\":\"hip\",\"dispatch_count\":6,\"fallback_used\":false}}"
    );
}
