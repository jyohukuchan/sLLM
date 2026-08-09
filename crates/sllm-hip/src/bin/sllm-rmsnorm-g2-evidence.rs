//! Dedicated real-weight G2 evidence binary.
//!
//! This executable has no CPU implementation.  It reads an explicitly
//! supplied locked slice, then uses the public Context/Queue/Buffer/RMSNorm
//! API.  The host HIP stub therefore returns HIP unavailable and the process
//! exits non-zero; it never reports synthetic numeric success.

use std::env;
use std::fs::File;
use std::io::Read;
use std::os::fd::FromRawFd;
use std::time::Duration;

use serde::Serialize;
use sllm_core::{DType, Encoding, TensorView};
use sllm_hip::{
    CompletionState, Context, HipBackend, PreparedRmsNorm, RmsNormDescriptor, RmsNormDispatchInfo,
    RuntimeError,
};

const N: usize = 2560;
const CASE_ROWS: [usize; 6] = [1, 2, 17, 255, 256, 257];
const CASE_SEEDS: [u32; 6] = [9201, 9202, 9217, 9255, 9256, 9257];
const CASE_IDS: [&str; 6] = [
    "g2-r1-n2560",
    "g2-r2-n2560",
    "g2-r17-n2560",
    "g2-r255-n2560",
    "g2-r256-n2560",
    "g2-r257-n2560",
];
const EPSILON: f32 = 1.0e-6;
const TIMEOUT: Duration = Duration::from_secs(30);
const MAX_PROTOCOL_BYTES: usize = 16 * 1024 * 1024;

#[derive(Serialize)]
struct RuntimeResult {
    schema_version: &'static str,
    state: &'static str,
    target: String,
    model_used: bool,
    full_model_used: bool,
    tokenizer_used: bool,
    generation_used: bool,
    selected_backend: &'static str,
    dispatch_count: u32,
    fallback_used: bool,
    cases: Vec<CaseResult>,
}

#[derive(Serialize)]
struct CaseResult {
    order: usize,
    id: &'static str,
    rows: usize,
    n: usize,
    input_seed: u32,
    request_b64: String,
    output_b64: String,
    dispatch: DispatchResult,
}

#[derive(Serialize)]
struct DispatchResult {
    backend: &'static str,
    kernel_id: u32,
    kernel_symbol: String,
    device_symbol: String,
    dispatch_count: u32,
    workgroup_size_x: u32,
    fallback_allowed: bool,
    fallback_used: bool,
}

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
    rows: usize,
    seed: u32,
) -> (Vec<u8>, RmsNormDispatchInfo) {
    let activation_bytes = activation(rows, seed);
    let output_len = activation_bytes.len();
    let activation_buffer = sllm_hip::Buffer::allocate(context, output_len as u64)
        .unwrap_or_else(|error| fail(format!("activation allocation failed: {error}")));
    let output_buffer = sllm_hip::Buffer::allocate(context, output_len as u64)
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
    let mut readback = vec![0_u8; output_len];
    let mut copy = queue
        .copy_to_host(&output_buffer, readback.len() as u64, 0)
        .unwrap_or_else(|error| fail(format!("output download failed: {error}")));
    wait_success(&mut copy, "output download");
    copy.read_into(&mut readback)
        .unwrap_or_else(|error| fail(format!("output readback failed: {error}")));
    if readback.len() != output_len || readback.len() > MAX_PROTOCOL_BYTES {
        fail("G2 output size changed or exceeded the protocol bound");
    }
    (readback, dispatch)
}

fn base64(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut encoded = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let first = chunk[0];
        let second = chunk.get(1).copied();
        let third = chunk.get(2).copied();
        encoded.push(TABLE[(first >> 2) as usize] as char);
        encoded.push(TABLE[((first & 0x03) << 4 | second.unwrap_or(0) >> 4) as usize] as char);
        encoded.push(match second {
            Some(value) => TABLE[((value & 0x0f) << 2 | third.unwrap_or(0) >> 6) as usize] as char,
            None => '=',
        });
        encoded.push(match third {
            Some(value) => TABLE[(value & 0x3f) as usize] as char,
            None => '=',
        });
    }
    encoded
}

fn execution_arguments() -> (String, i32) {
    let mut target = None;
    let mut slice_fd = None;
    let mut args = env::args().skip(1);
    while let Some(current) = args.next() {
        match current.as_str() {
            "--target" => {
                if target.is_some() {
                    fail("duplicate --target argument");
                }
                target = Some(
                    args.next()
                        .unwrap_or_else(|| fail("missing value for --target")),
                );
            }
            "--slice-fd" => {
                if slice_fd.is_some() {
                    fail("duplicate --slice-fd argument");
                }
                let value = args
                    .next()
                    .unwrap_or_else(|| fail("missing value for --slice-fd"));
                let fd = value
                    .parse::<i32>()
                    .unwrap_or_else(|_| fail("--slice-fd must be an integer"));
                if fd < 0 {
                    fail("--slice-fd must be non-negative");
                }
                slice_fd = Some(fd);
            }
            _ => fail(format!("unknown execution argument {current}")),
        }
    }
    (
        target.unwrap_or_else(|| fail("missing required argument --target")),
        slice_fd.unwrap_or_else(|| fail("missing required argument --slice-fd")),
    )
}

fn read_slice_fd(raw_fd: i32) -> Vec<u8> {
    // SAFETY: the runner passes an owned CLOEXEC descriptor to this process;
    // this child takes ownership and closes it exactly once on drop.
    let file = unsafe { File::from_raw_fd(raw_fd) };
    let mut bytes = Vec::with_capacity(N * 2);
    file.take((N * 2 + 1) as u64)
        .read_to_end(&mut bytes)
        .unwrap_or_else(|error| fail(format!("cannot read G2 slice fd: {error}")));
    if bytes.len() != N * 2 {
        fail("G2 slice fd must contain exactly 5120 bytes");
    }
    bytes
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
    let (target, slice_fd) = execution_arguments();
    if target != "gfx1030" && target != "gfx1201" {
        fail("target must be gfx1030 or gfx1201");
    }
    let raw_scale = read_slice_fd(slice_fd);
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
    let mut cases = Vec::with_capacity(CASE_ROWS.len());
    for (order, rows) in CASE_ROWS.into_iter().enumerate() {
        let request = activation(rows, CASE_SEEDS[order]);
        let (output, dispatch) = run_case(
            &backend,
            &context,
            &queue,
            &raw_scale_buffer,
            rows,
            CASE_SEEDS[order],
        );
        cases.push(CaseResult {
            order,
            id: CASE_IDS[order],
            rows,
            n: N,
            input_seed: CASE_SEEDS[order],
            request_b64: base64(&request),
            output_b64: base64(&output),
            dispatch: DispatchResult {
                backend: "hip",
                kernel_id: dispatch.kernel_id,
                kernel_symbol: dispatch.kernel_symbol,
                device_symbol: dispatch.device_symbol,
                dispatch_count: dispatch.dispatch_count,
                workgroup_size_x: dispatch.workgroup_size_x,
                fallback_allowed: dispatch.fallback_allowed,
                fallback_used: dispatch.fallback_used,
            },
        });
    }
    Context::drain_cleanup(8)
        .unwrap_or_else(|error: RuntimeError| fail(format!("cleanup failed: {error}")));
    let result = RuntimeResult {
        schema_version: "rmsnorm-g2-runtime-result-v1",
        state: "PASS",
        target,
        model_used: true,
        full_model_used: false,
        tokenizer_used: false,
        generation_used: false,
        selected_backend: "hip",
        dispatch_count: 6,
        fallback_used: false,
        cases,
    };
    let encoded = serde_json::to_vec(&result)
        .unwrap_or_else(|error| fail(format!("G2 protocol serialization failed: {error}")));
    if encoded.len() > MAX_PROTOCOL_BYTES {
        fail("G2 protocol exceeded the bounded output size");
    }
    println!(
        "{}",
        String::from_utf8(encoded).unwrap_or_else(|_| fail("G2 protocol is not UTF-8"))
    );
}
