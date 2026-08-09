//! Dedicated P0 producer for the public SemanticOpDescriptor -> HIP path.
//!
//! The producer has no host implementation and emits exactly one canonical
//! runtime-result document only after all cases have completed on the selected
//! HIP device.  The host stub therefore exits non-zero without producing JSON.

use std::collections::BTreeMap;
use std::env;
use std::io::Write;
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use sllm_core::{DType, Encoding, SemanticOpKind, TensorView};
use sllm_hip::{CompletionState, Context, HipBackend, RmsNormDescriptor};

const EPSILON: f32 = 1.0e-6;
const WARMUPS: usize = 5;
const MEASUREMENTS: usize = 21;
const TIMEOUT: Duration = Duration::from_secs(30);
const BLOCK: u32 = 256;
const KERNEL_SYMBOL: &str = "rmsnorm.baseline.wave32.v1";
const DEVICE_SYMBOL: &str = "sllm_rmsnorm_baseline_wave32_v1";

const CASES: [(&str, usize, usize, u32, &str); 5] = [
    ("p0-r3-n37", 3, 37, 9301, "synthetic-nonaligned"),
    ("p0-r1-n2560", 1, 2560, 9302, "locked-model-hidden-size"),
    ("p0-r1-n255", 1, 255, 9303, "dispatch-b-minus-1"),
    ("p0-r1-n256", 1, 256, 9304, "dispatch-b"),
    ("p0-r1-n257", 1, 257, 9305, "dispatch-b-plus-1"),
];

fn fail(message: impl AsRef<str>) -> ! {
    eprintln!("sllm-rmsnorm-p0-evidence: FAIL: {}", message.as_ref());
    std::process::exit(1);
}

fn required(arguments: &BTreeMap<String, String>, name: &str) -> String {
    arguments
        .get(name)
        .cloned()
        .unwrap_or_else(|| fail(format!("missing required argument {name}")))
}

fn parse_arguments() -> BTreeMap<String, String> {
    let values: Vec<String> = env::args().skip(1).collect();
    if values.len() % 2 != 0 {
        fail("arguments must be name/value pairs");
    }
    let known = [
        "--target",
        "--case-set-sha256",
        "--warmup",
        "--iterations",
        "--timing-contract",
        "--reviewed-sha",
        "--tested-sha",
        "--workflow-sha",
        "--tree-oid",
        "--artifact-id",
        "--artifact-sha256",
        "--binary-sha256",
        "--binary-sidecar-sha256",
        "--source-set-sha256",
        "--matrix-sha256",
        "--model-lock-sha256",
        "--physical-hip-index",
        "--device-bdf",
        "--device-uuid",
        "--device-product",
    ];
    let mut result = BTreeMap::new();
    for pair in values.chunks_exact(2) {
        if !known.contains(&pair[0].as_str()) {
            fail(format!("unknown argument {}", pair[0]));
        }
        if result.insert(pair[0].clone(), pair[1].clone()).is_some() {
            fail(format!("duplicate argument {}", pair[0]));
        }
    }
    result
}

fn positive_sha(value: &str, label: &str) {
    if value.len() != 64
        || value
            .chars()
            .any(|character| !character.is_ascii_hexdigit())
        || value != value.to_ascii_lowercase()
        || value == "0".repeat(64)
    {
        fail(format!("{label} is not a nonzero lowercase SHA-256"));
    }
}

fn positive_sha40(value: &str, label: &str) {
    if value.len() != 40
        || value
            .chars()
            .any(|character| !character.is_ascii_hexdigit())
        || value != value.to_ascii_lowercase()
        || value == "0".repeat(40)
    {
        fail(format!("{label} is not a nonzero full SHA"));
    }
}

fn canonical_sha(value: &Value) -> String {
    let mut bytes = serde_json::to_vec(value).unwrap_or_else(|error| fail(error.to_string()));
    bytes.push(b'\n');
    let mut digest = Sha256::new();
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}

fn bfloat16(value: f32) -> [u8; 2] {
    let bits = value.to_bits();
    let upper = bits >> 16;
    let lower = bits & 0xffff;
    let rounded = upper + u32::from(lower > 0x8000 || (lower == 0x8000 && upper & 1 != 0));
    (rounded as u16).to_le_bytes()
}

fn activation(rows: usize, n: usize, seed: u32) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(rows * n * 2);
    for row in 0..rows {
        for index in 0..n {
            let lane = ((seed as usize + row * 17 + index * 13) % 257) as f32;
            bytes.extend_from_slice(&bfloat16((lane - 128.0) / 64.0));
        }
    }
    bytes
}

fn wait_success(submission: &mut sllm_hip::RmsNormSubmission, label: &str) {
    match submission.wait(TIMEOUT) {
        Ok(CompletionState::Success) => {}
        Ok(state) => fail(format!("{label} completed with {state:?}")),
        Err(error) => fail(format!("{label} failed: {error}")),
    }
}

fn dispatch_value(
    dispatch: &sllm_hip::RmsNormDispatchInfo,
    iteration: usize,
    expected_id: u64,
    rows: usize,
    n: usize,
) -> Value {
    if dispatch.dispatch_id != expected_id
        || dispatch.dispatch_count != 1
        || dispatch.kernel_id != 1
        || dispatch.workgroup_size_x != BLOCK
        || dispatch.row_count != rows as u64
        || dispatch.normalized_size != n as u64
        || dispatch.backend != sllm_hip_sys::SLLM_BACKEND_HIP
        || dispatch.fallback_allowed
        || dispatch.fallback_used
        || dispatch.kernel_symbol != KERNEL_SYMBOL
        || dispatch.device_symbol != DEVICE_SYMBOL
    {
        fail("RMSNorm dispatch metadata violated the no-fallback public contract");
    }
    json!({
        "iteration": iteration,
        "dispatch_id": dispatch.dispatch_id,
        "dispatch_count": 1,
        "kernel_id": 1,
        "kernel_symbol": KERNEL_SYMBOL,
        "device_symbol": DEVICE_SYMBOL,
        "fallback_used": false,
    })
}

fn median(values: &[u64]) -> u64 {
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    sorted[sorted.len() / 2]
}

fn run_case(
    backend: &HipBackend,
    context: &Context,
    queue: &sllm_hip::Queue,
    raw_scale: &sllm_hip::Buffer,
    expected_dispatch_id: &mut u64,
    order: usize,
    case: (&str, usize, usize, u32, &str),
) -> Value {
    let (case_id, rows, n, seed, classification) = case;
    let activation_bytes = activation(rows, n, seed);
    let activation_buffer = sllm_hip::Buffer::allocate(context, activation_bytes.len() as u64)
        .unwrap_or_else(|error| fail(format!("{case_id} activation allocation failed: {error}")));
    let output_buffer = sllm_hip::Buffer::allocate(context, activation_bytes.len() as u64)
        .unwrap_or_else(|error| fail(format!("{case_id} output allocation failed: {error}")));
    let mut upload = queue
        .copy_to_device(&activation_buffer, &activation_bytes, 0)
        .unwrap_or_else(|error| fail(format!("{case_id} activation upload failed: {error}")));
    match upload.wait(TIMEOUT) {
        Ok(CompletionState::Success) => {}
        Ok(state) => fail(format!(
            "{case_id} activation upload completed with {state:?}"
        )),
        Err(error) => fail(format!("{case_id} activation upload failed: {error}")),
    }

    let activation_view =
        TensorView::new(DType::Bf16, Encoding::Unquantized, &[rows, n], &[n, 1], 0)
            .unwrap_or_else(|error| fail(format!("{case_id} activation view failed: {error}")));
    let scale_view = TensorView::contiguous(DType::Bf16, &[n])
        .unwrap_or_else(|error| fail(format!("{case_id} scale view failed: {error}")));
    let output_view = TensorView::new(DType::Bf16, Encoding::Unquantized, &[rows, n], &[n, 1], 0)
        .unwrap_or_else(|error| fail(format!("{case_id} output view failed: {error}")));
    let descriptor = RmsNormDescriptor::new(
        activation_buffer.binding(activation_view),
        raw_scale.binding(scale_view),
        output_buffer.binding(output_view),
        EPSILON,
    )
    .unwrap_or_else(|error| fail(format!("{case_id} descriptor failed: {error}")));
    if descriptor.semantic().kind() != SemanticOpKind::RmsNorm
        || descriptor.semantic().rms_norm_contract().is_none()
    {
        fail("RMSNorm producer did not retain the canonical semantic descriptor");
    }
    let prepared = backend
        .prepare_rms_norm(context, descriptor)
        .unwrap_or_else(|error| fail(format!("{case_id} prepare failed: {error}")));

    let mut warmup_dispatches = Vec::with_capacity(WARMUPS);
    for iteration in 0..WARMUPS {
        let (mut submission, dispatch) = prepared
            .execute(queue)
            .unwrap_or_else(|error| fail(format!("{case_id} warmup dispatch failed: {error}")));
        let value = dispatch_value(&dispatch, iteration, *expected_dispatch_id, rows, n);
        *expected_dispatch_id += 1;
        wait_success(&mut submission, &format!("{case_id} warmup {iteration}"));
        warmup_dispatches.push(value);
    }

    let mut samples = Vec::with_capacity(MEASUREMENTS);
    let mut kernel_values = Vec::with_capacity(MEASUREMENTS);
    let mut wall_values = Vec::with_capacity(MEASUREMENTS);
    for iteration in 0..MEASUREMENTS {
        let started = Instant::now();
        let (mut submission, dispatch) = prepared.execute(queue).unwrap_or_else(|error| {
            fail(format!("{case_id} measurement dispatch failed: {error}"))
        });
        let dispatch_record = dispatch_value(&dispatch, iteration, *expected_dispatch_id, rows, n);
        *expected_dispatch_id += 1;
        wait_success(
            &mut submission,
            &format!("{case_id} measurement {iteration}"),
        );
        let kernel_ns = submission
            .kernel_elapsed_ns()
            .unwrap_or_else(|error| fail(format!("{case_id} kernel timing failed: {error}")));
        let wall_ns = started.elapsed().as_nanos();
        let wall_ns = u64::try_from(wall_ns)
            .unwrap_or_else(|_| fail(format!("{case_id} wall timing overflowed u64")));
        if kernel_ns == 0 || wall_ns == 0 || wall_ns < kernel_ns {
            fail(format!("{case_id} timing is non-positive or wall < kernel"));
        }
        kernel_values.push(kernel_ns);
        wall_values.push(wall_ns);
        let mut sample = dispatch_record;
        sample["kernel_latency_ns"] = json!(kernel_ns);
        sample["wall_latency_ns"] = json!(wall_ns);
        samples.push(sample);
    }
    let kernel_median = median(&kernel_values);
    let wall_median = median(&wall_values);
    let kernel_deviations: Vec<u64> = kernel_values
        .iter()
        .map(|value| value.abs_diff(kernel_median))
        .collect();
    let wall_deviations: Vec<u64> = wall_values
        .iter()
        .map(|value| value.abs_diff(wall_median))
        .collect();
    json!({
        "order": order,
        "id": case_id,
        "rows": rows,
        "n": n,
        "input_seed": seed,
        "classification": classification,
        "state": "PASS",
        "warmup_dispatches": warmup_dispatches,
        "samples": samples,
        "summary": {
            "kernel_median_ns": kernel_median,
            "kernel_mad_ns": median(&kernel_deviations),
            "wall_median_ns": wall_median,
            "wall_mad_ns": median(&wall_deviations),
            "sample_set_sha256": canonical_sha(&json!(samples)),
        },
    })
}

fn main() {
    let arguments = parse_arguments();
    let target = required(&arguments, "--target");
    match target.as_str() {
        "gfx1030" | "gfx1201" => {}
        _ => fail("target must be gfx1030 or gfx1201"),
    }
    // The runner exposes exactly one physical GPU, so the runtime-visible
    // device is always logical index zero. Retain the independently observed
    // physical routing tuple separately in the result.
    let target_index = 0_u32;
    let physical_hip_index = required(&arguments, "--physical-hip-index")
        .parse::<u32>()
        .unwrap_or_else(|_| fail("physical HIP index must be an unsigned integer"));
    let device_bdf = required(&arguments, "--device-bdf");
    let device_uuid = required(&arguments, "--device-uuid");
    let device_product = required(&arguments, "--device-product");
    if device_bdf.is_empty() || device_uuid.is_empty() || device_product.is_empty() {
        fail("physical device routing tuple must be nonempty");
    }
    if required(&arguments, "--warmup").parse::<usize>().ok() != Some(WARMUPS)
        || required(&arguments, "--iterations").parse::<usize>().ok() != Some(MEASUREMENTS)
        || required(&arguments, "--timing-contract") != "rmsnorm-p0-timing-v1"
    {
        fail("warmup, measurement, or timing contract is noncanonical");
    }
    let expected_case_set = json!([
        {"order": 0, "id": "p0-r3-n37", "rows": 3, "n": 37, "input_seed": 9301, "classification": "synthetic-nonaligned"},
        {"order": 1, "id": "p0-r1-n2560", "rows": 1, "n": 2560, "input_seed": 9302, "classification": "locked-model-hidden-size"},
        {"order": 2, "id": "p0-r1-n255", "rows": 1, "n": 255, "input_seed": 9303, "classification": "dispatch-b-minus-1"},
        {"order": 3, "id": "p0-r1-n256", "rows": 1, "n": 256, "input_seed": 9304, "classification": "dispatch-b"},
        {"order": 4, "id": "p0-r1-n257", "rows": 1, "n": 257, "input_seed": 9305, "classification": "dispatch-b-plus-1"}
    ]);
    let case_set_sha256 = required(&arguments, "--case-set-sha256");
    positive_sha(&case_set_sha256, "case set");
    if canonical_sha(&expected_case_set) != case_set_sha256 {
        fail("case set identity is not canonical");
    }

    let reviewed_sha = required(&arguments, "--reviewed-sha");
    let tested_sha = required(&arguments, "--tested-sha");
    let workflow_sha = required(&arguments, "--workflow-sha");
    let tree_oid = required(&arguments, "--tree-oid");
    for (label, value) in [
        ("reviewed SHA", &reviewed_sha),
        ("tested SHA", &tested_sha),
        ("workflow SHA", &workflow_sha),
        ("tree OID", &tree_oid),
    ] {
        positive_sha40(value, label);
    }
    if reviewed_sha != tested_sha || reviewed_sha != workflow_sha {
        fail("candidate SHA identities differ");
    }
    let artifact_id = required(&arguments, "--artifact-id");
    let artifact_sha256 = required(&arguments, "--artifact-sha256");
    let binary_sha256 = required(&arguments, "--binary-sha256");
    let binary_sidecar_sha256 = required(&arguments, "--binary-sidecar-sha256");
    let source_set_sha256 = required(&arguments, "--source-set-sha256");
    let matrix_sha256 = required(&arguments, "--matrix-sha256");
    let model_lock_sha256 = required(&arguments, "--model-lock-sha256");
    for (label, value) in [
        ("artifact", &artifact_sha256),
        ("binary", &binary_sha256),
        ("binary sidecar", &binary_sidecar_sha256),
        ("source set", &source_set_sha256),
        ("matrix", &matrix_sha256),
        ("model lock", &model_lock_sha256),
    ] {
        positive_sha(value, label);
    }
    if artifact_id != format!("rmsnorm-p0-{target}-{binary_sha256}") {
        fail("artifact identity is not bound to the dedicated binary");
    }

    let backend =
        HipBackend::connect().unwrap_or_else(|error| fail(format!("HIP unavailable: {error}")));
    let context = Context::create(target_index, &target)
        .unwrap_or_else(|error| fail(format!("HIP context unavailable: {error}")));
    let device = Context::query_device(target_index)
        .unwrap_or_else(|error| fail(format!("HIP device query failed: {error}")));
    if device.device_index != target_index
        || device.gcn_arch_name != target
        || device.wavefront_size != 32
        || device.name != device_product
    {
        fail("actual HIP device identity does not match the canonical target");
    }
    let queue = sllm_hip::Queue::create(&context)
        .unwrap_or_else(|error| fail(format!("HIP queue unavailable: {error}")));
    let raw_scale_bytes = vec![0_u8; 2560 * 2];
    let raw_scale = sllm_hip::Buffer::allocate(&context, raw_scale_bytes.len() as u64)
        .unwrap_or_else(|error| fail(format!("raw scale allocation failed: {error}")));
    let mut scale_upload = queue
        .copy_to_device(&raw_scale, &raw_scale_bytes, 0)
        .unwrap_or_else(|error| fail(format!("raw scale upload failed: {error}")));
    match scale_upload.wait(TIMEOUT) {
        Ok(CompletionState::Success) => {}
        Ok(state) => fail(format!("raw scale upload completed with {state:?}")),
        Err(error) => fail(format!("raw scale upload failed: {error}")),
    }

    let mut expected_dispatch_id = 1_u64;
    let cases: Vec<Value> = CASES
        .into_iter()
        .enumerate()
        .map(|(order, case)| {
            run_case(
                &backend,
                &context,
                &queue,
                &raw_scale,
                &mut expected_dispatch_id,
                order,
                case,
            )
        })
        .collect();
    if expected_dispatch_id != 131 {
        fail("P0 dispatch count is not exactly 130");
    }
    let cases_value = Value::Array(cases);
    let candidate = json!({
        "reviewed_sha": reviewed_sha,
        "tested_sha": tested_sha,
        "workflow_sha": workflow_sha,
        "git_tree_oid": tree_oid,
        "worktree_clean": true,
        "revision_input": "full-sha",
    });
    let device_identity = json!({
        "bdf": device_bdf,
        "uuid": device_uuid,
        "product": device_product,
        "target": target,
        "physical_hip_index": physical_hip_index,
        "logical_device_index": target_index,
    });
    let result = json!({
        "schema_version": "rmsnorm-p0-runtime-result-v1",
        "state": "PASS",
        "row_id": format!("rmsnorm-p0-{target}"),
        "target": target,
        "candidate": candidate,
        "artifact": {"artifact_id": artifact_id, "artifact_sha256": artifact_sha256, "binary_sha256": binary_sha256, "binary_sidecar_sha256": binary_sidecar_sha256, "source_set_sha256": source_set_sha256, "binary_role": "dedicated-p0-public-rmsnorm-producer"},
        "matrix": {"path": "ci/matrix/rmsnorm-p0-v1.json", "sha256": matrix_sha256},
        "case_set_sha256": case_set_sha256,
        "model_lock": {"path": "docs/models/locks/qwen3.5-4b-bf16.json", "sha256": model_lock_sha256, "fingerprint": "sha256:32265444b7cdd2a00e4e4e3e6aa8375a05acf6cddfcb9ffc348f54f67a7cd935", "resolved_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", "used": false},
        "source_set_sha256": source_set_sha256,
        "dtype": {"activation": "BF16", "weight": "BF16", "output": "BF16", "accumulation": "F32", "scale_mode": "offset-one", "epsilon": "1e-6"},
        "scope": {"selected_backend": "hip", "gpu_execution": true, "public_rmsnorm_path": true, "semantic_op_used": true, "model_used": false, "fallback_allowed": false, "fallback_used": false, "cpu_fallback_used": false},
        "device": device_identity,
        "timing": {"timing_contract": "rmsnorm-p0-timing-v1", "unit": "ns", "kernel_source": "hip-event-elapsed-time", "wall_source": "steady-clock-monotonic", "warmup_iterations": 5, "measurement_iterations": 21, "location": "median", "robust_spread": "median-absolute-deviation"},
        "dispatch": {"backend": "hip", "kernel_id": 1, "kernel_symbol": KERNEL_SYMBOL, "device_symbol": DEVICE_SYMBOL, "workgroup_size_x": BLOCK, "dispatch_count": 130, "fallback_allowed": false, "fallback_used": false},
        "cases": cases_value,
        "measurement_sha256": canonical_sha(&cases_value),
    });
    let bytes = serde_json::to_vec(&result).unwrap_or_else(|error| fail(error.to_string()));
    let mut stdout = std::io::stdout().lock();
    stdout
        .write_all(&bytes)
        .unwrap_or_else(|error| fail(error.to_string()));
    stdout
        .write_all(b"\n")
        .unwrap_or_else(|error| fail(error.to_string()));
}
