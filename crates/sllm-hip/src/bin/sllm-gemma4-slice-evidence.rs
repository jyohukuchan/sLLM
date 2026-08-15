//! Focused real-weight Gemma 4 final-RMSNorm slice evidence.

use std::env;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;
use sllm_core::{
    AccessMode, Backend, BoundSemanticOp, DType, ExecutionSessionRequest, ExecutionState,
    RmsNormScaleMode, SemanticOpDescriptor, SemanticOpKind, SplitHalfRotaryContract, TensorDType,
    TensorView, VerifiedCache, WindowedCausalAttentionContract, parse_gemma4_model_lock,
};
use sllm_hip::HipBackend;

const ROWS: usize = 3;
const WIDTH: usize = 3_840;
const EPSILON: f32 = 1.0e-6;
const WAIT_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Debug)]
struct Config {
    lock: PathBuf,
    cache: PathBuf,
    device_index: u32,
    target: String,
}

#[derive(Serialize)]
struct Report {
    schema_version: &'static str,
    state: &'static str,
    model: &'static str,
    resolved_revision: String,
    lock_fingerprint: String,
    tensor: String,
    tensor_bytes: usize,
    rows: usize,
    width: usize,
    target: String,
    device_index: u32,
    fallback_allowed: bool,
    fallback_used: bool,
    operations: usize,
    embedding_tensor: String,
    embedding_exact_match: bool,
    embedding_kernel_symbol: String,
    embedding_device_symbol: String,
    rmsnorm_kernel_symbol: String,
    rmsnorm_device_symbol: String,
    mlp_gate_kernel_symbol: String,
    mlp_up_kernel_symbol: String,
    mlp_activation_kernel_symbol: String,
    mlp_down_kernel_symbol: String,
    tied_logits_kernel_symbol: String,
    small_graph_max_abs: f32,
    qkv_projection_max_abs: f32,
    rotary_kernel_symbol: String,
    rotary_max_abs: f32,
    attention_kernel_symbol: String,
    attention_max_abs: f32,
    max_abs: f32,
    max_scaled_rel: f32,
    cleanup_retryable: usize,
    cleanup_durable: usize,
}

fn parse_config() -> Result<Config, String> {
    let mut lock = None;
    let mut cache = None;
    let mut device_index = None;
    let mut target = None;
    let mut arguments = env::args().skip(1);
    while let Some(argument) = arguments.next() {
        let value = arguments
            .next()
            .ok_or_else(|| format!("{argument} requires a value"))?;
        match argument.as_str() {
            "--lock" if lock.is_none() => lock = Some(PathBuf::from(value)),
            "--cache" if cache.is_none() => cache = Some(PathBuf::from(value)),
            "--device-index" if device_index.is_none() => {
                device_index = Some(
                    value
                        .parse::<u32>()
                        .map_err(|_| "--device-index must be a u32".to_owned())?,
                );
            }
            "--target" if target.is_none() && matches!(value.as_str(), "gfx1030" | "gfx1201") => {
                target = Some(value)
            }
            "--target" => return Err("--target must be exactly gfx1030 or gfx1201".to_owned()),
            _ => return Err(format!("duplicate or unknown argument: {argument}")),
        }
    }
    Ok(Config {
        lock: lock.ok_or_else(|| "missing --lock".to_owned())?,
        cache: cache.ok_or_else(|| "missing --cache".to_owned())?,
        device_index: device_index.ok_or_else(|| "missing --device-index".to_owned())?,
        target: target.ok_or_else(|| "missing --target".to_owned())?,
    })
}

fn bf16_to_f32(value: u16) -> f32 {
    f32::from_bits(u32::from(value) << 16)
}

fn f32_to_bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    if bits & 0x7f80_0000 == 0x7f80_0000 {
        if bits & 0x007f_ffff != 0 {
            return ((bits >> 16) as u16 & 0x803f) | 0x7fc0;
        }
        return (bits >> 16) as u16;
    }
    let upper = bits >> 16;
    let lower = bits & 0xffff;
    (upper + u32::from(lower > 0x8000 || (lower == 0x8000 && upper & 1 != 0))) as u16
}

fn words_to_bytes(words: &[u16]) -> Arc<[u8]> {
    words
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect::<Vec<_>>()
        .into()
}

fn bytes_to_words(bytes: &[u8]) -> Result<Vec<u16>, String> {
    if bytes.len() % 2 != 0 {
        return Err("BF16 payload has an odd byte length".to_owned());
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect())
}

fn make_activation() -> Vec<u16> {
    (0..ROWS * WIDTH)
        .map(|index| {
            let mixed = (index * 37 + 11) % 509;
            f32_to_bf16_rne((mixed as f32 - 254.0) / 64.0)
        })
        .collect()
}

fn reference(activation: &[u16], scale: &[u16]) -> Vec<u16> {
    let mut output = vec![0_u16; activation.len()];
    for row in 0..ROWS {
        let base = row * WIDTH;
        let sum = activation[base..base + WIDTH]
            .iter()
            .map(|value| {
                let value = bf16_to_f32(*value);
                value * value
            })
            .sum::<f32>();
        let inverse = (sum / WIDTH as f32 + EPSILON).sqrt().recip();
        for column in 0..WIDTH {
            output[base + column] = f32_to_bf16_rne(
                bf16_to_f32(activation[base + column]) * inverse * bf16_to_f32(scale[column]),
            );
        }
    }
    output
}

fn wait_success(
    state: Result<ExecutionState, sllm_core::ExecutionError>,
    label: &str,
) -> Result<(), String> {
    match state.map_err(|error| format!("{label}: {error}"))? {
        ExecutionState::Success => Ok(()),
        ExecutionState::Pending => Err(format!("{label} remained pending")),
        ExecutionState::Failure => Err(format!("{label} reported failure")),
    }
}

fn execute_embedding_slice(
    session: &sllm_core::ExecutionSession,
    queue: &sllm_core::ExecutionQueue,
    cache: &VerifiedCache,
    target: &str,
) -> Result<(sllm_core::DispatchEvidence, bool), String> {
    const TENSOR: &str = "model.language_model.embed_tokens.weight";
    const VOCAB: usize = 3;
    const IDS: [i32; 3] = [2, 0, 2];
    let row_bytes = WIDTH * 2;
    let mut weight_bytes = Vec::with_capacity(VOCAB * row_bytes);
    for row in 0..VOCAB {
        weight_bytes.extend(
            cache
                .read_tensor_range(TENSOR, (row * row_bytes) as u64, row_bytes)
                .map_err(|error| format!("cannot read verified embedding row: {error}"))?,
        );
    }
    let id_bytes: Vec<u8> = IDS.iter().flat_map(|id| id.to_le_bytes()).collect();
    let expected: Vec<u8> = IDS
        .iter()
        .flat_map(|id| {
            let base = *id as usize * row_bytes;
            weight_bytes[base..base + row_bytes].iter().copied()
        })
        .collect();
    let weight_buffer = session
        .allocate(weight_bytes.len() as u64)
        .map_err(|error| format!("embedding weight allocation failed: {error}"))?;
    let id_buffer = session
        .allocate(id_bytes.len() as u64)
        .map_err(|error| format!("embedding ID allocation failed: {error}"))?;
    let output_buffer = session
        .allocate(expected.len() as u64)
        .map_err(|error| format!("embedding output allocation failed: {error}"))?;
    for (label, buffer, bytes) in [
        ("embedding weight", &weight_buffer, weight_bytes.as_slice()),
        ("embedding IDs", &id_buffer, id_bytes.as_slice()),
    ] {
        let mut upload = session
            .upload(
                queue,
                buffer
                    .range(0, bytes.len() as u64)
                    .map_err(|error| error.to_string())?,
                Arc::from(bytes),
            )
            .map_err(|error| format!("{label} upload failed: {error}"))?;
        wait_success(upload.wait(WAIT_TIMEOUT), label)?;
    }
    let weight_view =
        TensorView::contiguous(DType::Bf16, &[VOCAB, WIDTH]).map_err(|error| error.to_string())?;
    let id_view =
        TensorView::contiguous(DType::I32, &[IDS.len()]).map_err(|error| error.to_string())?;
    let output_view = TensorView::contiguous(DType::Bf16, &[IDS.len(), WIDTH])
        .map_err(|error| error.to_string())?;
    let descriptor = Arc::new(
        SemanticOpDescriptor::new(
            SemanticOpKind::Embedding,
            vec![weight_view.clone(), id_view.clone()],
            vec![output_view.clone()],
        )
        .map_err(|error| error.to_string())?,
    );
    let operation = Arc::new(
        BoundSemanticOp::new(
            descriptor,
            vec![
                session
                    .bind(&weight_buffer, weight_view, AccessMode::Read)
                    .map_err(|error| error.to_string())?,
                session
                    .bind(&id_buffer, id_view, AccessMode::Read)
                    .map_err(|error| error.to_string())?,
            ],
            vec![
                session
                    .bind(&output_buffer, output_view, AccessMode::Write)
                    .map_err(|error| error.to_string())?,
            ],
        )
        .map_err(|error| error.to_string())?,
    );
    let prepared = session
        .prepare(operation)
        .map_err(|error| error.to_string())?;
    let mut submission = session
        .submit(&prepared, queue)
        .map_err(|error| error.to_string())?;
    wait_success(submission.wait(WAIT_TIMEOUT), "embedding execution")?;
    let dispatch = submission.dispatch().clone();
    if dispatch.dispatch_count != 1
        || dispatch.target != target
        || dispatch.fallback_allowed
        || dispatch.fallback_used
        || dispatch.kernel_symbol != "embedding.gather.bf16_i32.v1"
        || dispatch.device_symbol != "sllm_embedding_gather_bf16_i32_v1"
    {
        return Err("embedding dispatch evidence is not exact/no-fallback".to_owned());
    }
    let mut readback = submission
        .start_output_readback(0)
        .map_err(|error| error.to_string())?;
    wait_success(readback.wait(WAIT_TIMEOUT), "embedding output readback")?;
    let mut observed = vec![0_u8; expected.len()];
    if readback
        .read_into(&mut observed)
        .map_err(|error| error.to_string())?
        != observed.len() as u64
    {
        return Err("embedding output readback length differs".to_owned());
    }
    let exact = observed == expected;
    if !exact {
        return Err("real-weight embedding gather differs bit-for-bit".to_owned());
    }
    drop(readback);
    drop(submission);
    drop(prepared);
    drop(output_buffer);
    drop(id_buffer);
    drop(weight_buffer);
    Ok((dispatch, exact))
}

struct MatmulSliceResult {
    output: Vec<u16>,
    dispatch: sllm_core::DispatchEvidence,
    max_abs: f32,
}

struct MatmulSliceRequest<'a> {
    tensor_name: &'a str,
    activation: &'a [u16],
    m: usize,
    k: usize,
    n: usize,
    target: &'a str,
}

fn execute_matmul_slice(
    session: &sllm_core::ExecutionSession,
    queue: &sllm_core::ExecutionQueue,
    cache: &VerifiedCache,
    request: MatmulSliceRequest<'_>,
) -> Result<MatmulSliceResult, String> {
    let MatmulSliceRequest {
        tensor_name,
        activation,
        m,
        k,
        n,
        target,
    } = request;
    if activation.len() != m * k {
        return Err("matmul slice activation shape differs".to_owned());
    }
    let tensor = cache
        .tensor(tensor_name)
        .ok_or_else(|| format!("verified tensor is missing: {tensor_name}"))?;
    if tensor.dtype != TensorDType::Bf16 || tensor.shape.len() != 2 {
        return Err(format!("matmul tensor is not rank-2 BF16: {tensor_name}"));
    }
    let source_n = usize::try_from(tensor.shape[0])
        .map_err(|_| "matmul source N does not fit usize".to_owned())?;
    let source_k = usize::try_from(tensor.shape[1])
        .map_err(|_| "matmul source K does not fit usize".to_owned())?;
    if n > source_n || k > source_k {
        return Err(format!(
            "matmul slice exceeds verified tensor: {tensor_name}"
        ));
    }
    let mut weight_bytes = Vec::with_capacity(n * k * 2);
    for row in 0..n {
        weight_bytes.extend(
            cache
                .read_tensor_range(tensor_name, (row * source_k * 2) as u64, k * 2)
                .map_err(|error| format!("cannot read verified matmul row: {error}"))?,
        );
    }
    let weight = bytes_to_words(&weight_bytes)?;
    let mut oracle = vec![0_u16; m * n];
    for row in 0..m {
        for column in 0..n {
            let mut sum = 0.0_f32;
            for inner in 0..k {
                sum += bf16_to_f32(activation[row * k + inner])
                    * bf16_to_f32(weight[column * k + inner]);
            }
            oracle[row * n + column] = f32_to_bf16_rne(sum);
        }
    }
    let activation_bytes = words_to_bytes(activation);
    let output_bytes = m * n * 2;
    let activation_buffer = session
        .allocate(activation_bytes.len() as u64)
        .map_err(|error| format!("matmul activation allocation failed: {error}"))?;
    let weight_buffer = session
        .allocate(weight_bytes.len() as u64)
        .map_err(|error| format!("matmul weight allocation failed: {error}"))?;
    let output_buffer = session
        .allocate(output_bytes as u64)
        .map_err(|error| format!("matmul output allocation failed: {error}"))?;
    for (label, buffer, bytes) in [
        (
            "matmul activation",
            &activation_buffer,
            activation_bytes.as_ref(),
        ),
        ("matmul weight", &weight_buffer, weight_bytes.as_slice()),
    ] {
        let mut upload = session
            .upload(
                queue,
                buffer
                    .range(0, bytes.len() as u64)
                    .map_err(|error| error.to_string())?,
                Arc::from(bytes),
            )
            .map_err(|error| format!("{label} upload failed: {error}"))?;
        wait_success(upload.wait(WAIT_TIMEOUT), label)?;
    }
    let activation_view =
        TensorView::contiguous(DType::Bf16, &[m, k]).map_err(|error| error.to_string())?;
    let weight_view =
        TensorView::contiguous(DType::Bf16, &[n, k]).map_err(|error| error.to_string())?;
    let output_view =
        TensorView::contiguous(DType::Bf16, &[m, n]).map_err(|error| error.to_string())?;
    let descriptor = Arc::new(
        SemanticOpDescriptor::new(
            SemanticOpKind::Matmul,
            vec![activation_view.clone(), weight_view.clone()],
            vec![output_view.clone()],
        )
        .map_err(|error| error.to_string())?,
    );
    let operation = Arc::new(
        BoundSemanticOp::new(
            descriptor,
            vec![
                session
                    .bind(&activation_buffer, activation_view, AccessMode::Read)
                    .map_err(|error| error.to_string())?,
                session
                    .bind(&weight_buffer, weight_view, AccessMode::Read)
                    .map_err(|error| error.to_string())?,
            ],
            vec![
                session
                    .bind(&output_buffer, output_view, AccessMode::Write)
                    .map_err(|error| error.to_string())?,
            ],
        )
        .map_err(|error| error.to_string())?,
    );
    let prepared = session
        .prepare(operation)
        .map_err(|error| error.to_string())?;
    let mut submission = session
        .submit(&prepared, queue)
        .map_err(|error| error.to_string())?;
    wait_success(submission.wait(WAIT_TIMEOUT), "matmul slice execution")?;
    let dispatch = submission.dispatch().clone();
    if dispatch.dispatch_count != 1
        || dispatch.target != target
        || dispatch.fallback_allowed
        || dispatch.fallback_used
        || !dispatch.kernel_symbol.starts_with("matmul.")
    {
        return Err("matmul slice dispatch is not exact/no-fallback".to_owned());
    }
    let mut readback = submission
        .start_output_readback(0)
        .map_err(|error| error.to_string())?;
    wait_success(readback.wait(WAIT_TIMEOUT), "matmul slice readback")?;
    let mut observed_bytes = vec![0_u8; output_bytes];
    if readback
        .read_into(&mut observed_bytes)
        .map_err(|error| error.to_string())?
        != output_bytes as u64
    {
        return Err("matmul slice readback length differs".to_owned());
    }
    let observed = bytes_to_words(&observed_bytes)?;
    let mut max_abs = 0.0_f32;
    for (actual, expected) in observed.iter().zip(&oracle) {
        let actual = bf16_to_f32(*actual);
        let expected = bf16_to_f32(*expected);
        let absolute = (actual - expected).abs();
        max_abs = max_abs.max(absolute);
        if !actual.is_finite() || absolute > 0.03125 + 0.03125 * expected.abs() {
            return Err(format!(
                "real-weight matmul slice differs: {tensor_name} actual={actual} expected={expected}"
            ));
        }
    }
    drop(readback);
    drop(submission);
    drop(prepared);
    drop(output_buffer);
    drop(weight_buffer);
    drop(activation_buffer);
    Ok(MatmulSliceResult {
        output: observed,
        dispatch,
        max_abs,
    })
}

fn execute_gelu_tanh_mul(
    session: &sllm_core::ExecutionSession,
    queue: &sllm_core::ExecutionQueue,
    gate: &[u16],
    up: &[u16],
    target: &str,
) -> Result<(Vec<u16>, sllm_core::DispatchEvidence), String> {
    if gate.len() != up.len() || gate.is_empty() {
        return Err("GELU-tanh slice inputs differ".to_owned());
    }
    let oracle: Vec<u16> = gate
        .iter()
        .zip(up)
        .map(|(gate, up)| {
            let value = bf16_to_f32(*gate);
            let inner = 0.797_884_6_f32 * (value + 0.044_715_f32 * value * value * value);
            let gelu = f32_to_bf16_rne(0.5 * value * (1.0 + inner.tanh()));
            f32_to_bf16_rne(bf16_to_f32(gelu) * bf16_to_f32(*up))
        })
        .collect();
    let gate_bytes = words_to_bytes(gate);
    let up_bytes = words_to_bytes(up);
    let buffers = [
        session.allocate(gate_bytes.len() as u64),
        session.allocate(up_bytes.len() as u64),
        session.allocate(gate_bytes.len() as u64),
    ]
    .into_iter()
    .collect::<Result<Vec<_>, _>>()
    .map_err(|error| error.to_string())?;
    for (label, buffer, bytes) in [
        ("GELU gate", &buffers[0], Arc::clone(&gate_bytes)),
        ("GELU up", &buffers[1], Arc::clone(&up_bytes)),
    ] {
        let mut upload = session
            .upload(
                queue,
                buffer
                    .range(0, bytes.len() as u64)
                    .map_err(|error| error.to_string())?,
                bytes,
            )
            .map_err(|error| format!("{label} upload failed: {error}"))?;
        wait_success(upload.wait(WAIT_TIMEOUT), label)?;
    }
    let view = TensorView::contiguous(DType::Bf16, &[ROWS, gate.len() / ROWS])
        .map_err(|error| error.to_string())?;
    let descriptor = Arc::new(
        SemanticOpDescriptor::new(
            SemanticOpKind::GeluTanhMul,
            vec![view.clone(), view.clone()],
            vec![view.clone()],
        )
        .map_err(|error| error.to_string())?,
    );
    let operation = Arc::new(
        BoundSemanticOp::new(
            descriptor,
            vec![
                session
                    .bind(&buffers[0], view.clone(), AccessMode::Read)
                    .map_err(|error| error.to_string())?,
                session
                    .bind(&buffers[1], view.clone(), AccessMode::Read)
                    .map_err(|error| error.to_string())?,
            ],
            vec![
                session
                    .bind(&buffers[2], view, AccessMode::Write)
                    .map_err(|error| error.to_string())?,
            ],
        )
        .map_err(|error| error.to_string())?,
    );
    let prepared = session
        .prepare(operation)
        .map_err(|error| error.to_string())?;
    let mut submission = session
        .submit(&prepared, queue)
        .map_err(|error| error.to_string())?;
    wait_success(submission.wait(WAIT_TIMEOUT), "GELU-tanh execution")?;
    let dispatch = submission.dispatch().clone();
    if dispatch.target != target
        || dispatch.fallback_allowed
        || dispatch.fallback_used
        || dispatch.kernel_symbol != "elementwise.gelu_tanh_mul.bf16_fp32.v1"
    {
        return Err("GELU-tanh dispatch is not exact/no-fallback".to_owned());
    }
    let mut readback = submission
        .start_output_readback(0)
        .map_err(|error| error.to_string())?;
    wait_success(readback.wait(WAIT_TIMEOUT), "GELU-tanh readback")?;
    let mut bytes = vec![0_u8; gate_bytes.len()];
    if readback
        .read_into(&mut bytes)
        .map_err(|error| error.to_string())?
        != bytes.len() as u64
    {
        return Err("GELU-tanh readback length differs".to_owned());
    }
    let output = bytes_to_words(&bytes)?;
    if output != oracle {
        return Err("GELU-tanh slice differs bit-for-bit".to_owned());
    }
    drop(readback);
    drop(submission);
    drop(prepared);
    drop(buffers);
    Ok((output, dispatch))
}

struct WordOpResult {
    output: Vec<u16>,
    dispatch: sllm_core::DispatchEvidence,
    max_abs: f32,
}

fn execute_direct_rmsnorm_words(
    session: &sllm_core::ExecutionSession,
    queue: &sllm_core::ExecutionQueue,
    input: &[u16],
    scale: &[u16],
    rows: usize,
    width: usize,
    target: &str,
) -> Result<WordOpResult, String> {
    if input.len() != rows * width || scale.len() != width {
        return Err("compact RMSNorm shape differs".to_owned());
    }
    let oracle = reference_rmsnorm(input, scale, rows, width);
    let input_bytes = words_to_bytes(input);
    let scale_bytes = words_to_bytes(scale);
    let buffers = [
        session.allocate(input_bytes.len() as u64),
        session.allocate(scale_bytes.len() as u64),
        session.allocate(input_bytes.len() as u64),
    ]
    .into_iter()
    .collect::<Result<Vec<_>, _>>()
    .map_err(|error| error.to_string())?;
    for (label, buffer, bytes) in [
        (
            "compact RMSNorm input",
            &buffers[0],
            Arc::clone(&input_bytes),
        ),
        (
            "compact RMSNorm scale",
            &buffers[1],
            Arc::clone(&scale_bytes),
        ),
    ] {
        let mut upload = session
            .upload(
                queue,
                buffer
                    .range(0, bytes.len() as u64)
                    .map_err(|error| error.to_string())?,
                bytes,
            )
            .map_err(|error| format!("{label} upload failed: {error}"))?;
        wait_success(upload.wait(WAIT_TIMEOUT), label)?;
    }
    let input_view =
        TensorView::contiguous(DType::Bf16, &[rows, width]).map_err(|error| error.to_string())?;
    let scale_view =
        TensorView::contiguous(DType::Bf16, &[width]).map_err(|error| error.to_string())?;
    let descriptor = Arc::new(
        SemanticOpDescriptor::new_rms_norm(
            vec![input_view.clone(), scale_view.clone()],
            vec![input_view.clone()],
            EPSILON,
            RmsNormScaleMode::Direct,
        )
        .map_err(|error| error.to_string())?,
    );
    let operation = Arc::new(
        BoundSemanticOp::new(
            descriptor,
            vec![
                session
                    .bind(&buffers[0], input_view.clone(), AccessMode::Read)
                    .map_err(|error| error.to_string())?,
                session
                    .bind(&buffers[1], scale_view, AccessMode::Read)
                    .map_err(|error| error.to_string())?,
            ],
            vec![
                session
                    .bind(&buffers[2], input_view, AccessMode::Write)
                    .map_err(|error| error.to_string())?,
            ],
        )
        .map_err(|error| error.to_string())?,
    );
    let prepared = session
        .prepare(operation)
        .map_err(|error| error.to_string())?;
    let mut submission = session
        .submit(&prepared, queue)
        .map_err(|error| error.to_string())?;
    wait_success(submission.wait(WAIT_TIMEOUT), "compact RMSNorm")?;
    let dispatch = submission.dispatch().clone();
    if dispatch.target != target
        || dispatch.fallback_allowed
        || dispatch.fallback_used
        || dispatch.kernel_symbol != "rmsnorm.baseline.wave32.v1"
    {
        return Err("compact RMSNorm dispatch is not exact/no-fallback".to_owned());
    }
    let mut readback = submission
        .start_output_readback(0)
        .map_err(|error| error.to_string())?;
    wait_success(readback.wait(WAIT_TIMEOUT), "compact RMSNorm readback")?;
    let mut bytes = vec![0_u8; input_bytes.len()];
    if readback
        .read_into(&mut bytes)
        .map_err(|error| error.to_string())?
        != bytes.len() as u64
    {
        return Err("compact RMSNorm readback length differs".to_owned());
    }
    let output = bytes_to_words(&bytes)?;
    let mut max_abs = 0.0_f32;
    for (actual, expected) in output.iter().zip(&oracle) {
        let actual = bf16_to_f32(*actual);
        let expected = bf16_to_f32(*expected);
        let absolute = (actual - expected).abs();
        max_abs = max_abs.max(absolute);
        if !actual.is_finite() || absolute > 0.03125 + 0.03125 * expected.abs() {
            return Err("compact RMSNorm differs from oracle".to_owned());
        }
    }
    drop(readback);
    drop(submission);
    drop(prepared);
    drop(buffers);
    Ok(WordOpResult {
        output,
        dispatch,
        max_abs,
    })
}

fn reference_rmsnorm(input: &[u16], scale: &[u16], rows: usize, width: usize) -> Vec<u16> {
    let mut output = vec![0_u16; input.len()];
    for row in 0..rows {
        let base = row * width;
        let sum = input[base..base + width]
            .iter()
            .map(|value| {
                let value = bf16_to_f32(*value);
                value * value
            })
            .sum::<f32>();
        let inverse = (sum / width as f32 + EPSILON).sqrt().recip();
        for column in 0..width {
            output[base + column] = f32_to_bf16_rne(
                bf16_to_f32(input[base + column]) * inverse * bf16_to_f32(scale[column]),
            );
        }
    }
    output
}

struct RotarySliceResult {
    query: Vec<u16>,
    key: Vec<u16>,
    dispatch: sllm_core::DispatchEvidence,
    max_abs: f32,
}

fn rotary_reference(input: &[u16], heads: usize, positions: &[i32]) -> Vec<u16> {
    const HEAD_DIM: usize = 6;
    const ROTARY_DIM: usize = 4;
    let mut output = input.to_vec();
    for (token, position) in positions.iter().enumerate() {
        for head in 0..heads {
            let base = (token * heads + head) * HEAD_DIM;
            for pair in 0..ROTARY_DIM / 2 {
                let exponent = -2.0 * pair as f32 / HEAD_DIM as f32;
                let angle = *position as f32 * 10_000.0_f32.powf(exponent);
                let left = bf16_to_f32(input[base + pair]);
                let right = bf16_to_f32(input[base + HEAD_DIM / 2 + pair]);
                output[base + pair] = f32_to_bf16_rne(left * angle.cos() - right * angle.sin());
                output[base + HEAD_DIM / 2 + pair] =
                    f32_to_bf16_rne(right * angle.cos() + left * angle.sin());
            }
        }
    }
    output
}

fn execute_rotary_slice(
    session: &sllm_core::ExecutionSession,
    queue: &sllm_core::ExecutionQueue,
    query: &[u16],
    key: &[u16],
    target: &str,
) -> Result<RotarySliceResult, String> {
    const M: usize = 3;
    const Q_HEADS: usize = 3;
    const KV_HEADS: usize = 1;
    const HEAD_DIM: usize = 6;
    if query.len() != M * Q_HEADS * HEAD_DIM || key.len() != M * KV_HEADS * HEAD_DIM {
        return Err("rotary real-weight projection shape differs".to_owned());
    }
    let positions = [0_i32, 1, 2];
    let position_bytes: Arc<[u8]> = positions
        .iter()
        .flat_map(|position| position.to_le_bytes())
        .collect::<Vec<_>>()
        .into();
    let query_bytes = words_to_bytes(query);
    let key_bytes = words_to_bytes(key);
    let buffers = [
        session.allocate(query_bytes.len() as u64),
        session.allocate(key_bytes.len() as u64),
        session.allocate(position_bytes.len() as u64),
        session.allocate(query_bytes.len() as u64),
        session.allocate(key_bytes.len() as u64),
    ]
    .into_iter()
    .collect::<Result<Vec<_>, _>>()
    .map_err(|error| error.to_string())?;
    for (label, buffer, bytes) in [
        ("rotary query", &buffers[0], Arc::clone(&query_bytes)),
        ("rotary key", &buffers[1], Arc::clone(&key_bytes)),
        ("rotary positions", &buffers[2], Arc::clone(&position_bytes)),
    ] {
        let mut upload = session
            .upload(
                queue,
                buffer
                    .range(0, bytes.len() as u64)
                    .map_err(|error| error.to_string())?,
                bytes,
            )
            .map_err(|error| format!("{label} upload failed: {error}"))?;
        wait_success(upload.wait(WAIT_TIMEOUT), label)?;
    }
    let query_view = TensorView::contiguous(DType::Bf16, &[M, Q_HEADS, HEAD_DIM])
        .map_err(|error| error.to_string())?;
    let key_view = TensorView::contiguous(DType::Bf16, &[M, KV_HEADS, HEAD_DIM])
        .map_err(|error| error.to_string())?;
    let position_view =
        TensorView::contiguous(DType::I32, &[M]).map_err(|error| error.to_string())?;
    let contract = SplitHalfRotaryContract::new(
        Q_HEADS as u32,
        KV_HEADS as u32,
        HEAD_DIM as u32,
        4,
        10_000.0,
        0,
        M as u64,
        262_144,
    )
    .map_err(|error| error.to_string())?;
    let descriptor = Arc::new(
        SemanticOpDescriptor::new_rotary(
            vec![query_view.clone(), key_view.clone(), position_view.clone()],
            vec![query_view.clone(), key_view.clone()],
            contract,
        )
        .map_err(|error| error.to_string())?,
    );
    let operation = Arc::new(
        BoundSemanticOp::new(
            descriptor,
            vec![
                session
                    .bind(&buffers[0], query_view.clone(), AccessMode::Read)
                    .map_err(|error| error.to_string())?,
                session
                    .bind(&buffers[1], key_view.clone(), AccessMode::Read)
                    .map_err(|error| error.to_string())?,
                session
                    .bind(&buffers[2], position_view, AccessMode::Read)
                    .map_err(|error| error.to_string())?,
            ],
            vec![
                session
                    .bind(&buffers[3], query_view, AccessMode::Write)
                    .map_err(|error| error.to_string())?,
                session
                    .bind(&buffers[4], key_view, AccessMode::Write)
                    .map_err(|error| error.to_string())?,
            ],
        )
        .map_err(|error| error.to_string())?,
    );
    let prepared = session
        .prepare(operation)
        .map_err(|error| error.to_string())?;
    let mut submission = session
        .submit(&prepared, queue)
        .map_err(|error| error.to_string())?;
    wait_success(submission.wait(WAIT_TIMEOUT), "rotary slice")?;
    let dispatch = submission.dispatch().clone();
    if dispatch.target != target
        || dispatch.fallback_allowed
        || dispatch.fallback_used
        || dispatch.kernel_symbol != "rotary.split_half.bf16_fp32.v1"
    {
        return Err("rotary slice dispatch is not exact/no-fallback".to_owned());
    }
    let mut outputs = Vec::with_capacity(2);
    for (index, length) in [query_bytes.len(), key_bytes.len()].into_iter().enumerate() {
        let mut readback = submission
            .start_output_readback(index)
            .map_err(|error| error.to_string())?;
        wait_success(readback.wait(WAIT_TIMEOUT), "rotary slice readback")?;
        let mut bytes = vec![0_u8; length];
        if readback
            .read_into(&mut bytes)
            .map_err(|error| error.to_string())?
            != length as u64
        {
            return Err("rotary slice readback length differs".to_owned());
        }
        outputs.push(bytes_to_words(&bytes)?);
    }
    let query_oracle = rotary_reference(query, Q_HEADS, &positions);
    let key_oracle = rotary_reference(key, KV_HEADS, &positions);
    let mut max_abs = 0.0_f32;
    for (actual, expected) in outputs[0]
        .iter()
        .zip(&query_oracle)
        .chain(outputs[1].iter().zip(&key_oracle))
    {
        let actual = bf16_to_f32(*actual);
        let expected = bf16_to_f32(*expected);
        let absolute = (actual - expected).abs();
        max_abs = max_abs.max(absolute);
        if !actual.is_finite() || absolute > 0.03125 + 0.03125 * expected.abs() {
            return Err("real-weight rotary slice differs from oracle".to_owned());
        }
    }
    drop(submission);
    drop(prepared);
    drop(buffers);
    Ok(RotarySliceResult {
        query: outputs.remove(0),
        key: outputs.remove(0),
        dispatch,
        max_abs,
    })
}

fn attention_reference(query: &[u16], key: &[u16], value: &[u16]) -> Vec<u16> {
    const M: usize = 3;
    const Q_HEADS: usize = 3;
    const HEAD_DIM: usize = 6;
    let mut output = vec![0_u16; query.len()];
    for row in 0..M {
        for head in 0..Q_HEADS {
            let query_base = (row * Q_HEADS + head) * HEAD_DIM;
            let mut scores = Vec::with_capacity(row + 1);
            let mut maximum = f32::NEG_INFINITY;
            for key_row in 0..=row {
                let key_base = key_row * HEAD_DIM;
                let mut score = 0.0_f32;
                for dimension in 0..HEAD_DIM {
                    score += bf16_to_f32(query[query_base + dimension])
                        * bf16_to_f32(key[key_base + dimension]);
                }
                maximum = maximum.max(score);
                scores.push(score);
            }
            let mut denominator = 0.0_f32;
            for score in &mut scores {
                *score = (*score - maximum).exp();
                denominator += *score;
            }
            for dimension in 0..HEAD_DIM {
                let mut result = 0.0_f32;
                for (key_row, score) in scores.iter().enumerate() {
                    result += *score * bf16_to_f32(value[key_row * HEAD_DIM + dimension]);
                }
                output[query_base + dimension] = f32_to_bf16_rne(result / denominator);
            }
        }
    }
    output
}

fn execute_attention_slice(
    session: &sllm_core::ExecutionSession,
    queue: &sllm_core::ExecutionQueue,
    query: &[u16],
    key: &[u16],
    value: &[u16],
    target: &str,
) -> Result<WordOpResult, String> {
    const M: usize = 3;
    const Q_HEADS: usize = 3;
    const KV_HEADS: usize = 1;
    const HEAD_DIM: usize = 6;
    if query.len() != M * Q_HEADS * HEAD_DIM
        || key.len() != M * KV_HEADS * HEAD_DIM
        || value.len() != key.len()
    {
        return Err("attention real-weight projection shape differs".to_owned());
    }
    let oracle = attention_reference(query, key, value);
    let query_bytes = words_to_bytes(query);
    let key_bytes = words_to_bytes(key);
    let value_bytes = words_to_bytes(value);
    let buffers = [
        session.allocate(query_bytes.len() as u64),
        session.allocate(key_bytes.len() as u64),
        session.allocate(value_bytes.len() as u64),
        session.allocate(query_bytes.len() as u64),
    ]
    .into_iter()
    .collect::<Result<Vec<_>, _>>()
    .map_err(|error| error.to_string())?;
    for (label, buffer, bytes) in [
        ("attention query", &buffers[0], Arc::clone(&query_bytes)),
        ("attention key", &buffers[1], Arc::clone(&key_bytes)),
        ("attention value", &buffers[2], Arc::clone(&value_bytes)),
    ] {
        let mut upload = session
            .upload(
                queue,
                buffer
                    .range(0, bytes.len() as u64)
                    .map_err(|error| error.to_string())?,
                bytes,
            )
            .map_err(|error| format!("{label} upload failed: {error}"))?;
        wait_success(upload.wait(WAIT_TIMEOUT), label)?;
    }
    let query_view = TensorView::contiguous(DType::Bf16, &[M, Q_HEADS, HEAD_DIM])
        .map_err(|error| error.to_string())?;
    let kv_view = TensorView::contiguous(DType::Bf16, &[M, KV_HEADS, HEAD_DIM])
        .map_err(|error| error.to_string())?;
    let contract = WindowedCausalAttentionContract::new(
        Q_HEADS as u32,
        KV_HEADS as u32,
        HEAD_DIM as u32,
        0,
        M as u64,
        M as u64,
        Some(1_024),
        1.0,
    )
    .map_err(|error| error.to_string())?;
    let descriptor = Arc::new(
        SemanticOpDescriptor::new_causal_attention(
            vec![query_view.clone(), kv_view.clone(), kv_view.clone()],
            vec![query_view.clone()],
            contract,
        )
        .map_err(|error| error.to_string())?,
    );
    let operation = Arc::new(
        BoundSemanticOp::new(
            descriptor,
            vec![
                session
                    .bind(&buffers[0], query_view.clone(), AccessMode::Read)
                    .map_err(|error| error.to_string())?,
                session
                    .bind(&buffers[1], kv_view.clone(), AccessMode::Read)
                    .map_err(|error| error.to_string())?,
                session
                    .bind(&buffers[2], kv_view, AccessMode::Read)
                    .map_err(|error| error.to_string())?,
            ],
            vec![
                session
                    .bind(&buffers[3], query_view, AccessMode::Write)
                    .map_err(|error| error.to_string())?,
            ],
        )
        .map_err(|error| error.to_string())?,
    );
    let prepared = session
        .prepare(operation)
        .map_err(|error| error.to_string())?;
    let mut submission = session
        .submit(&prepared, queue)
        .map_err(|error| error.to_string())?;
    wait_success(submission.wait(WAIT_TIMEOUT), "attention slice")?;
    let dispatch = submission.dispatch().clone();
    if dispatch.target != target
        || dispatch.fallback_allowed
        || dispatch.fallback_used
        || dispatch.kernel_symbol != "gemma_causal_attention.online_softmax_gqa_bf16.v1"
    {
        return Err("attention slice dispatch is not exact/no-fallback".to_owned());
    }
    let mut readback = submission
        .start_output_readback(0)
        .map_err(|error| error.to_string())?;
    wait_success(readback.wait(WAIT_TIMEOUT), "attention slice readback")?;
    let mut bytes = vec![0_u8; query_bytes.len()];
    if readback
        .read_into(&mut bytes)
        .map_err(|error| error.to_string())?
        != bytes.len() as u64
    {
        return Err("attention slice readback length differs".to_owned());
    }
    let output = bytes_to_words(&bytes)?;
    let mut max_abs = 0.0_f32;
    for (actual, expected) in output.iter().zip(&oracle) {
        let actual = bf16_to_f32(*actual);
        let expected = bf16_to_f32(*expected);
        let absolute = (actual - expected).abs();
        max_abs = max_abs.max(absolute);
        if !actual.is_finite() || absolute > 0.015625 + 0.03125 * expected.abs() {
            return Err("real-weight attention slice differs from oracle".to_owned());
        }
    }
    drop(readback);
    drop(submission);
    drop(prepared);
    drop(buffers);
    Ok(WordOpResult {
        output,
        dispatch,
        max_abs,
    })
}

fn run(config: Config) -> Result<Report, String> {
    let lock_bytes =
        std::fs::read(&config.lock).map_err(|error| format!("cannot read Gemma lock: {error}"))?;
    let lock = parse_gemma4_model_lock(&lock_bytes)
        .map_err(|error| format!("invalid Gemma lock: {error}"))?;
    let cache = lock
        .verify_cache(&config.cache)
        .map_err(|error| format!("Gemma cache verification failed: {error}"))?;
    let slice = &lock.model.slice_contract;
    if slice.shape != [WIDTH as u64]
        || slice.dtype != TensorDType::Bf16
        || slice.byte_size != (WIDTH * 2) as u64
    {
        return Err("locked Gemma slice is not the reviewed final RMSNorm weight".to_owned());
    }
    let scale_bytes = cache
        .read_tensor_range(&slice.tensor_name, 0, WIDTH * 2)
        .map_err(|error| format!("cannot read verified Gemma tensor slice: {error}"))?;
    let scale = bytes_to_words(&scale_bytes)?;
    let activation = make_activation();
    let oracle = reference(&activation, &scale);

    let backend = HipBackend::connect().map_err(|error| format!("HIP connect failed: {error}"))?;
    let request = ExecutionSessionRequest::new(config.device_index, config.target.clone())
        .map_err(|error| format!("invalid execution request: {error}"))?;
    let session = backend
        .open_execution_session(request)
        .map_err(|error| format!("cannot open HIP execution session: {error}"))?;
    let queue = session
        .create_queue()
        .map_err(|error| format!("queue creation failed: {error}"))?;
    let (embedding_dispatch, embedding_exact_match) =
        execute_embedding_slice(&session, &queue, &cache, &config.target)?;
    let small_activation: Vec<u16> = (0..ROWS * 17)
        .map(|index| f32_to_bf16_rne(((index * 19 + 7) % 97) as f32 / 32.0 - 1.5))
        .collect();
    let gate = execute_matmul_slice(
        &session,
        &queue,
        &cache,
        MatmulSliceRequest {
            tensor_name: "model.language_model.layers.0.mlp.gate_proj.weight",
            activation: &small_activation,
            m: ROWS,
            k: 17,
            n: 3,
            target: &config.target,
        },
    )?;
    let up = execute_matmul_slice(
        &session,
        &queue,
        &cache,
        MatmulSliceRequest {
            tensor_name: "model.language_model.layers.0.mlp.up_proj.weight",
            activation: &small_activation,
            m: ROWS,
            k: 17,
            n: 3,
            target: &config.target,
        },
    )?;
    let (activated, activation_dispatch) =
        execute_gelu_tanh_mul(&session, &queue, &gate.output, &up.output, &config.target)?;
    let down = execute_matmul_slice(
        &session,
        &queue,
        &cache,
        MatmulSliceRequest {
            tensor_name: "model.language_model.layers.0.mlp.down_proj.weight",
            activation: &activated,
            m: ROWS,
            k: 3,
            n: 3,
            target: &config.target,
        },
    )?;
    let logits = execute_matmul_slice(
        &session,
        &queue,
        &cache,
        MatmulSliceRequest {
            tensor_name: "model.language_model.embed_tokens.weight",
            activation: &down.output,
            m: ROWS,
            k: 3,
            n: 3,
            target: &config.target,
        },
    )?;
    let small_graph_max_abs = gate
        .max_abs
        .max(up.max_abs)
        .max(down.max_abs)
        .max(logits.max_abs);
    let query_projection = execute_matmul_slice(
        &session,
        &queue,
        &cache,
        MatmulSliceRequest {
            tensor_name: "model.language_model.layers.0.self_attn.q_proj.weight",
            activation: &small_activation,
            m: ROWS,
            k: 17,
            n: 18,
            target: &config.target,
        },
    )?;
    let key_projection = execute_matmul_slice(
        &session,
        &queue,
        &cache,
        MatmulSliceRequest {
            tensor_name: "model.language_model.layers.0.self_attn.k_proj.weight",
            activation: &small_activation,
            m: ROWS,
            k: 17,
            n: 6,
            target: &config.target,
        },
    )?;
    let value_projection = execute_matmul_slice(
        &session,
        &queue,
        &cache,
        MatmulSliceRequest {
            tensor_name: "model.language_model.layers.0.self_attn.v_proj.weight",
            activation: &small_activation,
            m: ROWS,
            k: 17,
            n: 6,
            target: &config.target,
        },
    )?;
    let q_scale = bytes_to_words(
        &cache
            .read_tensor_range(
                "model.language_model.layers.0.self_attn.q_norm.weight",
                0,
                12,
            )
            .map_err(|error| format!("cannot read q_norm slice: {error}"))?,
    )?;
    let k_scale = bytes_to_words(
        &cache
            .read_tensor_range(
                "model.language_model.layers.0.self_attn.k_norm.weight",
                0,
                12,
            )
            .map_err(|error| format!("cannot read k_norm slice: {error}"))?,
    )?;
    let query_norm = execute_direct_rmsnorm_words(
        &session,
        &queue,
        &query_projection.output,
        &q_scale,
        ROWS * 3,
        6,
        &config.target,
    )?;
    let key_norm = execute_direct_rmsnorm_words(
        &session,
        &queue,
        &key_projection.output,
        &k_scale,
        ROWS,
        6,
        &config.target,
    )?;
    let value_norm = execute_direct_rmsnorm_words(
        &session,
        &queue,
        &value_projection.output,
        &[f32_to_bf16_rne(1.0); 6],
        ROWS,
        6,
        &config.target,
    )?;
    let rotary = execute_rotary_slice(
        &session,
        &queue,
        &query_norm.output,
        &key_norm.output,
        &config.target,
    )?;
    let attention = execute_attention_slice(
        &session,
        &queue,
        &rotary.query,
        &rotary.key,
        &value_norm.output,
        &config.target,
    )?;
    let qkv_projection_max_abs = query_projection
        .max_abs
        .max(key_projection.max_abs)
        .max(value_projection.max_abs)
        .max(query_norm.max_abs)
        .max(key_norm.max_abs)
        .max(value_norm.max_abs);
    let activation_bytes = words_to_bytes(&activation);
    let activation_buffer = session
        .allocate(activation_bytes.len() as u64)
        .map_err(|error| format!("activation allocation failed: {error}"))?;
    let scale_buffer = session
        .allocate(scale_bytes.len() as u64)
        .map_err(|error| format!("scale allocation failed: {error}"))?;
    let output_buffer = session
        .allocate(activation_bytes.len() as u64)
        .map_err(|error| format!("output allocation failed: {error}"))?;
    let mut activation_upload = session
        .upload(
            &queue,
            activation_buffer
                .range(0, activation_bytes.len() as u64)
                .map_err(|error| error.to_string())?,
            Arc::clone(&activation_bytes),
        )
        .map_err(|error| format!("activation upload failed: {error}"))?;
    wait_success(activation_upload.wait(WAIT_TIMEOUT), "activation upload")?;
    drop(activation_upload);
    let mut scale_upload = session
        .upload(
            &queue,
            scale_buffer
                .range(0, scale_bytes.len() as u64)
                .map_err(|error| error.to_string())?,
            Arc::from(scale_bytes.clone()),
        )
        .map_err(|error| format!("scale upload failed: {error}"))?;
    wait_success(scale_upload.wait(WAIT_TIMEOUT), "scale upload")?;
    drop(scale_upload);

    let activation_view =
        TensorView::contiguous(DType::Bf16, &[ROWS, WIDTH]).map_err(|error| error.to_string())?;
    let scale_view =
        TensorView::contiguous(DType::Bf16, &[WIDTH]).map_err(|error| error.to_string())?;
    let descriptor = Arc::new(
        SemanticOpDescriptor::new_rms_norm(
            vec![activation_view.clone(), scale_view.clone()],
            vec![activation_view.clone()],
            EPSILON,
            RmsNormScaleMode::Direct,
        )
        .map_err(|error| format!("RMSNorm descriptor failed: {error}"))?,
    );
    let operation = Arc::new(
        BoundSemanticOp::new(
            descriptor,
            vec![
                session
                    .bind(
                        &activation_buffer,
                        activation_view.clone(),
                        AccessMode::Read,
                    )
                    .map_err(|error| error.to_string())?,
                session
                    .bind(&scale_buffer, scale_view, AccessMode::Read)
                    .map_err(|error| error.to_string())?,
            ],
            vec![
                session
                    .bind(&output_buffer, activation_view, AccessMode::Write)
                    .map_err(|error| error.to_string())?,
            ],
        )
        .map_err(|error| format!("RMSNorm binding failed: {error}"))?,
    );
    let prepared = session
        .prepare(operation)
        .map_err(|error| format!("RMSNorm prepare failed: {error}"))?;
    let mut submission = session
        .submit(&prepared, &queue)
        .map_err(|error| format!("RMSNorm submit failed: {error}"))?;
    wait_success(submission.wait(WAIT_TIMEOUT), "RMSNorm execution")?;
    let dispatch = submission.dispatch().clone();
    if dispatch.dispatch_count != 1
        || dispatch.target != config.target
        || dispatch.fallback_allowed
        || dispatch.fallback_used
        || dispatch.kernel_symbol != "rmsnorm.baseline.wave32.v1"
        || dispatch.device_symbol != "sllm_rmsnorm_baseline_wave32_v1"
    {
        return Err("RMSNorm dispatch evidence is not exact/no-fallback".to_owned());
    }
    let mut readback = submission
        .start_output_readback(0)
        .map_err(|error| format!("output readback failed: {error}"))?;
    wait_success(readback.wait(WAIT_TIMEOUT), "output readback")?;
    let mut output_bytes = vec![0_u8; activation_bytes.len()];
    if readback
        .read_into(&mut output_bytes)
        .map_err(|error| format!("output read failed: {error}"))?
        != output_bytes.len() as u64
    {
        return Err("output readback length differs".to_owned());
    }
    let output = bytes_to_words(&output_bytes)?;
    let mut max_abs = 0.0_f32;
    let mut max_scaled_rel = 0.0_f32;
    for (observed, expected) in output.iter().zip(&oracle) {
        let observed = bf16_to_f32(*observed);
        let expected = bf16_to_f32(*expected);
        let absolute = (observed - expected).abs();
        max_abs = max_abs.max(absolute);
        max_scaled_rel = max_scaled_rel.max(absolute / expected.abs().max(0.03125));
        if !observed.is_finite() || absolute > 0.03125 + 0.03125 * expected.abs() {
            return Err(format!(
                "real-weight RMSNorm differs from oracle: observed={observed} expected={expected}"
            ));
        }
    }
    drop(readback);
    drop(submission);
    drop(prepared);
    drop(output_buffer);
    drop(scale_buffer);
    drop(activation_buffer);
    drop(queue);
    let cleanup = session
        .shutdown(Duration::from_secs(16))
        .map_err(|error| format!("session cleanup failed: {error}"))?;
    if cleanup.retryable_cleanup != 0 || cleanup.durable_quarantine != 0 {
        return Err("session cleanup retained resources".to_owned());
    }
    Ok(Report {
        schema_version: "gemma4-real-weight-slice-v1",
        state: "PASS",
        model: "google/gemma-4-12B",
        resolved_revision: lock.model.resolved_revision.clone(),
        lock_fingerprint: lock.fingerprint().to_owned(),
        tensor: slice.tensor_name.clone(),
        tensor_bytes: scale_bytes.len(),
        rows: ROWS,
        width: WIDTH,
        target: config.target,
        device_index: config.device_index,
        fallback_allowed: dispatch.fallback_allowed,
        fallback_used: dispatch.fallback_used,
        operations: 15,
        embedding_tensor: "model.language_model.embed_tokens.weight".to_owned(),
        embedding_exact_match,
        embedding_kernel_symbol: embedding_dispatch.kernel_symbol,
        embedding_device_symbol: embedding_dispatch.device_symbol,
        rmsnorm_kernel_symbol: dispatch.kernel_symbol,
        rmsnorm_device_symbol: dispatch.device_symbol,
        mlp_gate_kernel_symbol: gate.dispatch.kernel_symbol,
        mlp_up_kernel_symbol: up.dispatch.kernel_symbol,
        mlp_activation_kernel_symbol: activation_dispatch.kernel_symbol,
        mlp_down_kernel_symbol: down.dispatch.kernel_symbol,
        tied_logits_kernel_symbol: logits.dispatch.kernel_symbol,
        small_graph_max_abs,
        qkv_projection_max_abs,
        rotary_kernel_symbol: rotary.dispatch.kernel_symbol,
        rotary_max_abs: rotary.max_abs,
        attention_kernel_symbol: attention.dispatch.kernel_symbol,
        attention_max_abs: attention.max_abs,
        max_abs,
        max_scaled_rel,
        cleanup_retryable: cleanup.retryable_cleanup,
        cleanup_durable: cleanup.durable_quarantine,
    })
}

fn main() -> ExitCode {
    match parse_config().and_then(run) {
        Ok(report) => match serde_json::to_string(&report) {
            Ok(json) => {
                println!("{json}");
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("cannot serialize evidence: {error}");
                ExitCode::FAILURE
            }
        },
        Err(error) => {
            eprintln!("Gemma 4 slice evidence failed: {error}");
            ExitCode::FAILURE
        }
    }
}
