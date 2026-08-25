//! Model-free numerical evidence for the public Qwen3.5 GDN state path.

use std::env;
use std::ffi::OsString;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;
#[cfg(test)]
use sllm_core::LinearAttentionLayout;
use sllm_core::{
    AccessMode, Backend, DType, Encoding, ExecutionSessionRequest, ExecutionState,
    LinearAttentionBindings, LinearAttentionDescriptor, LinearAttentionStateDescriptor,
    OwnedTensorBinding, TensorView,
};
use sllm_hip::HipBackend;

const CASE_TOKENS: [usize; 7] = [1, 3, 17, 32, 127, 128, 129];
const PHASE12_CASE_TOKENS: [usize; 3] = [1, 3, 17];
const QK_HEADS: usize = 16;
const VALUE_HEADS: usize = 32;
const HEAD_DIM: usize = 128;
const CONV_KERNEL: usize = 4;
const QKV_WIDTH: usize = (2 * QK_HEADS + VALUE_HEADS) * HEAD_DIM;
const OUTPUT_WIDTH: usize = VALUE_HEADS * HEAD_DIM;
const WAIT: Duration = Duration::from_secs(30);

#[derive(Debug)]
struct Config {
    device_index: u32,
    target: String,
    phase12_subset: bool,
}

#[derive(Serialize)]
struct CaseEvidence {
    tokens: usize,
    dispatch_count: u32,
    recurrent_kernel_id: u32,
    kernel_symbol: String,
    recurrent_device_symbol: String,
    max_abs_error: f32,
    max_rel_error: f32,
    state_length_after: u64,
}

#[derive(Serialize)]
struct Report {
    schema_version: &'static str,
    state: &'static str,
    target: String,
    device_index: u32,
    selected_backend: &'static str,
    model_used: bool,
    layout: [usize; 4],
    cases: Vec<CaseEvidence>,
    continuation: Option<ContinuationEvidence>,
    fallback_allowed: bool,
    fallback_used: bool,
    cpu_fallback_used: bool,
    cleanup_retryable: usize,
    cleanup_durable: usize,
}

#[derive(Serialize)]
struct ContinuationEvidence {
    first_tokens: usize,
    second_tokens: usize,
    first_kernel_symbol: String,
    first_recurrent_device_symbol: String,
    second_kernel_symbol: String,
    second_recurrent_device_symbol: String,
    second_max_abs_error: f32,
    second_max_rel_error: f32,
    state_length_after_first: u64,
    final_state_length: u64,
    final_state_layout: [usize; 4],
}

struct EnvironmentRestore {
    name: &'static str,
    value: Option<OsString>,
}

impl EnvironmentRestore {
    fn set(name: &'static str, value: &str) -> Self {
        let previous = env::var_os(name);
        // This evidence binary performs submissions serially on the main
        // thread. The temporary mutation is scoped across exactly one native
        // dispatch and restored on every return path.
        unsafe { env::set_var(name, value) };
        Self {
            name,
            value: previous,
        }
    }
}

impl Drop for EnvironmentRestore {
    fn drop(&mut self) {
        if let Some(value) = &self.value {
            unsafe { env::set_var(self.name, value) };
        } else {
            unsafe { env::remove_var(self.name) };
        }
    }
}

fn parse_config_from<I, S>(arguments: I) -> Result<Config, String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut device_index = None;
    let mut target = None;
    let mut phase12_subset = false;
    let mut args = arguments.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_ref() {
            "--device-index" => {
                device_index = Some(
                    args.next()
                        .ok_or_else(|| "--device-index requires a value".to_owned())?
                        .as_ref()
                        .parse::<u32>()
                        .map_err(|_| "--device-index must be a u32".to_owned())?,
                );
            }
            "--target" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--target requires a value".to_owned())?;
                if !matches!(value.as_ref(), "gfx1030" | "gfx1201" | "gfx942") {
                    return Err("--target must be gfx1030, gfx1201, or gfx942".to_owned());
                }
                target = Some(value.as_ref().to_owned());
            }
            "--phase12-subset" => {
                if phase12_subset {
                    return Err("duplicate --phase12-subset".to_owned());
                }
                phase12_subset = true;
            }
            other => return Err(format!("unexpected argument `{other}`")),
        }
    }
    Ok(Config {
        device_index: device_index.ok_or_else(|| "missing --device-index".to_owned())?,
        target: target.ok_or_else(|| "missing --target".to_owned())?,
        phase12_subset,
    })
}

fn parse_config() -> Result<Config, String> {
    parse_config_from(env::args().skip(1))
}

fn selected_case_tokens(phase12_subset: bool) -> &'static [usize] {
    if phase12_subset {
        &PHASE12_CASE_TOKENS
    } else {
        &CASE_TOKENS
    }
}

fn short_column_state_enabled(
    target: &str,
    tokens: usize,
    force_baseline: bool,
    opt_in: Option<&std::ffi::OsStr>,
) -> bool {
    !force_baseline
        && target == "gfx1030"
        && opt_in.is_none_or(|value| value == "1")
        && (17..128).contains(&tokens)
        && QK_HEADS == 16
        && VALUE_HEADS == 32
        && HEAD_DIM == 128
}

fn column_provider_enabled(
    target: &str,
    tokens: usize,
    force_baseline: bool,
    short_opt_in: Option<&std::ffi::OsStr>,
    gfx942_wave64_opt_in: Option<&std::ffi::OsStr>,
) -> bool {
    (!force_baseline && tokens >= 128 && matches!(target, "gfx1030" | "gfx1201"))
        || short_column_state_enabled(target, tokens, force_baseline, short_opt_in)
        || gfx942_wave64_column_state_enabled(target, tokens, force_baseline, gfx942_wave64_opt_in)
}

fn gfx942_wave64_column_state_enabled(
    target: &str,
    tokens: usize,
    force_baseline: bool,
    opt_in: Option<&std::ffi::OsStr>,
) -> bool {
    !force_baseline
        && target == "gfx942:sramecc+:xnack-"
        && opt_in.is_some_and(|value| value == "1")
        && tokens >= 128
        && QK_HEADS == 16
        && VALUE_HEADS == 32
        && HEAD_DIM == 128
}

fn f32_to_bf16(value: f32) -> u16 {
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

fn bf16_to_f32(value: u16) -> f32 {
    f32::from_bits(u32::from(value) << 16)
}

fn bf16_bytes(words: &[u16]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

fn f32_bytes(words: &[f32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
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

fn upload_binding(
    session: &sllm_core::ExecutionSession,
    queue: &sllm_core::ExecutionQueue,
    dtype: DType,
    shape: &[usize],
    access: AccessMode,
    bytes: Vec<u8>,
) -> Result<(sllm_core::ExecutionBuffer, OwnedTensorBinding), String> {
    let view = TensorView::with_encoding(dtype, Encoding::Unquantized, shape)
        .map_err(|error| format!("tensor view failed: {error}"))?;
    if view.payload_bytes() != bytes.len() as u64 {
        return Err("input byte length does not match its tensor view".to_owned());
    }
    let buffer = session
        .allocate(bytes.len() as u64)
        .map_err(|error| format!("allocation failed: {error}"))?;
    let mut upload = session
        .upload(
            queue,
            buffer
                .range(0, bytes.len() as u64)
                .map_err(|error| error.to_string())?,
            Arc::<[u8]>::from(bytes),
        )
        .map_err(|error| format!("upload failed: {error}"))?;
    wait_success(upload.wait(WAIT), "upload")?;
    let binding = session
        .bind(&buffer, view, access)
        .map_err(|error| format!("binding failed: {error}"))?;
    Ok((buffer, binding))
}

#[allow(clippy::too_many_arguments)]
fn upload_continuation_bindings(
    session: &sllm_core::ExecutionSession,
    queue: &sllm_core::ExecutionQueue,
    tokens: usize,
    qkv: &[u16],
    z: &[u16],
    b_input: &[u16],
    a_input: &[u16],
    conv_weight: &[u16],
    a_log: &[f32],
    dt_bias: &[u16],
    norm_weight: &[f32],
) -> Result<(LinearAttentionBindings, sllm_core::ExecutionBuffer), String> {
    let (_, qkv_binding) = upload_binding(
        session,
        queue,
        DType::Bf16,
        &[tokens, QKV_WIDTH],
        AccessMode::Read,
        bf16_bytes(qkv),
    )?;
    let (_, z_binding) = upload_binding(
        session,
        queue,
        DType::Bf16,
        &[tokens, OUTPUT_WIDTH],
        AccessMode::Read,
        bf16_bytes(z),
    )?;
    let (_, b_binding) = upload_binding(
        session,
        queue,
        DType::Bf16,
        &[tokens, VALUE_HEADS],
        AccessMode::Read,
        bf16_bytes(b_input),
    )?;
    let (_, a_binding) = upload_binding(
        session,
        queue,
        DType::Bf16,
        &[tokens, VALUE_HEADS],
        AccessMode::Read,
        bf16_bytes(a_input),
    )?;
    let (_, conv_binding) = upload_binding(
        session,
        queue,
        DType::Bf16,
        &[QKV_WIDTH, 1, CONV_KERNEL],
        AccessMode::Read,
        bf16_bytes(conv_weight),
    )?;
    let (_, a_log_binding) = upload_binding(
        session,
        queue,
        DType::F32,
        &[VALUE_HEADS],
        AccessMode::Read,
        f32_bytes(a_log),
    )?;
    let (_, dt_binding) = upload_binding(
        session,
        queue,
        DType::Bf16,
        &[VALUE_HEADS],
        AccessMode::Read,
        bf16_bytes(dt_bias),
    )?;
    let (_, norm_binding) = upload_binding(
        session,
        queue,
        DType::F32,
        &[HEAD_DIM],
        AccessMode::Read,
        f32_bytes(norm_weight),
    )?;
    let output_bytes = tokens * OUTPUT_WIDTH * 2;
    let output_buffer = session
        .allocate(output_bytes as u64)
        .map_err(|error| format!("continuation output allocation failed: {error}"))?;
    let output_view =
        TensorView::with_encoding(DType::Bf16, Encoding::Unquantized, &[tokens, OUTPUT_WIDTH])
            .map_err(|error| error.to_string())?;
    let output_binding = session
        .bind(&output_buffer, output_view, AccessMode::Write)
        .map_err(|error| format!("continuation output binding failed: {error}"))?;
    Ok((
        LinearAttentionBindings::new(
            qkv_binding,
            z_binding,
            b_binding,
            a_binding,
            conv_binding,
            a_log_binding,
            dt_binding,
            norm_binding,
            output_binding,
        ),
        output_buffer,
    ))
}

#[allow(clippy::type_complexity)]
fn inputs(
    tokens: usize,
) -> (
    Vec<u16>,
    Vec<u16>,
    Vec<u16>,
    Vec<u16>,
    Vec<u16>,
    Vec<f32>,
    Vec<u16>,
    Vec<f32>,
) {
    let qkv = (0..tokens * QKV_WIDTH)
        .map(|index| f32_to_bf16(((index * 17 + 3) % 31) as f32 / 64.0 - 0.234375))
        .collect();
    let z = (0..tokens * OUTPUT_WIDTH)
        .map(|index| f32_to_bf16(((index * 13 + 5) % 23) as f32 / 32.0 - 0.25))
        .collect();
    let b_input = (0..tokens * VALUE_HEADS)
        .map(|index| f32_to_bf16(index as f32 / 32.0 - 0.125))
        .collect();
    let a_input = (0..tokens * VALUE_HEADS)
        .map(|index| f32_to_bf16(index as f32 / 64.0 - 0.25))
        .collect();
    let mut conv_weight = vec![0_u16; QKV_WIDTH * CONV_KERNEL];
    for channel in 0..QKV_WIDTH {
        conv_weight[channel * CONV_KERNEL + CONV_KERNEL - 1] = f32_to_bf16(1.0);
    }
    let a_log = (0..VALUE_HEADS)
        .map(|index| -0.75 + (index % 11) as f32 / 32.0)
        .collect();
    let dt_bias = (0..VALUE_HEADS)
        .map(|index| f32_to_bf16(0.0625 + (index % 7) as f32 / 64.0))
        .collect();
    let norm_weight = (0..HEAD_DIM)
        .map(|index| 0.75 + (index % 17) as f32 / 64.0)
        .collect();
    (
        qkv,
        z,
        b_input,
        a_input,
        conv_weight,
        a_log,
        dt_bias,
        norm_weight,
    )
}

#[allow(clippy::too_many_arguments)]
fn oracle(
    tokens: usize,
    qkv: &[u16],
    z: &[u16],
    b_input: &[u16],
    a_input: &[u16],
    a_log: &[f32],
    dt_bias: &[u16],
    norm_weight: &[f32],
) -> Vec<u16> {
    let convolved: Vec<f32> = qkv
        .iter()
        .map(|&bits| {
            let value = bf16_to_f32(bits);
            bf16_to_f32(f32_to_bf16(value / (1.0 + (-value).exp())))
        })
        .collect();
    let mut recurrent = vec![0.0_f32; VALUE_HEADS * HEAD_DIM * HEAD_DIM];
    let mut result = vec![0_u16; tokens * OUTPUT_WIDTH];
    for token in 0..tokens {
        let row = token * QKV_WIDTH;
        for value_head in 0..VALUE_HEADS {
            let qk_head = value_head / (VALUE_HEADS / QK_HEADS);
            let q_base = row + qk_head * HEAD_DIM;
            let k_base = row + QK_HEADS * HEAD_DIM + qk_head * HEAD_DIM;
            let mut q = convolved[q_base..q_base + HEAD_DIM].to_vec();
            let mut k = convolved[k_base..k_base + HEAD_DIM].to_vec();
            let q_inverse =
                1.0 / (q.iter().map(|value| value * value).sum::<f32>() + 1.0e-6).sqrt();
            let k_inverse =
                1.0 / (k.iter().map(|value| value * value).sum::<f32>() + 1.0e-6).sqrt();
            for dimension in 0..HEAD_DIM {
                q[dimension] =
                    bf16_to_f32(f32_to_bf16(q[dimension] * q_inverse)) / (HEAD_DIM as f32).sqrt();
                k[dimension] = bf16_to_f32(f32_to_bf16(k[dimension] * k_inverse));
            }
            let scalar = token * VALUE_HEADS + value_head;
            let beta = bf16_to_f32(f32_to_bf16(
                1.0 / (1.0 + (-bf16_to_f32(b_input[scalar])).exp()),
            ));
            let a = bf16_to_f32(a_input[scalar]) + bf16_to_f32(dt_bias[value_head]);
            let softplus = a.max(0.0) + (-a.abs()).exp().ln_1p();
            let decay = (-a_log[value_head].exp() * softplus).exp();
            let mut output_values = vec![0.0_f32; HEAD_DIM];
            for dimension in 0..HEAD_DIM {
                let state_offset = (value_head * HEAD_DIM + dimension) * HEAD_DIM;
                let state_row = &mut recurrent[state_offset..state_offset + HEAD_DIM];
                for value in state_row.iter_mut() {
                    *value *= decay;
                }
                let previous = state_row
                    .iter()
                    .zip(&k)
                    .map(|(state, key)| state * key)
                    .sum::<f32>();
                let value =
                    convolved[row + 2 * QK_HEADS * HEAD_DIM + value_head * HEAD_DIM + dimension];
                let residual = value - previous;
                let mut projection = 0.0_f32;
                for key_dimension in 0..HEAD_DIM {
                    state_row[key_dimension] += beta * residual * k[key_dimension];
                    projection += state_row[key_dimension] * q[key_dimension];
                }
                output_values[dimension] = bf16_to_f32(f32_to_bf16(projection));
            }
            let inverse_rms = 1.0
                / (output_values.iter().map(|value| value * value).sum::<f32>() / HEAD_DIM as f32
                    + 1.0e-6)
                    .sqrt();
            for dimension in 0..HEAD_DIM {
                let normalized = bf16_to_f32(f32_to_bf16(output_values[dimension] * inverse_rms));
                let output_index = token * OUTPUT_WIDTH + value_head * HEAD_DIM + dimension;
                let z_value = bf16_to_f32(z[output_index]);
                let z_silu = z_value / (1.0 + (-z_value).exp());
                result[output_index] = f32_to_bf16(normalized * norm_weight[dimension] * z_silu);
            }
        }
    }
    result
}

fn run_case(
    session: &sllm_core::ExecutionSession,
    queue: &sllm_core::ExecutionQueue,
    tokens: usize,
    target: &str,
) -> Result<CaseEvidence, String> {
    let (qkv, z, b_input, a_input, conv_weight, a_log, dt_bias, norm_weight) = inputs(tokens);
    let expected = oracle(
        tokens,
        &qkv,
        &z,
        &b_input,
        &a_input,
        &a_log,
        &dt_bias,
        &norm_weight,
    );
    let (_, qkv_binding) = upload_binding(
        session,
        queue,
        DType::Bf16,
        &[tokens, QKV_WIDTH],
        AccessMode::Read,
        bf16_bytes(&qkv),
    )?;
    let (_, z_binding) = upload_binding(
        session,
        queue,
        DType::Bf16,
        &[tokens, OUTPUT_WIDTH],
        AccessMode::Read,
        bf16_bytes(&z),
    )?;
    let (_, b_binding) = upload_binding(
        session,
        queue,
        DType::Bf16,
        &[tokens, VALUE_HEADS],
        AccessMode::Read,
        bf16_bytes(&b_input),
    )?;
    let (_, a_binding) = upload_binding(
        session,
        queue,
        DType::Bf16,
        &[tokens, VALUE_HEADS],
        AccessMode::Read,
        bf16_bytes(&a_input),
    )?;
    let (_, conv_binding) = upload_binding(
        session,
        queue,
        DType::Bf16,
        &[QKV_WIDTH, 1, CONV_KERNEL],
        AccessMode::Read,
        bf16_bytes(&conv_weight),
    )?;
    let (_, a_log_binding) = upload_binding(
        session,
        queue,
        DType::F32,
        &[VALUE_HEADS],
        AccessMode::Read,
        f32_bytes(&a_log),
    )?;
    let (_, dt_binding) = upload_binding(
        session,
        queue,
        DType::Bf16,
        &[VALUE_HEADS],
        AccessMode::Read,
        bf16_bytes(&dt_bias),
    )?;
    let (_, norm_binding) = upload_binding(
        session,
        queue,
        DType::F32,
        &[HEAD_DIM],
        AccessMode::Read,
        f32_bytes(&norm_weight),
    )?;
    let output_bytes = tokens * OUTPUT_WIDTH * 2;
    let output_buffer = session
        .allocate(output_bytes as u64)
        .map_err(|error| format!("output allocation failed: {error}"))?;
    let output_view =
        TensorView::with_encoding(DType::Bf16, Encoding::Unquantized, &[tokens, OUTPUT_WIDTH])
            .map_err(|error| error.to_string())?;
    let output_binding = session
        .bind(&output_buffer, output_view, AccessMode::Write)
        .map_err(|error| format!("output binding failed: {error}"))?;
    let bindings = LinearAttentionBindings::new(
        qkv_binding,
        z_binding,
        b_binding,
        a_binding,
        conv_binding,
        a_log_binding,
        dt_binding,
        norm_binding,
        output_binding,
    );
    let state_descriptor = LinearAttentionStateDescriptor::new_with_layout(
        12,
        tokens as u64,
        QK_HEADS,
        VALUE_HEADS,
        HEAD_DIM,
        CONV_KERNEL,
    )
    .map_err(|error| error.to_string())?;
    let state = session
        .create_linear_attention_state(state_descriptor)
        .map_err(|error| format!("state creation failed: {error}"))?;
    let descriptor = LinearAttentionDescriptor::new(0, tokens as u64, tokens as u64)
        .map_err(|error| error.to_string())?;
    let mut submission = session
        .linear_attention(&state, queue, bindings, descriptor)
        .map_err(|error| format!("GDN submission failed: {error}"))?;
    let dispatch = submission.dispatch().clone();
    let force_baseline = env::var_os("SLLM_GDN_FORCE_BASELINE").is_some_and(|value| value == "1");
    let short_opt_in = env::var_os("SLLM_LINEAR_ATTENTION_GFX1030_SHORT_COLUMN_STATE");
    let gfx942_wave64_opt_in = env::var_os("SLLM_LINEAR_ATTENTION_GFX942_WAVE64_COLUMN_STATE");
    let selection_target = if target == "gfx942" {
        "gfx942:sramecc+:xnack-"
    } else {
        target
    };
    let use_gfx942_wave64_column_provider = gfx942_wave64_column_state_enabled(
        selection_target,
        tokens,
        force_baseline,
        gfx942_wave64_opt_in.as_deref(),
    );
    let use_column_provider = column_provider_enabled(
        selection_target,
        tokens,
        force_baseline,
        short_opt_in.as_deref(),
        gfx942_wave64_opt_in.as_deref(),
    );
    let use_decode_pair_provider = tokens == 1 && !force_baseline && target == "gfx1030";
    if dispatch.dispatch_count != if use_column_provider { 4 } else { 2 }
        || dispatch.kernel_id != 2
        || dispatch.workgroup_size_x
            != if use_gfx942_wave64_column_provider || use_decode_pair_provider {
                256
            } else {
                128
            }
        || dispatch.grid_size_x
            != if use_column_provider {
                (VALUE_HEADS * HEAD_DIM / 4) as u32
            } else if use_decode_pair_provider {
                QK_HEADS as u32
            } else {
                VALUE_HEADS as u32
            }
        || dispatch.row_count != tokens as u64
        || dispatch.normalized_size != HEAD_DIM as u64
        || dispatch.fallback_allowed
        || dispatch.fallback_used
        || dispatch.kernel_symbol
            != if use_gfx942_wave64_column_provider {
                "linear_attention.gdn.column_state.gfx942_wave64.v3"
            } else if use_column_provider {
                "linear_attention.gdn.column_state.v2"
            } else if use_decode_pair_provider {
                "linear_attention.gdn.decode_pair.v1"
            } else {
                "linear_attention.gdn.v1"
            }
        || dispatch.device_symbol
            != if use_gfx942_wave64_column_provider {
                "sllm_linear_attention_column_state_wave64_v3"
            } else if use_column_provider {
                "sllm_linear_attention_recurrent_column_state_v2"
            } else if use_decode_pair_provider {
                "sllm_linear_attention_recurrent_gated_norm_decode_pair_v1"
            } else {
                "sllm_linear_attention_recurrent_gated_norm_v1"
            }
        || dispatch.target != target
    {
        return Err(format!(
            "GDN dispatch metadata violated the exact contract: {dispatch:?}"
        ));
    }
    wait_success(submission.wait(WAIT), "GDN completion")?;
    let state_length_after = state
        .snapshot(session)
        .map_err(|error| format!("state snapshot failed: {error}"))?
        .length();
    if state_length_after != tokens as u64 {
        return Err("GDN state publication length mismatch".to_owned());
    }
    let mut readback = session
        .readback(
            queue,
            output_buffer
                .range(0, output_bytes as u64)
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| format!("output readback failed: {error}"))?;
    wait_success(readback.wait(WAIT), "output readback")?;
    let mut actual_bytes = vec![0_u8; output_bytes];
    readback
        .read_into(&mut actual_bytes)
        .map_err(|error| format!("output read failed: {error}"))?;
    let actual: Vec<u16> = actual_bytes
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect();
    let mut max_abs_error = 0.0_f32;
    let mut max_rel_error = 0.0_f32;
    for (&actual_bits, &expected_bits) in actual.iter().zip(&expected) {
        let actual = bf16_to_f32(actual_bits);
        let expected = bf16_to_f32(expected_bits);
        let absolute = (actual - expected).abs();
        let relative = if expected == 0.0 {
            absolute
        } else {
            absolute / expected.abs()
        };
        max_abs_error = max_abs_error.max(absolute);
        max_rel_error = max_rel_error.max(relative);
        if absolute > 0.015625 + 0.03125 * expected.abs() {
            return Err(format!(
                "GDN numerical mismatch for tokens={tokens}: actual={actual} expected={expected}"
            ));
        }
    }
    drop(readback);
    drop(submission);
    drop(state);
    drop(output_buffer);
    Ok(CaseEvidence {
        tokens,
        dispatch_count: dispatch.dispatch_count,
        recurrent_kernel_id: dispatch.kernel_id,
        kernel_symbol: dispatch.kernel_symbol,
        recurrent_device_symbol: dispatch.device_symbol,
        max_abs_error,
        max_rel_error,
        state_length_after,
    })
}

fn run_gfx942_mixed_provider_continuation(
    session: &sllm_core::ExecutionSession,
    queue: &sllm_core::ExecutionQueue,
    target: &str,
) -> Result<Option<ContinuationEvidence>, String> {
    const FIRST_TOKENS: usize = 128;
    const SECOND_TOKENS: usize = 128;
    const TOTAL_TOKENS: usize = FIRST_TOKENS + SECOND_TOKENS;
    const FORCE_NAME: &str = "SLLM_GDN_FORCE_BASELINE";
    let candidate_opt_in = env::var_os("SLLM_LINEAR_ATTENTION_GFX942_WAVE64_COLUMN_STATE")
        .is_some_and(|value| value == "1");
    let force_baseline = env::var_os(FORCE_NAME).is_some_and(|value| value == "1");
    if target != "gfx942" || !candidate_opt_in || force_baseline {
        return Ok(None);
    }

    let (qkv, z, b_input, a_input, conv_weight, a_log, dt_bias, norm_weight) = inputs(TOTAL_TOKENS);
    let expected = oracle(
        TOTAL_TOKENS,
        &qkv,
        &z,
        &b_input,
        &a_input,
        &a_log,
        &dt_bias,
        &norm_weight,
    );
    let state_descriptor = LinearAttentionStateDescriptor::new_with_layout(
        13,
        TOTAL_TOKENS as u64,
        QK_HEADS,
        VALUE_HEADS,
        HEAD_DIM,
        CONV_KERNEL,
    )
    .map_err(|error| error.to_string())?;
    let state = session
        .create_linear_attention_state(state_descriptor)
        .map_err(|error| format!("continuation state creation failed: {error}"))?;

    let (first_bindings, first_output) = upload_continuation_bindings(
        session,
        queue,
        FIRST_TOKENS,
        &qkv[..FIRST_TOKENS * QKV_WIDTH],
        &z[..FIRST_TOKENS * OUTPUT_WIDTH],
        &b_input[..FIRST_TOKENS * VALUE_HEADS],
        &a_input[..FIRST_TOKENS * VALUE_HEADS],
        &conv_weight,
        &a_log,
        &dt_bias,
        &norm_weight,
    )?;
    let first_descriptor =
        LinearAttentionDescriptor::new(0, FIRST_TOKENS as u64, FIRST_TOKENS as u64)
            .map_err(|error| error.to_string())?;
    let mut first_submission = session
        .linear_attention(&state, queue, first_bindings, first_descriptor)
        .map_err(|error| format!("continuation candidate submission failed: {error}"))?;
    let first_dispatch = first_submission.dispatch().clone();
    if first_dispatch.dispatch_count != 4
        || first_dispatch.workgroup_size_x != 256
        || first_dispatch.grid_size_x != (VALUE_HEADS * HEAD_DIM / 4) as u32
        || first_dispatch.kernel_symbol != "linear_attention.gdn.column_state.gfx942_wave64.v3"
        || first_dispatch.device_symbol != "sllm_linear_attention_column_state_wave64_v3"
        || first_dispatch.target != "gfx942"
        || first_dispatch.fallback_allowed
        || first_dispatch.fallback_used
    {
        return Err(format!(
            "continuation first dispatch did not select exact gfx942 v3: {first_dispatch:?}"
        ));
    }
    wait_success(
        first_submission.wait(WAIT),
        "continuation candidate completion",
    )?;
    let first_snapshot = state
        .snapshot(session)
        .map_err(|error| format!("continuation first snapshot failed: {error}"))?;
    if first_snapshot.length() != FIRST_TOKENS as u64
        || first_snapshot.descriptor() != state_descriptor
    {
        return Err("continuation first state publication/layout mismatch".to_owned());
    }
    drop(first_submission);
    drop(first_output);

    let qkv_offset = FIRST_TOKENS * QKV_WIDTH;
    let output_offset = FIRST_TOKENS * OUTPUT_WIDTH;
    let scalar_offset = FIRST_TOKENS * VALUE_HEADS;
    let (second_bindings, second_output) = upload_continuation_bindings(
        session,
        queue,
        SECOND_TOKENS,
        &qkv[qkv_offset..qkv_offset + SECOND_TOKENS * QKV_WIDTH],
        &z[output_offset..output_offset + SECOND_TOKENS * OUTPUT_WIDTH],
        &b_input[scalar_offset..scalar_offset + SECOND_TOKENS * VALUE_HEADS],
        &a_input[scalar_offset..scalar_offset + SECOND_TOKENS * VALUE_HEADS],
        &conv_weight,
        &a_log,
        &dt_bias,
        &norm_weight,
    )?;
    let second_descriptor = LinearAttentionDescriptor::new(
        FIRST_TOKENS as u64,
        SECOND_TOKENS as u64,
        TOTAL_TOKENS as u64,
    )
    .map_err(|error| error.to_string())?;
    let force_restore = EnvironmentRestore::set(FORCE_NAME, "1");
    let mut second_submission = session
        .linear_attention(&state, queue, second_bindings, second_descriptor)
        .map_err(|error| format!("continuation forced-baseline submission failed: {error}"))?;
    let second_dispatch = second_submission.dispatch().clone();
    if second_dispatch.dispatch_count != 2
        || second_dispatch.workgroup_size_x != 128
        || second_dispatch.grid_size_x != VALUE_HEADS as u32
        || second_dispatch.kernel_symbol != "linear_attention.gdn.v1"
        || second_dispatch.device_symbol != "sllm_linear_attention_recurrent_gated_norm_v1"
        || second_dispatch.target != "gfx942"
        || second_dispatch.fallback_allowed
        || second_dispatch.fallback_used
    {
        return Err(format!(
            "continuation second dispatch did not force baseline: {second_dispatch:?}"
        ));
    }
    wait_success(
        second_submission.wait(WAIT),
        "continuation forced-baseline completion",
    )?;
    let final_snapshot = state
        .snapshot(session)
        .map_err(|error| format!("continuation final snapshot failed: {error}"))?;
    if final_snapshot.length() != TOTAL_TOKENS as u64
        || final_snapshot.descriptor() != state_descriptor
    {
        return Err("continuation final state publication/layout mismatch".to_owned());
    }
    let second_output_bytes = SECOND_TOKENS * OUTPUT_WIDTH * 2;
    let mut readback = session
        .readback(
            queue,
            second_output
                .range(0, second_output_bytes as u64)
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| format!("continuation output readback failed: {error}"))?;
    wait_success(readback.wait(WAIT), "continuation output readback")?;
    let mut actual_bytes = vec![0_u8; second_output_bytes];
    readback
        .read_into(&mut actual_bytes)
        .map_err(|error| format!("continuation output read failed: {error}"))?;
    let actual = actual_bytes
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]));
    let expected_second = &expected[output_offset..];
    let mut max_abs_error = 0.0_f32;
    let mut max_rel_error = 0.0_f32;
    for (actual_bits, &expected_bits) in actual.zip(expected_second) {
        let actual_value = bf16_to_f32(actual_bits);
        let expected_value = bf16_to_f32(expected_bits);
        let absolute = (actual_value - expected_value).abs();
        let relative = if expected_value == 0.0 {
            absolute
        } else {
            absolute / expected_value.abs()
        };
        max_abs_error = max_abs_error.max(absolute);
        max_rel_error = max_rel_error.max(relative);
        if absolute > 0.015625 + 0.03125 * expected_value.abs() {
            return Err(format!(
                "continuation second output mismatch: actual={actual_value} expected={expected_value}"
            ));
        }
    }
    let layout = final_snapshot.descriptor().layout();
    let evidence = ContinuationEvidence {
        first_tokens: FIRST_TOKENS,
        second_tokens: SECOND_TOKENS,
        first_kernel_symbol: first_dispatch.kernel_symbol,
        first_recurrent_device_symbol: first_dispatch.device_symbol,
        second_kernel_symbol: second_dispatch.kernel_symbol,
        second_recurrent_device_symbol: second_dispatch.device_symbol,
        second_max_abs_error: max_abs_error,
        second_max_rel_error: max_rel_error,
        state_length_after_first: first_snapshot.length(),
        final_state_length: final_snapshot.length(),
        final_state_layout: [
            layout.qk_heads(),
            layout.value_heads(),
            layout.head_dim(),
            layout.conv_kernel_size(),
        ],
    };
    drop(readback);
    drop(second_submission);
    drop(second_output);
    drop(force_restore);
    drop(state);
    Ok(Some(evidence))
}

fn run(config: &Config) -> Result<Report, String> {
    let backend = HipBackend::connect().map_err(|error| format!("HIP connect failed: {error}"))?;
    let session = backend
        .open_execution_session(
            ExecutionSessionRequest::new(config.device_index, config.target.clone())
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| format!("session open failed: {error}"))?;
    let result: Result<(Vec<CaseEvidence>, Option<ContinuationEvidence>), String> = (|| {
        let queue = session
            .create_queue()
            .map_err(|error| format!("queue creation failed: {error}"))?;
        let cases = selected_case_tokens(config.phase12_subset)
            .iter()
            .copied()
            .map(|tokens| run_case(&session, &queue, tokens, &config.target))
            .collect::<Result<Vec<_>, _>>()?;
        let continuation =
            run_gfx942_mixed_provider_continuation(&session, &queue, &config.target)?;
        drop(queue);
        Ok((cases, continuation))
    })();
    let cleanup = match session.shutdown(Duration::from_secs(16)) {
        Ok(cleanup) => cleanup,
        Err(cleanup_error) => {
            return Err(match result {
                Ok(_) => format!("session shutdown failed: {cleanup_error}"),
                Err(case_error) => {
                    format!("{case_error}; session shutdown also failed: {cleanup_error}")
                }
            });
        }
    };
    let (cases, continuation) = result?;
    if cleanup.retryable_cleanup != 0 || cleanup.durable_quarantine != 0 {
        return Err("GDN cleanup did not return to zero".to_owned());
    }
    Ok(Report {
        schema_version: "linear-attention-g1-report-v1",
        state: "PASS",
        target: config.target.clone(),
        device_index: config.device_index,
        selected_backend: "hip",
        model_used: false,
        layout: [QK_HEADS, VALUE_HEADS, HEAD_DIM, CONV_KERNEL],
        cases,
        continuation,
        fallback_allowed: false,
        fallback_used: false,
        cpu_fallback_used: false,
        cleanup_retryable: cleanup.retryable_cleanup,
        cleanup_durable: cleanup.durable_quarantine,
    })
}

fn main() -> ExitCode {
    match parse_config().and_then(|config| run(&config)) {
        Ok(report) => match serde_json::to_string(&report) {
            Ok(json) => {
                println!("{json}");
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("linear-attention-g1 serialization failed: {error}");
                ExitCode::from(2)
            }
        },
        Err(error) => {
            eprintln!("linear-attention-g1: {error}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_free_layout_keeps_the_production_head_boundary() {
        let layout =
            LinearAttentionLayout::new(QK_HEADS, VALUE_HEADS, HEAD_DIM, CONV_KERNEL).unwrap();
        assert_eq!(layout.qkv_width(), QKV_WIDTH);
        assert_eq!(layout.output_width(), OUTPUT_WIDTH);
        assert_eq!(CASE_TOKENS, [1, 3, 17, 32, 127, 128, 129]);
    }

    #[test]
    fn phase12_subset_has_exact_token_membership_and_count() {
        assert_eq!(selected_case_tokens(true), &[1, 3, 17]);
        assert_eq!(selected_case_tokens(true).len(), 3);
        assert_eq!(selected_case_tokens(false), &CASE_TOKENS);
        assert_eq!(selected_case_tokens(false).len(), CASE_TOKENS.len());
    }

    #[test]
    fn parser_accepts_phase12_subset_and_defaults_to_full_matrix() {
        let full = parse_config_from(["--device-index", "0", "--target", "gfx942"]).unwrap();
        assert!(!full.phase12_subset);
        let subset = parse_config_from([
            "--device-index",
            "0",
            "--target",
            "gfx942",
            "--phase12-subset",
        ])
        .unwrap();
        assert!(subset.phase12_subset);
    }

    #[test]
    fn parser_rejects_duplicate_phase12_subset() {
        let error = parse_config_from([
            "--device-index",
            "0",
            "--target",
            "gfx942",
            "--phase12-subset",
            "--phase12-subset",
        ])
        .unwrap_err();
        assert_eq!(error, "duplicate --phase12-subset");
    }

    #[test]
    fn oracle_is_nonzero_and_stateful() {
        let (qkv, z, b, a, _, a_log, dt, norm) = inputs(3);
        let output = oracle(3, &qkv, &z, &b, &a, &a_log, &dt, &norm);
        assert_eq!(output.len(), 3 * OUTPUT_WIDTH);
        assert!(output.iter().any(|value| *value != 0));
        assert_ne!(
            &output[..OUTPUT_WIDTH],
            &output[OUTPUT_WIDTH..2 * OUTPUT_WIDTH]
        );
    }

    #[test]
    fn short_column_state_selector_covers_boundaries_and_guards() {
        let enabled = Some(std::ffi::OsStr::new("1"));
        let disabled = Some(std::ffi::OsStr::new("0"));
        let unknown = Some(std::ffi::OsStr::new("unknown"));
        for tokens in [16_usize, 17, 32, 33, 127, 128, 129] {
            assert_eq!(
                short_column_state_enabled("gfx1030", tokens, false, enabled),
                (17..128).contains(&tokens)
            );
            assert_eq!(
                short_column_state_enabled("gfx1030", tokens, false, None),
                (17..128).contains(&tokens)
            );
        }
        assert!(!short_column_state_enabled("gfx1030", 17, true, enabled));
        assert!(!short_column_state_enabled("gfx1030", 17, false, disabled));
        assert!(!short_column_state_enabled("gfx1030", 17, false, unknown));
        assert!(!short_column_state_enabled("gfx1201", 17, false, enabled));
        assert!(!short_column_state_enabled("unknown", 17, false, enabled));
        assert!(column_provider_enabled(
            "gfx1201", 128, false, disabled, None
        ));
        assert!(!column_provider_enabled(
            "gfx1201", 17, false, enabled, None
        ));
        assert!(column_provider_enabled("gfx1030", 17, false, None, None));
        assert!(!column_provider_enabled(
            "gfx1030", 128, true, enabled, None
        ));
    }

    #[test]
    fn gfx942_wave64_column_selector_requires_exact_suffix_and_opt_in() {
        let enabled = Some(std::ffi::OsStr::new("1"));
        let disabled = Some(std::ffi::OsStr::new("0"));
        for tokens in [127_usize, 128, 129] {
            assert_eq!(
                gfx942_wave64_column_state_enabled(
                    "gfx942:sramecc+:xnack-",
                    tokens,
                    false,
                    enabled,
                ),
                tokens >= 128
            );
        }
        for target in [
            "gfx942",
            "gfx942:sramecc-:xnack-",
            "gfx942:sramecc+:xnack+",
            "gfx1030",
            "gfx1201",
            "unknown",
        ] {
            assert!(!gfx942_wave64_column_state_enabled(
                target, 128, false, enabled,
            ));
        }
        assert!(!gfx942_wave64_column_state_enabled(
            "gfx942:sramecc+:xnack-",
            128,
            true,
            enabled,
        ));
        assert!(!gfx942_wave64_column_state_enabled(
            "gfx942:sramecc+:xnack-",
            128,
            false,
            None,
        ));
        assert!(!gfx942_wave64_column_state_enabled(
            "gfx942:sramecc+:xnack-",
            128,
            false,
            disabled,
        ));
    }
}
