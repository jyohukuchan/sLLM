//! Bounded C3b causal GQA full-attention evidence.
//!
//! The device operation is reached only through the public Rust execution
//! session and the request-local C3a2 append path. The scalar oracle is
//! independent of the HIP wrapper and compares the final BF16 words. This
//! runner has no CPU or generic fallback: unavailable hardware is not PASS.

use std::env;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;
use sllm_core::{
    AccessMode, Backend, DType, Encoding, ExecutionSession, ExecutionSessionRequest,
    ExecutionState, KvStateDescriptor, TensorView,
};
use sllm_hip::HipBackend;

const WAIT_TIMEOUT: Duration = Duration::from_secs(30);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(16);
const Q_HEADS: usize = 16;
const KV_HEADS: usize = 4;
const HEAD_DIM: usize = 256;
const WORKGROUP_SIZE: u32 = 256;
const KERNEL_ID: u32 = sllm_hip_sys::SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_STABLE_SOFTMAX_V1;
const KERNEL_SYMBOL: &str = "causal_attention.stable_softmax_gqa.v1";
const DEVICE_SYMBOL: &str = "sllm_causal_attention_stable_softmax_gqa_v1";

#[derive(Clone, Copy, Debug)]
struct Case {
    id: &'static str,
    m: usize,
    start_position: u64,
}

// Non-Cartesian coverage: prefill M boundaries plus decode prefixes. The
// prefill M=1/start=0 case is also the decode-prefix-zero boundary.
const CASES: [Case; 10] = [
    Case {
        id: "prefill-m1",
        m: 1,
        start_position: 0,
    },
    Case {
        id: "prefill-m3",
        m: 3,
        start_position: 0,
    },
    Case {
        id: "prefill-m17",
        m: 17,
        start_position: 0,
    },
    Case {
        id: "prefill-m255",
        m: 255,
        start_position: 0,
    },
    Case {
        id: "prefill-m256",
        m: 256,
        start_position: 0,
    },
    Case {
        id: "prefill-m257",
        m: 257,
        start_position: 0,
    },
    Case {
        id: "decode-prefix3",
        m: 1,
        start_position: 3,
    },
    Case {
        id: "decode-prefix255",
        m: 1,
        start_position: 255,
    },
    Case {
        id: "decode-prefix256",
        m: 1,
        start_position: 256,
    },
    Case {
        id: "decode-prefix257",
        m: 1,
        start_position: 257,
    },
];

#[derive(Debug)]
struct Config {
    device_index: u32,
    target: String,
}

#[derive(Debug, Serialize)]
struct CaseEvidence {
    id: &'static str,
    m: usize,
    start_position: u64,
    committed_kv_length: u64,
    numerical_match: bool,
    nonuniform_softmax_checked: bool,
    subnormal_score_contribution_checked: bool,
    causal_visibility_match: bool,
    gqa_mapping_match: bool,
    metadata_match: bool,
    no_fallback: bool,
}

#[derive(Debug, Serialize)]
struct OracleEvidence {
    scalar_ordered_dot_softmax_v: bool,
    fp16_subnormal_affects_score: bool,
    final_bf16_rne_checked: bool,
    gqa_heads_checked: bool,
}

#[derive(Debug, Serialize)]
struct CleanupEvidence {
    retryable_cleanup: usize,
    durable_quarantine: usize,
    terminal_zero: bool,
}

#[derive(Debug, Serialize)]
struct Report {
    schema_version: &'static str,
    state: &'static str,
    pass: bool,
    target: String,
    device_index: u32,
    selected_backend: &'static str,
    gpu_execution: bool,
    cpu_fallback_used: bool,
    fallback_allowed: bool,
    fallback_used: bool,
    cases: Vec<CaseEvidence>,
    oracle: OracleEvidence,
    cleanup: CleanupEvidence,
    error: Option<String>,
}

fn parse_config() -> Result<Config, String> {
    let mut device_index = None;
    let mut target = None;
    let mut arguments = env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--device-index" => {
                if device_index.is_some() {
                    return Err("duplicate --device-index".to_owned());
                }
                device_index = Some(
                    arguments
                        .next()
                        .ok_or_else(|| "--device-index requires a value".to_owned())?
                        .parse::<u32>()
                        .map_err(|_| "--device-index must be a u32".to_owned())?,
                );
            }
            "--target" => {
                if target.is_some() {
                    return Err("duplicate --target".to_owned());
                }
                let value = arguments
                    .next()
                    .ok_or_else(|| "--target requires a value".to_owned())?;
                if !matches!(value.as_str(), "gfx1030" | "gfx1201") {
                    return Err("--target must be gfx1030 or gfx1201".to_owned());
                }
                target = Some(value);
            }
            other => return Err(format!("unexpected argument {other}")),
        }
    }
    Ok(Config {
        device_index: device_index.ok_or_else(|| "missing --device-index".to_owned())?,
        target: target.ok_or_else(|| "missing --target".to_owned())?,
    })
}

fn words_to_bytes(words: &[u16]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

fn float_to_bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    if bits & 0x7f80_0000 == 0x7f80_0000 {
        if bits & 0x007f_ffff != 0 {
            let sign = ((bits >> 16) as u16) & 0x8000;
            let payload = ((bits >> 16) as u16) & 0x003f;
            return sign | 0x7fc0 | payload;
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

fn f16_to_f32(value: u16) -> f32 {
    let sign = u32::from(value & 0x8000) << 16;
    let exponent = u32::from((value >> 10) & 0x1f);
    let fraction = u32::from(value & 0x03ff);
    let bits = if exponent == 0 {
        if fraction == 0 {
            sign
        } else {
            let mut normalized = fraction;
            let mut shift = 0;
            while normalized & 0x0400 == 0 {
                normalized <<= 1;
                shift += 1;
            }
            sign | ((127 - 14 - shift) << 23) | ((normalized & 0x03ff) << 13)
        }
    } else if exponent == 0x1f {
        sign | 0x7f80_0000 | (fraction << 13)
    } else {
        sign | ((exponent + 112) << 23) | (fraction << 13)
    };
    f32::from_bits(bits)
}

fn input_q_words(m: usize) -> Vec<u16> {
    let mut words = vec![0_u16; m * Q_HEADS * HEAD_DIM];
    for row in 0..m {
        for head in 0..Q_HEADS {
            let offset = (row * Q_HEADS + head) * HEAD_DIM;
            words[offset] = float_to_bf16_rne(16.0);
            words[offset + 1] = float_to_bf16_rne(1.0 + (head % 4) as f32);
            // Keep the FP16-subnormal K lane numerically live. 2^20 times
            // the smallest FP16 subnormal survives the ordered F32 dot and
            // therefore tests more than merely executing the decoder path.
            words[offset + 2] = float_to_bf16_rne(1_048_576.0);
        }
    }
    words
}

fn input_k_words(token_count: usize, start_position: u64) -> Vec<u16> {
    let mut words = vec![0_u16; token_count * KV_HEADS * HEAD_DIM];
    for token in 0..token_count {
        let absolute = start_position + token as u64;
        for head in 0..KV_HEADS {
            let offset = (token * KV_HEADS + head) * HEAD_DIM;
            words[offset] = float_to_bf16_rne((absolute % 5) as f32 - 2.0);
            words[offset + 1] = float_to_bf16_rne(head as f32 * 0.25);
            // Exercise the corrected FP16 subnormal decoder on every device.
            words[offset + 2] = if (absolute + head as u64) % 2 == 0 {
                0x3380
            } else {
                0xb380
            };
        }
    }
    words
}

fn input_v_words(token_count: usize, seed: u64) -> Vec<u16> {
    let mut words = vec![0_u16; token_count * KV_HEADS * HEAD_DIM];
    for token in 0..token_count {
        for head in 0..KV_HEADS {
            let value = 1.0 + token as f32 + head as f32 + (seed % 2) as f32;
            let word = float_to_bf16_rne(value);
            for dimension in 0..HEAD_DIM {
                words[(token * KV_HEADS + head) * HEAD_DIM + dimension] = word;
            }
        }
    }
    words
}

fn make_binding(
    session: &ExecutionSession,
    buffer: &sllm_core::ExecutionBuffer,
    shape: &[usize],
    access: AccessMode,
) -> Result<sllm_core::OwnedTensorBinding, String> {
    let view = TensorView::with_encoding(DType::Bf16, Encoding::Unquantized, shape)
        .map_err(|error| format!("tensor view construction failed: {error}"))?;
    session
        .bind(buffer, view, access)
        .map_err(|error| format!("tensor binding failed: {error}"))
}

fn upload_words(
    session: &ExecutionSession,
    queue: &sllm_core::ExecutionQueue,
    buffer: &sllm_core::ExecutionBuffer,
    words: &[u16],
) -> Result<(), String> {
    let bytes = words_to_bytes(words);
    let range = buffer
        .range(0, bytes.len() as u64)
        .map_err(|error| format!("upload range construction failed: {error}"))?;
    let mut transfer = session
        .upload(queue, range, Arc::<[u8]>::from(bytes))
        .map_err(|error| format!("upload failed: {error}"))?;
    if transfer
        .wait(WAIT_TIMEOUT)
        .map_err(|error| format!("upload wait failed: {error}"))?
        != ExecutionState::Success
    {
        return Err("upload did not reach success".to_owned());
    }
    Ok(())
}

fn append_tokens(
    session: &ExecutionSession,
    queue: &sllm_core::ExecutionQueue,
    state: &sllm_core::KvState,
    key_words: &[u16],
    value_words: &[u16],
    expected_length: u64,
) -> Result<(), String> {
    let token_count = key_words.len() / (KV_HEADS * HEAD_DIM);
    if token_count == 0 || value_words.len() != key_words.len() {
        return Err("malformed non-empty KV append input".to_owned());
    }
    let bytes = std::mem::size_of_val(key_words) as u64;
    let key_buffer = session
        .allocate(bytes)
        .map_err(|error| format!("K allocation failed: {error}"))?;
    let value_buffer = session
        .allocate(bytes)
        .map_err(|error| format!("V allocation failed: {error}"))?;
    upload_words(session, queue, &key_buffer, key_words)?;
    upload_words(session, queue, &value_buffer, value_words)?;
    let shape = [token_count, KV_HEADS, HEAD_DIM];
    let key = make_binding(session, &key_buffer, &shape, AccessMode::Read)?;
    let value = make_binding(session, &value_buffer, &shape, AccessMode::Read)?;
    let mut append = session
        .append_kv_state(state, queue, key, value, expected_length, expected_length)
        .map_err(|error| format!("KV append failed: {error}"))?;
    if append
        .wait(WAIT_TIMEOUT)
        .map_err(|error| format!("KV append wait failed: {error}"))?
        != ExecutionState::Success
    {
        return Err("KV append did not reach success".to_owned());
    }
    drop(append);
    drop(key_buffer);
    drop(value_buffer);
    Ok(())
}

fn scalar_oracle(
    query_words: &[u16],
    key_words: &[u16],
    value_words: &[u16],
    m: usize,
    start_position: u64,
) -> Result<(Vec<u16>, bool, bool), String> {
    let expected_length = start_position
        .checked_add(m as u64)
        .ok_or_else(|| "oracle position overflow".to_owned())?;
    if query_words.len() != m * Q_HEADS * HEAD_DIM
        || key_words.len() != expected_length as usize * KV_HEADS * HEAD_DIM
        || value_words.len() != key_words.len()
    {
        return Err("oracle input lengths do not match the fixed contract".to_owned());
    }
    let mut output = vec![0_u16; query_words.len()];
    let mut nonuniform_softmax_checked = false;
    let mut subnormal_score_contribution_checked = false;
    for row in 0..m {
        let position = start_position + row as u64;
        for head in 0..Q_HEADS {
            let kv_head = head / 4;
            let query_offset = (row * Q_HEADS + head) * HEAD_DIM;
            let mut scores = Vec::with_capacity((position + 1) as usize);
            for key_position in 0..=position {
                let key_offset = (key_position as usize * KV_HEADS + kv_head) * HEAD_DIM;
                let mut dot = 0.0_f32;
                let mut dot_without_subnormal_lane = 0.0_f32;
                for dimension in 0..HEAD_DIM {
                    let product = bf16_to_f32(query_words[query_offset + dimension])
                        * f16_to_f32(sllm_hip::bf16_to_f16_bits(
                            key_words[key_offset + dimension],
                        ));
                    dot += product;
                    if dimension != 2 {
                        dot_without_subnormal_lane += product;
                    }
                }
                let score = dot * (1.0 / 16.0);
                let score_without_subnormal_lane = dot_without_subnormal_lane * (1.0 / 16.0);
                subnormal_score_contribution_checked |=
                    score.to_bits() != score_without_subnormal_lane.to_bits();
                scores.push(score);
            }
            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            nonuniform_softmax_checked |= scores
                .windows(2)
                .any(|pair| pair[0].to_bits() != pair[1].to_bits());
            let mut denominator = 0.0_f32;
            for score in &scores {
                denominator += (*score - maximum).exp();
            }
            for dimension in 0..HEAD_DIM {
                let mut accumulation = 0.0_f32;
                for (index, score) in scores.iter().enumerate() {
                    let value_offset = (index * KV_HEADS + kv_head) * HEAD_DIM + dimension;
                    let probability = (*score - maximum).exp() / denominator;
                    accumulation += probability
                        * f16_to_f32(sllm_hip::bf16_to_f16_bits(value_words[value_offset]));
                }
                output[query_offset + dimension] = float_to_bf16_rne(accumulation);
            }
        }
    }
    Ok((
        output,
        nonuniform_softmax_checked,
        subnormal_score_contribution_checked,
    ))
}

fn metadata_matches(
    dispatch: &sllm_core::DispatchEvidence,
    case: Case,
    expected_target: &str,
) -> bool {
    dispatch.abi_version == sllm_hip_sys::SLLM_HIP_ABI_VERSION
        && dispatch.info_version == sllm_hip_sys::SLLM_HIP_CAUSAL_ATTENTION_DISPATCH_INFO_VERSION
        && dispatch.dispatch_id != 0
        && dispatch.dispatch_count == 1
        && dispatch.kernel_id == KERNEL_ID
        && dispatch.workgroup_size_x == WORKGROUP_SIZE
        && dispatch.grid_size_x == (case.m * Q_HEADS) as u32
        && dispatch.row_count == case.m as u64
        && dispatch.normalized_size == HEAD_DIM as u64
        && dispatch.backend == sllm_hip_sys::SLLM_BACKEND_HIP
        && !dispatch.fallback_allowed
        && !dispatch.fallback_used
        && dispatch.kernel_symbol == KERNEL_SYMBOL
        && dispatch.device_symbol == DEVICE_SYMBOL
        && dispatch.target == expected_target
}

fn run_case(
    session: &ExecutionSession,
    queue: &sllm_core::ExecutionQueue,
    config: &Config,
    case: Case,
    seed: u64,
) -> Result<CaseEvidence, String> {
    let committed_length = case
        .start_position
        .checked_add(case.m as u64)
        .ok_or_else(|| "case committed length overflow".to_owned())?;
    let capacity = committed_length;
    let state = session
        .create_kv_state(
            KvStateDescriptor::new(seed as u32, capacity)
                .map_err(|error| format!("KV descriptor failed: {error}"))?,
        )
        .map_err(|error| format!("KV state creation failed: {error}"))?;
    let prefix_key = input_k_words(case.start_position as usize, 0);
    let prefix_value = input_v_words(case.start_position as usize, seed + 1);
    if case.start_position != 0 {
        append_tokens(session, queue, &state, &prefix_key, &prefix_value, 0)?;
    }
    let key_words = input_k_words(case.m, case.start_position);
    let value_words = input_v_words(case.m, seed + 2);
    append_tokens(
        session,
        queue,
        &state,
        &key_words,
        &value_words,
        case.start_position,
    )?;
    let snapshot = state
        .snapshot(session)
        .map_err(|error| format!("KV snapshot failed: {error}"))?;
    let query_words = input_q_words(case.m);
    let query_bytes = words_to_bytes(&query_words);
    let query_buffer = session
        .allocate(query_bytes.len() as u64)
        .map_err(|error| format!("Q allocation failed: {error}"))?;
    let output_buffer = session
        .allocate(query_bytes.len() as u64)
        .map_err(|error| format!("output allocation failed: {error}"))?;
    upload_words(session, queue, &query_buffer, &query_words)?;
    let shape = [case.m, Q_HEADS, HEAD_DIM];
    let query = make_binding(session, &query_buffer, &shape, AccessMode::Read)?;
    let output = make_binding(session, &output_buffer, &shape, AccessMode::Write)?;
    let descriptor = sllm_core::CausalAttentionDescriptor::new(
        case.start_position,
        case.m as u64,
        committed_length,
    )
    .map_err(|error| format!("causal descriptor failed: {error}"))?;
    let mut attention = session
        .causal_attention(&state, queue, query, output, descriptor)
        .map_err(|error| format!("causal attention submission failed: {error}"))?;
    let dispatch = attention.dispatch().clone();
    let metadata_match = metadata_matches(&dispatch, case, &config.target);
    if attention
        .wait(WAIT_TIMEOUT)
        .map_err(|error| format!("causal attention wait failed: {error}"))?
        != ExecutionState::Success
    {
        return Err("causal attention did not reach success".to_owned());
    }
    drop(attention);

    let (expected, nonuniform_softmax_checked, subnormal_score_contribution_checked) =
        scalar_oracle(
            &query_words,
            &[prefix_key.as_slice(), key_words.as_slice()].concat(),
            &[prefix_value.as_slice(), value_words.as_slice()].concat(),
            case.m,
            case.start_position,
        )?;
    let mut readback = session
        .readback(
            queue,
            output_buffer
                .range(0, query_bytes.len() as u64)
                .map_err(|error| format!("output readback range failed: {error}"))?,
        )
        .map_err(|error| format!("output readback failed: {error}"))?;
    if readback
        .wait(WAIT_TIMEOUT)
        .map_err(|error| format!("output readback wait failed: {error}"))?
        != ExecutionState::Success
    {
        return Err("output readback did not reach success".to_owned());
    }
    let mut actual_bytes = vec![0_u8; query_bytes.len()];
    readback
        .read_into(&mut actual_bytes)
        .map_err(|error| format!("output readback read failed: {error}"))?;
    let actual = actual_bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect::<Vec<_>>();
    let numerical_match = actual == expected;
    let causal_visibility_match = snapshot.length() == committed_length;
    let gqa_mapping_match = actual
        .chunks_exact(HEAD_DIM)
        .step_by(Q_HEADS)
        .zip(actual.chunks_exact(HEAD_DIM).skip(4).step_by(Q_HEADS))
        .any(|(left, right)| left != right);
    drop(readback);
    drop(state);
    drop(query_buffer);
    drop(output_buffer);
    Ok(CaseEvidence {
        id: case.id,
        m: case.m,
        start_position: case.start_position,
        committed_kv_length: committed_length,
        numerical_match,
        nonuniform_softmax_checked,
        subnormal_score_contribution_checked,
        causal_visibility_match,
        gqa_mapping_match,
        metadata_match,
        no_fallback: !dispatch.fallback_allowed && !dispatch.fallback_used,
    })
}

fn unavailable_report(config: &Config, error: String) -> Report {
    Report {
        schema_version: "sllm-full-attention-g1-evidence-v1",
        state: "UNAVAILABLE",
        pass: false,
        target: config.target.clone(),
        device_index: config.device_index,
        selected_backend: "hip",
        gpu_execution: false,
        cpu_fallback_used: false,
        fallback_allowed: false,
        fallback_used: false,
        cases: Vec::new(),
        oracle: OracleEvidence {
            scalar_ordered_dot_softmax_v: false,
            fp16_subnormal_affects_score: false,
            final_bf16_rne_checked: false,
            gqa_heads_checked: false,
        },
        cleanup: CleanupEvidence {
            retryable_cleanup: 0,
            durable_quarantine: 0,
            terminal_zero: false,
        },
        error: Some(error),
    }
}

fn run(config: &Config) -> Report {
    let backend = match HipBackend::connect() {
        Ok(backend) => backend,
        Err(error) => return unavailable_report(config, format!("HIP connect failed: {error}")),
    };
    let request = match ExecutionSessionRequest::new(config.device_index, config.target.clone()) {
        Ok(request) => request,
        Err(error) => return unavailable_report(config, error.to_string()),
    };
    let session = match backend.open_execution_session(request) {
        Ok(session) => session,
        Err(error) => {
            return unavailable_report(config, format!("execution-session open failed: {error}"));
        }
    };
    let queue = match session.create_queue() {
        Ok(queue) => queue,
        Err(error) => {
            let _ = session.shutdown(SHUTDOWN_TIMEOUT);
            return unavailable_report(config, format!("queue creation failed: {error}"));
        }
    };
    let mut cases = Vec::new();
    let operation = (|| {
        for (index, case) in CASES.into_iter().enumerate() {
            cases.push(run_case(&session, &queue, config, case, index as u64)?);
        }
        Ok::<(), String>(())
    })();
    drop(queue);
    let cleanup = session.shutdown(SHUTDOWN_TIMEOUT);
    match (operation, cleanup) {
        (Ok(()), Ok(cleanup)) => {
            let oracle = OracleEvidence {
                scalar_ordered_dot_softmax_v: cases
                    .iter()
                    .any(|case| case.nonuniform_softmax_checked && case.numerical_match),
                fp16_subnormal_affects_score: cases
                    .iter()
                    .all(|case| case.subnormal_score_contribution_checked),
                final_bf16_rne_checked: cases.iter().all(|case| case.numerical_match),
                gqa_heads_checked: cases.iter().all(|case| case.gqa_mapping_match),
            };
            let all_cases = cases.iter().all(|case| {
                case.numerical_match
                    && case.causal_visibility_match
                    && case.gqa_mapping_match
                    && case.metadata_match
                    && case.no_fallback
            });
            let pass = all_cases
                && oracle.scalar_ordered_dot_softmax_v
                && oracle.fp16_subnormal_affects_score
                && oracle.final_bf16_rne_checked
                && oracle.gqa_heads_checked
                && cleanup.retryable_cleanup == 0
                && cleanup.durable_quarantine == 0;
            Report {
                schema_version: "sllm-full-attention-g1-evidence-v1",
                state: if pass { "PASS" } else { "INCOMPLETE" },
                pass,
                target: config.target.clone(),
                device_index: config.device_index,
                selected_backend: "hip",
                gpu_execution: true,
                cpu_fallback_used: false,
                fallback_allowed: false,
                fallback_used: false,
                cases,
                oracle,
                cleanup: CleanupEvidence {
                    retryable_cleanup: cleanup.retryable_cleanup,
                    durable_quarantine: cleanup.durable_quarantine,
                    terminal_zero: cleanup.retryable_cleanup == 0
                        && cleanup.durable_quarantine == 0,
                },
                error: None,
            }
        }
        (Err(error), Ok(cleanup)) => {
            let mut report = unavailable_report(config, error);
            report.state = "FAIL";
            report.cases = cases;
            report.cleanup = CleanupEvidence {
                retryable_cleanup: cleanup.retryable_cleanup,
                durable_quarantine: cleanup.durable_quarantine,
                terminal_zero: cleanup.retryable_cleanup == 0 && cleanup.durable_quarantine == 0,
            };
            report
        }
        (operation, cleanup) => unavailable_report(
            config,
            format!("operation={operation:?}; cleanup={cleanup:?}"),
        ),
    }
}

fn main() -> ExitCode {
    match parse_config() {
        Ok(config) => {
            let report = run(&config);
            match serde_json::to_string(&report) {
                Ok(output) => {
                    println!("{output}");
                    if report.pass {
                        ExitCode::SUCCESS
                    } else {
                        ExitCode::from(1)
                    }
                }
                Err(error) => {
                    eprintln!(
                        "sllm-full-attention-g1-evidence: report serialization failed: {error}"
                    );
                    ExitCode::from(2)
                }
            }
        }
        Err(error) => {
            eprintln!("sllm-full-attention-g1-evidence: {error}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_oracle_covers_causal_visibility_and_gqa_mapping() {
        let query = input_q_words(2);
        let keys = input_k_words(3, 0);
        let values = input_v_words(3, 0);
        let (output, nonuniform, subnormal_score_contribution) =
            scalar_oracle(&query, &keys, &values, 2, 1).unwrap();
        // For row zero only absolute key 0/1 are visible; head four maps to
        // KV head one, so both causal visibility and GQA affect the result.
        assert_ne!(output[0], output[4 * HEAD_DIM]);
        assert_ne!(output[0], output[Q_HEADS * HEAD_DIM]);
        assert_eq!(output.len(), query.len());
        assert!(nonuniform);
        assert!(subnormal_score_contribution);
    }

    #[test]
    fn boundary_case_set_is_bounded_and_non_cartesian() {
        assert_eq!(CASES.len(), 10);
        assert!(CASES.iter().any(|case| case.m == 257));
        assert!(CASES.iter().any(|case| case.start_position == 257));
        assert!(CASES.iter().all(|case| case.m > 0));
    }

    #[test]
    fn bf16_rne_preserves_specials_and_ties() {
        assert_eq!(float_to_bf16_rne(f32::from_bits(0x3f80_8000)), 0x3f80);
        assert_eq!(float_to_bf16_rne(f32::from_bits(0x3f81_8000)), 0x3f82);
        assert_eq!(float_to_bf16_rne(f32::INFINITY), 0x7f80);
        assert_eq!(float_to_bf16_rne(f32::NEG_INFINITY), 0xff80);
        assert_eq!(float_to_bf16_rne(f32::from_bits(0x7fc1_2345)), 0x7fc1);
    }

    #[test]
    fn fp16_decoder_preserves_subnormals_zero_and_specials() {
        assert_eq!(f16_to_f32(0x0000).to_bits(), 0x0000_0000);
        assert_eq!(f16_to_f32(0x8000).to_bits(), 0x8000_0000);
        assert_eq!(f16_to_f32(0x0001).to_bits(), 0x3380_0000);
        assert_eq!(f16_to_f32(0x8001).to_bits(), 0xb380_0000);
        assert_eq!(f16_to_f32(0x03ff).to_bits(), 0x387f_c000);
        assert_eq!(f16_to_f32(0x83ff).to_bits(), 0xb87f_c000);
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert_eq!(f16_to_f32(0xfc00), f32::NEG_INFINITY);
        assert!(f16_to_f32(0x7e01).is_nan());
    }
}
