//! Bounded GQA6 FP16-KV causal-attention evidence and benchmark.
//!
//! This runner uses the public Rust execution session and the real KV-state
//! append path.  Inputs are uploaded as BF16, so the oracle deliberately
//! models BF16 -> FP16 KV storage -> F32 attention arithmetic.  The
//! modes compare the generic route, the existing GQA6 qtile1 control, the
//! opt-in Q_TILE4 FP16-KV candidates with K tiles 4, 8, 16, and 32, the
//! blockwise online-softmax candidates (including the gfx1201-only QTile8
//! variant), and the two-stage P32/P64 shared-KV
//! decode candidates.

use std::env;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;
use sha2::{Digest, Sha256};
use sllm_core::{
    AccessMode, Backend, DType, Encoding, ExecutionSession, ExecutionSessionRequest,
    ExecutionState, KvCacheEncoding, KvStateDescriptor, TensorView,
};
use sllm_hip::HipBackend;

const WAIT_TIMEOUT: Duration = Duration::from_secs(30);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(16);
const Q_HEADS: usize = 24;
const KV_HEADS: usize = 4;
const GQA_RATIO: usize = 6;
const HEAD_DIM: usize = 256;
const WORKGROUP_SIZE: u32 = 256;
const TIMING_WARMUPS: usize = 5;
const TIMING_MEASURED: usize = 21;

const KERNEL_BASELINE: &str = "causal_attention.online_softmax_gqa.v2";
const DEVICE_BASELINE: &str = "sllm_causal_attention_online_softmax_gqa_v2";
const KERNEL_WAVE1201: &str = "causal_attention.online_softmax_gqa.gfx1201_wave.v4";
const DEVICE_WAVE1201: &str = "sllm_causal_attention_gfx1201_wave_v4";
const KERNEL_QTILE1: &str = "causal_attention.prefill.gqa6_qtile4.v1";
const DEVICE_QTILE1: &str = "sllm_causal_attention_prefill_gqa6_qtile4_v1";
const KERNEL_K4: &str = "causal_attention.prefill.gqa6_qtile4_k4.fp16.v1";
const DEVICE_K4: &str = "sllm_causal_attention_prefill_gqa6_qtile4_k4_fp16_v1";
const KERNEL_K8: &str = "causal_attention.prefill.gqa6_qtile4_k8.fp16.v1";
const DEVICE_K8: &str = "sllm_causal_attention_prefill_gqa6_qtile4_k8_fp16_v1";
const KERNEL_K16: &str = "causal_attention.prefill.gqa6_qtile4_k16.fp16.v1";
const DEVICE_K16: &str = "sllm_causal_attention_prefill_gqa6_qtile4_k16_fp16_v1";
const KERNEL_K32: &str = "causal_attention.prefill.gqa6_qtile4_k32.fp16.v1";
const DEVICE_K32: &str = "sllm_causal_attention_prefill_gqa6_qtile4_k32_fp16_v1";
const KERNEL_BLOCKSOFTMAX: &str = "causal_attention.prefill.gqa6_blocksoftmax.fp16.v1";
const DEVICE_BLOCKSOFTMAX: &str = "sllm_causal_attention_prefill_gqa6_blocksoftmax_fp16_v1";
const KERNEL_BLOCKSOFTMAX_Q8: &str = "causal_attention.prefill.gqa6_blocksoftmax_q8.fp16.v1";
const DEVICE_BLOCKSOFTMAX_Q8: &str = "sllm_causal_attention_prefill_gqa6_blocksoftmax_q8_fp16_v1";
const KERNEL_DECODE_WAVE: &str = "causal_attention.decode.wave8_split.v5";
const DEVICE_DECODE_WAVE: &str = "sllm_causal_attention_decode_wave8_split_v5";
const KERNEL_DECODE_WAVE_Q_PRELOAD: &str = "causal_attention.decode.wave8_split.q_preload.v1";
const DEVICE_DECODE_WAVE_Q_PRELOAD: &str = "sllm_causal_attention_decode_wave8_split_q_preload_v1";
const KERNEL_DECODE_GQA6_P32: &str = "causal_attention.decode.gqa6_split_p32.fp16.v1";
const DEVICE_DECODE_GQA6_P32: &str = "sllm_causal_attention_decode_gqa6_split_p32_v1";
const KERNEL_DECODE_GQA6_P64: &str = "causal_attention.decode.gqa6_split_p64.fp16.v1";
const DEVICE_DECODE_GQA6_P64: &str = "sllm_causal_attention_decode_gqa6_split_p64_v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    Baseline,
    QTile1,
    K4,
    K8,
    K16,
    K32,
    BlockSoftmax,
    BlockSoftmaxQ8,
    DecodeP32,
    DecodeP64,
}

impl Mode {
    const fn name(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::QTile1 => "qtile1-control",
            Self::K4 => "k4-candidate",
            Self::K8 => "k8-candidate",
            Self::K16 => "k16-candidate",
            Self::K32 => "k32-candidate",
            Self::BlockSoftmax => "blocksoftmax-candidate",
            Self::BlockSoftmaxQ8 => "blocksoftmax-q8-candidate",
            Self::DecodeP32 => "decode-p32-candidate",
            Self::DecodeP64 => "decode-p64-candidate",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Case {
    id: &'static str,
    m: usize,
    start_position: u64,
}

const CASES: [Case; 15] = [
    Case {
        id: "prefill-m3",
        m: 3,
        start_position: 0,
    },
    Case {
        id: "prefill-m4",
        m: 4,
        start_position: 0,
    },
    Case {
        id: "prefill-m5",
        m: 5,
        start_position: 0,
    },
    Case {
        id: "prefill-m127",
        m: 127,
        start_position: 0,
    },
    Case {
        id: "prefill-m128",
        m: 128,
        start_position: 0,
    },
    Case {
        id: "prefill-m129",
        m: 129,
        start_position: 0,
    },
    Case {
        id: "prefill-m219",
        m: 219,
        start_position: 0,
    },
    Case {
        id: "v620-last-full-context9216",
        m: 512,
        start_position: 8_704,
    },
    Case {
        id: "r9700-last-full-context9216",
        m: 1_024,
        start_position: 8_192,
    },
    Case {
        id: "context31",
        m: 1,
        start_position: 30,
    },
    Case {
        id: "context32",
        m: 1,
        start_position: 31,
    },
    Case {
        id: "context33",
        m: 1,
        start_position: 32,
    },
    Case {
        id: "decode-context4096",
        m: 1,
        start_position: 4_095,
    },
    Case {
        id: "decode-context9435",
        m: 1,
        start_position: 9_434,
    },
    Case {
        id: "long-tail-context9435",
        m: 219,
        start_position: 9_216,
    },
];

#[derive(Debug)]
struct Config {
    device_index: u32,
    target: String,
    mode: Mode,
    benchmark: bool,
    case_filter: Option<String>,
    benchmark_filter: Option<String>,
}

#[derive(Debug, Serialize)]
struct CaseEvidence {
    id: &'static str,
    m: usize,
    start_position: u64,
    committed_kv_length: u64,
    numerical_match: bool,
    oracle_scope: &'static str,
    oracle_rows: Vec<usize>,
    max_abs_error: f64,
    max_relative_error: f64,
    max_bf16_ulp: u32,
    bf16_ulp_mismatch_count: u64,
    nonuniform_softmax_checked: bool,
    bf16_to_fp16_to_f32_checked: bool,
    causal_visibility_match: bool,
    gqa_mapping_match: bool,
    metadata_match: bool,
    hip_execution: bool,
    no_fallback: bool,
    output_bytes_sha256: String,
    timing_warmups: usize,
    timing_samples_ns: Vec<u64>,
    timing_median_ns: Option<u64>,
    timing_mad_ns: Option<u64>,
}

#[derive(Debug, Serialize)]
struct OracleEvidence {
    scalar_ordered_dot_softmax_v: bool,
    all_24_query_heads_checked: bool,
    bf16_kv_roundtrip_checked: bool,
    final_bf16_rne_checked: bool,
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
    mode: &'static str,
    target: String,
    device_index: u32,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    kv_storage: &'static str,
    selected_backend: &'static str,
    gpu_execution: bool,
    cpu_fallback_used: bool,
    fallback_allowed: bool,
    fallback_used: bool,
    benchmark_warmups: usize,
    benchmark_measured: usize,
    cases: Vec<CaseEvidence>,
    oracle: OracleEvidence,
    cleanup: CleanupEvidence,
    error: Option<String>,
}

fn parse_config() -> Result<Config, String> {
    let mut device_index = None;
    let mut target = None;
    let mut mode = None;
    let mut benchmark = false;
    let mut case_filter = None;
    let mut benchmark_filter = None;
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
            "--mode" => {
                if mode.is_some() {
                    return Err("duplicate --mode".to_owned());
                }
                mode = Some(
                    match arguments
                        .next()
                        .ok_or_else(|| {
                            "--mode requires baseline, qtile1, k4, k8, k16, k32, blocksoftmax, blocksoftmax-q8, decode-p32, or decode-p64"
                                .to_owned()
                        })?
                        .as_str()
                    {
                        "baseline" => Mode::Baseline,
                        "qtile1" | "qtile1-control" => Mode::QTile1,
                        "k4" | "k4-candidate" => Mode::K4,
                        "k8" | "k8-candidate" => Mode::K8,
                        "k16" | "k16-candidate" => Mode::K16,
                        "k32" | "k32-candidate" | "candidate" => Mode::K32,
                        "blocksoftmax" | "blocksoftmax-candidate" => Mode::BlockSoftmax,
                        "blocksoftmax-q8" | "blocksoftmax-q8-candidate" => Mode::BlockSoftmaxQ8,
                        "decode-p32" | "decode-p32-candidate" => Mode::DecodeP32,
                        "decode-p64" | "decode-p64-candidate" => Mode::DecodeP64,
                        _ => {
                            return Err(
                                "--mode must be baseline, qtile1, k4, k8, k16, k32, blocksoftmax, blocksoftmax-q8, decode-p32, or decode-p64"
                                    .to_owned(),
                            );
                        }
                    },
                );
            }
            "--benchmark" => benchmark = true,
            "--cases" => {
                if case_filter.is_some() {
                    return Err("duplicate --cases".to_owned());
                }
                case_filter = Some(
                    arguments
                        .next()
                        .ok_or_else(|| "--cases requires a comma-separated value".to_owned())?,
                );
            }
            "--benchmark-cases" => {
                if benchmark_filter.is_some() {
                    return Err("duplicate --benchmark-cases".to_owned());
                }
                benchmark_filter = Some(arguments.next().ok_or_else(|| {
                    "--benchmark-cases requires a comma-separated value".to_owned()
                })?);
            }
            other => return Err(format!("unexpected argument {other}")),
        }
    }
    let mode = mode
        .or_else(
            || match env::var("SLLM_GQA6_ATTENTION_MODE").ok()?.as_str() {
                "baseline" => Some(Mode::Baseline),
                "qtile1" | "qtile1-control" => Some(Mode::QTile1),
                "k4" | "k4-candidate" => Some(Mode::K4),
                "k8" | "k8-candidate" => Some(Mode::K8),
                "k16" | "k16-candidate" => Some(Mode::K16),
                "k32" | "k32-candidate" | "candidate" => Some(Mode::K32),
                "blocksoftmax" | "blocksoftmax-candidate" => Some(Mode::BlockSoftmax),
                "blocksoftmax-q8" | "blocksoftmax-q8-candidate" => Some(Mode::BlockSoftmaxQ8),
                "decode-p32" | "decode-p32-candidate" => Some(Mode::DecodeP32),
                "decode-p64" | "decode-p64-candidate" => Some(Mode::DecodeP64),
                _ => None,
            },
        )
        .ok_or_else(|| "missing --mode or SLLM_GQA6_ATTENTION_MODE".to_owned())?;
    Ok(Config {
        device_index: device_index.ok_or_else(|| "missing --device-index".to_owned())?,
        target: target.ok_or_else(|| "missing --target".to_owned())?,
        mode,
        benchmark,
        case_filter: case_filter.or_else(|| env::var("SLLM_GQA6_ATTENTION_CASES").ok()),
        benchmark_filter: benchmark_filter
            .or_else(|| env::var("SLLM_GQA6_ATTENTION_BENCHMARK_CASES").ok()),
    })
}

fn selected_cases(filter: Option<&str>) -> Result<Vec<Case>, String> {
    let Some(filter) = filter else {
        return Ok(CASES.to_vec());
    };
    let requested = filter.split(',').map(str::trim).filter(|id| !id.is_empty());
    let mut selected = Vec::new();
    for id in requested {
        let case = CASES
            .iter()
            .find(|case| case.id == id)
            .ok_or_else(|| format!("unknown case id {id}"))?;
        if !selected.contains(case) {
            selected.push(*case);
        }
    }
    if selected.is_empty() {
        return Err("case filter selected no cases".to_owned());
    }
    Ok(selected)
}

fn benchmark_case(case: Case, filter: Option<&str>) -> Result<bool, String> {
    let Some(filter) = filter else {
        return Ok(matches!(case.id, "prefill-m128" | "prefill-m219"));
    };
    let requested = filter.split(',').map(str::trim).filter(|id| !id.is_empty());
    let mut found = false;
    for id in requested {
        found = true;
        if !CASES.iter().any(|known| known.id == id) {
            return Err(format!("unknown benchmark case id {id}"));
        }
        if id == case.id {
            return Ok(true);
        }
    }
    if !found {
        return Err("benchmark case filter selected no cases".to_owned());
    }
    Ok(false)
}

fn configure_mode(mode: Mode, target: &str) {
    const FORCE: &str = "SLLM_CAUSAL_ATTENTION_FORCE_BASELINE";
    const QTILE1: &str = "SLLM_CAUSAL_ATTENTION_GQA6_QTILE4";
    const K4: &str = "SLLM_CAUSAL_ATTENTION_GQA6_QTILE4_K4_FP16";
    const K8: &str = "SLLM_CAUSAL_ATTENTION_GQA6_QTILE4_K8_FP16";
    const K16: &str = "SLLM_CAUSAL_ATTENTION_GQA6_QTILE4_K16_FP16";
    const K32: &str = "SLLM_CAUSAL_ATTENTION_GQA6_QTILE4_K32_FP16";
    const DECODE_P32: &str = "SLLM_CAUSAL_ATTENTION_GQA6_DECODE_SPLIT_P32";
    const DECODE_P64: &str = "SLLM_CAUSAL_ATTENTION_GQA6_DECODE_SPLIT_P64";
    const BLOCKSOFTMAX_GFX1030: &str = "SLLM_CAUSAL_ATTENTION_GQA6_PREFILL_BLOCKSOFTMAX_GFX1030";
    const BLOCKSOFTMAX_GFX1201: &str = "SLLM_CAUSAL_ATTENTION_GQA6_PREFILL_BLOCKSOFTMAX_GFX1201";
    const BLOCKSOFTMAX_Q8_GFX1201: &str =
        "SLLM_CAUSAL_ATTENTION_GQA6_PREFILL_BLOCKSOFTMAX_Q8_GFX1201";
    // This binary configures its process environment before opening the
    // execution session; no worker threads are started by this function.
    unsafe {
        env::remove_var(FORCE);
        env::remove_var(QTILE1);
        env::remove_var(K4);
        env::remove_var(K8);
        env::remove_var(K16);
        env::remove_var(K32);
        env::remove_var(DECODE_P32);
        env::remove_var(DECODE_P64);
        env::remove_var(BLOCKSOFTMAX_GFX1030);
        env::remove_var(BLOCKSOFTMAX_GFX1201);
        env::remove_var(BLOCKSOFTMAX_Q8_GFX1201);
        match mode {
            Mode::Baseline => {
                env::set_var(FORCE, "1");
            }
            Mode::QTile1 => {
                env::set_var(QTILE1, "1");
            }
            Mode::K4 => {
                env::set_var(K4, "1");
            }
            Mode::K8 => {
                env::set_var(K8, "1");
            }
            Mode::K16 => {
                env::set_var(K16, "1");
            }
            Mode::K32 => {
                env::set_var(K32, "1");
            }
            Mode::DecodeP32 => {
                env::set_var(DECODE_P32, "1");
            }
            Mode::DecodeP64 => {
                env::set_var(DECODE_P64, "1");
            }
            Mode::BlockSoftmax => {
                let variable = match target {
                    "gfx1030" => BLOCKSOFTMAX_GFX1030,
                    "gfx1201" => BLOCKSOFTMAX_GFX1201,
                    _ => return,
                };
                env::set_var(variable, "1");
            }
            Mode::BlockSoftmaxQ8 => {
                if target == "gfx1201" {
                    env::set_var(BLOCKSOFTMAX_Q8_GFX1201, "1");
                }
            }
        }
    }
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

fn f64_to_bf16_rne(value: f64) -> u16 {
    if value.is_nan() {
        return if value.is_sign_negative() {
            0xffc0
        } else {
            0x7fc0
        };
    }
    if value == f64::INFINITY {
        return 0x7f80;
    }
    if value == f64::NEG_INFINITY {
        return 0xff80;
    }
    if value == 0.0 {
        return if value.is_sign_negative() { 0x8000 } else { 0 };
    }
    float_to_bf16_rne(value as f32)
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

fn input_query_words(m: usize) -> Vec<u16> {
    let mut words = vec![0_u16; m * Q_HEADS * HEAD_DIM];
    for row in 0..m {
        for head in 0..Q_HEADS {
            for dimension in 0..HEAD_DIM {
                let code = ((row * 17 + head * 31 + dimension * 7) % 97) as i32 - 48;
                let value = code as f32 / 64.0 + (head % GQA_RATIO) as f32 / 32.0;
                words[(row * Q_HEADS + head) * HEAD_DIM + dimension] = float_to_bf16_rne(value);
            }
        }
    }
    words
}

fn input_key_words(token_count: usize, start_position: u64) -> Vec<u16> {
    let mut words = vec![0_u16; token_count * KV_HEADS * HEAD_DIM];
    for token in 0..token_count {
        let absolute = start_position + token as u64;
        for head in 0..KV_HEADS {
            for dimension in 0..HEAD_DIM {
                let code =
                    ((absolute * 13 + head as u64 * 23 + dimension as u64 * 7) % 89) as i32 - 44;
                let value = code as f32 / 64.0 + head as f32 / 16.0;
                words[(token * KV_HEADS + head) * HEAD_DIM + dimension] = float_to_bf16_rne(value);
            }
        }
    }
    // 2^-25 is representable in BF16 but rounds to zero in FP16. Keep one
    // numerically negligible fixture element so the evidence proves that the
    // actual BF16 -> FP16 KV conversion, rather than a BF16 mirror, was used.
    if let Some(first) = words.first_mut() {
        *first = 0x3300;
    }
    words
}

fn input_value_words(token_count: usize, start_position: u64) -> Vec<u16> {
    let mut words = vec![0_u16; token_count * KV_HEADS * HEAD_DIM];
    for token in 0..token_count {
        let absolute = start_position + token as u64;
        for head in 0..KV_HEADS {
            for dimension in 0..HEAD_DIM {
                let code =
                    ((absolute * 29 + head as u64 * 37 + dimension as u64 * 11) % 113) as i32 - 56;
                let value = code as f32 / 32.0 + head as f32 * 0.5;
                words[(token * KV_HEADS + head) * HEAD_DIM + dimension] = float_to_bf16_rne(value);
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

fn oracle_rows(case: Case) -> Vec<usize> {
    let work = case
        .m
        .saturating_mul(case.start_position as usize + case.m)
        .saturating_mul(Q_HEADS)
        .saturating_mul(HEAD_DIM);
    if case.m == 1 {
        vec![0]
    } else if work > 50_000_000 {
        vec![0, case.m / 2, case.m - 1]
    } else {
        (0..case.m).collect()
    }
}

fn bf16_ulp_distance(observed: u16, expected: u16) -> u32 {
    fn ordered(value: u16) -> i32 {
        if value & 0x8000 != 0 {
            i32::from(!value)
        } else {
            i32::from(value | 0x8000)
        }
    }
    ordered(observed).abs_diff(ordered(expected))
}

fn scalar_oracle(
    query_words: &[u16],
    key_words: &[u16],
    value_words: &[u16],
    case: Case,
    rows: &[usize],
) -> Result<AttentionProbeOutputs, String> {
    let committed = case
        .start_position
        .checked_add(case.m as u64)
        .ok_or_else(|| "oracle position overflow".to_owned())?;
    if query_words.len() != case.m * Q_HEADS * HEAD_DIM
        || key_words.len() != committed as usize * KV_HEADS * HEAD_DIM
        || value_words.len() != key_words.len()
    {
        return Err("oracle input lengths do not match fixed contract".to_owned());
    }
    let mut output_rows = Vec::with_capacity(rows.len());
    let mut nonuniform = false;
    let bf16_roundtrip_checked = key_words
        .iter()
        .chain(value_words.iter())
        .any(|&word| bf16_to_f32(word) != f16_to_f32(sllm_hip::bf16_to_f16_bits(word)));
    for &row in rows {
        if row >= case.m {
            return Err("oracle row out of bounds".to_owned());
        }
        let position = case.start_position + row as u64;
        let mut output = vec![0_u16; Q_HEADS * HEAD_DIM];
        for head in 0..Q_HEADS {
            let kv_head = head / GQA_RATIO;
            let query_offset = (row * Q_HEADS + head) * HEAD_DIM;
            let mut scores = Vec::with_capacity((position + 1) as usize);
            for key_position in 0..=position {
                let key_offset = (key_position as usize * KV_HEADS + kv_head) * HEAD_DIM;
                let mut dot = 0.0_f64;
                for dimension in 0..HEAD_DIM {
                    let query = f64::from(bf16_to_f32(query_words[query_offset + dimension]));
                    let key = f64::from(f16_to_f32(sllm_hip::bf16_to_f16_bits(
                        key_words[key_offset + dimension],
                    )));
                    dot += query * key;
                }
                scores.push(dot * (1.0 / 16.0));
            }
            nonuniform |= scores
                .windows(2)
                .any(|pair| pair[0].to_bits() != pair[1].to_bits());
            let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let denominator: f64 = scores.iter().map(|score| (*score - maximum).exp()).sum();
            for dimension in 0..HEAD_DIM {
                let mut accumulation = 0.0_f64;
                for (index, score) in scores.iter().enumerate() {
                    let value_offset = (index * KV_HEADS + kv_head) * HEAD_DIM + dimension;
                    let value = f64::from(f16_to_f32(sllm_hip::bf16_to_f16_bits(
                        value_words[value_offset],
                    )));
                    accumulation += (*score - maximum).exp() / denominator * value;
                }
                output[head * HEAD_DIM + dimension] = f64_to_bf16_rne(accumulation);
            }
        }
        output_rows.push((row, output));
    }
    Ok((output_rows, nonuniform, bf16_roundtrip_checked))
}

fn compare_rows(
    actual: &[u16],
    expected_rows: &[(usize, Vec<u16>)],
    case: Case,
) -> (bool, f64, f64, u32, u64) {
    let mut matches = true;
    let mut max_abs_error = 0.0_f64;
    let mut max_relative_error = 0.0_f64;
    let mut max_ulp = 0_u32;
    let mut ulp_mismatches = 0_u64;
    for &(row, ref expected) in expected_rows {
        let begin = row * Q_HEADS * HEAD_DIM;
        let end = begin + Q_HEADS * HEAD_DIM;
        let Some(observed) = actual.get(begin..end) else {
            return (false, f64::INFINITY, f64::INFINITY, u32::MAX, u64::MAX);
        };
        for (&observed, &reference) in observed.iter().zip(expected) {
            let observed_f32 = f64::from(bf16_to_f32(observed));
            let reference_f32 = f64::from(bf16_to_f32(reference));
            let error = (observed_f32 - reference_f32).abs();
            let relative_error = error / reference_f32.abs().max(1.0e-12);
            let ulp = bf16_ulp_distance(observed, reference);
            max_abs_error = max_abs_error.max(error);
            max_relative_error = max_relative_error.max(relative_error);
            max_ulp = max_ulp.max(ulp);
            if ulp != 0 {
                ulp_mismatches += 1;
            }
            matches &= error <= 0.016;
        }
    }
    let _ = case;
    (
        matches,
        max_abs_error,
        max_relative_error,
        max_ulp,
        ulp_mismatches,
    )
}

fn expected_kernel(config: &Config, case: Case) -> (&'static str, &'static str, u32, u32, u32) {
    let committed = case.start_position + case.m as u64;
    let p64_context = (config.target == "gfx1030" && committed >= 8192)
        || (config.target == "gfx1201" && committed >= 4096);
    if case.m == 1 && p64_context && config.mode == Mode::DecodeP64 {
        return (KERNEL_DECODE_GQA6_P64, DEVICE_DECODE_GQA6_P64, 256, 2, 192);
    }
    if case.m == 1 && committed >= 4096 && config.mode == Mode::DecodeP32 {
        return (KERNEL_DECODE_GQA6_P32, DEVICE_DECODE_GQA6_P32, 128, 2, 192);
    }
    if case.m == 1 && committed >= 1024 {
        if config.target == "gfx1030" {
            return (
                KERNEL_DECODE_WAVE_Q_PRELOAD,
                DEVICE_DECODE_WAVE_Q_PRELOAD,
                Q_HEADS as u32,
                1,
                WORKGROUP_SIZE,
            );
        }
        return (
            KERNEL_DECODE_WAVE,
            DEVICE_DECODE_WAVE,
            Q_HEADS as u32,
            1,
            WORKGROUP_SIZE,
        );
    }
    if case.m >= 128 {
        let candidate = match config.mode {
            Mode::K4 => Some((KERNEL_K4, DEVICE_K4)),
            Mode::K8 => Some((KERNEL_K8, DEVICE_K8)),
            Mode::K16 => Some((KERNEL_K16, DEVICE_K16)),
            Mode::K32 => Some((KERNEL_K32, DEVICE_K32)),
            Mode::BlockSoftmax => Some((KERNEL_BLOCKSOFTMAX, DEVICE_BLOCKSOFTMAX)),
            Mode::BlockSoftmaxQ8 if config.target == "gfx1201" => {
                Some((KERNEL_BLOCKSOFTMAX_Q8, DEVICE_BLOCKSOFTMAX_Q8))
            }
            Mode::Baseline | Mode::QTile1 | Mode::DecodeP32 | Mode::DecodeP64 => None,
            Mode::BlockSoftmaxQ8 => None,
        };
        if let Some((kernel, device)) = candidate {
            return (
                kernel,
                device,
                (if config.mode == Mode::BlockSoftmaxQ8 {
                    case.m.div_ceil(8)
                } else {
                    case.m.div_ceil(4)
                } * KV_HEADS) as u32,
                1,
                WORKGROUP_SIZE,
            );
        }
    }
    if config.mode == Mode::QTile1 && case.m >= 128 {
        return (
            KERNEL_QTILE1,
            DEVICE_QTILE1,
            (case.m.div_ceil(4) * KV_HEADS) as u32,
            1,
            WORKGROUP_SIZE,
        );
    }
    if config.target == "gfx1201" && (case.m == 1 || case.m >= 32) {
        return (
            KERNEL_WAVE1201,
            DEVICE_WAVE1201,
            (case.m * Q_HEADS) as u32,
            1,
            WORKGROUP_SIZE,
        );
    }
    (
        KERNEL_BASELINE,
        DEVICE_BASELINE,
        (case.m * Q_HEADS) as u32,
        1,
        WORKGROUP_SIZE,
    )
}

fn metadata_matches(dispatch: &sllm_core::DispatchEvidence, config: &Config, case: Case) -> bool {
    let (kernel, device, grid, dispatch_count, workgroup_size) = expected_kernel(config, case);
    dispatch.abi_version == sllm_hip_sys::SLLM_HIP_ABI_VERSION
        && dispatch.info_version == sllm_hip_sys::SLLM_HIP_CAUSAL_ATTENTION_DISPATCH_INFO_VERSION
        && dispatch.dispatch_id != 0
        && dispatch.dispatch_count == dispatch_count
        && dispatch.kernel_id == sllm_hip_sys::SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_ONLINE_SOFTMAX_V2
        && dispatch.workgroup_size_x == workgroup_size
        && dispatch.grid_size_x == grid
        && dispatch.row_count == case.m as u64
        && dispatch.normalized_size == HEAD_DIM as u64
        && dispatch.backend == sllm_hip_sys::SLLM_BACKEND_HIP
        && !dispatch.fallback_allowed
        && !dispatch.fallback_used
        && dispatch.kernel_symbol == kernel
        && dispatch.device_symbol == device
        && dispatch.target == config.target
}

fn median(values: &[u64]) -> Option<u64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    Some(sorted[sorted.len() / 2])
}

fn median_mad(values: &[u64]) -> (Option<u64>, Option<u64>) {
    let Some(middle) = median(values) else {
        return (None, None);
    };
    let deviations = values
        .iter()
        .map(|value| value.abs_diff(middle))
        .collect::<Vec<_>>();
    (Some(middle), median(&deviations))
}

fn run_case(
    session: &ExecutionSession,
    queue: &sllm_core::ExecutionQueue,
    config: &Config,
    case: Case,
    seed: u64,
    benchmark: bool,
) -> Result<CaseEvidence, String> {
    let committed = case
        .start_position
        .checked_add(case.m as u64)
        .ok_or_else(|| "case committed length overflow".to_owned())?;
    let descriptor = KvStateDescriptor::new_with_storage(
        seed as u32,
        committed,
        KV_HEADS,
        HEAD_DIM,
        KvCacheEncoding::Fp16,
    )
    .map_err(|error| format!("KV descriptor failed: {error}"))?;
    let state = session
        .create_kv_state(descriptor)
        .map_err(|error| format!("KV state creation failed: {error}"))?;
    let prefix_key = input_key_words(case.start_position as usize, 0);
    let prefix_value = input_value_words(case.start_position as usize, 0);
    if case.start_position != 0 {
        append_tokens(session, queue, &state, &prefix_key, &prefix_value, 0)?;
    }
    let key_words = input_key_words(case.m, case.start_position);
    let value_words = input_value_words(case.m, case.start_position);
    append_tokens(
        session,
        queue,
        &state,
        &key_words,
        &value_words,
        case.start_position,
    )?;
    let query_words = input_query_words(case.m);
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
    let attention_descriptor =
        sllm_core::CausalAttentionDescriptor::new(case.start_position, case.m as u64, committed)
            .map_err(|error| format!("causal descriptor failed: {error}"))?;
    let mut attention = session
        .causal_attention(&state, queue, query, output, attention_descriptor)
        .map_err(|error| format!("causal attention submission failed: {error}"))?;
    let dispatch = attention.dispatch().clone();
    let metadata_match = metadata_matches(&dispatch, config, case);
    if attention
        .wait(WAIT_TIMEOUT)
        .map_err(|error| format!("causal attention wait failed: {error}"))?
        != ExecutionState::Success
    {
        return Err("causal attention did not reach success".to_owned());
    }
    if attention
        .kernel_elapsed_ns()
        .map_err(|error| format!("causal attention timing failed: {error}"))?
        .is_none_or(|elapsed| elapsed == 0)
    {
        return Err("HIP causal attention returned zero device time".to_owned());
    }
    drop(attention);

    let mut timing_samples_ns = Vec::new();
    if benchmark {
        timing_samples_ns.reserve(TIMING_MEASURED);
        for repetition in 0..(TIMING_WARMUPS + TIMING_MEASURED) {
            let timing_query = make_binding(session, &query_buffer, &shape, AccessMode::Read)?;
            let timing_output = make_binding(session, &output_buffer, &shape, AccessMode::Write)?;
            let mut measured = session
                .causal_attention(
                    &state,
                    queue,
                    timing_query,
                    timing_output,
                    attention_descriptor,
                )
                .map_err(|error| format!("timed attention submission failed: {error}"))?;
            if measured
                .wait(WAIT_TIMEOUT)
                .map_err(|error| format!("timed attention wait failed: {error}"))?
                != ExecutionState::Success
            {
                return Err("timed causal attention did not reach success".to_owned());
            }
            let elapsed = measured
                .kernel_elapsed_ns()
                .map_err(|error| format!("timed attention timing failed: {error}"))?
                .ok_or_else(|| "timed attention omitted device timing".to_owned())?;
            if elapsed == 0 {
                return Err("timed attention returned zero device time".to_owned());
            }
            if repetition >= TIMING_WARMUPS {
                timing_samples_ns.push(elapsed);
            }
        }
    }

    let all_key_words = [prefix_key.as_slice(), key_words.as_slice()].concat();
    let all_value_words = [prefix_value.as_slice(), value_words.as_slice()].concat();
    let rows = oracle_rows(case);
    let (expected_rows, nonuniform_softmax_checked, bf16_roundtrip_checked) =
        scalar_oracle(&query_words, &all_key_words, &all_value_words, case, &rows)?;
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
    let (numerical_match, max_abs_error, max_relative_error, max_bf16_ulp, bf16_ulp_mismatch_count) =
        compare_rows(&actual, &expected_rows, case);
    let output_bytes_sha256 = format!("sha256:{:x}", Sha256::digest(&actual_bytes));
    let causal_visibility_match = state
        .snapshot(session)
        .map(|snapshot| snapshot.length() == committed)
        .unwrap_or(false);
    let gqa_mapping_match = numerical_match;
    let (timing_median_ns, timing_mad_ns) = median_mad(&timing_samples_ns);
    drop(readback);
    drop(state);
    drop(query_buffer);
    drop(output_buffer);
    Ok(CaseEvidence {
        id: case.id,
        m: case.m,
        start_position: case.start_position,
        committed_kv_length: committed,
        numerical_match,
        oracle_scope: if rows.len() == case.m {
            "all-rows-all-24-heads-scalar-v1"
        } else {
            "sampled-rows-all-24-heads-scalar-v1"
        },
        oracle_rows: rows,
        max_abs_error,
        max_relative_error,
        max_bf16_ulp,
        bf16_ulp_mismatch_count,
        nonuniform_softmax_checked,
        bf16_to_fp16_to_f32_checked: bf16_roundtrip_checked,
        causal_visibility_match,
        gqa_mapping_match,
        metadata_match,
        hip_execution: true,
        no_fallback: !dispatch.fallback_allowed && !dispatch.fallback_used,
        output_bytes_sha256,
        timing_warmups: if benchmark { TIMING_WARMUPS } else { 0 },
        timing_samples_ns,
        timing_median_ns,
        timing_mad_ns,
    })
}

fn unavailable_report(config: &Config, error: String) -> Report {
    Report {
        schema_version: "sllm-gqa6-attention-evidence-v1",
        state: "UNAVAILABLE",
        pass: false,
        mode: config.mode.name(),
        target: config.target.clone(),
        device_index: config.device_index,
        q_heads: Q_HEADS,
        kv_heads: KV_HEADS,
        head_dim: HEAD_DIM,
        kv_storage: "bf16-input-to-fp16-state",
        selected_backend: "hip",
        gpu_execution: false,
        cpu_fallback_used: false,
        fallback_allowed: false,
        fallback_used: false,
        benchmark_warmups: if config.benchmark { TIMING_WARMUPS } else { 0 },
        benchmark_measured: if config.benchmark { TIMING_MEASURED } else { 0 },
        cases: Vec::new(),
        oracle: OracleEvidence {
            scalar_ordered_dot_softmax_v: false,
            all_24_query_heads_checked: false,
            bf16_kv_roundtrip_checked: false,
            final_bf16_rne_checked: false,
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
    configure_mode(config.mode, &config.target);
    let selected = match selected_cases(config.case_filter.as_deref()) {
        Ok(selected) => selected,
        Err(error) => return unavailable_report(config, error),
    };
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
        for (index, case) in selected.iter().copied().enumerate() {
            let benchmark =
                config.benchmark && benchmark_case(case, config.benchmark_filter.as_deref())?;
            cases.push(run_case(
                &session,
                &queue,
                config,
                case,
                index as u64 + 1,
                benchmark,
            )?);
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
                    .any(|case| case.nonuniform_softmax_checked),
                all_24_query_heads_checked: cases.iter().all(|case| case.gqa_mapping_match),
                bf16_kv_roundtrip_checked: cases
                    .iter()
                    .all(|case| case.bf16_to_fp16_to_f32_checked),
                final_bf16_rne_checked: cases.iter().all(|case| case.numerical_match),
            };
            let pass = cases.iter().all(|case| {
                case.numerical_match
                    && case.causal_visibility_match
                    && case.gqa_mapping_match
                    && case.metadata_match
                    && case.hip_execution
                    && case.no_fallback
            }) && oracle.scalar_ordered_dot_softmax_v
                && oracle.all_24_query_heads_checked
                && oracle.bf16_kv_roundtrip_checked
                && oracle.final_bf16_rne_checked
                && cleanup.retryable_cleanup == 0
                && cleanup.durable_quarantine == 0;
            Report {
                schema_version: "sllm-gqa6-attention-evidence-v1",
                state: if pass { "PASS" } else { "INCOMPLETE" },
                pass,
                mode: config.mode.name(),
                target: config.target.clone(),
                device_index: config.device_index,
                q_heads: Q_HEADS,
                kv_heads: KV_HEADS,
                head_dim: HEAD_DIM,
                kv_storage: "bf16-input-to-fp16-state",
                selected_backend: "hip",
                gpu_execution: true,
                cpu_fallback_used: false,
                fallback_allowed: false,
                fallback_used: false,
                benchmark_warmups: if config.benchmark { TIMING_WARMUPS } else { 0 },
                benchmark_measured: if config.benchmark { TIMING_MEASURED } else { 0 },
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

type AttentionProbeOutputs = (Vec<(usize, Vec<u16>)>, bool, bool);

fn main() -> ExitCode {
    match parse_config() {
        Ok(config) => match serde_json::to_string(&run(&config)) {
            Ok(output) => {
                println!("{output}");
                if output.contains("\"pass\":true") {
                    ExitCode::SUCCESS
                } else {
                    ExitCode::from(1)
                }
            }
            Err(error) => {
                eprintln!("sllm-gqa6-attention-evidence: report serialization failed: {error}");
                ExitCode::from(2)
            }
        },
        Err(error) => {
            eprintln!("sllm-gqa6-attention-evidence: {error}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixture_keeps_six_query_heads_distinct_per_kv_head() {
        let words = input_query_words(1);
        for group in 0..KV_HEADS {
            let first = (group * GQA_RATIO) * HEAD_DIM;
            let second = first + HEAD_DIM;
            assert_ne!(words[first], words[second]);
        }
    }

    #[test]
    fn case_boundaries_cover_requested_non_aligned_shapes() {
        for m in [3, 4, 5, 127, 128, 129, 219] {
            assert!(
                CASES
                    .iter()
                    .any(|case| case.m == m && case.start_position == 0)
            );
        }
        for committed in [31, 32, 33] {
            assert!(
                CASES
                    .iter()
                    .any(|case| { case.start_position + case.m as u64 == committed })
            );
        }
        assert!(
            CASES
                .iter()
                .any(|case| { case.start_position == 9_216 && case.m == 219 })
        );
    }

    #[test]
    fn key_fixture_exercises_bf16_to_fp16_rounding() {
        let words = input_key_words(1, 0);
        assert_eq!(words[0], 0x3300);
        assert_ne!(
            bf16_to_f32(words[0]),
            f16_to_f32(sllm_hip::bf16_to_f16_bits(words[0]))
        );
    }

    #[test]
    fn median_and_mad_are_deterministic() {
        assert_eq!(median(&[9, 1, 5]), Some(5));
        assert_eq!(median_mad(&[9, 1, 5, 5]), (Some(5), Some(4)));
    }

    #[test]
    fn decode_p64_metadata_is_reserved_for_long_m1_decode() {
        let config = Config {
            device_index: 0,
            target: "gfx1030".to_owned(),
            mode: Mode::DecodeP64,
            benchmark: false,
            case_filter: None,
            benchmark_filter: None,
        };
        assert_eq!(
            expected_kernel(
                &config,
                Case {
                    id: "decode-context4096",
                    m: 1,
                    start_position: 4_095,
                },
            )
            .0,
            KERNEL_DECODE_WAVE_Q_PRELOAD
        );
        assert_eq!(
            expected_kernel(
                &config,
                Case {
                    id: "decode-context9435",
                    m: 1,
                    start_position: 9_434,
                },
            ),
            (KERNEL_DECODE_GQA6_P64, DEVICE_DECODE_GQA6_P64, 256, 2, 192,)
        );
        assert_ne!(
            expected_kernel(
                &config,
                Case {
                    id: "context33",
                    m: 1,
                    start_position: 32,
                },
            )
            .0,
            KERNEL_DECODE_GQA6_P64
        );
    }

    #[test]
    fn blocksoftmax_metadata_is_reserved_for_prefill_boundary() {
        let config = Config {
            device_index: 0,
            target: "gfx1030".to_owned(),
            mode: Mode::BlockSoftmax,
            benchmark: false,
            case_filter: None,
            benchmark_filter: None,
        };
        assert_eq!(
            expected_kernel(
                &config,
                Case {
                    id: "prefill-m128",
                    m: 128,
                    start_position: 0,
                },
            ),
            (
                KERNEL_BLOCKSOFTMAX,
                DEVICE_BLOCKSOFTMAX,
                128,
                1,
                WORKGROUP_SIZE,
            )
        );
        assert_eq!(
            expected_kernel(
                &config,
                Case {
                    id: "prefill-m127",
                    m: 127,
                    start_position: 0,
                },
            )
            .0,
            KERNEL_BASELINE
        );
    }

    #[test]
    fn blocksoftmax_q8_metadata_is_gfx1201_only_and_uses_q8_grid() {
        let config = Config {
            device_index: 0,
            target: "gfx1201".to_owned(),
            mode: Mode::BlockSoftmaxQ8,
            benchmark: false,
            case_filter: None,
            benchmark_filter: None,
        };
        assert_eq!(
            expected_kernel(
                &config,
                Case {
                    id: "prefill-m128",
                    m: 128,
                    start_position: 0,
                },
            ),
            (
                KERNEL_BLOCKSOFTMAX_Q8,
                DEVICE_BLOCKSOFTMAX_Q8,
                64,
                1,
                WORKGROUP_SIZE,
            )
        );
        let gfx1030 = Config {
            device_index: config.device_index,
            target: "gfx1030".to_owned(),
            mode: config.mode,
            benchmark: config.benchmark,
            case_filter: None,
            benchmark_filter: None,
        };
        assert_eq!(
            expected_kernel(
                &gfx1030,
                Case {
                    id: "prefill-m128",
                    m: 128,
                    start_position: 0,
                },
            )
            .0,
            KERNEL_BASELINE
        );
        assert_eq!(
            expected_kernel(
                &config,
                Case {
                    id: "prefill-m127",
                    m: 127,
                    start_position: 0,
                },
            )
            .0,
            KERNEL_WAVE1201
        );
    }
}
