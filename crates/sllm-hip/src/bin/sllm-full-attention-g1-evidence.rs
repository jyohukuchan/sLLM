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
use sha2::{Digest, Sha256};
use sllm_core::{
    AccessMode, Backend, DType, Encoding, ExecutionSession, ExecutionSessionRequest,
    ExecutionState, KvCacheEncoding, KvMemoryKind, KvStateDescriptor, TensorView, decode_e2m1,
    decode_e4m3fn, encode_e2m1, encode_e4m3fn,
};
use sllm_hip::HipBackend;

const WAIT_TIMEOUT: Duration = Duration::from_secs(30);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(16);
const Q_HEADS: usize = 16;
const KV_HEADS: usize = 4;
const HEAD_DIM: usize = 256;
const WORKGROUP_SIZE: u32 = 256;
const TIMING_WARMUPS: usize = 5;
const TIMING_MEASURED: usize = 21;

#[derive(Clone, Copy, Debug)]
struct Case {
    id: &'static str,
    m: usize,
    start_position: u64,
}

// Non-Cartesian coverage: prefill M boundaries plus decode prefixes. The
// prefill M=1/start=0 case is also the decode-prefix-zero boundary.
const CASES: [Case; 29] = [
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
        id: "prefill-m37",
        m: 37,
        start_position: 0,
    },
    Case {
        id: "prefill-m63",
        m: 63,
        start_position: 0,
    },
    Case {
        id: "prefill-m64",
        m: 64,
        start_position: 0,
    },
    Case {
        id: "prefill-m65",
        m: 65,
        start_position: 0,
    },
    Case {
        id: "prefill-m64-start3",
        m: 64,
        start_position: 3,
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
    Case {
        id: "decode-kv1023",
        m: 1,
        start_position: 1022,
    },
    Case {
        id: "decode-kv1024",
        m: 1,
        start_position: 1023,
    },
    Case {
        id: "decode-kv1025",
        m: 1,
        start_position: 1024,
    },
    Case {
        id: "decode-long-kv8193",
        m: 1,
        start_position: 8192,
    },
    Case {
        id: "decode-mixed-kv4097",
        m: 1,
        start_position: 4096,
    },
    Case {
        id: "special-query-nan",
        m: 1,
        start_position: 0,
    },
    Case {
        id: "special-value-pos-inf",
        m: 1,
        start_position: 0,
    },
    Case {
        id: "special-decode1024-query-nan",
        m: 1,
        start_position: 1023,
    },
    Case {
        id: "special-decode1024-value-pos-inf",
        m: 1,
        start_position: 1023,
    },
    Case {
        id: "special-prefill64-query-nan",
        m: 64,
        start_position: 0,
    },
    Case {
        id: "special-prefill64-value-pos-inf",
        m: 64,
        start_position: 0,
    },
];

// Phase 12's original 16-case matrix, retained as an explicit subset for
// current-main regression runs. The default matrix above remains unchanged.
const PHASE12_CASES: [Case; 16] = [
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
        id: "prefill-m37",
        m: 37,
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
    Case {
        id: "decode-kv1023",
        m: 1,
        start_position: 1022,
    },
    Case {
        id: "decode-kv1024",
        m: 1,
        start_position: 1023,
    },
    Case {
        id: "decode-kv1025",
        m: 1,
        start_position: 1024,
    },
    Case {
        id: "special-query-nan",
        m: 1,
        start_position: 0,
    },
    Case {
        id: "special-value-pos-inf",
        m: 1,
        start_position: 0,
    },
];

// Focused Phase 49 operator rows for prefill-provider boundary and long-row audit.
const PHASE49_OPERATOR_CASES: [Case; 22] = [
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
        id: "prefill-m1024",
        m: 1024,
        start_position: 0,
    },
    Case {
        id: "prefill-m1023",
        m: 1023,
        start_position: 0,
    },
    Case {
        id: "prefill-m1025",
        m: 1025,
        start_position: 0,
    },
    Case {
        id: "prefill-m4096",
        m: 4096,
        start_position: 0,
    },
    Case {
        id: "prefill-m10001",
        m: 10_001,
        start_position: 0,
    },
    Case {
        id: "prefill-m1024-start257",
        m: 1_024,
        start_position: 257,
    },
    Case {
        id: "special-prefill1024-query-nan",
        m: 1_024,
        start_position: 0,
    },
    Case {
        id: "special-prefill1024-value-pos-inf",
        m: 1_024,
        start_position: 0,
    },
    Case {
        id: "special-prefill1024-value-pos-neg-inf",
        m: 1_024,
        start_position: 0,
    },
    Case {
        id: "special-prefill1024-value-nan-pos-inf",
        m: 1_024,
        start_position: 0,
    },
    Case {
        id: "decode-kv1023-operator",
        m: 1,
        start_position: 1_022,
    },
    Case {
        id: "decode-kv1024-operator",
        m: 1,
        start_position: 1_023,
    },
    Case {
        id: "decode-kv1025-operator",
        m: 1,
        start_position: 1_024,
    },
    Case {
        id: "decode-kv4096-operator",
        m: 1,
        start_position: 4_095,
    },
    Case {
        id: "decode-kv8192-operator",
        m: 1,
        start_position: 8_191,
    },
    Case {
        id: "decode-kv16384-operator",
        m: 1,
        start_position: 16_383,
    },
];

// Focused short-decode boundary cases. With the opt-in environment flag these
// exercise the candidate at both sides of its KV-length gate.
const PHASE49_SHORT_DECODE_CASES: [Case; 6] = [
    Case {
        id: "decode-kv31-short",
        m: 1,
        start_position: 30,
    },
    Case {
        id: "decode-kv32-short",
        m: 1,
        start_position: 31,
    },
    Case {
        id: "decode-kv33-short",
        m: 1,
        start_position: 32,
    },
    Case {
        id: "decode-kv128-short",
        m: 1,
        start_position: 127,
    },
    Case {
        id: "decode-kv287-short",
        m: 1,
        start_position: 286,
    },
    Case {
        id: "decode-kv1023-short",
        m: 1,
        start_position: 1022,
    },
];

#[derive(Debug)]
struct Config {
    device_index: u32,
    target: String,
    kv_encoding: KvCacheEncoding,
    phase12_subset: bool,
    phase49_operator: bool,
    phase49_decode_operator: bool,
    phase49_decode_short: bool,
}

#[derive(Debug, Serialize)]
struct CaseEvidence {
    id: &'static str,
    m: usize,
    start_position: u64,
    committed_kv_length: u64,
    memory_kind: &'static str,
    physical_page_bytes: u64,
    mapped_token_capacity: u64,
    committed_bytes_per_plane: u64,
    fp16_committed_bytes_per_plane: u64,
    logical_bytes_per_plane: u64,
    fp16_logical_bytes_per_plane: u64,
    logical_byte_reduction_fraction: f64,
    committed_byte_reduction_fraction: f64,
    numerical_match: bool,
    numerical_oracle_scope: &'static str,
    numerical_oracle_rows: Vec<usize>,
    output_bytes_sha256: String,
    max_abs_error: f64,
    nonuniform_softmax_checked: bool,
    subnormal_score_contribution_checked: bool,
    causal_visibility_match: bool,
    gqa_mapping_match: bool,
    sampled_rows_causal_visibility_match: bool,
    sampled_rows_gqa_mapping_match: bool,
    metadata_match: bool,
    no_fallback: bool,
    timing_warmups: usize,
    timing_samples_ns: Vec<u64>,
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
    kv_encoding: &'static str,
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
    parse_config_from(env::args().skip(1))
}

fn parse_config_from<I>(arguments: I) -> Result<Config, String>
where
    I: IntoIterator<Item = String>,
{
    let mut device_index = None;
    let mut target = None;
    let mut kv_encoding = KvCacheEncoding::Fp16;
    let mut phase12_subset = false;
    let mut phase49_operator = false;
    let mut phase49_decode_operator = false;
    let mut phase49_decode_short = false;
    let mut arguments = arguments.into_iter();
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
                if !matches!(value.as_str(), "gfx1030" | "gfx1201" | "gfx942") {
                    return Err("--target must be gfx1030, gfx1201, or gfx942".to_owned());
                }
                target = Some(value);
            }
            "--kv-encoding" => {
                let value = arguments
                    .next()
                    .ok_or_else(|| "--kv-encoding requires a value".to_owned())?;
                kv_encoding = match value.as_str() {
                    "fp16" => KvCacheEncoding::Fp16,
                    "fp8" => KvCacheEncoding::Fp8E4M3Fn,
                    "fp8-static" => KvCacheEncoding::Fp8E4M3FnStatic,
                    "nvfp4" => KvCacheEncoding::Nvfp4,
                    _ => {
                        return Err(
                            "--kv-encoding must be fp16, fp8, fp8-static, or nvfp4".to_owned()
                        );
                    }
                };
            }
            "--phase12-subset" => {
                if phase12_subset {
                    return Err("duplicate --phase12-subset".to_owned());
                }
                phase12_subset = true;
            }
            "--phase49-operator" => {
                if phase49_operator {
                    return Err("duplicate --phase49-operator".to_owned());
                }
                phase49_operator = true;
            }
            "--phase49-decode-operator" => {
                if phase49_decode_operator {
                    return Err("duplicate --phase49-decode-operator".to_owned());
                }
                phase49_decode_operator = true;
            }
            "--phase49-decode-short" => {
                if phase49_decode_short {
                    return Err("duplicate --phase49-decode-short".to_owned());
                }
                phase49_decode_short = true;
            }
            other => return Err(format!("unexpected argument {other}")),
        }
    }
    Ok(Config {
        device_index: device_index.ok_or_else(|| "missing --device-index".to_owned())?,
        target: target.ok_or_else(|| "missing --target".to_owned())?,
        kv_encoding,
        phase12_subset,
        phase49_operator,
        phase49_decode_operator,
        phase49_decode_short,
    })
}

fn selected_cases(
    phase12_subset: bool,
    phase49_operator: bool,
    phase49_decode_operator: bool,
    phase49_decode_short: bool,
) -> &'static [Case] {
    if phase49_decode_short {
        &PHASE49_SHORT_DECODE_CASES
    } else if phase49_decode_operator {
        &PHASE49_OPERATOR_CASES[16..]
    } else if phase49_operator {
        &PHASE49_OPERATOR_CASES
    } else if phase12_subset {
        &PHASE12_CASES
    } else {
        &CASES
    }
}

fn kv_encoding_name(encoding: KvCacheEncoding) -> &'static str {
    match encoding {
        KvCacheEncoding::Fp16 => "fp16-v1",
        KvCacheEncoding::Fp8E4M3Fn => "kv-fp8-v1",
        KvCacheEncoding::Fp8E4M3FnStatic => "kv-fp8-static-v1",
        KvCacheEncoding::Nvfp4 => "kv-nvfp4-v1",
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
        return if value.is_sign_negative() {
            0x8000
        } else {
            0x0000
        };
    }
    let magnitude = value.abs();
    let exponent = magnitude.log2().floor() as i32;
    let quantum_exponent = if exponent < -126 { -133 } else { exponent - 7 };
    let quantum = 2.0_f64.powi(quantum_exponent);
    let rounded = (magnitude / quantum).round_ties_even() * quantum;
    float_to_bf16_rne(if value.is_sign_negative() {
        -(rounded as f32)
    } else {
        rounded as f32
    })
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
            // Keep adjacent KV heads distinct after BF16 rounding even at the
            // 1023/1024/1025 committed-length boundaries.
            let value = 1.0 + token as f32 + head as f32 * 32.0 + (seed % 2) as f32;
            let word = float_to_bf16_rne(value);
            for dimension in 0..HEAD_DIM {
                words[(token * KV_HEADS + head) * HEAD_DIM + dimension] = word;
            }
        }
    }
    words
}

fn input_mixed_q_words(m: usize) -> Vec<u16> {
    let mut words = vec![0_u16; m * Q_HEADS * HEAD_DIM];
    for row in 0..m {
        for head in 0..Q_HEADS {
            for dimension in 0..HEAD_DIM {
                let code = ((row * 19 + head * 11 + dimension * 7) % 31) as i32 - 15;
                words[(row * Q_HEADS + head) * HEAD_DIM + dimension] =
                    float_to_bf16_rne(code as f32 / 16.0);
            }
        }
    }
    words
}

fn input_mixed_k_words(token_count: usize, start_position: u64) -> Vec<u16> {
    let mut words = vec![0_u16; token_count * KV_HEADS * HEAD_DIM];
    for token in 0..token_count {
        let absolute = start_position + token as u64;
        for head in 0..KV_HEADS {
            for dimension in 0..HEAD_DIM {
                let code =
                    ((absolute * 13 + head as u64 * 17 + dimension as u64 * 5) % 37) as i32 - 18;
                words[(token * KV_HEADS + head) * HEAD_DIM + dimension] =
                    float_to_bf16_rne(code as f32 / 32.0);
            }
        }
    }
    words
}

fn input_mixed_v_words(token_count: usize, start_position: u64) -> Vec<u16> {
    let mut words = vec![0_u16; token_count * KV_HEADS * HEAD_DIM];
    for token in 0..token_count {
        let absolute = start_position + token as u64;
        for head in 0..KV_HEADS {
            for dimension in 0..HEAD_DIM {
                let code =
                    ((absolute * 29 + head as u64 * 23 + dimension as u64 * 3) % 127) as i32 - 63;
                words[(token * KV_HEADS + head) * HEAD_DIM + dimension] =
                    float_to_bf16_rne(code as f32 / 32.0);
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
    encoding: KvCacheEncoding,
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
    let key_values = quantized_kv_values(key_words, encoding)?;
    let value_values = quantized_kv_values(value_words, encoding)?;
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
                let mut dot = 0.0_f64;
                let mut dot_without_subnormal_lane = 0.0_f64;
                for dimension in 0..HEAD_DIM {
                    let product = f64::from(bf16_to_f32(query_words[query_offset + dimension]))
                        * f64::from(key_values[key_offset + dimension]);
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
            let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            nonuniform_softmax_checked |= scores
                .windows(2)
                .any(|pair| pair[0].to_bits() != pair[1].to_bits());
            let mut denominator = 0.0_f64;
            for score in &scores {
                denominator += (*score - maximum).exp();
            }
            for dimension in 0..HEAD_DIM {
                let mut accumulation = 0.0_f64;
                for (index, score) in scores.iter().enumerate() {
                    let value_offset = (index * KV_HEADS + kv_head) * HEAD_DIM + dimension;
                    let probability = (*score - maximum).exp() / denominator;
                    accumulation += probability * f64::from(value_values[value_offset]);
                }
                output[query_offset + dimension] = f64_to_bf16_rne(accumulation);
            }
        }
    }
    Ok((
        output,
        nonuniform_softmax_checked,
        subnormal_score_contribution_checked || encoding != KvCacheEncoding::Fp16,
    ))
}

fn quantized_kv_values(words: &[u16], encoding: KvCacheEncoding) -> Result<Vec<f32>, String> {
    let input = words.iter().copied().map(bf16_to_f32).collect::<Vec<_>>();
    match encoding {
        KvCacheEncoding::Fp16 => Ok(words
            .iter()
            .copied()
            .map(|word| f16_to_f32(sllm_hip::bf16_to_f16_bits(word)))
            .collect()),
        KvCacheEncoding::Fp8E4M3Fn => {
            let mut output = Vec::with_capacity(input.len());
            for row in input.chunks_exact(HEAD_DIM) {
                let maximum = row
                    .iter()
                    .filter(|value| value.is_finite())
                    .fold(0.0_f32, |current, value| current.max(value.abs()));
                let scale = if maximum == 0.0 { 1.0 } else { maximum / 448.0 };
                output.extend(
                    row.iter()
                        .map(|value| decode_e4m3fn(encode_e4m3fn(*value / scale)) * scale),
                );
            }
            Ok(output)
        }
        KvCacheEncoding::Fp8E4M3FnStatic => Ok(input
            .into_iter()
            .map(|value| decode_e4m3fn(encode_e4m3fn(value)))
            .collect()),
        KvCacheEncoding::Nvfp4 => {
            let mut output = Vec::with_capacity(input.len());
            for row in input.chunks_exact(HEAD_DIM) {
                let maximum = row
                    .iter()
                    .filter(|value| value.is_finite())
                    .fold(0.0_f32, |current, value| current.max(value.abs()));
                let outer = if maximum == 0.0 {
                    1.0
                } else {
                    maximum / (448.0 * 6.0)
                };
                for block in row.chunks(16) {
                    let mut block_maximum = block
                        .iter()
                        .filter(|value| value.is_finite())
                        .fold(0.0_f32, |current, value| current.max(value.abs()));
                    if block.iter().any(|value| value.is_infinite()) {
                        block_maximum = 448.0 * 6.0 * outer;
                    }
                    let decoded_scale = decode_e4m3fn(encode_e4m3fn((block_maximum / 6.0) / outer));
                    output.extend(block.iter().map(|value| {
                        if decoded_scale == 0.0 {
                            0.0
                        } else {
                            decode_e2m1(encode_e2m1(*value / (decoded_scale * outer)))
                                * decoded_scale
                                * outer
                        }
                    }));
                }
            }
            Ok(output)
        }
    }
}

fn decode_wave_split_q_preload_enabled(
    expected_target: &str,
    use_decode_wave_split: bool,
    q_preload_opt_in: Option<&std::ffi::OsStr>,
) -> bool {
    expected_target == "gfx1030"
        && use_decode_wave_split
        && q_preload_opt_in.is_none_or(|value| value == "1")
}

fn decode_wave_split_fp16_pair_enabled(
    expected_target: &str,
    case: Case,
    encoding: KvCacheEncoding,
    opt_in: Option<&std::ffi::OsStr>,
    force_baseline: bool,
) -> bool {
    !force_baseline
        && expected_target == "gfx1030"
        && opt_in.is_none_or(|value| value == "1")
        && case.m == 1
        && case.start_position + case.m as u64 >= 1024
        && Q_HEADS == 16
        && KV_HEADS == 4
        && HEAD_DIM == 256
        && encoding == KvCacheEncoding::Fp16
}

fn decode_gqa4_split_enabled(
    expected_target: &str,
    case: Case,
    encoding: KvCacheEncoding,
    opt_in: Option<&std::ffi::OsStr>,
    force_baseline: bool,
) -> bool {
    !force_baseline
        && expected_target == "gfx1030"
        && opt_in.is_some_and(|value| value == "1")
        && case.m == 1
        && case.start_position + case.m as u64 >= 4096
        && Q_HEADS == 16
        && KV_HEADS == 4
        && HEAD_DIM == 256
        && encoding == KvCacheEncoding::Fp16
}

fn decode_wave_split_short_enabled(
    expected_target: &str,
    case: Case,
    encoding: KvCacheEncoding,
    short_decode_opt_in: Option<&std::ffi::OsStr>,
) -> bool {
    expected_target == "gfx1030"
        && short_decode_opt_in.is_none_or(|value| value == "1")
        && case.m == 1
        && (32..1024).contains(&(case.start_position + case.m as u64))
        && Q_HEADS == 16
        && KV_HEADS == 4
        && HEAD_DIM == 256
        && encoding == KvCacheEncoding::Fp16
}

fn decode_wave_split_short_q_preload_enabled(
    use_decode_wave_split_short: bool,
    short_q_preload_opt_in: Option<&std::ffi::OsStr>,
) -> bool {
    use_decode_wave_split_short && short_q_preload_opt_in.is_none_or(|value| value == "1")
}

fn scaled_prefill_gemm_enabled(
    expected_target: &str,
    case: Case,
    encoding: KvCacheEncoding,
    opt_in: Option<&std::ffi::OsStr>,
    force_baseline: bool,
) -> bool {
    !force_baseline
        && expected_target == "gfx1030"
        && opt_in.is_none_or(|value| value == "1")
        && case.m >= 1024
        && Q_HEADS == 16
        && KV_HEADS == 4
        && HEAD_DIM == 256
        && encoding == KvCacheEncoding::Fp16
}

fn long_prefill_v2_enabled(
    expected_target: &str,
    case: Case,
    encoding: KvCacheEncoding,
    opt_in: Option<&std::ffi::OsStr>,
    force_baseline: bool,
) -> bool {
    !force_baseline
        && expected_target == "gfx1030"
        && opt_in.is_some_and(|value| value == "1")
        && case.m >= 1024
        && Q_HEADS == 16
        && KV_HEADS == 4
        && HEAD_DIM == 256
        && encoding == KvCacheEncoding::Fp16
}

fn metadata_matches(
    dispatch: &sllm_core::DispatchEvidence,
    case: Case,
    expected_target: &str,
    encoding: KvCacheEncoding,
) -> bool {
    let use_phase33_common_provider = is_phase33_common_target(expected_target);
    let use_gfx1201_wave_provider = expected_target == "gfx1201" && (case.m == 1 || case.m >= 32);
    let use_decode_wave_split_long =
        use_phase33_common_provider && case.m == 1 && case.start_position + 1 >= 1024;
    let force_baseline =
        env::var_os("SLLM_CAUSAL_ATTENTION_FORCE_BASELINE").is_some_and(|value| value == "1");
    let short_decode_opt_in = env::var_os("SLLM_CAUSAL_ATTENTION_GFX1030_DECODE_WAVE_SHORT");
    let use_decode_wave_split_short = decode_wave_split_short_enabled(
        expected_target,
        case,
        encoding,
        short_decode_opt_in.as_deref(),
    );
    let use_decode_wave_split = use_decode_wave_split_long || use_decode_wave_split_short;
    let fp16_pair_opt_in = env::var_os("SLLM_CAUSAL_ATTENTION_GFX1030_DECODE_WAVE_FP16_PAIR");
    let gqa4_split_opt_in = env::var_os("SLLM_CAUSAL_ATTENTION_GFX1030_DECODE_GQA4_SPLIT");
    let gqa4_split_p32_opt_in = env::var_os("SLLM_CAUSAL_ATTENTION_GFX1030_DECODE_GQA4_SPLIT_P32");
    let use_decode_gqa4_split = decode_gqa4_split_enabled(
        expected_target,
        case,
        encoding,
        gqa4_split_opt_in.as_deref(),
        force_baseline,
    ) && !decode_gqa4_split_enabled(
        expected_target,
        case,
        encoding,
        gqa4_split_p32_opt_in.as_deref(),
        force_baseline,
    );
    let use_decode_gqa4_split_p32 = decode_gqa4_split_enabled(
        expected_target,
        case,
        encoding,
        gqa4_split_p32_opt_in.as_deref(),
        force_baseline,
    );
    let use_decode_wave_split_fp16_pair = decode_wave_split_fp16_pair_enabled(
        expected_target,
        case,
        encoding,
        fp16_pair_opt_in.as_deref(),
        force_baseline,
    ) && !use_decode_gqa4_split
        && !use_decode_gqa4_split_p32;
    let q_preload_opt_in = env::var_os("SLLM_CAUSAL_ATTENTION_GFX1030_Q_PRELOAD");
    let use_decode_wave_split_q_preload_long = decode_wave_split_q_preload_enabled(
        expected_target,
        use_decode_wave_split_long,
        q_preload_opt_in.as_deref(),
    );
    let short_q_preload_opt_in =
        env::var_os("SLLM_CAUSAL_ATTENTION_GFX1030_DECODE_WAVE_SHORT_Q_PRELOAD");
    let use_decode_wave_split_q_preload_short = decode_wave_split_short_q_preload_enabled(
        use_decode_wave_split_short,
        short_q_preload_opt_in.as_deref(),
    );
    let use_decode_wave_split_q_preload =
        use_decode_wave_split_q_preload_long || use_decode_wave_split_q_preload_short;
    let use_prefill_gqa4 = use_phase33_common_provider && case.m >= 64;
    let scaled_prefill_opt_in = env::var_os("SLLM_CAUSAL_ATTENTION_GFX1030_SCALED_PREFILL_GEMM");
    let long_prefill_v2_opt_in = env::var_os("SLLM_CAUSAL_ATTENTION_GFX1030_LONG_PREFILL_V2");
    let use_long_prefill_v2 = long_prefill_v2_enabled(
        expected_target,
        case,
        encoding,
        long_prefill_v2_opt_in.as_deref(),
        force_baseline,
    );
    let use_scaled_prefill_gemm = scaled_prefill_gemm_enabled(
        expected_target,
        case,
        encoding,
        scaled_prefill_opt_in.as_deref(),
        force_baseline,
    ) && !use_long_prefill_v2;
    let use_prefill_gqa4_qtile4 =
        use_prefill_gqa4 && case.m >= 128 && !force_baseline && !use_scaled_prefill_gemm;
    let (kernel_id, baseline_kernel_symbol, baseline_device_symbol) =
        if encoding == KvCacheEncoding::Fp16 {
            (
                sllm_hip_sys::SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_ONLINE_SOFTMAX_V2,
                "causal_attention.online_softmax_gqa.v2",
                "sllm_causal_attention_online_softmax_gqa_v2",
            )
        } else {
            (
                sllm_hip_sys::SLLM_HIP_CAUSAL_ATTENTION_KERNEL_ID_PACKED_KV_V3,
                "causal_attention.online_softmax_gqa.packed_kv.v3",
                "sllm_causal_attention_online_softmax_gqa_packed_kv_v3",
            )
        };
    let (kernel_symbol, device_symbol) = if use_decode_gqa4_split_p32 {
        (
            "causal_attention.decode.gqa4_tiled_split.p32.v1",
            "sllm_causal_attention_decode_gqa4_split_p32_v1",
        )
    } else if use_decode_gqa4_split {
        (
            "causal_attention.decode.gqa4_tiled_split.v1",
            "sllm_causal_attention_decode_gqa4_tiled_split_v1",
        )
    } else if use_decode_wave_split_fp16_pair {
        (
            "causal_attention.decode.wave8_split.fp16_pair.v1",
            "sllm_causal_attention_decode_wave8_split_fp16_pair_v1",
        )
    } else if use_decode_wave_split {
        if use_decode_wave_split_q_preload {
            (
                "causal_attention.decode.wave8_split.q_preload.v1",
                "sllm_causal_attention_decode_wave8_split_q_preload_v1",
            )
        } else {
            (
                "causal_attention.decode.wave8_split.v5",
                "sllm_causal_attention_decode_wave8_split_v5",
            )
        }
    } else if use_scaled_prefill_gemm {
        (
            "causal_attention.prefill.gfx1030_hipblas_scaled_fp16.v1",
            "sllm_causal_attention_prefill_gfx1030_hipblas_scaled_fp16_v1",
        )
    } else if use_long_prefill_v2 {
        (
            "causal_attention.prefill.gfx1030_qtile8_split.v2",
            "sllm_causal_attention_prefill_gfx1030_qtile8_split_v2",
        )
    } else if use_prefill_gqa4_qtile4 {
        (
            "causal_attention.prefill.gqa4_qtile4.v7",
            "sllm_causal_attention_prefill_gqa4_qtile4_v7",
        )
    } else if use_prefill_gqa4 {
        (
            "causal_attention.prefill.gqa4_shared.v6",
            "sllm_causal_attention_prefill_gqa4_shared_v6",
        )
    } else if use_gfx1201_wave_provider {
        if encoding == KvCacheEncoding::Fp16 {
            (
                "causal_attention.online_softmax_gqa.gfx1201_wave.v4",
                "sllm_causal_attention_gfx1201_wave_v4",
            )
        } else {
            (
                "causal_attention.online_softmax_gqa.packed_kv.gfx1201_wave.v4",
                "sllm_causal_attention_packed_gfx1201_wave_v4",
            )
        }
    } else {
        (baseline_kernel_symbol, baseline_device_symbol)
    };
    dispatch.abi_version == sllm_hip_sys::SLLM_HIP_ABI_VERSION
        && dispatch.info_version == sllm_hip_sys::SLLM_HIP_CAUSAL_ATTENTION_DISPATCH_INFO_VERSION
        && dispatch.dispatch_id != 0
        && dispatch.dispatch_count
            == if use_decode_gqa4_split || use_decode_gqa4_split_p32 || use_long_prefill_v2 {
                2
            } else {
                1
            }
        && dispatch.kernel_id == kernel_id
        && dispatch.workgroup_size_x
            == if use_decode_gqa4_split || use_decode_gqa4_split_p32 {
                128
            } else {
                WORKGROUP_SIZE
            }
        && dispatch.grid_size_x
            == if use_decode_gqa4_split_p32 {
                128
            } else if use_decode_gqa4_split {
                64
            } else if use_scaled_prefill_gemm {
                case.m.div_ceil(256) as u32 * KV_HEADS as u32
            } else if use_long_prefill_v2 {
                case.m.div_ceil(8) as u32 * KV_HEADS as u32 * 16
            } else if use_prefill_gqa4_qtile4 {
                (case.m.div_ceil(4) * KV_HEADS) as u32
            } else if use_prefill_gqa4 {
                (case.m * KV_HEADS) as u32
            } else {
                (case.m * Q_HEADS) as u32
            }
        && dispatch.row_count == case.m as u64
        && dispatch.normalized_size == HEAD_DIM as u64
        && dispatch.backend == sllm_hip_sys::SLLM_BACKEND_HIP
        && !dispatch.fallback_allowed
        && !dispatch.fallback_used
        && dispatch.kernel_symbol == kernel_symbol
        && dispatch.device_symbol == device_symbol
        && dispatch.target == expected_target
}

fn is_phase33_common_target(target: &str) -> bool {
    matches!(target, "gfx1030" | "gfx1201")
}

fn phase49_oracle_rows(m: usize) -> Vec<usize> {
    let mut rows = vec![0, m / 2, m - 1];
    for boundary in [255, 256, 257] {
        if boundary < m {
            rows.push(boundary);
        }
    }
    rows.sort_unstable();
    rows.dedup();
    rows
}

fn compare_bf16_words(
    observed_word: u16,
    reference_word: u16,
    encoding: KvCacheEncoding,
) -> (bool, f64) {
    let observed = f64::from(bf16_to_f32(observed_word));
    let reference = f64::from(bf16_to_f32(reference_word));
    if reference.is_nan() {
        return (observed.is_nan(), 0.0);
    }
    if reference.is_infinite() {
        return (observed == reference, 0.0);
    }
    if !observed.is_finite() {
        return (false, 0.0);
    }
    let error = (observed - reference).abs();
    let (absolute_tolerance, relative_tolerance) = match encoding {
        KvCacheEncoding::Fp16 => (0.016, 0.0),
        KvCacheEncoding::Fp8E4M3Fn => (0.03125, 0.04),
        KvCacheEncoding::Fp8E4M3FnStatic => (0.03125, 0.04),
        KvCacheEncoding::Nvfp4 => (0.125, 0.25),
    };
    (
        error <= absolute_tolerance + relative_tolerance * reference.abs(),
        error,
    )
}

fn sampled_scalar_oracle(
    query_words: &[u16],
    key_words: &[u16],
    value_words: &[u16],
    m: usize,
    start_position: u64,
    encoding: KvCacheEncoding,
    actual: &[u16],
) -> Result<(bool, f64, bool, bool, bool, bool), String> {
    let rows = phase49_oracle_rows(m);
    let row_elements = Q_HEADS * HEAD_DIM;
    let mut numerical_match = true;
    let mut max_abs_error = 0.0_f64;
    let mut nonuniform_softmax_checked = false;
    let mut subnormal_score_contribution_checked = true;
    let mut sampled_rows_causal_visibility_match = true;
    let mut sampled_rows_gqa_mapping_match = true;
    let debug_mismatch = env::var_os("SLLM_SCALED_PREFILL_DEBUG").is_some();
    let mut debug_mismatch_count = 0_u32;
    for row in rows {
        let query_start = row
            .checked_mul(row_elements)
            .ok_or_else(|| "sampled oracle query offset overflowed".to_owned())?;
        let query_end = query_start
            .checked_add(row_elements)
            .ok_or_else(|| "sampled oracle query end overflowed".to_owned())?;
        let query_row = query_words
            .get(query_start..query_end)
            .ok_or_else(|| "sampled oracle query row was out of bounds".to_owned())?;
        let committed_words = (start_position + row as u64 + 1)
            .checked_mul((KV_HEADS * HEAD_DIM) as u64)
            .ok_or_else(|| "sampled oracle KV offset overflowed".to_owned())?
            as usize;
        sampled_rows_causal_visibility_match &=
            committed_words == (start_position + row as u64 + 1) as usize * KV_HEADS * HEAD_DIM;
        let key_prefix = key_words
            .get(..committed_words)
            .ok_or_else(|| "sampled oracle key prefix was out of bounds".to_owned())?;
        let value_prefix = value_words
            .get(..committed_words)
            .ok_or_else(|| "sampled oracle value prefix was out of bounds".to_owned())?;
        let (expected, nonuniform, subnormal) = scalar_oracle(
            query_row,
            key_prefix,
            value_prefix,
            1,
            start_position + row as u64,
            encoding,
        )?;
        nonuniform_softmax_checked |= nonuniform;
        subnormal_score_contribution_checked &= subnormal;
        let actual_row = actual
            .get(query_start..query_end)
            .ok_or_else(|| "sampled oracle output row was out of bounds".to_owned())?;
        if expected.len() != actual_row.len() {
            numerical_match = false;
            sampled_rows_gqa_mapping_match = false;
            continue;
        }
        let mut row_matches = true;
        for (index, (&observed, &reference)) in actual_row.iter().zip(&expected).enumerate() {
            let (matches, error) = compare_bf16_words(observed, reference, encoding);
            numerical_match &= matches;
            row_matches &= matches;
            max_abs_error = max_abs_error.max(error);
            if debug_mismatch && !matches && debug_mismatch_count < 16 {
                let head = index / HEAD_DIM;
                let dimension = index % HEAD_DIM;
                eprintln!(
                    "scaled-prefill mismatch row={row} head={head} dim={dimension} observed=0x{observed:04x} reference=0x{reference:04x} observed_f32={} reference_f32={} error={error}",
                    bf16_to_f32(observed),
                    bf16_to_f32(reference)
                );
                debug_mismatch_count += 1;
            }
        }
        // scalar_oracle returns every Q head; a row match therefore checks all
        // four GQA query-to-KV groups rather than one representative head.
        sampled_rows_gqa_mapping_match &= row_matches;
    }
    Ok((
        numerical_match,
        max_abs_error,
        nonuniform_softmax_checked,
        subnormal_score_contribution_checked,
        sampled_rows_causal_visibility_match,
        sampled_rows_gqa_mapping_match,
    ))
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
    let descriptor = if config.kv_encoding == KvCacheEncoding::Fp8E4M3FnStatic {
        KvStateDescriptor::new_with_static_fp8(seed as u32, capacity, KV_HEADS, HEAD_DIM, 1.0, 1.0)
    } else {
        KvStateDescriptor::new_with_storage(
            seed as u32,
            capacity,
            KV_HEADS,
            HEAD_DIM,
            config.kv_encoding,
        )
    }
    .map_err(|error| format!("KV descriptor failed: {error}"))?;
    let state = session
        .create_kv_state(descriptor)
        .map_err(|error| format!("KV state creation failed: {error}"))?;
    let mixed = case.id == "decode-mixed-kv4097";
    let prefix_key = if mixed {
        input_mixed_k_words(case.start_position as usize, 0)
    } else {
        input_k_words(case.start_position as usize, 0)
    };
    let prefix_value = if mixed {
        input_mixed_v_words(case.start_position as usize, 0)
    } else {
        input_v_words(case.start_position as usize, seed + 1)
    };
    if case.start_position != 0 {
        append_tokens(session, queue, &state, &prefix_key, &prefix_value, 0)?;
    }
    let key_words = if mixed {
        input_mixed_k_words(case.m, case.start_position)
    } else {
        input_k_words(case.m, case.start_position)
    };
    let mut value_words = if mixed {
        input_mixed_v_words(case.m, case.start_position)
    } else {
        input_v_words(case.m, seed + 2)
    };
    if case.id.contains("value-pos-inf") {
        value_words.fill(0x7f80);
    }
    if case.id.contains("value-pos-neg-inf") {
        for token in 0..case.m {
            let raw = if token % 2 == 0 { 0x7f80 } else { 0xff80 };
            let begin = token * KV_HEADS * HEAD_DIM;
            let end = begin + KV_HEADS * HEAD_DIM;
            value_words[begin..end].fill(raw);
        }
    }
    if case.id.contains("value-nan-pos-inf") {
        for token in 0..case.m {
            let raw = if token % 2 == 0 { 0x7fc1 } else { 0x7f80 };
            let begin = token * KV_HEADS * HEAD_DIM;
            let end = begin + KV_HEADS * HEAD_DIM;
            value_words[begin..end].fill(raw);
        }
    }
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
    let physical = snapshot
        .physical_memory()
        .ok_or_else(|| "KV snapshot omitted physical-memory evidence".to_owned())?;
    let memory_kind = match physical.memory_kind() {
        KvMemoryKind::VirtualContiguous => "virtual-contiguous",
        KvMemoryKind::ContiguousResident => "contiguous-resident",
    };
    let descriptor = snapshot.descriptor();
    let logical_bytes_per_plane = descriptor
        .resident_bytes_per_plane()
        .ok_or_else(|| "KV logical resident-byte calculation overflowed".to_owned())?;
    let fp16_logical_bytes_per_plane = capacity
        .checked_mul(KV_HEADS as u64)
        .and_then(|value| value.checked_mul(HEAD_DIM as u64))
        .and_then(|value| value.checked_mul(2))
        .ok_or_else(|| "FP16 KV logical resident-byte calculation overflowed".to_owned())?;
    let logical_byte_reduction_fraction =
        1.0 - logical_bytes_per_plane as f64 / fp16_logical_bytes_per_plane as f64;
    let fp16_committed_bytes_per_plane = match physical.memory_kind() {
        KvMemoryKind::VirtualContiguous => fp16_logical_bytes_per_plane
            .checked_add(physical.physical_page_bytes() - 1)
            .map(|value| value / physical.physical_page_bytes())
            .and_then(|pages| pages.checked_mul(physical.physical_page_bytes()))
            .ok_or_else(|| "FP16 KV committed-byte calculation overflowed".to_owned())?,
        KvMemoryKind::ContiguousResident => fp16_logical_bytes_per_plane,
    };
    let committed_byte_reduction_fraction =
        1.0 - physical.committed_bytes_per_plane() as f64 / fp16_committed_bytes_per_plane as f64;
    let mut query_words = if mixed {
        input_mixed_q_words(case.m)
    } else {
        input_q_words(case.m)
    };
    if case.id.contains("query-nan") {
        for row in 0..case.m {
            for head in 0..Q_HEADS {
                query_words[(row * Q_HEADS + head) * HEAD_DIM] = 0x7fc1;
            }
        }
    }
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
    let metadata_match = metadata_matches(&dispatch, case, &config.target, config.kv_encoding);
    if attention
        .wait(WAIT_TIMEOUT)
        .map_err(|error| format!("causal attention wait failed: {error}"))?
        != ExecutionState::Success
    {
        return Err("causal attention did not reach success".to_owned());
    }
    let first_elapsed = attention
        .kernel_elapsed_ns()
        .map_err(|error| format!("causal attention timing failed: {error}"))?
        .ok_or_else(|| "HIP causal attention omitted device timing".to_owned())?;
    if first_elapsed == 0 {
        return Err("HIP causal attention returned zero device time".to_owned());
    }
    drop(attention);

    let mut timing_samples_ns = Vec::with_capacity(TIMING_MEASURED);
    for repetition in 0..(TIMING_WARMUPS + TIMING_MEASURED) {
        let timing_query = make_binding(session, &query_buffer, &shape, AccessMode::Read)?;
        let timing_output = make_binding(session, &output_buffer, &shape, AccessMode::Write)?;
        let mut measured = session
            .causal_attention(&state, queue, timing_query, timing_output, descriptor)
            .map_err(|error| format!("timed causal attention submission failed: {error}"))?;
        if measured
            .wait(WAIT_TIMEOUT)
            .map_err(|error| format!("timed causal attention wait failed: {error}"))?
            != ExecutionState::Success
        {
            return Err("timed causal attention did not reach success".to_owned());
        }
        let elapsed = measured
            .kernel_elapsed_ns()
            .map_err(|error| format!("timed causal attention timing failed: {error}"))?
            .ok_or_else(|| "timed HIP causal attention omitted device timing".to_owned())?;
        if elapsed == 0 {
            return Err("timed HIP causal attention returned zero device time".to_owned());
        }
        if repetition >= TIMING_WARMUPS {
            timing_samples_ns.push(elapsed);
        }
    }

    let all_key_words = [prefix_key.as_slice(), key_words.as_slice()].concat();
    let all_value_words = [prefix_value.as_slice(), value_words.as_slice()].concat();
    let sampled_oracle = config.phase49_operator && case.m >= 1024;
    let oracle_rows = if sampled_oracle {
        phase49_oracle_rows(case.m)
    } else {
        Vec::new()
    };
    let (expected, mut nonuniform_softmax_checked, mut subnormal_score_contribution_checked) =
        if sampled_oracle {
            (Vec::new(), false, false)
        } else {
            scalar_oracle(
                &query_words,
                &all_key_words,
                &all_value_words,
                case.m,
                case.start_position,
                config.kv_encoding,
            )?
        };
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
    let output_bytes_sha256 = format!("sha256:{:x}", Sha256::digest(&actual_bytes));
    let (
        numerical_match,
        max_abs_error,
        sampled_rows_causal_visibility_match,
        sampled_rows_gqa_mapping_match,
    ) = if sampled_oracle {
        let (matches, error, nonuniform, subnormal, sampled_visibility, sampled_gqa) =
            sampled_scalar_oracle(
                &query_words,
                &all_key_words,
                &all_value_words,
                case.m,
                case.start_position,
                config.kv_encoding,
                &actual,
            )?;
        nonuniform_softmax_checked = nonuniform;
        subnormal_score_contribution_checked = subnormal;
        (matches, error, sampled_visibility, sampled_gqa)
    } else {
        let mut matches = actual.len() == expected.len();
        let mut error_max = 0.0_f64;
        for (&observed_word, &reference_word) in actual.iter().zip(&expected) {
            let (word_matches, error) =
                compare_bf16_words(observed_word, reference_word, config.kv_encoding);
            matches &= word_matches;
            error_max = error_max.max(error);
        }
        (matches, error_max, true, true)
    };
    let causal_visibility_match = snapshot.length() == committed_length;
    let gqa_mapping_match = if sampled_oracle {
        sampled_rows_gqa_mapping_match
    } else {
        // The full scalar oracle covers every query head and applies the
        // reviewed Q-head to KV-head mapping.  Distinct query heads may
        // legitimately produce identical rows, so output inequality is not
        // a valid GQA oracle.
        numerical_match
    };
    drop(readback);
    drop(state);
    drop(query_buffer);
    drop(output_buffer);
    Ok(CaseEvidence {
        id: case.id,
        m: case.m,
        start_position: case.start_position,
        committed_kv_length: committed_length,
        memory_kind,
        physical_page_bytes: physical.physical_page_bytes(),
        mapped_token_capacity: physical.mapped_token_capacity(),
        committed_bytes_per_plane: physical.committed_bytes_per_plane(),
        fp16_committed_bytes_per_plane,
        logical_bytes_per_plane,
        fp16_logical_bytes_per_plane,
        logical_byte_reduction_fraction,
        committed_byte_reduction_fraction,
        numerical_match,
        numerical_oracle_scope: if sampled_oracle {
            "sampled-rows-scalar-v1"
        } else {
            "full-scalar-v1"
        },
        numerical_oracle_rows: oracle_rows,
        output_bytes_sha256,
        max_abs_error,
        nonuniform_softmax_checked,
        subnormal_score_contribution_checked,
        causal_visibility_match,
        gqa_mapping_match,
        sampled_rows_causal_visibility_match,
        sampled_rows_gqa_mapping_match,
        metadata_match,
        no_fallback: !dispatch.fallback_allowed && !dispatch.fallback_used,
        timing_warmups: TIMING_WARMUPS,
        timing_samples_ns,
    })
}

fn unavailable_report(config: &Config, error: String) -> Report {
    Report {
        schema_version: "sllm-full-attention-g1-evidence-v2",
        state: "UNAVAILABLE",
        pass: false,
        target: config.target.clone(),
        kv_encoding: kv_encoding_name(config.kv_encoding),
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
        for (index, case) in selected_cases(
            config.phase12_subset,
            config.phase49_operator,
            config.phase49_decode_operator,
            config.phase49_decode_short,
        )
        .iter()
        .copied()
        .enumerate()
        {
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
                    .filter(|case| !case.id.starts_with("special-"))
                    .all(|case| case.subnormal_score_contribution_checked),
                final_bf16_rne_checked: cases.iter().all(|case| case.numerical_match),
                gqa_heads_checked: cases.iter().all(|case| case.gqa_mapping_match),
            };
            let all_cases = cases.iter().all(|case| {
                case.numerical_match
                    && case.causal_visibility_match
                    && case.gqa_mapping_match
                    && case.sampled_rows_causal_visibility_match
                    && case.sampled_rows_gqa_mapping_match
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
                schema_version: "sllm-full-attention-g1-evidence-v2",
                state: if pass { "PASS" } else { "INCOMPLETE" },
                pass,
                target: config.target.clone(),
                kv_encoding: kv_encoding_name(config.kv_encoding),
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
            scalar_oracle(&query, &keys, &values, 2, 1, KvCacheEncoding::Fp16).unwrap();
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
        assert_eq!(CASES.len(), 29);
        assert!(CASES.iter().any(|case| case.m == 37));
        for query_count in [63, 64, 65, 127, 128, 129] {
            assert!(CASES.iter().any(|case| case.m == query_count));
        }
        assert!(CASES.iter().any(|case| case.m == 257));
        assert!(CASES.iter().any(|case| case.start_position == 257));
        for committed_length in [1023, 1024, 1025] {
            assert!(CASES.iter().any(|case| {
                case.start_position + u64::try_from(case.m).unwrap() == committed_length
            }));
        }
        assert!(
            CASES
                .iter()
                .any(|case| { case.start_position + u64::try_from(case.m).unwrap() == 8193 })
        );
        assert!(CASES.iter().any(|case| case.id == "special-query-nan"));
        assert!(CASES.iter().any(|case| case.id == "special-value-pos-inf"));
        assert!(
            CASES
                .iter()
                .any(|case| case.id == "special-decode1024-query-nan")
        );
        assert!(
            CASES
                .iter()
                .any(|case| case.id == "special-decode1024-value-pos-inf")
        );
        assert!(
            CASES
                .iter()
                .any(|case| case.id == "special-prefill64-query-nan")
        );
        assert!(
            CASES
                .iter()
                .any(|case| case.id == "special-prefill64-value-pos-inf")
        );
        assert!(CASES.iter().all(|case| case.m > 0));
    }

    #[test]
    fn phase12_subset_selects_the_original_sixteen_cases() {
        let ids = selected_cases(true, false, false, false)
            .iter()
            .map(|case| case.id)
            .collect::<Vec<_>>();
        assert_eq!(
            ids,
            vec![
                "prefill-m1",
                "prefill-m3",
                "prefill-m17",
                "prefill-m37",
                "prefill-m255",
                "prefill-m256",
                "prefill-m257",
                "decode-prefix3",
                "decode-prefix255",
                "decode-prefix256",
                "decode-prefix257",
                "decode-kv1023",
                "decode-kv1024",
                "decode-kv1025",
                "special-query-nan",
                "special-value-pos-inf",
            ]
        );
        assert_eq!(selected_cases(true, false, false, false).len(), 16);
        assert_eq!(
            selected_cases(false, false, false, false).len(),
            CASES.len()
        );
        assert_eq!(
            selected_cases(false, false, false, false)[0].id,
            CASES[0].id
        );
        assert_eq!(
            selected_cases(false, false, false, false)[28].id,
            CASES[28].id
        );
        assert_eq!(
            selected_cases(false, true, false, false).len(),
            PHASE49_OPERATOR_CASES.len()
        );
        assert_eq!(selected_cases(false, true, false, false)[0].m, 127);
        assert_eq!(
            selected_cases(false, true, false, false)
                .iter()
                .map(|case| case.m)
                .collect::<Vec<_>>(),
            vec![
                127, 128, 129, 255, 256, 257, 1024, 1023, 1025, 4096, 10_001, 1024, 1024, 1024,
                1024, 1024, 1, 1, 1, 1, 1, 1,
            ]
        );
        assert_eq!(selected_cases(false, false, true, false).len(), 6);
        assert_eq!(
            selected_cases(false, false, true, false)
                .iter()
                .map(|case| case.start_position + 1)
                .collect::<Vec<_>>(),
            vec![1023, 1024, 1025, 4096, 8192, 16384]
        );
        assert_eq!(selected_cases(false, false, false, true).len(), 6);
        assert_eq!(
            selected_cases(false, false, false, true)
                .iter()
                .map(|case| case.start_position + 1)
                .collect::<Vec<_>>(),
            vec![31, 32, 33, 128, 287, 1023]
        );
    }

    #[test]
    fn phase12_subset_flag_is_parsed_without_changing_defaults() {
        let default = parse_config_from(
            vec!["--device-index", "0", "--target", "gfx942"]
                .into_iter()
                .map(String::from),
        )
        .unwrap();
        assert!(!default.phase12_subset);
        assert!(!default.phase49_operator);
        assert!(!default.phase49_decode_operator);
        assert!(!default.phase49_decode_short);

        let subset = parse_config_from(
            vec![
                "--device-index",
                "0",
                "--target",
                "gfx942",
                "--phase12-subset",
            ]
            .into_iter()
            .map(String::from),
        )
        .unwrap();
        assert!(subset.phase12_subset);
        assert!(!subset.phase49_operator);
        assert!(!subset.phase49_decode_operator);
        assert!(!subset.phase49_decode_short);
        assert_eq!(subset.device_index, default.device_index);
        assert_eq!(subset.target, default.target);
        assert_eq!(subset.kv_encoding, default.kv_encoding);

        let operator = parse_config_from(
            vec![
                "--device-index",
                "0",
                "--target",
                "gfx1030",
                "--phase49-operator",
            ]
            .into_iter()
            .map(String::from),
        )
        .unwrap();
        assert!(operator.phase49_operator);
        assert!(!operator.phase12_subset);
        assert!(!operator.phase49_decode_operator);
        assert!(!operator.phase49_decode_short);

        let decode_operator = parse_config_from(
            vec![
                "--device-index",
                "0",
                "--target",
                "gfx1030",
                "--phase49-decode-operator",
            ]
            .into_iter()
            .map(String::from),
        )
        .unwrap();
        assert!(decode_operator.phase49_decode_operator);
        assert!(!decode_operator.phase49_operator);
        assert!(!decode_operator.phase12_subset);
        assert!(!decode_operator.phase49_decode_short);

        let decode_short = parse_config_from(
            vec![
                "--device-index",
                "0",
                "--target",
                "gfx1030",
                "--phase49-decode-short",
            ]
            .into_iter()
            .map(String::from),
        )
        .unwrap();
        assert!(decode_short.phase49_decode_short);
        assert!(!decode_short.phase49_decode_operator);
        assert!(!decode_short.phase49_operator);
        assert!(!decode_short.phase12_subset);

        let duplicate = parse_config_from(
            vec![
                "--device-index",
                "0",
                "--target",
                "gfx942",
                "--phase12-subset",
                "--phase12-subset",
            ]
            .into_iter()
            .map(String::from),
        );
        assert_eq!(
            duplicate.unwrap_err(),
            "duplicate --phase12-subset".to_owned()
        );
    }

    #[test]
    fn phase49_operator_oracle_uses_deterministic_boundary_rows() {
        assert_eq!(phase49_oracle_rows(1024), vec![0, 255, 256, 257, 512, 1023]);
        assert_eq!(
            phase49_oracle_rows(10_001),
            vec![0, 255, 256, 257, 5_000, 10_000]
        );
        assert_eq!(phase49_oracle_rows(1), vec![0]);
    }

    #[test]
    fn phase49_operator_cases_use_bounded_scalar_oracle_only_at_large_m() {
        assert!(
            PHASE49_OPERATOR_CASES
                .iter()
                .filter(|case| case.m < 1024)
                .all(|case| phase49_oracle_rows(case.m).len() <= 6)
        );
        assert_eq!(PHASE49_OPERATOR_CASES[6].m, 1024);
        assert_eq!(PHASE49_OPERATOR_CASES[10].m, 10_001);
        assert_eq!(
            PHASE49_OPERATOR_CASES[16..]
                .iter()
                .map(|case| case.start_position + 1)
                .collect::<Vec<_>>(),
            vec![1023, 1024, 1025, 4096, 8192, 16384]
        );
    }

    #[test]
    fn phase49_short_decode_cases_cover_the_requested_kv_boundaries() {
        assert_eq!(
            PHASE49_SHORT_DECODE_CASES
                .iter()
                .map(|case| case.start_position + case.m as u64)
                .collect::<Vec<_>>(),
            vec![31, 32, 33, 128, 287, 1023]
        );
        assert!(PHASE49_SHORT_DECODE_CASES.iter().all(|case| case.m == 1));
    }

    #[test]
    fn phase33_and_phase35_prefill_providers_remain_rdna_scoped() {
        assert!(is_phase33_common_target("gfx1030"));
        assert!(is_phase33_common_target("gfx1201"));
        assert!(!is_phase33_common_target("gfx942"));
    }

    #[test]
    fn decode_q_preload_guard_defaults_on_and_accepts_only_explicit_disable() {
        assert!(decode_wave_split_q_preload_enabled("gfx1030", true, None));
        assert!(decode_wave_split_q_preload_enabled(
            "gfx1030",
            true,
            Some(std::ffi::OsStr::new("1"))
        ));
        assert!(!decode_wave_split_q_preload_enabled(
            "gfx1030",
            true,
            Some(std::ffi::OsStr::new("0"))
        ));
        assert!(!decode_wave_split_q_preload_enabled(
            "gfx1030",
            true,
            Some(std::ffi::OsStr::new("invalid"))
        ));
        assert!(!decode_wave_split_q_preload_enabled("gfx1201", true, None));
        assert!(!decode_wave_split_q_preload_enabled("gfx1030", false, None));
    }

    #[test]
    fn decode_fp16_pair_guard_defaults_on_for_long_gfx1030_shape_and_force_safe() {
        let enabled = Some(std::ffi::OsStr::new("1"));
        let boundary_case = |start_position| Case {
            id: "fp16-pair",
            m: 1,
            start_position,
        };
        assert!(decode_wave_split_fp16_pair_enabled(
            "gfx1030",
            boundary_case(1023),
            KvCacheEncoding::Fp16,
            enabled,
            false,
        ));
        assert!(decode_wave_split_fp16_pair_enabled(
            "gfx1030",
            Case {
                id: "fp16-pair-long",
                m: 1,
                start_position: 99_999,
            },
            KvCacheEncoding::Fp16,
            enabled,
            false,
        ));
        assert!(!decode_wave_split_fp16_pair_enabled(
            "gfx1030",
            boundary_case(1022),
            KvCacheEncoding::Fp16,
            enabled,
            false,
        ));
        assert!(!decode_wave_split_fp16_pair_enabled(
            "gfx1030",
            Case {
                id: "fp16-pair-batch",
                m: 2,
                start_position: 1023,
            },
            KvCacheEncoding::Fp16,
            enabled,
            false,
        ));
        assert!(decode_wave_split_fp16_pair_enabled(
            "gfx1030",
            boundary_case(1023),
            KvCacheEncoding::Fp16,
            None,
            false,
        ));
        for opt_in in [
            Some(std::ffi::OsStr::new("0")),
            Some(std::ffi::OsStr::new("unknown")),
        ] {
            assert!(!decode_wave_split_fp16_pair_enabled(
                "gfx1030",
                boundary_case(1023),
                KvCacheEncoding::Fp16,
                opt_in,
                false,
            ));
        }
        assert!(!decode_wave_split_fp16_pair_enabled(
            "gfx1201",
            boundary_case(1023),
            KvCacheEncoding::Fp16,
            enabled,
            false,
        ));
        assert!(!decode_wave_split_fp16_pair_enabled(
            "gfx1030",
            boundary_case(1023),
            KvCacheEncoding::Fp8E4M3Fn,
            enabled,
            false,
        ));
        assert!(!decode_wave_split_fp16_pair_enabled(
            "gfx1030",
            boundary_case(1023),
            KvCacheEncoding::Fp16,
            enabled,
            true,
        ));
    }

    #[test]
    fn decode_short_wave_guard_is_exact_target_shape_encoding_and_default_on() {
        let enabled = Some(std::ffi::OsStr::new("1"));
        assert!(decode_wave_split_short_enabled(
            "gfx1030",
            PHASE49_SHORT_DECODE_CASES[1],
            KvCacheEncoding::Fp16,
            enabled,
        ));
        assert!(decode_wave_split_short_enabled(
            "gfx1030",
            PHASE49_SHORT_DECODE_CASES[5],
            KvCacheEncoding::Fp16,
            enabled,
        ));
        assert!(decode_wave_split_short_enabled(
            "gfx1030",
            PHASE49_SHORT_DECODE_CASES[1],
            KvCacheEncoding::Fp16,
            None,
        ));
        assert!(!decode_wave_split_short_enabled(
            "gfx1030",
            PHASE49_SHORT_DECODE_CASES[0],
            KvCacheEncoding::Fp16,
            enabled,
        ));
        assert!(!decode_wave_split_short_enabled(
            "gfx1030",
            PHASE49_SHORT_DECODE_CASES[1],
            KvCacheEncoding::Fp8E4M3Fn,
            enabled,
        ));
        assert!(!decode_wave_split_short_enabled(
            "gfx1201",
            PHASE49_SHORT_DECODE_CASES[1],
            KvCacheEncoding::Fp16,
            enabled,
        ));
        assert!(!decode_wave_split_short_enabled(
            "gfx1030",
            PHASE49_SHORT_DECODE_CASES[1],
            KvCacheEncoding::Fp16,
            Some(std::ffi::OsStr::new("0")),
        ));
        assert!(!decode_wave_split_short_enabled(
            "gfx1030",
            PHASE49_SHORT_DECODE_CASES[1],
            KvCacheEncoding::Fp16,
            Some(std::ffi::OsStr::new("unknown")),
        ));
        assert!(!decode_wave_split_short_q_preload_enabled(
            false,
            Some(std::ffi::OsStr::new("1")),
        ));
        assert!(!decode_wave_split_short_q_preload_enabled(
            true,
            Some(std::ffi::OsStr::new("0")),
        ));
        assert!(decode_wave_split_short_q_preload_enabled(true, None));
        assert!(!decode_wave_split_short_q_preload_enabled(
            true,
            Some(std::ffi::OsStr::new("unknown")),
        ));
        assert!(decode_wave_split_short_q_preload_enabled(
            true,
            Some(std::ffi::OsStr::new("1")),
        ));
    }

    #[test]
    fn scaled_prefill_gemm_guard_and_oracle_cover_long_shape_edges() {
        let enabled = Some(std::ffi::OsStr::new("1"));
        for m in [1024, 4096, 10_001, 100_000, 1 << 20] {
            assert!(scaled_prefill_gemm_enabled(
                "gfx1030",
                Case {
                    id: "scaled",
                    m,
                    start_position: 257,
                },
                KvCacheEncoding::Fp16,
                enabled,
                false,
            ));
        }
        assert!(scaled_prefill_gemm_enabled(
            "gfx1030",
            Case {
                id: "scaled",
                m: 1024,
                start_position: 257,
            },
            KvCacheEncoding::Fp16,
            None,
            false,
        ));
        for value in ["0", "unknown"] {
            assert!(!scaled_prefill_gemm_enabled(
                "gfx1030",
                Case {
                    id: "scaled",
                    m: 1024,
                    start_position: 257,
                },
                KvCacheEncoding::Fp16,
                Some(std::ffi::OsStr::new(value)),
                false,
            ));
        }
        assert!(!scaled_prefill_gemm_enabled(
            "gfx1030",
            Case {
                id: "scaled",
                m: 1024,
                start_position: 257,
            },
            KvCacheEncoding::Fp8E4M3Fn,
            enabled,
            false,
        ));
        assert!(!scaled_prefill_gemm_enabled(
            "gfx1201",
            Case {
                id: "scaled",
                m: 1024,
                start_position: 257,
            },
            KvCacheEncoding::Fp16,
            enabled,
            false,
        ));
        // A power-of-two scale keeps 2^20 in the FP16 normal range.  The
        // tiny lane remains a representable FP16 subnormal after the same scale,
        // while NaN/Inf retain their IEEE classes for the special oracle.
        let scale = 2.0_f32.powi(-5);
        assert_eq!(
            sllm_hip::bf16_to_f16_bits(float_to_bf16_rne(2.0_f32.powi(20) * scale)),
            0x7800
        );
        assert_eq!(
            sllm_hip::bf16_to_f16_bits(float_to_bf16_rne(2.0_f32.powi(-19) * scale)),
            0x0001
        );
        assert!(f32::NAN.is_nan());
        assert!(f32::INFINITY.is_infinite());
    }

    #[test]
    fn long_prefill_v2_is_explicit_gfx1030_fp16_opt_in() {
        let enabled = Some(std::ffi::OsStr::new("1"));
        for m in [1024, 4096, 10_001, 100_000] {
            assert!(long_prefill_v2_enabled(
                "gfx1030",
                Case {
                    id: "long-v2",
                    m,
                    start_position: 257,
                },
                KvCacheEncoding::Fp16,
                enabled,
                false,
            ));
        }
        let case = Case {
            id: "long-v2",
            m: 1024,
            start_position: 257,
        };
        assert!(!long_prefill_v2_enabled(
            "gfx1030",
            case,
            KvCacheEncoding::Fp16,
            None,
            false,
        ));
        assert!(!long_prefill_v2_enabled(
            "gfx1030",
            case,
            KvCacheEncoding::Fp16,
            enabled,
            true,
        ));
        assert!(!long_prefill_v2_enabled(
            "gfx1201",
            case,
            KvCacheEncoding::Fp16,
            enabled,
            false,
        ));
        assert!(!long_prefill_v2_enabled(
            "gfx1030",
            Case { m: 1023, ..case },
            KvCacheEncoding::Fp16,
            enabled,
            false,
        ));
        assert!(!long_prefill_v2_enabled(
            "gfx1030",
            case,
            KvCacheEncoding::Fp8E4M3Fn,
            enabled,
            false,
        ));
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
    fn f64_oracle_rounds_directly_to_bf16() {
        assert_eq!(f64_to_bf16_rne(1.0 + 2.0_f64.powi(-8)), 0x3f80);
        assert_eq!(f64_to_bf16_rne(1.0 + 3.0 * 2.0_f64.powi(-8)), 0x3f82);
        assert_eq!(f64_to_bf16_rne(3.0 * 2.0_f64.powi(-134)), 0x0002);
        assert!(bf16_to_f32(f64_to_bf16_rne(f64::NAN)).is_nan());
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
