//! Exact host/device byte oracle for standard OCP MXFP8 E4 KV append and direct attention.
//!
//! The Phase 53 block16 route is retained below only as historical oracle code.  Argument
//! parsing rejects it, so this executable cannot admit the retired runtime path.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;
use sha2::{Digest, Sha256};
use sllm_core::{
    AccessMode, Backend, DType, Encoding, ExecutionSession, ExecutionSessionRequest,
    ExecutionState, KvCacheEncoding, KvFp8PhysicalVariant, KvMxfp8Descriptor, KvStateDescriptor,
    StatePlaneKindV1, TensorView, quantize_kv_fp8_block16, quantize_kv_mxfp8,
};
use sllm_hip::HipBackend;

const DIMENSIONS: [usize; 6] = [15, 16, 17, 255, 256, 257];
const MX_DIMENSIONS: [usize; 6] = [31, 32, 33, 255, 256, 257];
const HEADS: usize = 4;
const Q_HEADS: usize = 16;
const WAIT: Duration = Duration::from_secs(30);
const SHUTDOWN: Duration = Duration::from_secs(16);

#[derive(Clone, Copy, Eq, PartialEq)]
enum EvidenceFormat {
    Block16,
    Mxfp8,
}

#[derive(Clone)]
struct Config {
    device_index: u32,
    target: String,
    policy_sha256: String,
    encoding: KvCacheEncoding,
    variant: KvFp8PhysicalVariant,
    format: EvidenceFormat,
}

#[derive(Debug, Serialize)]
struct HostEvidence {
    pass: bool,
    variants: Vec<&'static str>,
    head_dimensions: Vec<usize>,
    special_values: Vec<&'static str>,
    tail_padding_zero: bool,
    key_value_scales_independent: bool,
}

#[derive(Debug, Serialize)]
struct CaseEvidence {
    head_dim: usize,
    value_bytes_exact: bool,
    key_scales_exact: bool,
    value_scales_exact: bool,
    key_value_scales_independent: bool,
    tail_padding_zero: bool,
    append_direct: bool,
    attention_direct: bool,
    numerical_match: bool,
}

#[derive(Serialize)]
struct ExecutionEvidence {
    selected_backend: &'static str,
    gpu_execution: bool,
    fallback_allowed: bool,
    fallback_used: bool,
    append_dispatches: u64,
    attention_dispatches: u64,
}

struct HostQuantized {
    values: Vec<u8>,
    scales: Vec<u8>,
    dequantized: Vec<f32>,
    padding_zero: bool,
}

#[derive(Serialize)]
struct CleanupEvidence {
    retryable: usize,
    durable: usize,
    terminal_zero: bool,
}

#[derive(Serialize)]
struct Report {
    #[serde(rename = "$schema")]
    schema: &'static str,
    schema_version: &'static str,
    state: &'static str,
    target: String,
    device_index: u32,
    encoding: &'static str,
    physical_variant: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    descriptor_id: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    scale_recipe: Option<&'static str>,
    policy_sha256: String,
    binary_sha256: String,
    host: HostEvidence,
    cases: Vec<CaseEvidence>,
    execution: ExecutionEvidence,
    cleanup: CleanupEvidence,
    error: Option<String>,
}

fn digest(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}

fn parse() -> Result<Config, String> {
    let mut device_index = None;
    let mut target = None;
    let mut policy = None;
    let mut format = EvidenceFormat::Mxfp8;
    let mut arguments = env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--device-index" => {
                device_index = Some(
                    arguments
                        .next()
                        .ok_or("--device-index needs a value")?
                        .parse::<u32>()
                        .map_err(|_| "--device-index must be u32")?,
                );
            }
            "--target" => target = Some(arguments.next().ok_or("--target needs a value")?),
            "--policy" => {
                policy = Some(PathBuf::from(
                    arguments.next().ok_or("--policy needs a value")?,
                ))
            }
            "--format" => {
                format = match arguments.next().ok_or("--format needs a value")?.as_str() {
                    "block16" => {
                        return Err("block16 KV evidence is retired; use --format mxfp8".to_owned());
                    }
                    "mxfp8" => EvidenceFormat::Mxfp8,
                    _ => return Err("--format must be mxfp8".to_owned()),
                }
            }
            other => return Err(format!("unexpected argument {other}")),
        }
    }
    let target = target.ok_or("missing --target")?;
    let (encoding, variant) = match (format, target.as_str()) {
        (EvidenceFormat::Mxfp8, "gfx942" | "gfx942:sramecc+:xnack-" | "gfx1201" | "gfx1030") => {
            (KvCacheEncoding::Mxfp8E4, KvFp8PhysicalVariant::OcpE4M3Fn)
        }
        (EvidenceFormat::Block16, _) => {
            return Err("block16 KV evidence is retired".to_owned());
        }
        _ => {
            return Err(
                "--target must be gfx942, gfx942:sramecc+:xnack-, gfx1201, or gfx1030".to_owned(),
            );
        }
    };
    let policy = policy.ok_or("missing --policy")?;
    let policy_bytes = fs::read(policy).map_err(|error| format!("read policy: {error}"))?;
    if policy_bytes.is_empty() {
        return Err("policy is empty".to_owned());
    }
    Ok(Config {
        device_index: device_index.ok_or("missing --device-index")?,
        target,
        policy_sha256: digest(&policy_bytes),
        encoding,
        variant,
        format,
    })
}

fn encoding_name(encoding: KvCacheEncoding) -> &'static str {
    match encoding {
        KvCacheEncoding::Fp8E4M3Block16 => "kv-fp8-e4-block16",
        KvCacheEncoding::Fp8E5M2Block16 => "kv-fp8-e5-block16",
        KvCacheEncoding::Mxfp8E4 => "kv-mxfp8-e4",
        KvCacheEncoding::Mxfp8E5 => "kv-mxfp8-e5",
        _ => unreachable!(),
    }
}

fn variant_name(variant: KvFp8PhysicalVariant, format: EvidenceFormat) -> &'static str {
    match variant {
        KvFp8PhysicalVariant::E4M3FnuZ => "E4M3-FNUZ",
        KvFp8PhysicalVariant::OcpE4M3Fn => "E4M3-OCP",
        KvFp8PhysicalVariant::OcpE5M2 if format == EvidenceFormat::Mxfp8 => "E5M2-OCP",
        KvFp8PhysicalVariant::OcpE5M2 => "E5M2-software",
    }
}

fn block16_descriptor_id(encoding: KvCacheEncoding) -> Option<&'static str> {
    match encoding {
        KvCacheEncoding::Fp8E4M3Block16 => Some("kv-fp8-e4-block16-v2"),
        KvCacheEncoding::Fp8E5M2Block16 => Some("kv-fp8-e5-block16-v2"),
        KvCacheEncoding::Mxfp8E4 | KvCacheEncoding::Mxfp8E5 => None,
        _ => unreachable!(),
    }
}

fn f32_to_bf16(value: f32) -> u16 {
    if value.is_nan() {
        return ((value.to_bits() >> 16) as u16) | 0x0040;
    }
    let bits = value.to_bits();
    ((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16) as u16
}

fn bf16_to_f32(value: u16) -> f32 {
    f32::from_bits(u32::from(value) << 16)
}

fn words_to_bytes(words: &[u16]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

fn source_words(head_dim: usize, value_plane: bool, variant: KvFp8PhysicalVariant) -> Vec<u16> {
    let mut words = Vec::with_capacity(HEADS * head_dim);
    let maximum = match variant {
        KvFp8PhysicalVariant::E4M3FnuZ => 240.0,
        KvFp8PhysicalVariant::OcpE4M3Fn => 448.0,
        KvFp8PhysicalVariant::OcpE5M2 => 57_344.0,
    };
    for head in 0..HEADS {
        for column in 0..head_dim {
            let multiplier = if value_plane { 1.0 / 64.0 } else { 1.0 };
            let base = match column {
                0 => 0.0,
                1 => -0.0,
                2 => 2.0_f32.powi(-120),
                3 if value_plane => maximum / (multiplier * 64.0),
                3 => maximum,
                4 if !value_plane => f32::NAN,
                5 if !value_plane => f32::INFINITY,
                6 if !value_plane => f32::NEG_INFINITY,
                7 => match variant {
                    KvFp8PhysicalVariant::OcpE5M2 => 65_535.0,
                    KvFp8PhysicalVariant::OcpE4M3Fn | KvFp8PhysicalVariant::E4M3FnuZ => 511.0,
                },
                _ => (((column * 13 + head * 7) % 61) as f32 - 30.0) / 8.0,
            };
            words.push(f32_to_bf16(base * multiplier));
        }
    }
    words
}

fn finite_attention_words(head_dim: usize, value_plane: bool) -> Vec<u16> {
    let multiplier = if value_plane { 1.0 / 64.0 } else { 1.0 };
    (0..HEADS * head_dim)
        .map(|index| {
            let head = index / head_dim;
            let column = index % head_dim;
            let value = (((column * 13 + head * 7) % 61) as f32 - 30.0) / 8.0;
            f32_to_bf16(value * multiplier)
        })
        .collect()
}

fn host_quantized(
    words: &[u16],
    head_dim: usize,
    variant: KvFp8PhysicalVariant,
    format: EvidenceFormat,
) -> Result<HostQuantized, String> {
    let input = words.iter().copied().map(bf16_to_f32).collect::<Vec<_>>();
    let (values, scales, dequantized, block_size) = match format {
        EvidenceFormat::Block16 => {
            let encoded = quantize_kv_fp8_block16(&input, HEADS, head_dim, variant)
                .map_err(|error| error.to_string())?;
            (
                encoded.values().to_vec(),
                encoded.scales().to_vec(),
                encoded.dequantize().map_err(|error| error.to_string())?,
                16,
            )
        }
        EvidenceFormat::Mxfp8 => {
            let descriptor = KvMxfp8Descriptor::new(
                match variant {
                    KvFp8PhysicalVariant::OcpE4M3Fn => KvCacheEncoding::Mxfp8E4,
                    KvFp8PhysicalVariant::OcpE5M2 => KvCacheEncoding::Mxfp8E5,
                    KvFp8PhysicalVariant::E4M3FnuZ => {
                        return Err("FNUZ is not standard MXFP8".to_owned());
                    }
                },
                variant,
            )
            .map_err(|error| error.to_string())?;
            let encoded = quantize_kv_mxfp8(&input, HEADS, head_dim, descriptor)
                .map_err(|error| error.to_string())?;
            (
                encoded.values().to_vec(),
                encoded.scales().to_vec(),
                encoded.dequantize().map_err(|error| error.to_string())?,
                32,
            )
        }
    };
    let blocks = head_dim.div_ceil(block_size);
    let mut padding_zero = true;
    for row in 0..HEADS {
        if head_dim % block_size != 0 {
            let begin = (row * blocks + blocks - 1) * block_size + head_dim % block_size;
            padding_zero &= values[begin..(row * blocks + blocks) * block_size]
                .iter()
                .all(|byte| *byte == 0);
        }
    }
    Ok(HostQuantized {
        // State images expose the physical row stride.  Preserve every
        // block-tail byte so compact native storage cannot accidentally pass.
        values,
        scales,
        dequantized,
        padding_zero,
    })
}

fn host_evidence(format: EvidenceFormat) -> HostEvidence {
    let variants: &[KvFp8PhysicalVariant] = match format {
        EvidenceFormat::Block16 => &[
            KvFp8PhysicalVariant::E4M3FnuZ,
            KvFp8PhysicalVariant::OcpE4M3Fn,
            KvFp8PhysicalVariant::OcpE5M2,
        ],
        EvidenceFormat::Mxfp8 => &[KvFp8PhysicalVariant::OcpE4M3Fn],
    };
    let dimensions = match format {
        EvidenceFormat::Block16 => DIMENSIONS.as_slice(),
        EvidenceFormat::Mxfp8 => MX_DIMENSIONS.as_slice(),
    };
    let mut pass = true;
    let mut padding = true;
    let mut independent = true;
    for &variant in variants {
        for &dimension in dimensions {
            let key = source_words(dimension, false, variant);
            let value = source_words(dimension, true, variant);
            match (
                host_quantized(&key, dimension, variant, format),
                host_quantized(&value, dimension, variant, format),
            ) {
                (Ok(key), Ok(value)) => {
                    padding &= key.padding_zero && value.padding_zero;
                    independent &= key.scales != value.scales;
                }
                _ => pass = false,
            }
        }
    }
    pass &= padding && independent;
    HostEvidence {
        pass,
        variants: match format {
            EvidenceFormat::Block16 => vec!["E4M3-FNUZ", "E4M3-OCP", "E5M2-software"],
            EvidenceFormat::Mxfp8 => vec!["E4M3-OCP"],
        },
        head_dimensions: dimensions.to_vec(),
        special_values: match format {
            EvidenceFormat::Block16 => vec![
                "zero",
                "tiny",
                "max-finite",
                "standard-mx-saturation-boundary",
                "nan",
                "positive-infinity",
                "negative-infinity",
                "positive-zero",
                "negative-zero",
            ],
            EvidenceFormat::Mxfp8 => vec![
                "zero",
                "tiny",
                "max-finite",
                "nan",
                "positive-infinity",
                "negative-infinity",
                "positive-zero",
                "negative-zero",
            ],
        },
        tail_padding_zero: padding,
        key_value_scales_independent: independent,
    }
}

fn binding(
    session: &ExecutionSession,
    buffer: &sllm_core::ExecutionBuffer,
    shape: &[usize],
    access: AccessMode,
) -> Result<sllm_core::OwnedTensorBinding, String> {
    let view = TensorView::with_encoding(DType::Bf16, Encoding::Unquantized, shape)
        .map_err(|error| error.to_string())?;
    session
        .bind(buffer, view, access)
        .map_err(|error| error.to_string())
}

fn upload(
    session: &ExecutionSession,
    queue: &sllm_core::ExecutionQueue,
    buffer: &sllm_core::ExecutionBuffer,
    words: &[u16],
) -> Result<(), String> {
    let bytes = words_to_bytes(words);
    let mut transfer = session
        .upload(
            queue,
            buffer
                .range(0, bytes.len() as u64)
                .map_err(|error| error.to_string())?,
            Arc::<[u8]>::from(bytes),
        )
        .map_err(|error| error.to_string())?;
    if transfer.wait(WAIT).map_err(|error| error.to_string())? != ExecutionState::Success {
        return Err("upload failed".to_owned());
    }
    Ok(())
}

fn plane(
    image: &sllm_core::ExecutionStateImageV1,
    kind: StatePlaneKindV1,
) -> Result<&[u8], String> {
    image
        .planes()
        .iter()
        .find(|plane| plane.plane == kind)
        .map(|plane| plane.bytes.as_slice())
        .ok_or_else(|| format!("state image omitted {kind:?}"))
}

fn run_attention(
    session: &ExecutionSession,
    queue: &sllm_core::ExecutionQueue,
    state: &sllm_core::KvState,
    expected_values: &[f32],
) -> Result<(bool, bool), String> {
    let query_words = vec![0_u16; Q_HEADS * 256];
    let bytes = words_to_bytes(&query_words);
    let query_buffer = session
        .allocate(bytes.len() as u64)
        .map_err(|error| error.to_string())?;
    let output_buffer = session
        .allocate(bytes.len() as u64)
        .map_err(|error| error.to_string())?;
    upload(session, queue, &query_buffer, &query_words)?;
    let shape = [1, Q_HEADS, 256];
    let query = binding(session, &query_buffer, &shape, AccessMode::Read)?;
    let output = binding(session, &output_buffer, &shape, AccessMode::Write)?;
    let descriptor =
        sllm_core::CausalAttentionDescriptor::new(0, 1, 1).map_err(|error| error.to_string())?;
    let mut submission = session
        .causal_attention(state, queue, query, output, descriptor)
        .map_err(|error| error.to_string())?;
    let dispatch = submission.dispatch().clone();
    if submission.wait(WAIT).map_err(|error| error.to_string())? != ExecutionState::Success {
        return Err("attention failed".to_owned());
    }
    drop(submission);
    let mut readback = session
        .readback(
            queue,
            output_buffer
                .range(0, bytes.len() as u64)
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
    if readback.wait(WAIT).map_err(|error| error.to_string())? != ExecutionState::Success {
        return Err("attention readback failed".to_owned());
    }
    let mut actual = vec![0_u8; bytes.len()];
    readback
        .read_into(&mut actual)
        .map_err(|error| error.to_string())?;
    let actual_words = actual
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect::<Vec<_>>();
    let mut expected = Vec::with_capacity(Q_HEADS * 256);
    for query_head in 0..Q_HEADS {
        let kv_head = query_head / (Q_HEADS / HEADS);
        expected.extend(
            expected_values[kv_head * 256..(kv_head + 1) * 256]
                .iter()
                .copied()
                .map(f32_to_bf16),
        );
    }
    let numerical_match = actual_words
        .iter()
        .zip(&expected)
        .all(|(actual, expected)| {
            actual == expected || (bf16_to_f32(*actual).is_nan() && bf16_to_f32(*expected).is_nan())
        });
    Ok((
        dispatch.backend != 0
            && !dispatch.fallback_allowed
            && !dispatch.fallback_used
            && dispatch.dispatch_count > 0
            && dispatch.kernel_symbol.contains("packed"),
        numerical_match,
    ))
}

fn run_cases(
    session: &ExecutionSession,
    config: &Config,
) -> Result<(Vec<CaseEvidence>, u64, u64), String> {
    let queue = session.create_queue().map_err(|error| error.to_string())?;
    let mut cases = Vec::new();
    let mut append_dispatches = 0_u64;
    let mut attention_dispatches = 0_u64;
    let dimensions = match config.format {
        EvidenceFormat::Block16 => DIMENSIONS.as_slice(),
        EvidenceFormat::Mxfp8 => MX_DIMENSIONS.as_slice(),
    };
    for (index, &dimension) in dimensions.iter().enumerate() {
        let descriptor = match config.format {
            EvidenceFormat::Block16 => KvStateDescriptor::new_with_kv_fp8_block16(
                index as u32,
                1,
                HEADS,
                dimension,
                config.encoding,
                config.variant,
            ),
            EvidenceFormat::Mxfp8 => KvStateDescriptor::new_with_kv_mxfp8(
                index as u32,
                1,
                HEADS,
                dimension,
                config.encoding,
                config.variant,
            ),
        }
        .map_err(|error| error.to_string())?;
        let state = session
            .create_kv_state(descriptor)
            .map_err(|error| error.to_string())?;
        let (key_words, value_words) = if dimension == 256 {
            (
                finite_attention_words(dimension, false),
                finite_attention_words(dimension, true),
            )
        } else {
            (
                source_words(dimension, false, config.variant),
                source_words(dimension, true, config.variant),
            )
        };
        let key_buffer = session
            .allocate((key_words.len() * 2) as u64)
            .map_err(|error| error.to_string())?;
        let value_buffer = session
            .allocate((value_words.len() * 2) as u64)
            .map_err(|error| error.to_string())?;
        upload(session, &queue, &key_buffer, &key_words)?;
        upload(session, &queue, &value_buffer, &value_words)?;
        let shape = [1, HEADS, dimension];
        let key = binding(session, &key_buffer, &shape, AccessMode::Read)?;
        let value = binding(session, &value_buffer, &shape, AccessMode::Read)?;
        let mut append = session
            .append_kv_state(&state, &queue, key, value, 0, 0)
            .map_err(|error| error.to_string())?;
        let append_dispatch = append.dispatch().clone();
        if append.wait(WAIT).map_err(|error| error.to_string())? != ExecutionState::Success {
            return Err(format!("head_dim={dimension} append failed"));
        }
        append_dispatches += u64::from(append_dispatch.dispatch_count);
        drop(append);
        let image = session
            .export_kv_state_image(&state)
            .map_err(|error| format!("head_dim={dimension} export: {error}"))?;
        let key_oracle = host_quantized(&key_words, dimension, config.variant, config.format)?;
        let value_oracle = host_quantized(&value_words, dimension, config.variant, config.format)?;
        let value_bytes_exact = plane(&image, StatePlaneKindV1::KvKey)? == key_oracle.values
            && plane(&image, StatePlaneKindV1::KvValue)? == value_oracle.values;
        let key_scales_exact = plane(&image, StatePlaneKindV1::KvKeyScale)? == key_oracle.scales;
        let value_scales_exact =
            plane(&image, StatePlaneKindV1::KvValueScale)? == value_oracle.scales;
        let append_direct = append_dispatch.backend != 0
            && !append_dispatch.fallback_allowed
            && !append_dispatch.fallback_used
            && append_dispatch.dispatch_count > 0
            && append_dispatch.kernel_symbol.contains(match config.format {
                EvidenceFormat::Block16 => "block16",
                EvidenceFormat::Mxfp8 => "mxfp8",
            });
        let (attention_direct, numerical_match) = if dimension == 256 {
            attention_dispatches += 1;
            run_attention(session, &queue, &state, &value_oracle.dequantized)?
        } else {
            (false, true)
        };
        cases.push(CaseEvidence {
            head_dim: dimension,
            value_bytes_exact,
            key_scales_exact,
            value_scales_exact,
            key_value_scales_independent: key_oracle.scales != value_oracle.scales,
            tail_padding_zero: key_oracle.padding_zero && value_oracle.padding_zero,
            append_direct,
            attention_direct,
            numerical_match,
        });
    }
    Ok((cases, append_dispatches, attention_dispatches))
}

fn base_report(
    config: &Config,
    host: HostEvidence,
    binary_sha256: String,
    state: &'static str,
    error: Option<String>,
) -> Report {
    Report {
        schema: match config.format {
            EvidenceFormat::Block16 => {
                "https://sllm.dev/schema/phase53-kv-fp8-block16-evidence-v2.schema.json"
            }
            EvidenceFormat::Mxfp8 => {
                "https://sllm.dev/schema/phase53-kv-mxfp8-evidence-v1.schema.json"
            }
        },
        schema_version: match config.format {
            EvidenceFormat::Block16 => "sllm-phase53-kv-fp8-block16-evidence-v2",
            EvidenceFormat::Mxfp8 => "sllm-phase53-kv-mxfp8-evidence-v1",
        },
        state,
        target: config.target.clone(),
        device_index: config.device_index,
        encoding: encoding_name(config.encoding),
        physical_variant: variant_name(config.variant, config.format),
        descriptor_id: block16_descriptor_id(config.encoding),
        scale_recipe: (config.format == EvidenceFormat::Block16)
            .then_some("standard-mx-floor-power-v1"),
        policy_sha256: config.policy_sha256.clone(),
        binary_sha256,
        host,
        cases: Vec::new(),
        execution: ExecutionEvidence {
            selected_backend: "hip",
            gpu_execution: false,
            fallback_allowed: false,
            fallback_used: false,
            append_dispatches: 0,
            attention_dispatches: 0,
        },
        cleanup: CleanupEvidence {
            retryable: 0,
            durable: 0,
            terminal_zero: false,
        },
        error,
    }
}

fn run(config: &Config) -> Report {
    let host = host_evidence(config.format);
    let binary_sha256 = env::current_exe()
        .ok()
        .and_then(|path| fs::read(path).ok())
        .map(|bytes| digest(&bytes))
        .unwrap_or_else(|| digest(b"unavailable-binary"));
    let backend = match HipBackend::connect() {
        Ok(backend) => backend,
        Err(error) => {
            return base_report(
                config,
                host,
                binary_sha256,
                "UNAVAILABLE",
                Some(error.to_string()),
            );
        }
    };
    let request = match ExecutionSessionRequest::new(config.device_index, config.target.clone()) {
        Ok(request) => request,
        Err(error) => {
            return base_report(config, host, binary_sha256, "FAIL", Some(error.to_string()));
        }
    };
    let session = match backend.open_execution_session(request) {
        Ok(session) => session,
        Err(error) => {
            return base_report(
                config,
                host,
                binary_sha256,
                "UNAVAILABLE",
                Some(error.to_string()),
            );
        }
    };
    let result = run_cases(&session, config);
    let cleanup = session.shutdown(SHUTDOWN);
    match (result, cleanup) {
        (Ok((cases, append_dispatches, attention_dispatches)), Ok(cleanup)) => {
            let pass = host.pass
                && cases.iter().all(|case| {
                    case.value_bytes_exact
                        && case.key_scales_exact
                        && case.value_scales_exact
                        && case.key_value_scales_independent
                        && case.tail_padding_zero
                        && case.append_direct
                        && case.numerical_match
                        && (case.head_dim != 256 || case.attention_direct)
                })
                && cleanup.retryable_cleanup == 0
                && cleanup.durable_quarantine == 0;
            Report {
                schema: match config.format {
                    EvidenceFormat::Block16 => {
                        "https://sllm.dev/schema/phase53-kv-fp8-block16-evidence-v2.schema.json"
                    }
                    EvidenceFormat::Mxfp8 => {
                        "https://sllm.dev/schema/phase53-kv-mxfp8-evidence-v1.schema.json"
                    }
                },
                schema_version: match config.format {
                    EvidenceFormat::Block16 => "sllm-phase53-kv-fp8-block16-evidence-v2",
                    EvidenceFormat::Mxfp8 => "sllm-phase53-kv-mxfp8-evidence-v1",
                },
                state: if pass { "PASS" } else { "FAIL" },
                target: config.target.clone(),
                device_index: config.device_index,
                encoding: encoding_name(config.encoding),
                physical_variant: variant_name(config.variant, config.format),
                descriptor_id: block16_descriptor_id(config.encoding),
                scale_recipe: (config.format == EvidenceFormat::Block16)
                    .then_some("standard-mx-floor-power-v1"),
                policy_sha256: config.policy_sha256.clone(),
                binary_sha256,
                host,
                cases,
                execution: ExecutionEvidence {
                    selected_backend: "hip",
                    gpu_execution: true,
                    fallback_allowed: false,
                    fallback_used: false,
                    append_dispatches,
                    attention_dispatches,
                },
                cleanup: CleanupEvidence {
                    retryable: cleanup.retryable_cleanup,
                    durable: cleanup.durable_quarantine,
                    terminal_zero: cleanup.retryable_cleanup == 0
                        && cleanup.durable_quarantine == 0,
                },
                error: None,
            }
        }
        (Err(error), Ok(cleanup)) => {
            let mut report = base_report(config, host, binary_sha256, "FAIL", Some(error));
            report.cleanup = CleanupEvidence {
                retryable: cleanup.retryable_cleanup,
                durable: cleanup.durable_quarantine,
                terminal_zero: cleanup.retryable_cleanup == 0 && cleanup.durable_quarantine == 0,
            };
            report
        }
        (operation, cleanup) => base_report(
            config,
            host,
            binary_sha256,
            "FAIL",
            Some(format!("operation={operation:?}; cleanup={cleanup:?}")),
        ),
    }
}

fn main() -> ExitCode {
    match parse() {
        Ok(config) => {
            let report = run(&config);
            let passed = report.state == "PASS";
            match serde_json::to_string_pretty(&report) {
                Ok(value) => {
                    println!("{value}");
                    if passed {
                        ExitCode::SUCCESS
                    } else {
                        ExitCode::FAILURE
                    }
                }
                Err(error) => {
                    eprintln!("serialize report: {error}");
                    ExitCode::from(2)
                }
            }
        }
        Err(error) => {
            eprintln!("sllm-kv-mxfp8-e4-evidence: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_matrix_covers_tail_specials_and_independent_planes() {
        let evidence = host_evidence(EvidenceFormat::Block16);
        assert!(evidence.pass, "{evidence:?}");
        assert!(evidence.tail_padding_zero);
        assert!(evidence.key_value_scales_independent);
        assert!(
            evidence
                .special_values
                .contains(&"standard-mx-saturation-boundary")
        );
    }

    #[test]
    fn block16_oracle_uses_standard_mx_floor_power_scale() {
        for (value, variant) in [
            (511.0, KvFp8PhysicalVariant::OcpE4M3Fn),
            (511.0, KvFp8PhysicalVariant::E4M3FnuZ),
            (65_535.0, KvFp8PhysicalVariant::OcpE5M2),
        ] {
            let encoded = quantize_kv_fp8_block16(&[value], 1, 1, variant).unwrap();
            assert_eq!(encoded.scales(), &[127]);
        }
    }

    #[test]
    fn host_mxfp8_matrix_covers_block32_tails_and_ocp_e4() {
        let evidence = host_evidence(EvidenceFormat::Mxfp8);
        assert!(evidence.pass);
        assert!(evidence.tail_padding_zero);
        assert!(evidence.key_value_scales_independent);
        assert_eq!(evidence.head_dimensions, MX_DIMENSIONS);
        assert_eq!(evidence.variants, ["E4M3-OCP"]);
    }

    #[test]
    fn bf16_roundtrip_keeps_signed_zero_and_nonfinite_classes() {
        assert_eq!(f32_to_bf16(-0.0), 0x8000);
        assert!(bf16_to_f32(f32_to_bf16(f32::NAN)).is_nan());
        assert!(bf16_to_f32(f32_to_bf16(f32::INFINITY)).is_infinite());
    }
}
