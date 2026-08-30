//! Phase 54 direct GPU oracle for research-only KV FP8 block16 recipes.
//!
//! This binary deliberately has a separate identity from the Phase 53
//! production evidence runner.  It consumes one closed candidate spec,
//! selects the process-local research recipe pair, and compares the native
//! append/export path with `quantize_kv_fp8_block16_research`.  The attention
//! case appends two tokens and uses a non-zero query so the K plane is part of
//! the numerical oracle rather than merely checking V reconstruction.
//! The Phase 54 V/O candidates use that same attention case as their reviewed
//! layer semantic carrier: V is permuted before append and the returned output
//! is restored with the self-inverse permutation before scalar comparison.

use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sllm_core::{
    AccessMode, Backend, DType, Encoding, ExecutionSession, ExecutionSessionRequest,
    ExecutionState, KvCacheEncoding, KvFp8PhysicalVariant, KvFp8ResearchScaleRecipe,
    KvStateDescriptor, PHASE54_KQ_TRANSFORM_DIGEST, PHASE54_KQ_TRANSFORM_SEMANTICS,
    PHASE54_VO_TRANSFORM_DIGEST, PHASE54_VO_TRANSFORM_LAYERS_19_31_DIGEST,
    PHASE54_VO_TRANSFORM_LAYERS_19_31_SELECTOR, PHASE54_VO_TRANSFORM_LAYERS_19_31_SEMANTICS,
    PHASE54_VO_TRANSFORM_SELECTOR, PHASE54_VO_TRANSFORM_SEMANTICS, StatePlaneKindV1, TensorView,
    quantize_kv_fp8_block16_research, transpose_bf16_words, transpose16x16_index,
};
use sllm_hip::HipBackend;

const HEADS: usize = 4;
const Q_HEADS: usize = 16;
const ATTENTION_DIM: usize = 256;
const DIMENSIONS: [usize; 6] = [15, 16, 17, 255, 256, 257];
const WAIT: Duration = Duration::from_secs(30);
const SHUTDOWN: Duration = Duration::from_secs(16);

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
enum Recipe {
    Floor,
    Ceil,
    NearestEven,
    Parent32Duplicate,
}

impl Recipe {
    #[allow(dead_code)]
    const fn runtime_value(self) -> u32 {
        match self {
            Self::Floor => 0,
            Self::Ceil => 1,
            Self::NearestEven => 2,
            Self::Parent32Duplicate => 3,
        }
    }

    const fn id(self) -> &'static str {
        match self {
            Self::Floor => "floor",
            Self::Ceil => "ceil",
            Self::NearestEven => "nearest-even",
            Self::Parent32Duplicate => "parent32-duplicate",
        }
    }

    const fn core(self) -> KvFp8ResearchScaleRecipe {
        match self {
            Self::Floor => KvFp8ResearchScaleRecipe::Floor,
            Self::Ceil => KvFp8ResearchScaleRecipe::Ceil,
            Self::NearestEven => KvFp8ResearchScaleRecipe::NearestEvenExponent,
            Self::Parent32Duplicate => KvFp8ResearchScaleRecipe::Parent32Duplicate,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct CandidateSpec {
    schema_version: String,
    candidate_id: String,
    scale_selector: String,
    rounding: String,
    k_recipe: Recipe,
    v_recipe: Recipe,
    transform: String,
    calibration_digest: Option<String>,
    descriptor_compatibility: String,
}

impl CandidateSpec {
    fn parent32_duplicate_candidate(&self) -> bool {
        self.k_recipe == Recipe::Parent32Duplicate
            && self.v_recipe == Recipe::Parent32Duplicate
            && self.transform == "none"
    }

    fn production_control(&self) -> bool {
        self.k_recipe == Recipe::Floor && self.v_recipe == Recipe::Floor && self.transform == "none"
    }

    fn transform_candidate(&self) -> bool {
        self.k_recipe == Recipe::Floor
            && self.v_recipe == Recipe::Floor
            && self.transform == "transpose16x16-all-full"
    }

    fn vo_transform_candidate(&self) -> bool {
        self.k_recipe == Recipe::Floor
            && self.v_recipe == Recipe::Floor
            && self.transform == PHASE54_VO_TRANSFORM_SELECTOR
    }

    fn vo_layers19_31_transform_candidate(&self) -> bool {
        self.k_recipe == Recipe::Floor
            && self.v_recipe == Recipe::Floor
            && self.transform == PHASE54_VO_TRANSFORM_LAYERS_19_31_SELECTOR
    }

    fn expected_id(&self) -> String {
        if self.production_control() {
            "production-control-v2".to_owned()
        } else if self.transform_candidate() {
            "phase54-kq-transpose16x16-all-full-v1".to_owned()
        } else if self.vo_transform_candidate() {
            "phase54-vo-transpose16x16-layer19-v1".to_owned()
        } else if self.vo_layers19_31_transform_candidate() {
            "phase54-vo-transpose16x16-layers19-31-v1".to_owned()
        } else {
            format!(
                "phase54-k-{}-v-{}-v1",
                self.k_recipe.id(),
                self.v_recipe.id()
            )
        }
    }

    fn validate(&self) -> Result<(), String> {
        let compatibility = if self.production_control() {
            "exact-production-v2"
        } else {
            "research-build-semantic-override-not-v2-compatible"
        };
        if self.schema_version != "sllm-phase54-kv-candidate-spec-v1"
            || self.candidate_id != self.expected_id()
            || self.scale_selector != "independent-k-v-closed-enum-v1"
            || self.rounding != "nearest-even"
            || self.calibration_digest.is_some()
            || self.descriptor_compatibility != compatibility
            || ((self.k_recipe == Recipe::Parent32Duplicate
                || self.v_recipe == Recipe::Parent32Duplicate)
                && !self.parent32_duplicate_candidate())
            || (!self.transform_candidate()
                && !self.vo_transform_candidate()
                && !self.vo_layers19_31_transform_candidate()
                && self.transform != "none")
        {
            return Err(
                "candidate spec does not match the closed Phase 54 recipe identity".to_owned(),
            );
        }
        Ok(())
    }
}

#[derive(Clone)]
struct Config {
    device_index: u32,
    target: String,
    encoding: KvCacheEncoding,
    variant: KvFp8PhysicalVariant,
    candidate: CandidateSpec,
    candidate_spec_sha256: String,
    output: Option<PathBuf>,
}

#[derive(Clone, Copy)]
enum ValuesKind {
    Boundary,
    SignedZero,
    RecipeDistinguishing,
    Attention,
}

struct HostQuantized {
    values: Vec<u8>,
    scales: Vec<u8>,
    dequantized: Vec<f32>,
    padding_zero: bool,
}

#[derive(Debug, Serialize)]
struct HostEvidence {
    pass: bool,
    head_dimensions: Vec<usize>,
    recipes: [String; 2],
    signed_zero_only: bool,
    recipe_distinguishing: bool,
    tail_padding_zero: bool,
    finite: bool,
}

#[derive(Debug, Serialize)]
struct CaseEvidence {
    id: String,
    head_dim: usize,
    token_count: usize,
    transform_applied: bool,
    signed_zero_only: bool,
    recipe_distinguishing: bool,
    key_values_exact: bool,
    value_values_exact: bool,
    key_scales_exact: bool,
    value_scales_exact: bool,
    tail_padding_zero: bool,
    append_direct: bool,
    attention_direct: bool,
    attention_numerical_match: bool,
    attention_key_contributes: bool,
    finite: bool,
}

fn case_passes(case: &CaseEvidence) -> bool {
    let attention_pass = if case.id == "attention-kv2" {
        case.attention_direct && case.attention_numerical_match && case.attention_key_contributes
    } else {
        !case.attention_direct
    };
    case.key_values_exact
        && case.value_values_exact
        && case.key_scales_exact
        && case.value_scales_exact
        && case.tail_padding_zero
        && case.append_direct
        && case.finite
        && attention_pass
}

#[derive(Debug, Serialize)]
struct TransformEvidence {
    identity: &'static str,
    semantics: &'static str,
    digest: &'static str,
    key_rows_transformed: bool,
    query_rows_transformed: bool,
    value_untransformed: bool,
    qk_invariant: bool,
    qk_max_abs_delta: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    value_rows_transformed: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_rows_inverse_transformed: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    vo_invariant: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    vo_max_abs_delta: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    semantic_layer: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    semantic_layers: Option<Vec<u32>>,
}

#[derive(Debug, Serialize)]
struct ExecutionEvidence {
    selected_backend: &'static str,
    gpu_execution: bool,
    fallback_allowed: bool,
    fallback_used: bool,
    append_dispatches: u64,
    attention_dispatches: u64,
    sequential_residents: bool,
}

#[derive(Debug, Serialize)]
struct CleanupEvidence {
    retryable: usize,
    durable: usize,
    terminal_zero: bool,
}

#[derive(Debug, Serialize)]
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
    production_descriptor_id: Option<&'static str>,
    descriptor_compatibility: String,
    candidate_spec: CandidateSpec,
    candidate_spec_sha256: String,
    binary_sha256: String,
    host: HostEvidence,
    cases: Vec<CaseEvidence>,
    #[serde(skip_serializing_if = "Option::is_none")]
    transform_evidence: Option<TransformEvidence>,
    execution: ExecutionEvidence,
    cleanup: CleanupEvidence,
    error: Option<String>,
}

fn digest(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}

fn parse_candidate(path: &Path) -> Result<(CandidateSpec, String), String> {
    let bytes = fs::read(path).map_err(|error| format!("read candidate spec: {error}"))?;
    let candidate: CandidateSpec =
        serde_json::from_slice(&bytes).map_err(|error| format!("parse candidate spec: {error}"))?;
    candidate.validate()?;
    let canonical = serde_json::to_vec(&candidate).map_err(|error| error.to_string())?;
    Ok((candidate, digest(&canonical)))
}

fn target_config(target: &str) -> Result<(KvCacheEncoding, KvFp8PhysicalVariant), String> {
    match target {
        "gfx1030" => Ok((
            KvCacheEncoding::Fp8E5M2Block16,
            KvFp8PhysicalVariant::OcpE5M2,
        )),
        "gfx1201" => Ok((
            KvCacheEncoding::Fp8E4M3Block16,
            KvFp8PhysicalVariant::OcpE4M3Fn,
        )),
        _ => Err("--target must be exact gfx1030 or gfx1201".to_owned()),
    }
}

fn parse() -> Result<Config, String> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let (device_index, target, supplied_encoding, candidate_path, output) = if args
        .first()
        .is_some_and(|argument| argument.starts_with("--"))
    {
        let mut device_index = None;
        let mut target = None;
        let mut supplied_encoding = None;
        let mut candidate_path = None;
        let mut output = None;
        let mut arguments = args.into_iter();
        while let Some(argument) = arguments.next() {
            let value = |label: &str, arguments: &mut std::vec::IntoIter<String>| {
                arguments
                    .next()
                    .ok_or_else(|| format!("{label} needs a value"))
            };
            match argument.as_str() {
                "--device-index" => {
                    device_index = Some(
                        value("--device-index", &mut arguments)?
                            .parse::<u32>()
                            .map_err(|_| "--device-index must be u32".to_owned())?,
                    );
                }
                "--target" => target = Some(value("--target", &mut arguments)?),
                "--encoding" => supplied_encoding = Some(value("--encoding", &mut arguments)?),
                "--candidate-spec" => {
                    candidate_path = Some(value("--candidate-spec", &mut arguments)?)
                }
                "--output" => output = Some(PathBuf::from(value("--output", &mut arguments)?)),
                other => return Err(format!("unexpected argument {other}")),
            }
        }
        (
            device_index.ok_or("missing --device-index")?,
            target.ok_or("missing --target")?,
            supplied_encoding,
            candidate_path.ok_or("missing --candidate-spec")?,
            output,
        )
    } else {
        if args.len() < 3 || args.len() > 5 {
            return Err(
                "usage: DEVICE_INDEX TARGET [ENCODING] CANDIDATE_SPEC_JSON [OUTPUT_JSON]"
                    .to_owned(),
            );
        }
        let device_index = args[0]
            .parse::<u32>()
            .map_err(|_| "DEVICE_INDEX must be u32".to_owned())?;
        let target = args[1].clone();
        let (supplied_encoding, candidate_index) = if args.len() >= 4
            && matches!(
                args[2].as_str(),
                "kv-fp8-e4-block16" | "kv-fp8-e5-block16" | "e4" | "e5"
            ) {
            (Some(args[2].clone()), 3)
        } else {
            (None, 2)
        };
        let candidate_path = args
            .get(candidate_index)
            .cloned()
            .ok_or("missing candidate spec path")?;
        let output = args.get(candidate_index + 1).map(PathBuf::from);
        (
            device_index,
            target,
            supplied_encoding,
            candidate_path,
            output,
        )
    };
    let (encoding, variant) = target_config(&target)?;
    if let Some(supplied) = supplied_encoding.as_deref() {
        let expected = encoding_name(encoding);
        let matches = supplied == expected
            || (encoding == KvCacheEncoding::Fp8E4M3Block16 && supplied == "e4")
            || (encoding == KvCacheEncoding::Fp8E5M2Block16 && supplied == "e5");
        if !matches {
            return Err(format!(
                "encoding {supplied} is incompatible with target {target} (expected {expected})"
            ));
        }
    }
    let (candidate, candidate_spec_sha256) = parse_candidate(Path::new(&candidate_path))?;
    Ok(Config {
        device_index,
        target,
        encoding,
        variant,
        candidate,
        candidate_spec_sha256,
        output,
    })
}

fn encoding_name(encoding: KvCacheEncoding) -> &'static str {
    match encoding {
        KvCacheEncoding::Fp8E4M3Block16 => "kv-fp8-e4-block16",
        KvCacheEncoding::Fp8E5M2Block16 => "kv-fp8-e5-block16",
        _ => unreachable!(),
    }
}

fn variant_name(variant: KvFp8PhysicalVariant) -> &'static str {
    match variant {
        KvFp8PhysicalVariant::OcpE4M3Fn => "E4M3-OCP",
        KvFp8PhysicalVariant::OcpE5M2 => "E5M2-software",
        KvFp8PhysicalVariant::E4M3FnuZ => "E4M3-FNUZ",
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

fn source_words(
    token_count: usize,
    head_dim: usize,
    value_plane: bool,
    variant: KvFp8PhysicalVariant,
    kind: ValuesKind,
) -> Vec<u16> {
    let maximum = match variant {
        KvFp8PhysicalVariant::E4M3FnuZ => 240.0,
        KvFp8PhysicalVariant::OcpE4M3Fn => 448.0,
        KvFp8PhysicalVariant::OcpE5M2 => 57_344.0,
    };
    let mut words = Vec::with_capacity(token_count * HEADS * head_dim);
    for token in 0..token_count {
        for head in 0..HEADS {
            for column in 0..head_dim {
                let value = match kind {
                    ValuesKind::SignedZero => {
                        if (token + head + column) % 2 == 0 {
                            0.0
                        } else {
                            -0.0
                        }
                    }
                    ValuesKind::RecipeDistinguishing => {
                        let base = if column == 0 { 3.25 } else { 0.75 };
                        base + (token * 3 + head) as f32 * 0.03125
                    }
                    ValuesKind::Attention => {
                        let base =
                            (((column * 13 + head * 7 + token * 11) % 61) as f32 - 30.0) / 32.0;
                        if value_plane { base / 8.0 } else { base }
                    }
                    ValuesKind::Boundary => {
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
                                KvFp8PhysicalVariant::OcpE4M3Fn
                                | KvFp8PhysicalVariant::E4M3FnuZ => 511.0,
                            },
                            _ => (((column * 13 + head * 7 + token * 5) % 61) as f32 - 30.0) / 8.0,
                        };
                        base * multiplier
                    }
                };
                words.push(f32_to_bf16(value));
            }
        }
    }
    words
}

fn host_quantized(
    words: &[u16],
    token_count: usize,
    head_dim: usize,
    variant: KvFp8PhysicalVariant,
    recipe: Recipe,
) -> Result<HostQuantized, String> {
    let rows = token_count
        .checked_mul(HEADS)
        .ok_or_else(|| "host row count overflow".to_owned())?;
    let input = words.iter().copied().map(bf16_to_f32).collect::<Vec<_>>();
    let (encoded, _) =
        quantize_kv_fp8_block16_research(&input, rows, head_dim, variant, recipe.core())
            .map_err(|error| error.to_string())?;
    let values = encoded.values().to_vec();
    let scales = encoded.scales().to_vec();
    let dequantized = encoded.dequantize().map_err(|error| error.to_string())?;
    let blocks = head_dim.div_ceil(16);
    let mut padding_zero = true;
    if head_dim % 16 != 0 {
        for row in 0..rows {
            let base = (row * blocks + blocks - 1) * 16;
            padding_zero &= values[base + head_dim % 16..(row * blocks + blocks) * 16]
                .iter()
                .all(|byte| *byte == 0);
        }
    }
    Ok(HostQuantized {
        values,
        scales,
        dequantized,
        padding_zero,
    })
}

fn host_evidence(config: &Config) -> HostEvidence {
    let mut pass = true;
    let mut padding = true;
    let mut finite = true;
    for &dimension in &DIMENSIONS {
        let key = source_words(1, dimension, false, config.variant, ValuesKind::Boundary);
        let value = source_words(1, dimension, true, config.variant, ValuesKind::Boundary);
        match (
            host_quantized(
                &key,
                1,
                dimension,
                config.variant,
                config.candidate.k_recipe,
            ),
            host_quantized(
                &value,
                1,
                dimension,
                config.variant,
                config.candidate.v_recipe,
            ),
        ) {
            (Ok(key), Ok(value)) => padding &= key.padding_zero && value.padding_zero,
            _ => pass = false,
        }
    }
    let zeros = source_words(1, 16, false, config.variant, ValuesKind::SignedZero);
    let recipe = source_words(
        1,
        32,
        false,
        config.variant,
        ValuesKind::RecipeDistinguishing,
    );
    let signed_zero_only = host_quantized(&zeros, 1, 16, config.variant, config.candidate.k_recipe)
        .is_ok_and(|quantized| {
            quantized.scales.iter().all(|scale| *scale == 127)
                && quantized.values.iter().all(|byte| *byte == 0)
                && quantized.dequantized.iter().all(|value| *value == 0.0)
        });
    let recipe_distinguishing =
        host_quantized(&recipe, 1, 32, config.variant, config.candidate.k_recipe)
            .and_then(|candidate| {
                host_quantized(&recipe, 1, 32, config.variant, Recipe::Floor)
                    .map(|floor| candidate.scales != floor.scales)
            })
            .unwrap_or(false);
    let attention = source_words(
        2,
        ATTENTION_DIM,
        false,
        config.variant,
        ValuesKind::Attention,
    );
    finite &= attention
        .iter()
        .copied()
        .map(bf16_to_f32)
        .all(|value| value.is_finite());
    pass &= padding && signed_zero_only && finite;
    HostEvidence {
        pass,
        head_dimensions: DIMENSIONS.to_vec(),
        recipes: [
            config.candidate.k_recipe.id().to_owned(),
            config.candidate.v_recipe.id().to_owned(),
        ],
        signed_zero_only,
        recipe_distinguishing,
        tail_padding_zero: padding,
        finite,
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
    let range = buffer
        .range(0, bytes.len() as u64)
        .map_err(|error| error.to_string())?;
    let mut transfer = session
        .upload(queue, range, Arc::<[u8]>::from(bytes))
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

fn f64_to_bf16(value: f64) -> u16 {
    f32_to_bf16(value as f32)
}

const QK_INVARIANCE_MAX_ABS_DELTA: f64 = 1.0e-12;
const VO_INVARIANCE_MAX_ABS_DELTA: f64 = 1.0e-12;

fn transform_key_rows(
    words: &[u16],
    token_count: usize,
    head_dim: usize,
    enabled: bool,
) -> Result<Vec<u16>, String> {
    if !enabled {
        return Ok(words.to_vec());
    }
    let rows = token_count
        .checked_mul(HEADS)
        .ok_or_else(|| "K/Q transform row count overflow".to_owned())?;
    transpose_bf16_words(words, rows, head_dim).map_err(|error| error.to_string())
}

fn fp64_qk_invariance(
    query_words: &[u16],
    transformed_query_words: &[u16],
    key_words: &[u16],
    transformed_key_words: &[u16],
    tokens: usize,
) -> Result<(bool, f64), String> {
    let query_len = Q_HEADS * ATTENTION_DIM;
    let key_len = tokens
        .checked_mul(HEADS)
        .and_then(|rows| rows.checked_mul(ATTENTION_DIM))
        .ok_or_else(|| "QK invariance key shape overflow".to_owned())?;
    if query_words.len() != query_len
        || transformed_query_words.len() != query_len
        || key_words.len() != key_len
        || transformed_key_words.len() != key_len
    {
        return Err("QK invariance shape differs".to_owned());
    }
    let mut maximum_delta = 0.0_f64;
    for query_head in 0..Q_HEADS {
        let kv_head = query_head / (Q_HEADS / HEADS);
        let query_offset = query_head * ATTENTION_DIM;
        for token in 0..tokens {
            let key_offset = (token * HEADS + kv_head) * ATTENTION_DIM;
            let mut original = 0.0_f64;
            let mut transformed = 0.0_f64;
            for dimension in 0..ATTENTION_DIM {
                original += f64::from(bf16_to_f32(query_words[query_offset + dimension]))
                    * f64::from(bf16_to_f32(key_words[key_offset + dimension]));
                transformed +=
                    f64::from(bf16_to_f32(
                        transformed_query_words[query_offset + dimension],
                    )) * f64::from(bf16_to_f32(transformed_key_words[key_offset + dimension]));
            }
            let delta = (original - transformed).abs();
            if !delta.is_finite() {
                return Err("QK invariance produced a non-finite delta".to_owned());
            }
            maximum_delta = maximum_delta.max(delta);
        }
    }
    Ok((maximum_delta <= QK_INVARIANCE_MAX_ABS_DELTA, maximum_delta))
}

#[allow(clippy::needless_range_loop)]
fn fp64_vo_invariance(
    query_words: &[u16],
    key_words: &[u16],
    value_words: &[u16],
    transformed_value_words: &[u16],
    tokens: usize,
) -> Result<(bool, f64), String> {
    let query_len = Q_HEADS * ATTENTION_DIM;
    let plane_len = tokens
        .checked_mul(HEADS)
        .and_then(|rows| rows.checked_mul(ATTENTION_DIM))
        .ok_or_else(|| "V/O invariance plane shape overflow".to_owned())?;
    if query_words.len() != query_len
        || key_words.len() != plane_len
        || value_words.len() != plane_len
        || transformed_value_words.len() != plane_len
    {
        return Err("V/O invariance shape differs".to_owned());
    }
    let mut maximum_delta = 0.0_f64;
    for query_head in 0..Q_HEADS {
        let kv_head = query_head / (Q_HEADS / HEADS);
        let query_offset = query_head * ATTENTION_DIM;
        let mut scores = Vec::with_capacity(tokens);
        for token in 0..tokens {
            let key_offset = (token * HEADS + kv_head) * ATTENTION_DIM;
            let mut dot = 0.0_f64;
            for dimension in 0..ATTENTION_DIM {
                dot += f64::from(bf16_to_f32(query_words[query_offset + dimension]))
                    * f64::from(bf16_to_f32(key_words[key_offset + dimension]));
            }
            scores.push(dot / 16.0);
        }
        let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let denominator = scores
            .iter()
            .map(|score| (*score - maximum).exp())
            .sum::<f64>();
        for dimension in 0..ATTENTION_DIM {
            let mut original = 0.0_f64;
            let mut transformed = 0.0_f64;
            for token in 0..tokens {
                let value_offset = (token * HEADS + kv_head) * ATTENTION_DIM + dimension;
                let probability = (scores[token] - maximum).exp() / denominator;
                original += probability * f64::from(bf16_to_f32(value_words[value_offset]));
                let transformed_offset =
                    (token * HEADS + kv_head) * ATTENTION_DIM + transpose16x16_index(dimension);
                transformed += probability
                    * f64::from(bf16_to_f32(transformed_value_words[transformed_offset]));
            }
            let delta = (original - transformed).abs();
            if !delta.is_finite() {
                return Err("V/O invariance produced a non-finite delta".to_owned());
            }
            maximum_delta = maximum_delta.max(delta);
        }
    }
    Ok((maximum_delta <= VO_INVARIANCE_MAX_ABS_DELTA, maximum_delta))
}

fn transpose_f32_rows(values: &[f32], rows: usize) -> Result<Vec<f32>, String> {
    let expected = rows
        .checked_mul(ATTENTION_DIM)
        .ok_or_else(|| "V/O f32 transform shape overflow".to_owned())?;
    if values.len() != expected {
        return Err("V/O f32 transform shape differs".to_owned());
    }
    let mut output = vec![0.0_f32; expected];
    for row in 0..rows {
        let base = row * ATTENTION_DIM;
        for column in 0..ATTENTION_DIM {
            output[base + column] = values[base + transpose16x16_index(column)];
        }
    }
    Ok(output)
}

#[allow(clippy::needless_range_loop)]
fn scalar_attention(
    query_words: &[u16],
    key: &[f32],
    value: &[f32],
    tokens: usize,
) -> Result<Vec<u16>, String> {
    if tokens < 2 || query_words.len() != Q_HEADS * ATTENTION_DIM {
        return Err("scalar attention shape differs".to_owned());
    }
    let mut output = vec![0_u16; query_words.len()];
    for query_head in 0..Q_HEADS {
        let kv_head = query_head / (Q_HEADS / HEADS);
        let query_offset = query_head * ATTENTION_DIM;
        let mut scores = Vec::with_capacity(tokens);
        for token in 0..tokens {
            let key_offset = (token * HEADS + kv_head) * ATTENTION_DIM;
            let mut dot = 0.0_f64;
            for dimension in 0..ATTENTION_DIM {
                dot += f64::from(bf16_to_f32(query_words[query_offset + dimension]))
                    * f64::from(key[key_offset + dimension]);
            }
            scores.push(dot / 16.0);
        }
        let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let denominator = scores
            .iter()
            .map(|score| (*score - maximum).exp())
            .sum::<f64>();
        for dimension in 0..ATTENTION_DIM {
            let mut accumulation = 0.0_f64;
            for token in 0..tokens {
                let value_offset = (token * HEADS + kv_head) * ATTENTION_DIM + dimension;
                let probability = (scores[token] - maximum).exp() / denominator;
                accumulation += probability * f64::from(value[value_offset]);
            }
            output[query_offset + dimension] = f64_to_bf16(accumulation);
        }
    }
    Ok(output)
}

fn attention_query() -> Vec<u16> {
    (0..Q_HEADS * ATTENTION_DIM)
        .map(|index| {
            let head = index / ATTENTION_DIM;
            let column = index % ATTENTION_DIM;
            f32_to_bf16(((column * 7 + head * 5) % 37) as f32 / 16.0 - 1.0)
        })
        .collect()
}

struct AttentionEvidence {
    direct: bool,
    numerical_match: bool,
    key_contributes: bool,
    qk_invariant: bool,
    qk_max_abs_delta: f64,
    query_transformed: bool,
    vo_invariant: bool,
    vo_max_abs_delta: f64,
    output_inverse_transformed: bool,
}

#[allow(clippy::too_many_arguments)]
fn run_attention(
    session: &ExecutionSession,
    queue: &sllm_core::ExecutionQueue,
    state: &sllm_core::KvState,
    kq_transform_enabled: bool,
    vo_transform_enabled: bool,
    query_words: &[u16],
    transformed_key_words: &[u16],
    original_key_words: &[u16],
    transformed_value_words: &[u16],
    original_value_words: &[u16],
    key_values: &[f32],
    value_values: &[f32],
) -> Result<AttentionEvidence, String> {
    let transform_query = kq_transform_enabled;
    let transformed_query_words = if transform_query {
        transpose_bf16_words(query_words, Q_HEADS, ATTENTION_DIM)
            .map_err(|error| error.to_string())?
    } else {
        query_words.to_vec()
    };
    let (qk_invariant, qk_max_abs_delta) = if transform_query {
        fp64_qk_invariance(
            query_words,
            &transformed_query_words,
            original_key_words,
            transformed_key_words,
            2,
        )?
    } else {
        (true, 0.0)
    };
    let (vo_invariant, vo_max_abs_delta) = if vo_transform_enabled {
        fp64_vo_invariance(
            query_words,
            original_key_words,
            original_value_words,
            transformed_value_words,
            2,
        )?
    } else {
        (true, 0.0)
    };
    let bytes = words_to_bytes(&transformed_query_words);
    let query_buffer = session
        .allocate(bytes.len() as u64)
        .map_err(|error| error.to_string())?;
    let output_buffer = session
        .allocate(bytes.len() as u64)
        .map_err(|error| error.to_string())?;
    upload(session, queue, &query_buffer, &transformed_query_words)?;
    let shape = [1, Q_HEADS, ATTENTION_DIM];
    let query = binding(session, &query_buffer, &shape, AccessMode::Read)?;
    let output = binding(session, &output_buffer, &shape, AccessMode::Write)?;
    let descriptor =
        sllm_core::CausalAttentionDescriptor::new(1, 1, 2).map_err(|error| error.to_string())?;
    let mut attention = session
        .causal_attention(state, queue, query, output, descriptor)
        .map_err(|error| error.to_string())?;
    let dispatch = attention.dispatch().clone();
    if attention.wait(WAIT).map_err(|error| error.to_string())? != ExecutionState::Success {
        return Err("attention failed".to_owned());
    }
    drop(attention);
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
    let mut actual_bytes = vec![0_u8; bytes.len()];
    readback
        .read_into(&mut actual_bytes)
        .map_err(|error| error.to_string())?;
    let actual = actual_bytes
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect::<Vec<_>>();
    let expected_value = if vo_transform_enabled {
        transpose_f32_rows(value_values, 2 * HEADS)?
    } else {
        value_values.to_vec()
    };
    let expected = scalar_attention(&transformed_query_words, key_values, &expected_value, 2)?;
    let zero_key = vec![0.0_f32; key_values.len()];
    let key_contributes =
        scalar_attention(&transformed_query_words, &zero_key, &expected_value, 2)? != expected;
    let (actual_for_compare, expected_for_compare, output_inverse_transformed) =
        if vo_transform_enabled {
            let actual = transpose_bf16_words(&actual, Q_HEADS, ATTENTION_DIM)
                .map_err(|error| error.to_string())?;
            (actual, expected, true)
        } else {
            (actual, expected, false)
        };
    let numerical_match = actual_for_compare == expected_for_compare;
    let direct = dispatch.backend != 0
        && !dispatch.fallback_allowed
        && !dispatch.fallback_used
        && dispatch.dispatch_count > 0
        && dispatch.kernel_symbol.contains("packed");
    Ok(AttentionEvidence {
        direct,
        numerical_match,
        key_contributes,
        qk_invariant,
        qk_max_abs_delta,
        query_transformed: transform_query,
        vo_invariant,
        vo_max_abs_delta,
        output_inverse_transformed,
    })
}

#[cfg(not(test))]
unsafe extern "C" {
    fn sllm_phase54_kv_research_set_recipe_pair_v1(key: u32, value: u32) -> i32;
    fn sllm_phase54_kv_research_get_recipe_pair_v1(key: *mut u32, value: *mut u32) -> i32;
}

#[cfg(not(test))]
fn set_and_verify_recipe_pair(key: Recipe, value: Recipe) -> Result<(), String> {
    // SAFETY: the ABI accepts only the closed recipe enum and writes to the
    // two valid stack pointers supplied to the getter.
    let status = unsafe {
        sllm_phase54_kv_research_set_recipe_pair_v1(key.runtime_value(), value.runtime_value())
    };
    if status != 0 {
        return Err(format!(
            "set Phase 54 recipe pair failed with status {status}"
        ));
    }
    let mut observed_key = u32::MAX;
    let mut observed_value = u32::MAX;
    // SAFETY: both pointers remain valid for the duration of this call.
    let status = unsafe {
        sllm_phase54_kv_research_get_recipe_pair_v1(&mut observed_key, &mut observed_value)
    };
    if status != 0 {
        return Err(format!(
            "get Phase 54 recipe pair failed with status {status}"
        ));
    }
    if (observed_key, observed_value) != (key.runtime_value(), value.runtime_value()) {
        return Err("Phase 54 recipe getter did not preserve the selected pair".to_owned());
    }
    Ok(())
}

#[cfg(test)]
fn set_and_verify_recipe_pair(_key: Recipe, _value: Recipe) -> Result<(), String> {
    Ok(())
}

struct RecipeResetGuard;

impl RecipeResetGuard {
    fn install() -> Result<Self, String> {
        set_and_verify_recipe_pair(Recipe::Floor, Recipe::Floor)?;
        Ok(Self)
    }
}

impl Drop for RecipeResetGuard {
    fn drop(&mut self) {
        let _ = set_and_verify_recipe_pair(Recipe::Floor, Recipe::Floor);
    }
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    session: &ExecutionSession,
    queue: &sllm_core::ExecutionQueue,
    config: &Config,
    index: usize,
    head_dim: usize,
    token_count: usize,
    kind: ValuesKind,
    attention_case: bool,
) -> Result<(CaseEvidence, u64, u64, Option<TransformEvidence>), String> {
    let original_key_words = source_words(token_count, head_dim, false, config.variant, kind);
    let original_value_words = source_words(token_count, head_dim, true, config.variant, kind);
    let kq_transform_applied = config.candidate.transform_candidate() && attention_case;
    let vo_transform_applied = (config.candidate.vo_transform_candidate()
        || config.candidate.vo_layers19_31_transform_candidate())
        && attention_case;
    let transform_applied = kq_transform_applied || vo_transform_applied;
    // The direct oracle has no layer graph. The attention case is the reviewed
    // layer semantic carrier for the V/O candidate; all boundary, signed-zero,
    // and recipe cases intentionally remain untransformed.
    let key_words = transform_key_rows(
        &original_key_words,
        token_count,
        head_dim,
        kq_transform_applied,
    )?;
    let value_words = transform_key_rows(
        &original_value_words,
        token_count,
        head_dim,
        vo_transform_applied,
    )?;
    let key_oracle = host_quantized(
        &key_words,
        token_count,
        head_dim,
        config.variant,
        config.candidate.k_recipe,
    )?;
    let value_oracle = host_quantized(
        &value_words,
        token_count,
        head_dim,
        config.variant,
        config.candidate.v_recipe,
    )?;
    let descriptor = KvStateDescriptor::new_with_kv_fp8_block16(
        index as u32,
        token_count as u64,
        HEADS,
        head_dim,
        config.encoding,
        config.variant,
    )
    .map_err(|error| error.to_string())?;
    let state = session
        .create_kv_state(descriptor)
        .map_err(|error| error.to_string())?;
    let key_buffer = session
        .allocate((key_words.len() * 2) as u64)
        .map_err(|error| error.to_string())?;
    let value_buffer = session
        .allocate((value_words.len() * 2) as u64)
        .map_err(|error| error.to_string())?;
    upload(session, queue, &key_buffer, &key_words)?;
    upload(session, queue, &value_buffer, &value_words)?;
    let shape = [token_count, HEADS, head_dim];
    let key = binding(session, &key_buffer, &shape, AccessMode::Read)?;
    let value = binding(session, &value_buffer, &shape, AccessMode::Read)?;
    let mut append = session
        .append_kv_state(&state, queue, key, value, 0, 0)
        .map_err(|error| error.to_string())?;
    let append_dispatch = append.dispatch().clone();
    if append.wait(WAIT).map_err(|error| error.to_string())? != ExecutionState::Success {
        return Err(format!("{} append failed", head_dim));
    }
    drop(append);
    let image = session
        .export_kv_state_image(&state)
        .map_err(|error| format!("{} export failed: {error}", head_dim))?;
    let key_values_exact = plane(&image, StatePlaneKindV1::KvKey)? == key_oracle.values;
    let value_values_exact = plane(&image, StatePlaneKindV1::KvValue)? == value_oracle.values;
    let key_scales_exact = plane(&image, StatePlaneKindV1::KvKeyScale)? == key_oracle.scales;
    let value_scales_exact = plane(&image, StatePlaneKindV1::KvValueScale)? == value_oracle.scales;
    let append_direct = append_dispatch.backend != 0
        && !append_dispatch.fallback_allowed
        && !append_dispatch.fallback_used
        && append_dispatch.dispatch_count > 0
        && append_dispatch.kernel_symbol.contains("block16");
    let (attention, transform_evidence) = if attention_case {
        let attention = run_attention(
            session,
            queue,
            &state,
            kq_transform_applied,
            vo_transform_applied,
            &attention_query(),
            &key_words,
            &original_key_words,
            &value_words,
            &original_value_words,
            &key_oracle.dequantized,
            &value_oracle.dequantized,
        )?;
        let transform_evidence = if kq_transform_applied {
            Some(TransformEvidence {
                identity: "transpose16x16-all-full",
                semantics: PHASE54_KQ_TRANSFORM_SEMANTICS,
                digest: PHASE54_KQ_TRANSFORM_DIGEST,
                key_rows_transformed: true,
                query_rows_transformed: attention.query_transformed,
                value_untransformed: true,
                qk_invariant: attention.qk_invariant,
                qk_max_abs_delta: attention.qk_max_abs_delta,
                value_rows_transformed: None,
                output_rows_inverse_transformed: None,
                vo_invariant: None,
                vo_max_abs_delta: None,
                semantic_layer: None,
                semantic_layers: None,
            })
        } else if vo_transform_applied {
            let layers19_31 = config.candidate.vo_layers19_31_transform_candidate();
            Some(TransformEvidence {
                identity: if layers19_31 {
                    PHASE54_VO_TRANSFORM_LAYERS_19_31_SELECTOR
                } else {
                    PHASE54_VO_TRANSFORM_SELECTOR
                },
                semantics: if layers19_31 {
                    PHASE54_VO_TRANSFORM_LAYERS_19_31_SEMANTICS
                } else {
                    PHASE54_VO_TRANSFORM_SEMANTICS
                },
                digest: if layers19_31 {
                    PHASE54_VO_TRANSFORM_LAYERS_19_31_DIGEST
                } else {
                    PHASE54_VO_TRANSFORM_DIGEST
                },
                key_rows_transformed: false,
                query_rows_transformed: false,
                value_untransformed: false,
                qk_invariant: attention.qk_invariant,
                qk_max_abs_delta: attention.qk_max_abs_delta,
                value_rows_transformed: Some(true),
                output_rows_inverse_transformed: Some(attention.output_inverse_transformed),
                vo_invariant: Some(attention.vo_invariant),
                vo_max_abs_delta: Some(attention.vo_max_abs_delta),
                semantic_layer: (!layers19_31).then_some(19),
                semantic_layers: layers19_31.then_some(vec![19, 31]),
            })
        } else {
            None
        };
        (Some(attention), transform_evidence)
    } else {
        (None, None)
    };
    let floor_scales = host_quantized(
        &key_words,
        token_count,
        head_dim,
        config.variant,
        Recipe::Floor,
    )?
    .scales;
    let recipe_distinguishing = key_oracle.scales != floor_scales;
    let id = if attention_case {
        "attention-kv2".to_owned()
    } else {
        format!("append-head-dim-{head_dim}")
    };
    let evidence = CaseEvidence {
        id,
        head_dim,
        token_count,
        transform_applied,
        signed_zero_only: matches!(kind, ValuesKind::SignedZero),
        recipe_distinguishing: matches!(kind, ValuesKind::RecipeDistinguishing)
            || recipe_distinguishing,
        key_values_exact,
        value_values_exact,
        key_scales_exact,
        value_scales_exact,
        tail_padding_zero: key_oracle.padding_zero && value_oracle.padding_zero,
        append_direct,
        attention_direct: attention.as_ref().is_some_and(|value| value.direct),
        attention_numerical_match: attention.as_ref().is_none_or(|value| value.numerical_match),
        attention_key_contributes: attention
            .as_ref()
            .is_some_and(|value| value.key_contributes),
        finite: key_words
            .iter()
            .chain(value_words.iter())
            .copied()
            .map(bf16_to_f32)
            .all(|value| value.is_finite())
            || !attention_case,
    };
    Ok((
        evidence,
        u64::from(append_dispatch.dispatch_count),
        u64::from(attention_case),
        transform_evidence,
    ))
}

fn base_report(config: &Config, host: HostEvidence, binary_sha256: String) -> Report {
    Report {
        schema: "https://sllm.dev/schema/phase54-kv-fp8-block16-research-evidence-v1.schema.json",
        schema_version: "sllm-phase54-kv-fp8-block16-research-evidence-v1",
        state: "UNAVAILABLE",
        target: config.target.clone(),
        device_index: config.device_index,
        encoding: encoding_name(config.encoding),
        physical_variant: variant_name(config.variant),
        production_descriptor_id: config.candidate.production_control().then_some(
            match config.encoding {
                KvCacheEncoding::Fp8E4M3Block16 => "kv-fp8-e4-block16-v2",
                KvCacheEncoding::Fp8E5M2Block16 => "kv-fp8-e5-block16-v2",
                _ => unreachable!(),
            },
        ),
        descriptor_compatibility: config.candidate.descriptor_compatibility.clone(),
        candidate_spec: config.candidate.clone(),
        candidate_spec_sha256: config.candidate_spec_sha256.clone(),
        binary_sha256,
        host,
        cases: Vec::new(),
        transform_evidence: if config.candidate.transform_candidate() {
            Some(TransformEvidence {
                identity: "transpose16x16-all-full",
                semantics: PHASE54_KQ_TRANSFORM_SEMANTICS,
                digest: PHASE54_KQ_TRANSFORM_DIGEST,
                key_rows_transformed: false,
                query_rows_transformed: false,
                value_untransformed: false,
                qk_invariant: false,
                qk_max_abs_delta: 0.0,
                value_rows_transformed: None,
                output_rows_inverse_transformed: None,
                vo_invariant: None,
                vo_max_abs_delta: None,
                semantic_layer: None,
                semantic_layers: None,
            })
        } else if config.candidate.vo_transform_candidate() {
            Some(TransformEvidence {
                identity: PHASE54_VO_TRANSFORM_SELECTOR,
                semantics: PHASE54_VO_TRANSFORM_SEMANTICS,
                digest: PHASE54_VO_TRANSFORM_DIGEST,
                key_rows_transformed: false,
                query_rows_transformed: false,
                value_untransformed: false,
                qk_invariant: true,
                qk_max_abs_delta: 0.0,
                value_rows_transformed: Some(false),
                output_rows_inverse_transformed: Some(false),
                vo_invariant: Some(false),
                vo_max_abs_delta: Some(0.0),
                semantic_layer: Some(19),
                semantic_layers: None,
            })
        } else if config.candidate.vo_layers19_31_transform_candidate() {
            Some(TransformEvidence {
                identity: PHASE54_VO_TRANSFORM_LAYERS_19_31_SELECTOR,
                semantics: PHASE54_VO_TRANSFORM_LAYERS_19_31_SEMANTICS,
                digest: PHASE54_VO_TRANSFORM_LAYERS_19_31_DIGEST,
                key_rows_transformed: false,
                query_rows_transformed: false,
                value_untransformed: false,
                qk_invariant: true,
                qk_max_abs_delta: 0.0,
                value_rows_transformed: Some(false),
                output_rows_inverse_transformed: Some(false),
                vo_invariant: Some(false),
                vo_max_abs_delta: Some(0.0),
                semantic_layer: None,
                semantic_layers: Some(vec![19, 31]),
            })
        } else {
            None
        },
        execution: ExecutionEvidence {
            selected_backend: "hip",
            gpu_execution: false,
            fallback_allowed: false,
            fallback_used: false,
            append_dispatches: 0,
            attention_dispatches: 0,
            sequential_residents: true,
        },
        cleanup: CleanupEvidence {
            retryable: 0,
            durable: 0,
            terminal_zero: false,
        },
        error: None,
    }
}

fn run(config: &Config) -> Report {
    let host = host_evidence(config);
    let binary_sha256 = env::current_exe()
        .ok()
        .and_then(|path| fs::read(path).ok())
        .map(|bytes| digest(&bytes))
        .unwrap_or_else(|| digest(b"unavailable-binary"));
    let backend = match HipBackend::connect() {
        Ok(backend) => backend,
        Err(error) => {
            let mut report = base_report(config, host, binary_sha256);
            report.error = Some(error.to_string());
            return report;
        }
    };
    let request = match ExecutionSessionRequest::new(config.device_index, config.target.clone()) {
        Ok(request) => request,
        Err(error) => {
            let mut report = base_report(config, host, binary_sha256);
            report.state = "FAIL";
            report.error = Some(error.to_string());
            return report;
        }
    };
    let session = match backend.open_execution_session(request) {
        Ok(session) => session,
        Err(error) => {
            let mut report = base_report(config, host, binary_sha256);
            report.error = Some(error.to_string());
            return report;
        }
    };
    let _recipe_guard = match RecipeResetGuard::install() {
        Ok(guard) => guard,
        Err(error) => {
            let mut report = base_report(config, host, binary_sha256);
            report.state = "FAIL";
            report.error = Some(error);
            let _ = session.shutdown(SHUTDOWN);
            return report;
        }
    };
    if let Err(error) =
        set_and_verify_recipe_pair(config.candidate.k_recipe, config.candidate.v_recipe)
    {
        let mut report = base_report(config, host, binary_sha256);
        report.state = "FAIL";
        report.error = Some(error);
        let _ = session.shutdown(SHUTDOWN);
        return report;
    }
    let queue = match session.create_queue() {
        Ok(queue) => queue,
        Err(error) => {
            let mut report = base_report(config, host, binary_sha256);
            report.state = "FAIL";
            report.error = Some(error.to_string());
            let _ = session.shutdown(SHUTDOWN);
            return report;
        }
    };
    let operation = (|| {
        let mut cases = Vec::new();
        let mut append_dispatches = 0_u64;
        let mut attention_dispatches = 0_u64;
        let mut transform_evidence = None;
        for (index, &dimension) in DIMENSIONS.iter().enumerate() {
            let attention = dimension == ATTENTION_DIM;
            let (case, append, attention_count, case_transform) = run_case(
                &session,
                &queue,
                config,
                index,
                dimension,
                if attention { 2 } else { 1 },
                if attention {
                    ValuesKind::Attention
                } else {
                    ValuesKind::Boundary
                },
                attention,
            )?;
            cases.push(case);
            append_dispatches += append;
            attention_dispatches += attention_count;
            if case_transform.is_some() {
                transform_evidence = case_transform;
            }
        }
        for (offset, (kind, id)) in [
            (ValuesKind::SignedZero, "signed-zero-only"),
            (ValuesKind::RecipeDistinguishing, "recipe-distinguishing"),
        ]
        .into_iter()
        .enumerate()
        {
            let head_dim = if matches!(kind, ValuesKind::RecipeDistinguishing) {
                32
            } else {
                16
            };
            let (mut case, append, attention_count, case_transform) = run_case(
                &session,
                &queue,
                config,
                DIMENSIONS.len() + offset,
                head_dim,
                1,
                kind,
                false,
            )?;
            case.id = id.to_owned();
            cases.push(case);
            append_dispatches += append;
            attention_dispatches += attention_count;
            if case_transform.is_some() {
                transform_evidence = case_transform;
            }
        }
        Ok::<_, String>((
            cases,
            append_dispatches,
            attention_dispatches,
            transform_evidence,
        ))
    })();
    let cleanup = session.shutdown(SHUTDOWN);
    match (operation, cleanup) {
        (Ok((cases, append_dispatches, attention_dispatches, transform_evidence)), Ok(cleanup)) => {
            let transform_pass = if config.candidate.transform_candidate() {
                transform_evidence.as_ref().is_some_and(|value| {
                    value.key_rows_transformed
                        && value.query_rows_transformed
                        && value.value_untransformed
                        && value.qk_invariant
                        && value.qk_max_abs_delta <= QK_INVARIANCE_MAX_ABS_DELTA
                })
            } else if config.candidate.vo_transform_candidate()
                || config.candidate.vo_layers19_31_transform_candidate()
            {
                transform_evidence.as_ref().is_some_and(|value| {
                    value.value_rows_transformed == Some(true)
                        && value.output_rows_inverse_transformed == Some(true)
                        && value.vo_invariant == Some(true)
                        && value
                            .vo_max_abs_delta
                            .is_some_and(|delta| delta <= VO_INVARIANCE_MAX_ABS_DELTA)
                })
            } else {
                true
            };
            let pass = host.pass
                && cases.iter().all(case_passes)
                && cases.iter().any(|case| case.signed_zero_only)
                && cases.iter().any(|case| case.recipe_distinguishing)
                && transform_pass
                && cleanup.retryable_cleanup == 0
                && cleanup.durable_quarantine == 0;
            let mut report = base_report(config, host, binary_sha256);
            report.state = if pass { "PASS" } else { "FAIL" };
            report.cases = cases;
            report.transform_evidence = transform_evidence;
            report.execution = ExecutionEvidence {
                selected_backend: "hip",
                gpu_execution: true,
                fallback_allowed: false,
                fallback_used: false,
                append_dispatches,
                attention_dispatches,
                sequential_residents: true,
            };
            report.cleanup = CleanupEvidence {
                retryable: cleanup.retryable_cleanup,
                durable: cleanup.durable_quarantine,
                terminal_zero: cleanup.retryable_cleanup == 0 && cleanup.durable_quarantine == 0,
            };
            report
        }
        (Err(error), Ok(cleanup)) => {
            let mut report = base_report(config, host, binary_sha256);
            report.state = "FAIL";
            report.cleanup = CleanupEvidence {
                retryable: cleanup.retryable_cleanup,
                durable: cleanup.durable_quarantine,
                terminal_zero: cleanup.retryable_cleanup == 0 && cleanup.durable_quarantine == 0,
            };
            report.error = Some(error);
            report
        }
        (operation, cleanup) => {
            let mut report = base_report(config, host, binary_sha256);
            report.state = "FAIL";
            report.error = Some(format!("operation={operation:?}; cleanup={cleanup:?}"));
            report
        }
    }
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if path.exists() {
        return Err("output already exists".to_owned());
    }
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(|error| format!("create output parent: {error}"))?;
    let name = path
        .file_name()
        .ok_or("output must name a file")?
        .to_string_lossy();
    let partial = parent.join(format!(".{name}.partial.{}", std::process::id()));
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&partial)
            .map_err(|error| error.to_string())?;
        file.write_all(bytes).map_err(|error| error.to_string())?;
        file.sync_all().map_err(|error| error.to_string())?;
        fs::rename(&partial, path).map_err(|error| error.to_string())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&partial);
    }
    result
}

fn main() -> ExitCode {
    let config = match parse() {
        Ok(config) => config,
        Err(error) => {
            eprintln!("sllm-kv-fp8-block16-research-evidence: {error}");
            return ExitCode::FAILURE;
        }
    };
    let report = run(&config);
    let passed = report.state == "PASS";
    let bytes = match serde_json::to_vec(&report) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("serialize report: {error}");
            return ExitCode::from(2);
        }
    };
    let output_result = if let Some(path) = &config.output {
        write_atomic(path, &bytes)
    } else {
        println!("{}", String::from_utf8_lossy(&bytes));
        Ok(())
    };
    match output_result {
        Ok(()) if passed => ExitCode::SUCCESS,
        Ok(()) => ExitCode::FAILURE,
        Err(error) => {
            eprintln!("write report: {error}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn passing_attention_case() -> CaseEvidence {
        CaseEvidence {
            id: "attention-kv2".to_owned(),
            head_dim: ATTENTION_DIM,
            token_count: 2,
            transform_applied: false,
            signed_zero_only: false,
            recipe_distinguishing: false,
            key_values_exact: true,
            value_values_exact: true,
            key_scales_exact: true,
            value_scales_exact: true,
            tail_padding_zero: true,
            append_direct: true,
            attention_direct: true,
            attention_numerical_match: true,
            attention_key_contributes: true,
            finite: true,
        }
    }

    fn candidate(k_recipe: Recipe, v_recipe: Recipe) -> CandidateSpec {
        let production = k_recipe == Recipe::Floor && v_recipe == Recipe::Floor;
        CandidateSpec {
            schema_version: "sllm-phase54-kv-candidate-spec-v1".to_owned(),
            candidate_id: if production {
                "production-control-v2".to_owned()
            } else {
                format!("phase54-k-{}-v-{}-v1", k_recipe.id(), v_recipe.id())
            },
            scale_selector: "independent-k-v-closed-enum-v1".to_owned(),
            rounding: "nearest-even".to_owned(),
            k_recipe,
            v_recipe,
            transform: "none".to_owned(),
            calibration_digest: None,
            descriptor_compatibility: if production {
                "exact-production-v2".to_owned()
            } else {
                "research-build-semantic-override-not-v2-compatible".to_owned()
            },
        }
    }

    fn transform_candidate() -> CandidateSpec {
        CandidateSpec {
            schema_version: "sllm-phase54-kv-candidate-spec-v1".to_owned(),
            candidate_id: "phase54-kq-transpose16x16-all-full-v1".to_owned(),
            scale_selector: "independent-k-v-closed-enum-v1".to_owned(),
            rounding: "nearest-even".to_owned(),
            k_recipe: Recipe::Floor,
            v_recipe: Recipe::Floor,
            transform: "transpose16x16-all-full".to_owned(),
            calibration_digest: None,
            descriptor_compatibility: "research-build-semantic-override-not-v2-compatible"
                .to_owned(),
        }
    }

    fn vo_transform_candidate() -> CandidateSpec {
        CandidateSpec {
            schema_version: "sllm-phase54-kv-candidate-spec-v1".to_owned(),
            candidate_id: "phase54-vo-transpose16x16-layer19-v1".to_owned(),
            scale_selector: "independent-k-v-closed-enum-v1".to_owned(),
            rounding: "nearest-even".to_owned(),
            k_recipe: Recipe::Floor,
            v_recipe: Recipe::Floor,
            transform: PHASE54_VO_TRANSFORM_SELECTOR.to_owned(),
            calibration_digest: None,
            descriptor_compatibility: "research-build-semantic-override-not-v2-compatible"
                .to_owned(),
        }
    }

    fn vo_layers19_31_transform_candidate() -> CandidateSpec {
        CandidateSpec {
            schema_version: "sllm-phase54-kv-candidate-spec-v1".to_owned(),
            candidate_id: "phase54-vo-transpose16x16-layers19-31-v1".to_owned(),
            scale_selector: "independent-k-v-closed-enum-v1".to_owned(),
            rounding: "nearest-even".to_owned(),
            k_recipe: Recipe::Floor,
            v_recipe: Recipe::Floor,
            transform: PHASE54_VO_TRANSFORM_LAYERS_19_31_SELECTOR.to_owned(),
            calibration_digest: None,
            descriptor_compatibility: "research-build-semantic-override-not-v2-compatible"
                .to_owned(),
        }
    }

    #[test]
    fn candidate_identity_is_closed_and_recipe_specific() {
        assert!(candidate(Recipe::Floor, Recipe::Floor).validate().is_ok());
        assert!(
            candidate(Recipe::Ceil, Recipe::NearestEven)
                .validate()
                .is_ok()
        );
        assert!(
            candidate(Recipe::Parent32Duplicate, Recipe::Parent32Duplicate,)
                .validate()
                .is_ok()
        );
        let transform = transform_candidate();
        assert!(transform.validate().is_ok());
        assert!(transform.transform_candidate());
        assert!(!transform.production_control());
        let vo_transform = vo_transform_candidate();
        assert!(vo_transform.validate().is_ok());
        assert!(vo_transform.vo_transform_candidate());
        assert!(!vo_transform.production_control());
        let vo_layers19_31 = vo_layers19_31_transform_candidate();
        assert!(vo_layers19_31.validate().is_ok());
        assert!(vo_layers19_31.vo_layers19_31_transform_candidate());
        assert!(!vo_layers19_31.vo_transform_candidate());
        assert_eq!(
            vo_layers19_31.expected_id(),
            "phase54-vo-transpose16x16-layers19-31-v1"
        );
        let mut invalid = candidate(Recipe::Floor, Recipe::Floor);
        invalid.candidate_id = "phase54-k-floor-v-floor-v1".to_owned();
        assert!(invalid.validate().is_err());
        let mut invalid_transform = transform_candidate();
        invalid_transform.k_recipe = Recipe::Ceil;
        assert!(invalid_transform.validate().is_err());
        let mut invalid_transform_id = transform_candidate();
        invalid_transform_id.candidate_id = "production-control-v2".to_owned();
        assert!(invalid_transform_id.validate().is_err());
        let mut invalid_vo_transform = vo_transform_candidate();
        invalid_vo_transform.v_recipe = Recipe::Ceil;
        assert!(invalid_vo_transform.validate().is_err());
        let mut invalid_vo_layers19_31 = vo_layers19_31_transform_candidate();
        invalid_vo_layers19_31.k_recipe = Recipe::Ceil;
        assert!(invalid_vo_layers19_31.validate().is_err());
    }

    #[test]
    fn attention_case_is_fail_closed_on_direct_and_oracle_evidence() {
        let passing = passing_attention_case();
        assert!(case_passes(&passing));

        let mut fallback = passing_attention_case();
        fallback.attention_direct = false;
        assert!(!case_passes(&fallback));

        let mut mismatch = passing_attention_case();
        mismatch.attention_numerical_match = false;
        assert!(!case_passes(&mismatch));

        let mut insensitive = passing_attention_case();
        insensitive.attention_key_contributes = false;
        assert!(!case_passes(&insensitive));
    }

    #[test]
    fn host_research_oracle_covers_tails_signed_zero_and_recipe_delta() {
        let config = Config {
            device_index: 0,
            target: "gfx1030".to_owned(),
            encoding: KvCacheEncoding::Fp8E5M2Block16,
            variant: KvFp8PhysicalVariant::OcpE5M2,
            candidate: candidate(Recipe::Ceil, Recipe::NearestEven),
            candidate_spec_sha256: digest(b"candidate"),
            output: None,
        };
        let evidence = host_evidence(&config);
        assert!(evidence.pass, "{evidence:?}");
        assert!(evidence.signed_zero_only);
        assert!(evidence.recipe_distinguishing);
        assert!(evidence.tail_padding_zero);
    }

    #[test]
    fn scalar_attention_is_nonzero_and_key_sensitive() {
        let query = attention_query();
        let key_words = source_words(
            2,
            ATTENTION_DIM,
            false,
            KvFp8PhysicalVariant::OcpE5M2,
            ValuesKind::Attention,
        );
        let value_words = source_words(
            2,
            ATTENTION_DIM,
            true,
            KvFp8PhysicalVariant::OcpE5M2,
            ValuesKind::Attention,
        );
        let key = host_quantized(
            &key_words,
            2,
            ATTENTION_DIM,
            KvFp8PhysicalVariant::OcpE5M2,
            Recipe::Ceil,
        )
        .unwrap();
        let value = host_quantized(
            &value_words,
            2,
            ATTENTION_DIM,
            KvFp8PhysicalVariant::OcpE5M2,
            Recipe::NearestEven,
        )
        .unwrap();
        let expected = scalar_attention(&query, &key.dequantized, &value.dequantized, 2).unwrap();
        assert!(expected.iter().any(|word| *word != 0));
        let zero_key = vec![0.0; key.dequantized.len()];
        assert_ne!(
            expected,
            scalar_attention(&query, &zero_key, &value.dequantized, 2).unwrap()
        );
    }

    #[test]
    fn transform_preserves_fp64_unquantized_qk_with_nonzero_query() {
        let query = attention_query();
        assert!(query.iter().any(|word| *word != 0));
        let key = source_words(
            2,
            ATTENTION_DIM,
            false,
            KvFp8PhysicalVariant::OcpE5M2,
            ValuesKind::Attention,
        );
        let transformed_key = transform_key_rows(&key, 2, ATTENTION_DIM, true).unwrap();
        let transformed_query = transpose_bf16_words(&query, Q_HEADS, ATTENTION_DIM).unwrap();
        let (invariant, maximum_delta) =
            fp64_qk_invariance(&query, &transformed_query, &key, &transformed_key, 2).unwrap();
        assert!(invariant, "maximum QK delta was {maximum_delta}");
        assert!(maximum_delta <= QK_INVARIANCE_MAX_ABS_DELTA);
        assert_ne!(key, transformed_key);
    }

    #[test]
    fn transform_key_oracle_is_floor_and_changes_block16_grouping() {
        let key = source_words(
            2,
            ATTENTION_DIM,
            false,
            KvFp8PhysicalVariant::OcpE5M2,
            ValuesKind::Attention,
        );
        let transformed = transform_key_rows(&key, 2, ATTENTION_DIM, true).unwrap();
        let original_oracle = host_quantized(
            &key,
            2,
            ATTENTION_DIM,
            KvFp8PhysicalVariant::OcpE5M2,
            Recipe::Floor,
        )
        .unwrap();
        let transformed_oracle = host_quantized(
            &transformed,
            2,
            ATTENTION_DIM,
            KvFp8PhysicalVariant::OcpE5M2,
            Recipe::Floor,
        )
        .unwrap();
        assert!(
            transformed_oracle.values != original_oracle.values
                || transformed_oracle.scales != original_oracle.scales
        );
    }

    #[test]
    fn vo_transform_preserves_unquantized_attention_v_output() {
        let query = attention_query();
        let key = source_words(
            2,
            ATTENTION_DIM,
            false,
            KvFp8PhysicalVariant::OcpE5M2,
            ValuesKind::Attention,
        );
        let value = source_words(
            2,
            ATTENTION_DIM,
            true,
            KvFp8PhysicalVariant::OcpE5M2,
            ValuesKind::Attention,
        );
        let transformed_value = transform_key_rows(&value, 2, ATTENTION_DIM, true).unwrap();
        let (invariant, maximum_delta) =
            fp64_vo_invariance(&query, &key, &value, &transformed_value, 2).unwrap();
        assert!(invariant, "maximum V/O delta was {maximum_delta}");
        assert!(maximum_delta <= VO_INVARIANCE_MAX_ABS_DELTA);
        assert_ne!(value, transformed_value);
        let values = value.iter().copied().map(bf16_to_f32).collect::<Vec<_>>();
        let transformed_values = transformed_value
            .iter()
            .copied()
            .map(bf16_to_f32)
            .collect::<Vec<_>>();
        assert_eq!(
            transpose_f32_rows(&transformed_values, 2 * HEADS).unwrap(),
            values
        );
        let key_quantized = host_quantized(
            &key,
            2,
            ATTENTION_DIM,
            KvFp8PhysicalVariant::OcpE5M2,
            Recipe::Floor,
        )
        .unwrap();
        let value_quantized = host_quantized(
            &transformed_value,
            2,
            ATTENTION_DIM,
            KvFp8PhysicalVariant::OcpE5M2,
            Recipe::Floor,
        )
        .unwrap();
        let restored_value = transpose_f32_rows(&value_quantized.dequantized, 2 * HEADS).unwrap();
        let expected =
            scalar_attention(&query, &key_quantized.dequantized, &restored_value, 2).unwrap();
        let zero_key = vec![0.0; key_quantized.dequantized.len()];
        assert_ne!(
            expected,
            scalar_attention(&query, &zero_key, &restored_value, 2).unwrap()
        );
    }

    #[test]
    fn vo_layers19_31_transform_reuses_vo_invariance_oracle() {
        let candidate = vo_layers19_31_transform_candidate();
        assert!(candidate.validate().is_ok());
        let query = attention_query();
        let key = source_words(
            2,
            ATTENTION_DIM,
            false,
            KvFp8PhysicalVariant::OcpE5M2,
            ValuesKind::Attention,
        );
        let value = source_words(
            2,
            ATTENTION_DIM,
            true,
            KvFp8PhysicalVariant::OcpE5M2,
            ValuesKind::Attention,
        );
        let transformed_value = transform_key_rows(&value, 2, ATTENTION_DIM, true).unwrap();
        let (invariant, maximum_delta) =
            fp64_vo_invariance(&query, &key, &value, &transformed_value, 2).unwrap();
        assert!(invariant, "maximum V/O delta was {maximum_delta}");
        assert!(maximum_delta <= VO_INVARIANCE_MAX_ABS_DELTA);
        assert_eq!(
            PHASE54_VO_TRANSFORM_LAYERS_19_31_SEMANTICS,
            "vo-fixed-permutation/transpose16x16-layers19-31-v1"
        );
        let evidence = TransformEvidence {
            identity: PHASE54_VO_TRANSFORM_LAYERS_19_31_SELECTOR,
            semantics: PHASE54_VO_TRANSFORM_LAYERS_19_31_SEMANTICS,
            digest: PHASE54_VO_TRANSFORM_LAYERS_19_31_DIGEST,
            key_rows_transformed: false,
            query_rows_transformed: false,
            value_untransformed: false,
            qk_invariant: true,
            qk_max_abs_delta: 0.0,
            value_rows_transformed: Some(true),
            output_rows_inverse_transformed: Some(true),
            vo_invariant: Some(invariant),
            vo_max_abs_delta: Some(maximum_delta),
            semantic_layer: None,
            semantic_layers: Some(vec![19, 31]),
        };
        assert!(evidence.semantic_layer.is_none());
        assert_eq!(evidence.semantic_layers.as_deref(), Some(&[19, 31][..]));
    }

    #[test]
    fn transform_keeps_boundary_and_signed_zero_codec_inputs_untransformed() {
        let config = Config {
            device_index: 0,
            target: "gfx1030".to_owned(),
            encoding: KvCacheEncoding::Fp8E5M2Block16,
            variant: KvFp8PhysicalVariant::OcpE5M2,
            candidate: transform_candidate(),
            candidate_spec_sha256: digest(b"candidate"),
            output: None,
        };
        let boundary = source_words(1, 256, false, config.variant, ValuesKind::Boundary);
        let signed_zero = source_words(1, 16, false, config.variant, ValuesKind::SignedZero);
        assert_eq!(
            transform_key_rows(&boundary, 1, 256, false).unwrap(),
            boundary
        );
        assert_eq!(
            transform_key_rows(&signed_zero, 1, 16, false).unwrap(),
            signed_zero
        );
        assert!(host_evidence(&config).pass);
    }

    #[test]
    fn report_identity_never_claims_v2_for_non_floor_recipe() {
        let candidate = candidate(Recipe::Ceil, Recipe::Floor);
        assert!(!candidate.production_control());
        assert_eq!(
            candidate.descriptor_compatibility,
            "research-build-semantic-override-not-v2-compatible"
        );
    }
}
