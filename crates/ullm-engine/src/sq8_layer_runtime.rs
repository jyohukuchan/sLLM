// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

use crate::decoder::{PagedDecodeShape, PagedDecodeState};
use crate::host_bytes::{decode_f32_le_values, encode_f32_to_bytes};
use crate::sq_canonical::Sq8CanonicalArtifact;
use crate::sq_runtime::{Sq8CanonicalResidentRuntimeTensor, load_sq8_canonical_resident_tensor};
use crate::sq8_layer_oracle::{
    QWEN3_14B_HEAD_DIM, QWEN3_14B_HIDDEN_SIZE, QWEN3_14B_INTERMEDIATE_SIZE, QWEN3_14B_KV_HEADS,
    QWEN3_14B_Q_HEADS, QWEN3_14B_RMS_NORM_EPSILON, QWEN3_14B_ROPE_THETA,
    QWEN3_14B_SQ8_LAYER_ORACLE_MAX_SEQUENCE_LEN, QWEN3_14B_VALUE_DIM,
};
use ullm_runtime_sys::{
    RuntimeBuffer, RuntimeContext, RuntimeStream, Sq8CkImplementation, Sq8CkQuantizedActivation,
    SqFp8ExecutionPath, add_f32, causal_attn_f32, rope_f32, segmented_rmsnorm_f32, silu_mul_f32,
    sq_fp8_matvec_block2d_batch_f32, sq8_ck_projection_buffer_bytes, sq8_ck_projection_f32,
};

pub const QWEN3_14B_SQ8_LAYER_ACTIVATION_QUANTIZATIONS: usize = 4;
pub const QWEN3_14B_SQ8_LAYER_PROJECTIONS: usize = 7;
pub const QWEN3_14B_SQ8_MAX_EXACT_F32_POSITION: usize = 1 << 24;
pub const QWEN3_14B_SQ8_REQUIRED_HIP_KERNEL_ENV: [&str; 5] = [
    "ULLM_REQUIRE_HIP_RMSNORM_KERNEL",
    "ULLM_REQUIRE_HIP_ROPE_KERNEL",
    "ULLM_REQUIRE_HIP_CAUSAL_ATTN_KERNEL",
    "ULLM_REQUIRE_HIP_ADD_KERNEL",
    "ULLM_REQUIRE_HIP_SILU_MUL_KERNEL",
];
pub const QWEN3_14B_SQ8_PAGED_REQUIRED_HIP_KERNEL_ENV: [&str; 2] = [
    "ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL",
    "ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sq8LayerExecutionProfile {
    ReferenceW8a16Block2d,
    Rdna4W8a8BlockCk,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sq8LayerProjectionExecution {
    Reference(SqFp8ExecutionPath),
    Ck(Sq8CkImplementation),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sq8LayerExecutionReport {
    pub profile: Sq8LayerExecutionProfile,
    pub q: Sq8LayerProjectionExecution,
    pub k: Sq8LayerProjectionExecution,
    pub v: Sq8LayerProjectionExecution,
    pub o: Sq8LayerProjectionExecution,
    pub gate: Sq8LayerProjectionExecution,
    pub up: Sq8LayerProjectionExecution,
    pub down: Sq8LayerProjectionExecution,
    pub activation_quantizations: usize,
    pub projection_calls: usize,
    pub fallback_used: bool,
}

impl Sq8LayerExecutionReport {
    pub fn all_reference_hip(&self) -> bool {
        [
            self.q, self.k, self.v, self.o, self.gate, self.up, self.down,
        ]
        .into_iter()
        .all(|execution| {
            execution == Sq8LayerProjectionExecution::Reference(SqFp8ExecutionPath::HipKernel)
        })
    }

    pub fn all_ck(&self) -> bool {
        [
            self.q, self.k, self.v, self.o, self.gate, self.up, self.down,
        ]
        .into_iter()
        .all(|execution| matches!(execution, Sq8LayerProjectionExecution::Ck(_)))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Qwen3Sq8LayerConfig {
    pub sequence_len: usize,
    pub position_offset: usize,
    rms_norm_epsilon: f32,
    rope_theta: f32,
}

impl Qwen3Sq8LayerConfig {
    pub fn qwen3_14b(sequence_len: usize, position_offset: usize) -> Result<Self, String> {
        let config = Self {
            sequence_len,
            position_offset,
            rms_norm_epsilon: QWEN3_14B_RMS_NORM_EPSILON,
            rope_theta: QWEN3_14B_ROPE_THETA,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(self) -> Result<(), String> {
        if !matches!(self.sequence_len, 1 | 2 | 4 | 8 | 16 | 32 | 128) {
            return Err(format!(
                "Qwen3-14B SQ8 layer sequence_len must be a measured M value, got {}",
                self.sequence_len
            ));
        }
        if self.sequence_len > QWEN3_14B_SQ8_LAYER_ORACLE_MAX_SEQUENCE_LEN {
            return Err(format!(
                "Qwen3-14B SQ8 layer sequence_len {} exceeds {}",
                self.sequence_len, QWEN3_14B_SQ8_LAYER_ORACLE_MAX_SEQUENCE_LEN
            ));
        }
        if self.position_offset != 0 {
            return Err(
                "Qwen3-14B SQ8 layer position_offset requires a past-KV input and is unsupported by the isolated layer runner"
                    .to_string(),
            );
        }
        let final_position = self
            .position_offset
            .checked_add(self.sequence_len - 1)
            .ok_or_else(|| "Qwen3-14B SQ8 layer final RoPE position overflows".to_string())?;
        if final_position > QWEN3_14B_SQ8_MAX_EXACT_F32_POSITION {
            return Err(format!(
                "Qwen3-14B SQ8 layer final RoPE position {final_position} exceeds exact F32 integer range"
            ));
        }
        if !self.rms_norm_epsilon.is_finite() || self.rms_norm_epsilon <= 0.0 {
            return Err("Qwen3-14B SQ8 layer RMSNorm epsilon must be finite and positive".into());
        }
        if !self.rope_theta.is_finite() || self.rope_theta <= 1.0 {
            return Err(
                "Qwen3-14B SQ8 layer RoPE theta must be finite and greater than one".into(),
            );
        }
        if self.rms_norm_epsilon.to_bits() != QWEN3_14B_RMS_NORM_EPSILON.to_bits()
            || self.rope_theta.to_bits() != QWEN3_14B_ROPE_THETA.to_bits()
        {
            return Err("Qwen3-14B SQ8 layer model constants do not match the oracle".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3Sq8LayerNormValues {
    pub input: Vec<f32>,
    pub post_attention: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
}

#[derive(Debug)]
pub struct Qwen3Sq8LayerWeights {
    pub layer_index: usize,
    pub q: Sq8CanonicalResidentRuntimeTensor,
    pub k: Sq8CanonicalResidentRuntimeTensor,
    pub v: Sq8CanonicalResidentRuntimeTensor,
    pub o: Sq8CanonicalResidentRuntimeTensor,
    pub gate: Sq8CanonicalResidentRuntimeTensor,
    pub up: Sq8CanonicalResidentRuntimeTensor,
    pub down: Sq8CanonicalResidentRuntimeTensor,
    input_norm: RuntimeBuffer,
    post_attention_norm: RuntimeBuffer,
    q_norm: RuntimeBuffer,
    k_norm: RuntimeBuffer,
}

pub fn qwen3_sq8_layer_tensor_names(layer_index: usize) -> [String; 7] {
    let prefix = format!("model.layers.{layer_index}");
    [
        format!("{prefix}.self_attn.q_proj.weight"),
        format!("{prefix}.self_attn.k_proj.weight"),
        format!("{prefix}.self_attn.v_proj.weight"),
        format!("{prefix}.self_attn.o_proj.weight"),
        format!("{prefix}.mlp.gate_proj.weight"),
        format!("{prefix}.mlp.up_proj.weight"),
        format!("{prefix}.mlp.down_proj.weight"),
    ]
}

pub fn load_qwen3_14b_sq8_layer_weights(
    context: &mut RuntimeContext,
    stream: &mut RuntimeStream,
    artifact: &Sq8CanonicalArtifact,
    layer_index: usize,
    norms: &Qwen3Sq8LayerNormValues,
    upload_chunk_bytes: usize,
) -> Result<Qwen3Sq8LayerWeights, String> {
    validate_norm_values(norms)?;
    let [
        q_name,
        k_name,
        v_name,
        o_name,
        gate_name,
        up_name,
        down_name,
    ] = qwen3_sq8_layer_tensor_names(layer_index);
    if upload_chunk_bytes == 0 {
        return Err("Qwen3-14B SQ8 layer upload chunk size must be greater than zero".into());
    }
    let chunk_bytes = upload_chunk_bytes;

    let q = load_sq8_canonical_resident_tensor(context, stream, artifact, &q_name, chunk_bytes)?;
    validate_resident_shape(&q, QWEN3_14B_HIDDEN_SIZE, QWEN3_14B_HIDDEN_SIZE, "q")?;
    let k = load_sq8_canonical_resident_tensor(context, stream, artifact, &k_name, chunk_bytes)?;
    validate_resident_shape(
        &k,
        QWEN3_14B_KV_HEADS * QWEN3_14B_HEAD_DIM,
        QWEN3_14B_HIDDEN_SIZE,
        "k",
    )?;
    let v = load_sq8_canonical_resident_tensor(context, stream, artifact, &v_name, chunk_bytes)?;
    validate_resident_shape(
        &v,
        QWEN3_14B_KV_HEADS * QWEN3_14B_VALUE_DIM,
        QWEN3_14B_HIDDEN_SIZE,
        "v",
    )?;
    let o = load_sq8_canonical_resident_tensor(context, stream, artifact, &o_name, chunk_bytes)?;
    validate_resident_shape(&o, QWEN3_14B_HIDDEN_SIZE, QWEN3_14B_HIDDEN_SIZE, "o")?;
    let gate =
        load_sq8_canonical_resident_tensor(context, stream, artifact, &gate_name, chunk_bytes)?;
    validate_resident_shape(
        &gate,
        QWEN3_14B_INTERMEDIATE_SIZE,
        QWEN3_14B_HIDDEN_SIZE,
        "gate",
    )?;
    let up = load_sq8_canonical_resident_tensor(context, stream, artifact, &up_name, chunk_bytes)?;
    validate_resident_shape(
        &up,
        QWEN3_14B_INTERMEDIATE_SIZE,
        QWEN3_14B_HIDDEN_SIZE,
        "up",
    )?;
    let down =
        load_sq8_canonical_resident_tensor(context, stream, artifact, &down_name, chunk_bytes)?;
    validate_resident_shape(
        &down,
        QWEN3_14B_HIDDEN_SIZE,
        QWEN3_14B_INTERMEDIATE_SIZE,
        "down",
    )?;

    let input_norm = upload_f32(context, stream, &norms.input, "input norm")?;
    let post_attention_norm = upload_f32(
        context,
        stream,
        &norms.post_attention,
        "post-attention norm",
    )?;
    let q_norm = upload_f32(context, stream, &norms.q, "q norm")?;
    let k_norm = upload_f32(context, stream, &norms.k, "k norm")?;

    Ok(Qwen3Sq8LayerWeights {
        layer_index,
        q,
        k,
        v,
        o,
        gate,
        up,
        down,
        input_norm,
        post_attention_norm,
        q_norm,
        k_norm,
    })
}

#[derive(Debug)]
pub struct Qwen3Sq8LayerWorkspace {
    config: Qwen3Sq8LayerConfig,
    last_profile: Option<Sq8LayerExecutionProfile>,
    poison_reason: Option<String>,
    input_normed: RuntimeBuffer,
    q_projected: RuntimeBuffer,
    k_projected: RuntimeBuffer,
    v_projected: RuntimeBuffer,
    q_normed: RuntimeBuffer,
    k_normed: RuntimeBuffer,
    q_rope: RuntimeBuffer,
    k_rope: RuntimeBuffer,
    attention: RuntimeBuffer,
    o_projected: RuntimeBuffer,
    attention_residual: RuntimeBuffer,
    post_normed: RuntimeBuffer,
    gate_projected: RuntimeBuffer,
    up_projected: RuntimeBuffer,
    mlp_activation: RuntimeBuffer,
    down_projected: RuntimeBuffer,
    output: RuntimeBuffer,
    projection_workspace: RuntimeBuffer,
    qkv_activation: Sq8CkQuantizedActivation,
    o_activation: Sq8CkQuantizedActivation,
    gate_up_activation: Sq8CkQuantizedActivation,
    down_activation: Sq8CkQuantizedActivation,
}

enum Sq8LayerAttentionMode<'a> {
    Causal,
    PrefillPaged(&'a mut PagedDecodeState),
    DecodePaged {
        position: usize,
        cache: &'a mut PagedDecodeState,
    },
}

impl Qwen3Sq8LayerWorkspace {
    pub fn allocate(
        context: &mut RuntimeContext,
        config: Qwen3Sq8LayerConfig,
    ) -> Result<Self, String> {
        config.validate()?;
        let m = config.sequence_len;
        let hidden_elements = checked_elements(m, QWEN3_14B_HIDDEN_SIZE, "hidden")?;
        let kv_elements = checked_elements(m, QWEN3_14B_KV_HEADS * QWEN3_14B_HEAD_DIM, "kv")?;
        let intermediate_elements =
            checked_elements(m, QWEN3_14B_INTERMEDIATE_SIZE, "intermediate")?;
        let input_normed = alloc_f32(context, hidden_elements, "input normed")?;
        let q_projected = alloc_f32(context, hidden_elements, "q projected")?;
        let k_projected = alloc_f32(context, kv_elements, "k projected")?;
        let v_projected = alloc_f32(context, kv_elements, "v projected")?;
        let q_normed = alloc_f32(context, hidden_elements, "q normed")?;
        let k_normed = alloc_f32(context, kv_elements, "k normed")?;
        let q_rope = alloc_f32(context, hidden_elements, "q rope")?;
        let k_rope = alloc_f32(context, kv_elements, "k rope")?;
        let attention = alloc_f32(context, hidden_elements, "attention")?;
        let o_projected = alloc_f32(context, hidden_elements, "o projected")?;
        let attention_residual = alloc_f32(context, hidden_elements, "attention residual")?;
        let post_normed = alloc_f32(context, hidden_elements, "post normed")?;
        let gate_projected = alloc_f32(context, intermediate_elements, "gate projected")?;
        let up_projected = alloc_f32(context, intermediate_elements, "up projected")?;
        let mlp_activation = alloc_f32(context, intermediate_elements, "MLP activation")?;
        let down_projected = alloc_f32(context, hidden_elements, "down projected")?;
        let output = alloc_f32(context, hidden_elements, "output")?;
        let (projection_workspace_bytes, _) =
            sq8_ck_projection_buffer_bytes(m, QWEN3_14B_INTERMEDIATE_SIZE)?;
        let projection_workspace = context.alloc_buffer(projection_workspace_bytes)?;
        let qkv_activation = Sq8CkQuantizedActivation::allocate(context, m, QWEN3_14B_HIDDEN_SIZE)?;
        let o_activation = Sq8CkQuantizedActivation::allocate(context, m, QWEN3_14B_HIDDEN_SIZE)?;
        let gate_up_activation =
            Sq8CkQuantizedActivation::allocate(context, m, QWEN3_14B_HIDDEN_SIZE)?;
        let down_activation =
            Sq8CkQuantizedActivation::allocate(context, m, QWEN3_14B_INTERMEDIATE_SIZE)?;

        Ok(Self {
            config,
            last_profile: None,
            poison_reason: None,
            input_normed,
            q_projected,
            k_projected,
            v_projected,
            q_normed,
            k_normed,
            q_rope,
            k_rope,
            attention,
            o_projected,
            attention_residual,
            post_normed,
            gate_projected,
            up_projected,
            mlp_activation,
            down_projected,
            output,
            projection_workspace,
            qkv_activation,
            o_activation,
            gate_up_activation,
            down_activation,
        })
    }

    pub fn config(&self) -> Qwen3Sq8LayerConfig {
        self.config
    }

    pub(crate) fn output_buffer(&self) -> &RuntimeBuffer {
        &self.output
    }

    pub fn is_poisoned(&self) -> bool {
        self.poison_reason.is_some()
    }

    pub fn poison_reason(&self) -> Option<&str> {
        self.poison_reason.as_deref()
    }

    pub fn run_synchronized(
        &mut self,
        weights: &Qwen3Sq8LayerWeights,
        input: &RuntimeBuffer,
        profile: Sq8LayerExecutionProfile,
        stream: &mut RuntimeStream,
    ) -> Result<Sq8LayerExecutionReport, String> {
        self.validate_synchronized_preconditions(weights, input)?;
        let report = match self.enqueue(weights, input, profile, stream) {
            Ok(report) => report,
            Err(operation_error) => {
                let error = match stream.synchronize() {
                    Ok(()) => operation_error,
                    Err(sync_error) => format!(
                        "{operation_error}; subsequent Qwen3-14B SQ8 layer stream recovery failed: {sync_error}"
                    ),
                };
                return Err(self.poison(error));
            }
        };
        if let Err(err) = stream.synchronize() {
            return Err(self.poison(format!(
                "failed to synchronize Qwen3-14B SQ8 layer execution: {err}"
            )));
        }
        Ok(report)
    }

    pub(crate) fn enqueue(
        &mut self,
        weights: &Qwen3Sq8LayerWeights,
        input: &RuntimeBuffer,
        profile: Sq8LayerExecutionProfile,
        stream: &mut RuntimeStream,
    ) -> Result<Sq8LayerExecutionReport, String> {
        self.enqueue_with_attention(
            weights,
            input,
            profile,
            Sq8LayerAttentionMode::Causal,
            stream,
        )
    }

    pub(crate) fn enqueue_prefill_with_paged_kv(
        &mut self,
        weights: &Qwen3Sq8LayerWeights,
        input: &RuntimeBuffer,
        profile: Sq8LayerExecutionProfile,
        cache: &mut PagedDecodeState,
        stream: &mut RuntimeStream,
    ) -> Result<Sq8LayerExecutionReport, String> {
        self.enqueue_with_attention(
            weights,
            input,
            profile,
            Sq8LayerAttentionMode::PrefillPaged(cache),
            stream,
        )
    }

    pub(crate) fn enqueue_paged_decode(
        &mut self,
        weights: &Qwen3Sq8LayerWeights,
        input: &RuntimeBuffer,
        profile: Sq8LayerExecutionProfile,
        position: usize,
        cache: &mut PagedDecodeState,
        stream: &mut RuntimeStream,
    ) -> Result<Sq8LayerExecutionReport, String> {
        self.enqueue_with_attention(
            weights,
            input,
            profile,
            Sq8LayerAttentionMode::DecodePaged { position, cache },
            stream,
        )
    }

    fn enqueue_with_attention(
        &mut self,
        weights: &Qwen3Sq8LayerWeights,
        input: &RuntimeBuffer,
        profile: Sq8LayerExecutionProfile,
        attention_mode: Sq8LayerAttentionMode<'_>,
        stream: &mut RuntimeStream,
    ) -> Result<Sq8LayerExecutionReport, String> {
        self.ensure_usable()?;
        self.last_profile = None;
        self.config.validate()?;
        validate_no_host_staging_contract()?;
        let m = self.config.sequence_len;
        let position_offset = match &attention_mode {
            Sq8LayerAttentionMode::Causal => self.config.position_offset,
            Sq8LayerAttentionMode::PrefillPaged(cache) => {
                validate_paged_cache_contract(cache, m)?;
                if cache.written_len() != 0 {
                    return Err(format!(
                        "Qwen3-14B SQ8 prefill requires an empty paged KV cache, got written_len={}",
                        cache.written_len()
                    ));
                }
                validate_paged_no_host_staging_contract()?;
                0
            }
            Sq8LayerAttentionMode::DecodePaged { position, cache } => {
                if m != 1 {
                    return Err(format!(
                        "Qwen3-14B SQ8 paged decode requires M=1, got M={m}"
                    ));
                }
                if *position != cache.written_len() {
                    return Err(format!(
                        "Qwen3-14B SQ8 paged decode position {position} does not match cache written_len {}",
                        cache.written_len()
                    ));
                }
                validate_paged_cache_contract(
                    cache,
                    position.checked_add(1).ok_or_else(|| {
                        "Qwen3-14B SQ8 paged decode position overflows".to_string()
                    })?,
                )?;
                validate_paged_no_host_staging_contract()?;
                *position
            }
        };
        if position_offset > QWEN3_14B_SQ8_MAX_EXACT_F32_POSITION {
            return Err(format!(
                "Qwen3-14B SQ8 layer position {position_offset} exceeds exact F32 integer range"
            ));
        }
        let hidden_elements = checked_elements(m, QWEN3_14B_HIDDEN_SIZE, "hidden")?;
        let intermediate_elements =
            checked_elements(m, QWEN3_14B_INTERMEDIATE_SIZE, "intermediate")?;

        segmented_rmsnorm_f32(
            input,
            &weights.input_norm,
            m,
            QWEN3_14B_HIDDEN_SIZE,
            self.config.rms_norm_epsilon,
            &mut self.input_normed,
            Some(&mut *stream),
        )?;

        let quantization_count = match profile {
            Sq8LayerExecutionProfile::ReferenceW8a16Block2d => 0,
            Sq8LayerExecutionProfile::Rdna4W8a8BlockCk => {
                self.qkv_activation
                    .quantize_f32(&self.input_normed, Some(&mut *stream))?;
                QWEN3_14B_SQ8_LAYER_ACTIVATION_QUANTIZATIONS
            }
        };
        let q = run_projection(
            profile,
            &weights.q,
            &self.input_normed,
            Some(&self.qkv_activation),
            &mut self.projection_workspace,
            &mut self.q_projected,
            m,
            stream,
        )?;
        let k = run_projection(
            profile,
            &weights.k,
            &self.input_normed,
            Some(&self.qkv_activation),
            &mut self.projection_workspace,
            &mut self.k_projected,
            m,
            stream,
        )?;
        let v = run_projection(
            profile,
            &weights.v,
            &self.input_normed,
            Some(&self.qkv_activation),
            &mut self.projection_workspace,
            &mut self.v_projected,
            m,
            stream,
        )?;

        segmented_rmsnorm_f32(
            &self.q_projected,
            &weights.q_norm,
            m * QWEN3_14B_Q_HEADS,
            QWEN3_14B_HEAD_DIM,
            self.config.rms_norm_epsilon,
            &mut self.q_normed,
            Some(&mut *stream),
        )?;
        segmented_rmsnorm_f32(
            &self.k_projected,
            &weights.k_norm,
            m * QWEN3_14B_KV_HEADS,
            QWEN3_14B_HEAD_DIM,
            self.config.rms_norm_epsilon,
            &mut self.k_normed,
            Some(&mut *stream),
        )?;
        rope_f32(
            &self.q_normed,
            m,
            QWEN3_14B_Q_HEADS,
            QWEN3_14B_HEAD_DIM,
            QWEN3_14B_HEAD_DIM,
            position_offset,
            self.config.rope_theta,
            &mut self.q_rope,
            Some(&mut *stream),
        )?;
        rope_f32(
            &self.k_normed,
            m,
            QWEN3_14B_KV_HEADS,
            QWEN3_14B_HEAD_DIM,
            QWEN3_14B_HEAD_DIM,
            position_offset,
            self.config.rope_theta,
            &mut self.k_rope,
            Some(&mut *stream),
        )?;
        let softmax_scale = 1.0 / (QWEN3_14B_HEAD_DIM as f32).sqrt();
        match attention_mode {
            Sq8LayerAttentionMode::Causal => causal_attn_f32(
                &self.q_rope,
                &self.k_rope,
                &self.v_projected,
                m,
                QWEN3_14B_Q_HEADS,
                QWEN3_14B_KV_HEADS,
                QWEN3_14B_HEAD_DIM,
                QWEN3_14B_VALUE_DIM,
                softmax_scale,
                &mut self.attention,
                Some(&mut *stream),
            )?,
            Sq8LayerAttentionMode::PrefillPaged(cache) => {
                let written =
                    cache.write_sequence_from_device(stream, &self.k_rope, &self.v_projected, m)?;
                if written != (0..m) {
                    return Err(format!(
                        "Qwen3-14B SQ8 prefill paged KV range mismatch: expected=0..{m} actual={written:?}"
                    ));
                }
                causal_attn_f32(
                    &self.q_rope,
                    &self.k_rope,
                    &self.v_projected,
                    m,
                    QWEN3_14B_Q_HEADS,
                    QWEN3_14B_KV_HEADS,
                    QWEN3_14B_HEAD_DIM,
                    QWEN3_14B_VALUE_DIM,
                    softmax_scale,
                    &mut self.attention,
                    Some(&mut *stream),
                )?;
            }
            Sq8LayerAttentionMode::DecodePaged { position, cache } => {
                let step = cache.decode_step_from_device(
                    stream,
                    &self.q_rope,
                    &self.k_rope,
                    &self.v_projected,
                    softmax_scale,
                )?;
                if step.cache_position != position || step.cache_len != position + 1 {
                    return Err(format!(
                        "Qwen3-14B SQ8 paged decode state mismatch: position={} cache_len={} expected_position={position}",
                        step.cache_position, step.cache_len
                    ));
                }
                self.attention
                    .copy_from_buffer(
                        0,
                        cache.output_buffer(),
                        0,
                        hidden_elements * std::mem::size_of::<f32>(),
                        Some(&mut *stream),
                    )
                    .map_err(|err| {
                        format!("failed to copy Qwen3-14B SQ8 paged attention output: {err}")
                    })?;
            }
        }

        if profile == Sq8LayerExecutionProfile::Rdna4W8a8BlockCk {
            self.o_activation
                .quantize_f32(&self.attention, Some(&mut *stream))?;
        }
        let o = run_projection(
            profile,
            &weights.o,
            &self.attention,
            Some(&self.o_activation),
            &mut self.projection_workspace,
            &mut self.o_projected,
            m,
            stream,
        )?;
        add_f32(
            input,
            &self.o_projected,
            hidden_elements,
            &mut self.attention_residual,
            Some(&mut *stream),
        )?;
        segmented_rmsnorm_f32(
            &self.attention_residual,
            &weights.post_attention_norm,
            m,
            QWEN3_14B_HIDDEN_SIZE,
            self.config.rms_norm_epsilon,
            &mut self.post_normed,
            Some(&mut *stream),
        )?;

        if profile == Sq8LayerExecutionProfile::Rdna4W8a8BlockCk {
            self.gate_up_activation
                .quantize_f32(&self.post_normed, Some(&mut *stream))?;
        }
        let gate = run_projection(
            profile,
            &weights.gate,
            &self.post_normed,
            Some(&self.gate_up_activation),
            &mut self.projection_workspace,
            &mut self.gate_projected,
            m,
            stream,
        )?;
        let up = run_projection(
            profile,
            &weights.up,
            &self.post_normed,
            Some(&self.gate_up_activation),
            &mut self.projection_workspace,
            &mut self.up_projected,
            m,
            stream,
        )?;
        silu_mul_f32(
            &self.gate_projected,
            &self.up_projected,
            intermediate_elements,
            &mut self.mlp_activation,
            Some(&mut *stream),
        )?;

        if profile == Sq8LayerExecutionProfile::Rdna4W8a8BlockCk {
            self.down_activation
                .quantize_f32(&self.mlp_activation, Some(&mut *stream))?;
        }
        let down = run_projection(
            profile,
            &weights.down,
            &self.mlp_activation,
            Some(&self.down_activation),
            &mut self.projection_workspace,
            &mut self.down_projected,
            m,
            stream,
        )?;
        add_f32(
            &self.attention_residual,
            &self.down_projected,
            hidden_elements,
            &mut self.output,
            Some(&mut *stream),
        )?;

        let fallback_used = [q, k, v, o, gate, up, down]
            .into_iter()
            .any(projection_fallback_used);
        let report = Sq8LayerExecutionReport {
            profile,
            q,
            k,
            v,
            o,
            gate,
            up,
            down,
            activation_quantizations: quantization_count,
            projection_calls: QWEN3_14B_SQ8_LAYER_PROJECTIONS,
            fallback_used,
        };
        self.last_profile = Some(profile);
        Ok(report)
    }

    pub fn read_trace(
        &mut self,
        stream: &mut RuntimeStream,
    ) -> Result<Sq8LayerRuntimeTrace, String> {
        self.ensure_usable()?;
        match self.read_trace_inner(stream) {
            Ok(trace) => Ok(trace),
            Err(err) => {
                Err(self.poison(format!("failed to read Qwen3-14B SQ8 layer trace: {err}")))
            }
        }
    }

    fn read_trace_inner(&self, stream: &mut RuntimeStream) -> Result<Sq8LayerRuntimeTrace, String> {
        self.ensure_usable()?;
        let profile = self
            .last_profile
            .ok_or_else(|| "Qwen3-14B SQ8 layer trace requires one successful run".to_string())?;
        let m = self.config.sequence_len;
        let hidden = checked_elements(m, QWEN3_14B_HIDDEN_SIZE, "trace hidden")?;
        let kv = checked_elements(m, QWEN3_14B_KV_HEADS * QWEN3_14B_HEAD_DIM, "trace kv")?;
        let intermediate = checked_elements(m, QWEN3_14B_INTERMEDIATE_SIZE, "trace intermediate")?;

        let input_normed = read_f32_synchronized(&self.input_normed, hidden, stream)?;
        let q_projected = read_f32_synchronized(&self.q_projected, hidden, stream)?;
        let k_projected = read_f32_synchronized(&self.k_projected, kv, stream)?;
        let v_projected = read_f32_synchronized(&self.v_projected, kv, stream)?;
        let q_normed = read_f32_synchronized(&self.q_normed, hidden, stream)?;
        let k_normed = read_f32_synchronized(&self.k_normed, kv, stream)?;
        let q_rope = read_f32_synchronized(&self.q_rope, hidden, stream)?;
        let k_rope = read_f32_synchronized(&self.k_rope, kv, stream)?;
        let attention = read_f32_synchronized(&self.attention, hidden, stream)?;
        let o_projected = read_f32_synchronized(&self.o_projected, hidden, stream)?;
        let attention_residual = read_f32_synchronized(&self.attention_residual, hidden, stream)?;
        let post_normed = read_f32_synchronized(&self.post_normed, hidden, stream)?;
        let gate_projected = read_f32_synchronized(&self.gate_projected, intermediate, stream)?;
        let up_projected = read_f32_synchronized(&self.up_projected, intermediate, stream)?;
        let mlp_activation = read_f32_synchronized(&self.mlp_activation, intermediate, stream)?;
        let down_projected = read_f32_synchronized(&self.down_projected, hidden, stream)?;
        let output = read_f32_synchronized(&self.output, hidden, stream)?;
        let activations = if profile == Sq8LayerExecutionProfile::Rdna4W8a8BlockCk {
            Some([
                read_activation_synchronized(&self.qkv_activation, stream)?,
                read_activation_synchronized(&self.o_activation, stream)?,
                read_activation_synchronized(&self.gate_up_activation, stream)?,
                read_activation_synchronized(&self.down_activation, stream)?,
            ])
        } else {
            None
        };

        let (qkv_activation, o_activation, gate_up_activation, down_activation) = match activations
        {
            Some([qkv, o, gate_up, down]) => (
                Some(qkv.decode()),
                Some(o.decode()),
                Some(gate_up.decode()),
                Some(down.decode()),
            ),
            None => (None, None, None, None),
        };

        Ok(Sq8LayerRuntimeTrace {
            input_normed: decode_f32_le_values(&input_normed),
            q_projected: decode_f32_le_values(&q_projected),
            k_projected: decode_f32_le_values(&k_projected),
            v_projected: decode_f32_le_values(&v_projected),
            q_normed: decode_f32_le_values(&q_normed),
            k_normed: decode_f32_le_values(&k_normed),
            q_rope: decode_f32_le_values(&q_rope),
            k_rope: decode_f32_le_values(&k_rope),
            attention: decode_f32_le_values(&attention),
            o_projected: decode_f32_le_values(&o_projected),
            attention_residual: decode_f32_le_values(&attention_residual),
            post_normed: decode_f32_le_values(&post_normed),
            gate_projected: decode_f32_le_values(&gate_projected),
            up_projected: decode_f32_le_values(&up_projected),
            mlp_activation: decode_f32_le_values(&mlp_activation),
            down_projected: decode_f32_le_values(&down_projected),
            output: decode_f32_le_values(&output),
            qkv_activation,
            o_activation,
            gate_up_activation,
            down_activation,
        })
    }

    fn validate_synchronized_preconditions(
        &self,
        weights: &Qwen3Sq8LayerWeights,
        input: &RuntimeBuffer,
    ) -> Result<(), String> {
        self.ensure_usable()?;
        self.config.validate()?;
        validate_no_host_staging_contract()?;
        let hidden_bytes =
            checked_elements(self.config.sequence_len, QWEN3_14B_HIDDEN_SIZE, "input")?
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| "Qwen3-14B SQ8 layer input byte size overflows".to_string())?;
        let actual_input_bytes = input.size()?;
        if actual_input_bytes != hidden_bytes {
            return Err(format!(
                "Qwen3-14B SQ8 layer input byte size mismatch: expected={hidden_bytes} actual={actual_input_bytes}"
            ));
        }
        validate_resident_shape(
            &weights.q,
            QWEN3_14B_HIDDEN_SIZE,
            QWEN3_14B_HIDDEN_SIZE,
            "q",
        )?;
        validate_resident_shape(
            &weights.k,
            QWEN3_14B_KV_HEADS * QWEN3_14B_HEAD_DIM,
            QWEN3_14B_HIDDEN_SIZE,
            "k",
        )?;
        validate_resident_shape(
            &weights.v,
            QWEN3_14B_KV_HEADS * QWEN3_14B_VALUE_DIM,
            QWEN3_14B_HIDDEN_SIZE,
            "v",
        )?;
        validate_resident_shape(
            &weights.o,
            QWEN3_14B_HIDDEN_SIZE,
            QWEN3_14B_HIDDEN_SIZE,
            "o",
        )?;
        validate_resident_shape(
            &weights.gate,
            QWEN3_14B_INTERMEDIATE_SIZE,
            QWEN3_14B_HIDDEN_SIZE,
            "gate",
        )?;
        validate_resident_shape(
            &weights.up,
            QWEN3_14B_INTERMEDIATE_SIZE,
            QWEN3_14B_HIDDEN_SIZE,
            "up",
        )?;
        validate_resident_shape(
            &weights.down,
            QWEN3_14B_HIDDEN_SIZE,
            QWEN3_14B_INTERMEDIATE_SIZE,
            "down",
        )?;
        for (label, buffer, expected_elements) in [
            ("input norm", &weights.input_norm, QWEN3_14B_HIDDEN_SIZE),
            (
                "post-attention norm",
                &weights.post_attention_norm,
                QWEN3_14B_HIDDEN_SIZE,
            ),
            ("q norm", &weights.q_norm, QWEN3_14B_HEAD_DIM),
            ("k norm", &weights.k_norm, QWEN3_14B_HEAD_DIM),
        ] {
            let expected_bytes = expected_elements
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| format!("Qwen3-14B SQ8 layer {label} byte size overflows"))?;
            let actual_bytes = buffer.size()?;
            if actual_bytes != expected_bytes {
                return Err(format!(
                    "Qwen3-14B SQ8 layer {label} byte size mismatch: expected={expected_bytes} actual={actual_bytes}"
                ));
            }
        }
        Ok(())
    }

    fn ensure_usable(&self) -> Result<(), String> {
        match &self.poison_reason {
            Some(reason) => Err(format!(
                "Qwen3-14B SQ8 layer workspace is permanently poisoned: {reason}"
            )),
            None => Ok(()),
        }
    }

    fn poison(&mut self, error: String) -> String {
        self.last_profile = None;
        if self.poison_reason.is_none() {
            self.poison_reason = Some(error.clone());
        }
        error
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sq8LayerQuantizedActivationTrace {
    pub m: usize,
    pub k: usize,
    pub values: Vec<u8>,
    pub scales: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sq8LayerRuntimeTrace {
    pub input_normed: Vec<f32>,
    pub q_projected: Vec<f32>,
    pub k_projected: Vec<f32>,
    pub v_projected: Vec<f32>,
    pub q_normed: Vec<f32>,
    pub k_normed: Vec<f32>,
    pub q_rope: Vec<f32>,
    pub k_rope: Vec<f32>,
    pub attention: Vec<f32>,
    pub o_projected: Vec<f32>,
    pub attention_residual: Vec<f32>,
    pub post_normed: Vec<f32>,
    pub gate_projected: Vec<f32>,
    pub up_projected: Vec<f32>,
    pub mlp_activation: Vec<f32>,
    pub down_projected: Vec<f32>,
    pub output: Vec<f32>,
    pub qkv_activation: Option<Sq8LayerQuantizedActivationTrace>,
    pub o_activation: Option<Sq8LayerQuantizedActivationTrace>,
    pub gate_up_activation: Option<Sq8LayerQuantizedActivationTrace>,
    pub down_activation: Option<Sq8LayerQuantizedActivationTrace>,
}

struct PendingActivationRead {
    m: usize,
    k: usize,
    values: Vec<u8>,
    scales: Vec<u8>,
}

impl PendingActivationRead {
    fn decode(self) -> Sq8LayerQuantizedActivationTrace {
        Sq8LayerQuantizedActivationTrace {
            m: self.m,
            k: self.k,
            values: self.values,
            scales: decode_f32_le_values(&self.scales),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_projection(
    profile: Sq8LayerExecutionProfile,
    weight: &Sq8CanonicalResidentRuntimeTensor,
    input: &RuntimeBuffer,
    activation: Option<&Sq8CkQuantizedActivation>,
    workspace: &mut RuntimeBuffer,
    output: &mut RuntimeBuffer,
    m: usize,
    stream: &mut RuntimeStream,
) -> Result<Sq8LayerProjectionExecution, String> {
    match profile {
        Sq8LayerExecutionProfile::ReferenceW8a16Block2d => {
            let path = sq_fp8_matvec_block2d_batch_f32(
                &weight.payload_buffer,
                &weight.scale_buffer,
                input,
                weight.rows,
                weight.cols,
                weight.block_rows,
                weight.block_cols,
                m,
                output,
                Some(stream),
            )?;
            Ok(Sq8LayerProjectionExecution::Reference(path))
        }
        Sq8LayerExecutionProfile::Rdna4W8a8BlockCk => {
            let activation = activation.ok_or_else(|| {
                format!(
                    "SQ8 CK projection {} is missing quantized activation",
                    weight.tensor_name
                )
            })?;
            if activation.m() != m || activation.k() != weight.cols {
                return Err(format!(
                    "SQ8 CK projection {} activation shape mismatch: activation=[{},{}] expected=[{},{}]",
                    weight.tensor_name,
                    activation.m(),
                    activation.k(),
                    m,
                    weight.cols
                ));
            }
            let implementation = sq8_ck_projection_f32(
                activation,
                &weight.payload_buffer,
                &weight.scale_buffer,
                weight.rows,
                workspace,
                output,
                Some(stream),
            )?;
            Ok(Sq8LayerProjectionExecution::Ck(implementation))
        }
    }
}

fn projection_fallback_used(execution: Sq8LayerProjectionExecution) -> bool {
    matches!(
        execution,
        Sq8LayerProjectionExecution::Reference(path) if path != SqFp8ExecutionPath::HipKernel
    )
}

pub(crate) fn validate_norm_values(norms: &Qwen3Sq8LayerNormValues) -> Result<(), String> {
    for (name, values, expected) in [
        ("input", norms.input.as_slice(), QWEN3_14B_HIDDEN_SIZE),
        (
            "post_attention",
            norms.post_attention.as_slice(),
            QWEN3_14B_HIDDEN_SIZE,
        ),
        ("q", norms.q.as_slice(), QWEN3_14B_HEAD_DIM),
        ("k", norms.k.as_slice(), QWEN3_14B_HEAD_DIM),
    ] {
        if values.len() != expected {
            return Err(format!(
                "Qwen3-14B SQ8 layer {name} norm length mismatch: expected={expected} actual={}",
                values.len()
            ));
        }
        if let Some((index, value)) = values
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(format!(
                "Qwen3-14B SQ8 layer {name} norm contains non-finite value {value} at index {index}"
            ));
        }
    }
    Ok(())
}

fn validate_no_host_staging_contract() -> Result<(), String> {
    let missing = QWEN3_14B_SQ8_REQUIRED_HIP_KERNEL_ENV
        .into_iter()
        .filter(|name| std::env::var_os(name).is_none())
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(format!(
            "Qwen3-14B SQ8 layer requires HIP-only primitive execution; set these no-staging guards before process start: {}",
            missing.join(",")
        ));
    }
    Ok(())
}

fn validate_paged_no_host_staging_contract() -> Result<(), String> {
    let missing = QWEN3_14B_SQ8_PAGED_REQUIRED_HIP_KERNEL_ENV
        .into_iter()
        .filter(|name| std::env::var_os(name).is_none())
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(format!(
            "Qwen3-14B SQ8 paged decode requires HIP-only KV primitives; set these no-staging guards before process start: {}",
            missing.join(",")
        ));
    }
    Ok(())
}

fn validate_paged_cache_contract(
    cache: &PagedDecodeState,
    required_len: usize,
) -> Result<(), String> {
    let shape = cache.shape();
    let expected = PagedDecodeShape {
        block_size: shape.block_size,
        cache_blocks: shape.cache_blocks,
        q_heads: QWEN3_14B_Q_HEADS,
        kv_heads: QWEN3_14B_KV_HEADS,
        head_dim: QWEN3_14B_HEAD_DIM,
        value_dim: QWEN3_14B_VALUE_DIM,
    };
    if shape != expected {
        return Err(format!(
            "Qwen3-14B SQ8 paged KV shape mismatch: expected q_heads={} kv_heads={} head_dim={} value_dim={} actual={shape:?}",
            QWEN3_14B_Q_HEADS, QWEN3_14B_KV_HEADS, QWEN3_14B_HEAD_DIM, QWEN3_14B_VALUE_DIM
        ));
    }
    let logical_capacity = cache
        .block_table()
        .len()
        .checked_mul(shape.block_size)
        .ok_or_else(|| "Qwen3-14B SQ8 paged KV logical capacity overflows".to_string())?;
    if required_len == 0 || required_len > logical_capacity {
        return Err(format!(
            "Qwen3-14B SQ8 paged KV required length {required_len} exceeds logical capacity {logical_capacity}"
        ));
    }
    Ok(())
}

fn validate_resident_shape(
    tensor: &Sq8CanonicalResidentRuntimeTensor,
    rows: usize,
    cols: usize,
    label: &str,
) -> Result<(), String> {
    if tensor.rows != rows || tensor.cols != cols {
        return Err(format!(
            "Qwen3-14B SQ8 layer {label} shape mismatch: expected=[{rows},{cols}] actual=[{},{}]",
            tensor.rows, tensor.cols
        ));
    }
    if tensor.block_rows != 128 || tensor.block_cols != 128 {
        return Err(format!(
            "Qwen3-14B SQ8 layer {label} requires 128x128 weight blocks, got {}x{}",
            tensor.block_rows, tensor.block_cols
        ));
    }
    let expected_scale_rows = rows.div_ceil(128);
    let expected_scale_cols = cols.div_ceil(128);
    if tensor.scale_rows != expected_scale_rows || tensor.scale_cols != expected_scale_cols {
        return Err(format!(
            "Qwen3-14B SQ8 layer {label} scale shape mismatch: expected=[{expected_scale_rows},{expected_scale_cols}] actual=[{},{}]",
            tensor.scale_rows, tensor.scale_cols
        ));
    }
    Ok(())
}

fn checked_elements(rows: usize, cols: usize, label: &str) -> Result<usize, String> {
    rows.checked_mul(cols)
        .ok_or_else(|| format!("Qwen3-14B SQ8 layer {label} element count overflows"))
}

fn alloc_f32(
    context: &mut RuntimeContext,
    elements: usize,
    label: &str,
) -> Result<RuntimeBuffer, String> {
    let bytes = elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| format!("Qwen3-14B SQ8 layer {label} byte size overflows"))?;
    context
        .alloc_buffer(bytes)
        .map_err(|err| format!("failed to allocate Qwen3-14B SQ8 layer {label}: {err}"))
}

fn upload_f32(
    context: &mut RuntimeContext,
    stream: &mut RuntimeStream,
    values: &[f32],
    label: &str,
) -> Result<RuntimeBuffer, String> {
    let bytes = encode_f32_to_bytes(values);
    let mut buffer = context
        .alloc_buffer(bytes.len())
        .map_err(|err| format!("failed to allocate Qwen3-14B SQ8 layer {label}: {err}"))?;
    buffer
        .copy_from_host(0, &bytes, Some(&mut *stream))
        .map_err(|err| format!("failed to upload Qwen3-14B SQ8 layer {label}: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed to synchronize Qwen3-14B SQ8 layer {label}: {err}"))?;
    Ok(buffer)
}

fn read_f32_synchronized(
    buffer: &RuntimeBuffer,
    elements: usize,
    stream: &mut RuntimeStream,
) -> Result<Vec<u8>, String> {
    let bytes = elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "Qwen3-14B SQ8 layer trace byte size overflows".to_string())?;
    let mut host = vec![0_u8; bytes];
    copy_to_host_synchronized(buffer, &mut host, stream)?;
    Ok(host)
}

fn read_activation_synchronized(
    activation: &Sq8CkQuantizedActivation,
    stream: &mut RuntimeStream,
) -> Result<PendingActivationRead, String> {
    let mut values = vec![0_u8; activation.quantized_bytes()];
    let mut scales = vec![0_u8; activation.scale_bytes()];
    copy_to_host_synchronized(activation.quantized_buffer(), &mut values, stream)?;
    copy_to_host_synchronized(activation.scale_buffer(), &mut scales, stream)?;
    Ok(PendingActivationRead {
        m: activation.m(),
        k: activation.k(),
        values,
        scales,
    })
}

fn copy_to_host_synchronized(
    buffer: &RuntimeBuffer,
    destination: &mut [u8],
    stream: &mut RuntimeStream,
) -> Result<(), String> {
    let copy_result = buffer.copy_to_host(0, destination, Some(&mut *stream));
    let synchronize_result = stream.synchronize();
    match (copy_result, synchronize_result) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(copy_error), Ok(())) => Err(copy_error),
        (Ok(()), Err(sync_error)) => Err(sync_error),
        (Err(copy_error), Err(sync_error)) => Err(format!(
            "runtime buffer read failed: {copy_error}; subsequent stream synchronization also failed: {sync_error}"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen3_14b_layer_config_accepts_only_measured_sequence_lengths() {
        for m in [1, 2, 4, 8, 16, 32, 128] {
            assert!(Qwen3Sq8LayerConfig::qwen3_14b(m, 0).is_ok());
        }
        for m in [0, 3, 64, 129] {
            assert!(Qwen3Sq8LayerConfig::qwen3_14b(m, 0).is_err());
        }
    }

    #[test]
    fn qwen3_14b_layer_config_rejects_inexact_rope_positions() {
        assert!(Qwen3Sq8LayerConfig::qwen3_14b(1, 1).is_err());
        assert!(Qwen3Sq8LayerConfig::qwen3_14b(2, QWEN3_14B_SQ8_MAX_EXACT_F32_POSITION).is_err());
    }

    #[test]
    fn qwen3_14b_layer_names_cover_exactly_seven_projections() {
        assert_eq!(
            qwen3_sq8_layer_tensor_names(0),
            [
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.k_proj.weight",
                "model.layers.0.self_attn.v_proj.weight",
                "model.layers.0.self_attn.o_proj.weight",
                "model.layers.0.mlp.gate_proj.weight",
                "model.layers.0.mlp.up_proj.weight",
                "model.layers.0.mlp.down_proj.weight",
            ]
            .map(str::to_string)
        );
    }

    #[test]
    fn workspace_allocates_on_cpu_without_enabling_ck_feature() {
        let mut context = RuntimeContext::create(0).unwrap();
        let config = Qwen3Sq8LayerConfig::qwen3_14b(1, 0).unwrap();
        let workspace = Qwen3Sq8LayerWorkspace::allocate(&mut context, config).unwrap();
        assert_eq!(
            workspace.output_buffer().size().unwrap(),
            QWEN3_14B_HIDDEN_SIZE * std::mem::size_of::<f32>()
        );
    }

    #[test]
    fn workspace_poison_is_permanent() {
        let mut context = RuntimeContext::create(0).unwrap();
        let config = Qwen3Sq8LayerConfig::qwen3_14b(1, 0).unwrap();
        let mut workspace = Qwen3Sq8LayerWorkspace::allocate(&mut context, config).unwrap();
        assert!(!workspace.is_poisoned());
        assert_eq!(
            workspace.poison("stream failure".to_string()),
            "stream failure"
        );
        assert!(workspace.is_poisoned());
        assert_eq!(workspace.poison_reason(), Some("stream failure"));
        assert!(workspace.ensure_usable().is_err());
        workspace.poison("replacement".to_string());
        assert_eq!(workspace.poison_reason(), Some("stream failure"));
    }

    #[test]
    fn reference_cpu_execution_is_reported_as_fallback() {
        assert!(!projection_fallback_used(
            Sq8LayerProjectionExecution::Reference(SqFp8ExecutionPath::HipKernel)
        ));
        assert!(projection_fallback_used(
            Sq8LayerProjectionExecution::Reference(SqFp8ExecutionPath::CpuReference)
        ));
        assert!(!projection_fallback_used(Sq8LayerProjectionExecution::Ck(
            Sq8CkImplementation::MemV1DefaultTile16x128x128
        )));
    }
}
