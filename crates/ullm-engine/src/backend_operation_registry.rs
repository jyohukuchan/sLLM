// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Typed, load-time backend operation selection.
//!
//! This registry is intentionally separate from [`crate::backend_dispatch`].  The latter is a
//! compatibility matcher for historical string catalogs; this module resolves semantic operations
//! to runnable entries, checked workspace requirements, and explicit state effects before device
//! execution starts.

use crate::execution_batch::ExecutionPhase;
use crate::model_graph::{NumericalFormat, TensorLayout};

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum ProbeFaultStage {
    None = 0,
    BeforeStart = 1,
    QkvPrepare = 2,
    Recurrent = 3,
    PagedPlain = 4,
    PagedGated = 5,
    PagedKvWrite = 6,
    FusedWriter = 7,
    Synchronize = 8,
    Aq4MatvecBatch = 9,
    QkvPrepareBatch = 10,
    RecurrentSequence = 11,
}

#[cfg(test)]
static FORCE_M1_PROBE_FAILURE_STAGE: std::sync::atomic::AtomicU8 =
    std::sync::atomic::AtomicU8::new(ProbeFaultStage::None as u8);

type ProbeCacheKey = (u8, Option<String>, i32, u32, u64);
static M1_PROBE_CACHE: std::sync::OnceLock<
    std::sync::Mutex<std::collections::BTreeMap<ProbeCacheKey, RuntimeFeatureSet>>,
> = std::sync::OnceLock::new();

fn probe_cache_key(capabilities: &DeviceCapabilities) -> ProbeCacheKey {
    let backend = match capabilities.backend {
        OperationBackend::Host => 0,
        OperationBackend::Hip => 1,
    };
    (
        backend,
        capabilities.architecture.clone(),
        capabilities.device_id,
        capabilities.abi_version,
        capabilities.runtime_features.bits(),
    )
}

#[cfg(test)]
static M1_PROBE_CHECKPOINT_COUNTS: [std::sync::atomic::AtomicUsize; 12] =
    [const { std::sync::atomic::AtomicUsize::new(0) }; 12];

#[cfg(test)]
std::thread_local! {
    static WRITER_SYS_CALL_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static PREPARE_SYS_CALL_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static RECURRENT_SYS_CALL_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn force_probe_failure(stage: ProbeFaultStage) {
    FORCE_M1_PROBE_FAILURE_STAGE.store(stage as u8, std::sync::atomic::Ordering::Release);
}

fn probe_fault_checkpoint(stage: u8, label: &str) -> Result<(), String> {
    #[cfg(test)]
    M1_PROBE_CHECKPOINT_COUNTS[usize::from(stage)]
        .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
    #[cfg(test)]
    if FORCE_M1_PROBE_FAILURE_STAGE
        .compare_exchange(
            stage,
            ProbeFaultStage::None as u8,
            std::sync::atomic::Ordering::AcqRel,
            std::sync::atomic::Ordering::Acquire,
        )
        .is_ok()
    {
        return Err(format!(
            "injected backend M1 capability probe failure at {label}"
        ));
    }
    let _ = (stage, label);
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Aq4ProbeBinding {
    scale_count: usize,
    group_size: usize,
    tensor_scale_bits: u32,
    row_scale_count: usize,
    rows: usize,
    cols: usize,
    batch_count: usize,
}

const fn aq4_probe_binding(batch_count: usize) -> Aq4ProbeBinding {
    Aq4ProbeBinding {
        scale_count: 2,
        group_size: 2,
        tensor_scale_bits: 1.0_f32.to_bits(),
        row_scale_count: 0,
        rows: 2,
        cols: 3,
        batch_count,
    }
}

/// A hard bound that prevents a package or plugin from growing registry memory without limit.
pub const MAX_BACKEND_OPERATION_IMPLEMENTATIONS: usize = 4_096;
pub const MAX_OPERATION_IDENTIFIER_BYTES: usize = 128;

/// Semantic operation requested by graph lowering.  These are not kernel names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationKind {
    /// Writes one plain K/V token into a paged cache.
    PagedKvWrite,
    /// Fused Q/K normalization, RoPE, gate split, and paged K/V write.
    FusedQkNormRopePagedKvWrite,
    /// Convolution-history update and Q/K/V preparation before the recurrent scan.
    LinearAttentionQkvPrepare,
    /// Sequence-width convolution-history update and Q/K/V preparation for one request.
    LinearAttentionQkvPrepareBatch,
    /// Recurrent-state update only; convolution/QKV preparation is a separate operation.
    GatedDeltaRuleScan,
    /// Sequence-width recurrent-state scan for one request.
    GatedDeltaRuleSequence,
    /// Sequence-width AQ4 projection for one or more rows.
    Aq4MatvecBatch,
    /// Read-only paged causal GQA over an already-written KV cache.
    PagedCausalGqaRead,
}

/// Concrete shape bucket used by the currently promoted width-one implementations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationGeometry {
    PagedKvWrite {
        kv_heads: usize,
        head_dim: usize,
        value_dim: usize,
        block_size: usize,
        cache_blocks: usize,
    },
    FusedQkNormRopePagedKvWrite {
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        value_dim: usize,
        rotary_dim: usize,
        rope_base_bits: u32,
        norm_epsilon_bits: u32,
        block_size: usize,
        cache_blocks: usize,
    },
    LinearAttentionQkvPrepare {
        key_heads: usize,
        value_heads: usize,
        key_dim: usize,
        value_dim: usize,
        kernel_size: usize,
        query_scale: QueryScale,
        qk_l2_norm: bool,
    },
    GatedDeltaRule {
        key_heads: usize,
        value_heads: usize,
        key_dim: usize,
        value_dim: usize,
    },
    Aq4MatvecBatch {
        rows: usize,
        cols: usize,
        group_size: usize,
        scale_count: usize,
        row_scale_count: usize,
        tensor_scale_bits: u32,
    },
    PagedCausalGqaRead {
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        value_dim: usize,
        block_size: usize,
        cache_blocks: usize,
        sigmoid_gate: bool,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryScale {
    InverseSqrtKeyDim,
    ExactF32Bits(u32),
}

/// Backend family is a typed capability field and never a model identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationBackend {
    Host,
    Hip,
}

/// Load-time device facts supplied by the runtime/profile capability probe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceCapabilities {
    pub device_id: i32,
    pub backend: OperationBackend,
    pub architecture: Option<String>,
    pub device_name: Option<String>,
    pub abi_version: u32,
    pub runtime_features: RuntimeFeatureSet,
    pub workspace_capacity_bytes: u64,
}

/// Runtime features that can be proven at load time by a served execution profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum RuntimeFeature {
    HipLinearAttentionRecurrent = 0,
    HipPagedDecodeAttention = 1,
    HipFusedQkNormRopePagedKvWrite = 2,
    HipLinearAttentionQkvPrepare = 3,
    HipPagedKvWrite = 4,
    HipAq4MatvecBatch = 5,
    HipLinearAttentionQkvPrepareBatch = 6,
    HipLinearAttentionRecurrentSequence = 7,
}

/// Canonical production guard for a probed runtime feature.
pub const fn runtime_feature_environment(feature: RuntimeFeature) -> &'static str {
    match feature {
        RuntimeFeature::HipLinearAttentionRecurrent => {
            "ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_KERNEL"
        }
        RuntimeFeature::HipPagedDecodeAttention => "ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL",
        RuntimeFeature::HipFusedQkNormRopePagedKvWrite => {
            "ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_PAGED_KV_WRITE_KERNEL"
        }
        RuntimeFeature::HipLinearAttentionQkvPrepare => "ULLM_REQUIRE_HIP_LINEAR_ATTN_KERNEL",
        RuntimeFeature::HipPagedKvWrite => "ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL",
        RuntimeFeature::HipAq4MatvecBatch => "ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL",
        RuntimeFeature::HipLinearAttentionQkvPrepareBatch => {
            "ULLM_REQUIRE_HIP_LINEAR_ATTN_QKV_PREPARE_BATCH_KERNEL"
        }
        RuntimeFeature::HipLinearAttentionRecurrentSequence => {
            "ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_SEQUENCE_KERNEL"
        }
    }
}

impl DeviceCapabilities {
    /// Probes the real runtime context and admits HIP features only when their production guards
    /// have already been validated to the exact value `1`.
    pub fn from_runtime_context(
        context: &ullm_runtime_sys::RuntimeContext,
    ) -> Result<Self, String> {
        let info = context
            .device_info()
            .map_err(|error| format!("failed to probe backend operation device: {error}"))?;
        let backend = match info.backend.as_str() {
            "cpu" => OperationBackend::Host,
            "hip" => OperationBackend::Hip,
            other => {
                return Err(format!(
                    "unsupported backend operation runtime backend {other}"
                ));
            }
        };
        let mut runtime_features = RuntimeFeatureSet::EMPTY;
        if backend == OperationBackend::Hip {
            for feature in [
                RuntimeFeature::HipLinearAttentionRecurrent,
                RuntimeFeature::HipPagedDecodeAttention,
                RuntimeFeature::HipFusedQkNormRopePagedKvWrite,
                RuntimeFeature::HipLinearAttentionQkvPrepare,
                RuntimeFeature::HipPagedKvWrite,
                RuntimeFeature::HipAq4MatvecBatch,
                RuntimeFeature::HipLinearAttentionQkvPrepareBatch,
                RuntimeFeature::HipLinearAttentionRecurrentSequence,
            ] {
                if std::env::var_os(runtime_feature_environment(feature)).as_deref()
                    == Some(std::ffi::OsStr::new("1"))
                {
                    runtime_features = runtime_features.with(feature);
                }
            }
        }
        Ok(Self {
            device_id: info.device_id,
            backend,
            architecture: normalized_device_architecture(&info)?,
            device_name: (!info.name.is_empty()).then_some(info.name),
            abi_version: ullm_runtime_sys::abi_version(),
            runtime_features,
            workspace_capacity_bytes: info.total_global_mem,
        })
    }

    /// Runs isolated M1 scratch calls once per physical runtime device. Environment guards are
    /// only policy inputs; features are returned only after the corresponding ABI calls and stream
    /// synchronization succeed.
    pub fn probe_m1_runtime_context(
        context: &mut ullm_runtime_sys::RuntimeContext,
        stream: &mut ullm_runtime_sys::RuntimeStream,
    ) -> Result<Self, String> {
        let mut capabilities = Self::from_runtime_context(context)?;
        probe_fault_checkpoint(1, "before-start")?;
        if capabilities.backend != OperationBackend::Hip {
            capabilities.runtime_features = RuntimeFeatureSet::EMPTY;
            return Ok(capabilities);
        }
        let policy = capabilities.runtime_features;
        let key = probe_cache_key(&capabilities);
        if let Some(features) = M1_PROBE_CACHE
            .get_or_init(Default::default)
            .lock()
            .map_err(|_| "backend capability probe cache is poisoned".to_string())?
            .get(&key)
            .copied()
        {
            capabilities.runtime_features = features;
            return Ok(capabilities);
        }

        fn zeros(
            context: &mut ullm_runtime_sys::RuntimeContext,
            stream: &mut ullm_runtime_sys::RuntimeStream,
            elements: usize,
        ) -> Result<ullm_runtime_sys::RuntimeBuffer, String> {
            let bytes = elements
                .checked_mul(4)
                .ok_or_else(|| "backend probe buffer bytes overflow".to_string())?;
            let mut buffer = context.alloc_buffer(bytes)?;
            buffer.zero(0, bytes, Some(stream))?;
            Ok(buffer)
        }
        let mut proven = RuntimeFeatureSet::EMPTY;

        if policy.contains(RuntimeFeature::HipLinearAttentionQkvPrepare) {
            let qkv = zeros(context, stream, 8_192)?;
            let conv_weight = zeros(context, stream, 8_192 * 4)?;
            let mut history = zeros(context, stream, 8_192 * 4)?;
            let mut conv_output = zeros(context, stream, 8_192)?;
            let mut q = zeros(context, stream, 16 * 128)?;
            let mut k = zeros(context, stream, 16 * 128)?;
            let mut v = zeros(context, stream, 32 * 128)?;
            ullm_runtime_sys::linear_attn_qkv_prepare_f32(
                &qkv,
                &conv_weight,
                &mut history,
                16,
                32,
                128,
                128,
                4,
                1.0 / 128.0_f32.sqrt(),
                true,
                &mut conv_output,
                &mut q,
                &mut k,
                &mut v,
                Some(stream),
            )?;
            probe_fault_checkpoint(2, "qkv-prepare")?;
            proven = proven.with(RuntimeFeature::HipLinearAttentionQkvPrepare);
        }
        if policy.contains(RuntimeFeature::HipLinearAttentionRecurrent) {
            let q = zeros(context, stream, 16 * 128)?;
            let k = zeros(context, stream, 16 * 128)?;
            let v = zeros(context, stream, 32 * 128)?;
            let gate = zeros(context, stream, 32)?;
            let beta = zeros(context, stream, 32)?;
            let mut state = zeros(context, stream, 32 * 128 * 128)?;
            let mut output = zeros(context, stream, 32 * 128)?;
            ullm_runtime_sys::linear_attn_recurrent_f32(
                &q,
                &k,
                &v,
                &gate,
                &beta,
                16,
                32,
                1,
                128,
                128,
                &mut state,
                &mut output,
                Some(stream),
            )?;
            probe_fault_checkpoint(3, "recurrent")?;
            proven = proven.with(RuntimeFeature::HipLinearAttentionRecurrent);
        }
        if policy.contains(RuntimeFeature::HipLinearAttentionRecurrentSequence) {
            const SEQUENCE_LEN: usize = 128;
            let q = zeros(context, stream, SEQUENCE_LEN * 16 * 128)?;
            let k = zeros(context, stream, SEQUENCE_LEN * 16 * 128)?;
            let v = zeros(context, stream, SEQUENCE_LEN * 32 * 128)?;
            let gate = zeros(context, stream, SEQUENCE_LEN * 32)?;
            let beta = zeros(context, stream, SEQUENCE_LEN * 32)?;
            let mut state = zeros(context, stream, 32 * 128 * 128)?;
            let mut output = zeros(context, stream, SEQUENCE_LEN * 32 * 128)?;
            ullm_runtime_sys::linear_attn_recurrent_f32(
                &q,
                &k,
                &v,
                &gate,
                &beta,
                16,
                32,
                SEQUENCE_LEN,
                128,
                128,
                &mut state,
                &mut output,
                Some(stream),
            )?;
            probe_fault_checkpoint(11, "recurrent-sequence")?;
            proven = proven.with(RuntimeFeature::HipLinearAttentionRecurrentSequence);
        }
        if policy.contains(RuntimeFeature::HipPagedDecodeAttention) {
            let q = zeros(context, stream, 32 * 128)?;
            let gate = zeros(context, stream, 32 * 128)?;
            let mut k_cache = zeros(context, stream, 16 * 256 * 4 * 256)?;
            let mut v_cache = zeros(context, stream, 16 * 256 * 4 * 256)?;
            let mut table = context.alloc_buffer(16 * 4)?;
            let table_bytes = (0_u32..16).flat_map(u32::to_le_bytes).collect::<Vec<_>>();
            table.copy_from_host(0, &table_bytes, Some(stream))?;
            let mut plain_output = zeros(context, stream, 32 * 128)?;
            let mut gated_output = zeros(context, stream, 32 * 128)?;
            ullm_runtime_sys::paged_decode_attn_f32(
                &q,
                &k_cache,
                &v_cache,
                &table,
                1,
                256,
                16,
                16,
                4,
                256,
                256,
                1.0 / 256.0_f32.sqrt(),
                &mut plain_output,
                Some(stream),
            )?;
            probe_fault_checkpoint(4, "paged-plain")?;
            ullm_runtime_sys::paged_decode_attn_sigmoid_gate_f32(
                &q,
                &gate,
                &k_cache,
                &v_cache,
                &table,
                1,
                256,
                16,
                16,
                4,
                256,
                256,
                1.0 / 256.0_f32.sqrt(),
                &mut gated_output,
                Some(stream),
            )?;
            probe_fault_checkpoint(5, "paged-gated")?;
            proven = proven.with(RuntimeFeature::HipPagedDecodeAttention);
            if policy.contains(RuntimeFeature::HipPagedKvWrite) {
                let k = zeros(context, stream, 4 * 256)?;
                let v = zeros(context, stream, 4 * 256)?;
                ullm_runtime_sys::paged_kv_write_f32(
                    &k,
                    &v,
                    &table,
                    0,
                    256,
                    16,
                    4,
                    256,
                    256,
                    &mut k_cache,
                    &mut v_cache,
                    Some(stream),
                )?;
                probe_fault_checkpoint(6, "paged-kv-write")?;
                proven = proven.with(RuntimeFeature::HipPagedKvWrite);
            }
            if policy.contains(RuntimeFeature::HipFusedQkNormRopePagedKvWrite) {
                let q_projected = zeros(context, stream, 2 * 16 * 256)?;
                let k_projected = zeros(context, stream, 4 * 256)?;
                let v_projected = zeros(context, stream, 4 * 256)?;
                let q_weight = zeros(context, stream, 256)?;
                let k_weight = zeros(context, stream, 256)?;
                let mut q_gate = zeros(context, stream, 16 * 256)?;
                let mut q_rope = zeros(context, stream, 16 * 256)?;
                ullm_runtime_sys::qwen35_qk_norm_rope_paged_kv_write_f32(
                    &q_projected,
                    &k_projected,
                    &v_projected,
                    &q_weight,
                    &k_weight,
                    &table,
                    16,
                    4,
                    256,
                    256,
                    64,
                    0,
                    10_000_000.0,
                    1e-5,
                    0,
                    256,
                    16,
                    &mut q_gate,
                    &mut q_rope,
                    &mut k_cache,
                    &mut v_cache,
                    Some(stream),
                )?;
                probe_fault_checkpoint(7, "fused-writer")?;
                proven = proven.with(RuntimeFeature::HipFusedQkNormRopePagedKvWrite);
            }
        }
        if policy.contains(RuntimeFeature::HipAq4MatvecBatch) {
            let binding = aq4_probe_binding(128);
            let mut index = context.alloc_buffer(3)?;
            index.copy_from_host(0, &[0x21_u8, 0x03, 0x54], Some(stream))?;
            let mut scale = context.alloc_buffer(3)?;
            scale.copy_from_host(0, &[0_u8, 1, 0], Some(stream))?;
            let mut codebook = context.alloc_buffer(16 * 4)?;
            let codebook_bytes = (0..16_u32)
                .flat_map(|value| (value as f32).to_le_bytes())
                .collect::<Vec<_>>();
            codebook.copy_from_host(0, &codebook_bytes, Some(stream))?;
            let mut scale_values = context.alloc_buffer(2 * 4)?;
            let scale_value_bytes = [0.5_f32, 2.0_f32]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>();
            scale_values.copy_from_host(0, &scale_value_bytes, Some(stream))?;
            let input = zeros(context, stream, binding.batch_count * binding.cols)?;
            let mut output = zeros(context, stream, binding.batch_count * binding.rows)?;
            ullm_runtime_sys::aq4_matvec_batch_f32(
                &index,
                &scale,
                &codebook,
                &scale_values,
                &input,
                None,
                binding.scale_count,
                binding.group_size,
                f32::from_bits(binding.tensor_scale_bits),
                binding.row_scale_count,
                binding.rows,
                binding.cols,
                binding.batch_count,
                &mut output,
                Some(stream),
            )?;
            probe_fault_checkpoint(9, "aq4-matvec-batch")?;
            proven = proven.with(RuntimeFeature::HipAq4MatvecBatch);
        }
        if policy.contains(RuntimeFeature::HipLinearAttentionQkvPrepareBatch) {
            const SEQUENCE_LEN: usize = 128;
            let qkv = zeros(context, stream, SEQUENCE_LEN * 8_192)?;
            let conv_weight = zeros(context, stream, 8_192 * 4)?;
            let mut history = zeros(context, stream, 8_192 * 4)?;
            let mut conv_output = zeros(context, stream, SEQUENCE_LEN * 8_192)?;
            let mut q = zeros(context, stream, SEQUENCE_LEN * 16 * 128)?;
            let mut k = zeros(context, stream, SEQUENCE_LEN * 16 * 128)?;
            let mut v = zeros(context, stream, SEQUENCE_LEN * 32 * 128)?;
            ullm_runtime_sys::linear_attn_qkv_prepare_batch_f32(
                &qkv,
                &conv_weight,
                &mut history,
                16,
                32,
                128,
                128,
                4,
                SEQUENCE_LEN,
                1.0 / 128.0_f32.sqrt(),
                true,
                &mut conv_output,
                &mut q,
                &mut k,
                &mut v,
                Some(stream),
            )?;
            probe_fault_checkpoint(10, "qkv-prepare-batch")?;
            proven = proven.with(RuntimeFeature::HipLinearAttentionQkvPrepareBatch);
        }
        probe_fault_checkpoint(8, "synchronize")?;
        stream.synchronize().map_err(|error| {
            format!("failed to synchronize backend M1 capability probes: {error}")
        })?;
        M1_PROBE_CACHE
            .get_or_init(Default::default)
            .lock()
            .map_err(|_| "backend capability probe cache is poisoned".to_string())?
            .insert(key, proven);
        capabilities.runtime_features = proven;
        Ok(capabilities)
    }

    pub fn require_features(&self, required: RuntimeFeatureSet) -> Result<(), String> {
        if !self.runtime_features.contains_all(required) {
            return Err("required backend operation runtime feature/guard is unavailable".into());
        }
        Ok(())
    }
}

pub fn normalized_device_architecture(
    info: &ullm_runtime_sys::DeviceInfo,
) -> Result<Option<String>, String> {
    if info.backend == "cpu" {
        return Ok(None);
    }
    let reported = info.gcn_arch_name.trim().to_ascii_lowercase();
    if !reported.is_empty() {
        let base = reported.split(':').next().unwrap_or(&reported);
        if base.starts_with("gfx")
            && base.len() >= 6
            && base[3..].bytes().all(|byte| byte.is_ascii_alphanumeric())
        {
            return Ok(Some(base.to_string()));
        }
        return Err(format!(
            "runtime reported invalid GPU architecture {reported}"
        ));
    }
    Err(format!(
        "HIP device {} does not expose gcnArchName through device properties",
        info.name
    ))
}

pub fn require_device_architecture(
    info: &ullm_runtime_sys::DeviceInfo,
    expected: &str,
) -> Result<(), String> {
    let actual = normalized_device_architecture(info)?;
    if actual.as_deref() != Some(expected) {
        return Err(format!(
            "runtime architecture mismatch: expected={expected} actual={}",
            actual.as_deref().unwrap_or("unavailable")
        ));
    }
    Ok(())
}

/// A fixed-size feature set. Unknown feature strings cannot enter the resolver.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RuntimeFeatureSet(u64);

impl RuntimeFeatureSet {
    pub const EMPTY: Self = Self(0);

    pub const fn from_feature(feature: RuntimeFeature) -> Self {
        Self(1_u64 << feature as u8)
    }

    pub const fn with(self, feature: RuntimeFeature) -> Self {
        Self(self.0 | (1_u64 << feature as u8))
    }

    pub const fn contains(self, feature: RuntimeFeature) -> bool {
        self.0 & (1_u64 << feature as u8) != 0
    }

    pub const fn contains_all(self, required: Self) -> bool {
        self.0 & required.0 == required.0
    }

    pub const fn len(self) -> u32 {
        self.0.count_ones()
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Stable policy/cache identity; individual bits remain assigned by [`RuntimeFeature`].
    pub const fn bits(self) -> u64 {
        self.0
    }
}

/// Typed persistent-state layout at the operation boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationStateLayout {
    None,
    ConvolutionHistory,
    RecurrentMatrix,
    PagedKvBlocks,
}

/// Request-owned state resources visible to the transaction coordinator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum StateResource {
    ConvolutionHistory = 0,
    RecurrentState = 1,
    PagedKvCache = 2,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StateResourceSet(u8);

impl StateResourceSet {
    pub const EMPTY: Self = Self(0);

    pub const fn from_resource(resource: StateResource) -> Self {
        Self(1_u8 << resource as u8)
    }

    pub const fn with(self, resource: StateResource) -> Self {
        Self(self.0 | (1_u8 << resource as u8))
    }

    pub const fn contains(self, resource: StateResource) -> bool {
        self.0 & (1_u8 << resource as u8) != 0
    }

    pub const fn contains_all(self, required: Self) -> bool {
        self.0 & required.0 == required.0
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// State visibility contract declared by a runnable implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StateEffect {
    pub reads: StateResourceSet,
    pub writes: StateResourceSet,
    pub prepares: StateResourceSet,
    pub commits: StateResourceSet,
    pub update_mode: StateUpdateMode,
    pub externally_visible_before_commit: bool,
}

/// Whether the current implementation provides a pending-state commit boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateUpdateMode {
    /// The implementation only reads request state.
    ReadOnly,
    /// The implementation mutates request state during the ABI call and must fail closed.
    InPlace,
    /// A future implementation prepares private state that the transaction owner commits later.
    PreparedCommit,
}

/// Checked workspace result used by admission before allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OperationWorkspace {
    pub persistent_bytes: u64,
    pub temporary_bytes: u64,
}

/// A bounded formula. Coefficients are bytes and are checked on every estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkspaceFormula {
    pub fixed_persistent_bytes: u64,
    pub fixed_temporary_bytes: u64,
    pub temporary_bytes_per_batch_item: u64,
    pub temporary_bytes_per_chunk_token: u64,
    pub maximum_total_bytes: u64,
}

impl WorkspaceFormula {
    pub const ZERO: Self = Self {
        fixed_persistent_bytes: 0,
        fixed_temporary_bytes: 0,
        temporary_bytes_per_batch_item: 0,
        temporary_bytes_per_chunk_token: 0,
        maximum_total_bytes: 0,
    };

    pub fn estimate(
        self,
        batch_width: u64,
        chunk_width: u64,
    ) -> Result<OperationWorkspace, String> {
        let temporary_bytes = self
            .temporary_bytes_per_batch_item
            .checked_mul(batch_width)
            .and_then(|value| {
                self.temporary_bytes_per_chunk_token
                    .checked_mul(chunk_width)
                    .and_then(|chunk| value.checked_add(chunk))
            })
            .and_then(|value| value.checked_add(self.fixed_temporary_bytes))
            .ok_or_else(|| "backend operation workspace estimate overflows u64".to_string())?;
        let total = self
            .fixed_persistent_bytes
            .checked_add(temporary_bytes)
            .ok_or_else(|| "backend operation workspace total overflows u64".to_string())?;
        if total > self.maximum_total_bytes {
            return Err(format!(
                "backend operation workspace {total} exceeds descriptor maximum {}",
                self.maximum_total_bytes
            ));
        }
        Ok(OperationWorkspace {
            persistent_bytes: self.fixed_persistent_bytes,
            temporary_bytes,
        })
    }
}

/// Production eligibility is separate from device capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PromotionStatus {
    Diagnostic,
    Reference,
    Production,
}

/// Typed runnable entry. The variant is checked again at the ABI call boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutableOperation {
    HipPagedKvWriteF32,
    HipFusedQkNormRopePagedKvWriteF32,
    HipLinearAttentionQkvPrepareF32,
    HipLinearAttentionQkvPrepareBatchF32,
    HipLinearAttentionRecurrentF32,
    HipLinearAttentionRecurrentSequenceF32,
    HipAq4MatvecBatchF32,
    HipPagedDecodeAttentionF32,
    HipPagedDecodeAttentionSigmoidGateF32,
}

/// Exact load-time request emitted from validated package geometry and served capabilities.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationRequest {
    pub kind: OperationKind,
    pub phase: ExecutionPhase,
    pub input_layout: TensorLayout,
    pub output_layout: TensorLayout,
    pub weight_layout: Option<TensorLayout>,
    pub state_layout: OperationStateLayout,
    pub activation_format: NumericalFormat,
    pub value_format: NumericalFormat,
    pub weight_format: Option<NumericalFormat>,
    pub state_format: NumericalFormat,
    pub geometry: OperationGeometry,
    pub batch_width: u64,
    pub chunk_width: u64,
    pub device: DeviceCapabilities,
    pub workspace_budget_bytes: u64,
    pub minimum_promotion: PromotionStatus,
}

/// Runnable implementation metadata owned by the immutable registry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImplementationDescriptor {
    pub id: &'static str,
    pub semantic_version: &'static str,
    pub kind: OperationKind,
    pub phases: PhaseSet,
    pub input_layout: TensorLayout,
    pub output_layout: TensorLayout,
    pub weight_layout: Option<TensorLayout>,
    pub state_layout: OperationStateLayout,
    pub activation_format: NumericalFormat,
    pub value_format: NumericalFormat,
    pub weight_format: Option<NumericalFormat>,
    pub state_format: NumericalFormat,
    pub geometry: OperationGeometry,
    pub minimum_batch_width: u64,
    pub maximum_batch_width: u64,
    pub minimum_chunk_width: u64,
    pub maximum_chunk_width: u64,
    pub backend: OperationBackend,
    pub architecture: Option<&'static str>,
    pub device_name: Option<&'static str>,
    pub minimum_abi_version: u32,
    pub required_features: RuntimeFeatureSet,
    pub workspace: WorkspaceFormula,
    pub state_effect: StateEffect,
    pub promotion: PromotionStatus,
    pub priority: i32,
    pub fallback_id: Option<&'static str>,
    pub deterministic: bool,
    pub executable: ExecutableOperation,
    pub runtime_build: &'static str,
}

/// Compact supported-phase bit set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhaseSet(u8);

impl PhaseSet {
    pub const ALL_CURRENT: Self = Self(0b111);

    pub const fn from_phase(phase: ExecutionPhase) -> Self {
        Self(1 << phase_index(phase))
    }

    pub const fn contains(self, phase: ExecutionPhase) -> bool {
        self.0 & (1 << phase_index(phase)) != 0
    }
}

const fn phase_index(phase: ExecutionPhase) -> u8 {
    match phase {
        ExecutionPhase::ColdPrefill => 0,
        ExecutionPhase::CachedPrefixPrefill => 1,
        ExecutionPhase::Decode => 2,
    }
}

/// Immutable catalog. Construction validates IDs, fallback contracts, and hard memory bounds.
#[derive(Debug, Clone)]
pub struct BackendOperationRegistry {
    implementations: Vec<ImplementationDescriptor>,
}

impl BackendOperationRegistry {
    pub fn try_from_iter(
        implementations: impl IntoIterator<Item = ImplementationDescriptor>,
    ) -> Result<Self, String> {
        let iterator = implementations.into_iter();
        if iterator
            .size_hint()
            .1
            .is_some_and(|upper| upper > MAX_BACKEND_OPERATION_IMPLEMENTATIONS)
        {
            return Err(format!(
                "backend operation registry exceeds maximum {MAX_BACKEND_OPERATION_IMPLEMENTATIONS}"
            ));
        }
        let mut bounded = Vec::new();
        bounded
            .try_reserve(
                iterator
                    .size_hint()
                    .0
                    .min(MAX_BACKEND_OPERATION_IMPLEMENTATIONS),
            )
            .map_err(|_| "backend operation registry allocation failed".to_string())?;
        for descriptor in iterator {
            if bounded.len() == MAX_BACKEND_OPERATION_IMPLEMENTATIONS {
                return Err(format!(
                    "backend operation registry exceeds maximum {MAX_BACKEND_OPERATION_IMPLEMENTATIONS}"
                ));
            }
            bounded.push(descriptor);
        }
        Self::new(bounded)
    }

    pub fn new(implementations: Vec<ImplementationDescriptor>) -> Result<Self, String> {
        if implementations.is_empty() {
            return Err("backend operation registry must not be empty".to_string());
        }
        if implementations.len() > MAX_BACKEND_OPERATION_IMPLEMENTATIONS {
            return Err(format!(
                "backend operation registry has {} entries; maximum is {MAX_BACKEND_OPERATION_IMPLEMENTATIONS}",
                implementations.len()
            ));
        }
        for (index, descriptor) in implementations.iter().enumerate() {
            validate_descriptor(descriptor)?;
            if implementations[..index]
                .iter()
                .any(|existing| existing.id == descriptor.id)
            {
                return Err(format!(
                    "duplicate backend operation implementation ID {}",
                    descriptor.id
                ));
            }
        }
        for descriptor in &implementations {
            if let Some(fallback_id) = descriptor.fallback_id {
                let fallback = implementations
                    .iter()
                    .find(|candidate| candidate.id == fallback_id)
                    .ok_or_else(|| {
                        format!(
                            "backend operation implementation {} names missing fallback {fallback_id}",
                            descriptor.id
                        )
                    })?;
                validate_fallback_compatibility(descriptor, fallback)?;
            }
        }
        for descriptor in &implementations {
            let mut cursor = descriptor;
            for _ in 0..=implementations.len() {
                let Some(next_id) = cursor.fallback_id else {
                    break;
                };
                if next_id == descriptor.id {
                    return Err(format!(
                        "backend operation fallback cycle includes {}",
                        descriptor.id
                    ));
                }
                cursor = implementations
                    .iter()
                    .find(|candidate| candidate.id == next_id)
                    .expect("fallback existence validated above");
            }
        }
        Ok(Self { implementations })
    }

    pub fn implementations(&self) -> &[ImplementationDescriptor] {
        &self.implementations
    }

    /// Resolves exactly once. A tied highest rank is an error, never declaration-order selection.
    fn resolve(&self, request: &OperationRequest) -> Result<ResolvedOperationPlan, String> {
        validate_request(request)?;
        let mut best: Option<(&ImplementationDescriptor, OperationWorkspace)> = None;
        let mut tied_id: Option<&str> = None;
        for descriptor in &self.implementations {
            if !descriptor_matches(descriptor, request) {
                continue;
            }
            let workspace = match descriptor
                .workspace
                .estimate(request.batch_width, request.chunk_width)
            {
                Ok(value)
                    if value
                        .persistent_bytes
                        .checked_add(value.temporary_bytes)
                        .is_some_and(|total| total <= request.workspace_budget_bytes) =>
                {
                    value
                }
                _ => continue,
            };
            match best {
                None => best = Some((descriptor, workspace)),
                Some((current, _)) if descriptor_rank(descriptor) > descriptor_rank(current) => {
                    best = Some((descriptor, workspace));
                    tied_id = None;
                }
                Some((current, _)) if descriptor_rank(descriptor) == descriptor_rank(current) => {
                    tied_id = Some(descriptor.id);
                }
                _ => {}
            }
        }
        if let (Some((selected, _)), Some(other)) = (best, tied_id) {
            return Err(format!(
                "ambiguous backend operation selection between {} and {other}",
                selected.id
            ));
        }
        let (descriptor, workspace) = best.ok_or_else(|| {
            format!(
                "unsupported backend operation {:?} for phase {:?}",
                request.kind, request.phase
            )
        })?;
        Ok(resolved_plan(
            descriptor,
            request,
            workspace,
            ResolutionKind::Primary,
        ))
    }

    pub fn admit(&self, request: OperationRequest) -> Result<PrestartAttempt<'_>, String> {
        let plan = self.resolve(&request)?;
        Ok(PrestartAttempt {
            registry: self,
            request,
            plan,
            visited: Vec::new(),
        })
    }
}

/// A load-time selection attempt. Fallback and start both consume the attempt.
#[derive(Debug)]
pub struct PrestartAttempt<'a> {
    registry: &'a BackendOperationRegistry,
    request: OperationRequest,
    plan: ResolvedOperationPlan,
    visited: Vec<&'static str>,
}

impl PrestartAttempt<'_> {
    pub fn fallback(mut self) -> Result<Self, String> {
        let unavailable_primary = self.plan.implementation_id;
        let fallback_id = self.plan.fallback_id.ok_or_else(|| {
            format!("backend operation {unavailable_primary} has no declared pre-start fallback")
        })?;
        if self.visited.len() >= MAX_BACKEND_OPERATION_IMPLEMENTATIONS
            || self.visited.contains(&fallback_id)
        {
            return Err("backend operation fallback chain is cyclic or too long".into());
        }
        self.visited.push(unavailable_primary);
        let fallback = self
            .registry
            .implementations
            .iter()
            .find(|candidate| candidate.id == fallback_id)
            .expect("registry construction validated fallback existence");
        if !descriptor_matches(fallback, &self.request) {
            return Err(format!(
                "declared backend operation fallback {fallback_id} is unavailable for the request"
            ));
        }
        let workspace = fallback
            .workspace
            .estimate(self.request.batch_width, self.request.chunk_width)?;
        let total = workspace
            .persistent_bytes
            .checked_add(workspace.temporary_bytes)
            .ok_or_else(|| "backend operation fallback workspace overflows u64".to_string())?;
        if total > self.request.workspace_budget_bytes {
            return Err(format!(
                "backend operation fallback {fallback_id} exceeds the admitted workspace budget"
            ));
        }
        self.plan = resolved_plan(
            fallback,
            &self.request,
            workspace,
            ResolutionKind::Fallback {
                unavailable_primary,
            },
        );
        Ok(self)
    }

    pub fn start(self) -> StartedOperationPlan<'static> {
        StartedOperationPlan {
            plan: StartedPlanStorage::Owned(self.plan),
        }
    }

    fn into_plan(self) -> ResolvedOperationPlan {
        self.plan
    }
}

fn resolved_plan(
    descriptor: &ImplementationDescriptor,
    request: &OperationRequest,
    workspace: OperationWorkspace,
    resolution: ResolutionKind,
) -> ResolvedOperationPlan {
    ResolvedOperationPlan {
        implementation_id: descriptor.id,
        semantic_version: descriptor.semantic_version,
        runtime_build: descriptor.runtime_build,
        kind: descriptor.kind,
        phase: request.phase,
        input_layout: request.input_layout.clone(),
        output_layout: request.output_layout.clone(),
        weight_layout: request.weight_layout.clone(),
        state_layout: request.state_layout,
        activation_format: request.activation_format.clone(),
        value_format: request.value_format.clone(),
        weight_format: request.weight_format.clone(),
        state_format: request.state_format.clone(),
        geometry: request.geometry,
        batch_width: request.batch_width,
        chunk_width: request.chunk_width,
        workspace,
        workspace_budget_bytes: request.workspace_budget_bytes,
        device: request.device.clone(),
        state_effect: descriptor.state_effect,
        promotion: descriptor.promotion,
        required_features: descriptor.required_features,
        deterministic: descriptor.deterministic,
        executable: descriptor.executable,
        fallback_id: descriptor.fallback_id,
        resolution,
    }
}

fn descriptor_rank(descriptor: &ImplementationDescriptor) -> (u32, i32) {
    let specificity = u32::from(descriptor.architecture.is_some())
        .saturating_add(u32::from(descriptor.device_name.is_some()))
        .saturating_add(descriptor.required_features.len());
    (specificity, descriptor.priority)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolutionKind {
    Primary,
    Fallback { unavailable_primary: &'static str },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationExecutionStatus {
    Started,
    Succeeded,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OperationExecutionRecord {
    pub implementation_id: &'static str,
    pub phase: ExecutionPhase,
    pub status: OperationExecutionStatus,
}

/// Fixed-cardinality request-terminal summary of registry-routed layer executions.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct OperationExecutionAudit {
    pub schema_version: &'static str,
    pub outcome: &'static str,
    pub expected_layers_per_step: usize,
    pub expected_records_per_layer: usize,
    pub cold_prefill_steps: u64,
    pub cached_prefix_prefill_steps: u64,
    pub decode_steps: u64,
    pub total_steps: u64,
    pub total_records: u64,
    /// Actual registry-routed operation invocations, including successful partial work.
    pub physical_operation_invocations: u64,
    /// M1-equivalent operation coverage. One fully covered Qwen3.5 token contributes 64.
    pub token_equivalent_operation_coverage: u64,
    pub prefill_chunks_executed: u64,
    pub prefill_tokens_executed: u64,
    pub prefill_tokens_committed: u64,
    /// Index is the execution width. Index zero is reserved and must remain zero.
    pub prefill_width_histogram: Vec<u64>,
    pub implementation_counts: [OperationExecutionCount; 8],
    #[serde(serialize_with = "serialize_sha256_hex")]
    pub deterministic_digest_sha256: [u8; 32],
    pub coverage_complete: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failed_phase: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failed_layer: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failed_operation: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct OperationExecutionCount {
    pub kind: &'static str,
    pub implementation_id: &'static str,
    pub count: u64,
}

fn serialize_sha256_hex<S>(digest: &[u8; 32], serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    use std::fmt::Write as _;
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        write!(&mut encoded, "{byte:02x}").map_err(serde::ser::Error::custom)?;
    }
    serializer.serialize_str(&encoded)
}

/// Stable trace view of a load-time resolution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationResolutionTrace {
    pub implementation_id: &'static str,
    pub semantic_version: &'static str,
    pub runtime_build: &'static str,
    pub kind: OperationKind,
    pub phase: ExecutionPhase,
    pub input_layout: TensorLayout,
    pub output_layout: TensorLayout,
    pub weight_layout: Option<TensorLayout>,
    pub state_layout: OperationStateLayout,
    pub activation_format: NumericalFormat,
    pub value_format: NumericalFormat,
    pub weight_format: Option<NumericalFormat>,
    pub state_format: NumericalFormat,
    pub geometry: OperationGeometry,
    pub batch_width: u64,
    pub chunk_width: u64,
    pub workspace: OperationWorkspace,
    pub workspace_budget_bytes: u64,
    pub device: DeviceCapabilities,
    pub state_effect: StateEffect,
    pub promotion: PromotionStatus,
    pub required_features: RuntimeFeatureSet,
    pub deterministic: bool,
    pub executable: ExecutableOperation,
    pub resolution: ResolutionKind,
}

impl OperationResolutionTrace {
    /// Bounded load-time audit snapshot. Device strings were bounded during request validation.
    pub fn audit_json(&self) -> String {
        serde_json::json!({
            "implementation_id": self.implementation_id,
            "semantic_version": self.semantic_version,
            "runtime_build": self.runtime_build,
            "kind": format!("{:?}", self.kind),
            "phase": format!("{:?}", self.phase),
            "backend": format!("{:?}", self.device.backend),
            "device_id": self.device.device_id,
            "device_name": self.device.device_name,
            "architecture": self.device.architecture,
            "abi_version": self.device.abi_version,
            "runtime_feature_count": self.device.runtime_features.len(),
            "input_layout": format!("{:?}", self.input_layout),
            "output_layout": format!("{:?}", self.output_layout),
            "weight_layout": format!("{:?}", self.weight_layout),
            "state_layout": format!("{:?}", self.state_layout),
            "activation_format": self.activation_format.as_str(),
            "value_format": self.value_format.as_str(),
            "weight_format": self.weight_format.as_ref().map(NumericalFormat::as_str),
            "state_format": self.state_format.as_str(),
            "geometry": format!("{:?}", self.geometry),
            "batch_width": self.batch_width,
            "chunk_width": self.chunk_width,
            "persistent_bytes": self.workspace.persistent_bytes,
            "temporary_bytes": self.workspace.temporary_bytes,
            "workspace_budget_bytes": self.workspace_budget_bytes,
            "state_effect": format!("{:?}", self.state_effect),
            "promotion": format!("{:?}", self.promotion),
            "executable": format!("{:?}", self.executable),
            "resolution": format!("{:?}", self.resolution),
        })
        .to_string()
    }
}

/// Immutable plan kept by the resident layer. Starting execution cannot trigger re-selection.
#[derive(Debug, PartialEq, Eq)]
pub struct ResolvedOperationPlan {
    implementation_id: &'static str,
    semantic_version: &'static str,
    runtime_build: &'static str,
    kind: OperationKind,
    phase: ExecutionPhase,
    input_layout: TensorLayout,
    output_layout: TensorLayout,
    weight_layout: Option<TensorLayout>,
    state_layout: OperationStateLayout,
    activation_format: NumericalFormat,
    value_format: NumericalFormat,
    weight_format: Option<NumericalFormat>,
    state_format: NumericalFormat,
    geometry: OperationGeometry,
    batch_width: u64,
    chunk_width: u64,
    workspace: OperationWorkspace,
    workspace_budget_bytes: u64,
    device: DeviceCapabilities,
    state_effect: StateEffect,
    promotion: PromotionStatus,
    required_features: RuntimeFeatureSet,
    deterministic: bool,
    executable: ExecutableOperation,
    fallback_id: Option<&'static str>,
    resolution: ResolutionKind,
}

impl ResolvedOperationPlan {
    pub const fn execution_record(
        &self,
        status: OperationExecutionStatus,
    ) -> OperationExecutionRecord {
        OperationExecutionRecord {
            implementation_id: self.implementation_id,
            phase: self.phase,
            status,
        }
    }
    pub fn trace(&self) -> OperationResolutionTrace {
        OperationResolutionTrace {
            implementation_id: self.implementation_id,
            semantic_version: self.semantic_version,
            runtime_build: self.runtime_build,
            kind: self.kind,
            phase: self.phase,
            input_layout: self.input_layout.clone(),
            output_layout: self.output_layout.clone(),
            weight_layout: self.weight_layout.clone(),
            state_layout: self.state_layout,
            activation_format: self.activation_format.clone(),
            value_format: self.value_format.clone(),
            weight_format: self.weight_format.clone(),
            state_format: self.state_format.clone(),
            geometry: self.geometry,
            batch_width: self.batch_width,
            chunk_width: self.chunk_width,
            workspace: self.workspace,
            workspace_budget_bytes: self.workspace_budget_bytes,
            device: self.device.clone(),
            state_effect: self.state_effect,
            promotion: self.promotion,
            required_features: self.required_features,
            deterministic: self.deterministic,
            executable: self.executable,
            resolution: self.resolution,
        }
    }

    pub const fn attempt(&self) -> OperationExecutionAttempt<'_> {
        OperationExecutionAttempt { plan: self }
    }
}

/// Started plans expose only their resolved executable; there is no post-start fallback API.
#[derive(Debug)]
pub struct OperationExecutionAttempt<'a> {
    plan: &'a ResolvedOperationPlan,
}

impl<'a> OperationExecutionAttempt<'a> {
    pub const fn start(self) -> StartedOperationPlan<'a> {
        StartedOperationPlan {
            plan: StartedPlanStorage::Borrowed(self.plan),
        }
    }
}

#[derive(Debug)]
enum StartedPlanStorage<'a> {
    Borrowed(&'a ResolvedOperationPlan),
    Owned(ResolvedOperationPlan),
}

#[derive(Debug)]
pub struct StartedOperationPlan<'a> {
    plan: StartedPlanStorage<'a>,
}

impl StartedOperationPlan<'_> {
    fn plan(&self) -> &ResolvedOperationPlan {
        match &self.plan {
            StartedPlanStorage::Borrowed(plan) => plan,
            StartedPlanStorage::Owned(plan) => plan,
        }
    }

    pub fn trace(&self) -> OperationResolutionTrace {
        self.plan().trace()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute_linear_attention_qkv_prepare_f32(
        self,
        qkv: &ullm_runtime_sys::RuntimeBuffer,
        convolution_weight: &ullm_runtime_sys::RuntimeBuffer,
        convolution_history: &mut ullm_runtime_sys::RuntimeBuffer,
        qkv_convolution_output: &mut ullm_runtime_sys::RuntimeBuffer,
        q: &mut ullm_runtime_sys::RuntimeBuffer,
        k: &mut ullm_runtime_sys::RuntimeBuffer,
        v: &mut ullm_runtime_sys::RuntimeBuffer,
        stream: &mut ullm_runtime_sys::RuntimeStream,
    ) -> Result<(), String> {
        let plan = self.plan();
        let OperationGeometry::LinearAttentionQkvPrepare {
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            kernel_size,
            query_scale,
            qk_l2_norm,
        } = plan.geometry
        else {
            return Err("resolved QKV prepare operation has incompatible geometry".into());
        };
        if plan.kind != OperationKind::LinearAttentionQkvPrepare
            || plan.executable != ExecutableOperation::HipLinearAttentionQkvPrepareF32
            || plan.batch_width != 1
            || plan.chunk_width != 1
        {
            return Err("resolved QKV prepare operation has incompatible binding".into());
        }
        let q_scale = match query_scale {
            QueryScale::InverseSqrtKeyDim => 1.0_f32 / (key_dim as f32).sqrt(),
            QueryScale::ExactF32Bits(bits) => f32::from_bits(bits),
        };
        if !q_scale.is_finite() || q_scale <= 0.0 {
            return Err("resolved QKV prepare query scale is invalid".into());
        }
        #[cfg(test)]
        PREPARE_SYS_CALL_COUNT.with(|count| count.set(count.get() + 1));
        ullm_runtime_sys::linear_attn_qkv_prepare_f32(
            qkv,
            convolution_weight,
            convolution_history,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            kernel_size,
            q_scale,
            qk_l2_norm,
            qkv_convolution_output,
            q,
            k,
            v,
            Some(stream),
        )
        .map_err(|error| error.to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute_linear_attention_qkv_prepare_batch_f32(
        self,
        qkv: &ullm_runtime_sys::RuntimeBuffer,
        convolution_weight: &ullm_runtime_sys::RuntimeBuffer,
        convolution_history: &mut ullm_runtime_sys::RuntimeBuffer,
        qkv_convolution_output: &mut ullm_runtime_sys::RuntimeBuffer,
        q: &mut ullm_runtime_sys::RuntimeBuffer,
        k: &mut ullm_runtime_sys::RuntimeBuffer,
        v: &mut ullm_runtime_sys::RuntimeBuffer,
        stream: &mut ullm_runtime_sys::RuntimeStream,
    ) -> Result<(), String> {
        let plan = self.plan();
        let OperationGeometry::LinearAttentionQkvPrepare {
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            kernel_size,
            query_scale,
            qk_l2_norm,
        } = plan.geometry
        else {
            return Err("resolved batch QKV prepare operation has incompatible geometry".into());
        };
        if plan.kind != OperationKind::LinearAttentionQkvPrepareBatch
            || plan.executable != ExecutableOperation::HipLinearAttentionQkvPrepareBatchF32
            || plan.batch_width != 1
            || !(2..=128).contains(&plan.chunk_width)
        {
            return Err("resolved batch QKV prepare operation has incompatible binding".into());
        }
        let sequence_len = usize::try_from(plan.chunk_width)
            .map_err(|_| "resolved batch QKV prepare width exceeds usize".to_string())?;
        let q_scale = match query_scale {
            QueryScale::InverseSqrtKeyDim => 1.0_f32 / (key_dim as f32).sqrt(),
            QueryScale::ExactF32Bits(bits) => f32::from_bits(bits),
        };
        if !q_scale.is_finite() || q_scale <= 0.0 {
            return Err("resolved batch QKV prepare query scale is invalid".into());
        }
        #[cfg(test)]
        PREPARE_SYS_CALL_COUNT.with(|count| count.set(count.get() + 1));
        ullm_runtime_sys::linear_attn_qkv_prepare_batch_f32(
            qkv,
            convolution_weight,
            convolution_history,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            kernel_size,
            sequence_len,
            q_scale,
            qk_l2_norm,
            qkv_convolution_output,
            q,
            k,
            v,
            Some(stream),
        )
        .map_err(|error| error.to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute_linear_attention_recurrent_f32(
        self,
        q: &ullm_runtime_sys::RuntimeBuffer,
        k: &ullm_runtime_sys::RuntimeBuffer,
        v: &ullm_runtime_sys::RuntimeBuffer,
        gate: &ullm_runtime_sys::RuntimeBuffer,
        beta: &ullm_runtime_sys::RuntimeBuffer,
        state: &mut ullm_runtime_sys::RuntimeBuffer,
        output: &mut ullm_runtime_sys::RuntimeBuffer,
        stream: &mut ullm_runtime_sys::RuntimeStream,
    ) -> Result<(), String> {
        let plan = self.plan();
        let OperationGeometry::GatedDeltaRule {
            key_heads,
            value_heads,
            key_dim,
            value_dim,
        } = plan.geometry
        else {
            return Err("resolved recurrent operation has incompatible geometry".into());
        };
        if plan.kind != OperationKind::GatedDeltaRuleScan
            || plan.executable != ExecutableOperation::HipLinearAttentionRecurrentF32
            || plan.batch_width != 1
            || plan.chunk_width != 1
        {
            return Err(format!(
                "resolved backend operation {} is not linear attention recurrent",
                plan.implementation_id
            ));
        }
        #[cfg(test)]
        RECURRENT_SYS_CALL_COUNT.with(|count| count.set(count.get() + 1));
        ullm_runtime_sys::linear_attn_recurrent_f32(
            q,
            k,
            v,
            gate,
            beta,
            key_heads,
            value_heads,
            1,
            key_dim,
            value_dim,
            state,
            output,
            Some(stream),
        )
        .map_err(|error| error.to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute_linear_attention_recurrent_sequence_f32(
        self,
        q: &ullm_runtime_sys::RuntimeBuffer,
        k: &ullm_runtime_sys::RuntimeBuffer,
        v: &ullm_runtime_sys::RuntimeBuffer,
        gate: &ullm_runtime_sys::RuntimeBuffer,
        beta: &ullm_runtime_sys::RuntimeBuffer,
        state: &mut ullm_runtime_sys::RuntimeBuffer,
        output: &mut ullm_runtime_sys::RuntimeBuffer,
        stream: &mut ullm_runtime_sys::RuntimeStream,
    ) -> Result<(), String> {
        let plan = self.plan();
        let OperationGeometry::GatedDeltaRule {
            key_heads,
            value_heads,
            key_dim,
            value_dim,
        } = plan.geometry
        else {
            return Err("resolved recurrent sequence operation has incompatible geometry".into());
        };
        if plan.kind != OperationKind::GatedDeltaRuleSequence
            || plan.executable != ExecutableOperation::HipLinearAttentionRecurrentSequenceF32
            || plan.batch_width != 1
            || !(2..=128).contains(&plan.chunk_width)
        {
            return Err(format!(
                "resolved backend operation {} is not a linear attention recurrent sequence",
                plan.implementation_id
            ));
        }
        let sequence_len = usize::try_from(plan.chunk_width)
            .map_err(|_| "resolved recurrent sequence width exceeds usize".to_string())?;
        #[cfg(test)]
        RECURRENT_SYS_CALL_COUNT.with(|count| count.set(count.get() + 1));
        ullm_runtime_sys::linear_attn_recurrent_f32(
            q,
            k,
            v,
            gate,
            beta,
            key_heads,
            value_heads,
            sequence_len,
            key_dim,
            value_dim,
            state,
            output,
            Some(stream),
        )
        .map_err(|error| error.to_string())
    }

    pub fn execute_aq4_matvec_batch_f32(
        self,
        index: &ullm_runtime_sys::RuntimeBuffer,
        scale: &ullm_runtime_sys::RuntimeBuffer,
        codebook: &ullm_runtime_sys::RuntimeBuffer,
        scale_values: &ullm_runtime_sys::RuntimeBuffer,
        input: &ullm_runtime_sys::RuntimeBuffer,
        row_scale: Option<&ullm_runtime_sys::RuntimeBuffer>,
        output: &mut ullm_runtime_sys::RuntimeBuffer,
        stream: &mut ullm_runtime_sys::RuntimeStream,
    ) -> Result<(), String> {
        let plan = self.plan();
        let OperationGeometry::Aq4MatvecBatch {
            rows,
            cols,
            group_size,
            scale_count,
            row_scale_count,
            tensor_scale_bits,
        } = plan.geometry
        else {
            return Err("resolved AQ4 matvec batch operation has incompatible geometry".into());
        };
        if plan.kind != OperationKind::Aq4MatvecBatch
            || plan.executable != ExecutableOperation::HipAq4MatvecBatchF32
            || plan.chunk_width != 1
            || !(2..=128).contains(&plan.batch_width)
        {
            return Err(format!(
                "resolved backend operation {} is not an AQ4 matvec batch",
                plan.implementation_id
            ));
        }
        if rows == 0 || cols == 0 || group_size == 0 || scale_count == 0 {
            return Err("resolved AQ4 matvec batch geometry contains zero dimensions".into());
        }
        let tensor_scale = f32::from_bits(tensor_scale_bits);
        if !tensor_scale.is_finite() || tensor_scale <= 0.0 {
            return Err("resolved AQ4 matvec batch tensor scale is invalid".into());
        }
        if row_scale_count == 0 && row_scale.is_some() {
            return Err("resolved AQ4 matvec batch has an unexpected row scale buffer".into());
        }
        if row_scale_count != 0 && row_scale.is_none() {
            return Err("resolved AQ4 matvec batch is missing its row scale buffer".into());
        }
        let batch_count = usize::try_from(plan.batch_width)
            .map_err(|_| "resolved AQ4 matvec batch width exceeds usize".to_string())?;
        ullm_runtime_sys::aq4_matvec_batch_f32(
            index,
            scale,
            codebook,
            scale_values,
            input,
            row_scale,
            scale_count,
            group_size,
            tensor_scale,
            row_scale_count,
            rows,
            cols,
            batch_count,
            output,
            Some(stream),
        )
        .map_err(|error| error.to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute_paged_kv_write_f32(
        self,
        k: &ullm_runtime_sys::RuntimeBuffer,
        v: &ullm_runtime_sys::RuntimeBuffer,
        block_table: &ullm_runtime_sys::RuntimeBuffer,
        cache_position: usize,
        k_cache: &mut ullm_runtime_sys::RuntimeBuffer,
        v_cache: &mut ullm_runtime_sys::RuntimeBuffer,
        stream: &mut ullm_runtime_sys::RuntimeStream,
    ) -> Result<(), String> {
        let plan = self.plan();
        let OperationGeometry::PagedKvWrite {
            kv_heads,
            head_dim,
            value_dim,
            block_size,
            cache_blocks,
        } = plan.geometry
        else {
            return Err("resolved paged KV writer has incompatible geometry".into());
        };
        if plan.kind != OperationKind::PagedKvWrite
            || plan.executable != ExecutableOperation::HipPagedKvWriteF32
            || cache_position >= block_size.saturating_mul(cache_blocks)
        {
            return Err("resolved paged KV writer has incompatible binding/position".into());
        }
        #[cfg(test)]
        WRITER_SYS_CALL_COUNT.with(|count| count.set(count.get() + 1));
        ullm_runtime_sys::paged_kv_write_f32(
            k,
            v,
            block_table,
            cache_position,
            block_size,
            cache_blocks,
            kv_heads,
            head_dim,
            value_dim,
            k_cache,
            v_cache,
            Some(stream),
        )
        .map_err(|error| error.to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute_fused_qk_norm_rope_paged_kv_write_f32(
        self,
        q_projected: &ullm_runtime_sys::RuntimeBuffer,
        k_projected: &ullm_runtime_sys::RuntimeBuffer,
        v_projected: &ullm_runtime_sys::RuntimeBuffer,
        q_norm_weight: &ullm_runtime_sys::RuntimeBuffer,
        k_norm_weight: &ullm_runtime_sys::RuntimeBuffer,
        block_table: &ullm_runtime_sys::RuntimeBuffer,
        actual_rotary_dim: usize,
        rope_position: usize,
        actual_rope_base: f32,
        actual_norm_epsilon: f32,
        cache_position: usize,
        q_gate: &mut ullm_runtime_sys::RuntimeBuffer,
        q_rope: &mut ullm_runtime_sys::RuntimeBuffer,
        k_cache: &mut ullm_runtime_sys::RuntimeBuffer,
        v_cache: &mut ullm_runtime_sys::RuntimeBuffer,
        stream: &mut ullm_runtime_sys::RuntimeStream,
    ) -> Result<(), String> {
        let plan = self.plan();
        let OperationGeometry::FusedQkNormRopePagedKvWrite {
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            rotary_dim,
            rope_base_bits,
            norm_epsilon_bits,
            block_size,
            cache_blocks,
        } = plan.geometry
        else {
            return Err("resolved fused QK/RoPE/KV writer has incompatible geometry".into());
        };
        if plan.kind != OperationKind::FusedQkNormRopePagedKvWrite
            || plan.executable != ExecutableOperation::HipFusedQkNormRopePagedKvWriteF32
            || cache_position >= block_size.saturating_mul(cache_blocks)
            || actual_rotary_dim != rotary_dim
            || actual_rope_base.to_bits() != rope_base_bits
            || actual_norm_epsilon.to_bits() != norm_epsilon_bits
        {
            return Err(
                "resolved fused QK/RoPE/KV writer has incompatible binding/position".into(),
            );
        }
        #[cfg(test)]
        WRITER_SYS_CALL_COUNT.with(|count| count.set(count.get() + 1));
        ullm_runtime_sys::qwen35_qk_norm_rope_paged_kv_write_f32(
            q_projected,
            k_projected,
            v_projected,
            q_norm_weight,
            k_norm_weight,
            block_table,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            actual_rotary_dim,
            rope_position,
            actual_rope_base,
            actual_norm_epsilon,
            cache_position,
            block_size,
            cache_blocks,
            q_gate,
            q_rope,
            k_cache,
            v_cache,
            Some(stream),
        )
        .map_err(|error| error.to_string())
    }

    pub fn execute_paged_decode_attention_f32(
        self,
        q: &ullm_runtime_sys::RuntimeBuffer,
        k_cache: &ullm_runtime_sys::RuntimeBuffer,
        v_cache: &ullm_runtime_sys::RuntimeBuffer,
        block_table: &ullm_runtime_sys::RuntimeBuffer,
        cache_len: usize,
        output: &mut ullm_runtime_sys::RuntimeBuffer,
        stream: &mut ullm_runtime_sys::RuntimeStream,
    ) -> Result<(), String> {
        let (q_heads, kv_heads, head_dim, value_dim, block_size, cache_blocks) =
            self.paged_geometry(false, ExecutableOperation::HipPagedDecodeAttentionF32)?;
        ullm_runtime_sys::paged_decode_attn_f32(
            q,
            k_cache,
            v_cache,
            block_table,
            cache_len,
            block_size,
            cache_blocks,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            1.0_f32 / (head_dim as f32).sqrt(),
            output,
            Some(stream),
        )
        .map_err(|error| error.to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute_paged_decode_attention_sigmoid_gate_f32(
        self,
        q: &ullm_runtime_sys::RuntimeBuffer,
        gate: &ullm_runtime_sys::RuntimeBuffer,
        k_cache: &ullm_runtime_sys::RuntimeBuffer,
        v_cache: &ullm_runtime_sys::RuntimeBuffer,
        block_table: &ullm_runtime_sys::RuntimeBuffer,
        cache_len: usize,
        output: &mut ullm_runtime_sys::RuntimeBuffer,
        stream: &mut ullm_runtime_sys::RuntimeStream,
    ) -> Result<(), String> {
        let (q_heads, kv_heads, head_dim, value_dim, block_size, cache_blocks) = self
            .paged_geometry(
                true,
                ExecutableOperation::HipPagedDecodeAttentionSigmoidGateF32,
            )?;
        ullm_runtime_sys::paged_decode_attn_sigmoid_gate_f32(
            q,
            gate,
            k_cache,
            v_cache,
            block_table,
            cache_len,
            block_size,
            cache_blocks,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            1.0_f32 / (head_dim as f32).sqrt(),
            output,
            Some(stream),
        )
        .map_err(|error| error.to_string())
    }

    fn paged_geometry(
        &self,
        expected_gate: bool,
        expected_executable: ExecutableOperation,
    ) -> Result<(usize, usize, usize, usize, usize, usize), String> {
        let plan = self.plan();
        let OperationGeometry::PagedCausalGqaRead {
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            block_size,
            cache_blocks,
            sigmoid_gate,
        } = plan.geometry
        else {
            return Err("resolved paged GQA operation has incompatible geometry".into());
        };
        if plan.kind != OperationKind::PagedCausalGqaRead
            || plan.executable != expected_executable
            || sigmoid_gate != expected_gate
            || plan.batch_width != 1
            || plan.chunk_width != 1
        {
            return Err(format!(
                "resolved backend operation {} has an incompatible paged GQA binding",
                plan.implementation_id
            ));
        }
        Ok((
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
            block_size,
            cache_blocks,
        ))
    }
}

fn validate_descriptor(descriptor: &ImplementationDescriptor) -> Result<(), String> {
    if descriptor.id.is_empty()
        || descriptor.semantic_version.is_empty()
        || descriptor.runtime_build.is_empty()
    {
        return Err(
            "backend operation descriptor IDs, versions, and builds must be nonempty".into(),
        );
    }
    for (value, label) in [
        (descriptor.id, "implementation ID"),
        (descriptor.semantic_version, "semantic version"),
        (descriptor.runtime_build, "runtime build"),
    ] {
        if value.len() > MAX_OPERATION_IDENTIFIER_BYTES {
            return Err(format!(
                "backend operation {label} exceeds {MAX_OPERATION_IDENTIFIER_BYTES} bytes"
            ));
        }
    }
    for (value, label) in [
        (descriptor.architecture, "architecture"),
        (descriptor.device_name, "device name"),
    ] {
        if value.is_some_and(|value| value.len() > MAX_OPERATION_IDENTIFIER_BYTES) {
            return Err(format!(
                "backend operation {label} exceeds {MAX_OPERATION_IDENTIFIER_BYTES} bytes"
            ));
        }
    }
    if descriptor.minimum_batch_width == 0
        || descriptor.minimum_chunk_width == 0
        || descriptor.minimum_batch_width > descriptor.maximum_batch_width
        || descriptor.minimum_chunk_width > descriptor.maximum_chunk_width
    {
        return Err(format!(
            "backend operation descriptor {} has an invalid width bound",
            descriptor.id
        ));
    }
    let semantic_binding_valid = match (descriptor.kind, descriptor.geometry, descriptor.executable)
    {
        (
            OperationKind::PagedKvWrite,
            OperationGeometry::PagedKvWrite { .. },
            ExecutableOperation::HipPagedKvWriteF32,
        ) => true,
        (
            OperationKind::FusedQkNormRopePagedKvWrite,
            OperationGeometry::FusedQkNormRopePagedKvWrite { .. },
            ExecutableOperation::HipFusedQkNormRopePagedKvWriteF32,
        ) => true,
        (
            OperationKind::LinearAttentionQkvPrepare,
            OperationGeometry::LinearAttentionQkvPrepare { .. },
            ExecutableOperation::HipLinearAttentionQkvPrepareF32,
        ) => true,
        (
            OperationKind::LinearAttentionQkvPrepareBatch,
            OperationGeometry::LinearAttentionQkvPrepare { .. },
            ExecutableOperation::HipLinearAttentionQkvPrepareBatchF32,
        ) => true,
        (
            OperationKind::GatedDeltaRuleScan,
            OperationGeometry::GatedDeltaRule { .. },
            ExecutableOperation::HipLinearAttentionRecurrentF32,
        ) => true,
        (
            OperationKind::GatedDeltaRuleSequence,
            OperationGeometry::GatedDeltaRule { .. },
            ExecutableOperation::HipLinearAttentionRecurrentSequenceF32,
        ) => true,
        (
            OperationKind::Aq4MatvecBatch,
            OperationGeometry::Aq4MatvecBatch { .. },
            ExecutableOperation::HipAq4MatvecBatchF32,
        ) => true,
        (
            OperationKind::PagedCausalGqaRead,
            OperationGeometry::PagedCausalGqaRead {
                sigmoid_gate: false,
                ..
            },
            ExecutableOperation::HipPagedDecodeAttentionF32,
        ) => true,
        (
            OperationKind::PagedCausalGqaRead,
            OperationGeometry::PagedCausalGqaRead {
                sigmoid_gate: true, ..
            },
            ExecutableOperation::HipPagedDecodeAttentionSigmoidGateF32,
        ) => true,
        _ => false,
    };
    if !semantic_binding_valid {
        return Err(format!(
            "backend operation descriptor {} has incompatible kind, geometry, and executable",
            descriptor.id
        ));
    }
    let effect = descriptor.state_effect;
    let effect_valid = match effect.update_mode {
        StateUpdateMode::ReadOnly => {
            (descriptor.kind == OperationKind::Aq4MatvecBatch || !effect.reads.is_empty())
                && effect.writes.is_empty()
                && effect.prepares.is_empty()
                && effect.commits.is_empty()
                && !effect.externally_visible_before_commit
        }
        StateUpdateMode::InPlace => {
            !effect.writes.is_empty()
                && effect.reads.contains_all(effect.writes)
                && effect.prepares.is_empty()
                && effect.commits.is_empty()
                && effect.externally_visible_before_commit
        }
        // Prepared device-state commit is not implemented by the M1 executor yet.
        StateUpdateMode::PreparedCommit => false,
    };
    let semantic_state_valid = match descriptor.kind {
        OperationKind::PagedKvWrite | OperationKind::FusedQkNormRopePagedKvWrite => {
            descriptor.state_layout == OperationStateLayout::PagedKvBlocks
                && effect.writes.contains(StateResource::PagedKvCache)
        }
        OperationKind::LinearAttentionQkvPrepare
        | OperationKind::LinearAttentionQkvPrepareBatch => {
            descriptor.state_layout == OperationStateLayout::ConvolutionHistory
                && effect.writes.contains(StateResource::ConvolutionHistory)
        }
        OperationKind::GatedDeltaRuleScan | OperationKind::GatedDeltaRuleSequence => {
            descriptor.state_layout == OperationStateLayout::RecurrentMatrix
                && effect.writes.contains(StateResource::RecurrentState)
        }
        OperationKind::Aq4MatvecBatch => {
            descriptor.state_layout == OperationStateLayout::None
                && effect.reads.is_empty()
                && effect.writes.is_empty()
        }
        OperationKind::PagedCausalGqaRead => {
            descriptor.state_layout == OperationStateLayout::PagedKvBlocks
                && effect.reads.contains(StateResource::PagedKvCache)
                && effect.writes.is_empty()
        }
    };
    if !effect_valid || !semantic_state_valid {
        return Err(format!(
            "backend operation descriptor {} has an invalid state effect",
            descriptor.id
        ));
    }
    descriptor.input_layout.validate()?;
    descriptor.output_layout.validate()?;
    if let Some(layout) = &descriptor.weight_layout {
        layout.validate()?;
    }
    descriptor.activation_format.validate()?;
    descriptor.value_format.validate()?;
    if let Some(format) = &descriptor.weight_format {
        format.validate()?;
    }
    descriptor.state_format.validate()?;
    descriptor.workspace.estimate(
        descriptor.maximum_batch_width,
        descriptor.maximum_chunk_width,
    )?;
    Ok(())
}

fn validate_request(request: &OperationRequest) -> Result<(), String> {
    if request.batch_width == 0 || request.chunk_width == 0 {
        return Err("backend operation request widths must be nonzero".into());
    }
    if request
        .device
        .architecture
        .as_ref()
        .is_some_and(|value| value.len() > MAX_OPERATION_IDENTIFIER_BYTES)
    {
        return Err(format!(
            "backend operation architecture exceeds {MAX_OPERATION_IDENTIFIER_BYTES} bytes"
        ));
    }
    if request
        .device
        .device_name
        .as_ref()
        .is_some_and(|value| value.len() > MAX_OPERATION_IDENTIFIER_BYTES)
    {
        return Err(format!(
            "backend operation device name exceeds {MAX_OPERATION_IDENTIFIER_BYTES} bytes"
        ));
    }
    request.input_layout.validate()?;
    request.output_layout.validate()?;
    if let Some(layout) = &request.weight_layout {
        layout.validate()?;
    }
    request.activation_format.validate()?;
    request.value_format.validate()?;
    if let Some(format) = &request.weight_format {
        format.validate()?;
    }
    request.state_format.validate()?;
    Ok(())
}

fn descriptor_matches(descriptor: &ImplementationDescriptor, request: &OperationRequest) -> bool {
    descriptor.kind == request.kind
        && descriptor.phases.contains(request.phase)
        && descriptor.input_layout == request.input_layout
        && descriptor.output_layout == request.output_layout
        && descriptor.weight_layout == request.weight_layout
        && descriptor.state_layout == request.state_layout
        && descriptor.activation_format == request.activation_format
        && descriptor.value_format == request.value_format
        && descriptor.weight_format == request.weight_format
        && descriptor.state_format == request.state_format
        && descriptor.geometry == request.geometry
        && request.batch_width >= descriptor.minimum_batch_width
        && request.batch_width <= descriptor.maximum_batch_width
        && request.chunk_width >= descriptor.minimum_chunk_width
        && request.chunk_width <= descriptor.maximum_chunk_width
        && descriptor.backend == request.device.backend
        && descriptor.minimum_abi_version <= request.device.abi_version
        && descriptor
            .architecture
            .is_none_or(|required| request.device.architecture.as_deref() == Some(required))
        && descriptor
            .device_name
            .is_none_or(|required| request.device.device_name.as_deref() == Some(required))
        && request
            .device
            .runtime_features
            .contains_all(descriptor.required_features)
        && descriptor.promotion >= request.minimum_promotion
}

fn validate_fallback_compatibility(
    primary: &ImplementationDescriptor,
    fallback: &ImplementationDescriptor,
) -> Result<(), String> {
    if primary.kind != fallback.kind
        || primary.phases != fallback.phases
        || primary.input_layout != fallback.input_layout
        || primary.output_layout != fallback.output_layout
        || primary.weight_layout != fallback.weight_layout
        || primary.state_layout != fallback.state_layout
        || primary.activation_format != fallback.activation_format
        || primary.value_format != fallback.value_format
        || primary.weight_format != fallback.weight_format
        || primary.state_format != fallback.state_format
        || primary.geometry != fallback.geometry
        || primary.state_effect != fallback.state_effect
        || primary.executable != fallback.executable
        || primary.minimum_batch_width != fallback.minimum_batch_width
        || primary.maximum_batch_width != fallback.maximum_batch_width
        || primary.minimum_chunk_width != fallback.minimum_chunk_width
        || primary.maximum_chunk_width != fallback.maximum_chunk_width
    {
        return Err(format!(
            "backend operation fallback {} is incompatible with primary {}",
            fallback.id, primary.id
        ));
    }
    Ok(())
}

/// The two width-one operations currently exercised by the resident M1 path.
pub fn qwen35_m1_production_registry() -> Result<BackendOperationRegistry, String> {
    let convolution_history = StateResourceSet::from_resource(StateResource::ConvolutionHistory);
    let recurrent_state = StateResourceSet::from_resource(StateResource::RecurrentState);
    let paged_kv = StateResourceSet::from_resource(StateResource::PagedKvCache);
    BackendOperationRegistry::new(vec![
        ImplementationDescriptor {
            id: "hip.linear-attention-qkv-prepare-f32.m1",
            semantic_version: "1.0.0",
            kind: OperationKind::LinearAttentionQkvPrepare,
            phases: PhaseSet::ALL_CURRENT,
            input_layout: TensorLayout::TokensHidden,
            output_layout: TensorLayout::TokensHidden,
            weight_layout: None,
            state_layout: OperationStateLayout::ConvolutionHistory,
            activation_format: NumericalFormat::F32,
            value_format: NumericalFormat::F32,
            weight_format: None,
            state_format: NumericalFormat::F32,
            geometry: OperationGeometry::LinearAttentionQkvPrepare {
                key_heads: 16,
                value_heads: 32,
                key_dim: 128,
                value_dim: 128,
                kernel_size: 4,
                query_scale: QueryScale::InverseSqrtKeyDim,
                qk_l2_norm: true,
            },
            minimum_batch_width: 1,
            maximum_batch_width: 1,
            minimum_chunk_width: 1,
            maximum_chunk_width: 1,
            backend: OperationBackend::Hip,
            architecture: Some("gfx1201"),
            device_name: None,
            minimum_abi_version: 1,
            required_features: RuntimeFeatureSet::from_feature(
                RuntimeFeature::HipLinearAttentionQkvPrepare,
            ),
            workspace: WorkspaceFormula {
                fixed_persistent_bytes: 131_072,
                fixed_temporary_bytes: 98_304,
                temporary_bytes_per_batch_item: 0,
                temporary_bytes_per_chunk_token: 0,
                maximum_total_bytes: 229_376,
            },
            state_effect: StateEffect {
                reads: convolution_history,
                writes: convolution_history,
                prepares: StateResourceSet::EMPTY,
                commits: StateResourceSet::EMPTY,
                update_mode: StateUpdateMode::InPlace,
                externally_visible_before_commit: true,
            },
            promotion: PromotionStatus::Production,
            priority: 100,
            fallback_id: None,
            deterministic: true,
            executable: ExecutableOperation::HipLinearAttentionQkvPrepareF32,
            runtime_build: env!("CARGO_PKG_VERSION"),
        },
        ImplementationDescriptor {
            id: "hip.linear-attention-recurrent-f32.m1",
            semantic_version: "1.0.0",
            kind: OperationKind::GatedDeltaRuleScan,
            phases: PhaseSet::ALL_CURRENT,
            input_layout: TensorLayout::TokensHidden,
            output_layout: TensorLayout::TokensHidden,
            weight_layout: None,
            state_layout: OperationStateLayout::RecurrentMatrix,
            activation_format: NumericalFormat::F32,
            value_format: NumericalFormat::F32,
            weight_format: None,
            state_format: NumericalFormat::F32,
            geometry: OperationGeometry::GatedDeltaRule {
                key_heads: 16,
                value_heads: 32,
                key_dim: 128,
                value_dim: 128,
            },
            minimum_batch_width: 1,
            maximum_batch_width: 1,
            minimum_chunk_width: 1,
            maximum_chunk_width: 1,
            backend: OperationBackend::Hip,
            architecture: Some("gfx1201"),
            device_name: None,
            minimum_abi_version: 1,
            required_features: RuntimeFeatureSet::from_feature(
                RuntimeFeature::HipLinearAttentionRecurrent,
            ),
            workspace: WorkspaceFormula {
                fixed_persistent_bytes: 2_097_152,
                fixed_temporary_bytes: 49_408,
                temporary_bytes_per_batch_item: 0,
                temporary_bytes_per_chunk_token: 0,
                maximum_total_bytes: 2_146_560,
            },
            state_effect: StateEffect {
                reads: recurrent_state,
                writes: recurrent_state,
                prepares: StateResourceSet::EMPTY,
                commits: StateResourceSet::EMPTY,
                update_mode: StateUpdateMode::InPlace,
                externally_visible_before_commit: true,
            },
            promotion: PromotionStatus::Production,
            priority: 100,
            fallback_id: None,
            deterministic: true,
            executable: ExecutableOperation::HipLinearAttentionRecurrentF32,
            runtime_build: env!("CARGO_PKG_VERSION"),
        },
        ImplementationDescriptor {
            id: "hip.linear-attention-qkv-prepare-batch-f32.m2-m128",
            semantic_version: "1.0.0",
            kind: OperationKind::LinearAttentionQkvPrepareBatch,
            phases: PhaseSet::ALL_CURRENT,
            input_layout: TensorLayout::TokensHidden,
            output_layout: TensorLayout::TokensHidden,
            weight_layout: None,
            state_layout: OperationStateLayout::ConvolutionHistory,
            activation_format: NumericalFormat::F32,
            value_format: NumericalFormat::F32,
            weight_format: None,
            state_format: NumericalFormat::F32,
            geometry: OperationGeometry::LinearAttentionQkvPrepare {
                key_heads: 16,
                value_heads: 32,
                key_dim: 128,
                value_dim: 128,
                kernel_size: 4,
                query_scale: QueryScale::InverseSqrtKeyDim,
                qk_l2_norm: true,
            },
            minimum_batch_width: 1,
            maximum_batch_width: 1,
            minimum_chunk_width: 2,
            maximum_chunk_width: 128,
            backend: OperationBackend::Hip,
            architecture: Some("gfx1201"),
            device_name: None,
            minimum_abi_version: 1,
            required_features: RuntimeFeatureSet::from_feature(
                RuntimeFeature::HipLinearAttentionQkvPrepareBatch,
            ),
            workspace: WorkspaceFormula {
                fixed_persistent_bytes: 131_072,
                fixed_temporary_bytes: 0,
                temporary_bytes_per_batch_item: 0,
                temporary_bytes_per_chunk_token: 98_304,
                maximum_total_bytes: 12_713_984,
            },
            state_effect: StateEffect {
                reads: convolution_history,
                writes: convolution_history,
                prepares: StateResourceSet::EMPTY,
                commits: StateResourceSet::EMPTY,
                update_mode: StateUpdateMode::InPlace,
                externally_visible_before_commit: true,
            },
            promotion: PromotionStatus::Production,
            priority: 100,
            fallback_id: None,
            deterministic: true,
            executable: ExecutableOperation::HipLinearAttentionQkvPrepareBatchF32,
            runtime_build: env!("CARGO_PKG_VERSION"),
        },
        ImplementationDescriptor {
            id: "hip.linear-attention-recurrent-sequence-f32.m2-m128",
            semantic_version: "1.0.0",
            kind: OperationKind::GatedDeltaRuleSequence,
            phases: PhaseSet::ALL_CURRENT,
            input_layout: TensorLayout::TokensHidden,
            output_layout: TensorLayout::TokensHidden,
            weight_layout: None,
            state_layout: OperationStateLayout::RecurrentMatrix,
            activation_format: NumericalFormat::F32,
            value_format: NumericalFormat::F32,
            weight_format: None,
            state_format: NumericalFormat::F32,
            geometry: OperationGeometry::GatedDeltaRule {
                key_heads: 16,
                value_heads: 32,
                key_dim: 128,
                value_dim: 128,
            },
            minimum_batch_width: 1,
            maximum_batch_width: 1,
            minimum_chunk_width: 2,
            maximum_chunk_width: 128,
            backend: OperationBackend::Hip,
            architecture: Some("gfx1201"),
            device_name: None,
            minimum_abi_version: 1,
            required_features: RuntimeFeatureSet::from_feature(
                RuntimeFeature::HipLinearAttentionRecurrentSequence,
            ),
            workspace: WorkspaceFormula {
                fixed_persistent_bytes: 2_097_152,
                fixed_temporary_bytes: 0,
                temporary_bytes_per_batch_item: 0,
                temporary_bytes_per_chunk_token: 49_408,
                maximum_total_bytes: 8_421_376,
            },
            state_effect: StateEffect {
                reads: recurrent_state,
                writes: recurrent_state,
                prepares: StateResourceSet::EMPTY,
                commits: StateResourceSet::EMPTY,
                update_mode: StateUpdateMode::InPlace,
                externally_visible_before_commit: true,
            },
            promotion: PromotionStatus::Production,
            priority: 100,
            fallback_id: None,
            deterministic: true,
            executable: ExecutableOperation::HipLinearAttentionRecurrentSequenceF32,
            runtime_build: env!("CARGO_PKG_VERSION"),
        },
        ImplementationDescriptor {
            id: "hip.paged-kv-write-f32.m1",
            semantic_version: "1.0.0",
            kind: OperationKind::PagedKvWrite,
            phases: PhaseSet::ALL_CURRENT,
            input_layout: TensorLayout::TokensHidden,
            output_layout: TensorLayout::TokensHidden,
            weight_layout: None,
            state_layout: OperationStateLayout::PagedKvBlocks,
            activation_format: NumericalFormat::F32,
            value_format: NumericalFormat::F32,
            weight_format: None,
            state_format: NumericalFormat::F32,
            geometry: OperationGeometry::PagedKvWrite {
                kv_heads: 4,
                head_dim: 256,
                value_dim: 256,
                block_size: 256,
                cache_blocks: 16,
            },
            minimum_batch_width: 1,
            maximum_batch_width: 1,
            minimum_chunk_width: 1,
            maximum_chunk_width: 1,
            backend: OperationBackend::Hip,
            architecture: Some("gfx1201"),
            device_name: None,
            minimum_abi_version: 1,
            required_features: RuntimeFeatureSet::from_feature(RuntimeFeature::HipPagedKvWrite),
            workspace: WorkspaceFormula {
                fixed_persistent_bytes: 33_554_432,
                fixed_temporary_bytes: 8_192,
                temporary_bytes_per_batch_item: 0,
                temporary_bytes_per_chunk_token: 0,
                maximum_total_bytes: 33_562_624,
            },
            state_effect: StateEffect {
                reads: paged_kv,
                writes: paged_kv,
                prepares: StateResourceSet::EMPTY,
                commits: StateResourceSet::EMPTY,
                update_mode: StateUpdateMode::InPlace,
                externally_visible_before_commit: true,
            },
            promotion: PromotionStatus::Production,
            priority: 100,
            fallback_id: None,
            deterministic: true,
            executable: ExecutableOperation::HipPagedKvWriteF32,
            runtime_build: env!("CARGO_PKG_VERSION"),
        },
        ImplementationDescriptor {
            id: "hip.fused-qk-norm-rope-paged-kv-write-f32.m1",
            semantic_version: "1.0.0",
            kind: OperationKind::FusedQkNormRopePagedKvWrite,
            phases: PhaseSet::ALL_CURRENT,
            input_layout: TensorLayout::TokensHidden,
            output_layout: TensorLayout::TokensHidden,
            weight_layout: Some(TensorLayout::RowMajor),
            state_layout: OperationStateLayout::PagedKvBlocks,
            activation_format: NumericalFormat::F32,
            value_format: NumericalFormat::F32,
            weight_format: Some(NumericalFormat::F32),
            state_format: NumericalFormat::F32,
            geometry: OperationGeometry::FusedQkNormRopePagedKvWrite {
                q_heads: 16,
                kv_heads: 4,
                head_dim: 256,
                value_dim: 256,
                rotary_dim: 64,
                rope_base_bits: 10_000_000.0_f32.to_bits(),
                norm_epsilon_bits: 1e-5_f32.to_bits(),
                block_size: 256,
                cache_blocks: 16,
            },
            minimum_batch_width: 1,
            maximum_batch_width: 1,
            minimum_chunk_width: 1,
            maximum_chunk_width: 1,
            backend: OperationBackend::Hip,
            architecture: Some("gfx1201"),
            device_name: None,
            minimum_abi_version: 1,
            required_features: RuntimeFeatureSet::from_feature(
                RuntimeFeature::HipFusedQkNormRopePagedKvWrite,
            ),
            workspace: WorkspaceFormula {
                fixed_persistent_bytes: 33_554_432,
                fixed_temporary_bytes: 73_728,
                temporary_bytes_per_batch_item: 0,
                temporary_bytes_per_chunk_token: 0,
                maximum_total_bytes: 33_628_160,
            },
            state_effect: StateEffect {
                reads: paged_kv,
                writes: paged_kv,
                prepares: StateResourceSet::EMPTY,
                commits: StateResourceSet::EMPTY,
                update_mode: StateUpdateMode::InPlace,
                externally_visible_before_commit: true,
            },
            promotion: PromotionStatus::Production,
            priority: 100,
            fallback_id: None,
            deterministic: true,
            executable: ExecutableOperation::HipFusedQkNormRopePagedKvWriteF32,
            runtime_build: env!("CARGO_PKG_VERSION"),
        },
        ImplementationDescriptor {
            id: "hip.paged-decode-attention-f32.m1-gqa",
            semantic_version: "1.0.0",
            kind: OperationKind::PagedCausalGqaRead,
            phases: PhaseSet::ALL_CURRENT,
            input_layout: TensorLayout::TokensHidden,
            output_layout: TensorLayout::TokensHidden,
            weight_layout: None,
            state_layout: OperationStateLayout::PagedKvBlocks,
            activation_format: NumericalFormat::F32,
            value_format: NumericalFormat::F32,
            weight_format: None,
            state_format: NumericalFormat::F32,
            geometry: OperationGeometry::PagedCausalGqaRead {
                q_heads: 16,
                kv_heads: 4,
                head_dim: 256,
                value_dim: 256,
                block_size: 256,
                cache_blocks: 16,
                sigmoid_gate: false,
            },
            minimum_batch_width: 1,
            maximum_batch_width: 1,
            minimum_chunk_width: 1,
            maximum_chunk_width: 1,
            backend: OperationBackend::Hip,
            architecture: Some("gfx1201"),
            device_name: None,
            minimum_abi_version: 1,
            required_features: RuntimeFeatureSet::from_feature(
                RuntimeFeature::HipPagedDecodeAttention,
            ),
            workspace: WorkspaceFormula {
                fixed_persistent_bytes: 33_554_432,
                fixed_temporary_bytes: 32_832,
                temporary_bytes_per_batch_item: 0,
                temporary_bytes_per_chunk_token: 0,
                maximum_total_bytes: 33_587_264,
            },
            state_effect: StateEffect {
                reads: paged_kv,
                writes: StateResourceSet::EMPTY,
                prepares: StateResourceSet::EMPTY,
                commits: StateResourceSet::EMPTY,
                update_mode: StateUpdateMode::ReadOnly,
                externally_visible_before_commit: false,
            },
            promotion: PromotionStatus::Production,
            priority: 100,
            fallback_id: None,
            deterministic: true,
            executable: ExecutableOperation::HipPagedDecodeAttentionF32,
            runtime_build: env!("CARGO_PKG_VERSION"),
        },
        ImplementationDescriptor {
            id: "hip.paged-decode-attention-sigmoid-gate-f32.m1-gqa",
            semantic_version: "1.0.0",
            kind: OperationKind::PagedCausalGqaRead,
            phases: PhaseSet::ALL_CURRENT,
            input_layout: TensorLayout::TokensHidden,
            output_layout: TensorLayout::TokensHidden,
            weight_layout: None,
            state_layout: OperationStateLayout::PagedKvBlocks,
            activation_format: NumericalFormat::F32,
            value_format: NumericalFormat::F32,
            weight_format: None,
            state_format: NumericalFormat::F32,
            geometry: OperationGeometry::PagedCausalGqaRead {
                q_heads: 16,
                kv_heads: 4,
                head_dim: 256,
                value_dim: 256,
                block_size: 256,
                cache_blocks: 16,
                sigmoid_gate: true,
            },
            minimum_batch_width: 1,
            maximum_batch_width: 1,
            minimum_chunk_width: 1,
            maximum_chunk_width: 1,
            backend: OperationBackend::Hip,
            architecture: Some("gfx1201"),
            device_name: None,
            minimum_abi_version: 1,
            required_features: RuntimeFeatureSet::from_feature(
                RuntimeFeature::HipPagedDecodeAttention,
            ),
            workspace: WorkspaceFormula {
                fixed_persistent_bytes: 33_554_432,
                fixed_temporary_bytes: 49_216,
                temporary_bytes_per_batch_item: 0,
                temporary_bytes_per_chunk_token: 0,
                maximum_total_bytes: 33_603_648,
            },
            state_effect: StateEffect {
                reads: paged_kv,
                writes: StateResourceSet::EMPTY,
                prepares: StateResourceSet::EMPTY,
                commits: StateResourceSet::EMPTY,
                update_mode: StateUpdateMode::ReadOnly,
                externally_visible_before_commit: false,
            },
            promotion: PromotionStatus::Production,
            priority: 100,
            fallback_id: None,
            deterministic: true,
            executable: ExecutableOperation::HipPagedDecodeAttentionSigmoidGateF32,
            runtime_build: env!("CARGO_PKG_VERSION"),
        },
    ])
}

/// Build the exact M1 request from geometry validated against package weights.
pub fn qwen35_m1_operation_request(
    kind: OperationKind,
    phase: ExecutionPhase,
    geometry: OperationGeometry,
    device: DeviceCapabilities,
    workspace_budget_bytes: u64,
) -> OperationRequest {
    OperationRequest {
        kind,
        phase,
        input_layout: TensorLayout::TokensHidden,
        output_layout: TensorLayout::TokensHidden,
        weight_layout: (kind == OperationKind::FusedQkNormRopePagedKvWrite)
            .then_some(TensorLayout::RowMajor),
        state_layout: match kind {
            OperationKind::PagedKvWrite | OperationKind::FusedQkNormRopePagedKvWrite => {
                OperationStateLayout::PagedKvBlocks
            }
            OperationKind::LinearAttentionQkvPrepare
            | OperationKind::LinearAttentionQkvPrepareBatch => {
                OperationStateLayout::ConvolutionHistory
            }
            OperationKind::GatedDeltaRuleScan | OperationKind::GatedDeltaRuleSequence => {
                OperationStateLayout::RecurrentMatrix
            }
            OperationKind::Aq4MatvecBatch => OperationStateLayout::None,
            OperationKind::PagedCausalGqaRead => OperationStateLayout::PagedKvBlocks,
        },
        activation_format: NumericalFormat::F32,
        value_format: NumericalFormat::F32,
        weight_format: (kind == OperationKind::FusedQkNormRopePagedKvWrite)
            .then_some(NumericalFormat::F32),
        state_format: NumericalFormat::F32,
        geometry,
        batch_width: 1,
        chunk_width: 1,
        device,
        workspace_budget_bytes,
        minimum_promotion: PromotionStatus::Production,
    }
}

/// Builds a one-request linear-attention sequence request for `2..=128` prompt tokens.
pub fn qwen35_sequence_operation_request(
    kind: OperationKind,
    phase: ExecutionPhase,
    geometry: OperationGeometry,
    sequence_len: u64,
    device: DeviceCapabilities,
    workspace_budget_bytes: u64,
) -> Result<OperationRequest, String> {
    let state_layout = match kind {
        OperationKind::LinearAttentionQkvPrepareBatch => OperationStateLayout::ConvolutionHistory,
        OperationKind::GatedDeltaRuleSequence => OperationStateLayout::RecurrentMatrix,
        _ => {
            return Err(format!(
                "operation kind {kind:?} is not a one-request linear-attention sequence"
            ));
        }
    };
    Ok(OperationRequest {
        kind,
        phase,
        input_layout: TensorLayout::TokensHidden,
        output_layout: TensorLayout::TokensHidden,
        weight_layout: None,
        state_layout,
        activation_format: NumericalFormat::F32,
        value_format: NumericalFormat::F32,
        weight_format: None,
        state_format: NumericalFormat::F32,
        geometry,
        batch_width: 1,
        chunk_width: sequence_len,
        device,
        workspace_budget_bytes,
        minimum_promotion: PromotionStatus::Production,
    })
}

/// Builds a shape-exact AQ4 projection request for a batch width in `2..=128`.
pub fn aq4_matvec_batch_operation_request(
    phase: ExecutionPhase,
    geometry: OperationGeometry,
    batch_count: u64,
    device: DeviceCapabilities,
    workspace_budget_bytes: u64,
) -> Result<OperationRequest, String> {
    if !matches!(geometry, OperationGeometry::Aq4MatvecBatch { .. }) {
        return Err("AQ4 matvec batch request requires AQ4 matvec batch geometry".into());
    }
    if !(2..=128).contains(&batch_count) {
        return Err(format!(
            "AQ4 matvec batch width must be in 2..=128, got {batch_count}"
        ));
    }
    Ok(OperationRequest {
        kind: OperationKind::Aq4MatvecBatch,
        phase,
        input_layout: TensorLayout::TokensHidden,
        output_layout: TensorLayout::TokensHidden,
        weight_layout: Some(TensorLayout::RowMajor),
        state_layout: OperationStateLayout::None,
        activation_format: NumericalFormat::F32,
        value_format: NumericalFormat::F32,
        weight_format: Some(NumericalFormat::Aq4_0),
        state_format: NumericalFormat::F32,
        geometry,
        batch_width: batch_count,
        chunk_width: 1,
        device,
        workspace_budget_bytes,
        minimum_promotion: PromotionStatus::Production,
    })
}

fn aq4_matvec_batch_descriptor(
    geometry: OperationGeometry,
    device: &DeviceCapabilities,
) -> Result<ImplementationDescriptor, String> {
    let OperationGeometry::Aq4MatvecBatch {
        rows,
        cols,
        group_size,
        scale_count,
        row_scale_count,
        tensor_scale_bits,
    } = geometry
    else {
        return Err("AQ4 matvec batch descriptor requires AQ4 matvec batch geometry".into());
    };
    if rows == 0 || cols == 0 || group_size == 0 || scale_count == 0 {
        return Err("AQ4 matvec batch descriptor geometry contains zero dimensions".into());
    }
    let tensor_scale = f32::from_bits(tensor_scale_bits);
    if !tensor_scale.is_finite() || tensor_scale <= 0.0 {
        return Err("AQ4 matvec batch descriptor tensor scale is invalid".into());
    }
    let elements = rows
        .checked_mul(cols)
        .ok_or_else(|| "AQ4 matvec batch descriptor matrix elements overflow".to_string())?;
    let _ = elements
        .checked_add(group_size - 1)
        .and_then(|value| value.checked_div(group_size))
        .ok_or_else(|| "AQ4 matvec batch descriptor group count overflow".to_string())?;
    if row_scale_count != 0 && row_scale_count != rows {
        return Err(format!(
            "AQ4 matvec batch descriptor row scale count {row_scale_count} must be zero or rows {rows}"
        ));
    }
    let bytes_per_item = rows
        .checked_add(cols)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| "AQ4 matvec batch descriptor workspace overflows".to_string())?;
    let bytes_per_item = u64::try_from(bytes_per_item)
        .map_err(|_| "AQ4 matvec batch descriptor workspace exceeds u64".to_string())?;
    let maximum_total_bytes = 1_u64 << 30;
    let maximum_estimate = bytes_per_item
        .checked_mul(128)
        .ok_or_else(|| "AQ4 matvec batch descriptor workspace overflows".to_string())?;
    if maximum_estimate > maximum_total_bytes {
        return Err("AQ4 matvec batch descriptor workspace exceeds bounded maximum".into());
    }
    let (backend, architecture, required_features, id) = match device.backend {
        OperationBackend::Host => (
            OperationBackend::Host,
            None,
            RuntimeFeatureSet::EMPTY,
            "host.aq4-matvec-batch-f32.m2-m128",
        ),
        OperationBackend::Hip => {
            if device.architecture.as_deref() != Some("gfx1201") {
                return Err(format!(
                    "AQ4 matvec batch production requires gfx1201, got {}",
                    device.architecture.as_deref().unwrap_or("unavailable")
                ));
            }
            (
                OperationBackend::Hip,
                Some("gfx1201"),
                RuntimeFeatureSet::from_feature(RuntimeFeature::HipAq4MatvecBatch),
                "hip.aq4-matvec-batch-f32.m2-m128",
            )
        }
    };
    Ok(ImplementationDescriptor {
        id,
        semantic_version: "1.0.0",
        kind: OperationKind::Aq4MatvecBatch,
        phases: PhaseSet::ALL_CURRENT,
        input_layout: TensorLayout::TokensHidden,
        output_layout: TensorLayout::TokensHidden,
        weight_layout: Some(TensorLayout::RowMajor),
        state_layout: OperationStateLayout::None,
        activation_format: NumericalFormat::F32,
        value_format: NumericalFormat::F32,
        weight_format: Some(NumericalFormat::Aq4_0),
        state_format: NumericalFormat::F32,
        geometry,
        minimum_batch_width: 2,
        maximum_batch_width: 128,
        minimum_chunk_width: 1,
        maximum_chunk_width: 1,
        backend,
        architecture,
        device_name: None,
        minimum_abi_version: 1,
        required_features,
        workspace: WorkspaceFormula {
            fixed_persistent_bytes: 0,
            fixed_temporary_bytes: 0,
            temporary_bytes_per_batch_item: bytes_per_item,
            temporary_bytes_per_chunk_token: 0,
            maximum_total_bytes,
        },
        state_effect: StateEffect {
            reads: StateResourceSet::EMPTY,
            writes: StateResourceSet::EMPTY,
            prepares: StateResourceSet::EMPTY,
            commits: StateResourceSet::EMPTY,
            update_mode: StateUpdateMode::ReadOnly,
            externally_visible_before_commit: false,
        },
        promotion: PromotionStatus::Production,
        priority: 100,
        fallback_id: None,
        deterministic: true,
        executable: ExecutableOperation::HipAq4MatvecBatchF32,
        runtime_build: env!("CARGO_PKG_VERSION"),
    })
}

/// Builds a production AQ4 batch registry for one exact resident matrix shape.
pub fn aq4_matvec_batch_production_registry(
    geometry: OperationGeometry,
    device: &DeviceCapabilities,
) -> Result<BackendOperationRegistry, String> {
    BackendOperationRegistry::new(vec![aq4_matvec_batch_descriptor(geometry, device)?])
}

/// One pre-resolved plan for every current execution phase.
#[derive(Debug, PartialEq, Eq)]
pub struct ResolvedPhasePlans {
    cold_prefill: ResolvedOperationPlan,
    cached_prefix_prefill: ResolvedOperationPlan,
    decode: ResolvedOperationPlan,
}

impl ResolvedPhasePlans {
    pub fn resolve_m1(
        registry: &BackendOperationRegistry,
        kind: OperationKind,
        geometry: OperationGeometry,
        device: &DeviceCapabilities,
        workspace_budget_bytes: u64,
    ) -> Result<Self, String> {
        Ok(Self {
            cold_prefill: registry
                .admit(qwen35_m1_operation_request(
                    kind,
                    ExecutionPhase::ColdPrefill,
                    geometry,
                    device.clone(),
                    workspace_budget_bytes,
                ))?
                .into_plan(),
            cached_prefix_prefill: registry
                .admit(qwen35_m1_operation_request(
                    kind,
                    ExecutionPhase::CachedPrefixPrefill,
                    geometry,
                    device.clone(),
                    workspace_budget_bytes,
                ))?
                .into_plan(),
            decode: registry
                .admit(qwen35_m1_operation_request(
                    kind,
                    ExecutionPhase::Decode,
                    geometry,
                    device.clone(),
                    workspace_budget_bytes,
                ))?
                .into_plan(),
        })
    }

    /// Resolves one-request sequence plans at the concrete runtime width for every current phase.
    ///
    /// Callers resolve only the width they are about to execute; descriptors enforce `2..=128`,
    /// and each returned plan retains that exact width in its trace and ABI binding.
    pub fn resolve_sequence(
        registry: &BackendOperationRegistry,
        kind: OperationKind,
        geometry: OperationGeometry,
        sequence_len: u64,
        device: &DeviceCapabilities,
        workspace_budget_bytes: u64,
    ) -> Result<Self, String> {
        let resolve = |phase| {
            registry
                .admit(qwen35_sequence_operation_request(
                    kind,
                    phase,
                    geometry,
                    sequence_len,
                    device.clone(),
                    workspace_budget_bytes,
                )?)
                .map(PrestartAttempt::into_plan)
        };
        Ok(Self {
            cold_prefill: resolve(ExecutionPhase::ColdPrefill)?,
            cached_prefix_prefill: resolve(ExecutionPhase::CachedPrefixPrefill)?,
            decode: resolve(ExecutionPhase::Decode)?,
        })
    }

    /// Resolves an exact resident AQ4 matrix shape for every current execution phase.
    pub fn resolve_aq4_batch(
        registry: &BackendOperationRegistry,
        geometry: OperationGeometry,
        batch_count: u64,
        device: &DeviceCapabilities,
        workspace_budget_bytes: u64,
    ) -> Result<Self, String> {
        let resolve = |phase| {
            registry
                .admit(aq4_matvec_batch_operation_request(
                    phase,
                    geometry,
                    batch_count,
                    device.clone(),
                    workspace_budget_bytes,
                )?)
                .map(PrestartAttempt::into_plan)
        };
        Ok(Self {
            cold_prefill: resolve(ExecutionPhase::ColdPrefill)?,
            cached_prefix_prefill: resolve(ExecutionPhase::CachedPrefixPrefill)?,
            decode: resolve(ExecutionPhase::Decode)?,
        })
    }

    pub const fn for_phase(&self, phase: ExecutionPhase) -> &ResolvedOperationPlan {
        match phase {
            ExecutionPhase::ColdPrefill => &self.cold_prefill,
            ExecutionPhase::CachedPrefixPrefill => &self.cached_prefix_prefill,
            ExecutionPhase::Decode => &self.decode,
        }
    }

    pub fn traces(&self) -> [OperationResolutionTrace; 3] {
        [
            self.cold_prefill.trace(),
            self.cached_prefix_prefill.trace(),
            self.decode.trace(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_hip_capabilities() -> DeviceCapabilities {
        DeviceCapabilities {
            device_id: 1,
            backend: OperationBackend::Hip,
            architecture: Some("gfx1201".into()),
            device_name: Some("test HIP device".into()),
            abi_version: 1,
            runtime_features: RuntimeFeatureSet::EMPTY
                .with(RuntimeFeature::HipLinearAttentionRecurrent)
                .with(RuntimeFeature::HipPagedDecodeAttention)
                .with(RuntimeFeature::HipFusedQkNormRopePagedKvWrite)
                .with(RuntimeFeature::HipLinearAttentionQkvPrepare)
                .with(RuntimeFeature::HipPagedKvWrite)
                .with(RuntimeFeature::HipAq4MatvecBatch)
                .with(RuntimeFeature::HipLinearAttentionQkvPrepareBatch)
                .with(RuntimeFeature::HipLinearAttentionRecurrentSequence),
            workspace_capacity_bytes: u64::MAX,
        }
    }

    fn recurrent_request(phase: ExecutionPhase) -> OperationRequest {
        qwen35_m1_operation_request(
            OperationKind::GatedDeltaRuleScan,
            phase,
            OperationGeometry::GatedDeltaRule {
                key_heads: 16,
                value_heads: 32,
                key_dim: 128,
                value_dim: 128,
            },
            test_hip_capabilities(),
            u64::MAX,
        )
    }

    fn recurrent_descriptor(registry: &BackendOperationRegistry) -> ImplementationDescriptor {
        registry
            .implementations()
            .iter()
            .find(|descriptor| descriptor.kind == OperationKind::GatedDeltaRuleScan)
            .unwrap()
            .clone()
    }

    #[test]
    fn m1_registry_resolves_all_current_phases_without_a_model_name_key() {
        let registry = qwen35_m1_production_registry().unwrap();
        for phase in [
            ExecutionPhase::ColdPrefill,
            ExecutionPhase::CachedPrefixPrefill,
            ExecutionPhase::Decode,
        ] {
            let plan = registry.resolve(&recurrent_request(phase)).unwrap();
            assert_eq!(plan.trace().phase, phase);
            assert_eq!(
                plan.trace().executable,
                ExecutableOperation::HipLinearAttentionRecurrentF32
            );
        }
    }

    fn sequence_request(
        kind: OperationKind,
        phase: ExecutionPhase,
        sequence_len: u64,
    ) -> OperationRequest {
        let geometry = match kind {
            OperationKind::LinearAttentionQkvPrepareBatch => {
                OperationGeometry::LinearAttentionQkvPrepare {
                    key_heads: 16,
                    value_heads: 32,
                    key_dim: 128,
                    value_dim: 128,
                    kernel_size: 4,
                    query_scale: QueryScale::InverseSqrtKeyDim,
                    qk_l2_norm: true,
                }
            }
            OperationKind::GatedDeltaRuleSequence => OperationGeometry::GatedDeltaRule {
                key_heads: 16,
                value_heads: 32,
                key_dim: 128,
                value_dim: 128,
            },
            other => panic!("unexpected sequence operation {other:?}"),
        };
        qwen35_sequence_operation_request(
            kind,
            phase,
            geometry,
            sequence_len,
            test_hip_capabilities(),
            u64::MAX,
        )
        .unwrap()
    }

    fn aq4_geometry() -> OperationGeometry {
        OperationGeometry::Aq4MatvecBatch {
            rows: 2,
            cols: 3,
            group_size: 2,
            scale_count: 2,
            row_scale_count: 0,
            tensor_scale_bits: 10.0_f32.to_bits(),
        }
    }

    #[test]
    fn aq4_batch_registry_is_shape_exact_and_width_bounded() {
        let context = ullm_runtime_sys::RuntimeContext::create(0).unwrap();
        let device = DeviceCapabilities::from_runtime_context(&context).unwrap();
        let registry = aq4_matvec_batch_production_registry(aq4_geometry(), &device).unwrap();
        assert_eq!(registry.implementations().len(), 1);
        for width in [2, 128] {
            let request = aq4_matvec_batch_operation_request(
                ExecutionPhase::Decode,
                aq4_geometry(),
                width,
                device.clone(),
                u64::MAX,
            )
            .unwrap();
            assert!(registry.admit(request).is_ok());
        }
        for width in [1, 129] {
            assert!(
                aq4_matvec_batch_operation_request(
                    ExecutionPhase::Decode,
                    aq4_geometry(),
                    width,
                    device.clone(),
                    device.workspace_capacity_bytes,
                )
                .is_err()
            );
        }
        let mut wrong_geometry = aq4_geometry();
        if let OperationGeometry::Aq4MatvecBatch { cols, .. } = &mut wrong_geometry {
            *cols = 4;
        }
        let request = aq4_matvec_batch_operation_request(
            ExecutionPhase::Decode,
            wrong_geometry,
            2,
            device.clone(),
            device.workspace_capacity_bytes,
        )
        .unwrap();
        assert!(registry.admit(request).is_err());

        let mut hip = test_hip_capabilities();
        hip.runtime_features = RuntimeFeatureSet::EMPTY;
        let hip_registry = aq4_matvec_batch_production_registry(aq4_geometry(), &hip).unwrap();
        let request = aq4_matvec_batch_operation_request(
            ExecutionPhase::Decode,
            aq4_geometry(),
            2,
            hip,
            u64::MAX,
        )
        .unwrap();
        assert!(hip_registry.admit(request).is_err());
    }

    #[test]
    fn planned_aq4_batch_wrapper_matches_direct_cpu_abi() {
        let mut context = ullm_runtime_sys::RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let mut index = context.alloc_buffer(3).unwrap();
        let mut scale = context.alloc_buffer(3).unwrap();
        let mut codebook = context.alloc_buffer(16 * 4).unwrap();
        let mut scale_values = context.alloc_buffer(2 * 4).unwrap();
        let mut input = context.alloc_buffer(2 * 3 * 4).unwrap();
        let mut direct_output = context.alloc_buffer(2 * 2 * 4).unwrap();
        let mut planned_output = context.alloc_buffer(2 * 2 * 4).unwrap();
        index
            .copy_from_host(0, &[0x21_u8, 0x03, 0x54], Some(&mut stream))
            .unwrap();
        scale
            .copy_from_host(0, &[0_u8, 1, 0], Some(&mut stream))
            .unwrap();
        let codebook_bytes = (0..16_u32)
            .flat_map(|value| (value as f32).to_le_bytes())
            .collect::<Vec<_>>();
        codebook
            .copy_from_host(0, &codebook_bytes, Some(&mut stream))
            .unwrap();
        scale_values
            .copy_from_host(
                0,
                &[0.5_f32.to_le_bytes(), 2.0_f32.to_le_bytes()]
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>(),
                Some(&mut stream),
            )
            .unwrap();
        input
            .copy_from_host(
                0,
                &[0.5_f32, -1.0, 2.0, 1.0, 0.0, -0.5]
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect::<Vec<_>>(),
                Some(&mut stream),
            )
            .unwrap();
        let device = DeviceCapabilities::from_runtime_context(&context).unwrap();
        let registry = aq4_matvec_batch_production_registry(aq4_geometry(), &device).unwrap();
        let request = aq4_matvec_batch_operation_request(
            ExecutionPhase::Decode,
            aq4_geometry(),
            2,
            device,
            u64::MAX,
        )
        .unwrap();
        ullm_runtime_sys::aq4_matvec_batch_f32(
            &index,
            &scale,
            &codebook,
            &scale_values,
            &input,
            None,
            2,
            2,
            10.0,
            0,
            2,
            3,
            2,
            &mut direct_output,
            Some(&mut stream),
        )
        .unwrap();
        registry
            .admit(request)
            .unwrap()
            .start()
            .execute_aq4_matvec_batch_f32(
                &index,
                &scale,
                &codebook,
                &scale_values,
                &input,
                None,
                &mut planned_output,
                &mut stream,
            )
            .unwrap();
        stream.synchronize().unwrap();
        let mut direct_bytes = vec![0_u8; 16];
        let mut planned_bytes = vec![0_u8; 16];
        direct_output
            .copy_to_host(0, &mut direct_bytes, Some(&mut stream))
            .unwrap();
        planned_output
            .copy_to_host(0, &mut planned_bytes, Some(&mut stream))
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(direct_bytes, planned_bytes);
    }

    #[test]
    fn sequence_registry_width_bounds_and_features_fail_closed_in_every_phase() {
        let registry = qwen35_m1_production_registry().unwrap();
        for phase in [
            ExecutionPhase::ColdPrefill,
            ExecutionPhase::CachedPrefixPrefill,
            ExecutionPhase::Decode,
        ] {
            for kind in [
                OperationKind::LinearAttentionQkvPrepareBatch,
                OperationKind::GatedDeltaRuleSequence,
            ] {
                assert!(registry.resolve(&sequence_request(kind, phase, 1)).is_err());
                for width in [2, 128] {
                    let trace = registry
                        .resolve(&sequence_request(kind, phase, width))
                        .unwrap()
                        .trace();
                    assert_eq!(trace.phase, phase);
                    assert_eq!(trace.batch_width, 1);
                    assert_eq!(trace.chunk_width, width);
                }
                assert!(
                    registry
                        .resolve(&sequence_request(kind, phase, 129))
                        .is_err()
                );

                let mut missing = sequence_request(kind, phase, 2);
                missing.device.runtime_features = RuntimeFeatureSet::EMPTY;
                assert!(registry.resolve(&missing).is_err());
            }
        }
        assert!(
            qwen35_sequence_operation_request(
                OperationKind::GatedDeltaRuleScan,
                ExecutionPhase::ColdPrefill,
                OperationGeometry::GatedDeltaRule {
                    key_heads: 16,
                    value_heads: 32,
                    key_dim: 128,
                    value_dim: 128,
                },
                2,
                test_hip_capabilities(),
                u64::MAX,
            )
            .is_err()
        );
    }

    #[test]
    fn equal_priority_sequence_descriptor_is_ambiguous() {
        let registry = qwen35_m1_production_registry().unwrap();
        let mut entries = registry.implementations().to_vec();
        let mut duplicate = entries
            .iter()
            .find(|descriptor| descriptor.kind == OperationKind::GatedDeltaRuleSequence)
            .unwrap()
            .clone();
        duplicate.id = "hip.linear-attention-recurrent-sequence-f32.ambiguous";
        entries.push(duplicate);
        let registry = BackendOperationRegistry::new(entries).unwrap();
        assert!(
            registry
                .resolve(&sequence_request(
                    OperationKind::GatedDeltaRuleSequence,
                    ExecutionPhase::ColdPrefill,
                    2,
                ))
                .unwrap_err()
                .contains("ambiguous")
        );
    }

    #[test]
    fn sequence_phase_plans_retain_the_concrete_execution_width() {
        let registry = qwen35_m1_production_registry().unwrap();
        for width in [2, 17, 128] {
            let plans = ResolvedPhasePlans::resolve_sequence(
                &registry,
                OperationKind::GatedDeltaRuleSequence,
                OperationGeometry::GatedDeltaRule {
                    key_heads: 16,
                    value_heads: 32,
                    key_dim: 128,
                    value_dim: 128,
                },
                width,
                &test_hip_capabilities(),
                u64::MAX,
            )
            .unwrap();
            for phase in [
                ExecutionPhase::ColdPrefill,
                ExecutionPhase::CachedPrefixPrefill,
                ExecutionPhase::Decode,
            ] {
                let trace = plans.for_phase(phase).trace();
                assert_eq!(trace.phase, phase);
                assert_eq!(trace.batch_width, 1);
                assert_eq!(trace.chunk_width, width);
            }
        }
        assert!(
            ResolvedPhasePlans::resolve_sequence(
                &registry,
                OperationKind::GatedDeltaRuleSequence,
                OperationGeometry::GatedDeltaRule {
                    key_heads: 16,
                    value_heads: 32,
                    key_dim: 128,
                    value_dim: 128,
                },
                1,
                &test_hip_capabilities(),
                u64::MAX,
            )
            .is_err()
        );
    }

    #[test]
    fn missing_feature_geometry_and_width_fail_closed() {
        let registry = qwen35_m1_production_registry().unwrap();
        let mut request = recurrent_request(ExecutionPhase::Decode);
        request.device.runtime_features = RuntimeFeatureSet::EMPTY;
        assert!(registry.resolve(&request).is_err());
        request = recurrent_request(ExecutionPhase::Decode);
        request.device.abi_version = 0;
        assert!(registry.resolve(&request).is_err());
        request = recurrent_request(ExecutionPhase::Decode);
        request.chunk_width = 2;
        assert!(registry.resolve(&request).is_err());
        request = recurrent_request(ExecutionPhase::Decode);
        request.geometry = OperationGeometry::GatedDeltaRule {
            key_heads: 8,
            value_heads: 32,
            key_dim: 128,
            value_dim: 128,
        };
        assert!(registry.resolve(&request).is_err());
    }

    #[test]
    fn equal_priority_is_ambiguous_instead_of_declaration_ordered() {
        let registry = qwen35_m1_production_registry().unwrap();
        let mut entries = registry.implementations().to_vec();
        let mut duplicate = recurrent_descriptor(&registry);
        duplicate.id = "hip.linear-attention-recurrent-f32.ambiguous";
        entries.push(duplicate);
        let registry = BackendOperationRegistry::new(entries).unwrap();
        assert!(
            registry
                .resolve(&recurrent_request(ExecutionPhase::Decode))
                .unwrap_err()
                .contains("ambiguous")
        );
    }

    #[test]
    fn workspace_overflow_and_descriptor_bound_are_rejected() {
        let formula = WorkspaceFormula {
            fixed_persistent_bytes: 1,
            fixed_temporary_bytes: 0,
            temporary_bytes_per_batch_item: u64::MAX,
            temporary_bytes_per_chunk_token: 0,
            maximum_total_bytes: u64::MAX,
        };
        assert!(formula.estimate(2, 1).is_err());
        let registry = qwen35_m1_production_registry().unwrap();
        let mut request = recurrent_request(ExecutionPhase::Decode);
        request.workspace_budget_bytes = 1;
        assert!(registry.admit(request).is_err());
    }

    #[test]
    fn descriptor_semantics_and_fallback_cycles_fail_at_construction() {
        let registry = qwen35_m1_production_registry().unwrap();
        let mut wrong = recurrent_descriptor(&registry);
        wrong.executable = ExecutableOperation::HipPagedDecodeAttentionF32;
        assert!(BackendOperationRegistry::new(vec![wrong]).is_err());

        let mut wrong_effect = recurrent_descriptor(&registry);
        wrong_effect.state_effect.update_mode = StateUpdateMode::ReadOnly;
        assert!(BackendOperationRegistry::new(vec![wrong_effect]).is_err());
        let mut unsupported_commit = recurrent_descriptor(&registry);
        unsupported_commit.state_effect.update_mode = StateUpdateMode::PreparedCommit;
        assert!(BackendOperationRegistry::new(vec![unsupported_commit]).is_err());

        let mut first = recurrent_descriptor(&registry);
        first.id = "recurrent-cycle-a";
        first.fallback_id = Some("recurrent-cycle-b");
        let mut second = recurrent_descriptor(&registry);
        second.id = "recurrent-cycle-b";
        second.fallback_id = Some("recurrent-cycle-a");
        assert!(BackendOperationRegistry::new(vec![first, second]).is_err());

        let prepare = registry
            .implementations()
            .iter()
            .find(|descriptor| descriptor.kind == OperationKind::LinearAttentionQkvPrepare)
            .unwrap();
        let mut wrong_scale = qwen35_m1_operation_request(
            OperationKind::LinearAttentionQkvPrepare,
            ExecutionPhase::Decode,
            OperationGeometry::LinearAttentionQkvPrepare {
                key_heads: 16,
                value_heads: 32,
                key_dim: 128,
                value_dim: 128,
                kernel_size: 4,
                query_scale: QueryScale::ExactF32Bits(1.0_f32.to_bits()),
                qk_l2_norm: true,
            },
            test_hip_capabilities(),
            u64::MAX,
        );
        wrong_scale.device.architecture = prepare.architecture.map(str::to_string);
        assert!(registry.admit(wrong_scale).is_err());
    }

    #[test]
    fn cpu_runtime_probe_never_claims_hip_features() {
        let context = ullm_runtime_sys::RuntimeContext::create(0).unwrap();
        let capabilities = DeviceCapabilities::from_runtime_context(&context).unwrap();
        assert_eq!(capabilities.backend, OperationBackend::Host);
        assert!(capabilities.runtime_features.is_empty());
        let mut request = recurrent_request(ExecutionPhase::Decode);
        request.device = capabilities;
        assert!(
            qwen35_m1_production_registry()
                .unwrap()
                .admit(request)
                .is_err()
        );
    }

    #[test]
    fn runtime_architecture_normalization_is_exact_and_fail_closed() {
        let info = ullm_runtime_sys::DeviceInfo {
            device_id: 0,
            backend: "hip".into(),
            name: "AMD Radeon AI PRO R9700".into(),
            total_global_mem: 1,
            compute_major: 12,
            compute_minor: 0,
            gcn_arch_name: "gfx1201:sramecc+:xnack-".into(),
            flags: 0,
        };
        assert_eq!(
            normalized_device_architecture(&info).unwrap(),
            Some("gfx1201".into())
        );
        let mut wrong = info.clone();
        wrong.gcn_arch_name = "gfx1100".into();
        assert_eq!(
            normalized_device_architecture(&wrong).unwrap(),
            Some("gfx1100".into())
        );
        let mut unknown = info.clone();
        unknown.gcn_arch_name.clear();
        unknown.name = "AMD Radeon AI PRO R9700".into();
        assert!(normalized_device_architecture(&unknown).is_err());
        require_device_architecture(&info, "gfx1201").unwrap();
        let mut gfx1200 = info.clone();
        gfx1200.gcn_arch_name = "gfx1200".into();
        assert!(require_device_architecture(&gfx1200, "gfx1201").is_err());
        assert!(require_device_architecture(&unknown, "gfx1201").is_err());
    }

    #[test]
    fn available_hip_device_reports_property_architecture_without_inference() {
        let hip = (1..ullm_runtime_sys::device_count().unwrap())
            .filter_map(|index| ullm_runtime_sys::device_info(index).ok())
            .find(|info| info.backend == "hip");
        let Some(info) = hip else {
            return;
        };
        let architecture = normalized_device_architecture(&info).unwrap();
        eprintln!(
            "property device={} architecture={}",
            info.name,
            architecture.as_deref().unwrap_or("missing")
        );
        assert!(architecture.is_some());
    }

    #[test]
    fn injected_probe_failure_is_fail_closed_before_feature_admission() {
        let mut context = ullm_runtime_sys::RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        force_probe_failure(ProbeFaultStage::BeforeStart);
        assert!(DeviceCapabilities::probe_m1_runtime_context(&mut context, &mut stream).is_err());
        let capabilities = DeviceCapabilities::from_runtime_context(&context).unwrap();
        assert!(capabilities.runtime_features.is_empty());
    }

    #[test]
    fn every_probe_fault_stage_is_named_and_consumed_once() {
        for (stage, checkpoint, label) in [
            (ProbeFaultStage::QkvPrepare, 2, "qkv-prepare"),
            (ProbeFaultStage::Recurrent, 3, "recurrent"),
            (ProbeFaultStage::PagedPlain, 4, "paged-plain"),
            (ProbeFaultStage::PagedGated, 5, "paged-gated"),
            (ProbeFaultStage::PagedKvWrite, 6, "paged-kv-write"),
            (ProbeFaultStage::FusedWriter, 7, "fused-writer"),
            (ProbeFaultStage::Aq4MatvecBatch, 9, "aq4-matvec-batch"),
            (ProbeFaultStage::QkvPrepareBatch, 10, "qkv-prepare-batch"),
            (ProbeFaultStage::RecurrentSequence, 11, "recurrent-sequence"),
            (ProbeFaultStage::Synchronize, 8, "synchronize"),
        ] {
            force_probe_failure(stage);
            let error = probe_fault_checkpoint(checkpoint, label).unwrap_err();
            assert!(error.contains(label));
            probe_fault_checkpoint(checkpoint, label).unwrap();
        }
    }

    #[test]
    fn aq4_m128_probe_binding_keeps_scale_count_distinct_from_batch_count() {
        let binding = aq4_probe_binding(128);
        assert_eq!(binding.scale_count, 2);
        assert_eq!(binding.group_size, 2);
        assert_eq!(binding.tensor_scale_bits, 1.0_f32.to_bits());
        assert_eq!(binding.row_scale_count, 0);
        assert_eq!(binding.rows, 2);
        assert_eq!(binding.cols, 3);
        assert_eq!(binding.batch_count, 128);
        assert_ne!(binding.scale_count, binding.batch_count);

        force_probe_failure(ProbeFaultStage::Aq4MatvecBatch);
        let error = probe_fault_checkpoint(9, "aq4-matvec-batch").unwrap_err();
        assert!(error.contains("aq4-matvec-batch"));
        probe_fault_checkpoint(9, "aq4-matvec-batch").unwrap();
    }

    #[test]
    fn probe_cache_key_separates_policy_backend_arch_device_and_abi() {
        let base = test_hip_capabilities();
        let base_key = probe_cache_key(&base);
        let mut variants = Vec::new();
        let mut value = base.clone();
        value.runtime_features = RuntimeFeatureSet::from_feature(RuntimeFeature::HipPagedKvWrite);
        variants.push(value);
        let mut value = base.clone();
        value.backend = OperationBackend::Host;
        variants.push(value);
        let mut value = base.clone();
        value.architecture = Some("gfx1200".into());
        variants.push(value);
        let mut value = base.clone();
        value.device_id += 1;
        variants.push(value);
        let mut value = base;
        value.abi_version += 1;
        variants.push(value);
        assert!(
            variants
                .iter()
                .all(|value| probe_cache_key(value) != base_key)
        );
    }

    #[test]
    #[ignore = "requires an isolated HIP device and all production kernel guards"]
    fn isolated_hip_probe_faults_never_publish_partial_cache_entries() {
        let hip_index = (1..ullm_runtime_sys::device_count().unwrap())
            .find(|index| {
                ullm_runtime_sys::device_info(*index)
                    .is_ok_and(|info| info.backend == "hip" && info.gcn_arch_name == "gfx1201")
            })
            .expect("isolated gfx1201 HIP device");
        for environment in [
            "ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_KERNEL",
            "ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL",
            "ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_PAGED_KV_WRITE_KERNEL",
            "ULLM_REQUIRE_HIP_LINEAR_ATTN_KERNEL",
            "ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL",
            "ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL",
            "ULLM_REQUIRE_HIP_LINEAR_ATTN_QKV_PREPARE_BATCH_KERNEL",
            "ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_SEQUENCE_KERNEL",
        ] {
            assert_eq!(std::env::var(environment).as_deref(), Ok("1"));
        }
        let mut context = ullm_runtime_sys::RuntimeContext::create(hip_index).unwrap();
        let mut stream = context.create_stream().unwrap();
        for stage in [
            ProbeFaultStage::QkvPrepare,
            ProbeFaultStage::Recurrent,
            ProbeFaultStage::PagedPlain,
            ProbeFaultStage::PagedGated,
            ProbeFaultStage::PagedKvWrite,
            ProbeFaultStage::FusedWriter,
            ProbeFaultStage::Aq4MatvecBatch,
            ProbeFaultStage::QkvPrepareBatch,
            ProbeFaultStage::RecurrentSequence,
            ProbeFaultStage::Synchronize,
        ] {
            M1_PROBE_CACHE
                .get_or_init(Default::default)
                .lock()
                .unwrap()
                .clear();
            for count in &M1_PROBE_CHECKPOINT_COUNTS {
                count.store(0, std::sync::atomic::Ordering::Release);
            }
            force_probe_failure(stage);
            assert!(
                DeviceCapabilities::probe_m1_runtime_context(&mut context, &mut stream).is_err()
            );
            assert!(M1_PROBE_CACHE.get().unwrap().lock().unwrap().is_empty());
            let failed_stage = stage as usize;
            assert_eq!(
                M1_PROBE_CHECKPOINT_COUNTS[failed_stage].load(std::sync::atomic::Ordering::Acquire),
                1
            );
            let capabilities =
                DeviceCapabilities::probe_m1_runtime_context(&mut context, &mut stream).unwrap();
            assert!(
                capabilities
                    .runtime_features
                    .contains_all(test_hip_capabilities().runtime_features)
            );
            assert_eq!(M1_PROBE_CACHE.get().unwrap().lock().unwrap().len(), 1);
            let before = M1_PROBE_CHECKPOINT_COUNTS
                .iter()
                .map(|count| count.load(std::sync::atomic::Ordering::Acquire))
                .collect::<Vec<_>>();
            DeviceCapabilities::probe_m1_runtime_context(&mut context, &mut stream).unwrap();
            let after = M1_PROBE_CHECKPOINT_COUNTS
                .iter()
                .map(|count| count.load(std::sync::atomic::Ordering::Acquire))
                .collect::<Vec<_>>();
            assert_eq!(after[1], before[1] + 1);
            assert_eq!(&after[2..], &before[2..], "cache hit re-ran scratch probes");
        }
    }

    #[test]
    fn planned_recurrent_wrapper_is_bit_exact_with_direct_cpu_abi() {
        fn bytes(values: &[f32]) -> Vec<u8> {
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect()
        }
        fn buffer(
            context: &mut ullm_runtime_sys::RuntimeContext,
            stream: &mut ullm_runtime_sys::RuntimeStream,
            values: &[f32],
        ) -> ullm_runtime_sys::RuntimeBuffer {
            let payload = bytes(values);
            let mut buffer = context.alloc_buffer(payload.len()).unwrap();
            buffer.copy_from_host(0, &payload, Some(stream)).unwrap();
            buffer
        }
        let mut context = ullm_runtime_sys::RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let q = buffer(&mut context, &mut stream, &vec![0.01; 16 * 128]);
        let k = buffer(&mut context, &mut stream, &vec![0.02; 16 * 128]);
        let v = buffer(&mut context, &mut stream, &vec![0.03; 32 * 128]);
        let gate = buffer(&mut context, &mut stream, &vec![-0.25; 32]);
        let beta = buffer(&mut context, &mut stream, &vec![0.5; 32]);
        let initial_state = vec![0.001; 32 * 128 * 128];
        let mut direct_state = buffer(&mut context, &mut stream, &initial_state);
        let mut planned_state = buffer(&mut context, &mut stream, &initial_state);
        let mut direct_output = buffer(&mut context, &mut stream, &vec![0.0; 32 * 128]);
        let mut planned_output = buffer(&mut context, &mut stream, &vec![0.0; 32 * 128]);

        ullm_runtime_sys::linear_attn_recurrent_f32(
            &q,
            &k,
            &v,
            &gate,
            &beta,
            16,
            32,
            1,
            128,
            128,
            &mut direct_state,
            &mut direct_output,
            Some(&mut stream),
        )
        .unwrap();
        let plan = qwen35_m1_production_registry()
            .unwrap()
            .admit(recurrent_request(ExecutionPhase::Decode))
            .unwrap()
            .into_plan();
        plan.attempt()
            .start()
            .execute_linear_attention_recurrent_f32(
                &q,
                &k,
                &v,
                &gate,
                &beta,
                &mut planned_state,
                &mut planned_output,
                &mut stream,
            )
            .unwrap();
        stream.synchronize().unwrap();

        for (direct, planned) in [
            (&direct_state, &planned_state),
            (&direct_output, &planned_output),
        ] {
            let size = direct.size().unwrap();
            let mut direct_bytes = vec![0; size];
            let mut planned_bytes = vec![0; size];
            direct
                .copy_to_host(0, &mut direct_bytes, Some(&mut stream))
                .unwrap();
            planned
                .copy_to_host(0, &mut planned_bytes, Some(&mut stream))
                .unwrap();
            stream.synchronize().unwrap();
            assert_eq!(direct_bytes, planned_bytes);
        }
    }

    #[test]
    fn planned_qkv_prepare_matches_direct_cpu_abi_state_and_outputs() {
        fn buffer(
            context: &mut ullm_runtime_sys::RuntimeContext,
            stream: &mut ullm_runtime_sys::RuntimeStream,
            elements: usize,
            value: f32,
        ) -> ullm_runtime_sys::RuntimeBuffer {
            let mut buffer = context.alloc_buffer(elements * 4).unwrap();
            let payload = (0..elements)
                .flat_map(|_| value.to_le_bytes())
                .collect::<Vec<_>>();
            buffer.copy_from_host(0, &payload, Some(stream)).unwrap();
            buffer
        }
        fn read(
            buffer: &ullm_runtime_sys::RuntimeBuffer,
            stream: &mut ullm_runtime_sys::RuntimeStream,
        ) -> Vec<u8> {
            let mut value = vec![0; buffer.size().unwrap()];
            buffer.copy_to_host(0, &mut value, Some(stream)).unwrap();
            stream.synchronize().unwrap();
            value
        }
        let mut context = ullm_runtime_sys::RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let qkv = buffer(&mut context, &mut stream, 8_192, 0.01);
        let conv_weight = buffer(&mut context, &mut stream, 8_192 * 4, 0.02);
        let mut direct_history = buffer(&mut context, &mut stream, 8_192 * 4, 0.03);
        let mut planned_history = buffer(&mut context, &mut stream, 8_192 * 4, 0.03);
        let mut direct_conv = buffer(&mut context, &mut stream, 8_192, 0.0);
        let mut direct_q = buffer(&mut context, &mut stream, 16 * 128, 0.0);
        let mut direct_k = buffer(&mut context, &mut stream, 16 * 128, 0.0);
        let mut direct_v = buffer(&mut context, &mut stream, 32 * 128, 0.0);
        let mut planned_conv = buffer(&mut context, &mut stream, 8_192, 0.0);
        let mut planned_q = buffer(&mut context, &mut stream, 16 * 128, 0.0);
        let mut planned_k = buffer(&mut context, &mut stream, 16 * 128, 0.0);
        let mut planned_v = buffer(&mut context, &mut stream, 32 * 128, 0.0);
        ullm_runtime_sys::linear_attn_qkv_prepare_f32(
            &qkv,
            &conv_weight,
            &mut direct_history,
            16,
            32,
            128,
            128,
            4,
            1.0 / 128.0_f32.sqrt(),
            true,
            &mut direct_conv,
            &mut direct_q,
            &mut direct_k,
            &mut direct_v,
            Some(&mut stream),
        )
        .unwrap();
        let request = qwen35_m1_operation_request(
            OperationKind::LinearAttentionQkvPrepare,
            ExecutionPhase::Decode,
            OperationGeometry::LinearAttentionQkvPrepare {
                key_heads: 16,
                value_heads: 32,
                key_dim: 128,
                value_dim: 128,
                kernel_size: 4,
                query_scale: QueryScale::InverseSqrtKeyDim,
                qk_l2_norm: true,
            },
            test_hip_capabilities(),
            u64::MAX,
        );
        qwen35_m1_production_registry()
            .unwrap()
            .admit(request)
            .unwrap()
            .start()
            .execute_linear_attention_qkv_prepare_f32(
                &qkv,
                &conv_weight,
                &mut planned_history,
                &mut planned_conv,
                &mut planned_q,
                &mut planned_k,
                &mut planned_v,
                &mut stream,
            )
            .unwrap();
        for (direct, planned) in [
            (&direct_history, &planned_history),
            (&direct_conv, &planned_conv),
            (&direct_q, &planned_q),
            (&direct_k, &planned_k),
            (&direct_v, &planned_v),
        ] {
            assert_eq!(read(direct, &mut stream), read(planned, &mut stream));
        }
    }

    #[test]
    fn planned_sequence_wrappers_are_bit_exact_with_direct_cpu_abi() {
        fn buffer(
            context: &mut ullm_runtime_sys::RuntimeContext,
            stream: &mut ullm_runtime_sys::RuntimeStream,
            values: &[f32],
        ) -> ullm_runtime_sys::RuntimeBuffer {
            let mut buffer = context.alloc_buffer(values.len() * 4).unwrap();
            let bytes = values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>();
            buffer.copy_from_host(0, &bytes, Some(stream)).unwrap();
            buffer
        }
        fn read(
            buffer: &ullm_runtime_sys::RuntimeBuffer,
            stream: &mut ullm_runtime_sys::RuntimeStream,
        ) -> Vec<u8> {
            let mut bytes = vec![0; buffer.size().unwrap()];
            buffer.copy_to_host(0, &mut bytes, Some(stream)).unwrap();
            stream.synchronize().unwrap();
            bytes
        }

        const SEQUENCE_LEN: usize = 2;
        const KEY_HEADS: usize = 16;
        const VALUE_HEADS: usize = 32;
        const KEY_DIM: usize = 128;
        const VALUE_DIM: usize = 128;
        const KERNEL_SIZE: usize = 4;
        const Q_ELEMENTS: usize = KEY_HEADS * KEY_DIM;
        const V_ELEMENTS: usize = VALUE_HEADS * VALUE_DIM;
        const CHANNELS: usize = Q_ELEMENTS * 2 + V_ELEMENTS;

        let mut context = ullm_runtime_sys::RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let qkv_values = (0..SEQUENCE_LEN * CHANNELS)
            .map(|index| (index % 29) as f32 * 0.002 - 0.025)
            .collect::<Vec<_>>();
        let weight_values = (0..CHANNELS * KERNEL_SIZE)
            .map(|index| (index % 7) as f32 * 0.01 - 0.02)
            .collect::<Vec<_>>();
        let history_values = (0..CHANNELS * KERNEL_SIZE)
            .map(|index| (index % 11) as f32 * 0.001)
            .collect::<Vec<_>>();
        let qkv = buffer(&mut context, &mut stream, &qkv_values);
        let weight = buffer(&mut context, &mut stream, &weight_values);
        let mut direct_history = buffer(&mut context, &mut stream, &history_values);
        let mut planned_history = buffer(&mut context, &mut stream, &history_values);
        let zeros = |elements| vec![0.0_f32; elements];
        let mut direct_conv = buffer(&mut context, &mut stream, &zeros(SEQUENCE_LEN * CHANNELS));
        let mut direct_q = buffer(&mut context, &mut stream, &zeros(SEQUENCE_LEN * Q_ELEMENTS));
        let mut direct_k = buffer(&mut context, &mut stream, &zeros(SEQUENCE_LEN * Q_ELEMENTS));
        let mut direct_v = buffer(&mut context, &mut stream, &zeros(SEQUENCE_LEN * V_ELEMENTS));
        let mut planned_conv = buffer(&mut context, &mut stream, &zeros(SEQUENCE_LEN * CHANNELS));
        let mut planned_q = buffer(&mut context, &mut stream, &zeros(SEQUENCE_LEN * Q_ELEMENTS));
        let mut planned_k = buffer(&mut context, &mut stream, &zeros(SEQUENCE_LEN * Q_ELEMENTS));
        let mut planned_v = buffer(&mut context, &mut stream, &zeros(SEQUENCE_LEN * V_ELEMENTS));
        ullm_runtime_sys::linear_attn_qkv_prepare_batch_f32(
            &qkv,
            &weight,
            &mut direct_history,
            KEY_HEADS,
            VALUE_HEADS,
            KEY_DIM,
            VALUE_DIM,
            KERNEL_SIZE,
            SEQUENCE_LEN,
            1.0 / (KEY_DIM as f32).sqrt(),
            true,
            &mut direct_conv,
            &mut direct_q,
            &mut direct_k,
            &mut direct_v,
            Some(&mut stream),
        )
        .unwrap();
        qwen35_m1_production_registry()
            .unwrap()
            .admit(sequence_request(
                OperationKind::LinearAttentionQkvPrepareBatch,
                ExecutionPhase::ColdPrefill,
                SEQUENCE_LEN as u64,
            ))
            .unwrap()
            .start()
            .execute_linear_attention_qkv_prepare_batch_f32(
                &qkv,
                &weight,
                &mut planned_history,
                &mut planned_conv,
                &mut planned_q,
                &mut planned_k,
                &mut planned_v,
                &mut stream,
            )
            .unwrap();
        for (direct, planned) in [
            (&direct_history, &planned_history),
            (&direct_conv, &planned_conv),
            (&direct_q, &planned_q),
            (&direct_k, &planned_k),
            (&direct_v, &planned_v),
        ] {
            assert_eq!(read(direct, &mut stream), read(planned, &mut stream));
        }

        let gate_values = (0..SEQUENCE_LEN * VALUE_HEADS)
            .map(|index| -0.01 * (1 + index % 5) as f32)
            .collect::<Vec<_>>();
        let beta_values = (0..SEQUENCE_LEN * VALUE_HEADS)
            .map(|index| 0.1 + 0.01 * (index % 7) as f32)
            .collect::<Vec<_>>();
        let gate = buffer(&mut context, &mut stream, &gate_values);
        let beta = buffer(&mut context, &mut stream, &beta_values);
        let state_values = (0..VALUE_HEADS * KEY_DIM * VALUE_DIM)
            .map(|index| (index % 13) as f32 * 0.0001)
            .collect::<Vec<_>>();
        let mut direct_state = buffer(&mut context, &mut stream, &state_values);
        let mut planned_state = buffer(&mut context, &mut stream, &state_values);
        let mut direct_output =
            buffer(&mut context, &mut stream, &zeros(SEQUENCE_LEN * V_ELEMENTS));
        let mut planned_output =
            buffer(&mut context, &mut stream, &zeros(SEQUENCE_LEN * V_ELEMENTS));
        ullm_runtime_sys::linear_attn_recurrent_f32(
            &direct_q,
            &direct_k,
            &direct_v,
            &gate,
            &beta,
            KEY_HEADS,
            VALUE_HEADS,
            SEQUENCE_LEN,
            KEY_DIM,
            VALUE_DIM,
            &mut direct_state,
            &mut direct_output,
            Some(&mut stream),
        )
        .unwrap();
        qwen35_m1_production_registry()
            .unwrap()
            .admit(sequence_request(
                OperationKind::GatedDeltaRuleSequence,
                ExecutionPhase::ColdPrefill,
                SEQUENCE_LEN as u64,
            ))
            .unwrap()
            .start()
            .execute_linear_attention_recurrent_sequence_f32(
                &planned_q,
                &planned_k,
                &planned_v,
                &gate,
                &beta,
                &mut planned_state,
                &mut planned_output,
                &mut stream,
            )
            .unwrap();
        assert_eq!(
            read(&direct_state, &mut stream),
            read(&planned_state, &mut stream)
        );
        assert_eq!(
            read(&direct_output, &mut stream),
            read(&planned_output, &mut stream)
        );
    }

    #[test]
    fn planned_paged_plain_and_gated_match_direct_cpu_abi_in_every_phase() {
        fn buffer(
            context: &mut ullm_runtime_sys::RuntimeContext,
            stream: &mut ullm_runtime_sys::RuntimeStream,
            elements: usize,
            value: f32,
        ) -> ullm_runtime_sys::RuntimeBuffer {
            let mut buffer = context.alloc_buffer(elements * 4).unwrap();
            let payload = (0..elements)
                .flat_map(|_| value.to_le_bytes())
                .collect::<Vec<_>>();
            buffer.copy_from_host(0, &payload, Some(stream)).unwrap();
            buffer
        }
        fn read(
            buffer: &ullm_runtime_sys::RuntimeBuffer,
            stream: &mut ullm_runtime_sys::RuntimeStream,
        ) -> Vec<u8> {
            let mut bytes = vec![0; buffer.size().unwrap()];
            buffer.copy_to_host(0, &mut bytes, Some(stream)).unwrap();
            stream.synchronize().unwrap();
            bytes
        }
        let mut context = ullm_runtime_sys::RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let q = buffer(&mut context, &mut stream, 32 * 128, 0.01);
        let gate = buffer(&mut context, &mut stream, 32 * 128, -0.25);
        let k_cache = buffer(&mut context, &mut stream, 16 * 256 * 8 * 128, 0.02);
        let v_cache = buffer(&mut context, &mut stream, 16 * 256 * 8 * 128, 0.03);
        let cache_before = (read(&k_cache, &mut stream), read(&v_cache, &mut stream));
        let mut table = context.alloc_buffer(16 * 4).unwrap();
        let table_bytes = (0_u32..16).flat_map(u32::to_le_bytes).collect::<Vec<_>>();
        table
            .copy_from_host(0, &table_bytes, Some(&mut stream))
            .unwrap();
        let mut direct_plain = buffer(&mut context, &mut stream, 32 * 128, 0.0);
        let mut direct_gated = buffer(&mut context, &mut stream, 32 * 128, 0.0);
        ullm_runtime_sys::paged_decode_attn_f32(
            &q,
            &k_cache,
            &v_cache,
            &table,
            1,
            256,
            16,
            16,
            4,
            256,
            256,
            1.0 / 256.0_f32.sqrt(),
            &mut direct_plain,
            Some(&mut stream),
        )
        .unwrap();
        ullm_runtime_sys::paged_decode_attn_sigmoid_gate_f32(
            &q,
            &gate,
            &k_cache,
            &v_cache,
            &table,
            1,
            256,
            16,
            16,
            4,
            256,
            256,
            1.0 / 256.0_f32.sqrt(),
            &mut direct_gated,
            Some(&mut stream),
        )
        .unwrap();
        let expected_plain = read(&direct_plain, &mut stream);
        let expected_gated = read(&direct_gated, &mut stream);

        for phase in [
            ExecutionPhase::ColdPrefill,
            ExecutionPhase::CachedPrefixPrefill,
            ExecutionPhase::Decode,
        ] {
            for (sigmoid_gate, expected) in [(false, &expected_plain), (true, &expected_gated)] {
                let request = qwen35_m1_operation_request(
                    OperationKind::PagedCausalGqaRead,
                    phase,
                    OperationGeometry::PagedCausalGqaRead {
                        q_heads: 16,
                        kv_heads: 4,
                        head_dim: 256,
                        value_dim: 256,
                        block_size: 256,
                        cache_blocks: 16,
                        sigmoid_gate,
                    },
                    test_hip_capabilities(),
                    u64::MAX,
                );
                let plan = qwen35_m1_production_registry()
                    .unwrap()
                    .admit(request)
                    .unwrap()
                    .into_plan();
                let mut output = buffer(&mut context, &mut stream, 32 * 128, 0.0);
                if sigmoid_gate {
                    plan.attempt()
                        .start()
                        .execute_paged_decode_attention_sigmoid_gate_f32(
                            &q,
                            &gate,
                            &k_cache,
                            &v_cache,
                            &table,
                            1,
                            &mut output,
                            &mut stream,
                        )
                        .unwrap();
                } else {
                    plan.attempt()
                        .start()
                        .execute_paged_decode_attention_f32(
                            &q,
                            &k_cache,
                            &v_cache,
                            &table,
                            1,
                            &mut output,
                            &mut stream,
                        )
                        .unwrap();
                }
                assert_eq!(&read(&output, &mut stream), expected);
            }
        }
        assert_eq!(read(&k_cache, &mut stream), cache_before.0);
        assert_eq!(read(&v_cache, &mut stream), cache_before.1);
    }

    #[test]
    fn planned_plain_and_fused_writers_match_direct_cpu_abi_in_every_phase() {
        fn buffer(
            context: &mut ullm_runtime_sys::RuntimeContext,
            stream: &mut ullm_runtime_sys::RuntimeStream,
            values: &[f32],
        ) -> ullm_runtime_sys::RuntimeBuffer {
            let mut buffer = context.alloc_buffer(values.len() * 4).unwrap();
            let bytes = values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>();
            buffer.copy_from_host(0, &bytes, Some(stream)).unwrap();
            buffer
        }
        fn read(
            buffer: &ullm_runtime_sys::RuntimeBuffer,
            stream: &mut ullm_runtime_sys::RuntimeStream,
        ) -> Vec<u8> {
            let mut bytes = vec![0; buffer.size().unwrap()];
            buffer.copy_to_host(0, &mut bytes, Some(stream)).unwrap();
            stream.synchronize().unwrap();
            bytes
        }
        let mut context = ullm_runtime_sys::RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();
        let table_values = (0_u32..16).flat_map(u32::to_le_bytes).collect::<Vec<_>>();
        let mut table = context.alloc_buffer(table_values.len()).unwrap();
        table
            .copy_from_host(0, &table_values, Some(&mut stream))
            .unwrap();
        let cache_elements = 16 * 256 * 4 * 256;
        let initial = (0..cache_elements)
            .map(|index| 0.25 + (index % 17) as f32 * 0.001)
            .collect::<Vec<_>>();
        let k = (0..4 * 256)
            .map(|index| index as f32 * 0.002 - 0.5)
            .collect::<Vec<_>>();
        let v = (0..4 * 256)
            .map(|index| index as f32 * -0.003 + 0.75)
            .collect::<Vec<_>>();
        let k = buffer(&mut context, &mut stream, &k);
        let v = buffer(&mut context, &mut stream, &v);

        for phase in [
            ExecutionPhase::ColdPrefill,
            ExecutionPhase::CachedPrefixPrefill,
            ExecutionPhase::Decode,
        ] {
            let mut direct_k_cache = buffer(&mut context, &mut stream, &initial);
            let mut direct_v_cache = buffer(&mut context, &mut stream, &initial);
            let mut planned_k_cache = buffer(&mut context, &mut stream, &initial);
            let mut planned_v_cache = buffer(&mut context, &mut stream, &initial);
            ullm_runtime_sys::paged_kv_write_f32(
                &k,
                &v,
                &table,
                257,
                256,
                16,
                4,
                256,
                256,
                &mut direct_k_cache,
                &mut direct_v_cache,
                Some(&mut stream),
            )
            .unwrap();
            let request = qwen35_m1_operation_request(
                OperationKind::PagedKvWrite,
                phase,
                OperationGeometry::PagedKvWrite {
                    kv_heads: 4,
                    head_dim: 256,
                    value_dim: 256,
                    block_size: 256,
                    cache_blocks: 16,
                },
                test_hip_capabilities(),
                u64::MAX,
            );
            qwen35_m1_production_registry()
                .unwrap()
                .admit(request)
                .unwrap()
                .start()
                .execute_paged_kv_write_f32(
                    &k,
                    &v,
                    &table,
                    257,
                    &mut planned_k_cache,
                    &mut planned_v_cache,
                    &mut stream,
                )
                .unwrap();
            assert_eq!(
                read(&direct_k_cache, &mut stream),
                read(&planned_k_cache, &mut stream)
            );
            assert_eq!(
                read(&direct_v_cache, &mut stream),
                read(&planned_v_cache, &mut stream)
            );

            let q_projected_values = (0..2 * 16 * 256)
                .map(|index| (index % 31) as f32 * 0.01 - 0.15)
                .collect::<Vec<_>>();
            let q_projected = buffer(&mut context, &mut stream, &q_projected_values);
            let k_projected = buffer(
                &mut context,
                &mut stream,
                &(0..4 * 256)
                    .map(|index| (index % 23) as f32 * 0.02 - 0.2)
                    .collect::<Vec<_>>(),
            );
            let v_projected = buffer(
                &mut context,
                &mut stream,
                &(0..4 * 256)
                    .map(|index| index as f32 * 0.001)
                    .collect::<Vec<_>>(),
            );
            let q_weight = buffer(&mut context, &mut stream, &vec![1.0; 256]);
            let k_weight = buffer(&mut context, &mut stream, &vec![0.75; 256]);
            let mut direct_gate = buffer(&mut context, &mut stream, &vec![0.0; 16 * 256]);
            let mut direct_q = buffer(&mut context, &mut stream, &vec![0.0; 16 * 256]);
            let mut planned_gate = buffer(&mut context, &mut stream, &vec![0.0; 16 * 256]);
            let mut planned_q = buffer(&mut context, &mut stream, &vec![0.0; 16 * 256]);
            let mut direct_k_cache = buffer(&mut context, &mut stream, &initial);
            let mut direct_v_cache = buffer(&mut context, &mut stream, &initial);
            let mut planned_k_cache = buffer(&mut context, &mut stream, &initial);
            let mut planned_v_cache = buffer(&mut context, &mut stream, &initial);
            ullm_runtime_sys::qwen35_qk_norm_rope_paged_kv_write_f32(
                &q_projected,
                &k_projected,
                &v_projected,
                &q_weight,
                &k_weight,
                &table,
                16,
                4,
                256,
                256,
                64,
                19,
                10_000_000.0,
                1e-5,
                257,
                256,
                16,
                &mut direct_gate,
                &mut direct_q,
                &mut direct_k_cache,
                &mut direct_v_cache,
                Some(&mut stream),
            )
            .unwrap();
            let request = qwen35_m1_operation_request(
                OperationKind::FusedQkNormRopePagedKvWrite,
                phase,
                OperationGeometry::FusedQkNormRopePagedKvWrite {
                    q_heads: 16,
                    kv_heads: 4,
                    head_dim: 256,
                    value_dim: 256,
                    rotary_dim: 64,
                    rope_base_bits: 10_000_000.0_f32.to_bits(),
                    norm_epsilon_bits: 1e-5_f32.to_bits(),
                    block_size: 256,
                    cache_blocks: 16,
                },
                test_hip_capabilities(),
                u64::MAX,
            );
            qwen35_m1_production_registry()
                .unwrap()
                .admit(request)
                .unwrap()
                .start()
                .execute_fused_qk_norm_rope_paged_kv_write_f32(
                    &q_projected,
                    &k_projected,
                    &v_projected,
                    &q_weight,
                    &k_weight,
                    &table,
                    64,
                    19,
                    10_000_000.0,
                    1e-5,
                    257,
                    &mut planned_gate,
                    &mut planned_q,
                    &mut planned_k_cache,
                    &mut planned_v_cache,
                    &mut stream,
                )
                .unwrap();
            for (direct, planned) in [
                (&direct_gate, &planned_gate),
                (&direct_q, &planned_q),
                (&direct_k_cache, &planned_k_cache),
                (&direct_v_cache, &planned_v_cache),
            ] {
                assert_eq!(read(direct, &mut stream), read(planned, &mut stream));
            }
        }
    }

    #[test]
    fn writer_geometry_mismatch_is_rejected_before_sys_dispatch() {
        WRITER_SYS_CALL_COUNT.with(|count| count.set(0));
        for (kind, geometry) in [
            (
                OperationKind::PagedKvWrite,
                OperationGeometry::PagedKvWrite {
                    kv_heads: 4,
                    head_dim: 128,
                    value_dim: 256,
                    block_size: 256,
                    cache_blocks: 16,
                },
            ),
            (
                OperationKind::FusedQkNormRopePagedKvWrite,
                OperationGeometry::FusedQkNormRopePagedKvWrite {
                    q_heads: 16,
                    kv_heads: 4,
                    head_dim: 256,
                    value_dim: 256,
                    rotary_dim: 128,
                    rope_base_bits: 10_000_000.0_f32.to_bits(),
                    norm_epsilon_bits: 1e-5_f32.to_bits(),
                    block_size: 256,
                    cache_blocks: 16,
                },
            ),
        ] {
            let request = qwen35_m1_operation_request(
                kind,
                ExecutionPhase::Decode,
                geometry,
                test_hip_capabilities(),
                u64::MAX,
            );
            assert!(
                qwen35_m1_production_registry()
                    .unwrap()
                    .admit(request)
                    .is_err()
            );
        }
        WRITER_SYS_CALL_COUNT.with(|count| assert_eq!(count.get(), 0));
    }

    #[test]
    fn actual_abi_failures_are_single_attempts_for_all_in_place_operations() {
        fn tiny(context: &mut ullm_runtime_sys::RuntimeContext) -> ullm_runtime_sys::RuntimeBuffer {
            context.alloc_buffer(4).unwrap()
        }
        let mut context = ullm_runtime_sys::RuntimeContext::create(0).unwrap();
        let mut stream = context.create_stream().unwrap();

        PREPARE_SYS_CALL_COUNT.with(|count| count.set(0));
        let prepare = qwen35_m1_production_registry()
            .unwrap()
            .admit(qwen35_m1_operation_request(
                OperationKind::LinearAttentionQkvPrepare,
                ExecutionPhase::Decode,
                OperationGeometry::LinearAttentionQkvPrepare {
                    key_heads: 16,
                    value_heads: 32,
                    key_dim: 128,
                    value_dim: 128,
                    kernel_size: 4,
                    query_scale: QueryScale::InverseSqrtKeyDim,
                    qk_l2_norm: true,
                },
                test_hip_capabilities(),
                u64::MAX,
            ))
            .unwrap()
            .into_plan();
        assert_eq!(prepare.trace().resolution, ResolutionKind::Primary);
        let qkv = tiny(&mut context);
        let weight = tiny(&mut context);
        let mut history = tiny(&mut context);
        let mut conv = tiny(&mut context);
        let mut q = tiny(&mut context);
        let mut k = tiny(&mut context);
        let mut v = tiny(&mut context);
        assert!(
            prepare
                .attempt()
                .start()
                .execute_linear_attention_qkv_prepare_f32(
                    &qkv,
                    &weight,
                    &mut history,
                    &mut conv,
                    &mut q,
                    &mut k,
                    &mut v,
                    &mut stream,
                )
                .is_err()
        );
        PREPARE_SYS_CALL_COUNT.with(|count| assert_eq!(count.get(), 1));

        RECURRENT_SYS_CALL_COUNT.with(|count| count.set(0));
        let recurrent = qwen35_m1_production_registry()
            .unwrap()
            .admit(recurrent_request(ExecutionPhase::Decode))
            .unwrap()
            .into_plan();
        let q = tiny(&mut context);
        let k = tiny(&mut context);
        let v = tiny(&mut context);
        let gate = tiny(&mut context);
        let beta = tiny(&mut context);
        let mut state = tiny(&mut context);
        let mut output = tiny(&mut context);
        assert!(
            recurrent
                .attempt()
                .start()
                .execute_linear_attention_recurrent_f32(
                    &q,
                    &k,
                    &v,
                    &gate,
                    &beta,
                    &mut state,
                    &mut output,
                    &mut stream,
                )
                .is_err()
        );
        RECURRENT_SYS_CALL_COUNT.with(|count| assert_eq!(count.get(), 1));

        WRITER_SYS_CALL_COUNT.with(|count| count.set(0));
        let writer = qwen35_m1_production_registry()
            .unwrap()
            .admit(qwen35_m1_operation_request(
                OperationKind::PagedKvWrite,
                ExecutionPhase::Decode,
                OperationGeometry::PagedKvWrite {
                    kv_heads: 4,
                    head_dim: 256,
                    value_dim: 256,
                    block_size: 256,
                    cache_blocks: 16,
                },
                test_hip_capabilities(),
                u64::MAX,
            ))
            .unwrap()
            .into_plan();
        let k = tiny(&mut context);
        let v = tiny(&mut context);
        let table = tiny(&mut context);
        let mut k_cache = tiny(&mut context);
        let mut v_cache = tiny(&mut context);
        assert!(
            writer
                .attempt()
                .start()
                .execute_paged_kv_write_f32(
                    &k,
                    &v,
                    &table,
                    0,
                    &mut k_cache,
                    &mut v_cache,
                    &mut stream,
                )
                .is_err()
        );
        WRITER_SYS_CALL_COUNT.with(|count| assert_eq!(count.get(), 1));
        // Each plan was consumed by `attempt().start().execute_*`; retry and fallback APIs are no
        // longer available after the ABI call. Layer tests cover the reset gate on request state.
    }

    #[test]
    fn specificity_precedes_priority_and_declared_fallback_is_prestart_only() {
        let registry = qwen35_m1_production_registry().unwrap();
        let mut generic = recurrent_descriptor(&registry);
        generic.id = "hip.linear-attention-recurrent-f32.generic";
        generic.required_features = RuntimeFeatureSet::EMPTY;
        generic.priority = 1_000;
        let mut specialized = recurrent_descriptor(&registry);
        specialized.id = "hip.linear-attention-recurrent-f32.gfx1201";
        specialized.architecture = Some("gfx1201");
        specialized.fallback_id = Some(generic.id);
        let generic_id = generic.id;
        let specialized_id = specialized.id;
        let registry = BackendOperationRegistry::new(vec![generic, specialized]).unwrap();
        let mut request = recurrent_request(ExecutionPhase::Decode);
        request.device.architecture = Some("gfx1201".into());
        let primary = registry.resolve(&request).unwrap();
        assert_eq!(primary.trace().implementation_id, specialized_id);
        let fallback = registry.admit(request).unwrap().fallback().unwrap();
        let fallback = fallback.into_plan();
        assert_eq!(fallback.trace().implementation_id, generic_id);
        assert_eq!(
            fallback.trace().resolution,
            ResolutionKind::Fallback {
                unavailable_primary: specialized_id
            }
        );
        let _started = fallback.attempt().start();
        // StartedOperationPlan deliberately has no API that can select another descriptor.
    }

    #[test]
    fn trace_exposes_state_effect_and_started_plan_has_no_fallback_transition() {
        let registry = qwen35_m1_production_registry().unwrap();
        let plan = registry
            .resolve(&recurrent_request(ExecutionPhase::Decode))
            .unwrap();
        let started = plan.attempt().start();
        let trace = started.trace();
        assert!(
            trace
                .state_effect
                .reads
                .contains(StateResource::RecurrentState)
        );
        assert_eq!(trace.state_effect.update_mode, StateUpdateMode::InPlace);
        assert!(trace.state_effect.externally_visible_before_commit);
        assert_eq!(trace.resolution, ResolutionKind::Primary);
        assert_eq!(trace.batch_width, 1);
        assert_eq!(trace.chunk_width, 1);
        assert_eq!(trace.device.backend, OperationBackend::Hip);
        assert_eq!(
            trace.executable,
            ExecutableOperation::HipLinearAttentionRecurrentF32
        );
        assert!(trace.workspace.persistent_bytes > 0);
        let audit: serde_json::Value = serde_json::from_str(&trace.audit_json()).unwrap();
        assert_eq!(audit["implementation_id"], trace.implementation_id);
        assert_eq!(audit["phase"], "Decode");
        assert_eq!(audit["backend"], "Hip");
        assert_eq!(audit["executable"], "HipLinearAttentionRecurrentF32");
    }
}
